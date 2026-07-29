#!/usr/bin/env python3
"""
build_dashboard.py
==================
Reads trade_log.jsonl and paper_portfolio.json and writes a single
self-contained dashboard.html: inline CSS and JS, no external
dependencies, no server, no network calls. Open it directly in a browser.

DELIBERATELY READ-ONLY. There is no kill switch, no dry-run toggle, no
agent-state indicator, and no pending-order queue in this page. It is a
static file generated after the fact and has no channel back to the
agent — a control that looked live but did nothing would be worse than no
control at all. To change what the agent does, change src/config.py and
run it again.

The page reads top to bottom as one causal chain:

    data in  ->  model proposed  ->  code decided  ->  what happened

so a single decision can be traced end to end. That ordering is the point;
density is not.

Usage:
    python scripts/build_dashboard.py
    python scripts/build_dashboard.py --open
    python scripts/build_dashboard.py --csv
"""

import argparse
import csv
import html
import json
import os
import sys
import webbrowser
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backtest import _buy_and_hold_series
from config import AgentConfig

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_LOG = os.path.join(REPO_ROOT, "trade_log.jsonl")
DEFAULT_PORTFOLIO = os.path.join(REPO_ROOT, "paper_portfolio.json")
DEFAULT_OUT = os.path.join(REPO_ROOT, "dashboard.html")
DEFAULT_CSV = os.path.join(REPO_ROOT, "trades.csv")

# Events that represent a decision about a specific proposed order, mapped
# to the verdict shown in the DECISIONS table.
DECISION_VERDICTS = {
    "order_blocked": "BLOCKED",
    "order_skipped": "SKIPPED",
    "auto_approved": "AUTO-APPROVED",
    "order_executed": "EXECUTED",
    "simulated_fill": "SIMULATED",
}

# Verdicts that indicate the guardrail layer or a human stopped an order.
# Visually distinguished because they carry the most information.
NEGATIVE_VERDICTS = {"BLOCKED", "SKIPPED"}

# Events produced only by dry_run cycles vs only by live cycles. Used to
# decide the PAPER / LIVE banner. Deliberately narrow: only events that
# unambiguously come from one mode.
PAPER_ONLY_EVENTS = {"simulated_fill"}
LIVE_ONLY_EVENTS = {"order_executed", "execution_verified", "execution_mismatch"}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_events(path: str) -> list[dict]:
    """Read a JSONL log, skipping malformed lines rather than failing —
    a half-written final line from an interrupted run must not make the
    whole history unreadable."""
    if not os.path.exists(path):
        return []
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def load_portfolio(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# --------------------------------------------------------------------------
# Pure reduction over the event log
# --------------------------------------------------------------------------


def _parse_ts(ts: str):
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def determine_mode(events: list[dict]) -> str:
    """Return 'PAPER', 'LIVE', 'MIXED', or 'UNKNOWN'.

    Simulated and real money must never be added into one figure, so this
    verdict gates how P&L is presented downstream.
    """
    has_paper = any(e.get("event") in PAPER_ONLY_EVENTS for e in events)
    has_live = any(e.get("event") in LIVE_ONLY_EVENTS for e in events)
    if has_paper and has_live:
        return "MIXED"
    if has_paper:
        return "PAPER"
    if has_live:
        return "LIVE"
    return "UNKNOWN"


def summarize_health(events: list[dict]) -> dict:
    """Health is driven by the MOST RECENT cycle only.

    A stale last-cycle timestamp means a scheduled run died silently; an
    incomplete snapshot on the latest cycle means the agent may not have
    actually seen the market. Both are invisible in P&L, which is exactly
    why they lead the page.
    """
    cycles = [e for e in events if e.get("event") == "cycle_report"]
    last_ts_raw = cycles[-1]["ts"] if cycles else None
    last_ts = _parse_ts(last_ts_raw) if last_ts_raw else None

    age_text = "never"
    if last_ts is not None:
        delta = datetime.now(timezone.utc) - last_ts
        seconds = max(delta.total_seconds(), 0)
        if seconds < 3600:
            age_text = f"{int(seconds // 60)} minutes ago"
        elif seconds < 86400:
            age_text = f"{int(seconds // 3600)} hours ago"
        else:
            age_text = f"{int(seconds // 86400)} days ago"

    # Only events at or after the last cycle_report describe the latest
    # cycle — earlier incomplete snapshots are already-known history.
    last_cycle_index = None
    for i, e in enumerate(events):
        if e.get("event") == "cycle_report":
            last_cycle_index = i
    tail = events[last_cycle_index:] if last_cycle_index is not None else []

    snapshot_ok = any(e.get("event") == "market_snapshot" for e in tail) and not any(
        e.get("event") == "market_snapshot_incomplete" for e in tail
    )

    missing = []
    for e in tail:
        if e.get("event") == "market_snapshot_incomplete":
            missing = e.get("missing_tickers", []) or []

    return {
        "cycle_count": len(cycles),
        "last_ts": last_ts_raw,
        "age_text": age_text,
        "status": "OK" if snapshot_ok else "WARNING",
        "missing_tickers": missing,
        "has_cycles": bool(cycles),
    }


def collect_snapshots(events: list[dict], limit: int = 10) -> list[dict]:
    """Most recent market snapshots, newest first."""
    snaps = [e for e in events if e.get("event") == "market_snapshot"]
    out = []
    for e in reversed(snaps[-limit:]):
        rows = []
        for ticker, data in sorted((e.get("snapshot") or {}).items()):
            if not isinstance(data, dict):
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "current_price": data.get("current_price"),
                    "day_change_pct": data.get("day_change_pct"),
                }
            )
        out.append({"ts": e.get("ts"), "rows": rows})
    return out


def _explain_arithmetic(order: dict, verdict: str, cfg: AgentConfig) -> list[str]:
    """Re-derive, in plain arithmetic, the numbers the guardrail compared.

    This is shown next to the guardrail's own reason string so a reader
    can check the decision rather than take it on faith. It reports only
    what the raw order fields support; it never guesses.
    """
    lines = []
    ticker = str(order.get("ticker", "")).upper().strip()
    side = str(order.get("side", "")).lower()
    dollars = order.get("dollars")

    if side == "buy" and isinstance(dollars, (int, float)):
        positions = order.get("positions") or {}
        if isinstance(positions, dict):
            held = 0.0
            for t, v in positions.items():
                if str(t).upper().strip() == ticker and isinstance(v, (int, float)):
                    held += float(v)
            lines.append(
                f"per-position: {held:.2f} + {dollars:.2f} = {held + dollars:.2f} "
                f"vs cap {cfg.max_position_dollars:.2f}"
                f"{'  → OVER' if held + dollars > cfg.max_position_dollars else '  → within'}"
            )

            total = sum(
                float(v) for v in positions.values() if isinstance(v, (int, float))
            )
            lines.append(
                f"account total: {total:.2f} + {dollars:.2f} = {total + dollars:.2f} "
                f"vs cap {cfg.max_total_dollars:.2f}"
                f"{'  → OVER' if total + dollars > cfg.max_total_dollars else '  → within'}"
            )

            for group_name, group_tickers in cfg.correlation_groups.items():
                if ticker not in group_tickers:
                    continue
                exposure = sum(
                    float(v)
                    for t, v in positions.items()
                    if str(t).upper().strip() in group_tickers
                    and isinstance(v, (int, float))
                )
                lines.append(
                    f"group '{group_name}': {exposure:.2f} + {dollars:.2f} = "
                    f"{exposure + dollars:.2f} vs cap {cfg.max_group_dollars:.2f}"
                    f"{'  → OVER' if exposure + dollars > cfg.max_group_dollars else '  → within'}"
                )

        dcp = order.get("day_change_pct")
        if isinstance(dcp, (int, float)):
            lines.append(
                f"dip trigger: {dcp:.2f}% vs required <= -{cfg.dip_trigger_pct:.2f}%"
                f"{'  → qualifies' if dcp <= -cfg.dip_trigger_pct else '  → does NOT qualify'}"
            )

    if side == "sell":
        price = order.get("current_price")
        avg_cost = order.get("avg_cost")
        if isinstance(price, (int, float)) and isinstance(avg_cost, (int, float)) and avg_cost:
            gain = (price - avg_cost) / avg_cost * 100
            lines.append(
                f"gain vs basis: ({price:.2f} - {avg_cost:.2f}) / {avg_cost:.2f} "
                f"= {gain:+.2f}%  (target {cfg.revert_target_pct:.2f}%, "
                f"disaster stop -{cfg.disaster_stop_pct:.2f}%)"
            )
        days_held = order.get("days_held")
        if isinstance(days_held, (int, float)):
            lines.append(
                f"time exit: held {int(days_held)} days vs max {cfg.max_hold_days}"
                f"{'  → forces exit' if days_held >= cfg.max_hold_days else '  → not yet'}"
            )

    return lines


def collect_decisions(events: list[dict], cfg: AgentConfig) -> list[dict]:
    """One row per decision about a proposed order, newest first."""
    rows = []
    for e in events:
        verdict = DECISION_VERDICTS.get(e.get("event"))
        if verdict is None:
            continue

        order = e.get("order") or {}
        ticker = str(order.get("ticker") or e.get("ticker") or "").upper().strip()
        side = str(order.get("side") or e.get("side") or "").lower()
        dollars = order.get("dollars", e.get("dollars"))
        price = order.get("current_price", e.get("simulated_price"))
        guardrail_reason = e.get("reason", "")

        # The model proposed this order at all, which means it believed the
        # order qualified. A guardrail rejection is therefore a direct
        # disagreement between the narrative and the arithmetic.
        disagrees = verdict == "BLOCKED"

        rows.append(
            {
                "ts": e.get("ts", ""),
                "date": str(e.get("ts", ""))[:10],
                "ticker": ticker,
                "side": side,
                "dollars": dollars,
                "price": price,
                "verdict": verdict,
                "reason": guardrail_reason,
                "model_reason": order.get("reason", ""),
                "model_numbers": {
                    k: order.get(k)
                    for k in (
                        "current_price",
                        "day_change_pct",
                        "positions",
                        "avg_cost",
                        "days_held",
                    )
                    if order.get(k) is not None
                },
                "arithmetic": _explain_arithmetic(order, verdict, cfg),
                "disagrees": disagrees,
            }
        )
    rows.reverse()
    return rows


def collect_api_cost(events: list[dict]) -> dict:
    usages = [e for e in events if e.get("event") == "api_usage"]
    total = sum(float(e.get("estimated_cost_usd", 0) or 0) for e in usages)
    by_call: dict[str, float] = {}
    for e in usages:
        by_call[e.get("call", "unknown")] = by_call.get(e.get("call", "unknown"), 0.0) + float(
            e.get("estimated_cost_usd", 0) or 0
        )
    return {
        "total_usd": total,
        "call_count": len(usages),
        "by_call": by_call,
        "input_tokens": sum(int(e.get("input_tokens", 0) or 0) for e in usages),
        "output_tokens": sum(int(e.get("output_tokens", 0) or 0) for e in usages),
    }


def build_equity_series(events: list[dict], cfg: AgentConfig) -> dict:
    """Reconstruct portfolio value at each snapshot, plus the equal-weight
    buy-and-hold benchmark over the same period.

    The benchmark is computed by backtest._buy_and_hold_series() — the same
    function the backtester uses — fed with synthetic single-price "bars"
    built from the logged snapshots. Reusing it means the dashboard and the
    backtest can never quietly disagree about what "buy and hold" means.
    """
    snaps = [e for e in events if e.get("event") == "market_snapshot"]
    if not snaps:
        return {"points": [], "benchmark": [], "dates": []}

    # Replay fills chronologically so the portfolio can be valued at each
    # snapshot using that snapshot's own prices.
    fills = [
        e
        for e in events
        if e.get("event") in ("simulated_fill", "order_executed")
    ]

    dates: list[str] = []
    bars: dict[str, list[dict]] = {}
    equity: list[float] = []

    cash = cfg.initial_cash
    shares: dict[str, float] = {}
    fill_idx = 0
    seen_dates: set[str] = set()

    for snap in snaps:
        ts = str(snap.get("ts", ""))
        date = ts[:10] or ts
        if not date or date in seen_dates:
            continue
        seen_dates.add(date)

        prices = {}
        for ticker, data in (snap.get("snapshot") or {}).items():
            if isinstance(data, dict) and isinstance(
                data.get("current_price"), (int, float)
            ):
                prices[str(ticker).upper().strip()] = float(data["current_price"])
        if not prices:
            continue

        # Apply every fill that happened at or before this snapshot.
        while fill_idx < len(fills) and str(fills[fill_idx].get("ts", "")) <= ts:
            fill = fills[fill_idx]
            fill_idx += 1
            order = fill.get("order") or {}
            t = str(order.get("ticker") or fill.get("ticker") or "").upper().strip()
            side = str(order.get("side") or fill.get("side") or "").lower()
            amount = order.get("dollars", fill.get("dollars"))
            fill_price = fill.get("simulated_price") or order.get("current_price")
            if not t or not isinstance(fill_price, (int, float)) or not fill_price:
                continue
            if side == "buy" and isinstance(amount, (int, float)):
                shares[t] = shares.get(t, 0.0) + float(amount) / float(fill_price)
                cash -= float(amount)
            elif side == "sell":
                cash += shares.get(t, 0.0) * float(fill_price)
                shares[t] = 0.0

        dates.append(date)
        equity.append(cash + sum(shares.get(t, 0.0) * p for t, p in prices.items()))

        for ticker, price in prices.items():
            bars.setdefault(ticker, []).append(
                {"date": date, "open": price, "high": price, "low": price, "close": price}
            )

    if not dates:
        return {"points": [], "benchmark": [], "dates": []}

    date_index = {
        ticker: {row["date"]: i for i, row in enumerate(rows)}
        for ticker, rows in bars.items()
    }
    benchmark = _buy_and_hold_series(
        [t for t in cfg.watchlist],
        bars,
        date_index,
        dates,
        cfg.initial_cash,
        cfg.slippage_pct,
    )

    return {"points": equity, "benchmark": benchmark or [], "dates": dates}


def compute_group_exposure(portfolio: dict | None, cfg: AgentConfig) -> list[dict]:
    positions = (portfolio or {}).get("positions") or {}
    values = {}
    for ticker, pos in positions.items():
        if isinstance(pos, dict):
            values[str(ticker).upper().strip()] = float(
                pos.get("shares", 0) or 0
            ) * float(pos.get("last_price", 0) or 0)

    out = []
    for group_name, tickers in cfg.correlation_groups.items():
        exposure = sum(values.get(str(t).upper().strip(), 0.0) for t in tickers)
        out.append(
            {
                "name": group_name,
                "tickers": list(tickers),
                "exposure": exposure,
                "cap": cfg.max_group_dollars,
                "pct": min(exposure / cfg.max_group_dollars * 100, 100.0)
                if cfg.max_group_dollars
                else 0.0,
                "over": exposure > cfg.max_group_dollars,
            }
        )
    return out


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _esc(value) -> str:
    return html.escape("" if value is None else str(value))


def _money(value) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _render_svg_chart(series: dict) -> str:
    points = series.get("points") or []
    benchmark = series.get("benchmark") or []
    dates = series.get("dates") or []

    if len(points) < 2:
        return (
            '<p class="empty">Not enough logged cycles yet to plot an equity '
            "curve. At least two cycles with a market snapshot are needed.</p>"
        )

    width, height, pad = 860, 280, 44
    all_values = points + [b for b in benchmark if isinstance(b, (int, float))]
    lo, hi = min(all_values), max(all_values)
    if hi == lo:
        hi = lo + 1.0

    def to_xy(i, value, count):
        x = pad + (width - 2 * pad) * (i / max(count - 1, 1))
        y = height - pad - (height - 2 * pad) * ((value - lo) / (hi - lo))
        return x, y

    def path(values):
        return " ".join(
            f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
            for i, (x, y) in enumerate(
                to_xy(i, v, len(values)) for i, v in enumerate(values)
            )
        )

    bench_path = (
        f'<path d="{path(benchmark)}" fill="none" stroke="#8b93a7" '
        'stroke-width="2" stroke-dasharray="6 4" />'
        if len(benchmark) >= 2
        else ""
    )

    bench_legend = (
        '<span class="legend"><i style="background:#8b93a7"></i>'
        "Equal-weight buy &amp; hold</span>"
        if len(benchmark) >= 2
        else '<span class="legend warn">Benchmark unavailable for this period</span>'
    )

    return f"""
<div class="chart-legend">
  <span class="legend"><i style="background:#3b82f6"></i>Strategy</span>
  {bench_legend}
</div>
<svg viewBox="0 0 {width} {height}" class="chart" role="img"
     aria-label="Portfolio value over time against an equal-weight buy-and-hold benchmark">
  <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#d5d9e3"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#d5d9e3"/>
  <text x="{pad - 6}" y="{pad + 4}" class="axis" text-anchor="end">{_money(hi)}</text>
  <text x="{pad - 6}" y="{height - pad}" class="axis" text-anchor="end">{_money(lo)}</text>
  <text x="{pad}" y="{height - pad + 18}" class="axis">{_esc(dates[0])}</text>
  <text x="{width - pad}" y="{height - pad + 18}" class="axis" text-anchor="end">{_esc(dates[-1])}</text>
  {bench_path}
  <path d="{path(points)}" fill="none" stroke="#3b82f6" stroke-width="2.5"/>
</svg>
"""


def _render_decisions(rows: list[dict]) -> str:
    if not rows:
        return '<p class="empty">No decisions logged yet.</p>'

    tickers = sorted({r["ticker"] for r in rows if r["ticker"]})
    verdicts = sorted({r["verdict"] for r in rows})

    ticker_opts = "".join(f'<option value="{_esc(t)}">{_esc(t)}</option>' for t in tickers)
    verdict_opts = "".join(f'<option value="{_esc(v)}">{_esc(v)}</option>' for v in verdicts)

    body = []
    for i, r in enumerate(rows):
        cls = "neg" if r["verdict"] in NEGATIVE_VERDICTS else ""
        if r["verdict"] == "AUTO-APPROVED":
            cls = "auto"

        numbers = "".join(
            f"<li><b>{_esc(k)}</b>: <code>{_esc(json.dumps(v))}</code></li>"
            for k, v in r["model_numbers"].items()
        ) or "<li class='empty'>The model reported no raw numbers for this order.</li>"

        arithmetic = "".join(
            f"<li><code>{_esc(line)}</code></li>" for line in r["arithmetic"]
        ) or "<li class='empty'>No re-derivable arithmetic for this order type.</li>"

        disagree_banner = (
            '<div class="disagree">MODEL AND CODE DISAGREE — the model proposed '
            "this order; the guardrails rejected it. The guardrail verdict is "
            "what actually happened.</div>"
            if r["disagrees"]
            else ""
        )

        body.append(f"""
<tr class="row {cls}" data-ticker="{_esc(r['ticker'])}" data-verdict="{_esc(r['verdict'])}"
    data-date="{_esc(r['date'])}" onclick="toggle({i})">
  <td>{_esc(r['ts'])}</td>
  <td><b>{_esc(r['ticker'])}</b></td>
  <td>{_esc(r['side'].upper())}</td>
  <td>{_money(r['dollars'])}</td>
  <td><span class="badge b-{_esc(r['verdict'].lower().replace(' ', '-'))}">{_esc(r['verdict'])}</span></td>
  <td class="reason">{_esc(r['reason'])}</td>
</tr>
<tr class="why" id="why-{i}" data-ticker="{_esc(r['ticker'])}" data-verdict="{_esc(r['verdict'])}"
    data-date="{_esc(r['date'])}">
  <td colspan="6">
    {disagree_banner}
    <div class="why-grid">
      <div class="half model">
        <h4>WHAT THE MODEL SAID</h4>
        <p class="caveat">A narrative the language model produced. Not verified.
           Shown so it can be compared against the check on the right.</p>
        <p class="prose">{_esc(r['model_reason']) or '<i>No prose reasoning recorded.</i>'}</p>
        <ul>{numbers}</ul>
      </div>
      <div class="half code">
        <h4>WHAT THE CODE DECIDED</h4>
        <p class="caveat">Deterministic. Computed by guardrails.py from the raw
           numbers, independently of the model's reasoning.</p>
        <p class="prose"><b>{_esc(r['reason'])}</b></p>
        <ul>{arithmetic}</ul>
      </div>
    </div>
  </td>
</tr>""")

    return f"""
<div class="filters">
  <label>Ticker
    <select id="f-ticker" onchange="applyFilters()">
      <option value="">All</option>{ticker_opts}
    </select>
  </label>
  <label>Verdict
    <select id="f-verdict" onchange="applyFilters()">
      <option value="">All</option>{verdict_opts}
    </select>
  </label>
  <label>Date
    <input type="date" id="f-date" onchange="applyFilters()">
  </label>
  <button onclick="clearFilters()">Clear</button>
  <span class="hint">Click any row to see why it was decided that way.</span>
</div>
<table class="decisions">
  <thead><tr>
    <th>Timestamp</th><th>Ticker</th><th>Side</th><th>Dollars</th>
    <th>Guardrail verdict</th><th>Reason</th>
  </tr></thead>
  <tbody>{''.join(body)}</tbody>
</table>
"""


def render_dashboard(
    events: list[dict], portfolio: dict | None, cfg: AgentConfig
) -> str:
    mode = determine_mode(events)
    health = summarize_health(events)
    snapshots = collect_snapshots(events)
    decisions = collect_decisions(events, cfg)
    cost = collect_api_cost(events)
    series = build_equity_series(events, cfg)
    groups = compute_group_exposure(portfolio, cfg)
    mismatches = [e for e in events if e.get("event") == "execution_mismatch"]

    mode_copy = {
        "PAPER": ("PAPER TRADING", "Every fill below is simulated. No real money moved."),
        "LIVE": ("LIVE TRADING", "These are real orders against a real account."),
        "MIXED": (
            "MIXED — PAPER AND LIVE IN ONE LOG",
            "This log contains both simulated and real fills. They are NOT "
            "combined into a single P&L figure below, and should not be read "
            "as one track record.",
        ),
        "UNKNOWN": (
            "NO FILLS RECORDED",
            "No simulated or real fills in this log yet, so no trading mode "
            "can be determined.",
        ),
    }[mode]

    # Health section
    if not health["has_cycles"]:
        health_detail = "No cycles have ever been logged."
    else:
        missing = health["missing_tickers"]
        health_detail = (
            f"Last cycle's market snapshot was incomplete — missing "
            f"{', '.join(missing)}. Inaction this cycle is UNEXPLAINED: it "
            "cannot be distinguished from a failed data fetch."
            if health["status"] == "WARNING" and missing
            else (
                "Last cycle did not record a complete market snapshot. "
                "Inaction cannot be distinguished from a failed data fetch."
                if health["status"] == "WARNING"
                else "Last cycle recorded a complete market snapshot for every "
                "watchlist ticker. A no-trade result is trustworthy."
            )
        )

    snapshot_html = ""
    for snap in snapshots:
        rows = "".join(
            f"<tr><td><b>{_esc(r['ticker'])}</b></td>"
            f"<td>{_money(r['current_price'])}</td>"
            f"<td class=\"{'neg-text' if isinstance(r['day_change_pct'], (int, float)) and r['day_change_pct'] < 0 else 'pos-text'}\">"
            f"{r['day_change_pct']:+.2f}%</td></tr>"
            if isinstance(r["day_change_pct"], (int, float))
            else f"<tr><td><b>{_esc(r['ticker'])}</b></td><td>{_money(r['current_price'])}</td><td>—</td></tr>"
            for r in snap["rows"]
        )
        snapshot_html += (
            f'<div class="snap"><h4>{_esc(snap["ts"])}</h4>'
            f'<table class="mini"><thead><tr><th>Ticker</th><th>Price</th>'
            f"<th>Day change</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )
    if not snapshot_html:
        snapshot_html = (
            '<p class="empty">No market snapshots logged yet. Until one appears, '
            "a zero-order cycle cannot be shown to have seen real data.</p>"
        )

    # Portfolio
    if portfolio:
        pos_rows = "".join(
            f"<tr><td><b>{_esc(t)}</b></td><td>{float(p.get('shares', 0)):.4f}</td>"
            f"<td>{_money(p.get('avg_cost'))}</td><td>{_money(p.get('last_price'))}</td>"
            f"<td>{_money(float(p.get('shares', 0)) * float(p.get('last_price', 0)))}</td></tr>"
            for t, p in sorted((portfolio.get("positions") or {}).items())
            if isinstance(p, dict)
        )
        positions_value = sum(
            float(p.get("shares", 0) or 0) * float(p.get("last_price", 0) or 0)
            for p in (portfolio.get("positions") or {}).values()
            if isinstance(p, dict)
        )
        cash = float(portfolio.get("cash", 0) or 0)
        total_pnl = cash + positions_value - cfg.initial_cash
        pnl_label = "Simulated P&L" if mode in ("PAPER", "MIXED") else "P&L"
        portfolio_html = f"""
<div class="cards">
  <div class="card"><div class="k">Cash</div><div class="v">{_money(cash)}</div></div>
  <div class="card"><div class="k">Positions value</div><div class="v">{_money(positions_value)}</div></div>
  <div class="card"><div class="k">{_esc(pnl_label)}</div>
    <div class="v {'pos-text' if total_pnl >= 0 else 'neg-text'}">{_money(total_pnl)}</div></div>
  <div class="card cost"><div class="k">Cumulative API cost (est.)</div>
    <div class="v">{_money(cost['total_usd'])}</div>
    <div class="sub">{cost['call_count']} calls · {cost['input_tokens']:,} in / {cost['output_tokens']:,} out tokens</div></div>
</div>
<p class="caveat">API cost sits next to P&amp;L deliberately: at small account
sizes the cost of running the agent can rival or exceed what it earns. The
cost figure is an estimate from operator-supplied prices in config.py, not a
billed amount.</p>
<table class="mini"><thead><tr><th>Ticker</th><th>Shares</th><th>Avg cost</th>
<th>Last price</th><th>Value</th></tr></thead><tbody>{pos_rows or '<tr><td colspan="5" class="empty">No open positions.</td></tr>'}</tbody></table>
"""
    else:
        portfolio_html = (
            '<p class="empty">No paper_portfolio.json found. Run a dry-run cycle '
            "to create one.</p>"
            f'<div class="cards"><div class="card cost"><div class="k">'
            f'Cumulative API cost (est.)</div><div class="v">{_money(cost["total_usd"])}</div>'
            f'<div class="sub">{cost["call_count"]} calls</div></div></div>'
        )

    group_html = "".join(
        f"""<div class="bar-row">
  <div class="bar-label"><b>{_esc(g['name'])}</b>
    <span class="muted">{_esc(', '.join(g['tickers']))}</span></div>
  <div class="bar"><div class="fill {'over' if g['over'] else ''}" style="width:{g['pct']:.1f}%"></div></div>
  <div class="bar-value">{_money(g['exposure'])} / {_money(g['cap'])}</div>
</div>"""
        for g in groups
    ) or '<p class="empty">No correlation groups configured.</p>'

    if mismatches:
        mismatch_html = "".join(
            f'<div class="mismatch"><b>{_esc(m.get("ts"))}</b>'
            f'<p>{_esc(m.get("reason"))}</p>'
            f'<pre>{_esc(json.dumps(m.get("order"), indent=2))}</pre></div>'
            for m in mismatches
        )
    else:
        mismatch_html = (
            '<p class="ok-note">No execution mismatches recorded. Every executed '
            "order matched what the guardrails approved, on symbol, side, count, "
            "and dollar amount.</p>"
        )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Dashboard</title>
<style>
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; padding:0 0 60px; font:15px/1.55 -apple-system,BlinkMacSystemFont,
  "Segoe UI",Roboto,Helvetica,Arial,sans-serif; color:#1a1d24; background:#f4f6fa; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:0 20px; }}
h1 {{ font-size:22px; margin:28px 0 4px; }}
h2 {{ font-size:13px; letter-spacing:.09em; text-transform:uppercase; color:#5b6478;
  margin:38px 0 12px; padding-bottom:7px; border-bottom:1px solid #dde1ea; }}
h4 {{ margin:0 0 6px; font-size:13px; }}
section {{ background:#fff; border:1px solid #e2e6ee; border-radius:10px;
  padding:18px 20px; margin-bottom:6px; }}
.mode {{ padding:20px; border-radius:10px; margin:18px 0 6px; color:#fff; }}
.mode h2 {{ border:0; margin:0 0 4px; color:#fff; font-size:20px; letter-spacing:.04em; }}
.mode p {{ margin:0; opacity:.95; }}
.mode.paper {{ background:#2563eb; }}
.mode.live {{ background:#b91c1c; }}
.mode.mixed {{ background:#b45309; }}
.mode.unknown {{ background:#4b5563; }}
.health {{ display:flex; align-items:center; gap:22px; flex-wrap:wrap; }}
.status {{ font-size:40px; font-weight:700; letter-spacing:.04em; padding:14px 30px;
  border-radius:10px; }}
.status.ok {{ background:#dcfce7; color:#14532d; }}
.status.warning {{ background:#fee2e2; color:#7f1d1d; }}
.health-meta {{ flex:1; min-width:260px; }}
.health-meta .big {{ font-size:17px; font-weight:600; }}
.muted {{ color:#6b7280; font-weight:400; }}
.caveat {{ font-size:12.5px; color:#6b7280; margin:6px 0; }}
.empty {{ color:#8a91a0; font-style:italic; }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
th {{ text-align:left; font-size:11.5px; text-transform:uppercase; letter-spacing:.05em;
  color:#6b7280; border-bottom:1px solid #e2e6ee; padding:7px 9px; }}
td {{ padding:8px 9px; border-bottom:1px solid #f0f2f7; vertical-align:top; }}
table.mini td, table.mini th {{ padding:6px 9px; }}
.snap {{ display:inline-block; vertical-align:top; min-width:290px; margin:0 18px 14px 0; }}
.snap h4 {{ color:#6b7280; font-weight:500; font-size:12px; }}
.pos-text {{ color:#15803d; }} .neg-text {{ color:#b91c1c; }}
.row {{ cursor:pointer; }}
.row:hover {{ background:#f7f9fc; }}
.row.neg {{ background:#fff7f7; }}
.row.neg:hover {{ background:#ffefef; }}
.row.auto {{ background:#fffbeb; }}
.badge {{ font-size:11px; font-weight:600; padding:3px 8px; border-radius:20px;
  white-space:nowrap; }}
.b-blocked {{ background:#fee2e2; color:#7f1d1d; }}
.b-skipped {{ background:#e5e7eb; color:#374151; }}
.b-auto-approved {{ background:#fef3c7; color:#78350f; }}
.b-executed {{ background:#dcfce7; color:#14532d; }}
.b-simulated {{ background:#dbeafe; color:#1e3a8a; }}
.reason {{ color:#4b5563; max-width:340px; }}
.why {{ display:none; background:#f8fafc; }}
.why.open {{ display:table-row; }}
.why-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.half {{ padding:13px 15px; border-radius:8px; }}
.half.model {{ background:#fdf7ed; border:1px solid #f0dcc0; }}
.half.code {{ background:#eef6ff; border:1px solid #cfe2f7; }}
.half h4 {{ letter-spacing:.06em; font-size:11.5px; }}
.half.model h4 {{ color:#92400e; }}
.half.code h4 {{ color:#1e40af; }}
.half ul {{ margin:8px 0 0; padding-left:17px; }}
.half li {{ margin:3px 0; font-size:12.5px; }}
.prose {{ margin:7px 0; font-size:13px; }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px;
  background:rgba(0,0,0,.05); padding:1px 4px; border-radius:3px; }}
.disagree {{ background:#7f1d1d; color:#fff; padding:9px 13px; border-radius:7px;
  font-size:12.5px; font-weight:600; margin-bottom:12px; }}
.filters {{ display:flex; gap:14px; align-items:flex-end; flex-wrap:wrap; margin-bottom:12px; }}
.filters label {{ display:flex; flex-direction:column; font-size:11.5px;
  text-transform:uppercase; letter-spacing:.05em; color:#6b7280; gap:4px; }}
.filters select, .filters input, .filters button {{ font:inherit; font-size:13px;
  padding:5px 9px; border:1px solid #d5d9e3; border-radius:6px; background:#fff; }}
.filters button {{ cursor:pointer; }}
.hint {{ font-size:12px; color:#8a91a0; margin-left:auto; }}
.cards {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:6px; }}
.card {{ flex:1; min-width:170px; background:#f8fafc; border:1px solid #e2e6ee;
  border-radius:9px; padding:13px 15px; }}
.card.cost {{ background:#fdf7ed; border-color:#f0dcc0; }}
.card .k {{ font-size:11.5px; text-transform:uppercase; letter-spacing:.05em; color:#6b7280; }}
.card .v {{ font-size:22px; font-weight:600; margin-top:3px; }}
.card .sub {{ font-size:11.5px; color:#8a91a0; margin-top:3px; }}
.chart {{ width:100%; height:auto; }}
.axis {{ font-size:11px; fill:#8a91a0; }}
.chart-legend {{ display:flex; gap:18px; font-size:12.5px; margin-bottom:6px; }}
.legend {{ display:flex; align-items:center; gap:6px; color:#4b5563; }}
.legend i {{ width:16px; height:3px; border-radius:2px; display:inline-block; }}
.legend.warn {{ color:#92400e; }}
.bar-row {{ display:flex; align-items:center; gap:14px; margin:11px 0; }}
.bar-label {{ min-width:210px; font-size:13px; }}
.bar {{ flex:1; height:19px; background:#eef1f6; border-radius:5px; overflow:hidden; }}
.fill {{ height:100%; background:#3b82f6; }}
.fill.over {{ background:#b91c1c; }}
.bar-value {{ min-width:150px; text-align:right; font-size:12.5px; color:#4b5563; }}
.mismatch {{ background:#fee2e2; border:1px solid #fca5a5; border-radius:8px;
  padding:13px 15px; margin-bottom:10px; }}
.mismatch pre {{ font-size:11.5px; overflow-x:auto; background:rgba(0,0,0,.04);
  padding:8px; border-radius:5px; }}
.ok-note {{ background:#dcfce7; color:#14532d; padding:11px 14px; border-radius:8px; }}
footer {{ text-align:center; color:#8a91a0; font-size:12px; margin-top:26px; }}
.readonly {{ background:#eef1f6; border:1px dashed #c3cad8; border-radius:8px;
  padding:10px 14px; font-size:12.5px; color:#5b6478; margin-top:14px; }}
</style></head>
<body><div class="wrap">

<h1>Agent Dashboard</h1>
<p class="muted">Generated {generated} · static snapshot of the log</p>

<div class="mode {mode.lower()}">
  <h2>{_esc(mode_copy[0])}</h2>
  <p>{_esc(mode_copy[1])}</p>
</div>

<h2>1 · Health</h2>
<section>
  <div class="health">
    <div class="status {health['status'].lower()}">{health['status']}</div>
    <div class="health-meta">
      <div class="big">Last cycle: {_esc(health['last_ts'] or 'never')} <span class="muted">({_esc(health['age_text'])})</span></div>
      <div class="muted">{health['cycle_count']} cycles logged in total</div>
      <p class="caveat">{_esc(health_detail)}</p>
    </div>
  </div>
  <p class="caveat">A stale timestamp is itself a finding: it means a scheduled
  run stopped without anyone noticing. Silence is not the same as no signal.</p>
</section>

<h2>2 · Market snapshots <span class="muted">— data in</span></h2>
<section>{snapshot_html}</section>

<h2>3 · Decisions <span class="muted">— model proposed → code decided</span></h2>
<section>{_render_decisions(decisions)}</section>

<h2>4 · Portfolio &amp; cost</h2>
<section>{portfolio_html}</section>

<h2>5 · Equity vs benchmark</h2>
<section>
  {_render_svg_chart(series)}
  <p class="caveat">The benchmark is an equal-weight buy-and-hold of the
  watchlist over the same period, computed by the same function the backtester
  uses. An equity line without a benchmark says nothing about whether the
  strategy added anything.</p>
</section>

<h2>6 · Group exposure</h2>
<section>{group_html}</section>

<h2>7 · Reconciliation</h2>
<section>{mismatch_html}</section>

<div class="readonly"><b>This page is read-only.</b> It is a static file
generated from the log and has no connection back to the agent — there is
deliberately no kill switch, mode toggle, or order queue here, because a
control that appeared to work but did nothing would be more dangerous than
none. Change src/config.py and re-run the agent.</div>

<footer>Generated by scripts/build_dashboard.py · no network calls, no external
dependencies</footer>
</div>
<script>
function toggle(i) {{
  var el = document.getElementById('why-' + i);
  if (el) el.classList.toggle('open');
}}
function applyFilters() {{
  var t = document.getElementById('f-ticker').value;
  var v = document.getElementById('f-verdict').value;
  var d = document.getElementById('f-date').value;
  document.querySelectorAll('tr.row').forEach(function (row) {{
    var show = (!t || row.dataset.ticker === t)
            && (!v || row.dataset.verdict === v)
            && (!d || row.dataset.date === d);
    row.style.display = show ? '' : 'none';
    var why = row.nextElementSibling;
    if (why && why.classList.contains('why')) {{
      if (!show) {{ why.classList.remove('open'); }}
      why.style.display = show ? '' : 'none';
      if (show && !why.classList.contains('open')) {{ why.style.display = 'none'; }}
    }}
  }});
}}
function clearFilters() {{
  document.getElementById('f-ticker').value = '';
  document.getElementById('f-verdict').value = '';
  document.getElementById('f-date').value = '';
  applyFilters();
}}
</script>
</body></html>"""


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp", "ticker", "side", "dollars", "price", "verdict", "reason"]
        )
        for r in rows:
            writer.writerow(
                [
                    r["ts"],
                    r["ticker"],
                    r["side"],
                    r["dollars"] if r["dollars"] is not None else "",
                    r["price"] if r["price"] is not None else "",
                    r["verdict"],
                    r["reason"],
                ]
            )


def build(
    log_path: str = DEFAULT_LOG,
    portfolio_path: str = DEFAULT_PORTFOLIO,
    out_path: str = DEFAULT_OUT,
    csv_path: str | None = None,
    cfg: AgentConfig | None = None,
) -> str:
    cfg = cfg or AgentConfig()
    events = load_events(log_path)
    portfolio = load_portfolio(portfolio_path)

    html_text = render_dashboard(events, portfolio, cfg)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_text)

    if csv_path:
        write_csv(collect_decisions(events, cfg), csv_path)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default=DEFAULT_LOG, help="Path to trade_log.jsonl")
    parser.add_argument(
        "--portfolio", default=DEFAULT_PORTFOLIO, help="Path to paper_portfolio.json"
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output HTML path")
    parser.add_argument("--open", action="store_true", help="Open the file in a browser")
    parser.add_argument(
        "--csv", action="store_true", help="Also write trades.csv alongside the HTML"
    )
    args = parser.parse_args()

    out = build(
        log_path=args.log,
        portfolio_path=args.portfolio,
        out_path=args.out,
        csv_path=DEFAULT_CSV if args.csv else None,
    )
    print(f"Wrote {out}")
    if args.csv:
        print(f"Wrote {DEFAULT_CSV}")
    if args.open:
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
