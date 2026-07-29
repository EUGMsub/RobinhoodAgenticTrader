"""
backtest.py
===========
A pure historical replay engine for the mean-reversion strategy. No network
calls, no LLM calls — it exists to answer "what would this strategy have
done" using the EXACT same guardrail functions that gate real orders.

Design constraint: validate_batch() from guardrails.py is called unchanged,
with the same GuardrailConfig the live agent uses (via
AgentConfig.guardrail_config()). This engine only supplies the day-by-day
inputs that function already expects (day_change_pct, positions,
current_price, avg_cost, days_held) and interprets its approve/reject
verdicts as trading decisions. If guardrails.py changes, this engine tests
the new behavior automatically — it is not a reimplementation of the rules.

Each trading day's full candidate set (every watchlist ticker's buy signal
plus every held position's sell signal) is validated in ONE validate_batch()
call rather than one validate_buy()/validate_sell() call per ticker. That
matters specifically for the correlation-group cap: if several watchlist
tickers dip on the same day, checking them one at a time against the same
positions snapshot would let each clear the group cap independently and
collectively overshoot it. validate_batch() checks them cumulatively against
a running snapshot instead, so the group cap holds for the whole day's batch,
not just per order.

Execution model:
- Each trading day, guardrails are evaluated against that day's CLOSE.
- An approved order is recorded PENDING and fills at the NEXT bar's OPEN
  for that ticker — never the signal day's own close. This avoids
  look-ahead bias: the "next open" is the first price the strategy could
  actually have traded at after seeing the signal.
- If a ticker has no bar after the signal day (end of the provided data),
  the pending order is discarded rather than filled, since there is no
  real price to fill it at.
- A pending buy is also discarded if there isn't enough cash to cover it;
  guardrails.py enforces policy limits (order/position/group/account
  caps), not account liquidity, so that check happens here at fill time.
- Every fill pays slippage against the next bar's raw open: worse for the
  side placing the order in both directions (buys pay up, sells receive
  less), matching how a real market order behaves. Both the raw open and
  the slippage-adjusted fill price are recorded on the trade.

Alongside the strategy replay, the same bar data is used to compute two
buy-and-hold benchmarks (equal-weight across the watchlist, and VOO alone)
over the identical date range, initial cash, and opening-purchase slippage,
so the strategy's numbers have something to sit next to.
"""

from dataclasses import dataclass
from datetime import date as Date

from config import AgentConfig
from guardrails import validate_batch


@dataclass
class Position:
    shares: float
    avg_cost: float
    acquired_date: str


@dataclass
class OrderRecord:
    ticker: str
    side: str  # "buy" | "sell"
    signal_date: str
    reason: str  # the guardrail's approval reason
    status: str  # "filled" | "discarded_no_next_bar" | "discarded_insufficient_cash"
    fill_date: str | None = None
    raw_open_price: float | None = None  # the next bar's open, before slippage
    fill_price: float | None = None  # raw_open_price adjusted for slippage
    shares: float | None = None
    dollars: float | None = None
    realized_pnl: float | None = None
    hold_days: int | None = None  # sells only: fill_date - position.acquired_date


@dataclass
class EquityPoint:
    date: str
    cash: float
    positions_value: float
    total_equity: float


@dataclass
class BenchmarkResult:
    name: str
    final_value: float
    total_return_pct: float
    max_drawdown_pct: float


@dataclass
class BacktestResult:
    orders: list[OrderRecord]
    equity_curve: list[EquityPoint]
    final_cash: float
    final_positions: dict[str, Position]
    final_portfolio_value: float
    total_return_pct: float
    max_drawdown_pct: float
    round_trips: int
    win_rate_pct: float
    avg_hold_days: float
    # None when the relevant ticker(s) have no bars in the provided data.
    equal_weight_benchmark: BenchmarkResult | None
    voo_benchmark: BenchmarkResult | None


def _parse_date(s: str) -> Date:
    return Date.fromisoformat(str(s)[:10])


def _max_drawdown_pct(values: list[float]) -> float:
    """Largest peak-to-trough decline in a chronological value series, as a
    positive percentage (0.0 if the series never drops below its running
    peak, or is empty)."""
    peak = None
    max_dd = 0.0
    for v in values:
        if peak is None or v > peak:
            peak = v
        if peak and peak > 0:
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd
    return max_dd


def _buy_and_hold_series(
    tickers: list[str],
    sorted_bars: dict[str, list[dict]],
    date_index: dict[str, dict[str, int]],
    master_dates: list[str],
    initial_cash: float,
    slippage_pct: float,
) -> list[float] | None:
    """Equal-weight buy-and-hold of `tickers` (each bought at its own first
    available bar's open, with buy-side slippage). Returns the chronological
    total-value series, or None if none of the tickers have any bars."""
    available = [t for t in tickers if sorted_bars.get(t)]
    if not available:
        return None

    allocation = initial_cash / len(available)
    cash = initial_cash
    shares: dict[str, float] = {}
    last_close: dict[str, float] = {}
    bought: set[str] = set()

    values: list[float] = []
    for today in master_dates:
        for ticker in available:
            idx = date_index[ticker].get(today)
            if idx is None:
                continue
            bar = sorted_bars[ticker][idx]
            if ticker not in bought:
                buy_price = float(bar["open"]) * (1 + slippage_pct / 100)
                shares[ticker] = allocation / buy_price
                cash -= allocation
                bought.add(ticker)
            last_close[ticker] = float(bar["close"])
        total = cash + sum(
            shares.get(t, 0.0) * last_close.get(t, 0.0) for t in available
        )
        values.append(total)
    return values


def _benchmark(
    name: str,
    tickers: list[str],
    sorted_bars: dict[str, list[dict]],
    date_index: dict[str, dict[str, int]],
    master_dates: list[str],
    initial_cash: float,
    slippage_pct: float,
) -> BenchmarkResult | None:
    values = _buy_and_hold_series(
        tickers, sorted_bars, date_index, master_dates, initial_cash, slippage_pct
    )
    if values is None:
        return None
    final_value = values[-1]
    return BenchmarkResult(
        name=name,
        final_value=final_value,
        total_return_pct=(final_value - initial_cash) / initial_cash * 100,
        max_drawdown_pct=_max_drawdown_pct(values),
    )


def run_backtest(
    bars: dict[str, list[dict]],
    cfg: AgentConfig,
    starting_cash: float | None = None,
) -> BacktestResult:
    """Replay `bars` (ticker -> chronological list of {date, open, high,
    low, close}) through the strategy's guardrails and return the trades,
    equity curve, summary metrics, and buy-and-hold benchmarks that would
    have resulted. `starting_cash` defaults to cfg.initial_cash.
    """
    gcfg = cfg.guardrail_config()
    starting_cash = cfg.initial_cash if starting_cash is None else starting_cash

    # Sort each ticker's bars chronologically and index them by date so
    # "this ticker's previous/next bar" can be found without assuming all
    # tickers share an identical calendar.
    sorted_bars: dict[str, list[dict]] = {
        ticker: sorted(rows, key=lambda r: r["date"]) for ticker, rows in bars.items()
    }
    date_index: dict[str, dict[str, int]] = {
        ticker: {row["date"]: i for i, row in enumerate(rows)}
        for ticker, rows in sorted_bars.items()
    }

    master_dates = sorted({row["date"] for rows in sorted_bars.values() for row in rows})

    cash = starting_cash
    positions: dict[str, Position] = {}
    last_close: dict[str, float] = {}
    # Pending fills keyed by (fill_date, ticker, side) -> the OrderRecord to
    # resolve. Side is part of the key because a buy and a sell for the
    # same ticker can both be approved on the same signal day (the buy's
    # day_change_pct and the sell's gain-vs-avg-cost use different
    # reference prices) and can therefore share the same next-bar fill
    # date without being the same order.
    pending: dict[tuple[str, str, str], OrderRecord] = {}
    orders: list[OrderRecord] = []
    equity_curve: list[EquityPoint] = []

    def _next_bar_date(ticker: str, idx: int) -> str | None:
        rows = sorted_bars[ticker]
        return rows[idx + 1]["date"] if idx + 1 < len(rows) else None

    for today in master_dates:
        # 1. Resolve any pending orders scheduled to fill today, at that
        #    ticker's OPEN price for today, adjusted for slippage. Sells
        #    resolve before buys so a sell always closes out the position
        #    exactly as it stood on its own signal day, never one already
        #    topped up by a same-day incoming buy fill.
        todays_fills = [key for key in pending if key[0] == today]
        todays_fills.sort(key=lambda key: 0 if key[2] == "sell" else 1)

        for key in todays_fills:
            fill_date, ticker, side = key
            record = pending.pop(key)
            idx = date_index[ticker][today]
            raw_open = float(sorted_bars[ticker][idx]["open"])
            slippage_mult = (
                1 + cfg.slippage_pct / 100
                if side == "buy"
                else 1 - cfg.slippage_pct / 100
            )
            fill_price = raw_open * slippage_mult

            if side == "buy":
                dollars = record.dollars
                if dollars > cash + 1e-9:
                    record.status = "discarded_insufficient_cash"
                    orders.append(record)
                    continue
                shares_bought = dollars / fill_price
                cash -= dollars
                existing = positions.get(ticker)
                if existing:
                    total_shares = existing.shares + shares_bought
                    existing.avg_cost = (
                        existing.shares * existing.avg_cost + dollars
                    ) / total_shares
                    existing.shares = total_shares
                else:
                    positions[ticker] = Position(
                        shares=shares_bought,
                        avg_cost=fill_price,
                        acquired_date=today,
                    )
                record.status = "filled"
                record.fill_date = today
                record.raw_open_price = raw_open
                record.fill_price = fill_price
                record.shares = shares_bought
                orders.append(record)
            else:  # sell
                position = positions.get(ticker)
                if position is None:
                    # Position was already closed by another path; nothing
                    # left to sell.
                    record.status = "discarded_no_next_bar"
                    orders.append(record)
                    continue
                proceeds = position.shares * fill_price
                cost_basis = position.shares * position.avg_cost
                cash += proceeds
                record.status = "filled"
                record.fill_date = today
                record.raw_open_price = raw_open
                record.fill_price = fill_price
                record.shares = position.shares
                record.dollars = proceeds
                record.realized_pnl = proceeds - cost_basis
                record.hold_days = (
                    _parse_date(today) - _parse_date(position.acquired_date)
                ).days
                orders.append(record)
                del positions[ticker]

        # 2. Update carried-forward close prices for every ticker with a
        #    bar today (used both for position valuation and tomorrow's
        #    day_change_pct).
        for ticker, idx_map in date_index.items():
            idx = idx_map.get(today)
            if idx is not None:
                last_close[ticker] = float(sorted_bars[ticker][idx]["close"])

        # 3. Build today's positions dict (ticker -> market value at
        #    today's close) exactly as agent.py builds it from live data.
        positions_value = {
            ticker: last_close.get(ticker, 0.0) * pos.shares
            for ticker, pos in positions.items()
        }

        # 4. Build today's full candidate set — every watchlist ticker's
        #    buy signal, every held position's sell signal — and validate
        #    it as ONE batch so a market-wide dip can't let each ticker's
        #    buy independently clear the group cap (see validate_batch()).
        candidates: list[dict] = []

        for ticker in cfg.watchlist:
            idx = date_index.get(ticker, {}).get(today)
            if idx is None or idx == 0:
                continue  # no bar today, or no prior close to diff against
            prev_close = float(sorted_bars[ticker][idx - 1]["close"])
            today_close = float(sorted_bars[ticker][idx]["close"])
            day_change_pct = (today_close - prev_close) / prev_close * 100
            candidates.append(
                {
                    "ticker": ticker,
                    "side": "buy",
                    "dollars": cfg.order_dollars,
                    "day_change_pct": day_change_pct,
                }
            )

        for ticker, pos in list(positions.items()):
            idx = date_index.get(ticker, {}).get(today)
            if idx is None:
                continue
            current_price = float(sorted_bars[ticker][idx]["close"])
            days_held = (_parse_date(today) - _parse_date(pos.acquired_date)).days
            candidates.append(
                {
                    "ticker": ticker,
                    "side": "sell",
                    "current_price": current_price,
                    "avg_cost": pos.avg_cost,
                    "days_held": days_held,
                }
            )

        for order, result in validate_batch(gcfg, candidates, positions_value):
            if not result.approved:
                continue

            ticker = order["ticker"]
            side = order["side"]
            idx = date_index[ticker][today]
            fill_date = _next_bar_date(ticker, idx)
            record = OrderRecord(
                ticker=ticker,
                side=side,
                signal_date=today,
                reason=result.reason,
                status="pending",
                dollars=cfg.order_dollars if side == "buy" else None,
            )
            if fill_date is None:
                record.status = "discarded_no_next_bar"
                orders.append(record)
            else:
                pending[(fill_date, ticker, side)] = record

        # 6. Snapshot equity for today.
        positions_value_today = sum(
            last_close.get(ticker, 0.0) * pos.shares for ticker, pos in positions.items()
        )
        equity_curve.append(
            EquityPoint(
                date=today,
                cash=cash,
                positions_value=positions_value_today,
                total_equity=cash + positions_value_today,
            )
        )

    # `pending` is always empty here: a fill_date is only ever set to one
    # of that same ticker's own bar dates, which is by construction part
    # of master_dates, so every pending order is resolved in the loop
    # above. Orders with no next bar are discarded at signal time instead
    # of ever entering `pending` (see the `fill_date is None` branches).

    total_equity_series = [pt.total_equity for pt in equity_curve]
    final_portfolio_value = total_equity_series[-1] if total_equity_series else starting_cash

    round_trip_orders = [o for o in orders if o.side == "sell" and o.status == "filled"]
    round_trips = len(round_trip_orders)
    win_rate_pct = (
        sum(1 for o in round_trip_orders if o.realized_pnl > 0) / round_trips * 100
        if round_trips
        else 0.0
    )
    avg_hold_days = (
        sum(o.hold_days for o in round_trip_orders) / round_trips if round_trips else 0.0
    )

    equal_weight_benchmark = _benchmark(
        "equal_weight_watchlist",
        list(cfg.watchlist),
        sorted_bars,
        date_index,
        master_dates,
        starting_cash,
        cfg.slippage_pct,
    )
    voo_benchmark = _benchmark(
        "voo_buy_and_hold",
        ["VOO"],
        sorted_bars,
        date_index,
        master_dates,
        starting_cash,
        cfg.slippage_pct,
    )

    return BacktestResult(
        orders=orders,
        equity_curve=equity_curve,
        final_cash=cash,
        final_positions=positions,
        final_portfolio_value=final_portfolio_value,
        total_return_pct=(final_portfolio_value - starting_cash) / starting_cash * 100,
        max_drawdown_pct=_max_drawdown_pct(total_equity_series),
        round_trips=round_trips,
        win_rate_pct=win_rate_pct,
        avg_hold_days=avg_hold_days,
        equal_weight_benchmark=equal_weight_benchmark,
        voo_benchmark=voo_benchmark,
    )
