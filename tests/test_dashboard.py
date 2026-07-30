"""
Unit tests for scripts/build_dashboard.py.

Every test runs against a hand-built sample log defined in this file, so
the expected output is known exactly. Nothing here touches the network,
reads the real trade_log.jsonl, or requires credentials — the dashboard is
a pure function of (events, portfolio, config).
"""

import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from build_dashboard import (
    build,
    collect_api_cost,
    collect_decisions,
    collect_recent_signals,
    determine_mode,
    render_dashboard,
    summarize_health,
)
from config import AgentConfig


def _cfg(**overrides):
    defaults = dict(
        watchlist=("AAPL", "MSFT"),
        correlation_groups={"us_large_cap": ("AAPL", "MSFT")},
        max_group_dollars=200.0,
        max_position_dollars=150.0,
        max_total_dollars=400.0,
        initial_cash=400.0,
        dip_trigger_pct=2.0,
        revert_target_pct=3.0,
        order_dollars=25.0,
        anthropic_api_key="k",
        mcp_url="u",
        robinhood_client_id="c",
        agentic_account_number="1",
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


# --- the sample log ------------------------------------------------------
# Two dry-run cycles. The first is healthy and produces one approved
# (simulated) buy plus one guardrail-blocked buy. The second is the
# degraded case: its snapshot is missing MSFT.

COMPLETE_SNAPSHOT_EVENTS = [
    {"ts": "2026-07-01T14:00:00+00:00", "event": "cycle_report", "report": "..."},
    {
        "ts": "2026-07-01T14:00:01+00:00",
        "event": "market_snapshot",
        "snapshot": {
            "AAPL": {"current_price": 180.0, "day_change_pct": -3.2},
            "MSFT": {"current_price": 400.0, "day_change_pct": -0.4},
        },
    },
    {
        "ts": "2026-07-01T14:00:02+00:00",
        "event": "api_usage",
        "call": "proposal",
        "model": "claude-sonnet-4-6",
        "input_tokens": 1_000_000,
        "output_tokens": 200_000,
        "estimated_cost_usd": 6.0,
    },
    {
        "ts": "2026-07-01T14:00:03+00:00",
        "event": "order_blocked",
        "reason": "would push AAPL position to $165.00, over the $150.00 per-position cap",
        "order": {
            "ticker": "AAPL",
            "side": "buy",
            "dollars": 25.0,
            "day_change_pct": -3.2,
            "current_price": 180.0,
            "positions": {"AAPL": 140.0},
            "reason": "AAPL dipped 3.2%, well past the trigger — clear buy",
        },
    },
    {
        "ts": "2026-07-01T14:00:04+00:00",
        "event": "simulated_fill",
        "ticker": "MSFT",
        "side": "buy",
        "dollars": 25.0,
        "simulated_price": 400.0,
        "reason": "MSFT down -2.50% (trigger 2.00%), within position, group, and account caps",
        "order": {
            "ticker": "MSFT",
            "side": "buy",
            "dollars": 25.0,
            "day_change_pct": -2.5,
            "current_price": 400.0,
            "positions": {},
            "reason": "MSFT qualifies on the dip trigger",
        },
    },
]

INCOMPLETE_TAIL_EVENTS = [
    {"ts": "2026-07-02T14:00:00+00:00", "event": "cycle_report", "report": "..."},
    {
        "ts": "2026-07-02T14:00:01+00:00",
        "event": "market_snapshot",
        "snapshot": {"AAPL": {"current_price": 182.0, "day_change_pct": 1.1}},
    },
    {
        "ts": "2026-07-02T14:00:02+00:00",
        "event": "market_snapshot_incomplete",
        "missing_tickers": ["MSFT"],
        "reason": "watchlist tickers absent from the model's market_snapshot",
    },
]

HEALTHY_SECOND_CYCLE = [
    {"ts": "2026-07-02T14:00:00+00:00", "event": "cycle_report", "report": "..."},
    {
        "ts": "2026-07-02T14:00:01+00:00",
        "event": "market_snapshot",
        "snapshot": {
            "AAPL": {"current_price": 182.0, "day_change_pct": 1.1},
            "MSFT": {"current_price": 404.0, "day_change_pct": 1.0},
        },
    },
]

CYCLE_WITH_RECENT_SIGNALS = [
    {"ts": "2026-07-03T14:00:00+00:00", "event": "cycle_report", "report": "..."},
    {
        "ts": "2026-07-03T14:00:01+00:00",
        "event": "market_snapshot",
        "snapshot": {
            "AAPL": {"current_price": 92.0, "day_change_pct": -8.0},
            "MSFT": {"current_price": 400.0, "day_change_pct": 0.0},
        },
    },
    {
        "ts": "2026-07-03T14:00:02+00:00",
        "event": "snapshot_recomputed",
        "ticker": "AAPL",
        "model_day_change_pct": -0.5,
        "recomputed_day_change_pct": -8.0,
        "price_session": "regular",
        "current_price": 92.0,
        "adjusted_previous_close": 100.0,
    },
    {
        "ts": "2026-07-03T14:00:03+00:00",
        "event": "session_divergence",
        "ticker": "MSFT",
        "last_trade_price": 400.0,
        "last_non_reg_trade_price": 432.0,
        "configured_session": "regular",
    },
]

SAMPLE_PORTFOLIO = {
    "cash": 375.0,
    "positions": {"MSFT": {"shares": 0.0625, "avg_cost": 400.0, "last_price": 404.0}},
    "realized_pnl": 0.0,
}


def _write_log(tmp_path, events):
    path = tmp_path / "trade_log.jsonl"
    path.write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
    )
    return str(path)


def _write_portfolio(tmp_path, portfolio=SAMPLE_PORTFOLIO):
    path = tmp_path / "paper_portfolio.json"
    path.write_text(json.dumps(portfolio), encoding="utf-8")
    return str(path)


class TestHealthBanner:
    def test_warning_when_last_cycle_snapshot_is_incomplete(self):
        events = COMPLETE_SNAPSHOT_EVENTS + INCOMPLETE_TAIL_EVENTS
        health = summarize_health(events)

        assert health["status"] == "WARNING"
        assert health["missing_tickers"] == ["MSFT"]
        assert health["cycle_count"] == 2

    def test_ok_when_last_cycle_snapshot_is_complete(self):
        events = COMPLETE_SNAPSHOT_EVENTS + HEALTHY_SECOND_CYCLE
        health = summarize_health(events)

        assert health["status"] == "OK"
        assert health["cycle_count"] == 2

    def test_earlier_incomplete_cycle_does_not_taint_a_healthy_latest_cycle(self):
        # Health is about the CURRENT state — an old failure that has since
        # recovered must not keep the banner red forever.
        events = (
            COMPLETE_SNAPSHOT_EVENTS + INCOMPLETE_TAIL_EVENTS + HEALTHY_SECOND_CYCLE
        )
        assert summarize_health(events)["status"] == "OK"

    def test_warning_banner_appears_in_rendered_html(self):
        events = COMPLETE_SNAPSHOT_EVENTS + INCOMPLETE_TAIL_EVENTS
        out = render_dashboard(events, SAMPLE_PORTFOLIO, _cfg())

        assert '<div class="status warning">WARNING</div>' in out
        assert '<div class="status ok">OK</div>' not in out

    def test_ok_banner_appears_in_rendered_html(self):
        events = COMPLETE_SNAPSHOT_EVENTS + HEALTHY_SECOND_CYCLE
        out = render_dashboard(events, SAMPLE_PORTFOLIO, _cfg())

        assert '<div class="status ok">OK</div>' in out
        assert '<div class="status warning">WARNING</div>' not in out


class TestModeBanner:
    def test_paper_banner_for_dry_run_sourced_events(self):
        assert determine_mode(COMPLETE_SNAPSHOT_EVENTS) == "PAPER"

        out = render_dashboard(COMPLETE_SNAPSHOT_EVENTS, SAMPLE_PORTFOLIO, _cfg())
        assert 'class="mode paper"' in out
        assert "PAPER TRADING" in out
        assert "LIVE TRADING" not in out

    def test_live_banner_for_real_execution_events(self):
        events = COMPLETE_SNAPSHOT_EVENTS[:3] + [
            {
                "ts": "2026-07-01T14:00:05+00:00",
                "event": "order_executed",
                "order": {"ticker": "AAPL", "side": "buy", "dollars": 25.0},
                "result": "filled",
            }
        ]
        assert determine_mode(events) == "LIVE"

        out = render_dashboard(events, None, _cfg())
        assert 'class="mode live"' in out
        assert "LIVE TRADING" in out

    def test_mixed_log_is_flagged_and_not_silently_combined(self):
        events = COMPLETE_SNAPSHOT_EVENTS + [
            {
                "ts": "2026-07-01T15:00:00+00:00",
                "event": "order_executed",
                "order": {"ticker": "AAPL", "side": "buy", "dollars": 25.0},
            }
        ]
        assert determine_mode(events) == "MIXED"

        out = render_dashboard(events, SAMPLE_PORTFOLIO, _cfg())
        assert 'class="mode mixed"' in out
        assert "NOT" in out and "single P&amp;L figure" in out

    def test_unknown_when_no_fills_at_all(self):
        assert determine_mode(HEALTHY_SECOND_CYCLE) == "UNKNOWN"


class TestDecisions:
    def test_blocked_and_simulated_rows_appear_in_rendered_html(self):
        out = render_dashboard(COMPLETE_SNAPSHOT_EVENTS, SAMPLE_PORTFOLIO, _cfg())

        assert "BLOCKED" in out
        assert "SIMULATED" in out
        assert "over the $150.00 per-position cap" in out

    def test_blocked_row_is_visually_distinguished_and_flagged_as_disagreement(self):
        out = render_dashboard(COMPLETE_SNAPSHOT_EVENTS, SAMPLE_PORTFOLIO, _cfg())

        assert 'class="row neg"' in out
        assert "MODEL AND CODE DISAGREE" in out

    def test_why_panel_separates_model_narrative_from_code_arithmetic(self):
        out = render_dashboard(COMPLETE_SNAPSHOT_EVENTS, SAMPLE_PORTFOLIO, _cfg())

        assert "WHAT THE MODEL SAID" in out
        assert "WHAT THE CODE DECIDED" in out
        # The model's own prose is shown, clearly attributed to the model.
        assert "well past the trigger" in out
        # The code's arithmetic is re-derived and shown explicitly.
        assert "140.00 + 25.00 = 165.00" in out

    def test_auto_approved_rows_are_distinguished(self):
        events = COMPLETE_SNAPSHOT_EVENTS + [
            {
                "ts": "2026-07-01T14:00:06+00:00",
                "event": "auto_approved",
                "reason": "AAPL down -3.20%, within caps",
                "order": {"ticker": "AAPL", "side": "buy", "dollars": 25.0},
            }
        ]
        out = render_dashboard(events, SAMPLE_PORTFOLIO, _cfg())

        assert "AUTO-APPROVED" in out
        assert 'class="row auto"' in out

    def test_decisions_are_newest_first(self):
        rows = collect_decisions(COMPLETE_SNAPSHOT_EVENTS, _cfg())
        assert [r["ticker"] for r in rows] == ["MSFT", "AAPL"]


class TestApiCost:
    def test_cost_is_totalled_and_shown_next_to_pnl(self):
        cost = collect_api_cost(COMPLETE_SNAPSHOT_EVENTS)
        assert cost["total_usd"] == pytest.approx(6.0)
        assert cost["call_count"] == 1

        out = render_dashboard(COMPLETE_SNAPSHOT_EVENTS, SAMPLE_PORTFOLIO, _cfg())
        assert "Cumulative API cost" in out
        assert "$6.00" in out


class TestReconciliationSection:
    def test_says_so_explicitly_when_there_are_no_mismatches(self):
        out = render_dashboard(COMPLETE_SNAPSHOT_EVENTS, SAMPLE_PORTFOLIO, _cfg())
        assert "No execution mismatches recorded" in out

    def test_mismatches_are_rendered_prominently(self):
        events = COMPLETE_SNAPSHOT_EVENTS + [
            {
                "ts": "2026-07-01T16:00:00+00:00",
                "event": "execution_mismatch",
                "reason": "2 option order(s) were placed — this strategy is equities-only",
                "order": {"ticker": "AAPL", "side": "buy", "dollars": 25.0},
            }
        ]
        out = render_dashboard(events, SAMPLE_PORTFOLIO, _cfg())

        assert 'class="mismatch"' in out
        assert "equities-only" in out
        assert "No execution mismatches recorded" not in out


class TestReadOnlyConstraints:
    def test_page_contains_no_control_affordances(self):
        out = render_dashboard(COMPLETE_SNAPSHOT_EVENTS, SAMPLE_PORTFOLIO, _cfg())
        lowered = out.lower()

        # No form posts, no fetch/XHR, no websockets — nothing that could
        # even appear to reach the agent.
        assert "<form" not in lowered
        assert "fetch(" not in lowered
        assert "xmlhttprequest" not in lowered
        assert "websocket" not in lowered

        # The only interactive controls are the decision-table filters and
        # the row-expand toggle. Every <button> must be one of those.
        import re

        onclicks = set(re.findall(r'onclick="(\w+)\(', out))
        assert onclicks <= {"toggle", "applyFilters", "clearFilters"}

    def test_read_only_nature_is_stated_explicitly_to_the_reader(self):
        # The absence of controls should be legible as a deliberate choice,
        # not read as an unfinished page.
        out = render_dashboard(COMPLETE_SNAPSHOT_EVENTS, SAMPLE_PORTFOLIO, _cfg())
        assert "This page is read-only" in out
        assert "deliberately no kill switch" in out

    def test_page_has_no_external_resource_references(self):
        out = render_dashboard(COMPLETE_SNAPSHOT_EVENTS, SAMPLE_PORTFOLIO, _cfg())
        assert "http://" not in out
        assert "https://" not in out
        assert "<script src" not in out
        assert "<link" not in out


class TestEquityChart:
    def test_benchmark_is_plotted_alongside_the_strategy(self):
        events = COMPLETE_SNAPSHOT_EVENTS + HEALTHY_SECOND_CYCLE
        out = render_dashboard(events, SAMPLE_PORTFOLIO, _cfg())

        assert "Equal-weight buy &amp; hold" in out
        assert "<svg" in out

    def test_single_cycle_shows_a_message_not_a_solo_line(self):
        out = render_dashboard(COMPLETE_SNAPSHOT_EVENTS, SAMPLE_PORTFOLIO, _cfg())
        assert "Not enough logged cycles" in out


class TestGroupExposure:
    def test_group_bar_shows_exposure_against_the_cap(self):
        out = render_dashboard(COMPLETE_SNAPSHOT_EVENTS, SAMPLE_PORTFOLIO, _cfg())

        assert "us_large_cap" in out
        assert "$200.00" in out  # the cap
        assert 'class="bar"' in out


class TestBuildAndCsv:
    def test_build_writes_a_self_contained_html_file(self, tmp_path):
        log = _write_log(tmp_path, COMPLETE_SNAPSHOT_EVENTS)
        portfolio = _write_portfolio(tmp_path)
        out_path = str(tmp_path / "dashboard.html")

        build(
            log_path=log,
            portfolio_path=portfolio,
            out_path=out_path,
            cfg=_cfg(),
        )

        content = open(out_path, encoding="utf-8").read()
        assert content.startswith("<!DOCTYPE html>")
        assert "<style>" in content and "<script>" in content

    def test_csv_has_correct_columns_and_row_count(self, tmp_path):
        log = _write_log(tmp_path, COMPLETE_SNAPSHOT_EVENTS)
        portfolio = _write_portfolio(tmp_path)
        csv_path = str(tmp_path / "trades.csv")

        build(
            log_path=log,
            portfolio_path=portfolio,
            out_path=str(tmp_path / "dashboard.html"),
            csv_path=csv_path,
            cfg=_cfg(),
        )

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        assert rows[0] == [
            "timestamp",
            "ticker",
            "side",
            "dollars",
            "price",
            "verdict",
            "reason",
        ]
        # Two decisions in the sample log: one blocked, one simulated fill.
        assert len(rows) == 3  # header + 2
        assert {r[1] for r in rows[1:]} == {"AAPL", "MSFT"}
        assert {r[5] for r in rows[1:]} == {"BLOCKED", "SIMULATED"}

    def test_missing_log_and_portfolio_do_not_crash(self, tmp_path):
        out_path = str(tmp_path / "dashboard.html")
        build(
            log_path=str(tmp_path / "nope.jsonl"),
            portfolio_path=str(tmp_path / "nope.json"),
            out_path=out_path,
            cfg=_cfg(),
        )
        content = open(out_path, encoding="utf-8").read()
        assert "No cycles have ever been logged" in content

    def test_malformed_log_line_is_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "trade_log.jsonl"
        path.write_text(
            json.dumps(COMPLETE_SNAPSHOT_EVENTS[0]) + "\n{ this is not json\n",
            encoding="utf-8",
        )
        out_path = str(tmp_path / "dashboard.html")

        build(log_path=str(path), portfolio_path=str(tmp_path / "x.json"),
              out_path=out_path, cfg=_cfg())

        assert os.path.exists(out_path)


class TestRecentSignals:
    """session_divergence and snapshot_recomputed were added to agent.py
    after this dashboard was first written — collect_recent_signals() and
    its HEALTH-section rendering are what make them visible instead of
    sitting unread in the log."""

    def test_collect_returns_empty_when_neither_event_type_is_in_the_log(self):
        events = COMPLETE_SNAPSHOT_EVENTS + HEALTHY_SECOND_CYCLE
        signals = collect_recent_signals(events)

        assert signals["session_divergences"] == []
        assert signals["snapshot_recomputations"] == []

    def test_collect_finds_both_event_types_from_the_most_recent_cycle(self):
        events = COMPLETE_SNAPSHOT_EVENTS + CYCLE_WITH_RECENT_SIGNALS
        signals = collect_recent_signals(events)

        assert len(signals["snapshot_recomputations"]) == 1
        recompute = signals["snapshot_recomputations"][0]
        assert recompute["ticker"] == "AAPL"
        assert recompute["model_day_change_pct"] == -0.5
        assert recompute["recomputed_day_change_pct"] == -8.0
        assert recompute["price_session"] == "regular"

        assert len(signals["session_divergences"]) == 1
        divergence = signals["session_divergences"][0]
        assert divergence["ticker"] == "MSFT"
        assert divergence["last_trade_price"] == 400.0
        assert divergence["last_non_reg_trade_price"] == 432.0
        assert divergence["configured_session"] == "regular"
        assert divergence["pct_gap"] == pytest.approx(8.0)

    def test_signals_from_an_earlier_cycle_do_not_leak_into_a_clean_latest_cycle(self):
        # The whole point of scoping to "most recent cycle" is that a
        # resolved issue from an earlier cycle must not keep showing up
        # forever, the same way summarize_health() already works.
        events = CYCLE_WITH_RECENT_SIGNALS + HEALTHY_SECOND_CYCLE
        signals = collect_recent_signals(events)

        assert signals["session_divergences"] == []
        assert signals["snapshot_recomputations"] == []

    def test_no_signal_block_renders_when_neither_event_type_is_present(self):
        events = COMPLETE_SNAPSHOT_EVENTS + HEALTHY_SECOND_CYCLE
        out = render_dashboard(events, None, _cfg())

        # Not just "no warning text" — no panel at all, since a log that
        # never logged these events either checked and found nothing, or
        # predates the feature entirely. Neither should render as a
        # reassuring all-clear.
        assert '<div class="signal-block' not in out

    def test_both_signal_blocks_render_with_their_data_when_present(self):
        events = COMPLETE_SNAPSHOT_EVENTS + CYCLE_WITH_RECENT_SIGNALS
        out = render_dashboard(events, None, _cfg())

        assert '<div class="signal-block">' in out
        assert '<div class="signal-block warn">' in out
        assert "Model/code arithmetic disagreement" in out
        assert "Session divergence" in out

        # The actual reported numbers must be on the page, not just the
        # section headers.
        assert "-0.50%" in out
        assert "-8.00%" in out
        assert "$400.00" in out
        assert "$432.00" in out
        assert "8.00%" in out
