"""
Unit tests for guardrails.py.

These tests exist to prove, independent of any LLM behavior, that the hard
limits actually hold. If a model ever proposed a malicious or mistaken
order, these are the checks standing between it and a real trade.
"""

import dataclasses
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from guardrails import (
    GuardrailConfig,
    validate_batch,
    validate_buy,
    validate_instrument,
    validate_sell,
)

CFG = GuardrailConfig(
    watchlist=("VOO", "AAPL", "MSFT"),
    order_dollars=25.0,
    max_position_dollars=150.0,
    max_total_dollars=400.0,
    dip_trigger_pct=2.0,
    revert_target_pct=3.0,
    max_hold_days=10,
    disaster_stop_pct=15.0,
    correlation_groups={"us_large_cap": ("VOO", "AAPL", "MSFT")},
    max_group_dollars=200.0,
)


class TestValidateInstrument:
    def test_accepts_plain_equity_symbols(self):
        for symbol in ("AAPL", "MSFT", "VOO", "F", "GOOGL"):
            result = validate_instrument(symbol)
            assert result.approved, f"{symbol} should be accepted"

    def test_rejects_occ_option_symbol_without_space(self):
        result = validate_instrument("AAPL260815C00200000")
        assert not result.approved

    def test_rejects_occ_option_symbol_with_space(self):
        result = validate_instrument("AAPL 260815C00200000")
        assert not result.approved

    def test_rejects_symbol_with_embedded_space(self):
        result = validate_instrument("AA PL")
        assert not result.approved

    def test_rejects_symbol_with_digits(self):
        result = validate_instrument("AAPL1")
        assert not result.approved

    def test_rejects_symbol_over_five_characters(self):
        result = validate_instrument("GOOGLE")
        assert not result.approved

    def test_accepts_symbol_at_five_characters(self):
        result = validate_instrument("GOOGL")
        assert result.approved

    def test_rejects_empty_symbol(self):
        result = validate_instrument("")
        assert not result.approved


class TestValidateBuy:
    def test_approves_qualifying_dip(self):
        result = validate_buy(
            CFG, "AAPL", 25.0, day_change_pct=-2.5, positions={},
        )
        assert result.approved

    def test_rejects_off_watchlist_ticker(self):
        result = validate_buy(
            CFG, "GME", 25.0, day_change_pct=-10.0, positions={},
        )
        assert not result.approved
        assert "watchlist" in result.reason

    def test_rejects_insufficient_dip(self):
        result = validate_buy(
            CFG, "AAPL", 25.0, day_change_pct=-1.0, positions={},
        )
        assert not result.approved

    def test_rejects_oversized_order_even_if_dip_qualifies(self):
        result = validate_buy(
            CFG, "AAPL", 1000.0, day_change_pct=-5.0, positions={},
        )
        assert not result.approved
        assert "exceeds" in result.reason

    def test_rejects_when_position_cap_would_be_exceeded(self):
        result = validate_buy(
            CFG, "AAPL", 25.0, day_change_pct=-5.0,
            positions={"AAPL": 140.0},
        )
        assert not result.approved
        assert "per-position cap" in result.reason

    def test_rejects_when_total_cap_would_be_exceeded(self):
        # Held value sits entirely outside any correlation group so only
        # the total account cap can be the thing that rejects this.
        result = validate_buy(
            CFG, "MSFT", 25.0, day_change_pct=-5.0,
            positions={"OTHER": 390.0},
        )
        assert not result.approved
        assert "account cap" in result.reason

    def test_boundary_exactly_at_trigger_is_approved(self):
        result = validate_buy(
            CFG, "VOO", 25.0, day_change_pct=-2.0, positions={},
        )
        assert result.approved

    def test_rejects_negative_or_zero_dollars(self):
        assert not validate_buy(
            CFG, "VOO", 0, day_change_pct=-5.0, positions={},
        ).approved
        assert not validate_buy(
            CFG, "VOO", -10, day_change_pct=-5.0, positions={},
        ).approved

    def test_ticker_is_case_insensitive(self):
        result = validate_buy(
            CFG, "aapl", 25.0, day_change_pct=-3.0, positions={},
        )
        assert result.approved

    def test_rejects_buy_purely_by_group_cap(self):
        # AAPL alone is well under its per-position cap, and total exposure
        # is well under the account cap — only the us_large_cap group cap
        # (VOO + AAPL + MSFT) can reject this.
        result = validate_buy(
            CFG, "AAPL", 25.0, day_change_pct=-5.0,
            positions={"VOO": 100.0, "MSFT": 90.0},
        )
        assert not result.approved
        assert "us_large_cap" in result.reason
        assert "group cap" in result.reason

    def test_boundary_exactly_at_group_cap_is_approved(self):
        result = validate_buy(
            CFG, "AAPL", 25.0, day_change_pct=-5.0,
            positions={"VOO": 100.0, "MSFT": 75.0},
        )
        assert result.approved

    def test_ticker_in_no_group_unaffected_by_group_cap(self):
        cfg = dataclasses.replace(CFG, watchlist=CFG.watchlist + ("TLT",))
        result = validate_buy(
            cfg, "TLT", 25.0, day_change_pct=-5.0,
            positions={"TLT": 100.0, "VOO": 190.0},
        )
        assert result.approved


class TestValidateSell:
    def test_approves_qualifying_revert(self):
        result = validate_sell(
            CFG, "AAPL", current_price=103.0, avg_cost=100.0, days_held=1
        )
        assert result.approved
        assert "profit_target" in result.reason

    def test_rejects_insufficient_gain(self):
        result = validate_sell(
            CFG, "AAPL", current_price=101.0, avg_cost=100.0, days_held=1
        )
        assert not result.approved

    def test_rejects_off_watchlist_ticker(self):
        result = validate_sell(
            CFG, "GME", current_price=200.0, avg_cost=100.0, days_held=1
        )
        assert not result.approved
        assert "watchlist" in result.reason

    def test_rejects_zero_or_negative_cost_basis(self):
        result = validate_sell(
            CFG, "AAPL", current_price=103.0, avg_cost=0, days_held=1
        )
        assert not result.approved

    def test_boundary_exactly_at_target_is_approved(self):
        result = validate_sell(
            CFG, "MSFT", current_price=103.0, avg_cost=100.0, days_held=1
        )
        assert result.approved
        assert "profit_target" in result.reason

    # --- time_exit ---

    def test_time_exit_approves_losing_position_past_max_hold(self):
        result = validate_sell(
            CFG, "VOO", current_price=95.0, avg_cost=100.0, days_held=10
        )
        assert result.approved
        assert "time_exit" in result.reason

    def test_boundary_exactly_at_max_hold_days_is_approved(self):
        result = validate_sell(
            CFG, "VOO", current_price=98.0, avg_cost=100.0, days_held=10
        )
        assert result.approved
        assert "time_exit" in result.reason

    def test_rejects_below_max_hold_days_with_no_other_exit(self):
        result = validate_sell(
            CFG, "VOO", current_price=98.0, avg_cost=100.0, days_held=9
        )
        assert not result.approved

    # --- disaster_stop ---

    def test_disaster_stop_approves_regardless_of_hold_time(self):
        result = validate_sell(
            CFG, "AAPL", current_price=84.0, avg_cost=100.0, days_held=1
        )
        assert result.approved
        assert "disaster_stop" in result.reason

    def test_boundary_exactly_at_disaster_stop_pct_is_approved(self):
        result = validate_sell(
            CFG, "AAPL", current_price=85.0, avg_cost=100.0, days_held=1
        )
        assert result.approved
        assert "disaster_stop" in result.reason

    # --- must still be rejected ---

    def test_rejects_loss_under_max_hold_and_above_disaster_stop(self):
        result = validate_sell(
            CFG, "VOO", current_price=90.0, avg_cost=100.0, days_held=5
        )
        assert not result.approved


class TestValidateBatch:
    def test_group_cap_admits_exactly_one_buy_and_prefers_the_largest_dip(self):
        # Existing group exposure is 175 (AAPL 100 + MSFT 75), leaving room
        # for exactly one more $25 buy before the 200 group cap. All three
        # tickers dip past the trigger, but by different amounts.
        positions = {"AAPL": 100.0, "MSFT": 75.0}
        orders = [
            {"ticker": "AAPL", "side": "buy", "dollars": 25.0, "day_change_pct": -6.0},
            {"ticker": "MSFT", "side": "buy", "dollars": 25.0, "day_change_pct": -3.0},
            {"ticker": "VOO", "side": "buy", "dollars": 25.0, "day_change_pct": -10.0},
        ]

        batch = validate_batch(CFG, orders, positions)
        by_ticker = {o["ticker"]: result for o, result in batch}

        assert by_ticker["VOO"].approved  # the largest dip (-10%) wins
        assert not by_ticker["AAPL"].approved
        assert "group cap" in by_ticker["AAPL"].reason
        assert not by_ticker["MSFT"].approved
        assert "group cap" in by_ticker["MSFT"].reason

    def test_buy_tiebreak_is_alphabetical_when_dips_are_equal(self):
        # Same setup, but all three dip by the identical amount — the
        # winner must be decided by ticker name alone, not input order.
        positions = {"AAPL": 100.0, "MSFT": 75.0}
        orders = [
            {"ticker": "MSFT", "side": "buy", "dollars": 25.0, "day_change_pct": -5.0},
            {"ticker": "VOO", "side": "buy", "dollars": 25.0, "day_change_pct": -5.0},
            {"ticker": "AAPL", "side": "buy", "dollars": 25.0, "day_change_pct": -5.0},
        ]

        batch = validate_batch(CFG, orders, positions)
        by_ticker = {o["ticker"]: result for o, result in batch}

        assert by_ticker["AAPL"].approved  # alphabetically first
        assert not by_ticker["MSFT"].approved
        assert not by_ticker["VOO"].approved

    def test_sell_in_batch_frees_capacity_for_a_buy_that_would_otherwise_fail(self):
        # Group is sitting exactly at the 200 cap. Without the MSFT sell,
        # the AAPL buy would push it to 225 and fail.
        positions = {"VOO": 100.0, "AAPL": 75.0, "MSFT": 25.0}
        orders = [
            {"ticker": "AAPL", "side": "buy", "dollars": 25.0, "day_change_pct": -5.0},
            {
                "ticker": "MSFT",
                "side": "sell",
                "current_price": 110.0,
                "avg_cost": 100.0,
                "days_held": 1,
            },
        ]

        batch = validate_batch(CFG, orders, positions)
        by_ticker = {o["ticker"]: result for o, result in batch}

        assert by_ticker["MSFT"].approved
        assert "profit_target" in by_ticker["MSFT"].reason
        assert by_ticker["AAPL"].approved

    def test_does_not_mutate_callers_positions_dict(self):
        positions = {"AAPL": 100.0, "MSFT": 75.0}
        original = dict(positions)
        orders = [
            {"ticker": "VOO", "side": "buy", "dollars": 25.0, "day_change_pct": -5.0},
            {
                "ticker": "AAPL",
                "side": "sell",
                "current_price": 110.0,
                "avg_cost": 100.0,
                "days_held": 1,
            },
        ]

        validate_batch(CFG, orders, positions)

        assert positions == original

    def test_sells_processed_before_buys_regardless_of_input_order(self):
        positions = {}
        orders = [
            {"ticker": "AAPL", "side": "buy", "dollars": 25.0, "day_change_pct": -5.0},
            {
                "ticker": "VOO",
                "side": "sell",
                "current_price": 110.0,
                "avg_cost": 100.0,
                "days_held": 1,
            },
        ]

        batch = validate_batch(CFG, orders, positions)

        assert [o["side"] for o, _ in batch] == ["sell", "buy"]

    def test_unrecognized_side_is_rejected(self):
        positions = {}
        orders = [{"ticker": "AAPL", "side": "hold", "dollars": 25.0}]

        batch = validate_batch(CFG, orders, positions)

        assert len(batch) == 1
        order, result = batch[0]
        assert not result.approved
        assert "unrecognized side" in result.reason
