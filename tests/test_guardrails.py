"""
Unit tests for guardrails.py.

These tests exist to prove, independent of any LLM behavior, that the hard
limits actually hold. If a model ever proposed a malicious or mistaken
order, these are the checks standing between it and a real trade.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from guardrails import GuardrailConfig, validate_buy, validate_sell

CFG = GuardrailConfig(
    watchlist=("VOO", "AAPL", "MSFT"),
    order_dollars=25.0,
    max_position_dollars=150.0,
    max_total_dollars=400.0,
    dip_trigger_pct=2.0,
    revert_target_pct=3.0,
)


class TestValidateBuy:
    def test_approves_qualifying_dip(self):
        result = validate_buy(
            CFG, "AAPL", 25.0, day_change_pct=-2.5,
            current_position_value=0, current_total_exposure=0,
        )
        assert result.approved

    def test_rejects_off_watchlist_ticker(self):
        result = validate_buy(
            CFG, "GME", 25.0, day_change_pct=-10.0,
            current_position_value=0, current_total_exposure=0,
        )
        assert not result.approved
        assert "watchlist" in result.reason

    def test_rejects_insufficient_dip(self):
        result = validate_buy(
            CFG, "AAPL", 25.0, day_change_pct=-1.0,
            current_position_value=0, current_total_exposure=0,
        )
        assert not result.approved

    def test_rejects_oversized_order_even_if_dip_qualifies(self):
        result = validate_buy(
            CFG, "AAPL", 1000.0, day_change_pct=-5.0,
            current_position_value=0, current_total_exposure=0,
        )
        assert not result.approved
        assert "exceeds" in result.reason

    def test_rejects_when_position_cap_would_be_exceeded(self):
        result = validate_buy(
            CFG, "AAPL", 25.0, day_change_pct=-5.0,
            current_position_value=140.0, current_total_exposure=140.0,
        )
        assert not result.approved
        assert "per-position cap" in result.reason

    def test_rejects_when_total_cap_would_be_exceeded(self):
        result = validate_buy(
            CFG, "MSFT", 25.0, day_change_pct=-5.0,
            current_position_value=0, current_total_exposure=390.0,
        )
        assert not result.approved
        assert "account cap" in result.reason

    def test_boundary_exactly_at_trigger_is_approved(self):
        result = validate_buy(
            CFG, "VOO", 25.0, day_change_pct=-2.0,
            current_position_value=0, current_total_exposure=0,
        )
        assert result.approved

    def test_rejects_negative_or_zero_dollars(self):
        assert not validate_buy(
            CFG, "VOO", 0, day_change_pct=-5.0,
            current_position_value=0, current_total_exposure=0,
        ).approved
        assert not validate_buy(
            CFG, "VOO", -10, day_change_pct=-5.0,
            current_position_value=0, current_total_exposure=0,
        ).approved

    def test_ticker_is_case_insensitive(self):
        result = validate_buy(
            CFG, "aapl", 25.0, day_change_pct=-3.0,
            current_position_value=0, current_total_exposure=0,
        )
        assert result.approved


class TestValidateSell:
    def test_approves_qualifying_revert(self):
        result = validate_sell(CFG, "AAPL", current_price=103.0, avg_cost=100.0)
        assert result.approved

    def test_rejects_insufficient_gain(self):
        result = validate_sell(CFG, "AAPL", current_price=101.0, avg_cost=100.0)
        assert not result.approved

    def test_rejects_off_watchlist_ticker(self):
        result = validate_sell(CFG, "GME", current_price=200.0, avg_cost=100.0)
        assert not result.approved
        assert "watchlist" in result.reason

    def test_rejects_zero_or_negative_cost_basis(self):
        result = validate_sell(CFG, "AAPL", current_price=103.0, avg_cost=0)
        assert not result.approved

    def test_boundary_exactly_at_target_is_approved(self):
        result = validate_sell(CFG, "MSFT", current_price=103.0, avg_cost=100.0)
        assert result.approved

    def test_loss_position_never_approved(self):
        result = validate_sell(CFG, "VOO", current_price=90.0, avg_cost=100.0)
        assert not result.approved
