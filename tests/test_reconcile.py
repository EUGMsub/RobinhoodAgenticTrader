"""
Unit tests for reconcile.py.

These prove, independent of any LLM behavior, that a mismatch between what
the broker actually recorded and what the human approved is always caught.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reconcile import reconcile_order

APPROVED = {"ticker": "AAPL", "side": "buy", "dollars": 25.0}


def _equity_order(symbol="AAPL", side="buy", dollar_amount=25.0):
    return {"symbol": symbol, "side": side, "dollar_amount": dollar_amount}


class TestReconcileOrder:
    def test_passes_on_exact_match(self):
        result = reconcile_order(APPROVED, [_equity_order()], [])
        assert result.passed

    def test_passes_within_dollar_tolerance(self):
        result = reconcile_order(
            APPROVED, [_equity_order(dollar_amount=25.2)], []
        )
        assert result.passed

    def test_boundary_exactly_at_one_percent_tolerance_passes(self):
        result = reconcile_order(
            APPROVED, [_equity_order(dollar_amount=25.25)], []
        )
        assert result.passed

    def test_fails_if_any_option_order_present(self):
        result = reconcile_order(
            APPROVED, [_equity_order()], [{"symbol": "AAPL260815C00200000"}]
        )
        assert not result.passed
        assert "option" in result.reason

    def test_fails_if_option_order_present_even_with_no_equity_orders(self):
        result = reconcile_order(APPROVED, [], [{"symbol": "AAPL260815C00200000"}])
        assert not result.passed
        assert "option" in result.reason

    def test_fails_if_zero_equity_orders(self):
        result = reconcile_order(APPROVED, [], [])
        assert not result.passed
        assert "1 new equity order" in result.reason

    def test_fails_if_more_than_one_equity_order(self):
        result = reconcile_order(
            APPROVED, [_equity_order(), _equity_order()], []
        )
        assert not result.passed
        assert "1 new equity order" in result.reason

    def test_fails_on_symbol_mismatch(self):
        result = reconcile_order(APPROVED, [_equity_order(symbol="MSFT")], [])
        assert not result.passed
        assert "symbol mismatch" in result.reason

    def test_fails_on_side_mismatch(self):
        result = reconcile_order(APPROVED, [_equity_order(side="sell")], [])
        assert not result.passed
        assert "side mismatch" in result.reason

    def test_fails_on_dollar_amount_over_tolerance(self):
        result = reconcile_order(
            APPROVED, [_equity_order(dollar_amount=30.0)], []
        )
        assert not result.passed
        assert "dollar amount mismatch" in result.reason

    def test_fails_just_over_one_percent_tolerance(self):
        result = reconcile_order(
            APPROVED, [_equity_order(dollar_amount=25.3)], []
        )
        assert not result.passed
        assert "dollar amount mismatch" in result.reason

    def test_ticker_and_side_comparison_is_case_insensitive(self):
        result = reconcile_order(
            APPROVED, [_equity_order(symbol="aapl", side="BUY")], []
        )
        assert result.passed
