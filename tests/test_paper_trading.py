"""
Unit tests for paper_trading.py.

apply_simulated_buy()/apply_simulated_sell()/summarize() are pure — these
tests never touch the filesystem. load/save round-tripping is covered
separately with tmp_path.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from paper_trading import (
    PaperPortfolio,
    PaperPosition,
    apply_simulated_buy,
    apply_simulated_sell,
    load_paper_portfolio,
    save_paper_portfolio,
    summarize,
)


class TestApplySimulatedBuy:
    def test_opens_a_new_position_and_debits_cash(self):
        portfolio = PaperPortfolio(cash=1000.0)
        result = apply_simulated_buy(portfolio, "AAPL", dollars=25.0, price=100.0)

        assert result.cash == 975.0
        assert result.positions["AAPL"].shares == 0.25
        assert result.positions["AAPL"].avg_cost == 100.0
        assert result.positions["AAPL"].last_price == 100.0

    def test_does_not_mutate_the_input_portfolio(self):
        portfolio = PaperPortfolio(cash=1000.0)
        apply_simulated_buy(portfolio, "AAPL", dollars=25.0, price=100.0)

        assert portfolio.cash == 1000.0
        assert portfolio.positions == {}

    def test_blends_into_an_existing_position_with_weighted_avg_cost(self):
        portfolio = PaperPortfolio(
            cash=975.0,
            positions={"AAPL": PaperPosition(shares=0.25, avg_cost=100.0, last_price=100.0)},
        )
        result = apply_simulated_buy(portfolio, "AAPL", dollars=25.0, price=50.0)

        # 0.25 sh @ 100 + 0.5 sh @ 50 -> 0.75 sh, cost = (25 + 25) / 0.75
        assert result.positions["AAPL"].shares == pytest.approx(0.75)
        assert result.positions["AAPL"].avg_cost == pytest.approx(50.0 / 0.75)
        assert result.positions["AAPL"].last_price == 50.0
        assert result.cash == 950.0


class TestApplySimulatedSell:
    def test_closes_the_position_and_credits_proceeds(self):
        portfolio = PaperPortfolio(
            cash=100.0,
            positions={"AAPL": PaperPosition(shares=1.0, avg_cost=100.0, last_price=100.0)},
        )
        result = apply_simulated_sell(portfolio, "AAPL", price=110.0)

        assert "AAPL" not in result.positions
        assert result.cash == 210.0
        assert result.realized_pnl == 10.0

    def test_does_not_mutate_the_input_portfolio(self):
        portfolio = PaperPortfolio(
            cash=100.0,
            positions={"AAPL": PaperPosition(shares=1.0, avg_cost=100.0, last_price=100.0)},
        )
        apply_simulated_sell(portfolio, "AAPL", price=110.0)

        assert "AAPL" in portfolio.positions
        assert portfolio.cash == 100.0

    def test_is_a_no_op_when_no_such_position_exists(self):
        portfolio = PaperPortfolio(cash=100.0)
        result = apply_simulated_sell(portfolio, "AAPL", price=110.0)

        assert result == portfolio

    def test_realizes_a_loss_correctly(self):
        portfolio = PaperPortfolio(
            cash=0.0,
            positions={"AAPL": PaperPosition(shares=2.0, avg_cost=100.0, last_price=100.0)},
        )
        result = apply_simulated_sell(portfolio, "AAPL", price=90.0)

        assert result.realized_pnl == -20.0
        assert result.cash == 180.0


class TestSummarize:
    def test_all_cash_no_positions(self):
        portfolio = PaperPortfolio(cash=400.0)
        summary = summarize(portfolio, initial_cash=400.0)

        assert summary["cash"] == 400.0
        assert summary["positions_value"] == 0.0
        assert summary["total_equity"] == 400.0
        assert summary["total_pnl"] == 0.0

    def test_unrealized_pnl_uses_last_price_not_avg_cost(self):
        portfolio = PaperPortfolio(
            cash=375.0,
            positions={"AAPL": PaperPosition(shares=0.25, avg_cost=100.0, last_price=120.0)},
        )
        summary = summarize(portfolio, initial_cash=400.0)

        assert summary["positions_value"] == pytest.approx(30.0)
        assert summary["unrealized_pnl"] == pytest.approx(5.0)
        assert summary["total_equity"] == pytest.approx(405.0)
        assert summary["total_pnl"] == pytest.approx(5.0)

    def test_total_pnl_combines_realized_and_unrealized(self):
        portfolio = PaperPortfolio(
            cash=420.0,
            positions={"AAPL": PaperPosition(shares=0.25, avg_cost=100.0, last_price=110.0)},
            realized_pnl=10.0,
        )
        summary = summarize(portfolio, initial_cash=400.0)

        # equity = 420 + 0.25*110 = 447.5; total_pnl = 447.5 - 400 = 47.5
        assert summary["total_pnl"] == pytest.approx(47.5)
        assert summary["realized_pnl"] == 10.0


class TestLoadSavePortfolio:
    def test_load_seeds_a_fresh_portfolio_when_no_file_exists(self, tmp_path):
        path = str(tmp_path / "paper_portfolio.json")
        portfolio = load_paper_portfolio(path, initial_cash=400.0)

        assert portfolio.cash == 400.0
        assert portfolio.positions == {}
        assert portfolio.realized_pnl == 0.0

    def test_save_then_load_round_trips(self, tmp_path):
        path = str(tmp_path / "paper_portfolio.json")
        portfolio = PaperPortfolio(
            cash=350.5,
            positions={"AAPL": PaperPosition(shares=0.5, avg_cost=99.0, last_price=101.0)},
            realized_pnl=12.34,
        )
        save_paper_portfolio(path, portfolio)
        loaded = load_paper_portfolio(path, initial_cash=999.0)

        assert loaded.cash == 350.5
        assert loaded.realized_pnl == 12.34
        assert loaded.positions["AAPL"] == PaperPosition(
            shares=0.5, avg_cost=99.0, last_price=101.0
        )

    def test_successive_runs_accumulate_state(self, tmp_path):
        path = str(tmp_path / "paper_portfolio.json")

        portfolio = load_paper_portfolio(path, initial_cash=400.0)
        portfolio = apply_simulated_buy(portfolio, "AAPL", dollars=25.0, price=100.0)
        save_paper_portfolio(path, portfolio)

        # Simulate a second, independent dry run picking up where the
        # first left off.
        portfolio2 = load_paper_portfolio(path, initial_cash=400.0)
        assert portfolio2.cash == 375.0
        assert portfolio2.positions["AAPL"].shares == 0.25

        portfolio2 = apply_simulated_buy(portfolio2, "AAPL", dollars=25.0, price=100.0)
        save_paper_portfolio(path, portfolio2)

        portfolio3 = load_paper_portfolio(path, initial_cash=400.0)
        assert portfolio3.cash == 350.0
        assert portfolio3.positions["AAPL"].shares == 0.5
