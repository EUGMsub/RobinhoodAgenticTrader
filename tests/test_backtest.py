"""
Unit tests for backtest.py.

These prove the replay engine's mechanics (fill timing, cash/position
bookkeeping, discard rules) AND that it genuinely reuses validate_buy() /
validate_sell() from guardrails.py rather than reimplementing the rules —
several tests exist purely to show a guardrail cap changes backtest
behavior exactly as it changes the unit-tested guardrail behavior.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backtest import run_backtest
from config import AgentConfig


def _cfg(**overrides):
    defaults = dict(
        watchlist=("AAPL",),
        dip_trigger_pct=2.0,
        revert_target_pct=3.0,
        order_dollars=25.0,
        max_position_dollars=150.0,
        max_total_dollars=400.0,
        max_hold_days=10,
        disaster_stop_pct=15.0,
        correlation_groups={},
        max_group_dollars=1_000_000.0,
        slippage_pct=0.0,  # most tests assert exact prices; opt in per-test
        initial_cash=400.0,
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


def _bar(date, open_, close, high=None, low=None):
    return {
        "date": date,
        "open": open_,
        "high": high if high is not None else max(open_, close),
        "low": low if low is not None else min(open_, close),
        "close": close,
    }


class TestFillTiming:
    def test_buy_fills_at_next_bars_open_not_signal_day_close(self):
        bars = {
            "AAPL": [
                _bar("2024-01-01", 100, 100),
                _bar("2024-01-02", 98, 97),  # -3% day change -> buy signal
                _bar("2024-01-03", 96, 98),  # fill day: open=96; close +1% (no new dip)
            ]
        }
        result = run_backtest(bars, _cfg(), starting_cash=1000.0)

        assert len(result.orders) == 1
        order = result.orders[0]
        assert order.side == "buy"
        assert order.status == "filled"
        assert order.signal_date == "2024-01-02"
        assert order.fill_date == "2024-01-03"
        assert order.fill_price == 96.0  # next bar's OPEN
        assert order.fill_price != 97.0  # not the signal day's close
        assert order.shares == 25.0 / 96.0

    def test_no_signal_on_first_bar_no_prior_close_to_compare(self):
        bars = {"AAPL": [_bar("2024-01-01", 100, 90)]}
        result = run_backtest(bars, _cfg(), starting_cash=1000.0)
        assert result.orders == []

    def test_signal_with_no_next_bar_is_discarded(self):
        bars = {
            "AAPL": [
                _bar("2024-01-01", 100, 100),
                _bar("2024-01-02", 95, 95),  # -5% signal, but last bar
            ]
        }
        result = run_backtest(bars, _cfg(), starting_cash=1000.0)

        assert len(result.orders) == 1
        assert result.orders[0].status == "discarded_no_next_bar"
        assert result.final_cash == 1000.0
        assert result.final_positions == {}


class TestSellExits:
    def test_profit_target_sell_reuses_guardrail_and_realizes_pnl(self):
        bars = {
            "AAPL": [
                _bar("2024-01-01", 100, 100),
                _bar("2024-01-02", 98, 97),   # buy signal (-3%)
                _bar("2024-01-03", 96, 96),   # buy fills at open=96
                _bar("2024-01-04", 96, 103),  # +7.3% vs avg_cost 96 -> sell signal
                _bar("2024-01-05", 104, 104), # sell fills at open=104
            ]
        }
        result = run_backtest(bars, _cfg(), starting_cash=1000.0)

        sells = [o for o in result.orders if o.side == "sell"]
        assert len(sells) == 1
        sell = sells[0]
        assert sell.status == "filled"
        assert "profit_target" in sell.reason
        assert sell.fill_date == "2024-01-05"
        assert sell.fill_price == 104.0
        expected_shares = 25.0 / 96.0
        assert sell.shares == expected_shares
        expected_pnl = expected_shares * 104.0 - 25.0
        assert sell.realized_pnl == expected_pnl
        assert result.final_positions == {}

    def test_no_exit_condition_leaves_position_open(self):
        # +1% vs avg_cost: below revert target, well under max_hold_days,
        # nowhere near the disaster stop. No bar after day4, so this also
        # confirms silence — no spurious signal is recorded either.
        bars = {
            "AAPL": [
                _bar("2024-01-01", 100, 100),
                _bar("2024-01-02", 98, 97),   # buy signal (-3%)
                _bar("2024-01-03", 96, 96),   # buy fills at open=96
                _bar("2024-01-04", 96, 97),   # +1.04% vs avg_cost 96
            ]
        }
        result = run_backtest(bars, _cfg(), starting_cash=1000.0)

        sells = [o for o in result.orders if o.side == "sell"]
        assert sells == []
        assert "AAPL" in result.final_positions

    def test_time_exit_fires_on_losing_position_past_max_hold_days(self):
        bars = {
            "AAPL": [
                _bar("2024-01-01", 100, 100),
                _bar("2024-01-02", 98, 97),   # buy signal
                _bar("2024-01-03", 96, 96),   # fills at 96
            ]
        }
        # Extend with flat days between the disaster stop and profit
        # target so only the passage of time forces the exit.
        for i in range(4, 10):
            bars["AAPL"].append(_bar(f"2024-01-{i:02d}", 96, 96))
        bars["AAPL"].append(_bar("2024-01-10", 96, 96))  # held 7 days here
        bars["AAPL"].append(_bar("2024-01-11", 97, 97))  # fill day

        result = run_backtest(bars, _cfg(max_hold_days=7), starting_cash=1000.0)

        sells = [o for o in result.orders if o.side == "sell"]
        assert len(sells) == 1
        assert "time_exit" in sells[0].reason
        assert sells[0].status == "filled"

    def test_disaster_stop_fires_regardless_of_hold_time(self):
        bars = {
            "AAPL": [
                _bar("2024-01-01", 100, 100),
                _bar("2024-01-02", 98, 97),   # buy signal
                _bar("2024-01-03", 96, 96),   # fills at 96
                _bar("2024-01-04", 80, 80),   # -16.7% vs avg_cost -> disaster stop
                _bar("2024-01-05", 79, 79),   # fill day
            ]
        }
        result = run_backtest(bars, _cfg(disaster_stop_pct=15.0), starting_cash=1000.0)

        sells = [o for o in result.orders if o.side == "sell"]
        assert len(sells) == 1
        assert "disaster_stop" in sells[0].reason
        assert sells[0].fill_price == 79.0


class TestGuardrailReuse:
    def test_position_cap_blocks_repeat_buys_same_as_unit_tests(self):
        # order_dollars=25, max_position_dollars=60: two fills (50) leave
        # no room for a third (would hit 75 > 60), even though every day
        # keeps dipping and would otherwise re-trigger a buy.
        bars = {
            "AAPL": [
                _bar("2024-01-01", 100, 100),
                _bar("2024-01-02", 95, 90),  # -10% -> buy #1 signal
                _bar("2024-01-03", 85, 80),  # fill #1 @85; -11% -> buy #2 signal
                _bar("2024-01-04", 75, 70),  # fill #2 @75; -12.5% dip, but
                                              # position cap now blocks a #3
            ]
        }
        result = run_backtest(
            bars, _cfg(max_position_dollars=60.0), starting_cash=1000.0
        )

        filled_buys = [o for o in result.orders if o.side == "buy" and o.status == "filled"]
        blocked_days = [
            o for o in result.orders if o.side == "buy" and o.status != "filled"
        ]
        assert len(filled_buys) == 2
        # The third signal was never even approved by validate_buy, so it
        # never became an order at all.
        assert len(blocked_days) == 0
        assert len(result.orders) == 2

    def test_off_watchlist_ticker_never_trades_even_with_bars(self):
        bars = {
            "AAPL": [_bar("2024-01-01", 100, 100), _bar("2024-01-02", 90, 90)],
            "GME": [_bar("2024-01-01", 50, 50), _bar("2024-01-02", 40, 40)],
        }
        result = run_backtest(bars, _cfg(watchlist=("AAPL",)), starting_cash=1000.0)
        assert all(o.ticker != "GME" for o in result.orders)


class TestCashAndEquity:
    def test_insufficient_cash_discards_the_buy(self):
        bars = {
            "AAPL": [
                _bar("2024-01-01", 100, 100),
                _bar("2024-01-02", 95, 90),
                _bar("2024-01-03", 90, 90),
            ]
        }
        result = run_backtest(bars, _cfg(order_dollars=25.0), starting_cash=10.0)

        assert len(result.orders) == 1
        assert result.orders[0].status == "discarded_insufficient_cash"
        assert result.final_cash == 10.0
        assert result.final_positions == {}

    def test_fractional_shares_allowed(self):
        bars = {
            "AAPL": [
                _bar("2024-01-01", 100, 100),
                _bar("2024-01-02", 95, 90),
                _bar("2024-01-03", 33, 33),  # 25 / 33 is not a whole number
            ]
        }
        result = run_backtest(bars, _cfg(), starting_cash=1000.0)
        assert result.orders[0].shares == 25.0 / 33.0

    def test_equity_curve_has_one_point_per_trading_day_and_totals_are_consistent(self):
        bars = {
            "AAPL": [
                _bar("2024-01-01", 100, 100),
                _bar("2024-01-02", 98, 97),
                _bar("2024-01-03", 96, 96),
                _bar("2024-01-04", 96, 96),
            ]
        }
        result = run_backtest(bars, _cfg(), starting_cash=1000.0)

        assert len(result.equity_curve) == 4
        for point in result.equity_curve:
            assert point.total_equity == point.cash + point.positions_value

        last = result.equity_curve[-1]
        expected_shares = 25.0 / 96.0
        assert last.positions_value == expected_shares * 96.0
        assert last.cash == 1000.0 - 25.0

    def test_run_backtest_reads_initial_cash_from_cfg_by_default(self):
        bars = {"AAPL": [_bar("2024-01-01", 100, 100)]}
        cfg = _cfg(initial_cash=777.0)
        result = run_backtest(bars, cfg)  # starting_cash omitted
        assert result.equity_curve[0].cash == 777.0
        assert result.final_cash == 777.0


def _round_trip_bars():
    return {
        "AAPL": [
            _bar("2024-01-01", 100, 100),
            _bar("2024-01-02", 98, 97),   # buy signal (-3%)
            _bar("2024-01-03", 96, 96),   # buy fills at open=96
            _bar("2024-01-04", 96, 103),  # sell signal (profit target)
            _bar("2024-01-05", 104, 104), # sell fills at open=104
        ]
    }


class TestSlippage:
    def test_slippage_strictly_reduces_returns_on_an_identical_run(self):
        bars = _round_trip_bars()
        zero = run_backtest(bars, _cfg(slippage_pct=0.0), starting_cash=1000.0)
        half = run_backtest(bars, _cfg(slippage_pct=0.5), starting_cash=1000.0)

        assert half.final_portfolio_value < zero.final_portfolio_value
        assert half.total_return_pct < zero.total_return_pct

    def test_fill_prices_reflect_slippage_in_the_correct_direction(self):
        bars = _round_trip_bars()
        result = run_backtest(bars, _cfg(slippage_pct=0.5), starting_cash=1000.0)

        buy = next(o for o in result.orders if o.side == "buy")
        sell = next(o for o in result.orders if o.side == "sell")

        # Buys pay UP (worse for the buyer); sells receive LESS (worse for
        # the seller) — slippage always costs the side placing the order.
        assert buy.fill_price > buy.raw_open_price
        assert buy.fill_price == pytest.approx(buy.raw_open_price * 1.005)
        assert sell.fill_price < sell.raw_open_price
        assert sell.fill_price == pytest.approx(sell.raw_open_price * 0.995)


class TestCorrelationGroupCap:
    def test_group_cap_holds_exactly_during_a_market_wide_drop(self):
        # All three tickers crash together at -4%/day, well past the 2%
        # dip trigger, every day — a real diversification failure, and
        # exactly the scenario validate_batch()'s cumulative check exists
        # for: within any one day's batch, each ticker's buy is checked
        # against the running exposure left by the others already approved
        # that same day, so the group cap now holds with NO overshoot —
        # not even the "one day's worth of simultaneous approvals" slack
        # the old per-order validation allowed.
        #
        # Bars are kept short enough (6 days) that no position ever
        # reaches the disaster stop — a sell would free capacity and
        # legitimately allow more lifetime spend, which would muddy a test
        # that's specifically about the cap holding, not about resets.
        tickers = ("VOO", "AAPL", "MSFT")
        bars = {}
        for ticker in tickers:
            rows = []
            price = 100.0
            for day in range(1, 7):
                date = f"2024-01-{day:02d}"
                close = price * 0.96
                rows.append(_bar(date, price, close))
                price = close
            bars[ticker] = rows

        base = dict(
            watchlist=tickers,
            correlation_groups={"us_large_cap": tickers},
            order_dollars=25.0,
            dip_trigger_pct=2.0,
            max_position_dollars=1000.0,
            max_total_dollars=1000.0,
            slippage_pct=0.0,
        )
        capped = run_backtest(
            bars, _cfg(**base, max_group_dollars=200.0), starting_cash=2000.0
        )
        uncapped = run_backtest(
            bars, _cfg(**base, max_group_dollars=1_000_000.0), starting_cash=2000.0
        )

        assert not any(o.side == "sell" for o in capped.orders)

        capped_spent = sum(
            o.dollars for o in capped.orders if o.side == "buy" and o.status == "filled"
        )
        uncapped_spent = sum(
            o.dollars for o in uncapped.orders if o.side == "buy" and o.status == "filled"
        )

        assert capped_spent == pytest.approx(200.0)
        assert uncapped_spent > capped_spent
        assert uncapped_spent > 200.0


class TestBenchmarks:
    def test_benchmark_returns_and_drawdowns_match_hand_computed_values(self):
        bars = {
            "VOO": [
                _bar("2024-01-01", 100, 100),
                _bar("2024-01-02", 105, 105),  # +5%
                _bar("2024-01-03", 90, 90),    # -14.29% from peak
            ],
            "AAPL": [
                _bar("2024-01-01", 50, 50),
                _bar("2024-01-02", 55, 55),   # +10%
                _bar("2024-01-03", 44, 44),   # -20%
            ],
            "MSFT": [
                _bar("2024-01-01", 200, 200),
                _bar("2024-01-02", 220, 220),  # +10%
                _bar("2024-01-03", 176, 176),  # -20%
            ],
        }
        cfg = _cfg(watchlist=("AAPL", "MSFT"), initial_cash=400.0, slippage_pct=0.0)
        result = run_backtest(bars, cfg, starting_cash=400.0)

        # Equal-weight AAPL+MSFT: $200 into each at day-1 open.
        # shares: AAPL 200/50=4.0, MSFT 200/200=1.0
        # day2 value = 4*55 + 1*220 = 440 (peak); day3 = 4*44 + 1*176 = 352
        ew = result.equal_weight_benchmark
        assert ew is not None
        assert ew.final_value == pytest.approx(352.0)
        assert ew.total_return_pct == pytest.approx(-12.0)
        assert ew.max_drawdown_pct == pytest.approx(20.0)

        # VOO alone: $400 at day-1 open=100 -> 4.0 shares.
        # day2 value = 4*105 = 420 (peak); day3 = 4*90 = 360
        voo = result.voo_benchmark
        assert voo is not None
        assert voo.final_value == pytest.approx(360.0)
        assert voo.total_return_pct == pytest.approx(-10.0)
        assert voo.max_drawdown_pct == pytest.approx(60.0 / 420.0 * 100)

    def test_voo_benchmark_is_none_when_voo_has_no_bars(self):
        bars = {"AAPL": [_bar("2024-01-01", 100, 100), _bar("2024-01-02", 110, 110)]}
        result = run_backtest(bars, _cfg(watchlist=("AAPL",)), starting_cash=400.0)
        assert result.voo_benchmark is None
        assert result.equal_weight_benchmark is not None
