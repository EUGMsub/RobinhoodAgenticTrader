#!/usr/bin/env python3
"""
run_backtest.py
================
Loads data/bars.json (see scripts/fetch_bars.py), replays it through
src/backtest.py, and prints a readable report: the strategy's performance
next to the equal-weight watchlist and VOO-alone buy-and-hold benchmarks.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --bars data/bars.json --cash 1000
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backtest import BenchmarkResult, run_backtest
from config import AgentConfig

DEFAULT_BARS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bars.json")


def _row(label: str, value: str) -> str:
    return f"  {label:<22} {value}"


def _print_strategy_report(result, starting_cash: float) -> None:
    print("STRATEGY (mean-reversion dip-buy)")
    print(_row("Starting cash", f"${starting_cash:,.2f}"))
    print(_row("Final portfolio value", f"${result.final_portfolio_value:,.2f}"))
    print(_row("Total return", f"{result.total_return_pct:+.2f}%"))
    print(_row("Max drawdown", f"{result.max_drawdown_pct:.2f}%"))
    print(_row("Round trips", str(result.round_trips)))
    print(_row("Win rate", f"{result.win_rate_pct:.1f}%"))
    print(_row("Avg hold (days)", f"{result.avg_hold_days:.1f}"))
    filled = [o for o in result.orders if o.status == "filled"]
    discarded = [o for o in result.orders if o.status != "filled"]
    print(_row("Orders filled / discarded", f"{len(filled)} / {len(discarded)}"))


def _print_benchmark(label: str, bm: BenchmarkResult | None) -> None:
    if bm is None:
        print(f"{label}: no data available")
        return
    print(label)
    print(_row("Final value", f"${bm.final_value:,.2f}"))
    print(_row("Total return", f"{bm.total_return_pct:+.2f}%"))
    print(_row("Max drawdown", f"{bm.max_drawdown_pct:.2f}%"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", default=DEFAULT_BARS_PATH, help="Path to bars.json")
    parser.add_argument(
        "--cash", type=float, default=None, help="Override starting cash (default: cfg.initial_cash)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.bars):
        sys.exit(
            f"{args.bars} not found. Run scripts/fetch_bars.py first to generate it."
        )
    with open(args.bars) as f:
        bars = json.load(f)

    cfg = AgentConfig()
    starting_cash = args.cash if args.cash is not None else cfg.initial_cash
    result = run_backtest(bars, cfg, starting_cash=starting_cash)

    print(f"Watchlist: {list(cfg.watchlist)}")
    print(
        f"Date range: {result.equity_curve[0].date} to "
        f"{result.equity_curve[-1].date}"
        if result.equity_curve
        else "Date range: (no trading days in bars)"
    )
    print(f"Slippage: {cfg.slippage_pct:.2f}% per fill")
    print()

    _print_strategy_report(result, starting_cash)
    print()
    _print_benchmark("BENCHMARK: equal-weight watchlist buy-and-hold", result.equal_weight_benchmark)
    print()
    _print_benchmark("BENCHMARK: VOO buy-and-hold", result.voo_benchmark)


if __name__ == "__main__":
    main()
