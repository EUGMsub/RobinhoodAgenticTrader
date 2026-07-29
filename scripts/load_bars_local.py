#!/usr/bin/env python3
"""
load_bars_local.py
===================
Credential-free alternative to scripts/fetch_bars.py: pulls daily OHLC bars
for the watchlist from Yahoo Finance via yfinance instead of the Robinhood
MCP connector, and writes data/bars.json in the exact same shape
run_backtest.py already expects: {ticker: [{date, open, high, low, close},
...]}, dates as "YYYY-MM-DD" strings, one entry per ticker in the watchlist
plus VOO.

Useful for backtesting without Robinhood Agentic Trading credentials —
nothing here touches ANTHROPIC_API_KEY or ROBINHOOD_*; it's a plain market
data pull. src/backtest.py doesn't know or care which script produced its
input, and this script never imports or calls it (the engine stays
untouched).

Note: yfinance's auto_adjust=True adjusts OHLC for both splits and
dividends, whereas fetch_bars.py requests split-only adjustment from
Robinhood. For a strategy that never reasons about dividends, this is a
reasonable local approximation, not a byte-for-byte match.

Usage:
    python scripts/load_bars_local.py
    python scripts/load_bars_local.py --start 2023-01-01 --end 2025-01-01
    python scripts/load_bars_local.py --out data/my_bars.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yfinance as yf

from config import AgentConfig

DEFAULT_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bars.json")
DEFAULT_LOOKBACK_DAYS = 730  # ~2 years


def load_bars(symbols: list[str], start: str, end: str) -> dict[str, list[dict]]:
    bars: dict[str, list[dict]] = {}
    for symbol in symbols:
        history = yf.Ticker(symbol).history(
            start=start, end=end, interval="1d", auto_adjust=True
        )
        rows = []
        for date, row in history.iterrows():
            if row[["Open", "High", "Low", "Close"]].isna().any():
                continue
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "open": round(float(row["Open"]), 4),
                    "high": round(float(row["High"]), 4),
                    "low": round(float(row["Low"]), 4),
                    "close": round(float(row["Close"]), 4),
                }
            )
        bars[symbol] = rows
    return bars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_start = (
        datetime.now(timezone.utc) - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    ).strftime("%Y-%m-%d")
    default_end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    parser.add_argument(
        "--start",
        default=default_start,
        help=f"Start date, e.g. 2023-01-01 (default: {default_start}, ~2 years back)",
    )
    parser.add_argument(
        "--end",
        default=default_end,
        help=f"End date, e.g. 2025-01-01 (default: {default_end}, today)",
    )
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="Output path for bars.json")
    args = parser.parse_args()

    cfg = AgentConfig()
    symbols = list(cfg.watchlist)
    if "VOO" not in symbols:
        # VOO is always needed as a standalone benchmark, even if it isn't
        # on the live watchlist — mirrors fetch_bars.py.
        symbols.append("VOO")

    print(f"Fetching {symbols} bars from {args.start} to {args.end} via yfinance...")
    bars = load_bars(symbols, args.start, args.end)

    for ticker, rows in bars.items():
        print(f"  {ticker}: {len(rows)} bars")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(bars, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
