#!/usr/bin/env python3
"""
fetch_bars.py
=============
Pulls daily OHLC bars for the watchlist via the Robinhood Trading MCP tool
get_equity_historicals and writes them to data/bars.json in the shape
backtest.py expects: {ticker: [{date, open, high, low, close}, ...]}.

This is deliberately separate from src/backtest.py — the engine stays pure
(no network calls) and only ever consumes a bars dict handed to it. This
script is the one place that talks to the network, and it does so the same
way agent.py does: via a Claude API call with the MCP server attached,
since the Robinhood MCP connector is only reachable through that channel.

Usage:
    python scripts/fetch_bars.py --start 2023-01-01 --end 2025-01-01
    python scripts/fetch_bars.py --start 2023-01-01  # end defaults to now
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import anthropic

from config import AgentConfig

MCP_BETA_HEADER = "mcp-client-2025-11-25"
DEFAULT_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bars.json")


def _mcp_server_block(cfg: AgentConfig) -> list[dict]:
    return [
        {
            "type": "url",
            "url": cfg.mcp_url,
            "name": "robinhood-trading",
            "authorization_token": cfg.mcp_token,
        }
    ]


def _extract_text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


def _parse_bars(text: str) -> dict[str, list[dict]]:
    if "```json" not in text:
        raise ValueError(f"model response contained no ```json block:\n{text}")
    raw = text.split("```json")[1].split("```")[0]
    return json.loads(raw)


def fetch_bars(
    cfg: AgentConfig,
    client: anthropic.Anthropic,
    start: str,
    end: str,
) -> dict[str, list[dict]]:
    symbols = list(cfg.watchlist)
    if "VOO" not in symbols:
        # VOO is always needed as a standalone benchmark, even if it isn't
        # on the live watchlist.
        symbols.append("VOO")

    prompt = (
        "Call get_equity_historicals ONCE with symbols="
        f"{symbols}, start_time='{start}', end_time='{end}', "
        "interval='day', bounds='regular', adjustment_type='split'. "
        "Then return ONLY a JSON block fenced with ```json, an object "
        "mapping each ticker symbol to a chronological list of bars, each "
        "bar an object with exactly these keys: date (YYYY-MM-DD), open, "
        "high, low, close (all numbers). Drop any bar where interpolated "
        "is true. No other text, no explanation."
    )

    response = client.beta.messages.create(
        model=cfg.model,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
        mcp_servers=_mcp_server_block(cfg),
        betas=[MCP_BETA_HEADER],
    )
    return _parse_bars(_extract_text(response))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="Start date, e.g. 2023-01-01")
    parser.add_argument(
        "--end",
        default=None,
        help="End date, e.g. 2025-01-01 (defaults to now)",
    )
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="Output path for bars.json")
    args = parser.parse_args()

    end = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cfg = AgentConfig()
    missing = [
        name
        for name, val in [
            ("ANTHROPIC_API_KEY", cfg.anthropic_api_key),
            ("ROBINHOOD_MCP_URL", cfg.mcp_url),
            ("ROBINHOOD_MCP_TOKEN", cfg.mcp_token),
        ]
        if not val
    ]
    if missing:
        sys.exit(
            f"Missing required environment variables: {', '.join(missing)}. "
            "See .env.example."
        )

    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    print(f"Fetching {list(cfg.watchlist)} + VOO bars from {args.start} to {end}...")
    bars = fetch_bars(cfg, client, args.start, end)

    for ticker, rows in bars.items():
        print(f"  {ticker}: {len(rows)} bars")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(bars, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
