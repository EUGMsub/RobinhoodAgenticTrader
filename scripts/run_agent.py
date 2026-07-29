#!/usr/bin/env python3
"""
Entry point. Run with:  python scripts/run_agent.py

Schedule with cron for unattended daily runs once you trust the behavior,
e.g.:
    0 15 * * 1-5  cd /path/to/repo && /path/to/venv/bin/python scripts/run_agent.py >> cron.log 2>&1

Paper-trade without ever placing a real order or prompting for approval:
    python scripts/run_agent.py --dry-run
"""

import argparse
import dataclasses
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import run_cycle
from config import AgentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Simulate the cycle against a persisted paper portfolio: no "
            "human prompt, no real order ever placed."
        ),
    )
    args = parser.parse_args()

    cfg = AgentConfig()
    if args.dry_run:
        cfg = dataclasses.replace(cfg, dry_run=True)

    try:
        cfg.validate()
    except EnvironmentError as e:
        sys.exit(str(e))

    print(f"Mean-Reversion Agent — {datetime.now():%Y-%m-%d %H:%M}")
    print(
        f"Watchlist: {list(cfg.watchlist)} | Approval mode: {cfg.approval_mode} "
        f"| Dry run: {cfg.dry_run}"
    )

    report = run_cycle(cfg)
    print("\n===== AGENT REPORT =====\n")
    print(report)


if __name__ == "__main__":
    main()
