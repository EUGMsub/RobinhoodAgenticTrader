#!/usr/bin/env python3
"""
Entry point. Run with:  python scripts/run_agent.py

Schedule with cron for unattended daily runs once you trust the behavior,
e.g.:
    0 15 * * 1-5  cd /path/to/repo && /path/to/venv/bin/python scripts/run_agent.py >> cron.log 2>&1

Paper-trade without ever placing a real order or prompting for approval:
    python scripts/run_agent.py --dry-run

Run a cycle and immediately inspect the result in a browser:
    python scripts/run_agent.py --dry-run --dashboard
"""

import argparse
import dataclasses
import os
import sys
import webbrowser
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

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
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help=(
            "Regenerate and open the local HTML dashboard after the cycle "
            "completes. The dashboard is read-only and cannot control the agent."
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

    if args.dashboard:
        # Imported lazily so a normal run never pays for it, and so a
        # dashboard-rendering bug can never prevent a trading cycle from
        # running — by this point the cycle has already completed.
        from build_dashboard import build

        out = build(log_path=cfg.log_file, portfolio_path=cfg.paper_portfolio_file)
        print(f"\nDashboard written to {out}")
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
