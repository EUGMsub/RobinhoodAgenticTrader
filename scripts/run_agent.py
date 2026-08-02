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

Unattended (cron, Task Scheduler): --scheduled requires --dry-run, so an
unattended invocation can never block on human approval or place a real
order. A cycle that raises is logged as "cycle_failed" before the process
exits non-zero, rather than vanishing silently.
    python scripts/run_agent.py --scheduled --dry-run
"""

import argparse
import dataclasses
import os
import sys
import webbrowser
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

# The model's reports routinely contain characters (≤, →, —, ❌, …) that
# Windows' default console encoding (cp1252) can't represent. Without
# this, print(report) crashes AFTER a cycle has already run and logged
# successfully — a confusing failure that looks like the cycle itself
# broke. UTF-8 with errors="replace" means the worst case is a stray "?"
# in the terminal, never a crash.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent import run_cycle
from config import AgentConfig
from logging_utils import log_event


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
    parser.add_argument(
        "--scheduled",
        action="store_true",
        help=(
            "Mark this as an unattended/scheduled invocation (e.g. cron, Task "
            "Scheduler). Requires --dry-run — a scheduled run must never be "
            "able to block on confirm_with_human() with no terminal attached, "
            "or reach the execution path unattended."
        ),
    )
    args = parser.parse_args()

    # Checked before anything else touches config or credentials: a
    # scheduled invocation with no human present must be structurally
    # incapable of reaching approval or execution, not just unlikely to.
    if args.scheduled and not args.dry_run:
        sys.exit(
            "--scheduled requires --dry-run: without it, a scheduled run "
            "with approval_mode=True would block forever on input() with no "
            "terminal attached, and one with approval_mode=False would place "
            "real orders unattended."
        )

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

    # Logged before the cycle actually runs so a process that dies
    # mid-flight (killed, OOM, power loss) is distinguishable in
    # trade_log.jsonl from one that never started at all.
    log_event(
        cfg.log_file,
        "cycle_started",
        scheduled=args.scheduled,
        dry_run=cfg.dry_run,
        approval_mode=cfg.approval_mode,
    )

    try:
        report = run_cycle(cfg)
    except Exception as e:
        # A scheduled run has no one watching the terminal — without this,
        # an unhandled exception leaves only a Task Scheduler/cron exit
        # code and no evidence in the one place this project's audit trail
        # actually lives. Re-raised unchanged so the process still exits
        # non-zero and the traceback still reaches the scheduler's log.
        log_event(
            cfg.log_file,
            "cycle_failed",
            exception_type=type(e).__name__,
            exception_message=str(e),
        )
        raise

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
