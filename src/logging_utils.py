"""
logging_utils.py
=================
Append-only structured logging. Every cycle report, every proposed order,
every human decision, and every execution result gets written here so the
agent's behavior is fully auditable after the fact.
"""

import json
from datetime import datetime, timezone


def log_event(log_file: str, event: str, **fields) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
