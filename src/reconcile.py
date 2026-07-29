"""
reconcile.py
============
Pure post-execution check: compares what the agent actually placed, as
reported by the broker's own order records, against what the human
approved. No I/O — it takes already-fetched order data and returns a
verdict.

This exists because the execution step gives Claude the full MCP toolset
and a natural-language instruction ("place NO other orders"). That
instruction is not a guarantee. This is the code-level check that verifies
it was actually followed, independent of what the model claims it did.
"""

from dataclasses import dataclass

DOLLAR_TOLERANCE_PCT = 1.0


@dataclass(frozen=True)
class ReconcileResult:
    passed: bool
    reason: str


def reconcile_order(
    approved: dict,
    equity_orders: list[dict],
    option_orders: list[dict],
) -> ReconcileResult:
    """Verify that exactly the approved equity order was placed.

    Fails (MISMATCH) if any of:
      - any option order was placed at all — this strategy is equities-only
      - the number of new equity orders is not exactly 1
      - the placed order's symbol or side differs from what was approved
      - the placed order's dollar amount differs from approved by more
        than DOLLAR_TOLERANCE_PCT

    `approved` is the same order dict guardrails.py already validated
    (ticker, side, dollars). `equity_orders` / `option_orders` are the raw
    order records fetched from the broker after execution.
    """
    if option_orders:
        return ReconcileResult(
            False,
            f"{len(option_orders)} option order(s) were placed — this "
            "strategy is equities-only",
        )

    if len(equity_orders) != 1:
        return ReconcileResult(
            False,
            f"expected exactly 1 new equity order, found {len(equity_orders)}",
        )

    order = equity_orders[0]

    approved_ticker = str(approved.get("ticker", "")).upper().strip()
    approved_side = str(approved.get("side", "")).lower().strip()
    approved_dollars = float(approved.get("dollars", 0))

    order_ticker = str(order.get("symbol", "")).upper().strip()
    order_side = str(order.get("side", "")).lower().strip()
    order_dollars = float(order.get("dollar_amount", 0))

    if order_ticker != approved_ticker:
        return ReconcileResult(
            False,
            f"symbol mismatch: approved {approved_ticker}, "
            f"executed {order_ticker or '(none)'}",
        )

    if order_side != approved_side:
        return ReconcileResult(
            False,
            f"side mismatch: approved {approved_side}, "
            f"executed {order_side or '(none)'}",
        )

    if approved_dollars <= 0:
        return ReconcileResult(
            False, "approved dollar amount is not positive; cannot verify"
        )

    pct_diff = abs(order_dollars - approved_dollars) / approved_dollars * 100
    if pct_diff > DOLLAR_TOLERANCE_PCT + 1e-9:
        return ReconcileResult(
            False,
            f"dollar amount mismatch: approved ${approved_dollars:.2f}, "
            f"executed ${order_dollars:.2f} ({pct_diff:.2f}% difference, "
            f"over the {DOLLAR_TOLERANCE_PCT:.1f}% tolerance)",
        )

    return ReconcileResult(
        True,
        f"executed order matches approved order: {approved_side} "
        f"${order_dollars:.2f} of {order_ticker}",
    )
