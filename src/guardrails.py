"""
guardrails.py
=============
Pure, deterministic safety checks for the trading agent.

These functions contain ZERO calls to the LLM or the broker. They exist so
that every trade the agent proposes is validated by ordinary, testable code
before it can ever reach a real order — the model's judgment is never the
last line of defense.

Design principle: an LLM can misread a number, get talked into something by
adversarial context, or simply make a mistake. Code that reasons over floats
and a fixed watchlist cannot. Keeping this logic separate and pure (no I/O)
means it can be exhaustively unit tested, which is exactly what tests/
does.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailConfig:
    watchlist: tuple[str, ...]
    order_dollars: float
    max_position_dollars: float
    max_total_dollars: float
    dip_trigger_pct: float
    revert_target_pct: float


@dataclass(frozen=True)
class GuardrailResult:
    approved: bool
    reason: str


def validate_buy(
    cfg: GuardrailConfig,
    ticker: str,
    dollars: float,
    day_change_pct: float,
    current_position_value: float,
    current_total_exposure: float,
) -> GuardrailResult:
    """Validate a proposed BUY order against every hard rule.

    All checks are inclusive-boundary and fail closed: if a value is
    missing, malformed, or borderline, the order is rejected rather than
    approved.
    """
    ticker = ticker.upper().strip()

    if ticker not in cfg.watchlist:
        return GuardrailResult(False, f"{ticker} is not on the approved watchlist")

    if dollars <= 0:
        return GuardrailResult(False, "order amount must be positive")

    if dollars > cfg.order_dollars + 1e-9:
        return GuardrailResult(
            False,
            f"${dollars:.2f} exceeds the per-order cap of ${cfg.order_dollars:.2f}",
        )

    if day_change_pct > -cfg.dip_trigger_pct:
        return GuardrailResult(
            False,
            f"{ticker} is only down {day_change_pct:.2f}%, "
            f"below the {cfg.dip_trigger_pct:.2f}% dip trigger",
        )

    if current_position_value + dollars > cfg.max_position_dollars + 1e-9:
        return GuardrailResult(
            False,
            f"would push {ticker} position to "
            f"${current_position_value + dollars:.2f}, "
            f"over the ${cfg.max_position_dollars:.2f} per-position cap",
        )

    if current_total_exposure + dollars > cfg.max_total_dollars + 1e-9:
        return GuardrailResult(
            False,
            f"would push total exposure to "
            f"${current_total_exposure + dollars:.2f}, "
            f"over the ${cfg.max_total_dollars:.2f} account cap",
        )

    return GuardrailResult(
        True,
        f"{ticker} down {day_change_pct:.2f}% (trigger {cfg.dip_trigger_pct:.2f}%), "
        f"within position and account caps",
    )


def validate_sell(
    cfg: GuardrailConfig,
    ticker: str,
    current_price: float,
    avg_cost: float,
) -> GuardrailResult:
    """Validate a proposed full-position SELL order."""
    ticker = ticker.upper().strip()

    if ticker not in cfg.watchlist:
        return GuardrailResult(False, f"{ticker} is not on the approved watchlist")

    if avg_cost <= 0:
        return GuardrailResult(False, "no valid cost basis to compare against")

    gain_pct = ((current_price - avg_cost) / avg_cost) * 100

    if gain_pct < cfg.revert_target_pct - 1e-9:
        return GuardrailResult(
            False,
            f"{ticker} is only up {gain_pct:.2f}%, "
            f"below the {cfg.revert_target_pct:.2f}% revert target",
        )

    return GuardrailResult(
        True, f"{ticker} up {gain_pct:.2f}%, meets the revert target"
    )
