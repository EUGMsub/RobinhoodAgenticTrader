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

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

DEFAULT_CORRELATION_GROUPS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"us_large_cap": ("VOO", "AAPL", "MSFT")}
)


@dataclass(frozen=True)
class GuardrailConfig:
    watchlist: tuple[str, ...]
    order_dollars: float
    max_position_dollars: float
    max_total_dollars: float
    dip_trigger_pct: float
    revert_target_pct: float
    max_hold_days: int = 10
    disaster_stop_pct: float = 15.0
    correlation_groups: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: DEFAULT_CORRELATION_GROUPS
    )
    max_group_dollars: float = 200.0


@dataclass(frozen=True)
class GuardrailResult:
    approved: bool
    reason: str


def validate_instrument(ticker: str) -> GuardrailResult:
    """Reject anything that is not a plain equity ticker symbol.

    Robinhood's agentic channel now supports long options orders, so the
    execution-stage model call (which has the full MCP toolset attached)
    could in principle place an option contract instead of the plain
    equity order the human approved. This is the last-line, code-level
    check against that: option-contract identifiers always fail here,
    whether OCC-style with an internal space ("AAPL 260815C00200000") or
    without ("AAPL260815C00200000"), because both contain whitespace or
    digits that no real equity ticker does.
    """
    raw = str(ticker)

    if any(ch.isspace() for ch in raw):
        return GuardrailResult(
            False,
            f"'{raw}' contains whitespace — not a plain equity symbol "
            "(possible option contract)",
        )

    symbol = raw.upper().strip()

    if not symbol:
        return GuardrailResult(False, "ticker is empty")

    if any(ch.isdigit() for ch in symbol):
        return GuardrailResult(
            False,
            f"'{symbol}' contains digits — not a plain equity symbol "
            "(possible option contract)",
        )

    if len(symbol) > 5:
        return GuardrailResult(
            False,
            f"'{symbol}' is longer than 5 characters — not a plain equity symbol",
        )

    if not symbol.isalpha():
        return GuardrailResult(
            False,
            f"'{symbol}' contains non-letter characters — not a plain equity symbol",
        )

    return GuardrailResult(True, f"{symbol} is a plain equity symbol")


def validate_buy(
    cfg: GuardrailConfig,
    ticker: str,
    dollars: float,
    day_change_pct: float,
    positions: dict[str, float],
) -> GuardrailResult:
    """Validate a proposed BUY order against every hard rule.

    All checks are inclusive-boundary and fail closed: if a value is
    missing, malformed, or borderline, the order is rejected rather than
    approved.

    `positions` maps ticker -> current market value for every position in
    the account. Per-ticker value, total exposure, and correlation-group
    exposure are all derived from it here, in code — the caller never gets
    to hand this function a pre-aggregated total to trust.
    """
    ticker = ticker.upper().strip()

    instrument_result = validate_instrument(ticker)
    if not instrument_result.approved:
        return instrument_result

    normalized_positions: dict[str, float] = {}
    for pos_ticker, value in positions.items():
        key = str(pos_ticker).upper().strip()
        normalized_positions[key] = normalized_positions.get(key, 0.0) + value

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

    current_position_value = normalized_positions.get(ticker, 0.0)
    if current_position_value + dollars > cfg.max_position_dollars + 1e-9:
        return GuardrailResult(
            False,
            f"would push {ticker} position to "
            f"${current_position_value + dollars:.2f}, "
            f"over the ${cfg.max_position_dollars:.2f} per-position cap",
        )

    for group_name, group_tickers in cfg.correlation_groups.items():
        if ticker not in group_tickers:
            continue
        group_exposure = sum(
            normalized_positions.get(t, 0.0) for t in group_tickers
        )
        if group_exposure + dollars > cfg.max_group_dollars + 1e-9:
            return GuardrailResult(
                False,
                f"would push the '{group_name}' correlation group "
                f"({', '.join(group_tickers)}) to "
                f"${group_exposure + dollars:.2f}, over the "
                f"${cfg.max_group_dollars:.2f} group cap",
            )

    current_total_exposure = sum(normalized_positions.values())
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
        f"within position, group, and account caps",
    )


def validate_sell(
    cfg: GuardrailConfig,
    ticker: str,
    current_price: float,
    avg_cost: float,
    days_held: int,
) -> GuardrailResult:
    """Validate a proposed full-position SELL order.

    A sale is approved if ANY exit condition is met:
      - profit_target: price has reverted to the target gain
      - time_exit: the position has been held at least max_hold_days
      - disaster_stop: price has fallen to the disaster stop loss

    This guarantees a losing position is never held indefinitely — either
    it reverts, or it is force-exited by time or by the disaster stop.
    """
    ticker = ticker.upper().strip()

    instrument_result = validate_instrument(ticker)
    if not instrument_result.approved:
        return instrument_result

    if ticker not in cfg.watchlist:
        return GuardrailResult(False, f"{ticker} is not on the approved watchlist")

    if avg_cost <= 0:
        return GuardrailResult(False, "no valid cost basis to compare against")

    gain_pct = ((current_price - avg_cost) / avg_cost) * 100

    if gain_pct >= cfg.revert_target_pct - 1e-9:
        return GuardrailResult(
            True,
            f"profit_target: {ticker} up {gain_pct:.2f}%, "
            f"meets the {cfg.revert_target_pct:.2f}% revert target",
        )

    if days_held >= cfg.max_hold_days:
        return GuardrailResult(
            True,
            f"time_exit: {ticker} held {days_held} days, "
            f"at or beyond the {cfg.max_hold_days}-day max hold",
        )

    if gain_pct <= -cfg.disaster_stop_pct + 1e-9:
        return GuardrailResult(
            True,
            f"disaster_stop: {ticker} down {gain_pct:.2f}%, "
            f"at or beyond the {cfg.disaster_stop_pct:.2f}% disaster stop",
        )

    return GuardrailResult(
        False,
        f"{ticker} is {gain_pct:.2f}% from cost basis (target "
        f"{cfg.revert_target_pct:.2f}%, disaster stop "
        f"-{cfg.disaster_stop_pct:.2f}%), held {days_held}/"
        f"{cfg.max_hold_days} days — no exit condition met",
    )


def validate_batch(
    cfg: GuardrailConfig,
    orders: list[dict],
    positions: dict[str, float],
) -> list[tuple[dict, GuardrailResult]]:
    """Validate a batch of proposed orders CUMULATIVELY instead of
    independently, closing a time-of-check-to-time-of-use gap: calling
    validate_buy() once per order against the same static `positions`
    snapshot lets every order in the batch pass as if it were the only
    one. If every ticker in a correlation group dips on the same day, each
    of their buys clears the group cap independently and, taken together,
    can overshoot it by up to (group_size - 1) * order_dollars — exactly
    the scenario max_group_dollars exists to prevent. This function checks
    orders one at a time against a running position snapshot that is
    updated after each approval, so later orders in the batch are checked
    against the true post-approval state, not the stale pre-batch one.

    Processing order is fixed and deterministic — it decides which ticker
    wins scarce capacity when the batch can't all fit, so it must be
    auditable rather than dependent on input order or dict iteration:

      1. All SELL orders, sorted alphabetically by ticker. Sells only free
         capacity (they reduce position/group/total exposure), so running
         them first gives every buy in the batch the benefit of whatever
         room they free — never less room than evaluating buys first would.
      2. All BUY orders, sorted by day_change_pct ascending (the largest
         dip first), ties broken alphabetically by ticker. When capacity
         is scarce, the ticker showing the strongest signal — furthest
         past its trigger — gets first claim on it.

    Each order dict must carry `ticker` and `side` ("buy" | "sell"), plus
    whatever validate_buy()/validate_sell() need for that side (buy:
    `dollars`, `day_change_pct`; sell: `current_price`, `avg_cost`,
    `days_held`). Orders with any other `side` are rejected outright.

    `positions` is never mutated — a normalized local copy tracks running
    state. Returns (order, GuardrailResult) pairs in the order they were
    evaluated (sells then sorted buys, not the caller's original order),
    so the sequence itself documents why each verdict landed where it did,
    including rejections caused by earlier approvals in the same batch.
    """
    running_positions: dict[str, float] = {}
    for pos_ticker, value in positions.items():
        key = str(pos_ticker).upper().strip()
        running_positions[key] = running_positions.get(key, 0.0) + value

    def _ticker(order: dict) -> str:
        return str(order.get("ticker", "")).upper().strip()

    sells = sorted((o for o in orders if o.get("side") == "sell"), key=_ticker)
    buys = sorted(
        (o for o in orders if o.get("side") == "buy"),
        key=lambda o: (float(o.get("day_change_pct", 0.0)), _ticker(o)),
    )
    unrecognized = [o for o in orders if o.get("side") not in ("buy", "sell")]

    results: list[tuple[dict, GuardrailResult]] = []

    for order in sells:
        ticker = _ticker(order)
        result = validate_sell(
            cfg,
            ticker=ticker,
            current_price=float(order.get("current_price", 0)),
            avg_cost=float(order.get("avg_cost", 0)),
            days_held=int(order.get("days_held", 0)),
        )
        if result.approved:
            running_positions[ticker] = 0.0
        results.append((order, result))

    for order in buys:
        ticker = _ticker(order)
        dollars = float(order.get("dollars", 0))
        result = validate_buy(
            cfg,
            ticker=ticker,
            dollars=dollars,
            day_change_pct=float(order.get("day_change_pct", 0)),
            positions=running_positions,
        )
        if result.approved:
            running_positions[ticker] = running_positions.get(ticker, 0.0) + dollars
        results.append((order, result))

    for order in unrecognized:
        results.append(
            (order, GuardrailResult(False, f"unrecognized side '{order.get('side')}'"))
        )

    return results
