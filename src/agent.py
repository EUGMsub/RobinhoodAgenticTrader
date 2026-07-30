"""
agent.py
========
Orchestration layer. Every live cycle follows the same four steps, in the
same order, regardless of config:

  1. propose  — one MCP call, always JSON proposals, always READ_ONLY_TOOLS.
     There is no "just execute directly" path; the model is never given
     an instruction or a tool that lets it place an order from this call.
  2. validate — validate_batch() re-derives every decision from
     guardrails.py, cumulatively over the whole proposed batch. A
     guardrail-rejected order never reaches step 3 or 4, no matter what
     approval_mode or dry_run are set to.
  3. approve  — approval_mode=True prompts a human per order via
     confirm_with_human(); approval_mode=False auto-approves whatever
     guardrails already approved (logged as "auto_approved") and never
     skips validation to get there.
  4. execute  — always through _execute_and_reconcile(), one order at a
     time, with EXECUTION_TOOLS and post-execution reconciliation.

dry_run stops after step 2 and simulates fills against a paper portfolio
instead of running steps 3/4 at all (see _run_dry_run_cycle()).

Nothing here trusts the model's arithmetic. The model is used for what LLMs
are good at (reading unstructured market data and applying a strategy in
natural language); guardrails.py is used for what plain code is good at
(enforcing hard numeric limits it cannot talk itself out of).

Tool surface is restricted per call, not just steered by prompt text: each
client.beta.messages.create() call passes a `tools` array (an MCPToolset,
see _mcp_toolset()) that allowlists only the tools that call actually needs.
A proposal or dry-run call never has place_equity_order / place_option_order
available at all — the model could not call them even if a prompt injection
talked it into trying.

Restriction lives in the top-level `tools` array as an MCPToolset entry
(default_config/configs), not as a `tool_configuration` key inside the
`mcp_servers` block. The latter was the pre-migration shape, deprecated
alongside the mcp-client-2025-04-04 beta; MCP_BETA_HEADER here targets a
later beta than the documented mcp-client-2025-11-20 migration point, so it
must use the current shape. See
docs.claude.com/en/docs/agents-and-tools/mcp-connector.
"""

import json
import sys
from datetime import datetime, timezone

import anthropic

from config import AgentConfig
from guardrails import validate_batch
from logging_utils import log_event
from paper_trading import (
    apply_simulated_buy,
    apply_simulated_sell,
    load_paper_portfolio,
    save_paper_portfolio,
    summarize,
)
from reconcile import reconcile_order
from strategy import build_approval_mode_instruction, build_system_prompt

# Verified against platform.claude.com/docs/en/agents-and-tools/mcp-connector
# on 2026-07-30 — "mcp-client-2025-11-20" is the current value; the previous
# "mcp-client-2025-11-25" was never valid and the API rejects it with a 400.
MCP_BETA_HEADER = "mcp-client-2025-11-20"
MCP_SERVER_NAME = "robinhood-trading"

# Verified against the live Robinhood Trading MCP tool list — not guessed.
# The quote/position/historical/tax-lot tools the strategy needs to read
# market data and form a proposal. No order-placing tool.
READ_ONLY_TOOLS: tuple[str, ...] = (
    "get_equity_quotes",
    "get_equity_positions",
    "get_accounts",
    "get_equity_historicals",
    "get_equity_tax_lots",
)

# Everything READ_ONLY_TOOLS has, plus the one tool that actually places a
# trade. Used only where an order is genuinely meant to be placed.
EXECUTION_TOOLS: tuple[str, ...] = READ_ONLY_TOOLS + ("place_equity_order",)

# Reconciliation only ever reads order records back — never re-derives a
# quote or places anything. get_option_orders stays in this list on
# purpose: reconciliation's whole job is to notice an option order that
# shouldn't be there, and it can't notice one it isn't allowed to fetch.
RECONCILE_TOOLS: tuple[str, ...] = ("get_equity_orders", "get_option_orders")


def _mcp_server_block(cfg: AgentConfig) -> list[dict]:
    return [
        {
            "type": "url",
            "url": cfg.mcp_url,
            "name": MCP_SERVER_NAME,
            "authorization_token": cfg.mcp_token,
        }
    ]


def _mcp_toolset(allowed_tools: tuple[str, ...] | None = None) -> list[dict]:
    """Build the `tools` array parameter that restricts which tools from
    the Robinhood MCP server are enabled for a given call.

    allowed_tools=None enables every tool on the server (used nowhere in
    this file, but available for completeness). Otherwise every tool is
    disabled by default and only the named tools are explicitly
    re-enabled — an allowlist, not a denylist, so a new tool added to the
    server in the future is excluded by default rather than silently
    exposed.
    """
    toolset: dict = {"type": "mcp_toolset", "mcp_server_name": MCP_SERVER_NAME}
    if allowed_tools is not None:
        toolset["default_config"] = {"enabled": False}
        toolset["configs"] = {tool: {"enabled": True} for tool in allowed_tools}
    return [toolset]


def _extract_text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


def _parse_proposed_orders(text: str) -> list[dict]:
    if "```json" not in text:
        return []
    try:
        raw = text.split("```json")[1].split("```")[0]
        return json.loads(raw).get("proposed_orders", [])
    except (json.JSONDecodeError, IndexError):
        return []


def _parse_market_snapshot(text: str) -> dict | None:
    """Extract the model's per-ticker quote snapshot from a cycle report.

    Returns None (not {}) when the block is missing or unparseable, so the
    caller can distinguish "the model reported an empty snapshot" from
    "there was no snapshot at all" — both are failures, but only the
    second means the response itself was malformed.
    """
    if "```json" not in text:
        return None
    try:
        raw = text.split("```json")[1].split("```")[0]
        snapshot = json.loads(raw).get("market_snapshot")
    except (json.JSONDecodeError, IndexError):
        return None
    return snapshot if isinstance(snapshot, dict) else None


def _log_market_snapshot(cfg: AgentConfig, text: str) -> None:
    """Log what the model actually saw this cycle, every cycle.

    Without this, a no-trade day is indistinguishable from a broken tool
    call: both produce zero orders. Logging the snapshot unconditionally
    means "fetched real prices, nothing qualified" leaves evidence, and a
    missing or partial snapshot raises a loud, specific signal instead of
    looking like a quiet, uneventful day.
    """
    snapshot = _parse_market_snapshot(text)

    if snapshot is None:
        print(
            "\n*** WARNING: no market_snapshot in this cycle's response — "
            "cannot confirm the agent actually fetched market data. Treat "
            "this cycle's inaction as UNEXPLAINED, not as 'nothing "
            "qualified'. ***"
        )
        log_event(
            cfg.log_file,
            "market_snapshot_incomplete",
            missing_tickers=list(cfg.watchlist),
            reason="no market_snapshot object in the model's response",
        )
        return

    normalized = {str(t).upper().strip(): v for t, v in snapshot.items()}
    missing = [t for t in cfg.watchlist if t.upper().strip() not in normalized]

    log_event(cfg.log_file, "market_snapshot", snapshot=normalized)

    if missing:
        print(
            f"\n*** WARNING: market_snapshot is missing {', '.join(missing)} "
            "— the agent may not have successfully fetched quotes for "
            "every watchlist ticker this cycle. ***"
        )
        log_event(
            cfg.log_file,
            "market_snapshot_incomplete",
            missing_tickers=missing,
            reason="watchlist tickers absent from the model's market_snapshot",
        )


def _log_api_usage(cfg: AgentConfig, call: str, response) -> None:
    """Record token usage and an ESTIMATED dollar cost for one API call.

    `call` names which step spent the tokens (proposal / execution /
    reconciliation) so cost can be attributed rather than just totalled.
    The dollar figure is derived from operator-supplied prices in
    AgentConfig and is an estimate; the token counts are the ground truth.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    def _tokens(name: str) -> int:
        value = getattr(usage, name, None)
        return int(value) if isinstance(value, (int, float)) else 0

    input_tokens = _tokens("input_tokens")
    output_tokens = _tokens("output_tokens")

    cost = (
        input_tokens / 1_000_000 * cfg.input_token_price_per_mtok
        + output_tokens / 1_000_000 * cfg.output_token_price_per_mtok
    )

    fields = {
        "call": call,
        "model": cfg.model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": cost,
    }

    # Cache fields only exist when prompt caching is in play. Record them
    # when present — they're a large part of real cost — but don't invent
    # zeros for them, which would misrepresent a non-caching call.
    for cache_field in (
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        value = getattr(usage, cache_field, None)
        if isinstance(value, (int, float)):
            fields[cache_field] = int(value)

    log_event(cfg.log_file, "api_usage", **fields)


def _parse_order_records(text: str) -> tuple[list[dict], list[dict]]:
    if "```json" not in text:
        return [], []
    try:
        raw = text.split("```json")[1].split("```")[0]
        data = json.loads(raw)
        return data.get("equity_orders", []), data.get("option_orders", [])
    except (json.JSONDecodeError, IndexError):
        return [], []


def confirm_with_human(prompt: str) -> bool:
    """Blocking CLI confirmation. Swap this out for a web/CLI UI as needed —
    the important part is that a human, not the model, makes the final call
    in approval mode."""
    ans = input(f"\n{prompt}\nType YES to approve, anything else to skip: ")
    return ans.strip() == "YES"


def _execute_and_reconcile(
    cfg: AgentConfig,
    client: anthropic.Anthropic,
    system_prompt: str,
    order: dict,
    running_positions: dict[str, float],
) -> None:
    """Place the human-approved order via MCP, then independently
    reconcile what the broker actually recorded. This is the ONLY place
    an order-placing MCP call happens — dry_run must never reach it.

    Guarded with an explicit raise rather than `assert`: assert statements
    are stripped entirely when Python runs with the `-O` flag, which would
    silently disable this guard under optimized execution. A RuntimeError
    is not optimized away.
    """
    if cfg.dry_run:
        raise RuntimeError(
            "execution path reached while dry_run=True — this should be "
            "unreachable; refusing to place an order"
        )

    ticker = str(order.get("ticker", "")).upper().strip()
    side = order.get("side")
    dollars = float(order.get("dollars", 0))

    execution_started_at = datetime.now(timezone.utc).isoformat()

    exec_response = client.beta.messages.create(
        model=cfg.model,
        max_tokens=2000,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": (
                    "The human approved exactly this order. Place it "
                    "now via the MCP order tools and confirm the "
                    f"result: {side} ${dollars:.2f} of {ticker}. "
                    "Place NO other orders."
                ),
            }
        ],
        mcp_servers=_mcp_server_block(cfg),
        tools=_mcp_toolset(EXECUTION_TOOLS),
        betas=[MCP_BETA_HEADER],
    )
    _log_api_usage(cfg, "execution", exec_response)
    exec_text = _extract_text(exec_response)
    print(f"\nEXECUTION RESULT:\n{exec_text}")

    if side == "buy":
        running_positions[ticker] = running_positions.get(ticker, 0.0) + dollars
    elif side == "sell":
        running_positions[ticker] = 0.0

    log_event(
        cfg.log_file,
        "order_executed",
        order=order,
        result=exec_text,
        positions_after=running_positions,
    )

    # Nothing above proves the model placed exactly (and only) the
    # approved order — it has the full MCP toolset attached and was
    # only told not to misuse it. Independently re-fetch what the
    # broker actually recorded and reconcile in plain code.
    reconcile_response = client.beta.messages.create(
        model=cfg.model,
        max_tokens=1500,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": (
                    "Fetch every order you just placed using the MCP "
                    "tools get_equity_orders and get_option_orders "
                    f"for account {cfg.agentic_account_number}, both "
                    "filtered to placed_agent='agentic' and "
                    f"created_at_gte='{execution_started_at}'. "
                    "Return ONLY a JSON block fenced with ```json "
                    "containing keys 'equity_orders' and "
                    "'option_orders', each the raw list of order "
                    "objects returned by the respective tool (empty "
                    "list if none). No other text."
                ),
            }
        ],
        mcp_servers=_mcp_server_block(cfg),
        tools=_mcp_toolset(RECONCILE_TOOLS),
        betas=[MCP_BETA_HEADER],
    )
    _log_api_usage(cfg, "reconciliation", reconcile_response)
    reconcile_text = _extract_text(reconcile_response)
    equity_orders, option_orders = _parse_order_records(reconcile_text)
    reconciliation = reconcile_order(order, equity_orders, option_orders)

    if not reconciliation.passed:
        print(
            "\n*** EXECUTION MISMATCH: what the broker recorded does "
            f"not match what was approved — {reconciliation.reason} ***"
        )
        log_event(
            cfg.log_file,
            "execution_mismatch",
            order=order,
            exec_result=exec_text,
            equity_orders=equity_orders,
            option_orders=option_orders,
            reason=reconciliation.reason,
        )
    else:
        log_event(
            cfg.log_file,
            "execution_verified",
            order=order,
            reason=reconciliation.reason,
        )


def _run_dry_run_cycle(cfg: AgentConfig, gcfg, proposed: list[dict]) -> None:
    """Simulate a cycle's approved orders against a persisted paper
    portfolio. No human prompt, no MCP order-placing call, no MCP
    reconciliation call — dry_run only ever reads market data (already
    fetched in the caller's cycle-report call) and simulates fills locally.
    """
    assert cfg.dry_run, "internal error: dry-run path reached with dry_run=False"

    portfolio = load_paper_portfolio(cfg.paper_portfolio_file, cfg.initial_cash)

    # The paper portfolio, not the model-reported (real, likely unfunded)
    # account, is the source of truth for guardrail checks in dry_run —
    # otherwise successive dry runs would never accumulate exposure and
    # the caps this whole project exists to test would never bind.
    positions = {
        ticker: pos.shares * pos.last_price for ticker, pos in portfolio.positions.items()
    }
    batch_results = validate_batch(gcfg, proposed, positions)

    for order, result in batch_results:
        ticker = str(order.get("ticker", "")).upper().strip()
        side = order.get("side")
        dollars = float(order.get("dollars", 0))

        if not result.approved:
            print(f"BLOCKED by guardrails: {ticker} {side} — {result.reason}")
            log_event(cfg.log_file, "order_blocked", order=order, reason=result.reason)
            continue

        price = float(order.get("current_price", 0))
        if price <= 0:
            print(f"SKIPPED (no current_price to simulate a fill): {ticker} {side}")
            log_event(
                cfg.log_file,
                "order_blocked",
                order=order,
                reason="missing current_price for simulated fill",
            )
            continue

        if side == "buy":
            portfolio = apply_simulated_buy(portfolio, ticker, dollars, price)
        elif side == "sell":
            portfolio = apply_simulated_sell(portfolio, ticker, price)
        else:
            continue

        print(
            f"SIMULATED FILL: {side.upper()} ${dollars:.2f} of {ticker} "
            f"@ ${price:.2f} — {result.reason}"
        )
        log_event(
            cfg.log_file,
            "simulated_fill",
            ticker=ticker,
            side=side,
            dollars=dollars,
            simulated_price=price,
            reason=result.reason,
            # The originating proposal, carried through so the model's own
            # stated numbers can later be compared against the guardrail's
            # verdict for this same order.
            order=order,
        )

    save_paper_portfolio(cfg.paper_portfolio_file, portfolio)

    summary = summarize(portfolio, cfg.initial_cash)
    print("\n===== PAPER PORTFOLIO (dry_run) =====")
    if portfolio.positions:
        for ticker, pos in sorted(portfolio.positions.items()):
            print(
                f"  {ticker}: {pos.shares:.4f} shares @ avg "
                f"${pos.avg_cost:.2f} (last ${pos.last_price:.2f})"
            )
    else:
        print("  (no open positions)")
    print(f"  Cash: ${summary['cash']:.2f}")
    print(f"  Total simulated P&L: ${summary['total_pnl']:+.2f}")


def run_cycle(cfg: AgentConfig, client: anthropic.Anthropic | None = None) -> str:
    cfg.validate()
    client = client or anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    system_prompt = build_system_prompt(cfg)
    user_msg = (
        "Run one cycle of the strategy now. Check quotes for the watchlist, "
        "check my agentic account positions and buying power, then apply "
        f"the rules. {build_approval_mode_instruction()}"
    )

    # 1. PROPOSE. Every live path — approval_mode on or off, dry_run or
    # not — makes exactly this one call to form a proposal: JSON output
    # only, READ_ONLY_TOOLS only. There is no "execute directly" prompt
    # variant and no call anywhere that gives this step an order-placing
    # tool; only step 4 (_execute_and_reconcile) ever gets EXECUTION_TOOLS.
    response = client.beta.messages.create(
        model=cfg.model,
        max_tokens=4000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
        mcp_servers=_mcp_server_block(cfg),
        tools=_mcp_toolset(READ_ONLY_TOOLS),
        betas=[MCP_BETA_HEADER],
    )
    _log_api_usage(cfg, "proposal", response)

    text = _extract_text(response)
    log_event(cfg.log_file, "cycle_report", report=text)

    # Log what the model saw BEFORE any validation or early return, so a
    # zero-order cycle still leaves proof that data was actually fetched.
    _log_market_snapshot(cfg, text)

    proposed = _parse_proposed_orders(text)
    if not proposed:
        return text

    # 2. VALIDATE. Always validate_batch() over the full proposal set,
    # cumulatively — so that (for example) three watchlist tickers dipping
    # on the same day can't each independently clear the group cap and
    # collectively overshoot it. See validate_batch() for the fixed,
    # auditable processing order. This runs unconditionally: dry_run,
    # approval_mode=True, and approval_mode=False all pass through here
    # before anything can be approved.
    gcfg = cfg.guardrail_config()

    if cfg.dry_run:
        _run_dry_run_cycle(cfg, gcfg, proposed)
        return text

    # Buy proposals each carry the model's snapshot of current account
    # positions; sells don't need one. Union them into a single baseline —
    # every order in the batch is checked against the same starting state.
    positions: dict[str, float] = {}
    for order in proposed:
        for t, v in dict(order.get("positions", {})).items():
            positions[str(t)] = float(v)

    batch_results = validate_batch(gcfg, proposed, positions)

    # Tracks what has actually been approved-and-executed so far this
    # cycle, as opposed to what validate_batch assumed when it computed
    # the verdicts above. An order that's skipped (approval_mode=True) or
    # never reached (guardrail-rejected) must never be treated as having
    # consumed capacity.
    running_positions = dict(positions)

    for order, result in batch_results:
        ticker = str(order.get("ticker", "")).upper().strip()
        side = order.get("side")
        dollars = float(order.get("dollars", 0))

        if not result.approved:
            print(f"BLOCKED by guardrails: {ticker} {side} — {result.reason}")
            log_event(cfg.log_file, "order_blocked", order=order, reason=result.reason)
            continue

        # 3. APPROVE. Only a guardrail-approved order reaches this point.
        if cfg.approval_mode:
            approved = confirm_with_human(
                f"PROPOSED: {side.upper()} ${dollars:.2f} of {ticker} — "
                f"guardrail check: {result.reason}"
            )
            if not approved:
                print("Skipped.")
                log_event(cfg.log_file, "order_skipped", order=order)
                continue
        else:
            print(
                f"AUTO-APPROVED (approval_mode off): {side.upper()} "
                f"${dollars:.2f} of {ticker} — {result.reason}"
            )
            log_event(cfg.log_file, "auto_approved", order=order, reason=result.reason)

        # 4. EXECUTE. Identical for both approval paths — the only
        # difference between them was who said yes.
        _execute_and_reconcile(cfg, client, system_prompt, order, running_positions)

    return text


if __name__ == "__main__":
    try:
        cfg = AgentConfig()
        report = run_cycle(cfg)
        print("\n===== AGENT REPORT =====\n")
        print(report)
    except EnvironmentError as e:
        sys.exit(str(e))
