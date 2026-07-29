"""
agent.py
========
Orchestration layer: calls the Claude API with the Robinhood Trading MCP
server attached, parses proposed orders, re-validates every proposal
against guardrails.py, gets human approval (if approval_mode is on), and
only then executes.

Nothing here trusts the model's arithmetic. The model is used for what LLMs
are good at (reading unstructured market data and applying a strategy in
natural language); guardrails.py is used for what plain code is good at
(enforcing hard numeric limits it cannot talk itself out of).
"""

import json
import sys
from datetime import datetime, timezone

import anthropic

from config import AgentConfig
from guardrails import validate_batch
from logging_utils import log_event
from reconcile import reconcile_order
from strategy import (
    build_approval_mode_instruction,
    build_execution_mode_instruction,
    build_system_prompt,
)

MCP_BETA_HEADER = "mcp-client-2025-11-25"


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


def _parse_proposed_orders(text: str) -> list[dict]:
    if "```json" not in text:
        return []
    try:
        raw = text.split("```json")[1].split("```")[0]
        return json.loads(raw).get("proposed_orders", [])
    except (json.JSONDecodeError, IndexError):
        return []


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


def run_cycle(cfg: AgentConfig, client: anthropic.Anthropic | None = None) -> str:
    cfg.validate()
    client = client or anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    system_prompt = build_system_prompt(cfg)
    mode_instruction = (
        build_approval_mode_instruction()
        if cfg.approval_mode
        else build_execution_mode_instruction()
    )
    user_msg = (
        "Run one cycle of the strategy now. Check quotes for the watchlist, "
        "check my agentic account positions and buying power, then apply "
        f"the rules. {mode_instruction}"
    )

    response = client.beta.messages.create(
        model=cfg.model,
        max_tokens=4000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
        mcp_servers=_mcp_server_block(cfg),
        betas=[MCP_BETA_HEADER],
    )

    text = _extract_text(response)
    log_event(cfg.log_file, "cycle_report", report=text)

    if not cfg.approval_mode:
        return text

    proposed = _parse_proposed_orders(text)
    if not proposed:
        return text

    gcfg = cfg.guardrail_config()

    # Buy proposals each carry the model's snapshot of current account
    # positions; sells don't need one. Union them into a single baseline —
    # every order in the batch is checked against the same starting state.
    positions: dict[str, float] = {}
    for order in proposed:
        for t, v in dict(order.get("positions", {})).items():
            positions[str(t)] = float(v)

    # Re-validate the WHOLE batch in plain code before any of it is shown
    # to the human — cumulatively, not order-by-order, so that (for
    # example) three watchlist tickers dipping on the same day can't each
    # independently clear the group cap and collectively overshoot it. See
    # validate_batch() for the fixed, auditable processing order.
    batch_results = validate_batch(gcfg, proposed, positions)

    # Tracks what has actually been approved-and-executed so far this
    # cycle, as opposed to what validate_batch assumed when it computed
    # the verdicts above. An order the human skips must never be treated
    # as having consumed capacity.
    running_positions = dict(positions)

    for order, result in batch_results:
        ticker = str(order.get("ticker", "")).upper().strip()
        side = order.get("side")
        dollars = float(order.get("dollars", 0))

        if not result.approved:
            print(f"BLOCKED by guardrails: {ticker} {side} — {result.reason}")
            log_event(cfg.log_file, "order_blocked", order=order, reason=result.reason)
            continue

        if confirm_with_human(
            f"PROPOSED: {side.upper()} ${dollars:.2f} of {ticker} — "
            f"guardrail check: {result.reason}"
        ):
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
                betas=[MCP_BETA_HEADER],
            )
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
                betas=[MCP_BETA_HEADER],
            )
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
        else:
            print("Skipped.")
            log_event(cfg.log_file, "order_skipped", order=order)

    return text


if __name__ == "__main__":
    try:
        cfg = AgentConfig()
        report = run_cycle(cfg)
        print("\n===== AGENT REPORT =====\n")
        print(report)
    except EnvironmentError as e:
        sys.exit(str(e))
