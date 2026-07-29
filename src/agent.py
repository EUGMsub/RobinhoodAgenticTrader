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

import anthropic

from config import AgentConfig
from guardrails import validate_buy, validate_sell
from logging_utils import log_event
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
    for order in proposed:
        ticker = str(order.get("ticker", "")).upper()
        side = order.get("side")
        dollars = float(order.get("dollars", 0))

        # Re-validate every proposal in plain code before it's even shown
        # to the human — the model's own claim that a rule is satisfied is
        # never sufficient on its own.
        if side == "buy":
            result = validate_buy(
                gcfg,
                ticker=ticker,
                dollars=dollars,
                day_change_pct=float(order.get("day_change_pct", 0)),
                current_position_value=float(order.get("current_position_value", 0)),
                current_total_exposure=float(order.get("current_total_exposure", 0)),
            )
        elif side == "sell":
            result = validate_sell(
                gcfg,
                ticker=ticker,
                current_price=float(order.get("current_price", 0)),
                avg_cost=float(order.get("avg_cost", 0)),
                days_held=int(order.get("days_held", 0)),
            )
        else:
            result = None

        if result is None or not result.approved:
            reason = result.reason if result else f"unrecognized side '{side}'"
            print(f"BLOCKED by guardrails: {ticker} {side} — {reason}")
            log_event(cfg.log_file, "order_blocked", order=order, reason=reason)
            continue

        if confirm_with_human(
            f"PROPOSED: {side.upper()} ${dollars:.2f} of {ticker} — "
            f"guardrail check: {result.reason}"
        ):
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
            log_event(cfg.log_file, "order_executed", order=order, result=exec_text)
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
