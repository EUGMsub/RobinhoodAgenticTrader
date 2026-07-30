"""
config.py
=========
Central, explicit configuration. Nothing here is hidden in prompt text —
every number that constrains the agent's behavior lives in one typed object
so it's obvious, reviewable, and testable.
"""

import os
from dataclasses import dataclass, field
from typing import Mapping

from dotenv import load_dotenv

from guardrails import DEFAULT_CORRELATION_GROUPS, GuardrailConfig

# Populate os.environ from .env (gitignored) before AgentConfig's
# default_factory fields read it below. No-op if .env doesn't exist or a
# variable is already set in the real environment (existing env wins).
load_dotenv()


@dataclass(frozen=True)
class AgentConfig:
    # --- strategy parameters ---
    watchlist: tuple[str, ...] = ("VOO", "AAPL", "MSFT")
    dip_trigger_pct: float = 2.0
    revert_target_pct: float = 3.0
    order_dollars: float = 3.0
    max_position_dollars: float = 6.0
    max_total_dollars: float = 10.0
    max_hold_days: int = 10
    disaster_stop_pct: float = 15.0
    correlation_groups: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: DEFAULT_CORRELATION_GROUPS
    )
    max_group_dollars: float = 8.0

    # --- backtest-only parameters (not part of live guardrails) ---
    slippage_pct: float = 0.05
    initial_cash: float = 10.0

    # --- operational parameters ---
    approval_mode: bool = True
    model: str = "claude-sonnet-4-6"
    log_file: str = "trade_log.jsonl"

    # --- API cost accounting ---
    # NOT AUTHORITATIVE. These are operator-supplied estimates used only to
    # attach an approximate dollar figure to logged token counts, so cost
    # can be compared against P&L. They are not fetched from Anthropic and
    # will drift as pricing changes or as you change `model` above.
    # Verify the current rates for your model at
    # docs.claude.com/en/docs/about-claude/pricing and update these
    # yourself. The raw input_tokens/output_tokens in each api_usage event
    # are the ground truth; the dollar figure is derived from these numbers
    # and is only as accurate as they are.
    input_token_price_per_mtok: float = 3.00
    output_token_price_per_mtok: float = 15.00

    # --- dry-run (paper trading) parameters ---
    # dry_run never prompts for approval and never places an order (see
    # agent.py) — it simulates fills against a persisted paper portfolio so
    # successive runs accumulate a track record without touching a real
    # (or real-but-unfunded) account.
    dry_run: bool = False
    paper_portfolio_file: str = "paper_portfolio.json"

    # --- credentials (populated from environment, never hardcoded) ---
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    mcp_url: str = field(default_factory=lambda: os.environ.get("ROBINHOOD_MCP_URL", ""))
    mcp_token: str = field(default_factory=lambda: os.environ.get("ROBINHOOD_MCP_TOKEN", ""))
    agentic_account_number: str = field(
        default_factory=lambda: os.environ.get("ROBINHOOD_ACCOUNT_NUMBER", "")
    )

    def guardrail_config(self) -> GuardrailConfig:
        return GuardrailConfig(
            watchlist=self.watchlist,
            order_dollars=self.order_dollars,
            max_position_dollars=self.max_position_dollars,
            max_total_dollars=self.max_total_dollars,
            dip_trigger_pct=self.dip_trigger_pct,
            revert_target_pct=self.revert_target_pct,
            max_hold_days=self.max_hold_days,
            disaster_stop_pct=self.disaster_stop_pct,
            correlation_groups=self.correlation_groups,
            max_group_dollars=self.max_group_dollars,
        )

    def validate(self) -> None:
        missing = [
            name
            for name, val in [
                ("ANTHROPIC_API_KEY", self.anthropic_api_key),
                ("ROBINHOOD_MCP_URL", self.mcp_url),
                ("ROBINHOOD_MCP_TOKEN", self.mcp_token),
                ("ROBINHOOD_ACCOUNT_NUMBER", self.agentic_account_number),
            ]
            if not val
        ]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "See .env.example."
            )
