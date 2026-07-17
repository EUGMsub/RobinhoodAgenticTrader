"""
config.py
=========
Central, explicit configuration. Nothing here is hidden in prompt text —
every number that constrains the agent's behavior lives in one typed object
so it's obvious, reviewable, and testable.
"""

import os
from dataclasses import dataclass, field

from guardrails import GuardrailConfig


@dataclass(frozen=True)
class AgentConfig:
    # --- strategy parameters ---
    watchlist: tuple[str, ...] = ("VOO", "AAPL", "MSFT")
    dip_trigger_pct: float = 2.0
    revert_target_pct: float = 3.0
    order_dollars: float = 25.0
    max_position_dollars: float = 150.0
    max_total_dollars: float = 400.0

    # --- operational parameters ---
    approval_mode: bool = True
    model: str = "claude-sonnet-4-6"
    log_file: str = "trade_log.jsonl"

    # --- credentials (populated from environment, never hardcoded) ---
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    mcp_url: str = field(default_factory=lambda: os.environ.get("ROBINHOOD_MCP_URL", ""))
    mcp_token: str = field(default_factory=lambda: os.environ.get("ROBINHOOD_MCP_TOKEN", ""))

    def guardrail_config(self) -> GuardrailConfig:
        return GuardrailConfig(
            watchlist=self.watchlist,
            order_dollars=self.order_dollars,
            max_position_dollars=self.max_position_dollars,
            max_total_dollars=self.max_total_dollars,
            dip_trigger_pct=self.dip_trigger_pct,
            revert_target_pct=self.revert_target_pct,
        )

    def validate(self) -> None:
        missing = [
            name
            for name, val in [
                ("ANTHROPIC_API_KEY", self.anthropic_api_key),
                ("ROBINHOOD_MCP_URL", self.mcp_url),
                ("ROBINHOOD_MCP_TOKEN", self.mcp_token),
            ]
            if not val
        ]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "See .env.example."
            )
