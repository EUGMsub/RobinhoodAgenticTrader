"""
Unit tests for config.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import AgentConfig


class TestCredentialsFromEnvironment:
    def test_agent_config_picks_up_credentials_from_environment(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        monkeypatch.setenv("ROBINHOOD_MCP_URL", "https://mcp.example.com")
        monkeypatch.setenv("ROBINHOOD_CLIENT_ID", "test-client-id")
        monkeypatch.setenv("ROBINHOOD_ACCOUNT_NUMBER", "602437931")

        cfg = AgentConfig()

        assert cfg.anthropic_api_key == "sk-test-key"
        assert cfg.mcp_url == "https://mcp.example.com"
        assert cfg.robinhood_client_id == "test-client-id"
        assert cfg.agentic_account_number == "602437931"

    def test_missing_credentials_default_to_empty_string(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ROBINHOOD_MCP_URL", raising=False)
        monkeypatch.delenv("ROBINHOOD_CLIENT_ID", raising=False)
        monkeypatch.delenv("ROBINHOOD_ACCOUNT_NUMBER", raising=False)

        cfg = AgentConfig()

        assert cfg.anthropic_api_key == ""
        assert cfg.mcp_url == ""
        assert cfg.robinhood_client_id == ""
        assert cfg.agentic_account_number == ""
