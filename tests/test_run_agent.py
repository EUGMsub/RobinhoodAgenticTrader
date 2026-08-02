"""
Unit tests for scripts/run_agent.py.

Nothing here makes a real network call: cfg.validate()'s credential check
is satisfied with fake env vars via monkeypatch, and run_cycle() itself is
monkeypatched out wherever a test needs to control what the "cycle" does,
rather than exercising the real MCP/Anthropic path.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import run_agent


def _set_fake_credentials(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    monkeypatch.setenv("ROBINHOOD_MCP_URL", "https://mcp.example.invalid")
    monkeypatch.setenv("ROBINHOOD_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("ROBINHOOD_ACCOUNT_NUMBER", "123456789")


def _read_log_events(log_file):
    with open(log_file) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestScheduledRequiresDryRun:
    def test_scheduled_without_dry_run_exits_non_zero(self, monkeypatch):
        # No credentials set at all — this must fail before cfg.validate()
        # or AgentConfig() are ever reached, so a missing-env-var error
        # can't be mistaken for the guard actually working.
        monkeypatch.setattr(sys, "argv", ["run_agent.py", "--scheduled"])

        with pytest.raises(SystemExit) as exc_info:
            run_agent.main()

        assert exc_info.value.code != 0
        assert "--dry-run" in str(exc_info.value.code)

    def test_scheduled_with_dry_run_does_not_hit_the_guard(
        self, tmp_path, monkeypatch
    ):
        _set_fake_credentials(monkeypatch)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["run_agent.py", "--scheduled", "--dry-run"])
        monkeypatch.setattr(run_agent, "run_cycle", lambda cfg: "ok")

        run_agent.main()  # must not raise SystemExit from the guard

        events = _read_log_events(tmp_path / "trade_log.jsonl")
        assert any(e["event"] == "cycle_started" for e in events)


class TestCycleStartedLogging:
    def test_cycle_started_is_logged_before_the_cycle_runs(
        self, tmp_path, monkeypatch
    ):
        _set_fake_credentials(monkeypatch)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["run_agent.py", "--dry-run"])
        monkeypatch.setattr(run_agent, "run_cycle", lambda cfg: "ok")

        run_agent.main()

        events = _read_log_events(tmp_path / "trade_log.jsonl")
        started = [e for e in events if e["event"] == "cycle_started"]
        assert len(started) == 1
        assert started[0]["dry_run"] is True
        assert started[0]["scheduled"] is False


class TestCycleFailedLogging:
    def test_exception_during_the_cycle_is_logged_and_still_propagates(
        self, tmp_path, monkeypatch
    ):
        _set_fake_credentials(monkeypatch)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["run_agent.py", "--dry-run"])

        def _explode(cfg):
            raise ValueError("simulated MCP failure")

        monkeypatch.setattr(run_agent, "run_cycle", _explode)

        # "still propagates" — a scheduled failure must exit non-zero via
        # an unhandled exception, not be swallowed into a clean exit.
        with pytest.raises(ValueError, match="simulated MCP failure"):
            run_agent.main()

        events = _read_log_events(tmp_path / "trade_log.jsonl")
        failures = [e for e in events if e["event"] == "cycle_failed"]
        assert len(failures) == 1
        assert failures[0]["exception_type"] == "ValueError"
        assert failures[0]["exception_message"] == "simulated MCP failure"

        # cycle_started must still be there — the failure happened AFTER
        # the cycle was recorded as having begun, not instead of it.
        assert any(e["event"] == "cycle_started" for e in events)

    def test_successful_cycle_never_logs_cycle_failed(self, tmp_path, monkeypatch):
        _set_fake_credentials(monkeypatch)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["run_agent.py", "--dry-run"])
        monkeypatch.setattr(run_agent, "run_cycle", lambda cfg: "all good")

        run_agent.main()

        events = _read_log_events(tmp_path / "trade_log.jsonl")
        assert not any(e["event"] == "cycle_failed" for e in events)
