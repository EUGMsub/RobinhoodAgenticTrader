"""
Unit tests for agent.py.

Two things are asserted throughout, with a fake Anthropic client and no
real credentials:

1. dry_run=True can NEVER reach the order-placing code path: not the
   execution MCP call, not the reconciliation MCP call, not even the human
   confirmation prompt. _execute_and_reconcile() — the only place an
   order-placing MCP call happens — raises RuntimeError immediately if
   dry_run is True, before touching the client.
2. approval_mode=False (no human in the loop) still goes through
   validate_batch() exactly like approval_mode=True does — it only skips
   the confirm_with_human() prompt, never the guardrail check. There is no
   "execute directly" path that bypasses validation.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import (
    _execute_and_reconcile,
    _recompute_snapshot,
    _session_prices_diverge,
    _snapshot_diverges_from_model,
    run_cycle,
)
from config import AgentConfig
from paper_trading import load_paper_portfolio


_UNSET = object()


@pytest.fixture(autouse=True)
def _fake_access_token(monkeypatch):
    """Every test in this file goes through agent._mcp_server_block(),
    which now calls oauth.get_valid_access_token() instead of reading a
    static config field. None of these tests are about OAuth — stub it out
    globally so they stay hermetic (no real token file, no network) and
    keep asserting what they were already asserting."""
    import agent as agent_module

    monkeypatch.setattr(agent_module, "get_valid_access_token", lambda client_id: "fake-access-token")


class _FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens, output_tokens, **cache_fields):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        for name, value in cache_fields.items():
            setattr(self, name, value)


class _FakeResponse:
    def __init__(self, text, usage=None):
        self.content = [_FakeBlock(text)]
        if usage is not None:
            self.usage = usage


class _FakeMessages:
    """Stands in for client.beta.messages. Each call pops the next canned
    response; calling past the end of `responses` raises immediately,
    which is exactly what we want if run_cycle ever makes a call it
    shouldn't in dry_run mode."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError(
                "unexpected extra call to client.beta.messages.create "
                f"(prompt: {kwargs.get('messages')})"
            )
        return self._responses.pop(0)


class _FakeBeta:
    def __init__(self, messages: _FakeMessages):
        self.messages = messages


class _FakeClient:
    def __init__(self, responses):
        self.beta = _FakeBeta(_FakeMessages(responses))


def _cfg(tmp_path, **overrides):
    defaults = dict(
        watchlist=("AAPL",),
        dip_trigger_pct=2.0,
        revert_target_pct=3.0,
        order_dollars=25.0,
        max_position_dollars=150.0,
        max_total_dollars=400.0,
        correlation_groups={},
        max_group_dollars=1_000_000.0,
        initial_cash=400.0,
        log_file=str(tmp_path / "trade_log.jsonl"),
        paper_portfolio_file=str(tmp_path / "paper_portfolio.json"),
        anthropic_api_key="test-key",
        mcp_url="https://example.invalid/mcp",
        robinhood_client_id="test-client-id",
        agentic_account_number="123456789",
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


def _snapshot_entry_for(current_price, day_change_pct):
    """Build raw market_snapshot fields (last_trade_price,
    adjusted_previous_close, ...) that agent._recompute_snapshot()
    reproduces exactly this current_price/day_change_pct from — so test
    fixtures state the number they care about instead of hand-solving the
    reverse algebra. Extended-hours fields are left null; tests that care
    about session divergence set them explicitly."""
    adjusted_previous_close = current_price / (1 + day_change_pct / 100)
    return {
        "last_trade_price": current_price,
        "last_trade_price_timestamp": "2026-01-01T20:00:00Z",
        "last_non_reg_trade_price": None,
        "last_non_reg_trade_price_timestamp": None,
        "adjusted_previous_close": adjusted_previous_close,
        "day_change_pct": day_change_pct,
    }


def _cycle_report(proposed_orders, market_snapshot=_UNSET):
    payload = {"proposed_orders": proposed_orders}
    if market_snapshot is _UNSET:
        # Default: derive one raw snapshot entry per distinct ticker
        # directly from proposed_orders' own current_price/day_change_pct,
        # so agent.py's code-side recomputation reproduces the exact
        # number these orders were already written to test against —
        # existing tests don't have to know recomputation exists. Falls
        # back to a fixed AAPL entry when there's nothing to derive from
        # (e.g. proposed_orders is empty).
        market_snapshot = {}
        for order in proposed_orders:
            ticker = order.get("ticker")
            current_price = order.get("current_price")
            day_change_pct = order.get("day_change_pct")
            if (
                ticker
                and ticker not in market_snapshot
                and isinstance(current_price, (int, float))
                and isinstance(day_change_pct, (int, float))
            ):
                market_snapshot[ticker] = _snapshot_entry_for(current_price, day_change_pct)
        if not market_snapshot:
            market_snapshot = {"AAPL": _snapshot_entry_for(100.0, -5.0)}
    if market_snapshot is not None:
        payload["market_snapshot"] = market_snapshot
    return f"Here's what I found.\n```json\n{json.dumps(payload)}\n```\n"


def _read_log_events(log_file):
    with open(log_file) as f:
        return [json.loads(line) for line in f if line.strip()]


def _reconcile_report(equity_orders, option_orders):
    payload = {"equity_orders": equity_orders, "option_orders": option_orders}
    return f"```json\n{json.dumps(payload)}\n```\n"


def _enabled_tools(tools_kwarg):
    """Extract the set of tool names actually enabled by a `tools=`
    kwarg built by agent._mcp_toolset(), asserting it's shaped as an
    allowlist (default_config disables everything, configs re-enables
    only the named tools) rather than trusting it loosely."""
    assert len(tools_kwarg) == 1
    toolset = tools_kwarg[0]
    assert toolset["type"] == "mcp_toolset"
    assert toolset["default_config"] == {"enabled": False}
    assert all(cfg == {"enabled": True} for cfg in toolset["configs"].values())
    return set(toolset["configs"].keys())


class TestDryRunNeverExecutes:
    def test_approved_buy_never_triggers_a_second_llm_call(self, tmp_path):
        cfg = _cfg(tmp_path, dry_run=True)
        report = _cycle_report(
            [
                {
                    "ticker": "AAPL",
                    "side": "buy",
                    "dollars": 25.0,
                    "day_change_pct": -5.0,
                    "current_price": 100.0,
                    "positions": {},
                    "reason": "down 5%",
                }
            ]
        )
        client = _FakeClient([_FakeResponse(report)])

        run_cycle(cfg, client=client)

        # Only the initial cycle-report call happened — no execution call,
        # no reconciliation call, regardless of the order being approved.
        assert len(client.beta.messages.calls) == 1

    def test_confirm_with_human_is_never_called(self, tmp_path, monkeypatch):
        def _explode(prompt):
            raise AssertionError("confirm_with_human must never be called in dry_run mode")

        import agent as agent_module

        monkeypatch.setattr(agent_module, "confirm_with_human", _explode)

        cfg = _cfg(tmp_path, dry_run=True)
        report = _cycle_report(
            [
                {
                    "ticker": "AAPL",
                    "side": "buy",
                    "dollars": 25.0,
                    "day_change_pct": -5.0,
                    "current_price": 100.0,
                    "positions": {},
                    "reason": "down 5%",
                }
            ]
        )
        client = _FakeClient([_FakeResponse(report)])

        run_cycle(cfg, client=client)  # would raise via _explode if reached

    def test_execution_mode_instruction_is_never_used_in_dry_run(self, tmp_path):
        # dry_run=True with approval_mode=False must still get the JSON
        # proposal instruction, never "execute qualifying orders directly".
        cfg = _cfg(tmp_path, dry_run=True, approval_mode=False)
        report = _cycle_report([])
        client = _FakeClient([_FakeResponse(report)])

        run_cycle(cfg, client=client)

        sent_prompt = client.beta.messages.calls[0]["messages"][0]["content"]
        assert "execute qualifying orders directly" not in sent_prompt


class TestExecutionGuard:
    def test_execute_and_reconcile_raises_when_dry_run_is_true(self, tmp_path):
        # Calls the guarded function directly — not through run_cycle, not
        # through confirm_with_human — so the guard itself is exercised
        # regardless of how the surrounding code routes to it. A plain
        # `assert` would vanish under `python -O`; RuntimeError does not.
        cfg = _cfg(tmp_path, dry_run=True)
        order = {"ticker": "AAPL", "side": "buy", "dollars": 25.0}

        with pytest.raises(RuntimeError, match="dry_run=True"):
            _execute_and_reconcile(
                cfg,
                client=None,  # must never be touched — the raise comes first
                system_prompt="irrelevant",
                order=order,
                running_positions={},
            )

    def test_execute_and_reconcile_does_not_raise_when_dry_run_is_false(self, tmp_path):
        # Sanity check on the other side of the guard: a live cfg must
        # reach the client, not the RuntimeError. It'll fail on the fake
        # client's canned-response exhaustion instead, which proves the
        # guard let it through.
        cfg = _cfg(tmp_path, dry_run=False)
        order = {"ticker": "AAPL", "side": "buy", "dollars": 25.0}
        client = _FakeClient([])  # no responses queued

        with pytest.raises(AssertionError, match="unexpected extra call"):
            _execute_and_reconcile(
                cfg,
                client=client,
                system_prompt="irrelevant",
                order=order,
                running_positions={},
            )


class TestDryRunSimulation:
    def test_approved_buy_updates_paper_portfolio_and_logs_simulated_fill(self, tmp_path):
        cfg = _cfg(tmp_path, dry_run=True)
        report = _cycle_report(
            [
                {
                    "ticker": "AAPL",
                    "side": "buy",
                    "dollars": 25.0,
                    "day_change_pct": -5.0,
                    "current_price": 100.0,
                    "positions": {},
                    "reason": "AAPL down 5%",
                }
            ]
        )
        client = _FakeClient([_FakeResponse(report)])

        run_cycle(cfg, client=client)

        portfolio = load_paper_portfolio(cfg.paper_portfolio_file, cfg.initial_cash)
        assert portfolio.cash == 375.0
        assert portfolio.positions["AAPL"].shares == 0.25
        assert portfolio.positions["AAPL"].avg_cost == 100.0

        events = _read_log_events(cfg.log_file)
        fills = [e for e in events if e["event"] == "simulated_fill"]
        assert len(fills) == 1
        assert fills[0]["ticker"] == "AAPL"
        assert fills[0]["side"] == "buy"
        assert fills[0]["dollars"] == 25.0
        assert fills[0]["simulated_price"] == 100.0
        # The GUARDRAIL's reason is logged, not the model-proposed one —
        # the whole point is re-deriving the decision, not trusting the
        # model's own stated justification.
        assert "AAPL down -5.00%" in fills[0]["reason"]
        assert "within position, group, and account caps" in fills[0]["reason"]

    def test_rejected_order_is_blocked_not_simulated(self, tmp_path):
        cfg = _cfg(tmp_path, dry_run=True)
        report = _cycle_report(
            [
                {
                    "ticker": "AAPL",
                    "side": "buy",
                    "dollars": 25.0,
                    "day_change_pct": -0.5,  # below the 2% trigger
                    "current_price": 100.0,
                    "positions": {},
                    "reason": "barely down",
                }
            ]
        )
        client = _FakeClient([_FakeResponse(report)])

        run_cycle(cfg, client=client)

        portfolio = load_paper_portfolio(cfg.paper_portfolio_file, cfg.initial_cash)
        assert portfolio.cash == 400.0
        assert portfolio.positions == {}

        events = _read_log_events(cfg.log_file)
        assert any(e["event"] == "order_blocked" for e in events)
        assert not any(e["event"] == "simulated_fill" for e in events)

    def test_successive_dry_run_cycles_accumulate_the_paper_portfolio(self, tmp_path):
        cfg = _cfg(tmp_path, dry_run=True)
        order = {
            "ticker": "AAPL",
            "side": "buy",
            "dollars": 25.0,
            "day_change_pct": -5.0,
            "current_price": 100.0,
            "positions": {},
            "reason": "down 5%",
        }

        client1 = _FakeClient([_FakeResponse(_cycle_report([order]))])
        run_cycle(cfg, client=client1)

        client2 = _FakeClient([_FakeResponse(_cycle_report([order]))])
        run_cycle(cfg, client=client2)

        portfolio = load_paper_portfolio(cfg.paper_portfolio_file, cfg.initial_cash)
        assert portfolio.cash == 350.0
        assert portfolio.positions["AAPL"].shares == pytest.approx(0.5)


class TestLiveModeUnaffected:
    def test_live_approval_mode_still_prompts_and_skips_cleanly(self, tmp_path, monkeypatch):
        # A regression check on the shared mode_instruction/early-return
        # logic: normal (non-dry-run) approval mode must still behave as
        # before — human is prompted, and a skip makes no further calls.
        import agent as agent_module

        monkeypatch.setattr(agent_module, "confirm_with_human", lambda prompt: False)

        cfg = _cfg(tmp_path, dry_run=False, approval_mode=True)
        report = _cycle_report(
            [
                {
                    "ticker": "AAPL",
                    "side": "buy",
                    "dollars": 25.0,
                    "day_change_pct": -5.0,
                    "current_price": 100.0,
                    "positions": {},
                    "reason": "down 5%",
                }
            ]
        )
        client = _FakeClient([_FakeResponse(report)])

        run_cycle(cfg, client=client)

        assert len(client.beta.messages.calls) == 1  # skipped, so no execution call
        events = _read_log_events(cfg.log_file)
        assert any(e["event"] == "order_skipped" for e in events)


class TestMcpToolRestriction:
    """Tool restriction is enforced by the request itself (an MCPToolset
    allowlist), not just by prompt wording — these tests inspect the
    actual `tools=` kwarg sent to client.beta.messages.create() rather
    than trusting behavior inferred from prompt text."""

    def test_dry_run_proposal_call_has_no_order_placing_tool(self, tmp_path):
        cfg = _cfg(tmp_path, dry_run=True)
        client = _FakeClient([_FakeResponse(_cycle_report([]))])

        run_cycle(cfg, client=client)

        enabled = _enabled_tools(client.beta.messages.calls[0]["tools"])
        assert "place_equity_order" not in enabled
        assert "place_option_order" not in enabled

    def test_live_approval_mode_proposal_call_has_no_order_placing_tool(
        self, tmp_path, monkeypatch
    ):
        import agent as agent_module

        monkeypatch.setattr(agent_module, "confirm_with_human", lambda prompt: False)

        cfg = _cfg(tmp_path, dry_run=False, approval_mode=True)
        client = _FakeClient([_FakeResponse(_cycle_report([]))])

        run_cycle(cfg, client=client)

        enabled = _enabled_tools(client.beta.messages.calls[0]["tools"])
        assert "place_equity_order" not in enabled
        assert "place_option_order" not in enabled

    def test_execution_call_is_the_only_one_with_place_equity_order(
        self, tmp_path, monkeypatch
    ):
        import agent as agent_module

        monkeypatch.setattr(agent_module, "confirm_with_human", lambda prompt: True)

        cfg = _cfg(tmp_path, dry_run=False, approval_mode=True)
        order = {
            "ticker": "AAPL",
            "side": "buy",
            "dollars": 25.0,
            "day_change_pct": -5.0,
            "current_price": 100.0,
            "positions": {},
            "reason": "down 5%",
        }
        client = _FakeClient(
            [
                _FakeResponse(_cycle_report([order])),
                _FakeResponse("Order placed."),
                _FakeResponse(
                    _reconcile_report(
                        [{"symbol": "AAPL", "side": "buy", "dollar_amount": 25.0}], []
                    )
                ),
            ]
        )

        run_cycle(cfg, client=client)

        proposal_call, execution_call, reconcile_call = client.beta.messages.calls

        assert "place_equity_order" not in _enabled_tools(proposal_call["tools"])
        assert "place_equity_order" in _enabled_tools(execution_call["tools"])
        assert "place_equity_order" not in _enabled_tools(reconcile_call["tools"])

    def test_reconciliation_call_retains_get_option_orders(self, tmp_path, monkeypatch):
        # get_option_orders exists precisely to catch an option order that
        # shouldn't be there — reconciliation must keep access to it even
        # though it never places anything.
        import agent as agent_module

        monkeypatch.setattr(agent_module, "confirm_with_human", lambda prompt: True)

        cfg = _cfg(tmp_path, dry_run=False, approval_mode=True)
        order = {
            "ticker": "AAPL",
            "side": "buy",
            "dollars": 25.0,
            "day_change_pct": -5.0,
            "current_price": 100.0,
            "positions": {},
            "reason": "down 5%",
        }
        client = _FakeClient(
            [
                _FakeResponse(_cycle_report([order])),
                _FakeResponse("Order placed."),
                _FakeResponse(
                    _reconcile_report(
                        [{"symbol": "AAPL", "side": "buy", "dollar_amount": 25.0}], []
                    )
                ),
            ]
        )

        run_cycle(cfg, client=client)

        reconcile_call = client.beta.messages.calls[2]
        enabled = _enabled_tools(reconcile_call["tools"])

        assert enabled == {"get_equity_orders", "get_option_orders"}
        assert "place_equity_order" not in enabled
        assert "place_option_order" not in enabled


def _approved_buy_order():
    return {
        "ticker": "AAPL",
        "side": "buy",
        "dollars": 25.0,
        "day_change_pct": -5.0,
        "current_price": 100.0,
        "positions": {},
        "reason": "down 5%",
    }


def _rejected_buy_order():
    # -0.5% is nowhere near the 2% dip trigger — validate_buy rejects it.
    return {
        "ticker": "AAPL",
        "side": "buy",
        "dollars": 25.0,
        "day_change_pct": -0.5,
        "current_price": 100.0,
        "positions": {},
        "reason": "barely down",
    }


class TestAutoApproveNoHumanInLoop:
    """approval_mode=False must still run every order through
    validate_batch() — it only removes the confirm_with_human() prompt,
    never the guardrail check itself."""

    def test_validate_batch_is_called(self, tmp_path, monkeypatch):
        import agent as agent_module
        from guardrails import validate_batch as real_validate_batch

        calls = []

        def _spy(cfg, orders, positions):
            calls.append((orders, positions))
            return real_validate_batch(cfg, orders, positions)

        monkeypatch.setattr(agent_module, "validate_batch", _spy)

        cfg = _cfg(tmp_path, approval_mode=False, dry_run=False)
        client = _FakeClient(
            [
                _FakeResponse(_cycle_report([_approved_buy_order()])),
                _FakeResponse("Order placed."),
                _FakeResponse(
                    _reconcile_report(
                        [{"symbol": "AAPL", "side": "buy", "dollar_amount": 25.0}], []
                    )
                ),
            ]
        )

        run_cycle(cfg, client=client)

        assert len(calls) == 1
        (orders, _positions) = calls[0]
        assert orders[0]["ticker"] == "AAPL"

    def test_confirm_with_human_is_never_called(self, tmp_path, monkeypatch):
        import agent as agent_module

        def _explode(prompt):
            raise AssertionError(
                "confirm_with_human must never be called when approval_mode=False"
            )

        monkeypatch.setattr(agent_module, "confirm_with_human", _explode)

        cfg = _cfg(tmp_path, approval_mode=False, dry_run=False)
        client = _FakeClient(
            [
                _FakeResponse(_cycle_report([_approved_buy_order()])),
                _FakeResponse("Order placed."),
                _FakeResponse(
                    _reconcile_report(
                        [{"symbol": "AAPL", "side": "buy", "dollar_amount": 25.0}], []
                    )
                ),
            ]
        )

        run_cycle(cfg, client=client)  # would raise via _explode if reached

    def test_guardrail_rejected_order_is_never_executed(self, tmp_path, monkeypatch):
        import agent as agent_module

        monkeypatch.setattr(
            agent_module,
            "confirm_with_human",
            lambda prompt: (_ for _ in ()).throw(
                AssertionError("should never be reached — order is rejected")
            ),
        )

        cfg = _cfg(tmp_path, approval_mode=False, dry_run=False)
        client = _FakeClient([_FakeResponse(_cycle_report([_rejected_buy_order()]))])

        run_cycle(cfg, client=client)

        # Only the proposal call happened — no execution, no reconciliation,
        # even though there was no human to say no.
        assert len(client.beta.messages.calls) == 1

        events = _read_log_events(cfg.log_file)
        assert any(e["event"] == "order_blocked" for e in events)
        assert not any(e["event"] == "auto_approved" for e in events)
        assert not any(e["event"] == "order_executed" for e in events)

    def test_approved_order_is_logged_auto_approved_then_executed(self, tmp_path, monkeypatch):
        import agent as agent_module

        monkeypatch.setattr(agent_module, "confirm_with_human", lambda prompt: True)

        cfg = _cfg(tmp_path, approval_mode=False, dry_run=False)
        client = _FakeClient(
            [
                _FakeResponse(_cycle_report([_approved_buy_order()])),
                _FakeResponse("Order placed."),
                _FakeResponse(
                    _reconcile_report(
                        [{"symbol": "AAPL", "side": "buy", "dollar_amount": 25.0}], []
                    )
                ),
            ]
        )

        run_cycle(cfg, client=client)

        assert len(client.beta.messages.calls) == 3  # proposal, execute, reconcile
        events = _read_log_events(cfg.log_file)
        assert any(e["event"] == "auto_approved" for e in events)
        assert any(e["event"] == "order_executed" for e in events)


class TestNoExecuteDirectlyInstructionAnywhere:
    def test_build_execution_mode_instruction_was_deleted_from_strategy(self):
        import strategy

        assert not hasattr(strategy, "build_execution_mode_instruction")

    def test_approval_mode_true_proposal_call_has_no_execute_directly_phrasing(
        self, tmp_path
    ):
        cfg = _cfg(tmp_path, approval_mode=True, dry_run=False)
        client = _FakeClient([_FakeResponse(_cycle_report([]))])

        run_cycle(cfg, client=client)

        prompt = client.beta.messages.calls[0]["messages"][0]["content"]
        assert "execute qualifying orders directly" not in prompt.lower()

    def test_approval_mode_false_proposal_call_has_no_execute_directly_phrasing(
        self, tmp_path
    ):
        # This is the exact path that used to bypass validate_batch()
        # entirely and send an execute-directly instruction instead.
        cfg = _cfg(tmp_path, approval_mode=False, dry_run=False)
        client = _FakeClient([_FakeResponse(_cycle_report([_rejected_buy_order()]))])

        run_cycle(cfg, client=client)

        prompt = client.beta.messages.calls[0]["messages"][0]["content"]
        assert "execute qualifying orders directly" not in prompt.lower()

    def test_dry_run_proposal_call_has_no_execute_directly_phrasing(self, tmp_path):
        cfg = _cfg(tmp_path, approval_mode=False, dry_run=True)
        client = _FakeClient([_FakeResponse(_cycle_report([]))])

        run_cycle(cfg, client=client)

        prompt = client.beta.messages.calls[0]["messages"][0]["content"]
        assert "execute qualifying orders directly" not in prompt.lower()


class TestMarketSnapshotLogging:
    """A no-trade day and a broken-tool day both produce zero orders. The
    snapshot event is what tells them apart after the fact."""

    def test_snapshot_is_logged_on_a_cycle_with_no_proposals(self, tmp_path):
        cfg = _cfg(tmp_path, dry_run=True)
        snapshot = {"AAPL": _snapshot_entry_for(187.42, -1.13)}
        client = _FakeClient([_FakeResponse(_cycle_report([], market_snapshot=snapshot))])

        run_cycle(cfg, client=client)

        events = _read_log_events(cfg.log_file)
        snapshots = [e for e in events if e["event"] == "market_snapshot"]
        assert len(snapshots) == 1
        assert snapshots[0]["snapshot"]["AAPL"]["current_price"] == 187.42
        assert snapshots[0]["snapshot"]["AAPL"]["day_change_pct"] == pytest.approx(-1.13)
        # A complete snapshot must NOT also raise the incomplete signal.
        assert not any(e["event"] == "market_snapshot_incomplete" for e in events)

    def test_missing_ticker_produces_the_incomplete_event(self, tmp_path):
        # Watchlist is AAPL + MSFT; the model only reported AAPL.
        cfg = _cfg(tmp_path, watchlist=("AAPL", "MSFT"), dry_run=True)
        snapshot = {"AAPL": _snapshot_entry_for(187.42, -1.13)}
        client = _FakeClient([_FakeResponse(_cycle_report([], market_snapshot=snapshot))])

        run_cycle(cfg, client=client)

        events = _read_log_events(cfg.log_file)
        incomplete = [e for e in events if e["event"] == "market_snapshot_incomplete"]
        assert len(incomplete) == 1
        assert incomplete[0]["missing_tickers"] == ["MSFT"]
        # The partial snapshot is still logged — partial data beats none.
        assert any(e["event"] == "market_snapshot" for e in events)

    def test_absent_snapshot_object_produces_the_incomplete_event(self, tmp_path):
        cfg = _cfg(tmp_path, dry_run=True)
        client = _FakeClient([_FakeResponse(_cycle_report([], market_snapshot=None))])

        run_cycle(cfg, client=client)

        events = _read_log_events(cfg.log_file)
        incomplete = [e for e in events if e["event"] == "market_snapshot_incomplete"]
        assert len(incomplete) == 1
        assert incomplete[0]["missing_tickers"] == ["AAPL"]
        assert not any(e["event"] == "market_snapshot" for e in events)

    def test_snapshot_is_logged_even_when_orders_are_proposed(self, tmp_path):
        cfg = _cfg(tmp_path, dry_run=True)
        client = _FakeClient([_FakeResponse(_cycle_report([_approved_buy_order()]))])

        run_cycle(cfg, client=client)

        events = _read_log_events(cfg.log_file)
        assert any(e["event"] == "market_snapshot" for e in events)


class TestApiUsageLogging:
    def test_usage_is_logged_with_correct_cost_arithmetic(self, tmp_path):
        cfg = _cfg(
            tmp_path,
            dry_run=True,
            input_token_price_per_mtok=3.0,
            output_token_price_per_mtok=15.0,
        )
        usage = _FakeUsage(input_tokens=1_000_000, output_tokens=200_000)
        client = _FakeClient([_FakeResponse(_cycle_report([]), usage=usage)])

        run_cycle(cfg, client=client)

        events = _read_log_events(cfg.log_file)
        usages = [e for e in events if e["event"] == "api_usage"]
        assert len(usages) == 1
        entry = usages[0]
        assert entry["call"] == "proposal"
        assert entry["input_tokens"] == 1_000_000
        assert entry["output_tokens"] == 200_000
        # 1.0 Mtok in @ $3 + 0.2 Mtok out @ $15 = 3.00 + 3.00 = 6.00
        assert entry["estimated_cost_usd"] == pytest.approx(6.0)

    def test_cache_token_fields_are_captured_when_present(self, tmp_path):
        cfg = _cfg(tmp_path, dry_run=True)
        usage = _FakeUsage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=2048,
            cache_read_input_tokens=4096,
        )
        client = _FakeClient([_FakeResponse(_cycle_report([]), usage=usage)])

        run_cycle(cfg, client=client)

        entry = [e for e in _read_log_events(cfg.log_file) if e["event"] == "api_usage"][0]
        assert entry["cache_creation_input_tokens"] == 2048
        assert entry["cache_read_input_tokens"] == 4096

    def test_cache_fields_are_omitted_when_absent_rather_than_zeroed(self, tmp_path):
        # Reporting 0 for a call that never used caching would misrepresent
        # it as "cached nothing" instead of "caching not in play".
        cfg = _cfg(tmp_path, dry_run=True)
        usage = _FakeUsage(input_tokens=100, output_tokens=50)
        client = _FakeClient([_FakeResponse(_cycle_report([]), usage=usage)])

        run_cycle(cfg, client=client)

        entry = [e for e in _read_log_events(cfg.log_file) if e["event"] == "api_usage"][0]
        assert "cache_read_input_tokens" not in entry
        assert "cache_creation_input_tokens" not in entry

    def test_each_call_is_attributed_separately(self, tmp_path, monkeypatch):
        import agent as agent_module

        monkeypatch.setattr(agent_module, "confirm_with_human", lambda prompt: True)

        cfg = _cfg(tmp_path, dry_run=False, approval_mode=True)
        client = _FakeClient(
            [
                _FakeResponse(
                    _cycle_report([_approved_buy_order()]),
                    usage=_FakeUsage(input_tokens=1000, output_tokens=100),
                ),
                _FakeResponse(
                    "Order placed.",
                    usage=_FakeUsage(input_tokens=2000, output_tokens=200),
                ),
                _FakeResponse(
                    _reconcile_report(
                        [{"symbol": "AAPL", "side": "buy", "dollar_amount": 25.0}], []
                    ),
                    usage=_FakeUsage(input_tokens=3000, output_tokens=300),
                ),
            ]
        )

        run_cycle(cfg, client=client)

        usages = [e for e in _read_log_events(cfg.log_file) if e["event"] == "api_usage"]
        assert [u["call"] for u in usages] == ["proposal", "execution", "reconciliation"]
        assert [u["input_tokens"] for u in usages] == [1000, 2000, 3000]

    def test_missing_usage_object_is_tolerated(self, tmp_path):
        # An SDK/mock without .usage must not crash a live trading cycle.
        cfg = _cfg(tmp_path, dry_run=True)
        client = _FakeClient([_FakeResponse(_cycle_report([]))])  # no usage

        run_cycle(cfg, client=client)

        events = _read_log_events(cfg.log_file)
        assert not any(e["event"] == "api_usage" for e in events)


class TestRecomputeSnapshotPureLogic:
    """_recompute_snapshot() is pure — no I/O, no client — so it's tested
    directly against hand-computed values rather than through a full
    run_cycle()."""

    def test_regular_session_recomputation_matches_a_hand_computed_value(self, tmp_path):
        cfg = _cfg(tmp_path)  # price_session defaults to "regular"
        raw = {
            "AAPL": {
                "last_trade_price": 200.0,
                "last_non_reg_trade_price": None,
                "adjusted_previous_close": 185.0,
            }
        }

        recomputed = _recompute_snapshot(cfg, raw)

        # Hand-computed: (200 - 185) / 185 * 100 = 8.108108108108108...
        assert recomputed["AAPL"]["current_price"] == 200.0
        assert recomputed["AAPL"]["day_change_pct"] == pytest.approx(8.108108108108108)

    def test_extended_session_uses_the_extended_price(self, tmp_path):
        cfg = _cfg(tmp_path, price_session="extended")
        raw = {
            "AAPL": {
                "last_trade_price": 200.0,
                "last_non_reg_trade_price": 184.0,
                "adjusted_previous_close": 200.0,
            }
        }

        recomputed = _recompute_snapshot(cfg, raw)

        assert recomputed["AAPL"]["current_price"] == 184.0
        assert recomputed["AAPL"]["day_change_pct"] == pytest.approx(-8.0)

    def test_extended_session_falls_back_to_regular_when_no_extended_trade(self, tmp_path):
        cfg = _cfg(tmp_path, price_session="extended")
        raw = {
            "AAPL": {
                "last_trade_price": 200.0,
                "last_non_reg_trade_price": None,
                "adjusted_previous_close": 185.0,
            }
        }

        recomputed = _recompute_snapshot(cfg, raw)

        assert recomputed["AAPL"]["current_price"] == 200.0

    def test_ticker_missing_adjusted_previous_close_is_dropped_not_guessed(self, tmp_path):
        cfg = _cfg(tmp_path)
        raw = {"AAPL": {"last_trade_price": 200.0, "last_non_reg_trade_price": None}}

        recomputed = _recompute_snapshot(cfg, raw)

        assert "AAPL" not in recomputed


class TestSnapshotDivergenceHelpers:
    def test_within_tolerance_does_not_diverge(self):
        entry = {"day_change_pct": 8.0, "model_day_change_pct": 8.03}
        assert _snapshot_diverges_from_model(entry) is False

    def test_beyond_tolerance_diverges(self):
        entry = {"day_change_pct": 8.0, "model_day_change_pct": 8.06}
        assert _snapshot_diverges_from_model(entry) is True

    def test_no_model_value_never_diverges(self):
        entry = {"day_change_pct": 8.0}
        assert _snapshot_diverges_from_model(entry) is False

    def test_session_prices_at_the_threshold_do_not_diverge(self):
        entry = {"last_trade_price": 200.0, "last_non_reg_trade_price": 202.0}  # exactly 1%
        assert _session_prices_diverge(entry) is False

    def test_session_prices_beyond_the_threshold_diverge(self):
        entry = {"last_trade_price": 200.0, "last_non_reg_trade_price": 202.5}  # 1.25%
        assert _session_prices_diverge(entry) is True

    def test_missing_extended_price_never_diverges(self):
        entry = {"last_trade_price": 200.0, "last_non_reg_trade_price": None}
        assert _session_prices_diverge(entry) is False


class TestRecomputedDayChangeOverridesModel:
    """The actual false-negative fix: whatever the model puts in an
    order's day_change_pct is not what guardrails see — the code-recomputed
    value from the raw quote fields is."""

    def test_model_understating_a_dip_is_corrected_and_approved(self, tmp_path):
        # The model reports a barely-down order (day_change_pct=-0.5, would
        # be rejected on its own), but the raw quote fields recompute (for
        # the default "regular" session) to a real -8% dip. The recomputed
        # value must win: the order should be APPROVED.
        cfg = _cfg(tmp_path, dry_run=True)
        snapshot = {
            "AAPL": {
                "last_trade_price": 92.0,
                "last_trade_price_timestamp": "2026-01-01T20:00:00Z",
                "last_non_reg_trade_price": None,
                "last_non_reg_trade_price_timestamp": None,
                "adjusted_previous_close": 100.0,  # (92-100)/100*100 = -8.0%
                "day_change_pct": -0.5,  # the model's own, wrong, claim
            }
        }
        order = {
            "ticker": "AAPL",
            "side": "buy",
            "dollars": 25.0,
            "day_change_pct": -0.5,
            "current_price": 92.0,
            "positions": {},
            "reason": "barely down (model's own, incorrect, claim)",
        }
        client = _FakeClient([_FakeResponse(_cycle_report([order], market_snapshot=snapshot))])

        run_cycle(cfg, client=client)

        events = _read_log_events(cfg.log_file)
        fills = [e for e in events if e["event"] == "simulated_fill"]
        assert len(fills) == 1, "recomputed -8% dip should have cleared the -2% trigger"
        assert "AAPL down -8.00%" in fills[0]["reason"]

        recompute_events = [e for e in events if e["event"] == "snapshot_recomputed"]
        assert len(recompute_events) == 1
        assert recompute_events[0]["model_day_change_pct"] == -0.5
        assert recompute_events[0]["recomputed_day_change_pct"] == pytest.approx(-8.0)

    def test_model_overclaiming_a_dip_is_corrected_and_rejected(self, tmp_path):
        # Inverse case: the model claims a qualifying -5% dip, but the raw
        # fields recompute to only -0.5% — nowhere near the trigger. The
        # order must be BLOCKED, not approved on the model's say-so.
        cfg = _cfg(tmp_path, dry_run=True)
        snapshot = {
            "AAPL": {
                "last_trade_price": 99.5,
                "last_trade_price_timestamp": "2026-01-01T20:00:00Z",
                "last_non_reg_trade_price": None,
                "last_non_reg_trade_price_timestamp": None,
                "adjusted_previous_close": 100.0,  # (99.5-100)/100*100 = -0.5%
                "day_change_pct": -5.0,  # the model's own, wrong, claim
            }
        }
        order = {
            "ticker": "AAPL",
            "side": "buy",
            "dollars": 25.0,
            "day_change_pct": -5.0,
            "current_price": 99.5,
            "positions": {},
            "reason": "down 5% (model's own, incorrect, claim)",
        }
        client = _FakeClient([_FakeResponse(_cycle_report([order], market_snapshot=snapshot))])

        run_cycle(cfg, client=client)

        events = _read_log_events(cfg.log_file)
        assert any(e["event"] == "order_blocked" for e in events)
        assert not any(e["event"] == "simulated_fill" for e in events)

    def test_agreeing_model_and_recomputed_values_do_not_log_a_divergence(self, tmp_path):
        cfg = _cfg(tmp_path, dry_run=True)
        client = _FakeClient([_FakeResponse(_cycle_report([_approved_buy_order()]))])

        run_cycle(cfg, client=client)

        events = _read_log_events(cfg.log_file)
        assert not any(e["event"] == "snapshot_recomputed" for e in events)


class TestSessionDivergenceLogging:
    def test_session_divergence_event_fires_above_the_threshold(self, tmp_path):
        cfg = _cfg(tmp_path, dry_run=True)
        snapshot = {
            "AAPL": {
                "last_trade_price": 100.0,
                "last_trade_price_timestamp": "2026-01-01T20:00:00Z",
                "last_non_reg_trade_price": 108.0,  # 8% away — well past 1%
                "last_non_reg_trade_price_timestamp": "2026-01-01T21:30:00Z",
                "adjusted_previous_close": 100.0,
                "day_change_pct": 0.0,
            }
        }
        client = _FakeClient([_FakeResponse(_cycle_report([], market_snapshot=snapshot))])

        run_cycle(cfg, client=client)

        events = _read_log_events(cfg.log_file)
        divergences = [e for e in events if e["event"] == "session_divergence"]
        assert len(divergences) == 1
        assert divergences[0]["ticker"] == "AAPL"
        assert divergences[0]["last_trade_price"] == 100.0
        assert divergences[0]["last_non_reg_trade_price"] == 108.0

    def test_no_session_divergence_event_within_the_threshold(self, tmp_path):
        cfg = _cfg(tmp_path, dry_run=True)
        snapshot = {
            "AAPL": {
                "last_trade_price": 100.0,
                "last_trade_price_timestamp": "2026-01-01T20:00:00Z",
                "last_non_reg_trade_price": 100.5,  # 0.5% away — within threshold
                "last_non_reg_trade_price_timestamp": "2026-01-01T21:30:00Z",
                "adjusted_previous_close": 100.0,
                "day_change_pct": 0.0,
            }
        }
        client = _FakeClient([_FakeResponse(_cycle_report([], market_snapshot=snapshot))])

        run_cycle(cfg, client=client)

        events = _read_log_events(cfg.log_file)
        assert not any(e["event"] == "session_divergence" for e in events)
