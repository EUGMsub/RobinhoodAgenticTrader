"""
Unit tests for mcp_client.py.

No real network call anywhere in this file — urllib.request.urlopen is
monkeypatched with a scripted fake transport that returns canned
SSE-framed responses shaped exactly like the ones captured against the
live Robinhood MCP server (see mcp_client.py's module docstring), and
oauth.get_valid_access_token is monkeypatched to avoid touching a real
token file.
"""

import io
import json
import os
import sys
import urllib.error

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mcp_client
from config import AgentConfig
from mcp_client import MCPClientError, fetch_orders, fetch_positions, fetch_quotes


def _cfg(**overrides):
    defaults = dict(
        mcp_url="https://example.invalid/mcp/trading",
        robinhood_client_id="test-client-id",
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


class _FakeHTTPResponse:
    def __init__(self, status: int, headers: dict, body_text: str):
        self.status = status
        self._headers = headers
        self._body = body_text.encode("utf-8")

    def getheaders(self):
        return list(self._headers.items())

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _sse(status: int, message: dict, session_id: str | None = None) -> _FakeHTTPResponse:
    headers = {"Content-Type": "text/event-stream"}
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    body = f"event: message\ndata: {json.dumps(message)}\n\n"
    return _FakeHTTPResponse(status, headers, body)


def _empty_202() -> _FakeHTTPResponse:
    return _FakeHTTPResponse(202, {"Content-Length": "0"}, "")


def _http_error(status: int, body: str = "") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.invalid/mcp/trading", status, "error", {}, io.BytesIO(body.encode()))


DEFAULT_SESSION_ID = "test-session-abc"


def _queue_key(payload: dict) -> str:
    """Requests are keyed by tool name for tools/call (since every tool
    call shares the JSON-RPC method "tools/call" — the tool itself is in
    params.name), and by method name otherwise."""
    method = payload.get("method")
    if method == "tools/call":
        return payload.get("params", {}).get("name", "")
    return method


class _ScriptedTransport:
    """Stands in for urllib.request.urlopen(). initialize and
    notifications/initialized get an auto-generated boilerplate success
    response unless a test explicitly queues an override — every test
    cares about the tool call itself, not the session handshake. Anything
    else must be explicitly queued via .queue(key, *items); each item is
    either a _FakeHTTPResponse, an exception instance to raise, or a
    zero-arg callable that produces one of those.
    """

    def __init__(self):
        self.requests: list[dict] = []
        self.raw_requests: list = []  # the urllib.request.Request objects themselves
        self._queues: dict[str, list] = {}

    def queue(self, key: str, *items) -> None:
        self._queues.setdefault(key, []).extend(items)

    def urlopen(self, req, timeout=None):
        payload = json.loads(req.data.decode("utf-8"))
        self.requests.append(payload)
        self.raw_requests.append(req)
        key = _queue_key(payload)

        if key not in self._queues:
            if payload.get("method") == "initialize":
                return _sse(
                    200,
                    {
                        "jsonrpc": "2.0",
                        "id": payload.get("id"),
                        "result": {
                            "protocolVersion": mcp_client.MCP_PROTOCOL_VERSION,
                            "serverInfo": {"name": "robinhood-trading", "version": "1.0.0"},
                        },
                    },
                    session_id=DEFAULT_SESSION_ID,
                )
            if payload.get("method") == "notifications/initialized":
                return _empty_202()
            raise AssertionError(f"no scripted response for {key!r} (payload={payload})")

        item = self._queues[key].pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item()
        return item


def _tool_result_response(request_id: int, structured_data: dict) -> _FakeHTTPResponse:
    return _sse(
        200,
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"structuredContent": {"data": structured_data}},
        },
    )


class TestFetchQuotes:
    def test_parses_structured_content_into_ticker_keyed_quotes(self, monkeypatch):
        transport = _ScriptedTransport()
        monkeypatch.setattr(mcp_client.urllib.request, "urlopen", transport.urlopen)
        monkeypatch.setattr(mcp_client, "get_valid_access_token", lambda client_id, force_refresh=False: "tok")

        # Shaped exactly like the live get_equity_quotes response captured
        # 2026-08-04 — numeric fields as strings, symbol nested under quote.
        transport.queue(
            "get_equity_quotes",
            _tool_result_response(
                2,
                {
                    "results": [
                        {
                            "quote": {
                                "symbol": "AAPL",
                                "last_trade_price": "309.380000",
                                "last_non_reg_trade_price": "310.660000",
                                "adjusted_previous_close": "303.420000",
                            },
                            "close": {"symbol": "AAPL", "price": "303.42"},
                        }
                    ]
                },
            ),
        )

        result = fetch_quotes(_cfg(), ["AAPL"])

        assert result == {
            "AAPL": {
                "symbol": "AAPL",
                "last_trade_price": "309.380000",
                "last_non_reg_trade_price": "310.660000",
                "adjusted_previous_close": "303.420000",
            }
        }

    def test_symbol_not_in_results_is_absent_not_guessed(self, monkeypatch):
        transport = _ScriptedTransport()
        monkeypatch.setattr(mcp_client.urllib.request, "urlopen", transport.urlopen)
        monkeypatch.setattr(mcp_client, "get_valid_access_token", lambda client_id, force_refresh=False: "tok")

        transport.queue("get_equity_quotes", _tool_result_response(2, {"results": []}))

        result = fetch_quotes(_cfg(), ["ZZZZ"])

        assert result == {}

    def test_sends_only_the_requested_symbols_argument(self, monkeypatch):
        transport = _ScriptedTransport()
        monkeypatch.setattr(mcp_client.urllib.request, "urlopen", transport.urlopen)
        monkeypatch.setattr(mcp_client, "get_valid_access_token", lambda client_id, force_refresh=False: "tok")

        transport.queue("get_equity_quotes", _tool_result_response(2, {"results": []}))

        fetch_quotes(_cfg(), ["AAPL", "MSFT"])

        tool_calls = [r for r in transport.requests if _queue_key(r) == "get_equity_quotes"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["params"]["arguments"] == {"symbols": ["AAPL", "MSFT"]}


class TestFetchPositions:
    def test_combines_positions_and_quotes_into_market_value(self, monkeypatch):
        transport = _ScriptedTransport()
        monkeypatch.setattr(mcp_client.urllib.request, "urlopen", transport.urlopen)
        monkeypatch.setattr(mcp_client, "get_valid_access_token", lambda client_id, force_refresh=False: "tok")

        transport.queue(
            "get_equity_positions",
            _tool_result_response(
                2,
                {
                    "positions": [
                        {"symbol": "AAPL", "quantity": "2.5", "average_buy_price": "100.00", "type": "long"}
                    ]
                },
            ),
        )
        transport.queue(
            "get_equity_quotes",
            _tool_result_response(
                2, {"results": [{"quote": {"symbol": "AAPL", "last_trade_price": "120.00"}}]}
            ),
        )

        result = fetch_positions(_cfg(), "602437931")

        assert result == {"AAPL": pytest.approx(300.0)}  # 2.5 shares * $120.00

    def test_no_positions_returns_empty_without_calling_quotes(self, monkeypatch):
        transport = _ScriptedTransport()
        monkeypatch.setattr(mcp_client.urllib.request, "urlopen", transport.urlopen)
        monkeypatch.setattr(mcp_client, "get_valid_access_token", lambda client_id, force_refresh=False: "tok")

        transport.queue("get_equity_positions", _tool_result_response(2, {"positions": []}))

        result = fetch_positions(_cfg(), "602437931")

        assert result == {}
        assert not any(_queue_key(r) == "get_equity_quotes" for r in transport.requests)

    def test_paginated_positions_response_raises_rather_than_truncating(self, monkeypatch):
        transport = _ScriptedTransport()
        monkeypatch.setattr(mcp_client.urllib.request, "urlopen", transport.urlopen)
        monkeypatch.setattr(mcp_client, "get_valid_access_token", lambda client_id, force_refresh=False: "tok")

        transport.queue(
            "get_equity_positions",
            _tool_result_response(2, {"positions": [], "next": "https://example.invalid/next-page"}),
        )

        with pytest.raises(MCPClientError, match="paginated"):
            fetch_positions(_cfg(), "602437931")


class TestFetchOrders:
    def test_normalizes_equity_orders_and_tags_instrument_type(self, monkeypatch):
        transport = _ScriptedTransport()
        monkeypatch.setattr(mcp_client.urllib.request, "urlopen", transport.urlopen)
        monkeypatch.setattr(mcp_client, "get_valid_access_token", lambda client_id, force_refresh=False: "tok")

        transport.queue(
            "get_equity_orders",
            _tool_result_response(
                2,
                {
                    "orders": [
                        {
                            "symbol": "AAPL",
                            "side": "buy",
                            "state": "filled",
                            # Real shape: nested dollar_based_amount, not a
                            # flat dollar_amount field.
                            "dollar_based_amount": {"amount": "3.00", "currency_code": "USD"},
                        }
                    ]
                },
            ),
        )
        transport.queue("get_option_orders", _tool_result_response(2, {"orders": []}))

        orders = fetch_orders(_cfg(), "602437931", "2026-08-01")

        assert len(orders) == 1
        assert orders[0]["symbol"] == "AAPL"
        assert orders[0]["side"] == "buy"
        assert orders[0]["instrument_type"] == "equity"
        # reconcile.reconcile_order() reads a flat dollar_amount — this is
        # what makes it work unmodified against directly-fetched data.
        assert orders[0]["dollar_amount"] == pytest.approx(3.00)

    def test_missing_dollar_based_amount_normalizes_to_zero_not_dropped(self, monkeypatch):
        transport = _ScriptedTransport()
        monkeypatch.setattr(mcp_client.urllib.request, "urlopen", transport.urlopen)
        monkeypatch.setattr(mcp_client, "get_valid_access_token", lambda client_id, force_refresh=False: "tok")

        transport.queue(
            "get_equity_orders",
            _tool_result_response(
                2, {"orders": [{"symbol": "AAPL", "side": "buy", "dollar_based_amount": None}]}
            ),
        )
        transport.queue("get_option_orders", _tool_result_response(2, {"orders": []}))

        orders = fetch_orders(_cfg(), "602437931", "2026-08-01")

        assert orders[0]["dollar_amount"] == 0.0

    def test_option_orders_are_tagged_and_passed_through_unmodified(self, monkeypatch):
        transport = _ScriptedTransport()
        monkeypatch.setattr(mcp_client.urllib.request, "urlopen", transport.urlopen)
        monkeypatch.setattr(mcp_client, "get_valid_access_token", lambda client_id, force_refresh=False: "tok")

        transport.queue("get_equity_orders", _tool_result_response(2, {"orders": []}))
        transport.queue(
            "get_option_orders",
            _tool_result_response(
                2, {"orders": [{"chain_symbol": "AAPL", "state": "filled", "direction": "debit"}]}
            ),
        )

        orders = fetch_orders(_cfg(), "602437931", "2026-08-01")

        assert len(orders) == 1
        assert orders[0]["instrument_type"] == "option"
        assert orders[0]["chain_symbol"] == "AAPL"
        assert "dollar_amount" not in orders[0]  # never inspected by reconcile_order() for options

    def test_sends_placed_agent_and_created_at_gte_to_both_tools(self, monkeypatch):
        transport = _ScriptedTransport()
        monkeypatch.setattr(mcp_client.urllib.request, "urlopen", transport.urlopen)
        monkeypatch.setattr(mcp_client, "get_valid_access_token", lambda client_id, force_refresh=False: "tok")

        transport.queue("get_equity_orders", _tool_result_response(2, {"orders": []}))
        transport.queue("get_option_orders", _tool_result_response(2, {"orders": []}))

        fetch_orders(_cfg(), "602437931", "2026-08-01T00:00:00Z")

        for tool_name in ("get_equity_orders", "get_option_orders"):
            call = next(r for r in transport.requests if _queue_key(r) == tool_name)
            assert call["params"]["arguments"] == {
                "account_number": "602437931",
                "placed_agent": "agentic",
                "created_at_gte": "2026-08-01T00:00:00Z",
            }

    def test_paginated_orders_response_raises_rather_than_truncating(self, monkeypatch):
        transport = _ScriptedTransport()
        monkeypatch.setattr(mcp_client.urllib.request, "urlopen", transport.urlopen)
        monkeypatch.setattr(mcp_client, "get_valid_access_token", lambda client_id, force_refresh=False: "tok")

        transport.queue(
            "get_equity_orders",
            _tool_result_response(2, {"orders": [], "next": "https://example.invalid/next-page"}),
        )

        with pytest.raises(MCPClientError, match="paginated"):
            fetch_orders(_cfg(), "602437931", "2026-08-01")


class TestNoOrderPlacingCapability:
    def test_every_tool_call_this_module_ever_makes_is_read_only(self, monkeypatch):
        # Not a mock of a hypothetical caller — this walks every request
        # actually captured across a run of every public function, and
        # asserts none of them ever named an order-placing tool. Nothing
        # in this module exposes a way to pass a tool name in from outside
        # (see mcp_client.py's module docstring), so this is confirming a
        # structural property, not testing input validation.
        transport = _ScriptedTransport()
        monkeypatch.setattr(mcp_client.urllib.request, "urlopen", transport.urlopen)
        monkeypatch.setattr(mcp_client, "get_valid_access_token", lambda client_id, force_refresh=False: "tok")

        transport.queue("get_equity_quotes", _tool_result_response(2, {"results": []}))
        transport.queue("get_equity_positions", _tool_result_response(2, {"positions": []}))
        transport.queue("get_equity_orders", _tool_result_response(2, {"orders": []}))
        transport.queue("get_option_orders", _tool_result_response(2, {"orders": []}))

        fetch_quotes(_cfg(), ["AAPL"])
        fetch_positions(_cfg(), "602437931")
        fetch_orders(_cfg(), "602437931", "2026-08-01")

        tool_calls = [r for r in transport.requests if r.get("method") == "tools/call"]
        tool_names = {c["params"]["name"] for c in tool_calls}
        assert tool_names == {
            "get_equity_quotes",
            "get_equity_positions",
            "get_equity_orders",
            "get_option_orders",
        }
        assert "place_equity_order" not in tool_names
        assert "place_option_order" not in tool_names


def _header(req, name: str):
    for key, value in req.headers.items():
        if key.lower() == name.lower():
            return value
    return None


class TestSessionHandshake:
    def test_session_id_from_initialize_is_echoed_on_the_tool_call(self, monkeypatch):
        transport = _ScriptedTransport()
        monkeypatch.setattr(mcp_client.urllib.request, "urlopen", transport.urlopen)
        monkeypatch.setattr(mcp_client, "get_valid_access_token", lambda client_id, force_refresh=False: "tok")

        transport.queue("get_equity_quotes", _tool_result_response(2, {"results": []}))

        fetch_quotes(_cfg(), ["AAPL"])

        # Three requests per call: initialize, notifications/initialized,
        # tools/call. The session id only exists after initialize's
        # response, so it can't be on request 0 — but must be on 1 and 2.
        init_req, initialized_req, tool_call_req = transport.raw_requests

        assert _header(init_req, "Mcp-Session-Id") is None
        assert _header(initialized_req, "Mcp-Session-Id") == DEFAULT_SESSION_ID
        assert _header(tool_call_req, "Mcp-Session-Id") == DEFAULT_SESSION_ID

    def test_notifications_initialized_has_no_id_field(self, monkeypatch):
        transport = _ScriptedTransport()
        monkeypatch.setattr(mcp_client.urllib.request, "urlopen", transport.urlopen)
        monkeypatch.setattr(mcp_client, "get_valid_access_token", lambda client_id, force_refresh=False: "tok")

        transport.queue("get_equity_quotes", _tool_result_response(2, {"results": []}))

        fetch_quotes(_cfg(), ["AAPL"])

        _init, initialized, _tool_call = transport.requests
        assert initialized["method"] == "notifications/initialized"
        assert "id" not in initialized


class TestFourZeroOneRetry:
    def test_401_triggers_exactly_one_refresh_and_retry(self, monkeypatch):
        transport = _ScriptedTransport()
        monkeypatch.setattr(mcp_client.urllib.request, "urlopen", transport.urlopen)

        force_refresh_calls = []

        def _fake_get_token(client_id, force_refresh=False):
            force_refresh_calls.append(force_refresh)
            return "token-2" if force_refresh else "token-1"

        monkeypatch.setattr(mcp_client, "get_valid_access_token", _fake_get_token)

        transport.queue(
            "get_equity_quotes",
            _http_error(401, "unauthorized"),
            _tool_result_response(2, {"results": []}),
        )

        result = fetch_quotes(_cfg(), ["AAPL"])

        assert result == {}
        assert force_refresh_calls == [False, True]

    def test_401_persisting_after_retry_fails_loudly(self, monkeypatch):
        transport = _ScriptedTransport()
        monkeypatch.setattr(mcp_client.urllib.request, "urlopen", transport.urlopen)
        monkeypatch.setattr(mcp_client, "get_valid_access_token", lambda client_id, force_refresh=False: "tok")

        transport.queue(
            "get_equity_quotes",
            _http_error(401, "unauthorized"),
            _http_error(401, "still unauthorized"),
        )

        with pytest.raises(MCPClientError, match="401"):
            fetch_quotes(_cfg(), ["AAPL"])

    def test_non_401_error_is_not_retried(self, monkeypatch):
        transport = _ScriptedTransport()
        monkeypatch.setattr(mcp_client.urllib.request, "urlopen", transport.urlopen)

        force_refresh_calls = []

        def _fake_get_token(client_id, force_refresh=False):
            force_refresh_calls.append(force_refresh)
            return "tok"

        monkeypatch.setattr(mcp_client, "get_valid_access_token", _fake_get_token)

        transport.queue("get_equity_quotes", _http_error(500, "server error"))

        with pytest.raises(MCPClientError, match="500"):
            fetch_quotes(_cfg(), ["AAPL"])

        # Only the first attempt's token fetch happened — a 500 doesn't
        # get the retry-with-forced-refresh treatment a 401 does.
        assert force_refresh_calls == [False]


class TestErrorHandling:
    def test_json_rpc_error_object_raises(self, monkeypatch):
        transport = _ScriptedTransport()
        monkeypatch.setattr(mcp_client.urllib.request, "urlopen", transport.urlopen)
        monkeypatch.setattr(mcp_client, "get_valid_access_token", lambda client_id, force_refresh=False: "tok")

        transport.queue(
            "get_equity_quotes",
            _sse(200, {"jsonrpc": "2.0", "id": 2, "error": {"code": -32602, "message": "bad params"}}),
        )

        with pytest.raises(MCPClientError, match="bad params"):
            fetch_quotes(_cfg(), ["AAPL"])

    def test_tool_level_is_error_result_raises(self, monkeypatch):
        transport = _ScriptedTransport()
        monkeypatch.setattr(mcp_client.urllib.request, "urlopen", transport.urlopen)
        monkeypatch.setattr(mcp_client, "get_valid_access_token", lambda client_id, force_refresh=False: "tok")

        transport.queue(
            "get_equity_quotes",
            _sse(200, {"jsonrpc": "2.0", "id": 2, "result": {"isError": True, "content": [{"type": "text", "text": "boom"}]}}),
        )

        with pytest.raises(MCPClientError, match="tool error"):
            fetch_quotes(_cfg(), ["AAPL"])

    def test_non_sse_body_raises(self, monkeypatch):
        transport = _ScriptedTransport()
        monkeypatch.setattr(mcp_client.urllib.request, "urlopen", transport.urlopen)
        monkeypatch.setattr(mcp_client, "get_valid_access_token", lambda client_id, force_refresh=False: "tok")

        transport.queue(
            "get_equity_quotes",
            _FakeHTTPResponse(200, {"Content-Type": "application/json"}, '{"jsonrpc": "2.0"}'),
        )

        with pytest.raises(MCPClientError, match="no SSE"):
            fetch_quotes(_cfg(), ["AAPL"])
