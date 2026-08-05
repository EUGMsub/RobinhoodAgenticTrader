"""
mcp_client.py
=============
A direct HTTP client for Robinhood's Agentic Trading MCP server — no model
in the loop.

Why this exists: three limitations shared one root cause — the model was
the only channel between this code and reality.
  - README Known Limitation #1: guardrails re-derived DECISIONS from
    model-reported INPUTS, never the inputs themselves. A model that
    misreported (or had injected into it) a quote field produced a
    validated decision on false data.
  - README Known Limitation #2: reconciliation asked the same model that
    executed an order to also report what it executed — a check with no
    teeth against anything adversarial.
  - tests/test_prompt_injection.py's input-falsification characterization
    test documented exactly this gap as a real, passing (i.e. undefended)
    test.

fetch_quotes()/fetch_positions()/fetch_orders() close all three: they hit
the MCP server directly with an HTTP POST and this repo's own OAuth token
(oauth.get_valid_access_token()), parse the response, and hand back plain
data. No LLM call happens anywhere in this module.

Wire protocol — discovered against the LIVE server on 2026-08-04, not
assumed from a generic MCP client:
  - Transport is MCP's "Streamable HTTP": a single POST endpoint. Every
    non-empty response observed came back framed as Server-Sent Events
    (`Content-Type: text/event-stream`, body `event: message\\ndata: {...}`)
    — never plain application/json, even though the Accept header offers
    both. A client MUST parse SSE framing, not json.loads() the body raw.
  - `initialize`'s response carries a session id in the `Mcp-Session-Id`
    response header. It must be echoed back on every subsequent request in
    the same session as a request header of the same name.
  - `notifications/initialized` is a JSON-RPC *notification* (no `id`) —
    the server replies `202 Accepted` with an empty body, no SSE framing.
  - Tool results carry the same payload twice: `content[0].text` (a
    double-JSON-encoded string, meant for a model to read) and
    `structuredContent` (the same data as real nested JSON). This module
    always reads `structuredContent` and ignores `content`.
  - Numeric fields inside tool results (prices, quantities, dollar
    amounts) come back as quoted STRINGS ("309.380000"), not JSON numbers.
    Every function here casts explicitly rather than trusting `float()` to
    be a no-op.
  - Each fetch_* call opens its own short-lived session (initialize ->
    notifications/initialized -> tools/call) instead of caching a session
    across calls. Simpler and more robust than session-lifetime management
    for a client that runs a handful of times per day; the extra two round
    trips are immaterial at this call volume.
  - List endpoints (get_equity_orders, get_option_orders, and — per the
    same tool family's documented shape — get_equity_positions) paginate
    via a `data.next` URL, empty when there are no more pages. This module
    does not implement pagination: if `next` ever comes back non-empty, it
    raises rather than silently returning a truncated page as if it were
    complete (see CLAUDE.md: "empty/partial data never renders as a
    passing check"). At this project's order volume (single-digit trades
    at $3 each), a second page has never been observed.

No order-placing capability exists anywhere in this module, structurally,
not just by convention: the only tool names ever sent are the four
hardcoded literals inside fetch_quotes/fetch_positions/fetch_orders. There
is no function here that accepts a caller-supplied tool name, so nothing
outside this module can make it call place_equity_order or
place_option_order even by mistake.
"""

import json
import urllib.error
import urllib.request

from config import AgentConfig
from oauth import get_valid_access_token

MCP_PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "robinhood-agentic-trader-direct-client", "version": "1.0.0"}

REQUEST_TIMEOUT_SECONDS = 20


class MCPClientError(Exception):
    """Raised on any failure talking to the MCP server: HTTP error, a
    JSON-RPC-level error object, a tool-level error result, or a response
    that doesn't parse as this server's known framing."""


class _HTTPStatusError(Exception):
    """Internal-only: carries the HTTP status code up to _call_tool() so it
    can decide whether a 401 is worth one retry. Never raised across this
    module's public boundary — callers only ever see MCPClientError."""

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


def _parse_sse_json_rpc(raw_text: str) -> dict:
    """Parse one `event: message\\ndata: {...}` SSE frame into its JSON-RPC
    message dict. Robinhood's MCP server frames every non-empty response
    this way (see module docstring) — plain JSON is never observed here,
    so this is the only parse path, not a fallback."""
    for line in raw_text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            candidate = line[len("data:"):].strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                raise MCPClientError(f"malformed SSE data line: {candidate!r}") from e
    raise MCPClientError(f"no SSE 'data:' line found in response body: {raw_text!r}")


def _post(
    url: str,
    token: str,
    payload: dict,
    session_id: str | None = None,
    protocol_version: str | None = None,
) -> tuple[int, dict, str]:
    """POST one JSON-RPC message. Returns (status, response_headers, raw_text).
    Raises _HTTPStatusError on any non-2xx status."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    if protocol_version:
        headers["MCP-Protocol-Version"] = protocol_version

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            return resp.status, dict(resp.getheaders()), resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise _HTTPStatusError(e.code, detail) from e


def _open_session(mcp_url: str, token: str) -> tuple[str | None, str | None]:
    """Run initialize + notifications/initialized. Returns (session_id,
    negotiated_protocol_version) — either may be None if the server didn't
    provide one, though the live server always has so far."""
    init_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        },
    }
    _status, headers, text = _post(mcp_url, token, init_payload)
    session_id = next((v for k, v in headers.items() if k.lower() == "mcp-session-id"), None)

    message = _parse_sse_json_rpc(text)
    if "error" in message:
        raise MCPClientError(f"initialize failed: {message['error']}")
    negotiated_version = (message.get("result") or {}).get("protocolVersion")

    _post(
        mcp_url,
        token,
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        session_id=session_id,
        protocol_version=negotiated_version,
    )
    return session_id, negotiated_version


def _call_tool(cfg: AgentConfig, tool_name: str, arguments: dict) -> dict:
    """Open a fresh session and call exactly one tool. On a 401, refresh
    the access token once (force_refresh=True bypasses the "not expired
    yet" cache check — a 401 despite a locally-unexpired token means the
    cache is wrong, e.g. server-side revocation) and retry exactly once;
    any other failure, or a second 401, fails loudly.

    `tool_name` is always a literal passed by one of this module's three
    public functions — never a caller-supplied value. That is what makes
    "no order-placing capability" a structural property of this module
    rather than a convention someone could bypass.
    """
    last_401: _HTTPStatusError | None = None

    for attempt in (1, 2):
        token = get_valid_access_token(cfg.robinhood_client_id, force_refresh=(attempt == 2))
        try:
            session_id, negotiated_version = _open_session(cfg.mcp_url, token)
            call_payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
            _status, _headers, text = _post(
                cfg.mcp_url,
                token,
                call_payload,
                session_id=session_id,
                protocol_version=negotiated_version,
            )
        except _HTTPStatusError as e:
            if e.status == 401 and attempt == 1:
                last_401 = e
                continue
            raise MCPClientError(f"{tool_name} failed: HTTP {e.status}: {e.body}") from e

        message = _parse_sse_json_rpc(text)
        if "error" in message:
            raise MCPClientError(f"{tool_name} failed: {message['error']}")
        result = message.get("result") or {}
        if result.get("isError"):
            raise MCPClientError(f"{tool_name} tool error: {result}")
        return result

    raise MCPClientError(
        f"{tool_name} failed: HTTP 401 persisted after one forced token refresh"
    ) from last_401


def _structured_data(result: dict) -> dict:
    return (result.get("structuredContent") or {}).get("data") or {}


def _reject_if_paginated(data: dict, tool_name: str) -> None:
    # Fail loudly rather than silently return page 1 as if it were
    # everything — see module docstring. Never observed in this project's
    # order volume, but "never observed" isn't "structurally impossible."
    if data.get("next"):
        raise MCPClientError(
            f"{tool_name} response is paginated (data.next is non-empty) — "
            "pagination is not implemented; refusing to silently return a "
            "partial result"
        )


def fetch_quotes(cfg: AgentConfig, symbols: list[str]) -> dict[str, dict]:
    """Fetch real-time quotes for `symbols` directly from the MCP server —
    no model in the loop. Returns each ticker's raw quote fields exactly as
    the server reports them (last_trade_price, last_non_reg_trade_price,
    adjusted_previous_close, etc. — all as the STRINGS the server sends),
    keyed by uppercased ticker. A symbol the server didn't return a quote
    for is simply absent from the result, the same "missing means missing"
    convention agent.py's own snapshot recomputation already uses.
    """
    result = _call_tool(cfg, "get_equity_quotes", {"symbols": list(symbols)})
    data = _structured_data(result)
    results = data.get("results") or []

    quotes: dict[str, dict] = {}
    for entry in results:
        if not isinstance(entry, dict):
            continue
        quote = entry.get("quote")
        if isinstance(quote, dict) and quote.get("symbol"):
            quotes[str(quote["symbol"]).upper().strip()] = quote
    return quotes


def fetch_positions(cfg: AgentConfig, account_number: str) -> dict[str, float]:
    """Fetch current equity positions for `account_number` directly from
    the MCP server, and their market value — the same shape
    guardrails.validate_buy()/validate_batch() expect for their
    `positions` argument (ticker -> current market value in dollars).

    get_equity_positions itself reports quantity and cost basis but NOT a
    current price (confirmed against the live server: its own response
    guide says exactly this — "No market price here — for current value
    ... call get_equity_quotes and multiply by quantity"). So this
    function makes a second call, fetch_quotes(), for whatever symbols
    come back, and marks each position at that quote's last_trade_price
    (regular session) — an approximation used only for guardrail exposure
    caps, not for pricing a trade decision.
    """
    result = _call_tool(cfg, "get_equity_positions", {"account_number": account_number})
    data = _structured_data(result)
    _reject_if_paginated(data, "get_equity_positions")
    raw_positions = data.get("positions") or []

    tickers = [
        str(p["symbol"]).upper().strip()
        for p in raw_positions
        if isinstance(p, dict) and p.get("symbol")
    ]
    if not tickers:
        return {}

    quotes = fetch_quotes(cfg, tickers)

    positions: dict[str, float] = {}
    for entry in raw_positions:
        if not isinstance(entry, dict) or not entry.get("symbol"):
            continue
        ticker = str(entry["symbol"]).upper().strip()
        quantity = entry.get("quantity")
        quote = quotes.get(ticker)
        if quantity is None or quote is None:
            continue
        last_trade_price = quote.get("last_trade_price")
        if last_trade_price is None:
            continue
        positions[ticker] = positions.get(ticker, 0.0) + float(quantity) * float(last_trade_price)
    return positions


def fetch_orders(cfg: AgentConfig, account_number: str, since: str) -> list[dict]:
    """Fetch every equity and option order for `account_number` placed at
    or after `since` (ISO-8601 or YYYY-MM-DD) by the agentic channel,
    directly from the MCP server.

    Each record carries `instrument_type` ("equity" or "option") so a
    caller can split them into the two lists reconcile.reconcile_order()
    expects. Equity records also carry a normalized `dollar_amount` (a
    plain float) alongside the raw fields: the live server reports it
    nested as `dollar_based_amount.amount` (a quoted string), but
    reconcile_order() — deliberately left unchanged by this module — reads
    a flat `order.get("dollar_amount", 0)`. Missing/null
    dollar_based_amount normalizes to 0.0, which fails reconciliation
    closed (a real approved order always has a positive dollar amount to
    match against) rather than silently passing.

    Option records are passed through with their raw fields plus
    `instrument_type` only — reconcile_order() never inspects an option
    order's fields, only whether the list is non-empty at all (this
    strategy is equities-only; any option order present is itself the
    violation).
    """
    orders: list[dict] = []

    for tool_name, instrument_type in (
        ("get_equity_orders", "equity"),
        ("get_option_orders", "option"),
    ):
        result = _call_tool(
            cfg,
            tool_name,
            {
                "account_number": account_number,
                "placed_agent": "agentic",
                "created_at_gte": since,
            },
        )
        data = _structured_data(result)
        _reject_if_paginated(data, tool_name)
        raw_orders = data.get("orders") or []

        for entry in raw_orders:
            if not isinstance(entry, dict):
                continue
            record = {**entry, "instrument_type": instrument_type}
            if instrument_type == "equity":
                dollar_based_amount = entry.get("dollar_based_amount") or {}
                amount = dollar_based_amount.get("amount") if isinstance(dollar_based_amount, dict) else None
                record["dollar_amount"] = float(amount) if amount is not None else 0.0
            orders.append(record)

    return orders
