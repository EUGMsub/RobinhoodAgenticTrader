# Architecture

## Goals

This project connects an LLM agent to a live brokerage account through
Robinhood's Agentic Trading MCP server, using the Claude API's native MCP
connector. The interesting problem isn't "can an LLM call a trading
tool" — it's **how do you let a probabilistic model take consequential,
real-money actions without trusting it to police its own limits, report
its own actions honestly, or even choose its own inputs correctly?**

The design answer: **the model proposes, deterministic code disposes.**
Every place that answer gets tested by a real failure mode — a leaked
cap, a silently wrong price field, a model misreporting what it did — is
a section below.

## The four-step pipeline

Every live path — `approval_mode=True` or `False`, `dry_run=True` or not —
runs the same four steps, in the same order, differing only in the human
step:

1. **PROPOSE** — one MCP call, read-only tools only. No "execute
   directly" variant of this call exists, ever.
2. **VALIDATE** — `validate_batch()` runs unconditionally, on every path,
   over the whole proposed batch at once.
3. **APPROVE** — a human types `YES` per order (`approval_mode=True`), or
   a guardrail-approved order auto-approves (`approval_mode=False`).
   `dry_run=True` stops here and simulates the fill instead.
4. **EXECUTE** — the approved order is placed via MCP, then independently
   reconciled against what the broker actually recorded.

Collapsing these into one path per step was a deliberate fix: an earlier
version had a separate "execute qualifying orders directly" prompt for
`approval_mode=False` that bypassed `validate_batch()` entirely.
`tests/test_agent.py` now asserts that phrase never appears in any
prompt, on any path — the regression is pinned down, not just fixed once.

## Data flow

```
Claude (READ_ONLY_TOOLS) ◀──▶ Robinhood MCP     1. PROPOSE
   token minted per call by oauth.py; raw quote fields back, not just a %
              │
              ▼
   agent._recompute_snapshot()   day_change_pct rebuilt from raw fields;
              │                  model's own number logged, never trusted
              ▼
   guardrails.validate_batch()   2. VALIDATE — pure, cumulative
              │ approved / rejected + reason, per order
              ▼
   dry_run? ──yes──▶ paper_trading.py simulates the fill; stops here
              │ no
              ▼
   human (CLI)                   3. APPROVE
              │
              ▼
Claude (EXECUTION_TOOLS) ────▶ Robinhood MCP    4. EXECUTE
              │
              ▼
   reconcile.py (RECONCILE_TOOLS)   re-fetch the broker's own order
              │                     records, compare to what was approved
              ▼
   logging_utils → trade_log.jsonl → build_dashboard.py
```

## Why `validate_batch()` is cumulative, not per-order

An earlier version called `validate_buy()`/`validate_sell()` once per
order, each check against the same static positions snapshot. That leaks:
if every ticker in a correlation group dips the same day, each buy clears
the group cap independently against the *pre-batch* snapshot, and the
batch collectively overshoots it by up to `(group_size - 1) *
order_dollars` — exactly what the group cap exists to prevent.

`validate_batch()` fixes this with one running snapshot, updated after
each approval, in a fixed order: sells first (they only free capacity),
then buys sorted by dip size (largest first, so the strongest signal
claims scarce capacity when the batch can't all fit). `backtest.py` calls
this same function unchanged rather than reimplementing the rules, so a
day's full candidate set is tested against the same cumulative behavior a
live cycle would hit — not a simplified stand-in for it.

## Per-call MCP tool allowlists

Telling the model "don't place orders during proposal" in a prompt is not
enforcement — it's a request the model could ignore, misread, or be
argued out of by adversarial content in tool output. `agent.py` instead
restricts which MCP tools are *reachable* per call, via an explicit
allowlist passed as an `mcp_toolset` entry:

- `READ_ONLY_TOOLS` — quotes, positions, account, historicals, tax lots.
  Used for every proposal call and the entire `dry_run` path.
- `EXECUTION_TOOLS` — `READ_ONLY_TOOLS` plus `place_equity_order`, used
  **only** inside `_execute_and_reconcile()`.
- `RECONCILE_TOOLS` — order-record lookups only; reconciliation can't
  notice an order it isn't allowed to fetch.

A proposal or dry-run call has no `place_equity_order` tool attached — the
model could not call it even if adversarial tool output talked it into
trying. The allowlist is default-deny (everything off, named tools
re-enabled), so a tool added later is excluded by default, not silently
exposed.

## Authentication: OAuth 2.1 authorization-code + PKCE

Robinhood's Agentic Trading MCP server does not accept a static bearer
token — its `.well-known/oauth-authorization-server` document requires
PKCE (`S256` only) and issues short-lived tokens via an authorization-code
grant, for an anonymous public client (`token_endpoint_auth_method:
"none"`). Confirmed against the real endpoint before any client code was
written, not assumed.

`src/oauth.py` implements the flow: `generate_pkce()` builds a verifier
and S256 challenge; `run_authorization_flow()` opens the system browser,
runs a one-shot local callback server, and verifies the returned `state`
matches before trusting anything else (CSRF protection, not optional);
`get_valid_access_token()` refreshes the stored token proactively, within
a leeway window rather than reactively after a call fails, before every
MCP call. Tokens live outside the repo (owner-only permissions) and are
never logged or printed anywhere in this codebase. `scripts/oauth_login.py`
runs the flow once, interactively, ahead of time, so the trading cycle
itself never blocks on a browser.

## Dry-run paper trading

`dry_run=True` exists so the agent can accumulate a track record without a
funded account, without ever reaching the order-placing code path.
`_execute_and_reconcile()` — the only function that places a real order —
raises `RuntimeError` immediately if `dry_run` is true, before touching
the client; a plain `assert` was rejected since `python -O` strips
assertions, silently deleting the guard.

`paper_trading.py` is a pure ledger: `apply_simulated_buy()`/
`apply_simulated_sell()` take a `PaperPortfolio` and return a new one, no
I/O, tested like `guardrails.py`. Only `load_`/`save_paper_portfolio()`
touch disk, persisting across cycles so dry runs accumulate exposure
instead of resetting — without that, the caps this project tests would
never bind.

## The backtest engine reuses guardrails, unchanged

`backtest.py` makes no network or LLM calls. It replays historical bars
day by day, evaluates each day's full candidate set through the exact
same `validate_batch()` the live agent calls (via
`AgentConfig.guardrail_config()`), and fills approved orders at the
**next** bar's open, not the signal day's own close, to avoid lookahead.
If `guardrails.py` changes, backtest behavior changes with it
automatically — a consumer of the rules, not a second implementation that
could quietly drift out of sync.

## Observability

A no-trade cycle and a broken-tool cycle both produce zero orders; nothing
about a proposal call failing looks different from a proposal call
correctly deciding to do nothing, unless the evidence is captured
separately:

- **Market snapshot logging.** Every cycle logs the model's raw quote
  fields for every watchlist ticker, whether or not an order was
  proposed — the only way to tell "fetched real data, nothing qualified"
  apart from "the tool call may have failed."
- **Code-side `day_change_pct` recomputation.** The model reports raw
  fields (`last_trade_price`, `last_non_reg_trade_price`, both
  timestamps, `adjusted_previous_close`); `agent.py` recomputes
  `day_change_pct` for the configured `price_session` and overwrites
  every order's value before `validate_batch()` sees it. Exists because a
  model once silently used the regular-session close over an 8%
  after-hours move — correct arithmetic, wrong field, chosen silently.
  `price_session` is now explicit config, not a per-run model choice;
  `snapshot_recomputed`/`session_divergence` log any disagreement.
- **Token cost logging.** Every call logs real token counts and an
  estimated dollar cost, attributed by call type, so cost can be checked
  against realized P&L instead of assumed negligible.
- **`build_dashboard.py`.** A static, read-only HTML file rendered from
  `trade_log.jsonl` — no server, no connection back to the agent. It
  re-derives its own arithmetic from raw logged numbers rather than
  parsing guardrail reason strings, so it's an independent check on the
  log, not a mirror of it. A health panel that predates the feature it
  would report on renders nothing — absence of data is not a passing
  check.

## The trust boundary

Three tiers of "how much is this trusted":

- **What the model supplies and code re-derives.** `day_change_pct`,
  `positions`, and raw quote fields all come from the model, but every
  number that gates a trade is re-derived from those inputs in plain
  Python before a decision is made. The model's *inputs* are still
  trusted; its *arithmetic* on them is not.
- **What is structurally unreachable, not just discouraged.** Order
  placement during proposal/dry-run, and any tool outside a call's
  allowlist. No prompt wording is load-bearing here.
- **What remains model-mediated.** Reconciliation asks the same model
  that just executed to report what it did, then verifies that report in
  code — catching accidental failures (tool errors, confused models,
  duplicate fills), but not a sufficiently adversarial actor controlling
  both the execution and the reconciliation read. The most honest
  remaining trust boundary here.

For the current, dated state of what's verified against the live server
versus still simulated or mocked, see the README's **Known limitations**
— that list changes with every live run and would go stale here if
duplicated.
