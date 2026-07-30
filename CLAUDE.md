# RobinhoodAgenticTrader — project context

## What this is

A guardrailed LLM trading agent: Claude reads market data via Robinhood's
Agentic Trading MCP server and proposes trades; pure Python re-derives every
decision and enforces limits. Portfolio piece for a job application. The value
is the safety architecture and honest measurement, **not** returns.

## Core principle (do not violate)

**The model proposes, deterministic code disposes.** Any safety claim enforced
only by prompt text is a bug. If a constraint matters, it lives in
`src/guardrails.py` as a pure function with unit tests.

Corollary: never trust the model's reasoning, arithmetic, or its report of what
it did. Re-derive independently.

## Pipeline (every live path is identical except the human step)

1. **PROPOSE** — read-only MCP tools only; order-placing tools not attached
2. **VALIDATE** — `validate_batch()` always runs, on every path
3. **APPROVE** — human types YES, or auto-approve if `approval_mode=False`;
   `dry_run=True` stops here and simulates the fill
4. **EXECUTE** — `_execute_and_reconcile()`, then compare broker's actual
   orders to what was approved

## Design decisions that must not be reverted

Each of these fixed a real bug. Reverting reintroduces it.

- **No tight stop-loss.** Contradicts mean reversion (sells exactly when the
  signal says hold). Exits are: profit target, time limit, wide disaster stop.
- **`validate_batch`, not per-order validation.** Independent validation let a
  market-wide dip clear the group cap by `(group_size-1) * order_dollars`.
- **Correlation groups are declared, not computed.** Measured correlations
  converge to 1 during crashes — they loosen exactly when they should tighten.
- **Guard uses `raise RuntimeError`, never `assert`.** `python -O` strips
  asserts, silently deleting the guard.
- **Per-call MCP tool allowlist.** Order-placing tools must be unreachable
  during proposal and dry-run, not merely discouraged.
- **Dashboard re-derives arithmetic** rather than parsing guardrail reason
  strings — it's an independent check, not a mirror.
- **Empty data never renders as a passing check.** Absence of evidence is not
  evidence of absence; this bug class has appeared 4x in this project.

## Conventions

- `src/guardrails.py`, `src/reconcile.py`, `src/paper_trading.py`,
  `src/backtest.py` are **pure** — no network, no LLM, no file I/O
- Tests never make network calls; use fake clients
- `backtest.py` reuses guardrail functions unchanged — never reimplement them
- Secrets only in `.env` (gitignored). Never hardcode account numbers.
- Run `pytest tests/ -q` before every commit

## Current state

- 183 tests (182 passing, 1 POSIX-only permission check skipped on Windows)
- Backtest done: strategy +8.06% vs VOO +38.67% over ~2yr. Underperforms; this
  is disclosed in the README, not hidden.
- Agentic account funded with $10. `order_dollars` must be ≤ ~$3 or orders will
  be rejected for insufficient funds.
- **First live run: 2026-07-30.** Real API calls made for the first time in
  this project's history. No order has ever been placed — every live call so
  far has been auth-only (login) or read-only (`--dry-run` proposal cycles).
  What today verified, precisely:
  - **OAuth (README #11):** `src/oauth.py`'s authorization-code + PKCE flow
    verified end-to-end against the live server — `scripts/oauth_login.py`
    completed a real login, state verification, code exchange, token storage.
    Still unverified: token refresh (no login so far has produced an expired
    access token) and refresh-token rotation handling.
  - **MCP tool calls (README #11):** `get_equity_quotes`, `get_accounts`, and
    `get_equity_positions` all worked against the live server — real quotes
    fetched, agentic account (`<agentic-account>`) confirmed with 0 positions.
    `EXECUTION_TOOLS`/`RECONCILE_TOOLS` remain unverified; no order has ever
    qualified for approval.
  - **MCP beta header (README #12):** resolved. `MCP_BETA_HEADER` was wrong
    (`mcp-client-2025-11-25`) for an unknown period; invisible to all 139
    tests passing at the time because every test uses a fake client. Caught
    only by the first real API call. Fixed to `mcp-client-2025-11-20`.
  - **Session-price false negative (README #13), NEW:** the model silently
    used regular-session price and missed a real ~8% after-hours move on
    AAPL. Fixed by making `price_session` an explicit `AgentConfig` value
    (`"regular"` by default), recomputing `day_change_pct` in code from raw
    quote fields instead of trusting the model's arithmetic, and logging
    `session_divergence` on >1% regular/extended disagreement — confirmed
    firing for real (AAPL, ~6.2% gap) on a later cycle the same day. The
    default still ignores after-hours moves by design; what changed is that
    the choice is explicit and logged, not silently made by the model.
  - Two real proposal cycles cost $0.0549 and $0.0616 — see README "Cost per
    trade" for what that implies at scale.

## Honesty rule

This project's credibility rests on accurate reporting. Never fabricate results,
never present sample/demo data as real, never let generated test data reach a
commit or a README screenshot.
