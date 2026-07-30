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

- 137 tests passing
- Backtest done: strategy +8.06% vs VOO +38.67% over ~2yr. Underperforms; this
  is disclosed in the README, not hidden.
- **Never run live.** No real API cycle has executed.
- Agentic account funded with $10. `order_dollars` must be ≤ ~$3 or orders will
  be rejected for insufficient funds.
- **Confirmed, not suspected:** Robinhood's Agentic Trading MCP uses OAuth
  with short-lived tokens issued to a client. A static bearer token in `.env`
  cannot authenticate, so `run_agent.py`'s direct-token path cannot work as
  designed. Cycles must run through an already-authenticated client (e.g.
  Claude Code holding the MCP connection) until a proper OAuth
  authorization-code flow with token refresh is implemented — that is not yet
  done.

## Honesty rule

This project's credibility rests on accurate reporting. Never fabricate results,
never present sample/demo data as real, never let generated test data reach a
commit or a README screenshot.
