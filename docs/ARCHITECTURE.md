# Architecture

## Goals

This project connects an LLM agent to a live brokerage account through
Robinhood's Agentic Trading MCP server, using the Claude API's native MCP
connector. The interesting engineering problem isn't "can an LLM call a
trading tool" — it's **how do you let a probabilistic model take
consequential, real-money actions without trusting it to police its own
limits?**

The design answer here is: **the model proposes, deterministic code
disposes.**

## Data flow

```
┌─────────────┐     watchlist quotes      ┌──────────────────────┐
│   Claude     │ ─────────────────────────▶│ Robinhood Trading MCP │
│ (strategy.py │◀─────────────────────────  │      (remote)         │
│  + agent.py) │   positions, buying power  └──────────────────────┘
└──────┬───────┘
       │ proposed_orders (JSON, includes raw numbers)
       ▼
┌──────────────┐
│ guardrails.py │  ← pure functions, no I/O, fully unit tested
└──────┬───────┘
       │ approved / rejected + reason
       ▼
┌──────────────┐
│ human (CLI)   │  ← approval_mode=True: nothing executes without this
└──────┬───────┘
       ▼
┌──────────────┐     order placement       ┌──────────────────────┐
│   Claude      │ ─────────────────────────▶│ Robinhood Trading MCP │
│ (execution)   │                            └──────────────────────┘
└──────┬───────┘
       ▼
┌──────────────┐
│ logging_utils │  →  trade_log.jsonl (append-only audit trail)
└──────────────┘
```

## Why re-validate the model's own numbers?

In approval mode, the model is asked to include the raw inputs behind its
own proposal (e.g. `day_change_pct`, `current_position_value`) alongside
its conclusion. `guardrails.py` re-derives the approve/reject decision from
those numbers independently, rather than trusting the model's stated
reasoning. This closes a specific failure mode: a model can state a
correct-sounding justification while being wrong about the underlying
arithmetic, or can be steered by adversarial content encountered while
reading market data or tool output. A pure function checking a float
against a fixed threshold cannot be talked into anything.

## Why separate `strategy.py`, `guardrails.py`, and `agent.py`?

- **`strategy.py`** — the *intent*, in natural language. Easy to review,
  easy to change, easy for a non-engineer to audit.
- **`guardrails.py`** — the *hard limits*, in plain Python. Zero
  dependencies on the model or network; fully covered by
  `tests/test_guardrails.py`.
- **`agent.py`** — the *orchestration* that wires them together, calls the
  Claude API with the MCP server attached, and enforces that guardrails run
  before anything reaches human approval or the broker.

Keeping these separate means the safety-critical logic (guardrails) can be
tested and reasoned about without ever making a network call, and the
strategy can be iterated on without touching the parts that make it safe.

## Known limitations

- Robinhood Agentic Trading is in beta and currently supports equities
  only — no options, crypto, or futures via this path.
- `approval_mode=False` (fully autonomous execution) is supported by the
  code but not recommended without an extended track record in approval
  mode first.
- The strategy itself (mean-reversion dip-buying) is a simple, disclosed
  baseline for demonstrating the architecture — it is not a claim of
  profitability. See the Disclaimer in the README.
