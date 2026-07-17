# Robinhood Agentic Trader

A guardrailed LLM trading agent built on the **Claude API's native MCP
connector** and **Robinhood's Agentic Trading MCP server**. The agent
proposes trades in natural language; a separate, fully unit-tested layer of
plain-Python guardrails independently re-validates every proposal before a
human approves it and it reaches the broker.

> **Disclaimer:** This is a software engineering demonstration, not
> financial advice, and not a claim that the included strategy is
> profitable. It trades real money if you connect a funded account — read
> [Safety design](#safety-design) before running it, and start small.

## Why this project

Anthropic's Claude API supports connecting Claude directly to remote MCP
servers via the `mcp_servers` parameter, and Robinhood recently opened an
official Agentic Trading product built on MCP — a live, real-stakes
environment where "the model made a mistake" has actual consequences. That
combination makes it a good test of a question that matters for any
agentic system, not just trading: **how do you scope down what an LLM is
*able* to do, rather than trusting it to stay within what it's *told* to
do?**

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full data-flow
diagram and design rationale. Short version:

```
Claude (reads market data via MCP) → proposes trade + raw numbers
                → guardrails.py (pure, unit-tested) independently re-checks
                → human approves in CLI
                → Claude executes via MCP
                → structured audit log
```

## Safety design

- **Hard limits live in code, not just in the prompt.** `src/guardrails.py`
  has zero dependency on the model or the network — it's a set of pure
  functions checking floats against fixed thresholds (watchlist
  membership, per-order size, per-position cap, total account cap, dip/
  revert thresholds). 15 unit tests in `tests/test_guardrails.py` cover the
  boundary conditions.
- **The model's own justification is re-derived, not trusted.** When
  proposing a trade, the model must include the raw numbers behind its
  reasoning; `guardrails.py` recomputes the approve/reject decision from
  those numbers independently.
- **Human-in-the-loop by default.** `approval_mode=True` means nothing
  executes until a human types `YES` for that specific order.
- **Everything is logged.** `trade_log.jsonl` is an append-only record of
  every cycle report, proposal, block, approval, skip, and execution
  result.
- **Scoped account.** Robinhood's Agentic Trading product requires trades
  to go through a dedicated Agentic account, separate from your main
  brokerage holdings — the agent physically cannot touch other accounts.

## The strategy

A disclosed, simple mean-reversion dip-buy, chosen because it's easy to
verify by hand (which matters far more here than backtested performance):

1. Buy a fixed dollar amount of a watchlisted ticker when it's down some
   threshold on the day.
2. Sell a full position once it's reverted a threshold above cost basis.
3. Otherwise, do nothing — most cycles should be no-ops.

All thresholds are configurable in `src/config.py`.

## Setup

**1. Get Robinhood Agentic Trading access** (currently a gradual rollout —
sign up at [robinhood.com/us/en/support/agentic-trading](https://robinhood.com/us/en/support/agentic-trading))
and create a dedicated Agentic account. This gives you an MCP server URL
and an OAuth token.

**2. Clone and install:**

```bash
git clone https://github.com/<your-username>/robinhood-agentic-trader.git
cd robinhood-agentic-trader
pip install -r requirements.txt
```

**3. Configure credentials:**

```bash
cp .env.example .env
# fill in ANTHROPIC_API_KEY, ROBINHOOD_MCP_URL, ROBINHOOD_MCP_TOKEN
export $(cat .env | xargs)   # or use a tool like python-dotenv / direnv
```

**4. Adjust strategy parameters** in `src/config.py` if desired (watchlist,
dollar caps, thresholds).

**5. Run the tests** (no credentials required — pure logic only):

```bash
pytest tests/ -v
```

**6. Run the agent:**

```bash
python scripts/run_agent.py
```

In approval mode (default), you'll see a report of what was checked and be
prompted to type `YES` for any proposed order before it executes.

## Project structure

```
robinhood-agentic-trader/
├── src/
│   ├── config.py         # typed, explicit configuration
│   ├── strategy.py       # the natural-language strategy prompt
│   ├── guardrails.py     # pure, unit-tested hard limits
│   ├── agent.py          # orchestration: MCP calls + guardrails + approval
│   └── logging_utils.py  # structured audit logging
├── tests/
│   └── test_guardrails.py
├── scripts/
│   └── run_agent.py      # entry point
└── docs/
    └── ARCHITECTURE.md
```

## License

MIT — see [LICENSE](LICENSE).
