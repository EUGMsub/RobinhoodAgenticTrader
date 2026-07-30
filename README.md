# Robinhood Agentic Trader

A guardrailed LLM trading agent built on the **Claude API's MCP connector** and
**Robinhood's Agentic Trading MCP server**. Claude reads live market data and
proposes trades in natural language; a separate layer of pure, unit-tested
Python independently re-derives every decision, enforces hard limits, and
requires human approval before anything reaches the broker.

> **Disclaimer:** This is a software engineering demonstration, not financial
> advice, and not a claim that the included strategy is profitable. The
> backtest below shows it underperforming a passive benchmark. It trades real
> money if you connect a funded account.

## The question this project is about

Connecting an LLM to a brokerage is a paste-one-URL affair. The interesting
engineering problem is what happens next: **how do you let a probabilistic
model take consequential, real-money actions without trusting it to police its
own limits?**

The answer implemented here is *the model proposes, deterministic code
disposes* — and, deliberately, the code does not trust the model's reasoning,
its arithmetic, or its report of what it did.

## Status

- ✅ Guardrail layer: 67 unit tests passing (`pytest tests/ -v`), no
  credentials required to verify.
- ✅ Backtested against ~2 years of real daily bars. Results below.
- ✅ Agent orchestration, MCP wiring, batch validation, instrument-type
  enforcement, post-execution reconciliation, CLI approval flow.
- ⏳ **Not yet run against a funded live account.** The Agentic account exists
  but is unfunded. The guardrail and backtest layers were built and validated
  first, deliberately, before putting real money behind it.

## Backtest results

This backtest was run under the previous configuration ($400 starting cash,
$25 orders, $150 per-position cap, $200 group cap). `src/config.py` now ships
smaller defaults sized for a $10 funded live account; the figures below are
historical and accurate as reported for the configuration they were run under.

Period: 2024-07-29 → 2026-07-27. Starting cash $400. Slippage 0.05% per side.
Signals computed on each bar's close, filled at the **next** bar's open — never
same-bar, to avoid lookahead.

| | Return | Max drawdown | Return / drawdown |
|---|---|---|---|
| **Strategy** | **+8.06%** | **2.60%** | **3.10** |
| Equal-weight buy & hold (VOO/AAPL/MSFT) | +28.92% | 23.83% | 1.21 |
| VOO buy & hold | +38.67% | 18.69% | 2.07 |

66 round trips · 72.7% win rate · both benchmarks beat the strategy on raw
return.

### Reading these numbers honestly

**The strategy lost on return and won on drawdown — but the drawdown win is
mostly low exposure, not risk management.** With $25 orders and a $200 group
cap against $400 of cash, the portfolio sat roughly 86% in cash. Being out of
the market is why the drawdown is shallow.

The fair test is a drawdown-matched passive comparison. A static 13.9% VOO /
86.1% cash portfolio has the same 2.60% max drawdown and returns 5.38% — which
the strategy's 8.06% beats. **However, this backtest credits no interest on
idle cash.** Over 2024–2026, short-term Treasuries and money market funds paid
roughly 4% annually. Crediting the cash leg at that rate brings the passive
drawdown-matched mix to roughly 12–13%, which **beats the strategy**.

Conclusion: at this account size and configuration, the strategy did not add
value over a passive mix at equivalent risk once cash yield is accounted for.

### Cost per trade, and why it may be the most useful result here

+8.06% on $400 is **$32.24 of profit across 66 round trips — about 49 cents per
round trip**, over two years.

Each daily cycle costs at least one Claude API call, plus additional calls on
days that trade (execution, then reconciliation). Across ~500 trading days,
[estimated, not yet measured] inference costs are plausibly in the same order
of magnitude as the returns. **At small account sizes, an LLM-driven trading
agent can cost roughly as much to operate as it earns.**

Measuring this precisely against a live run is the most interesting open
question in this project.

### On the 72.7% win rate

Win rate is a vanity metric and this project is a good demonstration of why. A
72.7% win rate that produces 49 cents per trade tells you almost nothing
without average win size, average loss size, and cost per trade. Promotional
material for AI trading systems leads with figures like this precisely because
they sound impressive in isolation.

## Architecture

Full data flow and design rationale: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

```
Claude (reads live quotes/positions via MCP) -> proposes trades + raw inputs
     |
validate_batch()  - pure, cumulative, unit-tested; re-derives every decision
     |
human approval (CLI, per order)
     |
Claude executes via MCP
     |
reconcile_order() - compares broker's actual orders to what was approved
     |
trade_log.jsonl   - append-only audit trail
```

### Safety design

- **Hard limits live in code, not the prompt.** `src/guardrails.py` has zero
  network or model dependencies — pure functions checking floats against fixed
  thresholds.
- **The model's reasoning is re-derived, not trusted.** Proposals must include
  the raw numbers behind them; guardrails recompute the decision independently.
- **Batch validation, not per-order.** `validate_batch()` evaluates a day's
  orders cumulatively against a running position snapshot. Validating orders
  independently allowed a market-wide dip to clear the group cap by up to
  `(group_size - 1) * order_dollars` — the exact scenario the cap exists to
  prevent.
- **Correlation groups, declared not computed.** VOO *contains* AAPL and MSFT,
  so they are capped as one bet. Structural grouping is used rather than
  rolling correlation because measured correlations converge toward 1 during
  crashes — loosening precisely when the constraint should tighten.
- **Exit rules that can fire on losers.** An earlier version could only sell
  above cost basis, so a declining position was held indefinitely. Exits now
  fire on profit target, a time limit (thesis invalidation), or a wide disaster
  stop. A tight stop-loss is deliberately *not* used: it contradicts a
  mean-reversion premise by selling exactly when the signal says hold.
- **Instrument-type enforcement.** Robinhood's agentic channel now supports
  long options orders. `validate_instrument()` rejects anything that isn't a
  plain equity symbol, including OCC-style option identifiers.
- **Post-execution reconciliation.** After execution, the actual orders on the
  account are fetched and compared to what was approved — symbol, side, count,
  and dollar amount within 1%.
- **Human-in-the-loop by default.** `approval_mode=True`; nothing executes
  without a per-order `YES`.
- **Scoped account.** Robinhood requires agent trades to route through a
  dedicated Agentic account, isolated from other holdings.

## Known limitations

Stated plainly because they're real, and because a reviewer will find them.

**Trust boundary**

1. **Guardrails re-derive decisions, not inputs.** The model supplies
   `day_change_pct`, `days_held`, and `positions`. A model that misreports an
   input gets a validated decision on false data. Narrowing this requires
   fetching market data independently of the model.
2. **Reconciliation is model-mediated.** The same model that executes is asked
   to report what it executed. This catches accidental failures — confused
   models, tool errors, duplicate fills — but is worth little against anything
   adversarial. A direct HTTP call to the MCP server from Python, with no model
   in the loop, would close it.
3. **Capacity/reconciliation seam.** `running_positions` updates on
   approve-and-execute, before reconciliation runs. If reconciliation flags a
   mismatch, capacity was already assumed spent. Fails conservative.
4. **No prompt-injection testing.** Market data and tool output are untrusted
   text reaching a model with a path to order placement. The guardrails bound
   the damage; there is no test suite proving it.

**Backtest**

5. **No interest credited on idle cash** — materially flatters the strategy
   relative to a passive mix, as discussed above.
6. **Ticker selection bias.** VOO/AAPL/MSFT were chosen in 2026 knowing they
   rose. This measures whether the *rules* beat holding those three names, not
   whether the strategy would have found them.
7. **One period, one regime, three correlated tickers.** Max drawdown from a
   single 2-year sample is a noisy statistic.
8. **Data source caveats.** `scripts/load_bars_local.py` uses `yfinance` (an
   unofficial Yahoo Finance wrapper) with `auto_adjust=True`, giving
   total-return series; `scripts/fetch_bars.py` requests Robinhood's split-only
   adjustment. The two are **not** interchangeable on dividend-paying tickers.
   Back-adjusted prices also embed a technical lookahead, immaterial at
   quarterly dividend scale against a 2% trigger.

**Operational**

9. **Time exits resolve at schedule granularity.** `max_hold_days = 10`
   evaluated on a weekly schedule means roughly 14 calendar days.
10. **The effective account cap is $200, not $400.** With every watchlist
    ticker in one correlation group and `max_group_dollars = 200`,
    `max_total_dollars = 400` can never bind.
11. **`run_agent.py`'s static-token auth cannot work as designed — confirmed,
    not suspected.** Robinhood's Agentic Trading MCP uses OAuth with
    short-lived tokens issued to a client, so a static bearer token pasted
    into `.env` as `ROBINHOOD_MCP_TOKEN` cannot authenticate. Cycles can only
    run through an already-authenticated client (e.g. Claude Code holding the
    MCP connection) rather than through `run_agent.py` standalone. The fix — a
    proper OAuth authorization-code flow with token refresh — is not yet
    implemented.
12. **The MCP beta header was wrong for an unknown period, caught only by a
    real API call.** `MCP_BETA_HEADER` was set to `mcp-client-2025-11-25`,
    which the Anthropic API rejects outright with a 400. All 139 tests passed
    anyway, because every test uses a fake client that never sends a real
    header — the bug was invisible to the entire suite. Fixed to the
    documented `mcp-client-2025-11-20`. A note on the limits of mocked
    testing: a wrong constant that's never exercised against the real API can
    sit behind full test coverage indefinitely.

## Setup

```bash
git clone https://github.com/EUGMsub/RobinhoodAgenticTrader.git
cd RobinhoodAgenticTrader
pip install -r requirements.txt
pytest tests/ -v          # 67 tests, no credentials needed
```

Reproduce the backtest (no credentials, no funded account):

```bash
python scripts/load_bars_local.py
python scripts/run_backtest.py
```

Run the live agent (requires credentials — see `.env.example`):

```bash
cp .env.example .env      # fill in ANTHROPIC_API_KEY, ROBINHOOD_MCP_URL,
                          # ROBINHOOD_MCP_TOKEN, ROBINHOOD_ACCOUNT_NUMBER
python scripts/run_agent.py
```

Access to Robinhood Agentic Trading and a dedicated Agentic account are
required for live use; setup is at
[robinhood.com/us/en/support/agentic-trading](https://robinhood.com/us/en/support/agentic-trading).

## Project structure

```
src/
  config.py         typed configuration - every constraint in one object
  strategy.py       the natural-language strategy prompt
  guardrails.py     pure, unit-tested hard limits + batch validation
  reconcile.py      pure post-execution order verification
  agent.py          orchestration: MCP calls, validation, approval, execution
  backtest.py       pure replay engine - reuses guardrails unchanged
  logging_utils.py  append-only structured audit log
tests/              67 tests, zero network calls
scripts/
  run_agent.py        live agent entry point
  load_bars_local.py  credential-free bar loader (yfinance)
  fetch_bars.py       bar loader via Robinhood MCP
  run_backtest.py     backtest runner + benchmark report
docs/ARCHITECTURE.md
```

## License

MIT — see [LICENSE](LICENSE).
