---
name: pre-commit-review
description: "Run a correctness and honesty review before committing or pushing in this repo. Use whenever the user says they're about to commit, push, or 'ship it', asks whether changes are ready, or finishes a feature. Checks for this project's recurring failure modes: safety claims enforced only by prompt text, asserts used as guards, empty data rendering as passing checks, fabricated data reaching commits, README claims drifting from reality, purity violations, and vacuous tests. Also use when reviewing a diff or after a Claude Code session that touched src/ or tests/."
---

# Pre-commit review

This repo's credibility rests on two things: safety claims that are actually
enforced, and documentation that matches reality. Both have broken before. Run
this review before any commit that touches `src/`, `tests/`, `scripts/`, or
`README.md`.

Report findings; do not auto-fix. Show the user what's wrong and let them
decide.

## Step 1 — Establish the diff

```bash
git status
git diff --stat
git diff
```

Only review what changed. Note which files are staged vs unstaged.

## Step 2 — The seven checks

### 1. Absence rendered as presence

**This bug class has appeared four times in this project.** It is the default
failure mode here, so check it first.

Look for any code path where missing data produces an affirmative or passing
result: a green banner with zero records, a "no mismatches" message when nothing
executed, a check that returns True on an empty set, a summary that reports "all
clear" without distinguishing "verified clean" from "nothing to verify."

Three states must be distinguishable everywhere: **no data**, **checked and
passing**, **checked and failing**. Two states is a bug.

```bash
grep -rn "no mismatches\|all clear\|No .* recorded\|len(.*) == 0" src/ scripts/
```

### 2. Safety enforced only by prompt text

Grep the strategy prompt for constraints, then verify each one has a
corresponding pure function in `src/guardrails.py` with a test.

```bash
grep -n "never\|must not\|do not\|only if" src/strategy.py
```

For each constraint found, ask: **if the model ignored this sentence, what stops
it?** If the answer is "nothing," that is a bug, not a limitation. Prompt text is
guidance; `guardrails.py` is enforcement.

Also verify no order-placing tool is attached to a proposal or dry-run call:

```bash
grep -n "READ_ONLY_TOOLS\|EXECUTION_TOOLS\|RECONCILE_TOOLS" src/agent.py
```

### 3. `assert` used as a runtime guard

`python -O` strips every `assert` statement, silently deleting the check.
Assertions are fine in tests. They are never acceptable as a safety guard in
`src/`.

```bash
grep -rn "assert " src/ scripts/
```

Any hit in `src/` or `scripts/` that is protecting an invariant must become an
explicit `raise`.

### 4. Fabricated or generated data reaching a commit

Sample logs, demo portfolios, and synthetic fixtures created to preview a UI
must never be committed and must never appear in a README screenshot.

```bash
git diff --cached --name-only
cat .gitignore
```

Confirm `trade_log.jsonl`, `paper_portfolio.json`, `data/`, `dashboard.html`,
`trades.csv`, and `.env` are all gitignored and absent from the staged set. If
any sample data was generated this session, confirm it is labeled as fabricated
and is not staged.

### 5. README drift

The README is the primary artifact a reviewer reads. Every factual claim in it
must be currently true.

```bash
pytest tests/ -q | tail -3
grep -n "tests passing\|tests, no credentials\|137\|Status" README.md
```

Verify:
- the stated test count equals the actual count
- the Status section reflects what has and has not run
- Known Limitations still lists only limitations that still exist, and lists
  every one that does
- any fixed bug has been moved out of Limitations and into Safety design, with
  the flaw and the fix both described

### 6. Purity violations

`src/guardrails.py`, `src/reconcile.py`, `src/paper_trading.py`, and
`src/backtest.py` are pure: no network, no LLM calls, no file I/O.

```bash
grep -n "import anthropic\|requests\|urllib\|open(\|http" src/guardrails.py src/reconcile.py src/paper_trading.py src/backtest.py
```

Also confirm no test makes a network call:

```bash
grep -rn "anthropic.Anthropic()\|requests\.\|urlopen" tests/
```

### 7. Vacuous tests

A test that passes without asserting anything meaningful is worse than no test,
because it creates false confidence in a safety claim.

Read the assertions in any new or modified test. Flag:
- assertions that would pass on an empty collection (`assert all(...)` over an
  empty list, `assert not any(...)` with nothing to iterate)
- tests with no `assert` at all
- tests asserting a value against itself, or against a constant they just set
- brittle assertions on HTML string contents that will break on any template
  edit without indicating a real regression

For each new test, ask: **what change to the source would make this fail?** If
nothing obvious, the test is not pulling weight.

## Step 3 — Design decisions not silently reverted

Check `CLAUDE.md`'s "Design decisions that must not be reverted" list against
the diff. Each entry fixed a real bug. If the diff undoes one, flag it loudly
and name the bug it reintroduces.

## Step 4 — Report

Output a short report:

- **Blockers** — things that must be fixed before committing (safety claims
  without enforcement, asserts as guards, fabricated data staged, false README
  claims)
- **Warnings** — things worth fixing but not blocking (brittle tests, minor
  doc drift)
- **Clean** — checks that passed, listed briefly so the user knows they ran

If everything passes, say so plainly and suggest a commit message describing
what actually changed. Do not pad a clean review with invented concerns.
