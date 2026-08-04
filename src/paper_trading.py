"""
paper_trading.py
=================
A minimal paper-trading ledger for dry_run mode: no real orders, no real
money, just a JSON file on disk that accumulates simulated fills across
successive runs so dry_run behaves like a persistent paper account rather
than resetting every cycle.

load_paper_portfolio() and save_paper_portfolio() are the only I/O in this
module. apply_simulated_buy(), apply_simulated_sell(), and summarize() are
pure — they take a PaperPortfolio and return a new one (or a summary dict)
without touching the filesystem, so they're testable the same way
guardrails.py and backtest.py are.
"""

from dataclasses import dataclass, field
import json
import os


@dataclass
class PaperPosition:
    shares: float
    avg_cost: float
    last_price: float


@dataclass
class PaperPortfolio:
    cash: float
    positions: dict[str, PaperPosition] = field(default_factory=dict)
    realized_pnl: float = 0.0


def load_paper_portfolio(path: str, initial_cash: float) -> PaperPortfolio:
    """Load the paper portfolio from `path`, or start a fresh one seeded
    with `initial_cash` if the file doesn't exist yet (e.g. the first
    dry_run)."""
    if not os.path.exists(path):
        return PaperPortfolio(cash=initial_cash)
    with open(path) as f:
        data = json.load(f)
    return PaperPortfolio(
        cash=data["cash"],
        positions={
            ticker: PaperPosition(**pos)
            for ticker, pos in data.get("positions", {}).items()
        },
        realized_pnl=data.get("realized_pnl", 0.0),
    )


def save_paper_portfolio(path: str, portfolio: PaperPortfolio) -> None:
    data = {
        "cash": portfolio.cash,
        "positions": {
            ticker: {
                "shares": pos.shares,
                "avg_cost": pos.avg_cost,
                "last_price": pos.last_price,
            }
            for ticker, pos in portfolio.positions.items()
        },
        "realized_pnl": portfolio.realized_pnl,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def apply_simulated_buy(
    portfolio: PaperPortfolio, ticker: str, dollars: float, price: float
) -> PaperPortfolio:
    """Return a NEW portfolio with a simulated buy applied — does not
    mutate `portfolio`. Blends into an existing position (weighted-average
    cost) exactly like backtest.py does for a real fill."""
    positions = dict(portfolio.positions)
    shares_bought = dollars / price

    existing = positions.get(ticker)
    if existing:
        total_shares = existing.shares + shares_bought
        avg_cost = (existing.shares * existing.avg_cost + dollars) / total_shares
        positions[ticker] = PaperPosition(
            shares=total_shares, avg_cost=avg_cost, last_price=price
        )
    else:
        positions[ticker] = PaperPosition(
            shares=shares_bought, avg_cost=price, last_price=price
        )

    return PaperPortfolio(
        cash=portfolio.cash - dollars,
        positions=positions,
        realized_pnl=portfolio.realized_pnl,
    )


def apply_simulated_sell(
    portfolio: PaperPortfolio, ticker: str, price: float
) -> PaperPortfolio:
    """Return a NEW portfolio with a simulated full-position sell applied —
    does not mutate `portfolio`. A no-op (returns `portfolio` unchanged) if
    there's no such position, since there's nothing to sell."""
    position = portfolio.positions.get(ticker)
    if position is None:
        return portfolio

    positions = dict(portfolio.positions)
    del positions[ticker]

    proceeds = position.shares * price
    realized = proceeds - position.shares * position.avg_cost

    return PaperPortfolio(
        cash=portfolio.cash + proceeds,
        positions=positions,
        realized_pnl=portfolio.realized_pnl + realized,
    )


def mark_to_market(portfolio: PaperPortfolio, prices: dict[str, float]) -> PaperPortfolio:
    """Return a NEW portfolio with last_price updated to `prices` for every
    held ticker `prices` has a quote for — does not mutate `portfolio`.
    Tickers absent from `prices` keep their existing last_price unchanged
    (e.g. a watchlist ticker whose snapshot failed to recompute this cycle).

    Callers should call this once per cycle with the latest recomputed
    market snapshot, before summarize(), so unrealized P&L reflects current
    prices rather than going stale between fills."""
    positions = dict(portfolio.positions)
    for ticker, price in prices.items():
        existing = positions.get(ticker)
        if existing is not None:
            positions[ticker] = PaperPosition(
                shares=existing.shares, avg_cost=existing.avg_cost, last_price=price
            )
    return PaperPortfolio(
        cash=portfolio.cash, positions=positions, realized_pnl=portfolio.realized_pnl
    )


def summarize(portfolio: PaperPortfolio, initial_cash: float) -> dict:
    """Mark-to-market summary using each position's last_price. That field
    reflects the latest market snapshot only if the caller kept it fresh via
    mark_to_market() this cycle — summarize() itself has no quote feed, so a
    stale last_price (e.g. from a ticker mark_to_market() didn't get a price
    for) produces stale unrealized P&L here too."""
    positions_value = sum(
        pos.shares * pos.last_price for pos in portfolio.positions.values()
    )
    cost_basis = sum(pos.shares * pos.avg_cost for pos in portfolio.positions.values())
    total_equity = portfolio.cash + positions_value
    return {
        "cash": portfolio.cash,
        "positions_value": positions_value,
        "total_equity": total_equity,
        "realized_pnl": portfolio.realized_pnl,
        "unrealized_pnl": positions_value - cost_basis,
        "total_pnl": total_equity - initial_cash,
    }
