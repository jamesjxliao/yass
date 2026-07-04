"""Broker-agnostic rebalance order logic, shared by every broker integration.

The Alpaca client lives in ``alpaca.py`` (as eToro's does in ``etoro.py`` and
Robinhood's in ``robinhood.py``) — nothing in this module is specific to any
one broker.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RebalanceOrder:
    ticker: str
    side: str  # "buy" or "sell"
    notional: float  # dollar amount
    status: str = "pending"
    trim: bool = False  # True = partial sell (reduce to target), False = full exit
    # Execution-quality instrumentation (populated live by each broker's
    # execute_orders; None on dry runs and until a fill lands). `arrival_price`
    # is the reference/market price captured at order-placement time;
    # `fill_price` is the broker's actual average execution price. The
    # live-vs-backtest tracker (evaluation/tracking.py) uses them to split its
    # per-period gap into timing drift (arrival vs the model's close) and
    # execution slippage/spread (fill vs arrival) — the one real-money leak the
    # backtest structurally cannot see.
    arrival_price: float | None = None
    fill_price: float | None = None


TOLERANCE = 0.05


def target_value_for(
    ticker: str,
    equity: float,
    target_weights: dict[str, float] | None,
    n_tickers: int,
) -> float:
    """Dollar target for one ticker.

    Weighted by ``target_weights[ticker]`` when provided; otherwise equal-weight
    ``equity / n_tickers``. Shared by ``compute_rebalance_orders`` (the order
    preview) and every broker's Phase-2 buy recompute, so the previewed sizing
    and the executed sizing can never diverge.
    """
    if target_weights is not None and ticker in target_weights:
        return equity * target_weights[ticker]
    return equity / n_tickers if n_tickers else 0.0


def weighting_label(target_weights: dict[str, float] | None) -> str:
    """Human-readable sizing mode for logs.

    NB: the production equal-weight path passes a *populated* {ticker: 1/N} dict
    (compute_weights(..., "equal")), NOT None — so a bare truthiness check on
    target_weights mislabels every equal-weight run as "inverse-vol". Detect the
    1/N signature instead.
    """
    if not target_weights:
        return "equal-weight"
    n = len(target_weights)
    if n and all(abs(w - 1.0 / n) < 1e-9 for w in target_weights.values()):
        return "equal-weight"
    return "inverse-vol"


def compute_rebalance_orders(
    target_tickers: list[str],
    equity: float,
    current: dict[str, float],
    target_weights: dict[str, float] | None = None,
) -> list[RebalanceOrder]:
    """Compute rebalance orders with a 5% tolerance band.

    Shared across all brokers. Broker-specific methods fetch equity/positions
    and delegate here.

    target_weights: optional {ticker: weight} (weights sum to ~1) for non-equal
        sizing (e.g. inverse-vol). When None, falls back to equal weight 1/N.
        The per-ticker target value and its tolerance band both scale with the
        ticker's own weight.
    """
    n = len(target_tickers)

    def target_value_of(ticker: str) -> float:
        return target_value_for(ticker, equity, target_weights, n)

    target_set = set(target_tickers)
    current_set = set(current.keys())

    orders: list[RebalanceOrder] = []

    for ticker in current_set - target_set:
        orders.append(RebalanceOrder(
            ticker=ticker, side="sell", notional=current[ticker],
        ))

    for ticker in current_set & target_set:
        target_value = target_value_of(ticker)
        diff = current[ticker] - target_value
        if diff > target_value * TOLERANCE:
            orders.append(RebalanceOrder(
                ticker=ticker, side="sell", notional=diff, trim=True,
            ))

    for ticker in target_set:
        target_value = target_value_of(ticker)
        current_value = current.get(ticker, 0)
        diff = target_value - current_value
        if diff > target_value * TOLERANCE:
            orders.append(RebalanceOrder(
                ticker=ticker, side="buy", notional=diff,
            ))

    return orders
