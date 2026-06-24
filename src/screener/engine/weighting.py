"""Position-weighting schemes for portfolio construction.

The screener ranks picks; this module decides how much capital each pick gets.

- ``equal``: classic 1/N. The historical default.
- ``inverse_vol``: size each pick proportional to 1/realized_vol_20d
  (risk-parity-lite). Down-weights the most volatile picks. Validated as a
  Pareto improvement on the production strategy (Sharpe 1.099 -> 1.204, MaxDD
  -23.5% -> -20.3%, identical turnover) and robust across both 2017-21 / 2021-26
  sub-periods. The canonical exponent of 1.0 is used deliberately — stronger
  tilts (1/vol^p, p>1) chase Sharpe but the gain is post-2021 regime luck.

Shared by the backtest runner and the live trade path so both size positions
identically.
"""
from __future__ import annotations

import statistics

import polars as pl

VALID_MODES = ("equal", "inverse_vol")


def compute_weights(picks: pl.DataFrame, mode: str = "equal") -> dict[str, float]:
    """Return {ticker: weight} summing to 1.0 over the picks.

    ``picks`` must have a ``ticker`` column; ``inverse_vol`` also needs
    ``realized_vol_20d``. A name with missing/non-positive vol gets the *median*
    inverse-vol score of the names that do have a usable vol — a neutral weight
    that neither dominates (as 1/0 would) nor is starved (as a flat 1/N share
    would be, since real 1/vol scores are ~5-12 while 1/N is ~0.1). If no name
    has a usable vol, the whole book degrades to equal weight.
    """
    tickers = picks["ticker"].to_list()
    n = len(tickers)
    if n == 0:
        return {}
    if mode == "equal":
        return {t: 1.0 / n for t in tickers}
    if mode != "inverse_vol":
        raise ValueError(f"Unknown weighting mode: {mode!r}. Valid: {VALID_MODES}")

    if "realized_vol_20d" not in picks.columns:
        import logging
        logging.getLogger(__name__).warning(
            "inverse_vol requested but realized_vol_20d is absent (universe-wide "
            "price gap?) — falling back to EQUAL weight; positions are NOT vol-sized."
        )
        return {t: 1.0 / n for t in tickers}

    vols = picks["realized_vol_20d"].to_list()
    scores = {
        t: 1.0 / float(v)
        for t, v in zip(tickers, vols)
        if v is not None and float(v) > 0
    }
    if not scores:
        return {t: 1.0 / n for t in tickers}  # no usable vol anywhere -> equal

    neutral = statistics.median(scores.values())
    raw = {t: scores.get(t, neutral) for t in tickers}
    total = sum(raw.values())
    return {t: w / total for t, w in raw.items()}
