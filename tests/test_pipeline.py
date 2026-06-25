"""Regression tests for enrich_with_price_data's short-history guards.

A freshly listed/cached ticker with only a handful of price days used to get a
spurious near-zero realized_vol_20d (which both falsely trips the low-vol filter
and, under inverse_vol weighting, explodes its weight ~1/vol) and a meaningless
partial SMA. Each derived column must now be null until enough history exists.
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from screener.engine.pipeline import enrich_with_price_data


def _price_series(ticker: str, n_days: int, start_close: float = 100.0) -> pl.DataFrame:
    base = date(2020, 1, 1)
    rows = []
    for i in range(n_days):
        # mild deterministic wiggle so vol is non-zero when measurable
        close = start_close * (1.0 + 0.01 * ((i % 5) - 2))
        rows.append({"ticker": ticker, "date": base + timedelta(days=i), "close": close})
    return pl.DataFrame(rows)


def test_thin_history_ticker_gets_null_vol_and_sma():
    fundamentals = pl.DataFrame({"ticker": ["FULL", "THIN"]})
    prices = pl.concat([
        _price_series("FULL", 300),   # plenty of history
        _price_series("THIN", 5),     # pathologically short
    ])

    out = enrich_with_price_data(fundamentals, prices)
    by_ticker = {r["ticker"]: r for r in out.to_dicts()}

    # Full-history name: all three derived columns are populated.
    assert by_ticker["FULL"]["realized_vol_20d"] is not None
    assert by_ticker["FULL"]["realized_vol_20d"] > 0
    assert by_ticker["FULL"]["sma_200"] is not None
    assert by_ticker["FULL"]["momentum_12m_return"] is not None

    # Thin-history name: every derived column is null (not a spurious tiny vol).
    assert by_ticker["THIN"]["realized_vol_20d"] is None
    assert by_ticker["THIN"]["sma_200"] is None
    assert by_ticker["THIN"]["momentum_12m_return"] is None


def test_medium_history_vol_measurable_but_sma_guarded():
    """~30 days: enough for a 20-day vol, not for a 200-day SMA or 12m momentum."""
    fundamentals = pl.DataFrame({"ticker": ["MED"]})
    prices = _price_series("MED", 30)

    row = enrich_with_price_data(fundamentals, prices).to_dicts()[0]

    assert row["realized_vol_20d"] is not None and row["realized_vol_20d"] > 0
    assert row["sma_200"] is None          # < 150 days
    assert row["momentum_12m_return"] is None  # < 252 days
