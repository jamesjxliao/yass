from __future__ import annotations

import polars as pl


class LowVolatilityFilter:
    """Exclude stocks with abnormally low realized volatility.

    Stocks under pending M&A trade in a tight range anchored to the deal
    price, producing near-zero volatility. These are deal-spread trades,
    not fundamentals picks — the screener's quality signals are irrelevant.
    """

    name = "low_volatility_filter"
    description = "Exclude stocks with collapsed volatility (pending M&A)"

    def __init__(self, min_vol: float = 0.10):
        self.min_vol = min_vol

    def apply(self, df: pl.DataFrame) -> pl.Series:
        if "realized_vol_20d" not in df.columns:
            return pl.Series([True] * len(df))
        return df["realized_vol_20d"].fill_null(1.0) >= self.min_vol
