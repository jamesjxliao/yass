from __future__ import annotations

import polars as pl


class PriceAboveSMAFilter:
    name = "price_above_sma"
    description = "Keep stocks trading above their 200-day simple moving average"

    def apply(self, df: pl.DataFrame) -> pl.Series:
        if "sma_200" not in df.columns:
            # If SMA not pre-computed, pass all stocks through
            return pl.Series([True] * len(df))
        return df["close"] > df["sma_200"]
