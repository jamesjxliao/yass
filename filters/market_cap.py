from __future__ import annotations

import polars as pl


class MarketCapFilter:
    name = "market_cap_filter"
    description = "Keep stocks above a minimum market cap"

    def __init__(self, min_cap: float = 1_000_000_000):
        self.min_cap = min_cap

    def apply(self, df: pl.DataFrame) -> pl.Series:
        return df["market_cap"] >= self.min_cap
