from __future__ import annotations

import polars as pl


class MarketCapFilter:
    name = "market_cap_filter"
    description = "Keep stocks above a minimum market cap"

    def __init__(self, min_cap: float = 1_000_000_000):
        self.min_cap = min_cap

    def apply(self, df: pl.DataFrame) -> pl.Series:
        # null market_cap -> null mask -> row dropped by df.filter (fail-closed:
        # a name of unknown size must not pass a size gate)
        return df["market_cap"] >= self.min_cap
