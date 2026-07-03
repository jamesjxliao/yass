from __future__ import annotations

import polars as pl


class VolumeFilter:
    name = "volume_filter"
    description = "Keep stocks above a minimum average daily volume"

    def __init__(self, min_avg_volume: float = 500_000):
        self.min_avg_volume = min_avg_volume

    def apply(self, df: pl.DataFrame) -> pl.Series:
        # null volume -> null mask -> row dropped by df.filter (fail-closed:
        # unknown liquidity must not pass a liquidity gate)
        return df["avg_volume_20d"] >= self.min_avg_volume
