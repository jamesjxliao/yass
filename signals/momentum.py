from __future__ import annotations

import polars as pl


class MomentumSignal:
    name = "momentum_12m"
    description = "12-month price momentum (excluding most recent month)"
    higher_is_better = True

    def compute(self, df: pl.DataFrame) -> pl.Series:
        # Prefer actual momentum computed from price history
        if "momentum_12m_return" in df.columns:
            return df["momentum_12m_return"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)

        # Fallback: revenue_growth proxy (mock data)
        if "revenue_growth" in df.columns:
            return df["revenue_growth"].cast(pl.Float64)

        # Last resort: z-score of close price
        close = df["close"].cast(pl.Float64)
        std = close.std()
        if std is None or std == 0:
            return pl.Series([0.0] * len(df))
        return (close - close.mean()) / std
