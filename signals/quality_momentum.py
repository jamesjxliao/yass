from __future__ import annotations

import polars as pl

from signals._normalize import minmax


class QualityMomentumSignal:
    name = "quality_momentum"
    description = "Momentum weighted by quality — stocks trending up with high ROE"
    higher_is_better = True

    def compute(self, df: pl.DataFrame) -> pl.Series:
        if "momentum_12m_return" not in df.columns or "roe" not in df.columns:
            return pl.Series([0.5] * len(df))

        mom = df["momentum_12m_return"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
        roe = df["roe"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)

        mom_n = minmax(mom)
        roe_n = minmax(roe)

        # Blend in ROIC if available for a richer quality measure
        if "roic" in df.columns:
            roic = df["roic"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            roic_n = minmax(roic)
            quality = 0.5 * roe_n + 0.5 * roic_n
        else:
            quality = roe_n

        # Income quality: earnings backed by cash flow
        if "income_quality" in df.columns:
            iq = df["income_quality"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            iq = iq.clip(-3.0, 3.0)
            iq_n = minmax(iq)
            quality = 0.75 * quality + 0.25 * iq_n

        # Geometric mean of momentum and quality
        combo = (mom_n * quality).sqrt()
        return combo.fill_null(0.5).fill_nan(0.5)
