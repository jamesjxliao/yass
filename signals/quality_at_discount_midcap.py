from __future__ import annotations

import polars as pl

from signals._normalize import minmax


class QualityAtDiscountMidcapSignal:
    """Mid-cap variant: removes op-margin overlap with margin_expansion."""

    name = "quality_at_discount_midcap"
    description = "High gross margins + negative momentum + strong FCF + low debt"
    higher_is_better = True

    def compute(self, df: pl.DataFrame) -> pl.Series:
        components = []

        # High gross margins (SaaS-like business model)
        if "gross_margin_current" in df.columns:
            s = df["gross_margin_current"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            components.append(minmax(s))

        # Negative momentum = beaten down (invert: more negative = higher score)
        if "momentum_12m_return" in df.columns:
            s = df["momentum_12m_return"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            components.append(1.0 - minmax(s))

        # Strong FCF yield: cash flow turning positive
        if "fcf_yield" in df.columns:
            s = df["fcf_yield"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            components.append(minmax(s))

        # Low debt: not burning cash to survive
        if "net_debt_to_ebitda" in df.columns:
            s = df["net_debt_to_ebitda"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            components.append(1.0 - minmax(s))

        if not components:
            return pl.Series([0.5] * len(df))

        stacked = pl.DataFrame({f"c{i}": c for i, c in enumerate(components)})
        return stacked.mean_horizontal()
