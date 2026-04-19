from __future__ import annotations

import polars as pl

from signals._normalize import minmax


class LowLeverageGrowthSignal:
    """Companies growing without relying on debt.

    Scores highly for companies with strong momentum, low debt, and high
    cash flow. The idea: companies that grow organically (funded by
    operations) are more sustainable than those funded by borrowing.

    This is a cross-factor signal that doesn't map cleanly to momentum,
    value, or quality alone.
    """

    name = "low_leverage_growth"
    description = "Growth funded by cash flow, not debt"
    higher_is_better = True

    def compute(self, df: pl.DataFrame) -> pl.Series:
        components = []

        # Momentum component (growth)
        if "momentum_12m_return" in df.columns:
            s = df["momentum_12m_return"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            components.append(minmax(s))

        # Low leverage (lower net_debt_to_ebitda is better — invert)
        if "net_debt_to_ebitda" in df.columns:
            s = df["net_debt_to_ebitda"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            components.append(1.0 - minmax(s))

        # Cash generation
        if "fcf_yield" in df.columns:
            s = df["fcf_yield"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            components.append(minmax(s))

        # Low capex intensity (lower is better — asset-light growth)
        if "capex_to_revenue" in df.columns:
            s = df["capex_to_revenue"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            s = s.clip(0.0, 0.5)
            components.append(1.0 - minmax(s))

        if not components:
            return pl.Series([0.5] * len(df))

        stacked = pl.DataFrame({f"c{i}": c for i, c in enumerate(components)})
        return stacked.mean_horizontal()
