from __future__ import annotations

import polars as pl

from signals._normalize import column


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
            components.append(column(df, "momentum_12m_return"))

        # Low leverage (lower net_debt_to_ebitda is better — invert)
        if "net_debt_to_ebitda" in df.columns:
            components.append(column(df, "net_debt_to_ebitda", invert=True))

        # Cash generation
        if "fcf_yield" in df.columns:
            components.append(column(df, "fcf_yield"))

        # Low capex intensity (lower is better — asset-light growth)
        if "capex_to_revenue" in df.columns:
            components.append(column(df, "capex_to_revenue", invert=True, clip=(0.0, 0.5)))

        if not components:
            return pl.Series([0.5] * len(df))

        stacked = pl.DataFrame({f"c{i}": c for i, c in enumerate(components)})
        return stacked.mean_horizontal()
