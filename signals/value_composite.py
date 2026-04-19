from __future__ import annotations

import polars as pl

from signals._normalize import minmax


class ValueCompositeSignal:
    name = "value_composite"
    description = "Composite value factor: earnings yield + FCF yield + EV/Sales"
    higher_is_better = True

    def compute(self, df: pl.DataFrame) -> pl.Series:
        """Average of available value metrics, each normalized to [0, 1] range."""
        components = []

        # Higher is better: earnings_yield, fcf_yield
        for col in ("earnings_yield", "fcf_yield"):
            if col in df.columns:
                s = df[col].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
                components.append(minmax(s))

        # Lower is better: ev_to_sales (invert)
        if "ev_to_sales" in df.columns:
            s = df["ev_to_sales"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            components.append(1.0 - minmax(s))

        # Fallback columns from mock data
        for col in ("book_to_price", "dividend_yield"):
            if col in df.columns:
                s = df[col].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
                components.append(minmax(s))

        if not components:
            return pl.Series([0.5] * len(df))

        stacked = pl.DataFrame({f"c{i}": c for i, c in enumerate(components)})
        return stacked.mean_horizontal()
