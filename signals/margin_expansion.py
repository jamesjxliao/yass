from __future__ import annotations

import polars as pl

from signals._normalize import minmax


class MarginExpansionSignal:
    """Margin expansion: companies with improving profitability."""

    name = "margin_expansion"
    description = "Gross + operating margin year-over-year improvement"
    higher_is_better = True

    def compute(self, df: pl.DataFrame) -> pl.Series:
        components = []

        if "gross_margin_current" in df.columns and "gross_margin_prior" in df.columns:
            curr = df["gross_margin_current"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            prior = df["gross_margin_prior"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            components.append(minmax((curr - prior).clip(-0.2, 0.2)))

        if "op_margin_current" in df.columns and "op_margin_prior" in df.columns:
            curr = df["op_margin_current"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            prior = df["op_margin_prior"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            components.append(minmax((curr - prior).clip(-0.2, 0.2)))

        if "sga_to_revenue" in df.columns and "sga_to_revenue_prior" in df.columns:
            curr = df["sga_to_revenue"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            prior = df["sga_to_revenue_prior"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            components.append(minmax((prior - curr).clip(-0.15, 0.15)))

        if not components:
            return pl.Series([0.5] * len(df))

        stacked = pl.DataFrame({f"c{i}": c for i, c in enumerate(components)})
        return stacked.mean_horizontal()
