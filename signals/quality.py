from __future__ import annotations

import polars as pl

from signals._normalize import minmax


class QualitySignal:
    name = "quality_score"
    description = "Quality factor: ROE, low debt, capital efficiency"
    higher_is_better = True

    def compute(self, df: pl.DataFrame) -> pl.Series:
        """Composite quality score from available metrics."""
        components = []

        for col in ("roe", "roic", "roa"):
            if col in df.columns:
                s = df[col].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
                components.append(minmax(s))

        if "net_debt_to_ebitda" in df.columns:
            s = df["net_debt_to_ebitda"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            components.append(1.0 - minmax(s))

        if "debt_to_equity" in df.columns and "net_debt_to_ebitda" not in df.columns:
            s = df["debt_to_equity"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            components.append(1.0 - minmax(s))

        if "rd_to_revenue" in df.columns:
            s = df["rd_to_revenue"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0).clip(0.0, 0.4)
            components.append(minmax(s))

        if "earnings_stability" in df.columns:
            s = df["earnings_stability"].cast(pl.Float64).fill_null(0.5).fill_nan(0.5)
            components.append(minmax(s))

        if not components:
            return pl.Series([0.5] * len(df))

        stacked = pl.DataFrame({f"c{i}": c for i, c in enumerate(components)})
        return stacked.mean_horizontal()
