from __future__ import annotations

import polars as pl

from signals._normalize import column


class QualitySignal:
    name = "quality_score"
    description = "Quality factor: ROE, low debt, capital efficiency"
    higher_is_better = True

    def compute(self, df: pl.DataFrame) -> pl.Series:
        """Composite quality score from available metrics."""
        components = []

        for col in ("roe", "roic", "roa"):
            if col in df.columns:
                components.append(column(df, col))

        if "net_debt_to_ebitda" in df.columns:
            components.append(column(df, "net_debt_to_ebitda", invert=True))
        elif "debt_to_equity" in df.columns:
            components.append(column(df, "debt_to_equity", invert=True))

        if "rd_to_revenue" in df.columns:
            components.append(column(df, "rd_to_revenue", clip=(0.0, 0.4)))

        if "earnings_stability" in df.columns:
            components.append(column(df, "earnings_stability", fill=0.5))

        if not components:
            return pl.Series([0.5] * len(df))

        stacked = pl.DataFrame({f"c{i}": c for i, c in enumerate(components)})
        return stacked.mean_horizontal()
