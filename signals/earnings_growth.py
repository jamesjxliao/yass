from __future__ import annotations

import polars as pl

from signals._normalize import minmax


class EarningsGrowthSignal:
    """Earnings growth acceleration.

    Companies whose earnings are growing faster than their historical
    average tend to outperform. This captures the "improving fundamentals"
    effect — not just high ROE (quality), but ROE that's getting better.

    Uses the difference between current earnings_yield and its
    cross-sectional rank to identify stocks with improving profitability
    relative to peers. Combined with ROIC to favor companies that are
    both improving AND efficient with capital.
    """

    name = "earnings_growth"
    description = "Earnings improvement + capital efficiency acceleration"
    higher_is_better = True

    def compute(self, df: pl.DataFrame) -> pl.Series:
        components = []

        # Earnings yield rank (higher = cheaper + more profitable)
        if "earnings_yield" in df.columns:
            s = df["earnings_yield"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            components.append(minmax(s))

        # ROIC improvement proxy: high ROIC relative to ROA suggests
        # efficient capital structure (leverage working for shareholders)
        if "roic" in df.columns and "roa" in df.columns:
            roic = df["roic"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            roa = df["roa"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            spread = (roic - roa).clip(0.0, None)
            components.append(minmax(spread))

        # Current ratio as a safety net — profitable AND liquid
        if "current_ratio" in df.columns:
            s = df["current_ratio"].cast(pl.Float64).fill_null(1.0).fill_nan(1.0)
            # Cap at 3 (above that is hoarding cash, not efficient)
            s = s.clip(0.0, 3.0)
            components.append(minmax(s))

        if not components:
            return pl.Series([0.5] * len(df))

        stacked = pl.DataFrame({f"c{i}": c for i, c in enumerate(components)})
        return stacked.mean_horizontal()
