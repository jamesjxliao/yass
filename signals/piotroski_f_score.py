from __future__ import annotations

import polars as pl


class PiotroskiFScoreSignal:
    """Simplified Piotroski F-Score.

    Classic 9-point checklist for financial strength, adapted to
    available FMP data. Scores 0-1 based on how many criteria pass:

    Profitability (from key-metrics):
    1. ROA > 0 (profitable)
    2. ROIC > ROA (leverage working for shareholders)
    3. FCF yield > 0 (generating real cash)
    4. Earnings yield > 0

    Leverage (from key-metrics):
    5. Current ratio > 1 (can pay short-term debts)
    6. Net debt/EBITDA < 2 (not over-leveraged)

    Efficiency:
    7. ROE > sector median (above-average profitability)
    """

    name = "piotroski_f"
    description = "Simplified Piotroski F-Score (financial strength checklist)"
    higher_is_better = True

    def compute(self, df: pl.DataFrame) -> pl.Series:
        n = len(df)
        score = pl.Series([0.0] * n)

        # 1. ROA > 0
        if "roa" in df.columns:
            score = score + (df["roa"].cast(pl.Float64).fill_null(0.0) > 0).cast(pl.Float64)

        # 2. ROIC > ROA (capital efficiency)
        if "roic" in df.columns and "roa" in df.columns:
            roic = df["roic"].cast(pl.Float64).fill_null(0.0)
            roa = df["roa"].cast(pl.Float64).fill_null(0.0)
            score = score + (roic > roa).cast(pl.Float64)

        # 3. FCF yield > 0
        if "fcf_yield" in df.columns:
            score = score + (df["fcf_yield"].cast(pl.Float64).fill_null(0.0) > 0).cast(pl.Float64)

        # 4. Earnings yield > 0
        if "earnings_yield" in df.columns:
            ey = df["earnings_yield"].cast(pl.Float64).fill_null(0.0)
            score = score + (ey > 0).cast(pl.Float64)

        # 5. Current ratio > 1
        if "current_ratio" in df.columns:
            cr = df["current_ratio"].cast(pl.Float64).fill_null(0.0)
            score = score + (cr > 1.0).cast(pl.Float64)

        # 6. Net debt/EBITDA < 2
        if "net_debt_to_ebitda" in df.columns:
            nde = df["net_debt_to_ebitda"].cast(pl.Float64).fill_null(5.0)
            score = score + (nde < 2.0).cast(pl.Float64)

        # 7. ROE > median
        if "roe" in df.columns:
            roe = df["roe"].cast(pl.Float64).fill_null(0.0)
            median_roe = roe.median()
            if median_roe is not None:
                score = score + (roe > median_roe).cast(pl.Float64)

        # Normalize to 0-1 range
        max_possible = 7.0
        return score / max_possible
