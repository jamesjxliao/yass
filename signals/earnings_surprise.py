from __future__ import annotations

import polars as pl

from signals._normalize import minmax


class EarningsSurpriseSignal:
    """Post-earnings announcement drift (PEAD) signal.

    Uses Standardized Unexpected Earnings (SUE) from SEC EDGAR.
    SUE = (EPS_q - EPS_same_quarter_last_year) / historical_std.
    Stocks with positive surprises tend to continue outperforming.
    """

    name = "earnings_surprise"
    description = "Post-earnings drift from standardized earnings surprise (SUE)"
    higher_is_better = True

    def compute(self, df: pl.DataFrame) -> pl.Series:
        if "earnings_surprise" not in df.columns:
            return pl.Series([0.5] * len(df))

        sue = df["earnings_surprise"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
        sue = sue.clip(-5.0, 5.0)
        return minmax(sue).fill_null(0.5)
