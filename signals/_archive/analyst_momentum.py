from __future__ import annotations

import polars as pl

from signals._normalize import minmax


class AnalystMomentumSignal:
    """Analyst sentiment momentum — rising buy consensus.

    Uses monthly analyst grades data (buy/hold/sell counts) from FMP.
    Signal = current buy ratio + change in buy ratio over 6 months.
    Stocks where analysts are upgrading outperform.
    """

    name = "analyst_momentum"
    description = "Rising analyst buy consensus (grade momentum)"
    higher_is_better = True

    def compute(self, df: pl.DataFrame) -> pl.Series:
        if "analyst_buy_ratio" not in df.columns:
            return pl.Series([0.5] * len(df))

        buy_ratio = (
            df["analyst_buy_ratio"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
        )
        ratio_n = minmax(buy_ratio)

        prior = (
            df["analyst_buy_ratio_prior"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            if "analyst_buy_ratio_prior" in df.columns
            else buy_ratio
        )
        change = (buy_ratio - prior).clip(-0.3, 0.3)
        change_n = minmax(change)
        return (0.6 * ratio_n + 0.4 * change_n).fill_null(0.5)
