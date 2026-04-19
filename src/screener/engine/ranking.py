from __future__ import annotations

import polars as pl

from screener.plugins.base import Signal


def winsorize(series: pl.Series, n_std: float = 3.0) -> pl.Series:
    """Clip values to within n_std standard deviations of the mean."""
    mean = series.mean()
    std = series.std()
    if std is None or std == 0 or mean is None:
        return series
    lower = mean - n_std * std
    upper = mean + n_std * std
    return series.clip(lower, upper)


def z_score_normalize(series: pl.Series, higher_is_better: bool = True) -> pl.Series:
    """Winsorize then z-score normalize. NaN filled with 0 (neutral)."""
    clipped = winsorize(series)
    mean = clipped.mean()
    std = clipped.std()
    if std is None or std == 0 or mean is None:
        return pl.Series([0.0] * len(series))
    z = (clipped - mean) / std
    if not higher_is_better:
        z = -z
    return z.fill_null(0.0).fill_nan(0.0)


def compute_composite_score(
    df: pl.DataFrame,
    signals_with_weights: list[tuple[Signal, float]],
) -> pl.DataFrame:
    """Compute weighted composite score from multiple signals.

    For each signal: compute raw -> winsorize -> z-score -> multiply by weight.
    Adds individual z-score columns and a 'composite_score' column.

    Returns the input DataFrame with added columns.
    """
    score_columns: dict[str, pl.Series] = {}
    weighted_sum = pl.Series([0.0] * len(df))

    for signal, weight in signals_with_weights:
        raw = signal.compute(df)
        z = z_score_normalize(raw, signal.higher_is_better)
        col_name = f"z_{signal.name}"
        score_columns[col_name] = z
        weighted_sum = weighted_sum + (z * weight)

    new_cols = [s.alias(name) for name, s in score_columns.items()]
    new_cols.append(weighted_sum.alias("composite_score"))
    result = df.with_columns(new_cols)

    return result
