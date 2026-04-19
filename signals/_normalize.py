from __future__ import annotations

import polars as pl


def minmax(s: pl.Series) -> pl.Series:
    """Normalize a Series to [0, 1] range. Returns 0.5 for constant/empty series."""
    mn, mx = s.min(), s.max()
    if mn is None or mx is None or mn == mx:
        return pl.Series([0.5] * len(s))
    return (s - mn) / (mx - mn)
