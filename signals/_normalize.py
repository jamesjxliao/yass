from __future__ import annotations

import polars as pl


def minmax(s: pl.Series) -> pl.Series:
    """Normalize a Series to [0, 1] range. Returns 0.5 for constant/empty series."""
    mn, mx = s.min(), s.max()
    if mn is None or mx is None or mn == mx:
        return pl.Series([0.5] * len(s))
    return (s - mn) / (mx - mn)


def column(
    df: pl.DataFrame,
    name: str,
    *,
    invert: bool = False,
    clip: tuple[float, float] | None = None,
    fill: float = 0.0,
) -> pl.Series:
    """A normalized [0, 1] component for a composite signal.

    Reproduces the cast → fill_null → fill_nan → (clip) → minmax → (invert) chain
    that was hand-written in every composite signal, so a new factor component is
    one line and null/normalization handling lives in one place.

    - ``invert=True`` for lower-is-better inputs (e.g. leverage, capex intensity).
    - ``clip`` caps the raw value before normalizing (e.g. R&D intensity at 0.4).
    - ``fill`` is the missing-value default applied before normalizing.
    """
    s = df[name].cast(pl.Float64).fill_null(fill).fill_nan(fill)
    if clip is not None:
        s = s.clip(*clip)
    m = minmax(s)
    return (1.0 - m) if invert else m
