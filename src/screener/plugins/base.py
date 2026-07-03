from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class Filter(Protocol):
    """Boolean gate: removes rows from the universe.

    Each filter must have a `name` and `description`, and implement `apply()`.

    Null convention: a null in the mask drops the row (``df.filter`` semantics),
    so a bare comparison against a null field is FAIL-CLOSED — the right default
    for gates on size/liquidity, where "unknown" must not pass. A filter that
    should instead be FAIL-OPEN on missing data (e.g. low_volatility, where a
    missing value is not the pattern being screened for) must say so explicitly
    with ``fill_null``. Missing COLUMN entirely -> return all-True (no-op).
    Every production filter's null polarity is pinned by tests
    (tests/test_plugins.py::TestNullPolarity).
    """

    name: str
    description: str

    def apply(self, df: pl.DataFrame) -> pl.Series:
        """Return a boolean Series. True = keep the row."""
        ...


@runtime_checkable
class Signal(Protocol):
    """Numeric factor: produces a score for ranking.

    Each signal must have a `name`, `description`, and `higher_is_better` flag.

    Null convention: return raw values and PRESERVE nulls/NaN — the ranking
    layer winsorizes, z-scores, and fills null with 0 (the cross-sectional
    neutral). Do not invent a neutral value inside the signal; pre-filling
    hides missing data from the ranking layer and skews the z distribution.
    Missing COLUMN entirely -> return a constant Series (z-scores to all-0
    anyway). Pinned by tests/test_plugins.py::TestNullPolarity.
    """

    name: str
    description: str
    higher_is_better: bool

    def compute(self, df: pl.DataFrame) -> pl.Series:
        """Return a float Series of the same length as df."""
        ...
