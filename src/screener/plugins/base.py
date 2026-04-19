from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class Filter(Protocol):
    """Boolean gate: removes rows from the universe.

    Each filter must have a `name` and `description`, and implement `apply()`.
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
    """

    name: str
    description: str
    higher_is_better: bool

    def compute(self, df: pl.DataFrame) -> pl.Series:
        """Return a float Series of the same length as df."""
        ...
