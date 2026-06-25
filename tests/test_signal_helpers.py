"""The column() composite-signal primitive: normalize, invert, clip, fill."""
from __future__ import annotations

import polars as pl

from signals._normalize import column


def test_column_minmax_passthrough():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    assert column(df, "x").to_list() == [0.0, 0.5, 1.0]


def test_column_invert():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    assert column(df, "x", invert=True).to_list() == [1.0, 0.5, 0.0]


def test_column_clip_before_normalize():
    df = pl.DataFrame({"x": [1.0, 2.0, 10.0]})
    # clip caps at 3 → [1,2,3] → minmax [0, 0.5, 1]; without clip the 10 would
    # compress the others toward 0.
    assert column(df, "x", clip=(0.0, 3.0)).to_list() == [0.0, 0.5, 1.0]


def test_column_fill_missing():
    df = pl.DataFrame({"x": [2.0, None, 4.0]})
    # null → fill 0.0 (the new minimum) → minmax [0.5, 0, 1].
    assert column(df, "x", fill=0.0).to_list() == [0.5, 0.0, 1.0]
