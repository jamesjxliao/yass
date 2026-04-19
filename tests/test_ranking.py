from __future__ import annotations

from pathlib import Path

import polars as pl
from screener.engine.pipeline import ScreeningPipeline
from screener.engine.ranking import compute_composite_score, winsorize, z_score_normalize
from screener.plugins.registry import discover_filters, discover_signals

from tests.test_data.fixtures import make_fundamentals

FILTERS_DIR = Path(__file__).parent.parent / "filters"
SIGNALS_DIR = Path(__file__).parent.parent / "signals"


def test_z_score_basic():
    s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = z_score_normalize(s)
    # Mean should be ~0
    assert abs(z.mean()) < 1e-10
    # Std should be ~1
    assert abs(z.std() - 1.0) < 0.1


def test_z_score_higher_is_better_false():
    s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z_high = z_score_normalize(s, higher_is_better=True)
    z_low = z_score_normalize(s, higher_is_better=False)
    # Should be negated
    assert abs(z_high[0] + z_low[0]) < 1e-10


def test_z_score_nan_handling():
    s = pl.Series([1.0, None, 3.0, 4.0, 5.0])
    z = z_score_normalize(s)
    assert z.null_count() == 0  # No nulls in output


def test_winsorize_clips_outliers():
    s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0])
    w = winsorize(s, n_std=2.0)
    assert w[-1] < 1000.0  # Should be clipped


def test_composite_score_with_real_plugins():
    signals_registry = discover_signals(SIGNALS_DIR)
    signals_with_weights = [
        (signals_registry["momentum_12m"], 0.4),
        (signals_registry["value_composite"], 0.35),
        (signals_registry["quality_score"], 0.25),
    ]
    df = make_fundamentals(10)
    result = compute_composite_score(df, signals_with_weights)

    assert "composite_score" in result.columns
    assert "z_momentum_12m" in result.columns
    assert "z_value_composite" in result.columns
    assert "z_quality_score" in result.columns
    assert len(result) == 10


def test_pipeline_end_to_end():
    filters = list(discover_filters(FILTERS_DIR).values())
    signals_registry = discover_signals(SIGNALS_DIR)
    signals_with_weights = [
        (signals_registry["momentum_12m"], 0.5),
        (signals_registry["value_composite"], 0.5),
    ]

    pipeline = ScreeningPipeline(
        filters=filters,
        signals_with_weights=signals_with_weights,
        top_n=5,
    )

    df = make_fundamentals(20)
    result = pipeline.run(df)

    assert len(result) <= 5
    assert "composite_score" in result.columns
    # Should be sorted descending
    scores = result["composite_score"].to_list()
    assert scores == sorted(scores, reverse=True)


def test_pipeline_sector_cap():
    """Pipeline should limit stocks per sector when max_per_sector is set."""
    signals_registry = discover_signals(SIGNALS_DIR)
    signals_with_weights = [
        (signals_registry["momentum_12m"], 0.5),
        (signals_registry["value_composite"], 0.5),
    ]

    pipeline = ScreeningPipeline(
        filters=[],
        signals_with_weights=signals_with_weights,
        top_n=10,
        max_per_sector=2,
    )

    df = make_fundamentals(20)
    # Add sector column — assign 3 sectors unevenly
    sectors = ["Technology"] * 10 + ["Healthcare"] * 5 + ["Energy"] * 5
    df = df.with_columns(pl.Series("sector", sectors))
    result = pipeline.run(df)

    # Count per sector — none should exceed 2
    sector_counts = result.group_by("sector").len()
    for row in sector_counts.iter_rows(named=True):
        assert row["len"] <= 2, f"Sector {row['sector']} has {row['len']} stocks (max 2)"


def test_pipeline_empty_after_filter():
    """Pipeline should handle case where all stocks are filtered out."""

    class RejectAllFilter:
        name = "reject_all"
        description = "Reject everything"

        def apply(self, df: pl.DataFrame) -> pl.Series:
            return pl.Series([False] * len(df))

    pipeline = ScreeningPipeline(
        filters=[RejectAllFilter()],
        signals_with_weights=[],
        top_n=5,
    )
    df = make_fundamentals(10)
    result = pipeline.run(df)
    assert len(result) == 0
