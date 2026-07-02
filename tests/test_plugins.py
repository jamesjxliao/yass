from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from screener.plugins.base import Filter, Signal
from screener.plugins.loader import load_filters, load_signals
from screener.plugins.registry import discover_filters, discover_signals

from tests.test_data.fixtures import make_fundamentals

FILTERS_DIR = Path(__file__).parent.parent / "filters"
SIGNALS_DIR = Path(__file__).parent.parent / "signals"


def test_load_signals_zero_total_weight_raises():
    """All-zero weights must raise, not silently yield an order-dependent,
    meaningless ranking (#26)."""
    with pytest.raises(ValueError):
        load_signals([{"name": "piotroski_f", "weight": 0.0}], SIGNALS_DIR)


def test_load_filters_unknown_param_raises():
    """A typo'd filter param (e.g. 'minvol' for 'min_vol') must raise, not run
    the filter silently at its default threshold."""
    with pytest.raises(ValueError, match="no parameter 'minvol'"):
        load_filters(
            [{"name": "low_volatility_filter", "params": {"minvol": 0.08}}],
            FILTERS_DIR,
        )


def test_load_filters_valid_param_sets_attribute():
    loaded = load_filters(
        [{"name": "low_volatility_filter", "params": {"min_vol": 0.08}}],
        FILTERS_DIR,
    )
    assert loaded[0].min_vol == 0.08


def test_discover_filters():
    filters = discover_filters(FILTERS_DIR)
    assert len(filters) == 3
    assert "market_cap_filter" in filters
    assert "volume_filter" in filters
    assert "low_volatility_filter" in filters


def test_discover_signals():
    signals = discover_signals(SIGNALS_DIR)
    assert len(signals) >= 3
    assert "momentum_12m" in signals
    assert "low_leverage_growth" in signals
    assert "quality_score" in signals


def test_filter_returns_bool_series():
    filters = discover_filters(FILTERS_DIR)
    df = make_fundamentals(10)
    for name, f in filters.items():
        result = f.apply(df)
        assert isinstance(result, pl.Series), f"Filter {name} did not return a Series"
        assert len(result) == len(df), f"Filter {name} returned wrong length"
        assert result.dtype == pl.Boolean, f"Filter {name} did not return Boolean series"


def test_signal_returns_float_series():
    signals = discover_signals(SIGNALS_DIR)
    df = make_fundamentals(10)
    for name, s in signals.items():
        result = s.compute(df)
        assert isinstance(result, pl.Series), f"Signal {name} did not return a Series"
        assert len(result) == len(df), f"Signal {name} returned wrong length"


def test_filter_protocol_check():
    filters = discover_filters(FILTERS_DIR)
    for f in filters.values():
        assert isinstance(f, Filter)


def test_signal_protocol_check():
    signals = discover_signals(SIGNALS_DIR)
    for s in signals.values():
        assert isinstance(s, Signal)
        assert hasattr(s, "higher_is_better")


def test_signals_handle_missing_columns():
    """EVERY discovered signal must degrade gracefully on a minimal frame —
    live data can lack any field, and consumers are expected to column-guard
    (this generalizes the old efficiency_acceleration-specific test)."""
    signals = discover_signals(SIGNALS_DIR)
    df = pl.DataFrame({"ticker": ["A", "B", "C"], "close": [100.0, 200.0, 300.0]})
    for name, s in signals.items():
        result = s.compute(df)
        assert len(result) == 3, f"Signal {name} returned wrong length"


def test_market_cap_filter_threshold():
    from filters.market_cap import MarketCapFilter

    f = MarketCapFilter(min_cap=5e9)
    df = make_fundamentals(10)
    mask = f.apply(df)
    # market_cap goes from 1e9 to 10e9, so 5e9+ means indices 4-9 (6 stocks)
    assert mask.sum() == 6
