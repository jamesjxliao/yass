from __future__ import annotations

from pathlib import Path

import polars as pl
from screener.plugins.base import Filter, Signal
from screener.plugins.registry import discover_filters, discover_signals

from tests.test_data.fixtures import make_fundamentals

FILTERS_DIR = Path(__file__).parent.parent / "filters"
SIGNALS_DIR = Path(__file__).parent.parent / "signals"


def test_discover_filters():
    filters = discover_filters(FILTERS_DIR)
    assert len(filters) == 3
    assert "market_cap_filter" in filters
    assert "volume_filter" in filters
    assert "price_above_sma" in filters


def test_discover_signals():
    signals = discover_signals(SIGNALS_DIR)
    assert len(signals) >= 3
    assert "momentum_12m" in signals
    assert "value_composite" in signals
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


def test_efficiency_acceleration_signal():
    from signals.efficiency_acceleration import EfficiencyAccelerationSignal

    signal = EfficiencyAccelerationSignal()
    df = make_fundamentals(10)
    result = signal.compute(df)
    assert len(result) == 10
    # All values should be in [0, 1] range (normalized)
    assert result.min() >= 0.0  # type: ignore[operator]
    assert result.max() <= 1.0  # type: ignore[operator]
    # With varied inputs, should not be all identical
    assert result.n_unique() > 1


def test_efficiency_acceleration_missing_columns():
    """Signal should gracefully handle missing columns."""
    from signals.efficiency_acceleration import EfficiencyAccelerationSignal

    signal = EfficiencyAccelerationSignal()
    df = pl.DataFrame({"ticker": ["A", "B", "C"], "close": [100.0, 200.0, 300.0]})
    result = signal.compute(df)
    assert len(result) == 3
    # Should return 0.5 fallback
    assert result[0] == 0.5


def test_market_cap_filter_threshold():
    from filters.market_cap import MarketCapFilter

    f = MarketCapFilter(min_cap=5e9)
    df = make_fundamentals(10)
    mask = f.apply(df)
    # market_cap goes from 1e9 to 10e9, so 5e9+ means indices 4-9 (6 stocks)
    assert mask.sum() == 6
