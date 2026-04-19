from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
from screener.backtest.metrics import (
    compute_max_drawdown,
    compute_metrics_from_returns,
    compute_psr,
    compute_sharpe,
)
from screener.backtest.pit_server import PITDataServer
from screener.backtest.runner import run_backtest
from screener.backtest.walkforward import WalkForwardConfig, generate_folds
from screener.data.cache import CacheManager
from screener.data.mock import MockProvider
from screener.data.universe import UniverseManager
from screener.engine.pipeline import ScreeningPipeline
from screener.plugins.registry import discover_filters, discover_signals

FILTERS_DIR = Path(__file__).parent.parent / "filters"
SIGNALS_DIR = Path(__file__).parent.parent / "signals"


def test_sharpe_ratio():
    # Positive returns with slight variation should have high Sharpe
    returns = pl.Series([0.01, 0.012, 0.009, 0.011, 0.01, 0.013,
                         0.008, 0.011, 0.01, 0.012, 0.009, 0.011])
    sharpe = compute_sharpe(returns, periods_per_year=12)
    assert sharpe > 3.0


def test_sharpe_ratio_zero_std():
    returns = pl.Series([0.0] * 12)
    assert compute_sharpe(returns) == 0.0


def test_max_drawdown():
    # Go up then down
    cumulative = pl.Series([1.0, 1.1, 1.2, 0.9, 1.0])
    dd = compute_max_drawdown(cumulative)
    assert dd < 0  # Negative value
    assert abs(dd - (-0.25)) < 0.01  # 0.9/1.2 - 1 = -0.25


def test_psr():
    # High Sharpe with many observations should have high PSR
    psr = compute_psr(sharpe=2.0, n=120)
    assert psr > 0.95

    # Low Sharpe with few observations should have lower PSR
    psr_low = compute_psr(sharpe=0.3, n=10)
    assert psr_low < psr  # Should be lower than high-Sharpe case


def test_metrics_from_returns():
    returns = pl.Series([0.02, -0.01, 0.03, 0.01, -0.02, 0.015] * 6)
    metrics = compute_metrics_from_returns(returns, periods_per_year=12, n_years=3.0)

    assert metrics.sharpe_ratio != 0.0
    assert metrics.max_drawdown <= 0.0
    assert metrics.sample_size == 36
    assert metrics.is_statistically_significant


def test_metrics_insufficient_data():
    returns = pl.Series([0.01] * 10)
    metrics = compute_metrics_from_returns(returns)
    assert not metrics.is_statistically_significant


def test_walk_forward_folds():
    config = WalkForwardConfig(
        train_months=36,
        test_months=12,
        step_months=12,
        start_date=date(2016, 1, 1),
        end_date=date(2026, 1, 1),
    )
    folds = generate_folds(config)
    # 10 years, 3yr train + 1yr test, step 1yr -> 7 folds
    assert len(folds) == 7

    # First fold
    train_start, train_end, test_start, test_end = folds[0]
    assert train_start == date(2016, 1, 1)
    assert test_end == date(2020, 1, 1)


def test_backtest_with_mock_data():
    """End-to-end backtest using mock provider."""
    cache = CacheManager(":memory:")
    provider = MockProvider()
    um = UniverseManager(cache)
    um.sync(provider, "sp500")

    pit_server = PITDataServer(provider, cache, um)

    filters = list(discover_filters(FILTERS_DIR).values())
    signals_registry = discover_signals(SIGNALS_DIR)
    signals_with_weights = [
        (signals_registry["momentum_12m"], 0.5),
        (signals_registry["value_composite"], 0.5),
    ]

    pipeline = ScreeningPipeline(
        filters=filters,
        signals_with_weights=signals_with_weights,
        top_n=10,
    )

    # Generate mock price data
    tickers = provider.get_universe("sp500")[:10]
    price_data = provider.get_prices(tickers, date(2024, 1, 1), date(2024, 6, 30))

    metrics = run_backtest(
        pipeline=pipeline,
        pit_server=pit_server,
        price_data=price_data,
        universe_index="sp500",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 1),
    )

    assert metrics.sample_size > 0
    assert isinstance(metrics.sharpe_ratio, float)

    cache.close()
