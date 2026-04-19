from __future__ import annotations

from pathlib import Path

from screener.backtest.metrics import BacktestMetrics
from screener.data.cache import CacheManager
from screener.research.guardrails import Guardrails


def test_guardrails_experiment_budget(cache: CacheManager):
    g = Guardrails(cache)
    assert g.check_experiment_budget("test_exp", max_experiments=2)

    # Log two experiments
    metrics = BacktestMetrics(
        sharpe_ratio=1.0, max_drawdown=-0.1, cagr=0.1,
        calmar_ratio=1.0, sample_size=50, psr=0.9,
        walk_forward_consistency=0.8, total_return=0.3,
    )
    g.log_experiment("id1", "test_exp", {}, metrics)
    g.log_experiment("id2", "test_exp", {}, metrics)

    assert not g.check_experiment_budget("test_exp", max_experiments=2)


def test_guardrails_holdout_tracking(cache: CacheManager):
    g = Guardrails(cache)
    assert g.check_holdout_usage("test_exp") == 0

    metrics = BacktestMetrics(
        sharpe_ratio=1.0, max_drawdown=-0.1, cagr=0.1,
        calmar_ratio=1.0, sample_size=50, psr=0.9,
        walk_forward_consistency=0.8, total_return=0.3,
    )
    g.log_experiment("id1", "test_exp", {}, metrics, holdout_used=True)
    assert g.check_holdout_usage("test_exp") == 1


def test_guardrails_validate_results():
    cache = CacheManager(":memory:")
    g = Guardrails(cache)

    # Good results
    good = BacktestMetrics(
        sharpe_ratio=1.5, max_drawdown=-0.15, cagr=0.12,
        calmar_ratio=0.8, sample_size=60, psr=0.95,
        walk_forward_consistency=0.8, total_return=0.4,
    )
    assert g.validate_results(good) == []

    # Bad results
    bad = BacktestMetrics(
        sharpe_ratio=0.1, max_drawdown=-0.45, cagr=0.01,
        calmar_ratio=0.02, sample_size=10, psr=0.3,
        walk_forward_consistency=0.2, total_return=0.02,
    )
    warnings = g.validate_results(bad)
    assert len(warnings) == 3  # low sample, low sharpe, severe drawdown

    cache.close()


def test_experiment_config_parsing():
    from screener.research.experiment import ExperimentConfig

    config = ExperimentConfig.from_json(
        Path(__file__).parent.parent / "config" / "experiments" / "example_experiment.json"
    )
    assert config.name == "momentum_value_blend_v1"
    assert len(config.variations) == 2
    assert config.guardrails.max_experiments == 50
