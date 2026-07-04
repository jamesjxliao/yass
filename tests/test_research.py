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


def test_loop_per_variation_budget_uses_absolute_cap(cache: CacheManager):
    """Regression: ResearchLoop's per-variation budget re-check must gate on the
    ABSOLUTE max_experiments, not ``max - var_idx``. Each completed variation
    logs one row, so ``count`` already grows with var_idx; subtracting var_idx
    too double-counts and aborted a fresh experiment at ~max/2 variations.

    This mirrors loop.py's per-iteration gate + log_experiment accounting.
    """
    g = Guardrails(cache)
    max_experiments = 6
    metrics = BacktestMetrics(
        sharpe_ratio=1.0, max_drawdown=-0.1, cagr=0.1,
        calmar_ratio=1.0, sample_size=50, psr=0.9,
        walk_forward_consistency=0.8, total_return=0.3,
    )

    completed = 0
    for var_idx in range(max_experiments):
        # Fixed code: absolute cap (NOT max_experiments - var_idx).
        if not g.check_experiment_budget("exp", max_experiments):
            break
        g.log_experiment(f"id{var_idx}", "exp", {}, metrics)
        completed += 1

    # All variations run — the old `max - var_idx` formula stopped at 3 (~max/2).
    assert completed == max_experiments
    # And the cap is still enforced: a 7th variation is refused.
    assert not g.check_experiment_budget("exp", max_experiments)


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


def test_loop_threads_production_backtest_knobs(cache: CacheManager, tmp_path, monkeypatch):
    """Regression: ResearchLoop must measure every variation under the base
    config's production knobs (hold_bonus, weighting, max_per_sector), not
    run_backtest's defaults (hold_bonus=0.0, weighting='equal', no sector cap).
    Otherwise `run-research` gates candidates on numbers that aren't comparable
    to the production baseline."""
    from datetime import date

    from screener.research import loop as loop_mod
    from screener.research.experiment import (
        ExperimentConfig,
        ExperimentGuardrails,
        ExperimentVariation,
    )
    from screener.research.loop import ResearchLoop

    cfg = tmp_path / "base.yaml"
    cfg.write_text(
        "universe: sp500\ntop_n: 10\nhold_bonus: 1.0\n"
        "weighting: inverse_vol\nmax_per_sector: 3\nfilters: []\n"
        "signals:\n  - name: quality_score\n    weight: 1.0\n"
    )

    captured: list[dict] = []

    def fake_run_backtest(**kwargs):
        captured.append(kwargs)
        return BacktestMetrics(
            sharpe_ratio=1.0, max_drawdown=-0.1, cagr=0.1, calmar_ratio=1.0,
            sample_size=50, psr=0.9, walk_forward_consistency=0.8, total_return=0.3,
        )

    monkeypatch.setattr(loop_mod, "run_backtest", fake_run_backtest)

    exp = ExperimentConfig(
        name="thread_test",
        hypothesis="",
        base_config=str(cfg),
        variations=[ExperimentVariation(
            name="v1", signals=[{"name": "quality_score", "weight": 1.0}])],
        backtest_start=date(2020, 1, 1),
        backtest_end=date(2020, 12, 31),
        rebalance_frequency="monthly",
        top_n=10,
        guardrails=ExperimentGuardrails(
            holdout_start=date(2020, 7, 1), max_experiments=50,
            min_sharpe_threshold=0.0, min_sample_size=1),
    )

    loop = ResearchLoop(cache, pit_server=None, price_data=None,
                        filters_dir=Path("filters"), signals_dir=Path("signals"))
    loop.run(exp)

    assert captured, "run_backtest was never called"
    for call in captured:  # train + holdout both must carry the knobs
        assert call["hold_bonus"] == 1.0
        assert call["weighting"] == "inverse_vol"
        assert call["frequency"] == "monthly"
        assert call["pipeline"].max_per_sector == 3


def test_experiment_config_parsing():
    from screener.research.experiment import ExperimentConfig

    config = ExperimentConfig.from_json(
        Path(__file__).parent.parent / "config" / "experiments" / "example_experiment.json"
    )
    assert config.name == "momentum_value_blend_v1"
    assert len(config.variations) == 2
    assert config.guardrails.max_experiments == 50
