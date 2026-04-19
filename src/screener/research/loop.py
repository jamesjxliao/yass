from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from screener.backtest.metrics import BacktestMetrics
from screener.backtest.pit_server import PITDataServer
from screener.backtest.runner import run_backtest
from screener.config import PipelineConfig
from screener.data.cache import CacheManager
from screener.engine.pipeline import ScreeningPipeline
from screener.plugins.loader import load_filters, load_signals
from screener.research.experiment import ExperimentConfig
from screener.research.guardrails import Guardrails

logger = logging.getLogger(__name__)


@dataclass
class VariationResult:
    name: str
    metrics: BacktestMetrics
    warnings: list[str]


@dataclass
class ResearchReport:
    experiment_name: str
    hypothesis: str
    results: list[VariationResult]
    best_variation: str | None
    holdout_metrics: BacktestMetrics | None

    def to_json(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "hypothesis": self.hypothesis,
            "best_variation": self.best_variation,
            "results": [
                {
                    "name": r.name,
                    "sharpe": r.metrics.sharpe_ratio,
                    "cagr": r.metrics.cagr,
                    "max_drawdown": r.metrics.max_drawdown,
                    "psr": r.metrics.psr,
                    "sample_size": r.metrics.sample_size,
                    "warnings": r.warnings,
                }
                for r in self.results
            ],
            "holdout": {
                "sharpe": self.holdout_metrics.sharpe_ratio,
                "cagr": self.holdout_metrics.cagr,
                "max_drawdown": self.holdout_metrics.max_drawdown,
            } if self.holdout_metrics else None,
        }

    def summary(self) -> str:
        lines = [
            f"Experiment: {self.experiment_name}",
            f"Hypothesis: {self.hypothesis}",
            "",
        ]
        for r in sorted(self.results, key=lambda x: x.metrics.sharpe_ratio, reverse=True):
            lines.append(f"  [{r.name}] {r.metrics.summary()}")
            for w in r.warnings:
                lines.append(f"    WARNING: {w}")

        if self.best_variation:
            lines.append(f"\nBest variation: {self.best_variation}")
        if self.holdout_metrics:
            lines.append(f"Holdout: {self.holdout_metrics.summary()}")

        return "\n".join(lines)


class ResearchLoop:
    def __init__(
        self,
        cache: CacheManager,
        pit_server: PITDataServer,
        price_data,
        filters_dir: Path,
        signals_dir: Path,
        universe_index: str = "sp500",
    ):
        self._cache = cache
        self._pit_server = pit_server
        self._price_data = price_data
        self._filters_dir = filters_dir
        self._signals_dir = signals_dir
        self._universe_index = universe_index
        self._guardrails = Guardrails(cache)

    def run(self, experiment: ExperimentConfig) -> ResearchReport:
        """Run all variations in an experiment and produce a report."""
        # Check experiment budget
        if not self._guardrails.check_experiment_budget(
            experiment.name, experiment.guardrails.max_experiments
        ):
            raise RuntimeError(
                f"Experiment budget exceeded for '{experiment.name}'"
            )

        # Load base config for default filters
        base_config = PipelineConfig.from_yaml(Path(experiment.base_config))

        results: list[VariationResult] = []

        for variation in experiment.variations:
            logger.info("Running variation: %s", variation.name)
            experiment_id = str(uuid.uuid4())

            # Build pipeline for this variation
            filters_config = variation.filters or base_config.filters
            filters = load_filters(filters_config, self._filters_dir)
            signals_with_weights = load_signals(variation.signals, self._signals_dir)

            pipeline = ScreeningPipeline(
                filters=filters,
                signals_with_weights=signals_with_weights,
                top_n=experiment.top_n,
            )

            # Run backtest (train period only, not holdout)
            metrics = run_backtest(
                pipeline=pipeline,
                pit_server=self._pit_server,
                price_data=self._price_data,
                universe_index=self._universe_index,
                start_date=experiment.backtest_start,
                end_date=experiment.guardrails.holdout_start,
            )

            warnings = self._guardrails.validate_results(
                metrics,
                min_sharpe=experiment.guardrails.min_sharpe_threshold,
                min_sample_size=experiment.guardrails.min_sample_size,
            )

            self._guardrails.log_experiment(
                experiment_id=experiment_id,
                experiment_name=experiment.name,
                config={"variation": variation.name, "signals": variation.signals},
                metrics=metrics,
            )

            results.append(VariationResult(
                name=variation.name,
                metrics=metrics,
                warnings=warnings,
            ))

        # Find best variation (by Sharpe, only among those without warnings)
        clean_results = [r for r in results if not r.warnings]
        best = max(clean_results, key=lambda r: r.metrics.sharpe_ratio) if clean_results else None

        # Run best on holdout (if available and not already used)
        holdout_metrics = None
        if best:
            holdout_uses = self._guardrails.check_holdout_usage(experiment.name)
            if holdout_uses > 0:
                logger.warning(
                    "Holdout already used %d times for '%s', skipping",
                    holdout_uses, experiment.name,
                )
            else:
                logger.info("Running holdout evaluation for '%s'", best.name)
                filters_config = next(
                    (v.filters for v in experiment.variations if v.name == best.name),
                    None,
                ) or base_config.filters
                filters = load_filters(filters_config, self._filters_dir)
                signals_with_weights = load_signals(
                    next(v.signals for v in experiment.variations if v.name == best.name),
                    self._signals_dir,
                )
                pipeline = ScreeningPipeline(
                    filters=filters,
                    signals_with_weights=signals_with_weights,
                    top_n=experiment.top_n,
                )
                holdout_metrics = run_backtest(
                    pipeline=pipeline,
                    pit_server=self._pit_server,
                    price_data=self._price_data,
                    universe_index=self._universe_index,
                    start_date=experiment.guardrails.holdout_start,
                    end_date=experiment.backtest_end,
                )

                self._guardrails.log_experiment(
                    experiment_id=str(uuid.uuid4()),
                    experiment_name=experiment.name,
                    config={"variation": best.name, "holdout": True},
                    metrics=holdout_metrics,
                    holdout_used=True,
                )

        return ResearchReport(
            experiment_name=experiment.name,
            hypothesis=experiment.hypothesis,
            results=results,
            best_variation=best.name if best else None,
            holdout_metrics=holdout_metrics,
        )
