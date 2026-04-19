from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from screener.backtest.metrics import BacktestMetrics
from screener.data.cache import CacheManager

logger = logging.getLogger(__name__)

RESEARCH_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_log (
    experiment_id   VARCHAR PRIMARY KEY,
    experiment_name VARCHAR NOT NULL,
    config_json     JSON NOT NULL,
    started_at      TIMESTAMP NOT NULL,
    completed_at    TIMESTAMP,
    holdout_used    BOOLEAN DEFAULT FALSE,
    metrics_json    JSON,
    status          VARCHAR DEFAULT 'running'
);
"""


class Guardrails:
    def __init__(self, cache: CacheManager):
        self._cache = cache
        self._cache._conn.execute(RESEARCH_LOG_SCHEMA)

    def check_experiment_budget(self, experiment_name: str, max_experiments: int) -> bool:
        """Return True if we haven't exceeded the experiment budget."""
        result = self._cache._conn.execute(
            "SELECT COUNT(*) FROM research_log WHERE experiment_name = ?",
            [experiment_name],
        ).fetchone()
        count = result[0] if result else 0
        if count >= max_experiments:
            logger.warning(
                "Experiment budget exceeded for '%s': %d/%d",
                experiment_name, count, max_experiments,
            )
            return False
        return True

    def check_holdout_usage(self, experiment_name: str) -> int:
        """Return number of times holdout has been used."""
        result = self._cache._conn.execute(
            "SELECT COUNT(*) FROM research_log WHERE experiment_name = ? AND holdout_used = TRUE",
            [experiment_name],
        ).fetchone()
        return result[0] if result else 0

    def validate_results(
        self, metrics: BacktestMetrics, min_sharpe: float = 0.5, min_sample_size: int = 30
    ) -> list[str]:
        """Return list of warnings about the results."""
        warnings = []
        if metrics.sample_size < min_sample_size:
            warnings.append(
                f"Insufficient sample size: {metrics.sample_size} < {min_sample_size}"
            )
        if metrics.sharpe_ratio < min_sharpe:
            warnings.append(
                f"Sharpe ratio below threshold: {metrics.sharpe_ratio:.3f} < {min_sharpe}"
            )
        if metrics.max_drawdown < -0.30:
            warnings.append(
                f"Severe max drawdown: {metrics.max_drawdown:.2%}"
            )
        return warnings

    def log_experiment(
        self,
        experiment_id: str,
        experiment_name: str,
        config: dict,
        metrics: BacktestMetrics | None = None,
        holdout_used: bool = False,
        status: str = "completed",
    ) -> None:
        """Record an experiment run in the research log."""
        now = datetime.now(UTC)
        metrics_json = json.dumps({
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "cagr": metrics.cagr,
            "psr": metrics.psr,
            "sample_size": metrics.sample_size,
            "total_return": metrics.total_return,
        }) if metrics else None

        self._cache._conn.execute(
            """INSERT OR REPLACE INTO research_log
               (experiment_id, experiment_name, config_json, started_at, completed_at,
                holdout_used, metrics_json, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                experiment_id,
                experiment_name,
                json.dumps(config),
                now,
                now,
                holdout_used,
                metrics_json,
                status,
            ],
        )
