from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from dateutil.relativedelta import relativedelta

from screener.backtest.metrics import BacktestMetrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    train_months: int = 36  # 3 years training
    test_months: int = 12  # 1 year testing
    step_months: int = 12  # step forward 1 year
    start_date: date = date(2016, 1, 1)
    end_date: date = date(2026, 1, 1)


@dataclass
class FoldResult:
    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    metrics: BacktestMetrics


def generate_folds(config: WalkForwardConfig) -> list[tuple[date, date, date, date]]:
    """Generate train/test date splits for walk-forward analysis.

    Returns list of (train_start, train_end, test_start, test_end) tuples.
    """
    folds = []
    current = config.start_date

    while True:
        train_start = current
        train_end = train_start + relativedelta(months=config.train_months)
        test_start = train_end
        test_end = test_start + relativedelta(months=config.test_months)

        if test_end > config.end_date:
            break

        folds.append((train_start, train_end, test_start, test_end))
        current = current + relativedelta(months=config.step_months)

    return folds


def compute_walk_forward_consistency(fold_results: list[FoldResult]) -> float:
    """Fraction of folds with positive Sharpe ratio."""
    if not fold_results:
        return 0.0
    positive = sum(1 for f in fold_results if f.metrics.sharpe_ratio > 0)
    return positive / len(fold_results)
