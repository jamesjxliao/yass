from __future__ import annotations

import math
from dataclasses import dataclass, field

import polars as pl


@dataclass
class BacktestMetrics:
    sharpe_ratio: float
    max_drawdown: float
    cagr: float
    calmar_ratio: float  # CAGR / |MaxDD|
    sample_size: int  # number of rebalance periods
    psr: float  # Probabilistic Sharpe Ratio
    walk_forward_consistency: float  # % of folds with Sharpe > 0
    total_return: float
    periodic_returns: list[float] = field(default_factory=list)

    @property
    def is_statistically_significant(self) -> bool:
        return self.sample_size >= 30

    def summary(self) -> str:
        flag = "" if self.is_statistically_significant else " [INSUFFICIENT DATA]"
        return (
            f"Sharpe: {self.sharpe_ratio:.3f} | "
            f"CAGR: {self.cagr:.2%} | "
            f"MaxDD: {self.max_drawdown:.2%} | "
            f"Calmar: {self.calmar_ratio:.3f} | "
            f"PSR: {self.psr:.3f} | "
            f"Trades: {self.sample_size}{flag}"
        )


def compute_sharpe(returns: pl.Series, periods_per_year: int = 12) -> float:
    """Annualized Sharpe ratio from periodic returns."""
    mean = returns.mean()
    std = returns.std()
    if std is None or std == 0 or mean is None:
        return 0.0
    return float(mean / std * math.sqrt(periods_per_year))


def compute_max_drawdown(cumulative_returns: pl.Series) -> float:
    """Maximum drawdown from a cumulative returns series (1.0 = start)."""
    peak = cumulative_returns.cum_max()
    drawdown = (cumulative_returns - peak) / peak
    return float(drawdown.min() or 0.0)


def compute_cagr(total_return: float, n_years: float) -> float:
    """Compound annual growth rate."""
    if n_years <= 0 or total_return <= -1:
        return 0.0
    return (1 + total_return) ** (1 / n_years) - 1


def compute_psr(
    sharpe: float,
    n: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio.

    Adjusts observed Sharpe for sample size, skewness, and kurtosis.
    Returns probability that the true Sharpe > 0.
    """
    if n < 2:
        return 0.0
    # Standard error of Sharpe estimate
    se = math.sqrt(
        (1 + 0.5 * sharpe**2 - skew * sharpe + ((kurtosis - 3) / 4) * sharpe**2) / (n - 1)
    )
    if se == 0:
        return 0.5
    # One-sided test: P(true Sharpe > 0)
    z = sharpe / se
    # Approximate normal CDF
    return _normal_cdf(z)


def _normal_cdf(x: float) -> float:
    """Approximation of the standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_metrics_from_returns(
    periodic_returns: pl.Series,
    periods_per_year: int = 12,
    n_years: float | None = None,
    walk_forward_consistency: float = 0.0,
) -> BacktestMetrics:
    """Compute all backtest metrics from a series of periodic returns."""
    cumulative = (1 + periodic_returns).cum_prod()
    total_return = float(cumulative[-1] - 1) if len(cumulative) > 0 else 0.0

    if n_years is None:
        n_years = len(periodic_returns) / periods_per_year

    sharpe = compute_sharpe(periodic_returns, periods_per_year)
    max_dd = compute_max_drawdown(cumulative)
    cagr = compute_cagr(total_return, n_years)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    skew_val = periodic_returns.skew()
    skew = float(skew_val) if skew_val is not None and not math.isnan(skew_val) else 0.0
    kurt_val = periodic_returns.kurtosis()
    kurtosis = (float(kurt_val) if kurt_val is not None and not math.isnan(kurt_val) else 0.0) + 3.0
    psr = compute_psr(sharpe, len(periodic_returns), skew, kurtosis)

    return BacktestMetrics(
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        cagr=cagr,
        calmar_ratio=calmar,
        sample_size=len(periodic_returns),
        psr=psr,
        walk_forward_consistency=walk_forward_consistency,
        total_return=total_return,
        periodic_returns=periodic_returns.to_list(),
    )
