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
    total_swaps: int = 0
    # Benchmark-relative metrics — populated only on paths that have a benchmark
    # series (evaluate / robustness); ``None`` on the benchmark-free fast path
    # (lab / mock / unit tests), so the core stays a pure function of the
    # strategy's own returns. See ``compute_alpha_beta``.
    beta: float | None = None
    alpha_annual: float | None = None

    @property
    def is_statistically_significant(self) -> bool:
        return self.sample_size >= 30

    def summary(self) -> str:
        flag = "" if self.is_statistically_significant else " [INSUFFICIENT DATA]"
        avg_swaps = self.total_swaps / self.sample_size if self.sample_size else 0
        extra = ""
        if self.beta is not None and self.alpha_annual is not None:
            extra = f" | Beta: {self.beta:.2f} | Alpha: {self.alpha_annual:+.1%}"
        return (
            f"Sharpe: {self.sharpe_ratio:.3f} | "
            f"CAGR: {self.cagr:.2%} | "
            f"MaxDD: {self.max_drawdown:.2%} | "
            f"Calmar: {self.calmar_ratio:.3f} | "
            f"PSR: {self.psr:.3f} | "
            f"Periods: {self.sample_size} | "
            f"Swaps: {self.total_swaps} ({avg_swaps:.1f}/mo){extra}{flag}"
        )


@dataclass
class AlphaBeta:
    """CAPM regression of strategy returns on a benchmark (e.g. SPY)."""

    beta: float
    alpha_period: float  # per-period intercept (e.g. monthly)
    alpha_annual: float  # alpha_period * periods_per_year (Jensen convention)
    t_alpha: float  # t-stat of the intercept (H0: alpha == 0)
    r_squared: float
    n: int  # number of aligned periods regressed

    def summary(self) -> str:
        sig = "sig" if abs(self.t_alpha) > 1.98 else "not sig"
        return (
            f"Beta: {self.beta:.3f} | "
            f"Alpha: {self.alpha_annual:+.2%}/yr (t={self.t_alpha:+.2f}, {sig} @95%) | "
            f"R²: {self.r_squared:.3f} | n={self.n}"
        )


def compute_alpha_beta(
    strategy_returns,
    benchmark_returns,
    periods_per_year: int = 12,
    rf_period: float = 0.0,
) -> AlphaBeta | None:
    """OLS CAPM regression: strategy = alpha + beta * benchmark (excess of rf).

    ``strategy_returns`` and ``benchmark_returns`` MUST be aligned period-by-period
    (same holding intervals). Build the benchmark series with
    ``runner.benchmark_period_returns`` so it lines up with the backtest's
    rebalance dates — a calendar-month resample is ~1 period out of phase and
    collapses the regression (R² -> 0, a long-only equity book then spuriously
    shows ~zero beta). ``rf_period`` is the per-period risk-free rate (default 0;
    with beta ~ 1 the Jensen alpha is nearly rf-invariant). Returns ``None`` when
    there is too little data (< 3 periods) or the benchmark has no variance.
    """
    import numpy as np

    y = np.asarray(list(strategy_returns), dtype=float)
    x = np.asarray(list(benchmark_returns), dtype=float)
    k = min(len(y), len(x))
    if k < 3:
        return None
    y = y[:k] - rf_period
    x = x[:k] - rf_period
    mean_x = float(x.mean())
    var_x = float(np.sum((x - mean_x) ** 2))
    # Treat an (effectively) constant benchmark as unregressable. A relative
    # threshold so float round-off on identical inputs (var_x ~ 1e-34, not exactly
    # 0) doesn't slip past into a rank-deficient polyfit.
    if var_x <= 1e-12 * (mean_x**2 * k + 1.0):
        return None
    beta, alpha = (float(v) for v in np.polyfit(x, y, 1))  # deg=1 -> [slope, intercept]
    resid = y - (alpha + beta * x)
    dof = k - 2
    sigma2 = float(np.sum(resid**2)) / dof if dof > 0 else 0.0
    se_alpha = math.sqrt(sigma2 * (1.0 / k + x.mean() ** 2 / var_x)) if sigma2 > 0 else 0.0
    t_alpha = alpha / se_alpha if se_alpha > 0 else 0.0
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else 0.0
    return AlphaBeta(
        beta=beta,
        alpha_period=alpha,
        alpha_annual=alpha * periods_per_year,
        t_alpha=t_alpha,
        r_squared=max(0.0, r_squared),
        n=k,
    )


def periods_per_year(frequency: str) -> int:
    """Annualization factor (rebalance periods per year) for a frequency.

    Single source of truth: a drift between the cadence used to annualize the
    strategy and the one used for the MC null / benchmarks silently corrupts
    Sharpe comparisons (see evaluation.report).
    """
    return {"weekly": 52, "quarterly": 4}.get(frequency, 12)


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
    benchmark: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio: P(true Sharpe > ``benchmark``).

    ``sharpe``, ``skew``, ``kurtosis`` and ``benchmark`` must all be at the SAME
    (per-period, non-annualized) frequency: the variance term mixes ``sharpe``
    with the per-period skew/kurtosis, so feeding an *annualized* Sharpe here
    silently saturates the result toward 1.0 (it inflates the numerator without
    touching the higher-moment terms). ``kurtosis`` is the full, non-excess
    kurtosis (normal == 3). A ``benchmark`` > 0 is the building block for the
    Deflated Sharpe Ratio (see ``evaluation.robustness``).
    """
    if n < 2:
        return 0.0
    # Standard error of the Sharpe estimate (Bailey & López de Prado).
    se = math.sqrt(
        (1 + 0.5 * sharpe**2 - skew * sharpe + ((kurtosis - 3) / 4) * sharpe**2) / (n - 1)
    )
    if se == 0:
        return 0.5
    z = (sharpe - benchmark) / se
    return _normal_cdf(z)


def _normal_cdf(x: float) -> float:
    """Approximation of the standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_metrics_from_returns(
    periodic_returns: pl.Series,
    periods_per_year: int = 12,
    n_years: float | None = None,
    walk_forward_consistency: float = 0.0,
    total_swaps: int = 0,
) -> BacktestMetrics:
    """Compute all backtest metrics from a series of periodic returns."""
    cumulative = (1 + periodic_returns).cum_prod()
    total_return = float(cumulative[-1] - 1) if len(cumulative) > 0 else 0.0

    if n_years is None:
        n_years = len(periodic_returns) / periods_per_year

    sharpe = compute_sharpe(periodic_returns, periods_per_year)
    # Prepend the starting equity (1.0) before measuring drawdown: cum_prod()
    # starts at 1+r[0], so a decline in the opening period(s) — before the first
    # new equity high — would be invisible (peak would start already-down). The
    # total_return above intentionally uses the un-prepended series.
    equity_curve = (
        pl.Series([1.0, *cumulative.to_list()]) if len(cumulative) else cumulative
    )
    max_dd = compute_max_drawdown(equity_curve)
    cagr = compute_cagr(total_return, n_years)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    skew_val = periodic_returns.skew()
    skew = float(skew_val) if skew_val is not None and not math.isnan(skew_val) else 0.0
    kurt_val = periodic_returns.kurtosis()
    kurtosis = (float(kurt_val) if kurt_val is not None and not math.isnan(kurt_val) else 0.0) + 3.0
    # PSR must use the per-period (non-annualized) Sharpe — same frequency as the
    # skew/kurtosis above. Passing the annualized ``sharpe`` mixes frequencies and
    # saturates PSR to ~1.0 for any decent strategy (the old behavior).
    sharpe_per_period = compute_sharpe(periodic_returns, 1)
    psr = compute_psr(sharpe_per_period, len(periodic_returns), skew, kurtosis)

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
        total_swaps=total_swaps,
    )
