from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import date
from typing import Any

import polars as pl
from dateutil.relativedelta import relativedelta

from screener.backtest.metrics import BacktestMetrics, compute_sharpe
from screener.backtest.pit_server import PITDataServer
from screener.backtest.runner import run_backtest
from screener.backtest.walkforward import (
    WalkForwardConfig,
    generate_folds,
)
from screener.engine.pipeline import ScreeningPipeline, enrich_with_price_data

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    strategy_sharpe: float
    random_sharpes: list[float]
    percentile: float  # 0-100

    def summary(self) -> str:
        avg = sum(self.random_sharpes) / len(self.random_sharpes)
        return (
            f"Strategy Sharpe: {self.strategy_sharpe:.3f}\n"
            f"Random mean Sharpe: {avg:.3f}\n"
            f"Percentile: {self.percentile:.1f}%\n"
            f"{'SIGNIFICANT' if self.percentile >= 95 else 'NOT SIGNIFICANT'} "
            f"(beats {self.percentile:.0f}% of random portfolios)"
        )


@dataclass
class FactorAttributionResult:
    alpha: float  # monthly alpha
    factor_loadings: dict[str, float]  # factor name -> beta
    r_squared: float

    def summary(self) -> str:
        lines = [f"Alpha (monthly): {self.alpha:.4f} ({self.alpha * 12:.2%} annualized)"]
        lines.append(f"R-squared: {self.r_squared:.3f}")
        lines.append("Factor loadings:")
        for name, loading in self.factor_loadings.items():
            lines.append(f"  {name}: {loading:.3f}")
        return "\n".join(lines)


@dataclass
class CorrelationResult:
    matrix: dict[str, dict[str, float]]  # signal_a -> signal_b -> correlation
    high_correlation_pairs: list[tuple[str, str, float]]

    def summary(self) -> str:
        signals = list(self.matrix.keys())
        lines = ["Signal correlations (cross-sectional rank):"]
        header = f"{'':>20}" + "".join(f"{s:>15}" for s in signals)
        lines.append(header)
        for s1 in signals:
            row = f"{s1:>20}"
            for s2 in signals:
                row += f"{self.matrix[s1][s2]:>15.3f}"
            lines.append(row)
        if self.high_correlation_pairs:
            lines.append("\nWARNING: Highly correlated pairs (>0.7):")
            for s1, s2, corr in self.high_correlation_pairs:
                lines.append(f"  {s1} <-> {s2}: {corr:.3f}")
        return "\n".join(lines)


@dataclass
class RegimeResult:
    regimes: dict[str, dict[str, float]]  # regime -> {avg_return, sharpe, n_months}

    def summary(self) -> str:
        lines = [f"{'Regime':<10} {'Months':>8} {'Avg Return':>12} {'Sharpe':>8}"]
        lines.append("-" * 42)
        for regime in ["Bull", "Flat", "Bear"]:
            if regime in self.regimes:
                r = self.regimes[regime]
                lines.append(
                    f"{regime:<10} {r['n_months']:>8.0f} "
                    f"{r['avg_return']:>11.2%} {r['sharpe']:>8.3f}"
                )
        return "\n".join(lines)


@dataclass
class EvaluationReport:
    strategy_metrics: BacktestMetrics
    monte_carlo: MonteCarloResult
    factor_attribution: FactorAttributionResult
    correlation: CorrelationResult
    regime: RegimeResult
    walk_forward_folds: list[dict[str, Any]]
    walk_forward_consistency: float

    def summary(self) -> str:
        sections = [
            "=" * 60,
            "SIGNAL EVALUATION REPORT",
            "=" * 60,
            "",
            "--- Strategy Performance ---",
            self.strategy_metrics.summary(),
            f"Total return: {self.strategy_metrics.total_return:.2%}",
            "",
            "--- Monte Carlo Significance ---",
            self.monte_carlo.summary(),
            "",
            "--- Factor Attribution ---",
            self.factor_attribution.summary(),
            "",
            "--- Signal Correlations ---",
            self.correlation.summary(),
            "",
            "--- Regime Analysis ---",
            self.regime.summary(),
            "",
            f"--- Walk-Forward ({len(self.walk_forward_folds)} folds) ---",
        ]
        for fold in self.walk_forward_folds:
            status = "PASS" if fold["sharpe"] > 0 else "FAIL"
            sections.append(
                f"  {fold['period']:<25} Sharpe: {fold['sharpe']:>7.3f}  {status}"
            )
        sections.append(f"Consistency: {self.walk_forward_consistency:.0%}")
        sections.append("")

        # Overall verdict
        signals = []
        if self.monte_carlo.percentile >= 95:
            signals.append("Monte Carlo: PASS")
        else:
            signals.append("Monte Carlo: FAIL")
        if self.factor_attribution.alpha * 12 > 0.01:
            signals.append("Alpha: POSITIVE")
        else:
            signals.append("Alpha: NEGLIGIBLE")
        if self.walk_forward_consistency >= 0.7:
            signals.append("Walk-forward: CONSISTENT")
        else:
            signals.append("Walk-forward: INCONSISTENT")

        sections.append("--- VERDICT ---")
        sections.extend(signals)

        pass_count = sum(1 for s in signals if "PASS" in s or "POSITIVE" in s or "CONSISTENT" in s)
        if pass_count == 3:
            sections.append("\nOVERALL: STRONG signal — worth paper trading")
        elif pass_count == 2:
            sections.append("\nOVERALL: MODERATE signal — promising but has weaknesses")
        elif pass_count == 1:
            sections.append("\nOVERALL: WEAK signal — likely not worth trading")
        else:
            sections.append("\nOVERALL: NO signal — do not trade")

        return "\n".join(sections)

    def to_json(self) -> dict:
        return {
            "strategy": {
                "sharpe": self.strategy_metrics.sharpe_ratio,
                "cagr": self.strategy_metrics.cagr,
                "max_drawdown": self.strategy_metrics.max_drawdown,
                "total_return": self.strategy_metrics.total_return,
            },
            "monte_carlo": {
                "percentile": self.monte_carlo.percentile,
                "strategy_sharpe": self.monte_carlo.strategy_sharpe,
            },
            "factor_attribution": {
                "alpha_annual": self.factor_attribution.alpha * 12,
                "r_squared": self.factor_attribution.r_squared,
                "loadings": self.factor_attribution.factor_loadings,
            },
            "correlation": {
                "matrix": self.correlation.matrix,
                "high_pairs": [
                    {"a": a, "b": b, "corr": c}
                    for a, b, c in self.correlation.high_correlation_pairs
                ],
            },
            "regime": self.regime.regimes,
            "walk_forward": {
                "consistency": self.walk_forward_consistency,
                "folds": self.walk_forward_folds,
            },
        }


def _generate_monthly_dates(start: date, end: date) -> list[date]:
    dates = []
    current = date(start.year, start.month, 1)
    while current <= end:
        dates.append(current)
        current += relativedelta(months=1)
    return dates


def _run_single_mc_iteration(args: tuple) -> float:
    """Run one Monte Carlo iteration. Designed for parallel execution."""
    seed, top_n, rebalance_dates, tickers_by_date, prices_by_period, cost = args
    rng = random.Random(seed)
    periodic_returns = []

    for i in range(len(rebalance_dates) - 1):
        tickers = tickers_by_date.get(i, [])
        if len(tickers) < top_n:
            periodic_returns.append(0.0)
            continue

        picks = rng.sample(tickers, top_n)
        period_returns = prices_by_period.get(i, {})

        if not period_returns:
            periodic_returns.append(0.0)
            continue

        returns = [period_returns[t] for t in picks if t in period_returns]
        if returns:
            periodic_returns.append(sum(returns) / len(returns) - cost)
        else:
            periodic_returns.append(0.0)

    return compute_sharpe(pl.Series(periodic_returns))


def run_monte_carlo(
    pipeline: ScreeningPipeline,
    pit_server: PITDataServer,
    price_data: pl.DataFrame,
    universe_index: str,
    start_date: date,
    end_date: date,
    n_iterations: int = 500,
    transaction_cost_bps: float = 10.0,
) -> MonteCarloResult:
    """Compare strategy Sharpe against randomly selected portfolios."""
    logger.info("Running Monte Carlo with %d iterations...", n_iterations)

    actual = run_backtest(
        pipeline=pipeline,
        pit_server=pit_server,
        price_data=price_data,
        universe_index=universe_index,
        start_date=start_date,
        end_date=end_date,
        transaction_cost_bps=transaction_cost_bps,
    )

    # Pre-compute data for all periods (avoid repeated filtering in threads)
    rebalance_dates = _generate_monthly_dates(start_date, end_date)
    tickers_by_date: dict[int, list[str]] = {}
    prices_by_period: dict[int, dict[str, float]] = {}

    for i in range(len(rebalance_dates) - 1):
        rebal = rebalance_dates[i]
        next_rebal = rebalance_dates[i + 1]

        tickers_by_date[i] = pit_server.get_tradeable_tickers(rebal)

        period_prices = price_data.filter(
            (pl.col("date") >= rebal) & (pl.col("date") < next_rebal)
        )
        if not period_prices.is_empty():
            tr = (
                period_prices.sort("date")
                .group_by("ticker")
                .agg([
                    pl.col("close").first().alias("s"),
                    pl.col("close").last().alias("e"),
                ])
                .with_columns(((pl.col("e") / pl.col("s")) - 1).alias("r"))
            )
            prices_by_period[i] = dict(zip(
                tr["ticker"].to_list(), tr["r"].to_list()
            ))

    cost = transaction_cost_bps / 10_000 * 2

    # Run iterations in parallel
    from concurrent.futures import ThreadPoolExecutor

    args = [
        (42 + i, pipeline.top_n, rebalance_dates, tickers_by_date,
         prices_by_period, cost)
        for i in range(n_iterations)
    ]

    with ThreadPoolExecutor(max_workers=4) as pool:
        random_sharpes = list(pool.map(_run_single_mc_iteration, args))

    n_below = sum(1 for s in random_sharpes if s < actual.sharpe_ratio)
    percentile = n_below / len(random_sharpes) * 100

    return MonteCarloResult(
        strategy_sharpe=actual.sharpe_ratio,
        random_sharpes=random_sharpes,
        percentile=percentile,
    )


def run_factor_attribution(
    strategy_returns: list[float],
    price_data: pl.DataFrame,
    pit_server: PITDataServer,
    universe_index: str,
    start_date: date,
    end_date: date,
) -> FactorAttributionResult:
    """Regress strategy returns against simple factor portfolios."""
    rebalance_dates = _generate_monthly_dates(start_date, end_date)

    factor_returns: dict[str, list[float]] = {
        "momentum": [],
        "value": [],
        "quality": [],
        "market": [],
    }

    for i in range(len(rebalance_dates) - 1):
        rebal = rebalance_dates[i]
        next_rebal = rebalance_dates[i + 1]

        tickers = pit_server.get_tradeable_tickers(rebal)
        if len(tickers) < 20:
            for f in factor_returns:
                factor_returns[f].append(0.0)
            continue

        # Get price data for sorting
        price_history = price_data.filter(pl.col("date") < rebal)
        period_prices = price_data.filter(
            (pl.col("date") >= rebal) & (pl.col("date") < next_rebal)
        )

        # Compute period returns for all tickers
        all_returns = (
            period_prices.filter(pl.col("ticker").is_in(tickers))
            .sort("date")
            .group_by("ticker")
            .agg([
                pl.col("close").first().alias("s"),
                pl.col("close").last().alias("e"),
            ])
            .with_columns(((pl.col("e") / pl.col("s")) - 1).alias("r"))
        )
        if all_returns.is_empty():
            for f in factor_returns:
                factor_returns[f].append(0.0)
            continue

        # Market return (equal-weight all)
        market_r = float(all_returns["r"].mean() or 0.0)
        factor_returns["market"].append(market_r)

        # Momentum factor: sort by recent returns, long top decile
        momentum_data = (
            price_history.filter(pl.col("ticker").is_in(tickers))
            .sort("date")
            .group_by("ticker")
            .agg([
                pl.col("close").first().alias("old"),
                pl.col("close").last().alias("recent"),
            ])
            .with_columns(((pl.col("recent") / pl.col("old")) - 1).alias("mom"))
            .sort("mom", descending=True)
        )
        n_decile = max(len(momentum_data) // 10, 1)
        mom_picks = momentum_data.head(n_decile)["ticker"].to_list()
        mom_r = all_returns.filter(pl.col("ticker").is_in(mom_picks))
        factor_returns["momentum"].append(float(mom_r["r"].mean() or 0.0))

        # Value + Quality from PIT data
        pit_data = pit_server.get_screening_data(tickers, rebal)
        if pit_data.is_empty() or "ticker" not in pit_data.columns:
            factor_returns["value"].append(market_r)
            factor_returns["quality"].append(market_r)
            continue

        pit_with_returns = pit_data.join(
            all_returns.select("ticker", "r"), on="ticker", how="inner"
        )

        # Value: sort by earnings_yield descending
        if "earnings_yield" in pit_with_returns.columns:
            val_sorted = pit_with_returns.sort("earnings_yield", descending=True, nulls_last=True)
            n_d = max(len(val_sorted) // 10, 1)
            factor_returns["value"].append(float(val_sorted.head(n_d)["r"].mean() or 0.0))
        else:
            factor_returns["value"].append(market_r)

        # Quality: sort by ROE descending
        if "roe" in pit_with_returns.columns:
            qual_sorted = pit_with_returns.sort("roe", descending=True, nulls_last=True)
            n_d = max(len(qual_sorted) // 10, 1)
            factor_returns["quality"].append(float(qual_sorted.head(n_d)["r"].mean() or 0.0))
        else:
            factor_returns["quality"].append(market_r)

    # Simple OLS regression: strategy = alpha + b1*mom + b2*val + b3*qual + b4*mkt
    n = min(len(strategy_returns), len(factor_returns["market"]))
    if n < 5:
        return FactorAttributionResult(alpha=0.0, factor_loadings={}, r_squared=0.0)

    y = strategy_returns[:n]
    X_names = ["momentum", "value", "quality", "market"]
    X = [[factor_returns[f][i] for f in X_names] for i in range(n)]

    # Normal equations: beta = (X'X)^-1 X'y (with intercept)
    alpha, betas, r_sq = _ols_with_intercept(y, X)

    return FactorAttributionResult(
        alpha=alpha,
        factor_loadings=dict(zip(X_names, betas)),
        r_squared=r_sq,
    )


def _ols_with_intercept(
    y: list[float], X: list[list[float]]
) -> tuple[float, list[float], float]:
    """Simple OLS regression with intercept. Returns (alpha, betas, r_squared)."""
    n = len(y)
    k = len(X[0]) if X else 0
    if n < k + 2:
        return 0.0, [0.0] * k, 0.0

    # Add intercept column
    X_aug = [[1.0] + row for row in X]
    k_aug = k + 1

    # X'X
    XtX = [[0.0] * k_aug for _ in range(k_aug)]
    for i in range(k_aug):
        for j in range(k_aug):
            XtX[i][j] = sum(X_aug[r][i] * X_aug[r][j] for r in range(n))

    # X'y
    Xty = [sum(X_aug[r][i] * y[r] for r in range(n)) for i in range(k_aug)]

    # Solve via Gaussian elimination
    try:
        beta = _solve_linear(XtX, Xty)
    except (ZeroDivisionError, ValueError):
        return 0.0, [0.0] * k, 0.0

    # R-squared
    y_mean = sum(y) / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    y_pred = [sum(X_aug[r][j] * beta[j] for j in range(k_aug)) for r in range(n)]
    ss_res = sum((y[r] - y_pred[r]) ** 2 for r in range(n))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return beta[0], beta[1:], max(0.0, r_sq)


def _solve_linear(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve Ax = b via Gaussian elimination."""
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[max_row] = M[max_row], M[col]
        if abs(M[col][col]) < 1e-12:
            raise ValueError("Singular matrix")
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def run_signal_correlation(
    pipeline: ScreeningPipeline,
    pit_server: PITDataServer,
    price_data: pl.DataFrame,
    universe_index: str,
    start_date: date,
    sample_dates: int = 12,
) -> CorrelationResult:
    """Compute average cross-sectional rank correlation between signals."""
    signals = [s for s, _ in pipeline.signals_with_weights]
    signal_names = [s.name for s in signals]

    if len(signals) < 2:
        return CorrelationResult(matrix={}, high_correlation_pairs=[])

    end = start_date + relativedelta(months=sample_dates)
    rebalance_dates = _generate_monthly_dates(start_date, end)
    corr_accum: dict[tuple[str, str], list[float]] = {}

    for rebal in rebalance_dates:
        tickers = pit_server.get_tradeable_tickers(rebal)
        if len(tickers) < 20:
            continue

        screening_data = pit_server.get_screening_data(tickers, rebal)
        if screening_data.is_empty():
            continue

        price_history = price_data.filter(pl.col("date") < rebal)
        screening_data = enrich_with_price_data(screening_data, price_history)

        # Compute each signal's output
        signal_values: dict[str, pl.Series] = {}
        for signal in signals:
            try:
                vals = signal.compute(screening_data)
                signal_values[signal.name] = vals
            except Exception:
                continue

        # Pairwise rank correlations
        for i, s1 in enumerate(signal_names):
            for j, s2 in enumerate(signal_names):
                if i >= j:
                    continue
                if s1 not in signal_values or s2 not in signal_values:
                    continue
                pair = (s1, s2)
                v1 = signal_values[s1].rank()
                v2 = signal_values[s2].rank()
                corr_df = pl.DataFrame({"a": v1, "b": v2})
                corr = corr_df.select(pl.corr("a", "b")).item()
                if corr is not None and not (isinstance(corr, float) and corr != corr):
                    corr_accum.setdefault(pair, []).append(float(corr))

    # Average correlations
    matrix: dict[str, dict[str, float]] = {s: {} for s in signal_names}
    high_pairs = []
    for s in signal_names:
        matrix[s][s] = 1.0

    for (s1, s2), vals in corr_accum.items():
        avg = sum(vals) / len(vals)
        matrix[s1][s2] = avg
        matrix[s2][s1] = avg
        if abs(avg) > 0.7:
            high_pairs.append((s1, s2, avg))

    return CorrelationResult(matrix=matrix, high_correlation_pairs=high_pairs)


def run_regime_analysis(
    strategy_returns: list[float],
    price_data: pl.DataFrame,
    start_date: date,
    end_date: date,
) -> RegimeResult:
    """Analyze strategy performance in bull/flat/bear market regimes."""
    rebalance_dates = _generate_monthly_dates(start_date, end_date)

    regimes: dict[str, list[float]] = {"Bull": [], "Flat": [], "Bear": []}

    for i in range(min(len(rebalance_dates) - 1, len(strategy_returns))):
        rebal = rebalance_dates[i]
        next_rebal = rebalance_dates[i + 1]

        # Compute market return for regime classification
        period = price_data.filter(
            (pl.col("date") >= rebal) & (pl.col("date") < next_rebal)
        )
        if period.is_empty():
            continue

        market_r = (
            period.sort("date")
            .group_by("ticker")
            .agg([
                pl.col("close").first().alias("s"),
                pl.col("close").last().alias("e"),
            ])
            .with_columns(((pl.col("e") / pl.col("s")) - 1).alias("r"))
        )
        mkt_avg = float(market_r["r"].mean() or 0.0)

        if mkt_avg > 0.02:
            regime = "Bull"
        elif mkt_avg < -0.02:
            regime = "Bear"
        else:
            regime = "Flat"

        regimes[regime].append(strategy_returns[i])

    result = {}
    for regime, returns in regimes.items():
        if returns:
            avg_r = sum(returns) / len(returns)
            sharpe = compute_sharpe(pl.Series(returns))
            result[regime] = {
                "n_months": len(returns),
                "avg_return": avg_r,
                "sharpe": sharpe,
            }

    return RegimeResult(regimes=result)


def run_full_evaluation(
    pipeline: ScreeningPipeline,
    pit_server: PITDataServer,
    price_data: pl.DataFrame,
    universe_index: str,
    start_date: date,
    end_date: date,
    monte_carlo_iterations: int = 500,
    transaction_cost_bps: float = 10.0,
    position_stop_loss: float = 0.0,
    hold_bonus: float = 0.0,
) -> EvaluationReport:
    """Run all evaluation analyses and produce a comprehensive report."""
    logger.info("Starting full evaluation...")

    # 1. Strategy backtest
    logger.info("Running strategy backtest...")
    metrics = run_backtest(
        pipeline=pipeline,
        pit_server=pit_server,
        price_data=price_data,
        universe_index=universe_index,
        start_date=start_date,
        end_date=end_date,
        transaction_cost_bps=transaction_cost_bps,
        position_stop_loss=position_stop_loss,
        hold_bonus=hold_bonus,
    )

    # 2. Monte Carlo
    logger.info("Running Monte Carlo significance test...")
    mc = run_monte_carlo(
        pipeline=pipeline,
        pit_server=pit_server,
        price_data=price_data,
        universe_index=universe_index,
        start_date=start_date,
        end_date=end_date,
        n_iterations=monte_carlo_iterations,
        transaction_cost_bps=transaction_cost_bps,
    )

    # 3. Factor attribution
    logger.info("Running factor attribution...")
    fa = run_factor_attribution(
        strategy_returns=metrics.periodic_returns,
        price_data=price_data,
        pit_server=pit_server,
        universe_index=universe_index,
        start_date=start_date,
        end_date=end_date,
    )

    # 4. Signal correlations
    logger.info("Computing signal correlations...")
    corr = run_signal_correlation(
        pipeline=pipeline,
        pit_server=pit_server,
        price_data=price_data,
        universe_index=universe_index,
        start_date=start_date + relativedelta(years=3),  # use middle of data
    )

    # 5. Regime analysis
    logger.info("Running regime analysis...")
    regime = run_regime_analysis(
        strategy_returns=metrics.periodic_returns,
        price_data=price_data,
        start_date=start_date,
        end_date=end_date,
    )

    # 6. Walk-forward
    logger.info("Running walk-forward analysis...")
    wf_config = WalkForwardConfig(
        train_months=36,
        test_months=12,
        step_months=12,
        start_date=start_date,
        end_date=end_date,
    )
    folds = generate_folds(wf_config)
    fold_results = []
    positive = 0
    for ts, te, test_s, test_e in folds:
        m = run_backtest(
            pipeline=pipeline,
            pit_server=pit_server,
            price_data=price_data,
            universe_index=universe_index,
            start_date=test_s,
            end_date=test_e,
            transaction_cost_bps=transaction_cost_bps,
        )
        if m.sharpe_ratio > 0:
            positive += 1
        fold_results.append({
            "period": f"{test_s} - {test_e}",
            "sharpe": m.sharpe_ratio,
            "cagr": m.cagr,
            "max_drawdown": m.max_drawdown,
        })

    wf_consistency = positive / len(folds) if folds else 0.0

    logger.info("Evaluation complete.")
    return EvaluationReport(
        strategy_metrics=metrics,
        monte_carlo=mc,
        factor_attribution=fa,
        correlation=corr,
        regime=regime,
        walk_forward_folds=fold_results,
        walk_forward_consistency=wf_consistency,
    )
