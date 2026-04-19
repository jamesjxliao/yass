from __future__ import annotations

import logging
from datetime import date

import polars as pl
from dateutil.relativedelta import relativedelta

from screener.backtest.metrics import BacktestMetrics, compute_metrics_from_returns
from screener.backtest.pit_server import PITDataServer
from screener.backtest.walkforward import (
    FoldResult,
    WalkForwardConfig,
    compute_walk_forward_consistency,
    generate_folds,
)
from screener.engine.pipeline import ScreeningPipeline, enrich_with_price_data

logger = logging.getLogger(__name__)


def _generate_rebalance_dates(
    start: date, end: date, frequency: str = "monthly"
) -> list[date]:
    """Generate rebalance dates between start and end."""
    dates = []
    if frequency == "weekly":
        from datetime import timedelta

        current = start
        # Align to Monday
        while current.weekday() != 0:
            current += timedelta(days=1)
        while current <= end:
            dates.append(current)
            current += timedelta(weeks=1)
    elif frequency == "quarterly":
        current = date(start.year, start.month, 1)
        while current <= end:
            dates.append(current)
            current += relativedelta(months=3)
    else:
        current = date(start.year, start.month, 1)
        while current <= end:
            dates.append(current)
            current += relativedelta(months=1)
    return dates


def _compute_turnover(prev_picks: set[str], new_picks: set[str]) -> float:
    """Fraction of positions that changed between rebalances.

    Returns 1.0 if all positions changed, 0.0 if none changed.
    """
    if not prev_picks and not new_picks:
        return 0.0
    if not prev_picks:
        return 1.0  # First period: all new
    all_positions = prev_picks | new_picks
    changed = len(prev_picks.symmetric_difference(new_picks))
    return changed / len(all_positions) if all_positions else 0.0


def run_backtest(
    pipeline: ScreeningPipeline,
    pit_server: PITDataServer,
    price_data: pl.DataFrame,
    universe_index: str,
    start_date: date,
    end_date: date,
    transaction_cost_bps: float = 10.0,
    frequency: str = "monthly",
    portfolio_stop_loss: float = 0.0,
    momentum_cap: float = 0.0,
    position_stop_loss: float = 0.0,
    hold_bonus: float = 0.0,
) -> BacktestMetrics:
    """Run a backtest with turnover-based transaction costs.

    Optional risk controls:
        portfolio_stop_loss: Go to cash if portfolio drops this much in a month (e.g. 0.15).
        momentum_cap: Exclude stocks with momentum_12m_return above this (e.g. 2.0).
        position_stop_loss: Replace stocks that drop this much from entry (e.g. 0.20).
        hold_bonus: Z-score bonus for current holdings to reduce turnover (e.g. 1.0).
    """
    rebalance_dates = _generate_rebalance_dates(start_date, end_date, frequency)
    periods_per_year = {"weekly": 52, "quarterly": 4}.get(frequency, 12)
    if len(rebalance_dates) < 2:
        logger.warning("Not enough rebalance dates for backtest")
        return compute_metrics_from_returns(pl.Series([], dtype=pl.Float64))

    periodic_returns = []
    prev_picks: set[str] = set()
    in_cash = False  # portfolio stop-loss state

    for i in range(len(rebalance_dates) - 1):
        rebal_date = rebalance_dates[i]
        next_rebal = rebalance_dates[i + 1]

        # Portfolio stop-loss: if last period was a big loss, sit in cash
        if portfolio_stop_loss > 0 and periodic_returns:
            if periodic_returns[-1] < -portfolio_stop_loss:
                in_cash = True
            elif in_cash:
                in_cash = False  # re-enter at next rebalance
        if in_cash:
            periodic_returns.append(0.0)
            prev_picks = set()
            continue

        # Get universe and screening data as of rebalance date
        tickers = pit_server.get_universe_as_of(universe_index, rebal_date)
        if not tickers:
            periodic_returns.append(0.0)
            prev_picks = set()
            continue

        screening_data = pit_server.get_screening_data(tickers, rebal_date)
        if screening_data.is_empty():
            periodic_returns.append(0.0)
            prev_picks = set()
            continue

        # Enrich with price-derived columns (PIT-safe: only data before rebalance)
        price_history = price_data.filter(pl.col("date") < rebal_date)
        screening_data = enrich_with_price_data(screening_data, price_history)

        # Momentum cap: exclude stocks with extreme momentum
        if momentum_cap > 0 and "momentum_12m_return" in screening_data.columns:
            screening_data = screening_data.filter(
                pl.col("momentum_12m_return").is_null()
                | (pl.col("momentum_12m_return") <= momentum_cap)
            )

        # Run pipeline to get top picks
        picks = pipeline.run(
            screening_data,
            hold_bonus_tickers=prev_picks if hold_bonus > 0 else None,
            hold_bonus=hold_bonus,
        )
        if picks.is_empty():
            periodic_returns.append(0.0)
            prev_picks = set()
            continue

        pick_tickers = set(picks["ticker"].to_list())

        # Compute turnover-based transaction cost
        turnover = _compute_turnover(prev_picks, pick_tickers)
        # First period: buy only (no sell), so charge 1x; subsequent: buy+sell = 2x
        cost_multiplier = 1 if not prev_picks else 2
        cost = turnover * (transaction_cost_bps / 10_000) * cost_multiplier
        prev_picks = pick_tickers

        # Compute equal-weight return over holding period
        period_prices = price_data.filter(
            pl.col("ticker").is_in(pick_tickers)
            & (pl.col("date") >= rebal_date)
            & (pl.col("date") < next_rebal)
        )

        if period_prices.is_empty():
            periodic_returns.append(-cost)
            continue

        ticker_returns = (
            period_prices.sort("date")
            .group_by("ticker")
            .agg([
                pl.col("close").first().alias("start_price"),
                pl.col("close").last().alias("end_price"),
            ])
            .with_columns(
                ((pl.col("end_price") / pl.col("start_price")) - 1).alias("return")
            )
        )

        # Apply position stop-loss: cap losses at the stop level
        if position_stop_loss > 0:
            ticker_returns = ticker_returns.with_columns(
                pl.when(pl.col("return") < -position_stop_loss)
                .then(-position_stop_loss)
                .otherwise(pl.col("return"))
                .alias("return")
            )

        avg_return = ticker_returns["return"].mean()
        net_return = (float(avg_return) if avg_return is not None else 0.0) - cost
        periodic_returns.append(net_return)

    returns_series = pl.Series(periodic_returns)
    n_years = (end_date - start_date).days / 365.25

    return compute_metrics_from_returns(
        returns_series,
        periods_per_year=periods_per_year,
        n_years=n_years,
    )


def run_walk_forward(
    pipeline: ScreeningPipeline,
    pit_server: PITDataServer,
    price_data: pl.DataFrame,
    universe_index: str,
    config: WalkForwardConfig,
    transaction_cost_bps: float = 10.0,
    frequency: str = "monthly",
    position_stop_loss: float = 0.0,
    hold_bonus: float = 0.0,
) -> tuple[list[FoldResult], BacktestMetrics]:
    """Run walk-forward analysis across multiple folds."""
    folds = generate_folds(config)
    fold_results = []

    risk_params = {
        "transaction_cost_bps": transaction_cost_bps,
        "frequency": frequency,
        "position_stop_loss": position_stop_loss,
        "hold_bonus": hold_bonus,
    }

    for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
        logger.info(
            "Fold %d: train %s-%s, test %s-%s",
            i, train_start, train_end, test_start, test_end,
        )

        metrics = run_backtest(
            pipeline=pipeline,
            pit_server=pit_server,
            price_data=price_data,
            universe_index=universe_index,
            start_date=test_start,
            end_date=test_end,
            **risk_params,
        )

        fold_results.append(FoldResult(
            fold_index=i,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            metrics=metrics,
        ))

    # Aggregate metrics over full out-of-sample period
    all_test_start = min(f.test_start for f in fold_results)
    all_test_end = max(f.test_end for f in fold_results)
    aggregate = run_backtest(
        pipeline=pipeline,
        pit_server=pit_server,
        price_data=price_data,
        universe_index=universe_index,
        start_date=all_test_start,
        end_date=all_test_end,
        **risk_params,
    )
    aggregate.walk_forward_consistency = compute_walk_forward_consistency(fold_results)

    return fold_results, aggregate
