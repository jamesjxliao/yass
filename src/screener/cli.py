import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import typer

from screener.config import PipelineConfig, Settings

app = typer.Typer(help="Modular Quantitative Research Screener")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _make_provider(settings: Settings):
    """Auto-select provider: CachedFMPProvider if API key set, else MockProvider."""
    from screener.data.cache import CacheManager

    cache = CacheManager(settings.db_path)

    if settings.fmp_api_key:
        from screener.data.fmp import CachedFMPProvider, FMPProvider

        fmp = FMPProvider(settings.fmp_api_key)
        provider = CachedFMPProvider(fmp, cache)
        typer.echo("Using FMP provider (cached)")
    else:
        from screener.data.mock import MockProvider

        provider = MockProvider()
        typer.echo("Using mock provider (set FMP_API_KEY in .env for real data)")

    return provider, cache


def _build_pipeline(pipeline_config: PipelineConfig, settings: Settings, top_n: int | None = None):
    """Build a ScreeningPipeline from config."""
    from screener.engine.pipeline import ScreeningPipeline
    from screener.plugins.loader import load_filters, load_signals

    filters = load_filters(pipeline_config.filters, settings.filters_dir)
    signals = load_signals(pipeline_config.signals, settings.signals_dir)
    return ScreeningPipeline(
        filters=filters,
        signals_with_weights=signals,
        top_n=top_n or pipeline_config.top_n,
        max_per_sector=pipeline_config.max_per_sector,
    )


def _make_pit_server(provider, cache, pipeline_config: PipelineConfig):
    """Create PIT server with universe sync."""
    from screener.backtest.pit_server import PITDataServer
    from screener.data.universe import UniverseManager

    um = UniverseManager(cache)
    um.sync(provider, pipeline_config.universe)
    return PITDataServer(provider, cache, um)


@app.command()
def screen(
    config: Annotated[Path, typer.Option(help="Config YAML")] = Path("config/default.yaml"),
    top_n: Annotated[int, typer.Option(help="Number of top candidates")] = 20,
    output: Annotated[str, typer.Option(help="Output: console, json, csv")] = "console",
    output_path: Annotated[str, typer.Option(help="Output file path")] = "",
    as_of: Annotated[str, typer.Option(help="Screen as of date (YYYY-MM-DD), uses PIT data")] = "",
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Run the screener with the given config and output top N candidates."""
    _setup_logging(verbose)

    from screener.engine.output import to_console, to_csv, to_json
    from screener.engine.pipeline import enrich_with_price_data

    settings = Settings()
    pipeline_config = PipelineConfig.from_yaml(config)
    pipeline = _build_pipeline(pipeline_config, settings, top_n=top_n)

    provider, cache = _make_provider(settings)

    if as_of:
        as_of_date = date.fromisoformat(as_of)
        pit_server = _make_pit_server(provider, cache, pipeline_config)
        tickers = pit_server.get_universe_as_of(
            pipeline_config.universe, as_of_date
        )
        df = pit_server.get_screening_data(tickers, as_of_date)
        prices = cache.get_prices(
            tickers, str(as_of_date - timedelta(days=400)), str(as_of_date)
        )
        df = enrich_with_price_data(df, prices)
        typer.echo(f"Screening as of {as_of_date} ({len(tickers)} tickers)")
    else:
        tickers = provider.get_universe(pipeline_config.universe)
        df = provider.get_fundamentals(tickers)
        yesterday = date.today() - timedelta(days=1)
        prices = provider.get_prices(tickers, yesterday - timedelta(days=400), yesterday)
        df = enrich_with_price_data(df, prices)

    result = pipeline.run(df)

    if output == "json" and output_path:
        to_json(result, Path(output_path))
        typer.echo(f"Results written to {output_path}")
    elif output == "csv" and output_path:
        to_csv(result, Path(output_path))
        typer.echo(f"Results written to {output_path}")
    else:
        typer.echo(to_console(result, top_n))

    cache.close()


@app.command()
def backtest(
    config: Annotated[Path, typer.Option(help="Config YAML")] = Path("config/default.yaml"),
    start_date: Annotated[str, typer.Option(help="Backtest start date")] = "2017-01-01",
    end_date: Annotated[str, typer.Option(help="Backtest end date")] = "2026-01-01",
    top_n: Annotated[int, typer.Option()] = 10,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Run backtest on a screening config."""
    _setup_logging(verbose)

    from screener.backtest.runner import run_backtest

    settings = Settings()
    pipeline_config = PipelineConfig.from_yaml(config)
    pipeline = _build_pipeline(pipeline_config, settings, top_n=top_n)

    provider, cache = _make_provider(settings)
    pit_server = _make_pit_server(provider, cache, pipeline_config)

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    tickers = provider.get_universe(pipeline_config.universe)
    price_data = provider.get_prices(
        tickers, start - timedelta(days=400), end
    )

    metrics = run_backtest(
        pipeline=pipeline,
        pit_server=pit_server,
        price_data=price_data,
        universe_index=pipeline_config.universe,
        start_date=start,
        end_date=end,
        frequency=pipeline_config.rebalance_frequency,
        position_stop_loss=pipeline_config.position_stop_loss,
        hold_bonus=pipeline_config.hold_bonus,
    )

    typer.echo(metrics.summary())
    cache.close()


@app.command("run-research")
def run_research(
    config: Annotated[Path, typer.Argument(help="Experiment config JSON")],
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Execute an AutoResearch experiment loop."""
    _setup_logging(verbose)

    from screener.backtest.pit_server import PITDataServer
    from screener.data.universe import UniverseManager
    from screener.research.experiment import ExperimentConfig
    from screener.research.loop import ResearchLoop

    settings = Settings()
    experiment = ExperimentConfig.from_json(config)

    provider, cache = _make_provider(settings)
    um = UniverseManager(cache)
    um.sync(provider, "sp500")

    pit_server = PITDataServer(provider, cache, um)

    tickers = provider.get_universe("sp500")
    price_data = provider.get_prices(
        tickers, experiment.backtest_start, experiment.backtest_end
    )

    loop = ResearchLoop(
        cache=cache,
        pit_server=pit_server,
        price_data=price_data,
        filters_dir=settings.filters_dir,
        signals_dir=settings.signals_dir,
    )

    report = loop.run(experiment)
    typer.echo(report.summary())

    report_path = Path(f"data/{experiment.name}_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report.to_json(), f, indent=2)
    typer.echo(f"\nJSON report: {report_path}")

    cache.close()


@app.command("fetch-history")
def fetch_history(
    years: Annotated[int, typer.Option(help="Years of history to fetch")] = 10,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Fetch historical data and populate PIT snapshots for backtesting."""
    _setup_logging(verbose)

    from screener.data.fmp import CachedFMPProvider

    settings = Settings()
    provider, cache = _make_provider(settings)

    if not isinstance(provider, CachedFMPProvider):
        typer.echo("fetch-history requires FMP provider. Set FMP_API_KEY in .env")
        raise typer.Exit(1)

    tickers = provider.get_universe("sp500")
    typer.echo(f"Fetching {years}yr history for {len(tickers)} tickers...")

    # Fetch historical data (parallel) then write PIT snapshots (bulk)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Fetch PIT snapshots for all tickers (parallel API, bulk DB write)
    typer.echo("Fetching PIT data from API (parallel)...")
    all_snapshots = []
    done = 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(provider.collect_pit_snapshots, ticker, years): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            all_snapshots.extend(future.result())
            done += 1
            if done % 50 == 0:
                typer.echo(f"  Fetched: {done}/{len(tickers)} ({len(all_snapshots)} snapshots)")

    typer.echo(f"Fetched {len(all_snapshots)} snapshots, writing to DB...")
    cache.bulk_store_pit_snapshots(all_snapshots)
    typer.echo(f"Stored {len(all_snapshots)} PIT snapshots")

    # Fetch historical prices
    start = date.today() - timedelta(days=365 * years)
    yesterday = date.today() - timedelta(days=1)
    typer.echo(f"Fetching prices {start} to {yesterday}...")
    provider.get_prices(tickers, start, yesterday)

    # Warm fundamentals cache so subsequent commands (trade, screen) get cache hits
    typer.echo("Warming fundamentals cache...")
    provider.get_fundamentals(tickers)

    typer.echo("Done. Historical data cached in DuckDB.")
    cache.close()


@app.command()
def evaluate(
    config: Annotated[Path, typer.Option(help="Config YAML")] = Path("config/default.yaml"),
    start_date: Annotated[str, typer.Option(help="Start date")] = "2017-01-01",
    end_date: Annotated[str, typer.Option(help="End date")] = "2026-01-01",
    monte_carlo: Annotated[int, typer.Option(help="Monte Carlo iterations")] = 100,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Run comprehensive signal evaluation: Monte Carlo, factor attribution, correlation, regime."""
    _setup_logging(verbose)

    import polars as pl
    from dateutil.relativedelta import relativedelta

    from screener.evaluation.report import run_full_evaluation

    settings = Settings()
    pipeline_config = PipelineConfig.from_yaml(config)
    pipeline = _build_pipeline(pipeline_config, settings)

    provider, cache = _make_provider(settings)
    pit_server = _make_pit_server(provider, cache, pipeline_config)

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    tickers = provider.get_universe(pipeline_config.universe)
    price_data = cache.get_prices(tickers, str(start - timedelta(days=400)), str(end))

    report = run_full_evaluation(
        pipeline=pipeline,
        pit_server=pit_server,
        price_data=price_data,
        universe_index=pipeline_config.universe,
        start_date=start,
        end_date=end,
        monte_carlo_iterations=monte_carlo,
        position_stop_loss=pipeline_config.position_stop_loss,
        hold_bonus=pipeline_config.hold_bonus,
    )

    typer.echo(report.summary())

    # Create timestamped output folder
    from datetime import datetime

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"results/evaluations/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Charts with SPY and QQQ benchmarks
    from screener.evaluation.charts import (
        plot_equity_curve,
        plot_holdings_timeline,
        plot_monthly_heatmap,
    )

    for etf in ["SPY", "QQQ"]:
        provider.get_prices([etf], start - timedelta(days=30), end)

    all_prices = cache.get_prices(
        tickers + ["SPY", "QQQ"], str(start - timedelta(days=30)), str(end)
    )

    plot_equity_curve(
        report.strategy_metrics.periodic_returns,
        start_date=start,
        output_path=out_dir / "equity_curve.png",
        price_data=all_prices,
        benchmark_tickers=["SPY", "QQQ"],
        metrics={
            "sharpe": report.strategy_metrics.sharpe_ratio,
            "cagr": report.strategy_metrics.cagr,
            "max_drawdown": report.strategy_metrics.max_drawdown,
        },
    )

    plot_monthly_heatmap(
        report.strategy_metrics.periodic_returns,
        start_date=start,
        output_path=out_dir / "monthly_returns.png",
    )

    # Collect holdings at each rebalance for timeline chart
    typer.echo("Generating holdings timeline...")
    from screener.engine.pipeline import enrich_with_price_data

    holdings = []
    curr = date.fromisoformat(start_date)
    while curr < end:
        universe = pit_server.get_tradeable_tickers(curr)
        screening = pit_server.get_screening_data(universe, curr)
        if not screening.is_empty():
            ph = price_data.filter(pl.col("date") < curr)
            enriched = enrich_with_price_data(screening, ph)
            result = pipeline.run(enriched)
            if not result.is_empty():
                holdings.append({
                    "date": curr,
                    "picks": result["ticker"].to_list(),
                })
        curr += relativedelta(months=1)

    plot_holdings_timeline(holdings, out_dir / "holdings_timeline.png")

    with open(out_dir / "report.json", "w") as f:
        json.dump(report.to_json(), f, indent=2)

    # Save summary text
    with open(out_dir / "summary.txt", "w") as f:
        f.write(report.summary())

    typer.echo(f"\nResults saved to {out_dir}/")
    typer.echo("  equity_curve.png")
    typer.echo("  monthly_returns.png")
    typer.echo("  holdings_timeline.png")
    typer.echo("  report.json")
    typer.echo("  summary.txt")

    cache.close()


@app.command()
def lab(
    config: Annotated[Path, typer.Option(help="Config YAML")] = Path("config/default.yaml"),
    start_date: Annotated[str, typer.Option(help="Start date")] = "2017-01-01",
    end_date: Annotated[str, typer.Option(help="End date")] = "2026-01-01",
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Fast signal evaluation for autonomous research. Backtest + metrics only."""
    _setup_logging(verbose)

    from screener.backtest.runner import run_backtest

    settings = Settings()
    pipeline_config = PipelineConfig.from_yaml(config)
    pipeline = _build_pipeline(pipeline_config, settings)

    provider, cache = _make_provider(settings)
    pit_server = _make_pit_server(provider, cache, pipeline_config)

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    tickers = provider.get_universe(pipeline_config.universe)
    price_data = cache.get_prices(tickers, str(start - timedelta(days=400)), str(end))

    metrics = run_backtest(
        pipeline=pipeline,
        pit_server=pit_server,
        price_data=price_data,
        universe_index=pipeline_config.universe,
        start_date=start,
        end_date=end,
        position_stop_loss=pipeline_config.position_stop_loss,
        hold_bonus=pipeline_config.hold_bonus,
    )

    # Machine-readable output for autonomous agent parsing
    typer.echo("---")
    typer.echo(f"sharpe: {metrics.sharpe_ratio:.6f}")
    typer.echo(f"cagr: {metrics.cagr:.6f}")
    typer.echo(f"max_drawdown: {metrics.max_drawdown:.6f}")
    typer.echo(f"calmar: {metrics.calmar_ratio:.6f}")
    typer.echo(f"total_return: {metrics.total_return:.6f}")
    typer.echo(f"sample_size: {metrics.sample_size}")
    typer.echo(f"total_swaps: {metrics.total_swaps}")

    # Benchmark comparison
    import numpy as np
    import polars as pl

    def _bench_metrics(closes: list, n_years: float) -> tuple:
        rets = np.array([(closes[i] / closes[i - 1]) - 1 for i in range(1, len(closes))])
        sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(12)) if np.std(rets) > 0 else 0
        total = (closes[-1] / closes[0]) - 1
        cagr = (1 + total) ** (1 / n_years) - 1
        cum = np.cumprod(1 + rets)
        maxdd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)))
        calmar = cagr / abs(maxdd) if maxdd != 0 else 0
        return sharpe, cagr, maxdd, calmar, total

    n_years = (end - start).days / 365.25
    hdr = f"{'':>10} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8} {'Total':>8}"
    typer.echo(hdr)
    typer.echo(
        f"{'Strategy':>10} {metrics.sharpe_ratio:>8.3f}"
        f" {metrics.cagr:>7.1%} {metrics.max_drawdown:>7.1%}"
        f" {metrics.calmar_ratio:>8.3f} {metrics.total_return:>7.0%}"
    )
    for bench in ["SPY", "QQQ"]:
        bp = cache.get_prices([bench], str(start - timedelta(days=30)), str(end))
        if bp.is_empty():
            continue
        bp = bp.sort("date")
        monthly = bp.group_by_dynamic("date", every="1mo").agg(
            pl.col("close").last()
        ).sort("date")
        closes = monthly["close"].to_list()
        if len(closes) < 2:
            continue
        s, c, d, cal, tot = _bench_metrics(closes, n_years)
        typer.echo(f"{bench:>10} {s:>8.3f} {c:>7.1%} {d:>7.1%} {cal:>8.3f} {tot:>7.0%}")

    cache.close()


@app.command()
def trade(
    config: Annotated[Path, typer.Option(help="Config YAML")] = Path("config/default.yaml"),
    dry_run: Annotated[bool, typer.Option(help="Show orders without executing")] = True,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Rebalance portfolio via Alpaca (paper or live trading)."""
    _setup_logging(verbose)

    from screener.engine.pipeline import enrich_with_price_data
    from screener.trading.broker import AlpacaBroker

    settings = Settings()
    if not settings.alpaca_api_key:
        typer.echo("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        raise typer.Exit(1)

    pipeline_config = PipelineConfig.from_yaml(config)
    # Screen with extra rows so we capture scores for held positions that hold
    # bonus may pull into picks. Take top N for actual picks.
    screen_top_n = max(50, pipeline_config.top_n * 5)
    pipeline = _build_pipeline(pipeline_config, settings, top_n=screen_top_n)

    provider, cache = _make_provider(settings)
    tickers = provider.get_universe(pipeline_config.universe)

    yesterday = date.today() - timedelta(days=1)
    prices = provider.get_prices(tickers, yesterday - timedelta(days=400), yesterday)
    fundamentals = provider.get_fundamentals(tickers)
    enriched = enrich_with_price_data(fundamentals, prices)
    extended_result = pipeline.run(enriched)

    if extended_result.is_empty():
        typer.echo("No stocks passed filters")
        raise typer.Exit(1)

    result = extended_result.head(pipeline_config.top_n)
    picks = result["ticker"].to_list()
    typer.echo(f"Target portfolio ({len(picks)} stocks): {', '.join(picks)}")

    # Connect to Alpaca
    broker = AlpacaBroker(
        settings.alpaca_api_key, settings.alpaca_secret_key,
        paper=settings.alpaca_paper,
    )
    typer.echo(f"Mode: {broker.mode}")

    account = broker.get_account()
    typer.echo(f"Equity: ${account['equity']:,.2f}  Cash: ${account['cash']:,.2f}")

    current = broker.get_positions()
    if current:
        typer.echo(f"Current positions: {', '.join(current.keys())}")
    else:
        typer.echo("No current positions")

    # Compute and display orders
    orders = broker.compute_rebalance_orders(picks, account)

    if not orders:
        typer.echo("Portfolio already balanced — no trades needed")
    else:
        typer.echo(f"\nOrders ({'DRY RUN' if dry_run else 'LIVE'}):")
        for o in orders:
            typer.echo(f"  {o.side.upper():<5} {o.ticker:<6} ${o.notional:>10,.2f}")

        if not dry_run and not settings.alpaca_paper:
            typer.echo("\n⚠️  LIVE TRADING — orders will execute with real money!")

        results = broker.execute_orders(
            orders, dry_run=dry_run,
            stop_loss_pct=pipeline_config.position_stop_loss,
        )

        submitted = sum(1 for o in results if o.status == "submitted")
        errors = sum(1 for o in results if o.status.startswith("error"))
        if not dry_run:
            typer.echo(f"\n{submitted} orders submitted, {errors} errors")

    if dry_run:
        typer.echo("\nThis was a dry run. Use --no-dry-run to execute trades.")

    # Save trade log
    from datetime import datetime

    trade_dir = Path("results/trades")
    trade_dir.mkdir(parents=True, exist_ok=True)
    log_file = trade_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    z_cols = [c for c in extended_result.columns if c.startswith("z_")]
    score_cols = ["ticker", "composite_score", "sector", *z_cols]
    if "company_name" in extended_result.columns:
        score_cols.append("company_name")
    screen_results = extended_result.select(score_cols).to_dicts()

    trade_log = {
        "date": str(date.today()),
        "mode": broker.mode,
        "dry_run": dry_run,
        "account": account,
        "picks": picks,
        "previous_positions": list(current.keys()),
        "orders": [
            {"ticker": o.ticker, "side": o.side,
             "notional": o.notional, "status": o.status}
            for o in (orders if orders else [])
        ],
        "screen_results": screen_results,
    }
    with open(log_file, "w") as f:
        json.dump(trade_log, f, indent=2)
    typer.echo(f"Trade log: {log_file}")

    cache.close()


@app.command("list-plugins")
def list_plugins(
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Show all discovered filters and signals."""
    _setup_logging(verbose)

    from screener.plugins.registry import discover_filters, discover_signals

    settings = Settings()

    typer.echo("Filters:")
    filters = discover_filters(settings.filters_dir)
    for name, f in sorted(filters.items()):
        typer.echo(f"  {name}: {f.description}")

    typer.echo("\nSignals:")
    signals = discover_signals(settings.signals_dir)
    for name, s in sorted(signals.items()):
        direction = "higher is better" if s.higher_is_better else "lower is better"
        typer.echo(f"  {name}: {s.description} ({direction})")


if __name__ == "__main__":
    app()
