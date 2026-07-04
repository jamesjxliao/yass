import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import typer

from screener.config import PipelineConfig, Settings
from screener.research.harness import (
    PRICE_LOOKBACK_DAYS,
    backtest_universe,
    build_pipeline,
    make_pit_server,
    make_provider,
)

app = typer.Typer(help="Modular Quantitative Research Screener")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _make_provider(settings: Settings):
    """Build the provider (via the shared harness) and echo the CLI selection."""
    provider, cache = make_provider(settings)
    name = type(provider).__name__
    if "Sharadar" in name:
        typer.echo("Using Sharadar provider (cached)")
    elif "FMP" in name:
        typer.echo("Using FMP provider (cached)")
    else:
        typer.echo(
            "Using mock provider (set NASDAQ_DATA_LINK_API_KEY or FMP_API_KEY "
            "in .env for real data)"
        )
    return provider, cache


# The pipeline/PIT-server/universe setup now lives in screener.research.harness
# (the canonical, CLI-agnostic research API). These aliases preserve the
# historical ``_``-prefixed names that the experiment scripts import.
_build_pipeline = build_pipeline
_make_pit_server = make_pit_server
_backtest_universe = backtest_universe


def _resolve_target_weights(result, weighting: str, picks: list) -> dict:
    """Compute per-ticker target weights for `picks` and echo them if non-equal.

    Shared by the live trade paths (Alpaca/eToro) so their sizing and display
    stay identical. `result` must cover the picks' rows (with `realized_vol_20d`).
    """
    import polars as pl

    from screener.engine.weighting import compute_weights

    target_weights = compute_weights(
        result.filter(pl.col("ticker").is_in(picks)), weighting
    )
    if weighting != "equal":
        typer.echo(f"Weighting: {weighting}")
        for t in picks:
            typer.echo(f"  {t:<6} {target_weights.get(t, 0):>6.1%}")
    return target_weights


def _write_trade_log(
    *,
    broker_mode: str,
    dry_run: bool,
    account: dict,
    picks: list,
    weighting: str,
    target_weights: dict,
    previous_positions: list,
    orders: list,
    extended_result,
    suffix: str = "",
    extra: dict | None = None,
    trade_dir: Path = Path("results/trades"),
) -> Path:
    """Write a rebalance trade log JSON. Shared by every broker's trade command so
    the on-disk schema can't drift between them. ``extra`` injects broker-specific
    top-level keys (e.g. ``{"broker": "etoro"}``) just after ``date``; ``suffix``
    distinguishes filenames (e.g. ``_etoro``).
    """
    from datetime import datetime

    trade_dir.mkdir(parents=True, exist_ok=True)
    log_file = trade_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}.json"

    z_cols = [c for c in extended_result.columns if c.startswith("z_")]
    score_cols = ["ticker", "composite_score", "sector", *z_cols]
    if "company_name" in extended_result.columns:
        score_cols.append("company_name")
    screen_results = extended_result.select(score_cols).to_dicts()

    trade_log = {
        "date": str(date.today()),
        **(extra or {}),
        "mode": broker_mode,
        "dry_run": dry_run,
        "account": account,
        "picks": picks,
        "weighting": weighting,
        "target_weights": target_weights,
        "previous_positions": previous_positions,
        "orders": [
            {"ticker": o.ticker, "side": o.side,
             "notional": o.notional, "status": o.status,
             # Execution-quality fields (None on dry runs / until filled) — the
             # tracker decomposes its gap from these. Kept even when null so the
             # schema is stable and `screener track` can tell "instrumented,
             # unfilled" from "pre-instrumentation log" (key absent).
             "arrival_price": getattr(o, "arrival_price", None),
             "fill_price": getattr(o, "fill_price", None)}
            for o in (orders or [])
        ],
        "screen_results": screen_results,
    }
    with open(log_file, "w") as f:
        json.dump(trade_log, f, indent=2)
    return log_file


def _run_rebalance(
    *,
    broker,
    provider,
    cache,
    pipeline,
    pipeline_config: PipelineConfig,
    dry_run: bool,
    get_holdings,
    is_live_real: bool,
    live_warning: str,
    finalize_picks=None,
    log_suffix: str = "",
    log_extra: dict | None = None,
    trade_dir: Path = Path("results/trades"),
) -> None:
    """Shared rebalance flow for every broker's trade command.

    The brokers diverge in only two places, injected as callbacks:
    - ``get_holdings(enriched) -> {ticker: market_value}`` — Alpaca reads positions
      directly; eToro must screen for candidates first, then resolve_positions.
    - ``finalize_picks(result, picks) -> picks`` — eToro drops picks it can't
      resolve to instrument IDs (and renormalizes weights over the rest); Alpaca
      passes ``None`` (identity).
    Everything else — screening, hold-bonus, weights, order compute, the
    dry-run/execute/report block, and the trade log — is identical, so a change to
    the rebalance contract lands once. The money path (picks → target_weights →
    compute_rebalance_orders → execute_orders) is byte-for-byte the prior logic.
    """
    from screener.engine.pipeline import enrich_with_price_data

    try:
        account = broker.get_account()
        typer.echo(f"Equity: ${account['equity']:,.2f}  Cash: ${account['cash']:,.2f}")

        tickers = provider.get_universe(pipeline_config.universe)
        today = date.today()
        prices = provider.get_prices(
            tickers, today - timedelta(days=PRICE_LOOKBACK_DAYS), today
        )
        fundamentals = provider.get_fundamentals(tickers)
        enriched = enrich_with_price_data(fundamentals, prices)

        current = get_holdings(enriched)
        hold_bonus_tickers = set(current.keys()) if current else None
        if hold_bonus_tickers:
            typer.echo(f"Current positions: {', '.join(sorted(hold_bonus_tickers))}")
        else:
            typer.echo("No current positions")

        extended_result = pipeline.run(
            enriched,
            hold_bonus_tickers=hold_bonus_tickers,
            hold_bonus=pipeline_config.hold_bonus,
        )
        if extended_result.is_empty():
            typer.echo("No stocks passed filters")
            raise typer.Exit(1)

        result = extended_result.head(pipeline_config.top_n)
        picks = result["ticker"].to_list()
        typer.echo(f"Target portfolio ({len(picks)} stocks): {', '.join(picks)}")

        if finalize_picks is not None:
            picks = finalize_picks(result, picks)

        # The screen was non-empty (guarded above), so picks started with top_n
        # names. If finalize_picks dropped every one, the broker could not resolve
        # ANY target instrument — an outage/format change, never a legitimate
        # "hold nothing" signal. Aborting here prevents compute_rebalance_orders([])
        # from emitting a full-exit sell for every held position (whole-account
        # liquidation with no buys).
        if not picks:
            typer.echo(
                "ERROR: no target instruments could be resolved for any pick "
                "(possible broker/market-data outage). Aborting without trading."
            )
            raise typer.Exit(1)

        target_weights = _resolve_target_weights(
            result, pipeline_config.weighting, picks
        )
        orders = broker.compute_rebalance_orders(picks, account, target_weights)

        # Once orders are computed the trade log MUST be written even if
        # execute_orders raises mid-flight (e.g. a 5xx on a post-sell re-read
        # after sells already executed) — otherwise real submitted orders leave
        # no results/trades record to reconcile against the broker. Capture any
        # execution error, always write the log (annotated), then re-raise.
        execution_error: Exception | None = None
        try:
            if not orders:
                typer.echo("Portfolio already balanced — no trades needed")
            else:
                typer.echo(f"\nOrders ({'DRY RUN' if dry_run else 'LIVE'}):")
                for o in orders:
                    typer.echo(f"  {o.side.upper():<5} {o.ticker:<6} ${o.notional:>10,.2f}")

                if not dry_run and is_live_real:
                    typer.echo(f"\n{live_warning}")

                results = broker.execute_orders(
                    orders, dry_run=dry_run,
                    stop_loss_pct=pipeline_config.position_stop_loss,
                    target_weights=target_weights,
                )

                submitted = sum(1 for o in results if o.status == "submitted")
                errors = sum(1 for o in results if o.status.startswith("error"))
                if not dry_run:
                    typer.echo(f"\n{submitted} orders submitted, {errors} errors")

            if dry_run:
                typer.echo("\nThis was a dry run. Use --no-dry-run to execute trades.")
        except Exception as e:  # noqa: BLE001 — record then re-raise below
            execution_error = e
            typer.echo(
                f"\nERROR during execution: {e!r}\n"
                "Writing trade log with the attempted orders before aborting."
            )

        log_extra_final = dict(log_extra or {})
        if execution_error is not None:
            log_extra_final["execution_error"] = repr(execution_error)
        log_file = _write_trade_log(
            broker_mode=broker.mode,
            dry_run=dry_run,
            account=account,
            picks=picks,
            weighting=pipeline_config.weighting,
            target_weights=target_weights,
            previous_positions=list(current.keys()),
            orders=orders,
            extended_result=extended_result,
            suffix=log_suffix,
            extra=log_extra_final,
            trade_dir=trade_dir,
        )
        typer.echo(f"Trade log: {log_file}")
        if execution_error is not None:
            raise execution_error
    finally:
        # Always release the DuckDB handle, even on the empty-screen early exit
        # (raise typer.Exit) — otherwise the cache file stays locked.
        cache.close()


@app.command()
def screen(
    config: Annotated[Path, typer.Option(help="Config YAML")] = Path("config/default.yaml"),
    top_n: Annotated[int, typer.Option(help="Number of top candidates")] = 20,
    output: Annotated[str, typer.Option(help="Output: console, json, csv")] = "console",
    output_path: Annotated[str, typer.Option(help="Output file path")] = "",
    as_of: Annotated[str, typer.Option(help="Screen as of date (YYYY-MM-DD), uses PIT data")] = "",
    hold_bonus_tickers: Annotated[
        str,
        typer.Option(
            help="Comma-separated tickers to apply the config's hold bonus to "
            "(currently-held names). Used by the /rebalance skill to keep picks "
            "stable across accounts. Empty = no hold bonus."
        ),
    ] = "",
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
        # End the price window the day BEFORE as_of_date: cache.get_prices is
        # end-inclusive, so ending at as_of_date would feed that day's own close
        # into momentum/SMA/vol — one day of lookahead vs the backtest's strict
        # `date < rebal` cutoff, so `screen --as-of D` would not reproduce what the
        # backtest screened for rebalance D.
        prices = cache.get_prices(
            tickers,
            str(as_of_date - timedelta(days=400)),
            str(as_of_date - timedelta(days=1)),
        )
        df = enrich_with_price_data(df, prices)
        typer.echo(f"Screening as of {as_of_date} ({len(tickers)} tickers)")
    else:
        tickers = provider.get_universe(pipeline_config.universe)
        df = provider.get_fundamentals(tickers)
        today = date.today()
        prices = provider.get_prices(tickers, today - timedelta(days=400), today)
        df = enrich_with_price_data(df, prices)

    held = {t.strip().upper() for t in hold_bonus_tickers.split(",") if t.strip()}
    result = pipeline.run(
        df,
        hold_bonus_tickers=held or None,
        hold_bonus=pipeline_config.hold_bonus if held else 0.0,
    )

    # Attach the config's intended position weight per pick (equal or inverse_vol),
    # so downstream consumers (rebalance/robinhood skills) size positions to match.
    import polars as pl

    from screener.engine.weighting import compute_weights

    weights = compute_weights(result, pipeline_config.weighting)
    result = result.with_columns(
        pl.col("ticker").replace_strict(weights, default=None).alias("target_weight")
    )

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
    top_n: Annotated[int | None, typer.Option(help="Override config top_n")] = None,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Run backtest on a screening config."""
    _setup_logging(verbose)

    from screener.backtest.runner import run_backtest

    settings = Settings()
    pipeline_config = PipelineConfig.from_yaml(config)
    # Honor the config's top_n unless explicitly overridden on the CLI (matches
    # lab/evaluate; the old hardcoded default of 10 silently overrode the config).
    effective_top_n = top_n if top_n is not None else pipeline_config.top_n
    pipeline = _build_pipeline(pipeline_config, settings, top_n=effective_top_n)

    provider, cache = _make_provider(settings)
    pit_server = _make_pit_server(provider, cache, pipeline_config)

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    tickers = _backtest_universe(cache, provider, pipeline_config.universe)
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
        **pipeline_config.backtest_kwargs(),
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

    tickers = _backtest_universe(cache, provider, "sp500")
    price_data = provider.get_prices(
        # 400-day lookback buffer so momentum_12m/sma_200 aren't null/biased for
        # the first ~12 months of the window (matches lab/backtest/evaluate).
        tickers, experiment.backtest_start - timedelta(days=400),
        experiment.backtest_end,
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


def _fetch_history_sharadar(provider, cache, universe: str, years: int) -> None:
    """Sharadar fetch-history: full membership + PIT + prices + sectors.

    Table-oriented batching (~2 calls per 50 tickers) instead of FMP's ~7 calls
    per ticker — a full 10yr refresh runs in minutes, not half an hour."""
    if universe != "sp500":
        typer.echo("Sharadar provider supports the sp500 universe only.")
        raise typer.Exit(1)

    typer.echo("Syncing survivorship-free sp500 membership (events since 1957)...")
    n = provider.sync_membership()
    typer.echo(f"  {n} membership stints loaded")

    # Every ticker whose stint overlaps the requested window gets PIT + prices,
    # so backtests over the window are survivorship-free out of the box.
    window_start = date.today() - timedelta(days=int(365.25 * years))
    members = cache.to_polars(
        """SELECT DISTINCT ticker FROM universe_membership
           WHERE index_name='sp500'
             AND (removed_date IS NULL OR removed_date >= ?)""",
        [str(window_start)],
    )["ticker"].to_list()
    typer.echo(f"Fetching {years}yr history for {len(members)} members (incl. departed)...")

    typer.echo("Fetching PIT fundamentals (batched)...")
    snaps = provider.collect_pit_snapshots_bulk(
        members, years,
        progress=lambda i, n: typer.echo(f"  batch {i}/{n}") if i % 6 == 0 else None,
    )
    cache.bulk_store_pit_snapshots(snaps)
    typer.echo(f"Stored {len(snaps)} PIT snapshots")

    yesterday = date.today() - timedelta(days=1)
    typer.echo(f"Fetching prices {window_start} to {yesterday} (incremental)...")
    provider.get_prices(members, window_start, yesterday)
    typer.echo("Fetching benchmark ETF prices (SPY, QQQ)...")
    provider.get_prices(["SPY", "QQQ"], window_start, yesterday)

    typer.echo("Storing sectors...")
    provider.store_sectors(members)

    typer.echo("Warming fundamentals cache...")
    current = provider.get_universe(universe)
    provider.get_fundamentals(current)

    from screener.data.staleness import find_stale_quarters

    stale = find_stale_quarters(cache)
    if stale:
        top = ", ".join(f"{s.ticker}(+{s.quarters_behind}Q)" for s in stale[:8])
        typer.echo(f"⚠ PIT freshness: {len(stale)} ticker(s) overdue: {top}"
                   f"{' ...' if len(stale) > 8 else ''}")
    else:
        typer.echo("PIT freshness: all tickers current.")
    typer.echo("Done. Historical data cached in DuckDB.")
    cache.close()


@app.command("fetch-history")
def fetch_history(
    universe: Annotated[str, typer.Option(help="Universe to fetch (sp500, sp400)")] = "sp500",
    years: Annotated[int, typer.Option(help="Years of history to fetch")] = 10,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Fetch historical data and populate PIT snapshots for backtesting."""
    _setup_logging(verbose)

    from screener.data.fmp import CachedFMPProvider
    from screener.data.sharadar import CachedSharadarProvider

    settings = Settings()
    provider, cache = _make_provider(settings)

    if isinstance(provider, CachedSharadarProvider):
        _fetch_history_sharadar(provider, cache, universe, years)
        return

    if not isinstance(provider, CachedFMPProvider):
        typer.echo(
            "fetch-history requires a real data provider. Set "
            "NASDAQ_DATA_LINK_API_KEY or FMP_API_KEY in .env"
        )
        raise typer.Exit(1)

    # Load PIT membership history for sp400
    if universe == "sp400":
        import json as _json
        from pathlib import Path

        from screener.data.universe import UniverseManager

        history_path = Path("data/sp400/membership_history.json")
        if history_path.exists():
            um = UniverseManager(cache)
            with open(history_path) as f:
                records = _json.load(f)
            um.bulk_load_history("sp400", records)
            typer.echo(f"Loaded {len(records)} sp400 membership records")
        else:
            typer.echo(
                "Warning: data/sp400/membership_history.json not found."
                " Build the sp400 PIT membership first."
            )

    tickers = provider.get_universe(universe)
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

    # Surface per-ticker staleness: FMP's quarterly key-metrics lags filings by a
    # variable amount, so some tickers can stay a quarter behind even after a full
    # re-fetch. The global max(report_date) check can't see a lone straggler.
    from screener.data.staleness import find_stale_quarters

    stale = find_stale_quarters(cache)
    if stale:
        top = ", ".join(f"{s.ticker}(+{s.quarters_behind}Q)" for s in stale[:8])
        typer.echo(
            f"⚠ PIT freshness: {len(stale)} ticker(s) overdue for a newer quarter "
            f"(FMP compute lag): {top}{' ...' if len(stale) > 8 else ''}"
        )
        typer.echo("  (normal FMP compute lag — clears on the next fetch-history)")
    else:
        typer.echo("PIT freshness: all tickers current.")

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

    from screener.evaluation.report import run_full_evaluation

    settings = Settings()
    pipeline_config = PipelineConfig.from_yaml(config)
    pipeline = _build_pipeline(pipeline_config, settings)

    provider, cache = _make_provider(settings)
    pit_server = _make_pit_server(provider, cache, pipeline_config)

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    tickers = _backtest_universe(cache, provider, pipeline_config.universe)
    price_data = cache.get_prices(tickers, str(start - timedelta(days=400)), str(end))

    # SPY for the CAPM alpha/beta section (fetched up front so it's cached for the
    # benchmark charts below too). Aligned to rebalance dates inside the report.
    provider.get_prices(["SPY"], start - timedelta(days=10), end)
    spy_prices = cache.get_prices(["SPY"], str(start - timedelta(days=10)), str(end))

    report = run_full_evaluation(
        pipeline=pipeline,
        pit_server=pit_server,
        price_data=price_data,
        universe_index=pipeline_config.universe,
        start_date=start,
        end_date=end,
        monte_carlo_iterations=monte_carlo,
        benchmark_prices=spy_prices,
        **pipeline_config.backtest_kwargs(),
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
    from screener.backtest.runner import _generate_rebalance_dates
    from screener.engine.pipeline import enrich_with_price_data

    holdings = []
    prev: set = set()
    for curr in _generate_rebalance_dates(
        date.fromisoformat(start_date), end, pipeline_config.rebalance_frequency
    ):
        # Mirror the backtest's selection: index∩tradeable universe, hold_bonus
        # stickiness, configured cadence — so the turnover chart matches the
        # evaluated portfolio instead of a pure monthly top-10 re-screen (#23).
        universe = pit_server.get_universe_as_of(pipeline_config.universe, curr)
        screening = pit_server.get_screening_data(universe, curr)
        if not screening.is_empty():
            ph = price_data.filter(pl.col("date") < curr)
            enriched = enrich_with_price_data(screening, ph)
            result = pipeline.run(
                enriched,
                hold_bonus_tickers=prev if pipeline_config.hold_bonus > 0 else None,
                hold_bonus=pipeline_config.hold_bonus,
            )
            if not result.is_empty():
                picks = result["ticker"].to_list()
                holdings.append({"date": curr, "picks": picks})
                prev = set(picks)

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

    from screener.research.harness import build_research_context, run_variation

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    ctx = build_research_context(config, start, end, provider_factory=_make_provider)
    cache = ctx.cache
    metrics = run_variation(ctx)

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


@app.command("trade-alpaca")
def trade_alpaca(
    config: Annotated[Path, typer.Option(help="Config YAML")] = Path("config/default.yaml"),
    dry_run: Annotated[bool, typer.Option(help="Show orders without executing")] = True,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Rebalance portfolio via Alpaca (paper or live trading)."""
    _setup_logging(verbose)

    from screener.trading.alpaca import AlpacaBroker

    settings = Settings()
    if not settings.alpaca_api_key:
        typer.echo("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        raise typer.Exit(1)

    pipeline_config = PipelineConfig.from_yaml(config)
    screen_top_n = max(50, pipeline_config.top_n * 5)
    pipeline = _build_pipeline(pipeline_config, settings, top_n=screen_top_n)

    broker = AlpacaBroker(
        settings.alpaca_api_key, settings.alpaca_secret_key,
        paper=settings.alpaca_paper,
    )
    typer.echo(f"Mode: {broker.mode}")
    provider, cache = _make_provider(settings)

    _run_rebalance(
        broker=broker, provider=provider, cache=cache, pipeline=pipeline,
        pipeline_config=pipeline_config, dry_run=dry_run,
        # Alpaca reads its positions directly (independent of the screen).
        get_holdings=lambda enriched: broker.get_positions(),
        is_live_real=not settings.alpaca_paper,
        live_warning="⚠️  LIVE TRADING — orders will execute with real money!",
        log_suffix="_alpaca",
        log_extra={"broker": "alpaca"},
    )


@app.command("trade-etoro")
def trade_etoro(
    config: Annotated[Path, typer.Option(help="Config YAML")] = Path("config/default.yaml"),
    dry_run: Annotated[bool, typer.Option(help="Show orders without executing")] = True,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Rebalance portfolio via eToro (demo or real trading)."""
    _setup_logging(verbose)

    from screener.trading.etoro import EtoroBroker

    settings = Settings()
    if not settings.etoro_api_key or not settings.etoro_user_key:
        typer.echo("Set ETORO_API_KEY and ETORO_USER_KEY in .env")
        raise typer.Exit(1)

    pipeline_config = PipelineConfig.from_yaml(config)
    screen_top_n = max(50, pipeline_config.top_n * 5)
    pipeline = _build_pipeline(pipeline_config, settings, top_n=screen_top_n)

    broker = EtoroBroker(
        settings.etoro_api_key, settings.etoro_user_key,
        demo=settings.etoro_demo,
    )
    typer.echo(f"Mode: {broker.mode}")
    provider, cache = _make_provider(settings)

    def get_holdings(enriched):
        # eToro can only map instruments it searches for, so screen for the top
        # candidates first, then resolve their positions.
        initial = pipeline.run(enriched)
        if initial.is_empty():
            return {}
        candidates = initial.head(pipeline_config.top_n)["ticker"].to_list()
        return broker.resolve_positions(candidates)

    def finalize_picks(result, picks):
        resolved = broker.resolve_instrument_ids(picks)
        unresolved = set(picks) - set(resolved.keys())
        if unresolved:
            typer.echo(
                f"Warning: could not resolve eToro instrument IDs for: "
                f"{', '.join(unresolved)}"
            )
            picks = [t for t in picks if t in resolved]
            if pipeline_config.weighting != "equal":
                typer.echo(
                    "  → weights renormalized over the remaining picks; eToro "
                    "sizing will differ from brokers that hold the dropped name(s)."
                )
        return picks

    _run_rebalance(
        broker=broker, provider=provider, cache=cache, pipeline=pipeline,
        pipeline_config=pipeline_config, dry_run=dry_run,
        get_holdings=get_holdings, finalize_picks=finalize_picks,
        is_live_real=not settings.etoro_demo,
        live_warning="WARNING: LIVE TRADING — orders will execute with real money!",
        log_suffix="_etoro", log_extra={"broker": "etoro"},
    )


@app.command()
def track(
    reseed: Annotated[bool, typer.Option(
        help="Rebuild the cash-flow ledger candidates")] = False,
) -> None:
    """Live-vs-backtest tracking: model returns of logged holdings vs realized,
    deposit-adjusted account equity per broker."""
    from screener.evaluation.tracking import report

    settings = Settings()
    report(settings.db_path, Path("results/trades"), Path("results/tracking"),
           reseed=reseed)


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
