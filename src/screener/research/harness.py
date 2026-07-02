"""Shared setup for backtests, experiments, and the CLI research commands.

Before this module, the ~8-line "load config → build pipeline → make provider →
sync PIT server → pull padded price history" incantation was duplicated across
every experiment harness (reaching into private ``cli`` helpers) and the
``lab``/``backtest``/``evaluate`` commands. A new experiment now reads:

    from screener.research.harness import build_research_context, run_variation
    ctx = build_research_context("config/example.yaml", start, end)
    metrics = run_variation(ctx)                      # reproduces `lab` exactly

These functions are deliberately CLI-agnostic (no ``typer``); the CLI layer adds
its own user-facing echoes around them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from screener.config import PipelineConfig, Settings

logger = logging.getLogger(__name__)

# Extra price history pulled *before* the backtest start so momentum_12m_return
# and sma_200 aren't biased during the first ~12 months (they need ~252 trading
# days of lookback). One named constant instead of a magic 400 copy-pasted at
# every price-fetch site.
PRICE_LOOKBACK_DAYS = 400


def make_provider(settings: Settings):
    """Select provider per settings.data_provider.

    "auto" prefers Sharadar (when NASDAQ_DATA_LINK_API_KEY is set), then FMP,
    then mock. DATA_PROVIDER=fmp forces FMP — the rollback path after the
    jul-2026 Sharadar migration."""
    from screener.data.cache import CacheManager

    cache = CacheManager(settings.db_path)
    choice = settings.data_provider
    if choice == "auto":
        if settings.nasdaq_data_link_api_key:
            choice = "sharadar"
        elif settings.fmp_api_key:
            choice = "fmp"
        else:
            choice = "mock"

    if choice == "sharadar":
        if not settings.nasdaq_data_link_api_key:
            raise ValueError("DATA_PROVIDER=sharadar but NASDAQ_DATA_LINK_API_KEY unset")
        from screener.data.sharadar import CachedSharadarProvider, SharadarProvider

        provider = CachedSharadarProvider(
            SharadarProvider(settings.nasdaq_data_link_api_key), cache
        )
    elif choice == "fmp":
        if not settings.fmp_api_key:
            raise ValueError("DATA_PROVIDER=fmp but FMP_API_KEY unset")
        from screener.data.fmp import CachedFMPProvider, FMPProvider

        provider = CachedFMPProvider(FMPProvider(settings.fmp_api_key), cache)
    elif choice == "mock":
        from screener.data.mock import MockProvider

        provider = MockProvider()
    else:
        raise ValueError(f"Unknown data_provider {choice!r} (auto|sharadar|fmp|mock)")
    # The CLI layer adds the user-facing "Using ... provider" echo; this logs
    # instead of printing so CLI commands don't double-announce — but it is
    # never fully silent, so a broken .env can't run an experiment on mock
    # without leaving a trace.
    logger.info("Data provider: %s (db: %s)", choice, settings.db_path)
    return provider, cache


def build_pipeline(
    pipeline_config: PipelineConfig, settings: Settings, top_n: int | None = None
):
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


def make_pit_server(provider, cache, pipeline_config: PipelineConfig):
    """Create a PIT server with the universe membership synced."""
    from screener.backtest.pit_server import PITDataServer
    from screener.data.universe import UniverseManager

    um = UniverseManager(cache)
    um.sync(provider, pipeline_config.universe)
    return PITDataServer(provider, cache, um)


def backtest_universe(cache, provider, universe: str) -> list[str]:
    """Full PIT membership (active + departed) for survivorship-correct price_data.

    ``get_universe`` returns only *current* members, so departed names the
    screener selects can't be priced — the backtest silently trades survivors and
    the Sharpe is inflated. Pricing the full historical membership fixes it. Falls
    back to current constituents when membership history isn't loaded (e.g. mock).
    """
    try:
        rows = cache.to_polars(
            "SELECT DISTINCT ticker FROM universe_membership WHERE index_name = ?",
            [universe],
        )
        members = [t for t in rows["ticker"].to_list() if "/" not in t]
        if members:
            return members
    except Exception:
        pass
    return provider.get_universe(universe)


@dataclass
class ResearchContext:
    """Everything a backtest/experiment needs, assembled once."""

    config: PipelineConfig
    settings: Settings
    pipeline: object
    provider: object
    cache: object
    pit_server: object
    price_data: pl.DataFrame
    start: date
    end: date


def build_research_context(
    config_path: str | Path,
    start: date,
    end: date,
    *,
    top_n: int | None = None,
    provider_factory=make_provider,
) -> ResearchContext:
    """Assemble the full backtest setup from a config path and date range.

    ``provider_factory`` lets the CLI inject its echoing provider builder so the
    command output is unchanged; experiments use the default (logger only).
    """
    settings = Settings()
    config = PipelineConfig.from_yaml(Path(config_path))
    provider, cache = provider_factory(settings)
    pipeline = build_pipeline(config, settings, top_n)
    pit_server = make_pit_server(provider, cache, config)
    tickers = backtest_universe(cache, provider, config.universe)
    price_data = cache.get_prices(
        tickers, str(start - timedelta(days=PRICE_LOOKBACK_DAYS)), str(end)
    )
    return ResearchContext(
        config=config,
        settings=settings,
        pipeline=pipeline,
        provider=provider,
        cache=cache,
        pit_server=pit_server,
        price_data=price_data,
        start=start,
        end=end,
    )


def run_variation(ctx: ResearchContext, *, pipeline=None, **overrides):
    """Run a backtest over ``ctx``, threading the config's backtest knobs.

    With no overrides this reproduces ``screener lab`` exactly. Pass ``pipeline``
    to test an alternative signal set, or keyword overrides (e.g. ``frequency``,
    ``weighting``) to vary a single knob.
    """
    from screener.backtest.runner import run_backtest

    cfg = ctx.config
    kwargs = dict(
        pipeline=pipeline or ctx.pipeline,
        pit_server=ctx.pit_server,
        price_data=ctx.price_data,
        universe_index=cfg.universe,
        start_date=ctx.start,
        end_date=ctx.end,
        **cfg.backtest_kwargs(),
    )
    kwargs.update(overrides)
    return run_backtest(**kwargs)
