"""Streamlit dashboard for the quantitative research screener.

Fully data-driven — new signals/filters added to the codebase
automatically appear in the UI without any changes here.
"""

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import streamlit as st
from dateutil.relativedelta import relativedelta
from screener.backtest.pit_server import PITDataServer
from screener.backtest.runner import run_backtest
from screener.config import PipelineConfig, Settings
from screener.data.cache import CacheManager
from screener.data.fmp import CachedFMPProvider, FMPProvider
from screener.data.mock import MockProvider
from screener.data.universe import UniverseManager
from screener.engine.pipeline import ScreeningPipeline, enrich_with_price_data
from screener.plugins.registry import discover_filters, discover_signals

st.set_page_config(page_title="Screener Dashboard", layout="wide")

BENCHMARK_TICKERS = ["SPY", "QQQ"]


# --- Cached resources ---


@st.cache_resource
def init_backend():
    """Initialize provider, cache, universe — shared across all sessions."""
    settings = Settings()
    cache = CacheManager(settings.db_path)
    if settings.fmp_api_key:
        fmp = FMPProvider(settings.fmp_api_key)
        provider = CachedFMPProvider(fmp, cache)
        provider_name = "FMP (cached)"
    else:
        provider = MockProvider()
        provider_name = "Mock"
    um = UniverseManager(cache)
    um.sync(provider, "sp500")
    return provider, cache, um, settings, provider_name


@st.cache_data(ttl=300)
def load_price_data(_cache, tickers, start_str, end_str):
    return _cache.get_prices(tickers, start_str, end_str)


@st.cache_resource
def discover_plugins(signals_dir, filters_dir):
    """Discover signals and filters once, cache across reruns."""
    return discover_signals(signals_dir), discover_filters(filters_dir)


def build_pipeline(settings, signal_weights, top_n):
    """Build pipeline from dynamic config. No hardcoded signal names."""
    signals_registry, filters_registry = discover_plugins(
        settings.signals_dir, settings.filters_dir
    )
    filters = list(filters_registry.values())
    signals_with_weights = [
        (signals_registry[name], w)
        for name, w in signal_weights.items()
        if w > 0 and name in signals_registry
    ]
    return ScreeningPipeline(
        filters=filters, signals_with_weights=signals_with_weights, top_n=top_n
    ), filters, signals_with_weights


def compute_etf_returns(price_data, etf, start, end):
    """Compute monthly cumulative returns for a benchmark ETF."""
    etf_prices = price_data.filter(
        (pl.col("ticker") == etf)
        & (pl.col("date") >= start) & (pl.col("date") <= end)
    ).sort("date")
    if etf_prices.is_empty():
        return []
    cum = [1.0]
    curr = date(start.year, start.month, 1)
    while curr < end:
        next_m = curr + relativedelta(months=1)
        period = etf_prices.filter(
            (pl.col("date") >= curr) & (pl.col("date") < next_m)
        )
        if not period.is_empty():
            ret = float(period["close"][-1]) / float(period["close"][0]) - 1
            cum.append(cum[-1] * (1 + ret))
        else:
            cum.append(cum[-1])  # carry forward on missing months
        curr = next_m
    return cum


# --- Sidebar (data-driven) ---


def render_sidebar(settings):
    """Render sidebar controls. Discovers signals dynamically."""
    with st.sidebar:
        st.header("Configuration")
        st.caption(f"Provider: {init_backend()[4]}")

        # Load default config for initial weights
        default_config = PipelineConfig.from_yaml(Path("config/default.yaml"))
        default_weights = {s["name"]: s.get("weight", 0.25) for s in default_config.signals}

        signals_registry, filters_registry = discover_plugins(
            settings.signals_dir, settings.filters_dir
        )

        st.subheader("Signal Weights")
        weights = {}
        for name in sorted(signals_registry.keys()):
            default = default_weights.get(name, 0.0)
            weights[name] = st.slider(
                f"{name}", 0.0, 1.0, value=default, step=0.05,
                help=signals_registry[name].description,
            )

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        st.subheader("Parameters")
        top_n = st.slider("Top N stocks", 5, 50, default_config.top_n)
        cost_bps = st.slider("Transaction cost (bps)", 0, 30, 10)

        st.subheader("Date Range")
        start = st.date_input("Start", date(2017, 1, 1))
        end = st.date_input("End", date(2026, 1, 1))

        st.subheader("Active Filters")
        for name, f in sorted(filters_registry.items()):
            st.caption(f"- {name}: {f.description}")

    return weights, top_n, cost_bps, start, end


# --- Tab renderers ---


def render_backtest_tab(pipeline, pit_server, price_data, start, end, cost_bps):
    """Backtest explorer tab."""
    st.header("Backtest Explorer")

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            metrics = run_backtest(
                pipeline=pipeline, pit_server=pit_server,
                price_data=price_data, universe_index="sp500",
                start_date=start, end_date=end,
                transaction_cost_bps=float(cost_bps),
            )

        # Metrics row
        cols = st.columns(5)
        cols[0].metric("Sharpe", f"{metrics.sharpe_ratio:.3f}")
        cols[1].metric("CAGR", f"{metrics.cagr:.2%}")
        cols[2].metric("Max Drawdown", f"{metrics.max_drawdown:.2%}")
        cols[3].metric("Total Return", f"{metrics.total_return:.2%}")
        cols[4].metric("PSR", f"{metrics.psr:.3f}")

        # Equity curve
        st.subheader("Equity Curve")
        strat_cum = [1.0]
        for r in metrics.periodic_returns:
            strat_cum.append(strat_cum[-1] * (1 + r))

        dates = []
        curr = date(start.year, start.month, 1)
        for _ in range(len(strat_cum)):
            dates.append(curr)
            curr += relativedelta(months=1)

        chart_data = {"Date": dates, "Strategy": strat_cum}
        for etf in BENCHMARK_TICKERS:
            etf_cum = compute_etf_returns(price_data, etf, start, end)
            if etf_cum:
                chart_data[etf] = etf_cum[:len(dates)]

        st.line_chart(
            pl.DataFrame(chart_data).to_pandas().set_index("Date")
        )

        # Monthly returns table
        st.subheader("Monthly Returns")
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly = {}
        curr = date(start.year, start.month, 1)
        for r in metrics.periodic_returns:
            monthly.setdefault(curr.year, {})[curr.strftime("%b")] = r
            curr += relativedelta(months=1)

        rows = []
        for year in sorted(monthly.keys()):
            row = {"Year": year}
            for m in month_names:
                val = monthly[year].get(m)
                row[m] = f"{val:.1%}" if val is not None else ""
            rows.append(row)

        st.dataframe(
            pl.DataFrame(rows).to_pandas().set_index("Year"),
            use_container_width=True,
        )

        st.session_state["last_metrics"] = metrics


def render_signal_tab(pipeline, filters, signals_with_weights, pit_server, price_data):
    """Signal deep-dive tab."""
    st.header("Signal Deep-Dive")

    dive_date = st.date_input("Inspect date", date(2025, 3, 1), key="dive_date")

    if st.button("Inspect", key="inspect_btn"):
        with st.spinner("Loading signals..."):
            tickers = pit_server.get_tradeable_tickers(dive_date)
            screening_data = pit_server.get_screening_data(tickers, dive_date)

            if screening_data.is_empty():
                st.warning("No data for this date")
                return

            price_hist = price_data.filter(pl.col("date") < dive_date)
            enriched = enrich_with_price_data(screening_data, price_hist)

            # Compute each signal's scores
            score_cols = {"ticker": enriched["ticker"]}
            for col in ["close", "market_cap"]:
                if col in enriched.columns:
                    if col == "market_cap":
                        score_cols["mkt_cap_B"] = (enriched[col] / 1e9).round(1)
                    else:
                        score_cols[col] = enriched[col].round(2)

            for signal, weight in signals_with_weights:
                try:
                    score_cols[f"{signal.name} ({weight:.0%})"] = signal.compute(enriched).round(4)
                except Exception as e:
                    st.warning(f"{signal.name}: {e}")

            # Mark which pass filters and which are selected
            mask = pl.Series([True] * len(enriched))
            for f in filters:
                try:
                    mask = mask & f.apply(enriched)
                except Exception as e:
                    st.warning(f"Filter {f.name} failed: {e}")

            result = pipeline.run(enriched)
            picked = set(result["ticker"].to_list()) if not result.is_empty() else set()

            scores_df = pl.DataFrame(score_cols).with_columns([
                mask.alias("passes_filters"),
                pl.col("ticker").is_in(list(picked)).alias("SELECTED"),
            ])

            # Display
            st.subheader(f"All Stocks ({len(scores_df)} total, {len(picked)} selected)")
            st.dataframe(
                scores_df.sort("SELECTED", descending=True).to_pandas(),
                use_container_width=True, height=500,
            )

            if picked:
                st.subheader(f"Selected Portfolio ({len(picked)} stocks)")
                st.dataframe(
                    scores_df.filter(pl.col("SELECTED"))
                    .drop("SELECTED", "passes_filters").to_pandas(),
                    use_container_width=True,
                )

            # Filter funnel
            st.subheader("Filter Funnel")
            remaining = enriched
            funnel = [{"Step": "Universe", "Stocks": len(remaining)}]
            for f in filters:
                try:
                    remaining = remaining.filter(f.apply(remaining))
                    funnel.append({"Step": f.name, "Stocks": len(remaining)})
                except Exception as e:
                    st.warning(f"Filter {f.name}: {e}")
            funnel.append({"Step": f"Top {pipeline.top_n}", "Stocks": len(picked)})

            cols = st.columns(len(funnel))
            for i, step in enumerate(funnel):
                cols[i].metric(step["Step"], step["Stocks"])


def render_screen_tab(pipeline, pit_server, tickers, price_data):
    """Current screen tab."""
    st.header("Current Screen")

    if st.button("Screen Now", type="primary", key="screen_btn"):
        with st.spinner("Screening (using cached data)..."):
            yesterday = date.today() - timedelta(days=1)
            # Use PIT server instead of live API — much faster, uses cached data
            fundamentals = pit_server.get_screening_data(tickers, yesterday)
            prices = price_data.filter(pl.col("date") <= yesterday)
            enriched = enrich_with_price_data(fundamentals, prices)
            result = pipeline.run(enriched)

            if result.is_empty():
                st.warning("No stocks passed filters")
                return

            # Auto-detect display columns
            display_cols = ["ticker", "composite_score"]
            display_cols.extend(c for c in result.columns if c.startswith("z_"))
            for c in ["close", "market_cap", "roe", "earnings_yield",
                       "momentum_12m_return", "sma_200"]:
                if c in result.columns:
                    display_cols.append(c)

            available = [c for c in display_cols if c in result.columns]
            df = result.select(available)

            if "market_cap" in df.columns:
                df = df.with_columns(
                    (pl.col("market_cap") / 1e9).round(1).alias("mkt_cap_B")
                ).drop("market_cap")

            st.dataframe(df.to_pandas(), use_container_width=True, height=600)


# --- Main ---


def main():
    provider, cache, um, settings, provider_name = init_backend()
    pit_server = PITDataServer(provider, cache, um)

    weights, top_n, cost_bps, start, end = render_sidebar(settings)
    pipeline, filters, signals_with_weights = build_pipeline(settings, weights, top_n)

    tickers = provider.get_universe("sp500")
    price_data = load_price_data(
        cache, tickers + BENCHMARK_TICKERS,
        str(start - timedelta(days=400)), str(end),
    )

    tab1, tab2, tab3 = st.tabs(["Backtest Explorer", "Signal Deep-Dive", "Current Screen"])

    with tab1:
        render_backtest_tab(pipeline, pit_server, price_data, start, end, cost_bps)
    with tab2:
        render_signal_tab(pipeline, filters, signals_with_weights, pit_server, price_data)
    with tab3:
        render_screen_tab(pipeline, pit_server, tickers, price_data)


if __name__ == "__main__":
    main()
