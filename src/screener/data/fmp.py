from __future__ import annotations

import logging
import threading
import time
from datetime import date, timedelta

import httpx
import polars as pl

from screener.data.cache import CacheManager

logger = logging.getLogger(__name__)

BASE_URL = "https://financialmodelingprep.com/stable"
MAX_CALLS_PER_MINUTE = 720  # FMP limit is 750, leave margin


class FMPProvider:
    """Financial Modeling Prep data provider (stable API)."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = httpx.Client(timeout=30.0)
        self._call_times: list[float] = []
        self._rate_lock = threading.Lock()

    def _rate_limit(self) -> None:
        """Evenly space API calls to stay under the rate limit. Thread-safe.

        Uses minimum interval between calls (60s / max_calls) instead of
        burst-then-sleep, which keeps throughput smooth and avoids long pauses.
        """
        min_interval = 60.0 / MAX_CALLS_PER_MINUTE  # ~0.083s at 720/min
        sleep_time = 0.0
        with self._rate_lock:
            now = time.monotonic()
            if self._call_times:
                elapsed = now - self._call_times[-1]
                if elapsed < min_interval:
                    sleep_time = min_interval - elapsed
            # Reserve our slot before releasing the lock
            self._call_times.append(now + sleep_time)
            # Trim old entries to prevent unbounded growth
            cutoff = now - 60
            self._call_times = [t for t in self._call_times if t > cutoff]
        # Sleep outside the lock so other threads can proceed
        if sleep_time > 0:
            time.sleep(sleep_time)

    def _get(
        self, endpoint: str, params: dict | None = None, retries: int = 2
    ) -> list[dict] | dict:
        self._rate_limit()
        params = params or {}
        params["apikey"] = self._api_key
        url = f"{BASE_URL}/{endpoint}"
        for attempt in range(retries + 1):
            try:
                resp = self._client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, str):
                    raise httpx.HTTPStatusError(
                        f"API error: {data}",
                        request=resp.request, response=resp,
                    )
                return data
            except (httpx.HTTPError, httpx.HTTPStatusError) as e:
                if attempt < retries:
                    wait = 2 ** attempt
                    logger.debug("Retry %d/%d for %s: %s", attempt + 1, retries, endpoint, e)
                    time.sleep(wait)
                else:
                    raise

    def get_key_metrics(self, ticker: str, limit: int = 1) -> list[dict]:
        """Fetch key financial metrics for a ticker. Returns list of annual records."""
        try:
            data = self._get(
                "key-metrics", params={"symbol": ticker, "period": "annual", "limit": limit}
            )
            if isinstance(data, list):
                return data
        except (httpx.HTTPError, httpx.HTTPStatusError):
            logger.debug("key-metrics unavailable for %s", ticker)
        return []

    def get_price_target_consensus(self, ticker: str) -> float | None:
        """Fetch analyst consensus price target."""
        try:
            data = self._get("price-target-consensus", params={"symbol": ticker})
            if isinstance(data, list) and data:
                return data[0].get("targetConsensus")
        except (httpx.HTTPError, httpx.HTTPStatusError):
            pass
        return None

    def get_insider_stats(self, ticker: str) -> float | None:
        """Fetch latest insider acquired/disposed ratio."""
        try:
            data = self._get("insider-trading/statistics", params={"symbol": ticker})
            if isinstance(data, list) and data:
                return data[0].get("acquiredDisposedRatio")
        except (httpx.HTTPError, httpx.HTTPStatusError):
            pass
        return None

    def get_financial_growth(
        self, ticker: str, limit: int = 3, period: str = "annual"
    ) -> list[dict]:
        """Fetch financial growth rates (annual or quarterly)."""
        try:
            data = self._get(
                "financial-growth",
                params={"symbol": ticker, "period": period, "limit": limit},
            )
            if isinstance(data, list):
                return data
        except (httpx.HTTPError, httpx.HTTPStatusError):
            logger.debug("financial-growth unavailable for %s", ticker)
        return []

    def get_ratios(self, ticker: str, limit: int = 2) -> list[dict]:
        """Fetch financial ratios (margins, P/E, P/B, etc.). Annual only."""
        try:
            data = self._get(
                "ratios", params={"symbol": ticker, "period": "annual", "limit": limit}
            )
            if isinstance(data, list):
                return data
        except (httpx.HTTPError, httpx.HTTPStatusError):
            logger.debug("ratios unavailable for %s", ticker)
        return []

    def get_quarterly_key_metrics(self, ticker: str, limit: int = 2) -> list[dict]:
        """Fetch quarterly key-metrics (Premium plan)."""
        try:
            data = self._get(
                "key-metrics",
                params={"symbol": ticker, "period": "quarter", "limit": limit},
            )
            if isinstance(data, list):
                return data
        except (httpx.HTTPError, httpx.HTTPStatusError):
            logger.debug("quarterly key-metrics unavailable for %s", ticker)
        return []

    def enrich_profile_row(self, row: dict, ticker: str) -> dict:
        """Merge key-metrics, quarterly key-metrics, growth, ratios, and alt data.

        Quarterly key-metrics (Premium) override annual for ROE, ROIC, ROA,
        current_ratio, FCF yield, earnings yield — at most 3 months stale.
        """
        # Start with annual key-metrics as baseline
        km = self.get_key_metrics(ticker, limit=1)
        if km:
            row.update(km[0])
        # Override with quarterly key-metrics (Premium plan, 1 API call replaces 3)
        # Values are single-quarter (not annualized) — consistent with PIT snapshots
        qkm = self.get_quarterly_key_metrics(ticker, limit=2)
        if qkm:
            # Override core ratios with fresher quarterly values
            # Uses FMP field names — _normalize_fundamentals handles renaming
            for fmp_field in [
                "returnOnEquity", "returnOnInvestedCapital", "returnOnAssets",
                "currentRatio", "netDebtToEBITDA",
                "earningsYield", "freeCashFlowYield",
            ]:
                val = qkm[0].get(fmp_field)
                if val is not None:
                    row[fmp_field] = val
            # Efficiency and quality ratios from same quarterly key-metrics call
            row["sga_to_revenue"] = qkm[0].get("salesGeneralAndAdministrativeToRevenue")
            row["rd_to_revenue"] = qkm[0].get("researchAndDevelopementToRevenue")
            row["sbc_to_revenue"] = qkm[0].get("stockBasedCompensationToRevenue")
            row["income_quality"] = qkm[0].get("incomeQuality")
            row["capex_to_revenue"] = qkm[0].get("capexToRevenue")
            row["intangibles_to_assets"] = qkm[0].get("intangiblesToTotalAssets")
            row["cash_conversion_cycle"] = qkm[0].get("cashConversionCycle")
            if len(qkm) >= 2:
                row["sga_to_revenue_prior"] = qkm[1].get(
                    "salesGeneralAndAdministrativeToRevenue"
                )
        # Growth rates (prefer quarterly)
        q_growth = self.get_financial_growth(ticker, limit=2, period="quarter")
        if len(q_growth) >= 2:
            row["eps_growth_current"] = q_growth[0].get("epsgrowth")
            row["eps_growth_prior"] = q_growth[1].get("epsgrowth")
            row["rev_growth_current"] = q_growth[0].get("revenueGrowth")
            row["rev_growth_prior"] = q_growth[1].get("revenueGrowth")
        else:
            # Fall back to annual growth
            growth = self.get_financial_growth(ticker, limit=2, period="annual")
            if len(growth) >= 2:
                row["eps_growth_current"] = growth[0].get("epsgrowth")
                row["eps_growth_prior"] = growth[1].get("epsgrowth")
                row["rev_growth_current"] = growth[0].get("revenueGrowth")
                row["rev_growth_prior"] = growth[1].get("revenueGrowth")
            elif len(growth) == 1:
                row["eps_growth_current"] = growth[0].get("epsgrowth")
                row["rev_growth_current"] = growth[0].get("revenueGrowth")
        # Margin data for expansion signal
        ratios = self.get_ratios(ticker, limit=2)
        if len(ratios) >= 2:
            row["gross_margin_current"] = ratios[0].get("grossProfitMargin")
            row["gross_margin_prior"] = ratios[1].get("grossProfitMargin")
            row["op_margin_current"] = ratios[0].get("operatingProfitMargin")
            row["op_margin_prior"] = ratios[1].get("operatingProfitMargin")
        elif len(ratios) == 1:
            row["gross_margin_current"] = ratios[0].get("grossProfitMargin")
            row["op_margin_current"] = ratios[0].get("operatingProfitMargin")
        # Analyst target and insider data
        target = self.get_price_target_consensus(ticker)
        if target is not None:
            row["analyst_target"] = target
        insider = self.get_insider_stats(ticker)
        if insider is not None:
            row["insider_buy_ratio"] = insider
        return row

    def get_fundamentals(self, tickers: list[str]) -> pl.DataFrame:
        rows = []
        for ticker in tickers:
            try:
                data = self._get("profile", params={"symbol": ticker})
                if isinstance(data, list) and data:
                    row = data[0]
                elif isinstance(data, dict) and data:
                    row = data
                else:
                    continue
                rows.append(self.enrich_profile_row(row, ticker))
            except (httpx.HTTPError, httpx.HTTPStatusError):
                logger.warning("Failed to fetch profile for %s", ticker)
        if not rows:
            return pl.DataFrame()
        return _normalize_fundamentals(pl.DataFrame(rows))

    def get_prices_single(self, ticker: str, start: date, end: date) -> pl.DataFrame:
        """Fetch historical EOD prices for a single ticker."""
        try:
            data = self._get(
                "historical-price-eod/full",
                params={"symbol": ticker, "from": str(start), "to": str(end)},
            )
            if isinstance(data, list) and data:
                return (
                    pl.DataFrame(data)
                    .select(
                        pl.col("symbol").alias("ticker"),
                        pl.col("date").str.to_date("%Y-%m-%d").alias("date"),
                        pl.col("open").cast(pl.Float64),
                        pl.col("high").cast(pl.Float64),
                        pl.col("low").cast(pl.Float64),
                        pl.col("close").cast(pl.Float64),
                        pl.col("volume").cast(pl.Float64),
                    )
                )
        except (httpx.HTTPError, httpx.HTTPStatusError):
            logger.warning("Failed to fetch prices for %s", ticker)
        return pl.DataFrame()

    def get_prices(self, tickers: list[str], start: date, end: date) -> pl.DataFrame:
        frames = []
        for ticker in tickers:
            df = self.get_prices_single(ticker, start, end)
            if not df.is_empty():
                frames.append(df)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames)

    def get_universe(self, index: str) -> list[str]:
        """Get index constituents. Falls back to hardcoded list on free tier."""
        endpoint_map = {
            "sp500": "sp500-constituent",
            "nasdaq100": "nasdaq-constituent",
            "dowjones": "dowjones-constituent",
        }
        if index not in endpoint_map:
            raise ValueError(f"Unknown index: {index}")
        try:
            data = self._get(endpoint_map[index])
            if isinstance(data, list) and data:
                return [row["symbol"] for row in data]
        except (httpx.HTTPError, httpx.HTTPStatusError):
            logger.warning("Constituent endpoint restricted, using hardcoded list")

        # Free tier fallback
        if index == "sp500":
            return _SP500_TICKERS
        raise ValueError(f"No fallback ticker list for index: {index}")

    def get_delisted(self) -> list[str]:
        try:
            data = self._get("delisted-companies")
            if isinstance(data, list):
                return [row["symbol"] for row in data]
        except (httpx.HTTPError, httpx.HTTPStatusError):
            logger.warning("Delisted endpoint restricted")
        return []


# Fields that need renaming (FMP name -> our name)
_RENAME_MAP = {
    "marketCap": "market_cap",
    "price": "close",
    "averageVolume": "avg_volume_20d",
    "returnOnEquity": "roe",
    "earningsYield": "earnings_yield",
    "freeCashFlowYield": "fcf_yield",
    "evToSales": "ev_to_sales",
    "returnOnAssets": "roa",
    "returnOnInvestedCapital": "roic",
    "currentRatio": "current_ratio",
    "netDebtToEBITDA": "net_debt_to_ebitda",
}

# Fields injected by enrich_profile_row (already use our naming)
_PASSTHROUGH_FIELDS = [
    "beta",
    "gross_margin_current", "gross_margin_prior",
    "op_margin_current", "op_margin_prior",
    "eps_growth_current", "eps_growth_prior",
    "rev_growth_current", "rev_growth_prior",
    "analyst_target", "insider_buy_ratio",
    "sga_to_revenue", "sga_to_revenue_prior",
    "rd_to_revenue", "sbc_to_revenue",
    "income_quality", "capex_to_revenue",
    "intangibles_to_assets", "cash_conversion_cycle",
]

# Combined map for PIT snapshot iteration
_FIELD_MAP = {**_RENAME_MAP, **{f: f for f in _PASSTHROUGH_FIELDS}}


def _normalize_fundamentals(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize FMP profile + key-metrics to standard column names."""
    exprs = [pl.col("symbol").alias("ticker")]

    for fmp_col, our_col in _RENAME_MAP.items():
        if fmp_col in df.columns:
            exprs.append(pl.col(fmp_col).cast(pl.Float64).alias(our_col))

    for col in _PASSTHROUGH_FIELDS:
        if col in df.columns:
            exprs.append(pl.col(col).cast(pl.Float64))

    if "sector" in df.columns:
        exprs.append(pl.col("sector"))

    return df.select(exprs)


class CachedFMPProvider:
    """FMP provider with DuckDB-backed caching.

    - Profiles: cached per ticker with 24h TTL
    - Prices: per-ticker incremental gap-fill, cached forever (immutable)
    - Universe: cached with 7-day TTL
    """

    PROFILE_TTL_HOURS = 7 * 24  # 7 days — fundamentals update quarterly
    UNIVERSE_TTL_HOURS = 30 * 24  # 30 days — S&P 500 changes ~2x/month
    # Bump this when enrich_profile_row adds new fields to invalidate stale cache
    _CACHE_VERSION = 4

    def __init__(self, fmp: FMPProvider, cache: CacheManager):
        self._fmp = fmp
        self._cache = cache

    def _fundamentals_cache_key(self, ticker: str) -> str:
        return f"fundamentals/v{self._CACHE_VERSION}/{ticker}"

    def get_fundamentals(self, tickers: list[str]) -> pl.DataFrame:
        """Fetch profiles + key-metrics, using cache where possible."""
        cached_rows = []
        uncached_tickers = []

        for ticker in tickers:
            hit = self._cache.get_or_fetch_if_cached(
                provider="fmp", endpoint=self._fundamentals_cache_key(ticker)
            )
            if hit is not None:
                cached_rows.extend(hit)
            else:
                uncached_tickers.append(ticker)

        if uncached_tickers:
            logger.info(
                "Fundamentals: %d cached, %d to fetch",
                len(tickers) - len(uncached_tickers), len(uncached_tickers),
            )

            def _fetch_one(ticker: str) -> dict | None:
                try:
                    data = self._fmp._get("profile", params={"symbol": ticker})
                    if isinstance(data, list) and data:
                        row = data[0]
                    elif isinstance(data, dict) and data:
                        row = data
                    else:
                        return None
                    return self._fmp.enrich_profile_row(row, ticker)
                except (httpx.HTTPError, httpx.HTTPStatusError):
                    logger.warning("Failed to fetch fundamentals for %s", ticker)
                    return None

            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=6) as pool:
                results = pool.map(_fetch_one, uncached_tickers)

            for ticker, row in zip(uncached_tickers, results):
                if row is not None:
                    self._cache.get_or_fetch(
                        provider="fmp",
                        endpoint=self._fundamentals_cache_key(ticker),
                        fetch_fn=lambda r=row: [r],
                        ttl_hours=self.PROFILE_TTL_HOURS,
                    )
                    if row.get("sector"):
                        self._cache.store_sector(ticker, row["sector"])
                    cached_rows.append(row)
        else:
            logger.info("Fundamentals: all %d cached", len(tickers))

        if not cached_rows:
            return pl.DataFrame()
        return _normalize_fundamentals(pl.DataFrame(cached_rows))

    def get_prices(self, tickers: list[str], start: date, end: date) -> pl.DataFrame:
        """Fetch prices with incremental gap-fill. Only fetches what's missing."""
        api_calls = 0
        for ticker in tickers:
            cached_range = self._cache.get_cached_price_range(ticker)

            if cached_range is not None:
                cached_start = date.fromisoformat(cached_range[0])
                cached_end = date.fromisoformat(cached_range[1])

                if start < cached_start:
                    gap_end = cached_start - timedelta(days=1)
                    df = self._fmp.get_prices_single(ticker, start, gap_end)
                    if not df.is_empty():
                        self._cache.store_prices(df, source="fmp")
                    api_calls += 1

                if end > cached_end:
                    gap_start = cached_end + timedelta(days=1)
                    df = self._fmp.get_prices_single(ticker, gap_start, end)
                    if not df.is_empty():
                        self._cache.store_prices(df, source="fmp")
                    api_calls += 1
            else:
                df = self._fmp.get_prices_single(ticker, start, end)
                if not df.is_empty():
                    self._cache.store_prices(df, source="fmp")
                api_calls += 1

        if api_calls > 0:
            logger.info("Prices: %d API calls for %d tickers", api_calls, len(tickers))
        else:
            logger.info("Prices: all %d tickers fully cached", len(tickers))

        return self._cache.get_prices(tickers, str(start), str(end))

    def get_universe(self, index: str) -> list[str]:
        data = self._cache.get_or_fetch(
            provider="fmp",
            endpoint=f"universe/{index}",
            fetch_fn=lambda: [{"symbol": t} for t in self._fmp.get_universe(index)],
            ttl_hours=self.UNIVERSE_TTL_HOURS,
        )
        return [row["symbol"] for row in data]

    def get_delisted(self) -> list[str]:
        data = self._cache.get_or_fetch(
            provider="fmp",
            endpoint="delisted",
            fetch_fn=lambda: [{"symbol": t} for t in self._fmp.get_delisted()],
            ttl_hours=self.UNIVERSE_TTL_HOURS,
        )
        return [row["symbol"] for row in data]

    def collect_pit_snapshots(self, ticker: str, years: int = 10) -> list[tuple]:
        """Fetch historical data and return as list of snapshot tuples.

        Thread-safe: only reads from API, no DB writes.
        Returns list of (ticker, field, value, report_date, source, observed_at).
        """
        snapshots: list[tuple] = []

        # Fields covered by quarterly key-metrics — skip in annual to avoid
        # annual observed_at (+90d) shadowing quarterly observed_at (+45d)
        quarterly_fields = {
            "returnOnEquity", "returnOnInvestedCapital", "returnOnAssets",
            "currentRatio", "netDebtToEBITDA", "earningsYield", "freeCashFlowYield",
            "salesGeneralAndAdministrativeToRevenue", "researchAndDevelopementToRevenue",
            "stockBasedCompensationToRevenue", "incomeQuality", "capexToRevenue",
            "intangiblesToTotalAssets", "cashConversionCycle",
        }

        # Key-metrics snapshots (annual only — quarterly fields handled below)
        records = self._fmp.get_key_metrics(ticker, limit=years)
        for record in records:
            report_date = record.get("date", "")
            if not report_date:
                continue
            try:
                rd = date.fromisoformat(report_date)
                observed_at = str(rd + timedelta(days=90))
            except ValueError:
                continue
            for fmp_field, our_field in _FIELD_MAP.items():
                if fmp_field in quarterly_fields:
                    continue
                if fmp_field in record and record[fmp_field] is not None:
                    snapshots.append((
                        ticker, our_field, float(record[fmp_field]),
                        report_date, "fmp", observed_at,
                    ))

        # Growth rate snapshots
        growth_records = self._fmp.get_financial_growth(ticker, limit=years)
        for i, record in enumerate(growth_records):
            report_date = record.get("date", "")
            if not report_date:
                continue
            try:
                rd = date.fromisoformat(report_date)
                observed_at = str(rd + timedelta(days=90))
            except ValueError:
                continue
            for fmp_field, our_field in [
                ("epsgrowth", "eps_growth_current"),
                ("revenueGrowth", "rev_growth_current"),
            ]:
                if fmp_field in record and record[fmp_field] is not None:
                    snapshots.append((
                        ticker, our_field, float(record[fmp_field]),
                        report_date, "fmp", observed_at,
                    ))
            if i + 1 < len(growth_records):
                prior = growth_records[i + 1]
                for fmp_field, our_field in [
                    ("epsgrowth", "eps_growth_prior"),
                    ("revenueGrowth", "rev_growth_prior"),
                ]:
                    if fmp_field in prior and prior[fmp_field] is not None:
                        snapshots.append((
                            ticker, our_field, float(prior[fmp_field]),
                            report_date, "fmp", observed_at,
                        ))

        # Margin snapshots from /ratios
        ratios = self._fmp.get_ratios(ticker, limit=years)
        for i, record in enumerate(ratios):
            report_date = record.get("date", "")
            if not report_date:
                continue
            try:
                rd = date.fromisoformat(report_date)
                observed_at = str(rd + timedelta(days=90))
            except ValueError:
                continue
            for fmp_field, our_field in [
                ("grossProfitMargin", "gross_margin_current"),
                ("operatingProfitMargin", "op_margin_current"),
            ]:
                if fmp_field in record and record[fmp_field] is not None:
                    snapshots.append((
                        ticker, our_field, float(record[fmp_field]),
                        report_date, "fmp", observed_at,
                    ))
            if i + 1 < len(ratios):
                prior = ratios[i + 1]
                for fmp_field, our_field in [
                    ("grossProfitMargin", "gross_margin_prior"),
                    ("operatingProfitMargin", "op_margin_prior"),
                ]:
                    if fmp_field in prior and prior[fmp_field] is not None:
                        snapshots.append((
                            ticker, our_field, float(prior[fmp_field]),
                            report_date, "fmp", observed_at,
                        ))

        # Quarterly key-metrics (Premium) — efficiency + core ratios in single pass
        q_km = self._fmp.get_quarterly_key_metrics(ticker, limit=years * 4)
        for i, record in enumerate(q_km):
            report_date = record.get("date", "")
            if not report_date:
                continue
            try:
                rd = date.fromisoformat(report_date)
                observed_at = str(rd + timedelta(days=45))
            except ValueError:
                continue
            for fmp_field, our_field in [
                ("salesGeneralAndAdministrativeToRevenue", "sga_to_revenue"),
                ("researchAndDevelopementToRevenue", "rd_to_revenue"),
                ("stockBasedCompensationToRevenue", "sbc_to_revenue"),
                ("incomeQuality", "income_quality"),
                ("capexToRevenue", "capex_to_revenue"),
                ("intangiblesToTotalAssets", "intangibles_to_assets"),
                ("cashConversionCycle", "cash_conversion_cycle"),
                ("returnOnEquity", "roe"),
                ("returnOnInvestedCapital", "roic"),
                ("returnOnAssets", "roa"),
                ("currentRatio", "current_ratio"),
                ("netDebtToEBITDA", "net_debt_to_ebitda"),
                ("earningsYield", "earnings_yield"),
                ("freeCashFlowYield", "fcf_yield"),
            ]:
                if fmp_field in record and record[fmp_field] is not None:
                    snapshots.append((
                        ticker, our_field, float(record[fmp_field]),
                        report_date, "fmp_qkm", observed_at,
                    ))
            if i + 1 < len(q_km):
                prior = q_km[i + 1]
                sga_prior = prior.get("salesGeneralAndAdministrativeToRevenue")
                if sga_prior is not None:
                    snapshots.append((
                        ticker, "sga_to_revenue_prior", float(sga_prior),
                        report_date, "fmp_qkm", observed_at,
                    ))

        return snapshots

    def populate_pit_snapshots(self, ticker: str, years: int = 10) -> int:
        """Collect and store PIT snapshots for a single ticker.

        Returns number of snapshots stored.
        """
        snapshots = self.collect_pit_snapshots(ticker, years)
        for t, field, value, report_date, source, observed_at in snapshots:
            self._cache.record_pit_snapshot(
                ticker=t, field=field, value=value,
                report_date=report_date, source=source,
                observed_at=observed_at,
            )
        return len(snapshots)


# Hardcoded S&P 500 tickers for free-tier fallback
_SP500_TICKERS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE",
    "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALK",
    "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN",
    "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APTV", "ARE", "ATO",
    "AVB", "AVGO", "AVY", "AWK", "AXP", "AZO", "BA", "BAC", "BAX",
    "BBWI", "BBY", "BDX", "BEN", "BF.B", "BIIB", "BIO", "BK", "BKNG", "BKR",
    "BLK", "BMY", "BR", "BRK.B", "BRO", "BSX", "BWA", "BXP", "C", "CAG",
    "CAH", "CARR", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL", "CDAY", "CDNS",
    "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF",
    "CL", "CLX", "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP",
    "COF", "COO", "COP", "COST", "CPB", "CPRT", "CPT", "CRL", "CRM", "CSCO",
    "CSGP", "CSX", "CTAS", "CTLT", "CTRA", "CTSH", "CTVA", "CVS", "CVX", "CZR",
    "D", "DAL", "DD", "DE", "DFS", "DG", "DGX", "DHI", "DHR", "DIS",
    "DLR", "DLTR", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA",
    "DVN", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EIX", "EL",
    "EMN", "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS",
    "ETN", "ETR", "ETSY", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR", "F",
    "FANG", "FAST", "FBHS", "FCX", "FDS", "FDX", "FE", "FFIV", "FIS", "FISV",
    "FITB", "FLT", "FMC", "FOXA", "FRT", "FTNT", "FTV", "GD",
    "GE", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOGL", "GPC",
    "GPN", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA", "HD", "PEAK",
    "HES", "HIG", "HII", "HLT", "HOLX", "HON", "HPE", "HPQ", "HRL", "HSIC",
    "HST", "HSY", "HUM", "HWM", "IBM", "ICE", "IDXX", "IEX", "IFF", "ILMN",
    "INCY", "INTC", "INTU", "INVH", "IP", "IPG", "IQV", "IR", "IRM", "ISRG",
    "IT", "ITW", "IVZ", "J", "JBHT", "JCI", "JKHY", "JNJ", "JNPR", "JPM",
    "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KLAC", "KMB", "KMI", "KMX",
    "KO", "KR", "L", "LDOS", "LEN", "LH", "LHX", "LIN", "LKQ", "LLY",
    "LMT", "LNC", "LNT", "LOW", "LRCX", "LUV", "LVS", "LW", "LYB",
    "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ",
    "MDT", "MET", "META", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM",
    "MNST", "MO", "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRO", "MS",
    "MSCI", "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NDSN",
    "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP",
    "NTRS", "NUE", "NVDA", "NVR", "NWL", "NWSA", "NXPI", "O", "ODFL",
    "OGN", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY", "PARA", "PAYC",
    "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG", "PG", "PGR", "PH",
    "PHM", "PKG", "PKI", "PLD", "PM", "PNC", "PNR", "PNW", "POOL", "PPG",
    "PPL", "PRU", "PSA", "PSX", "PTC", "PVH", "PWR", "PXD", "PYPL", "QCOM",
    "QRVO", "RCL", "RE", "REG", "REGN", "RF", "RHI", "RJF", "RL", "RMD",
    "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "SBAC", "SBUX", "SCHW",
    "SEE", "SHW", "SJM", "SLB", "SNA", "SNPS", "SO", "SPG", "SPGI",
    "SRE", "STE", "STT", "STX", "STZ", "SWK", "SWKS", "SYF", "SYK", "SYY",
    "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT",
    "TJX", "TMO", "TMUS", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA",
    "TSN", "TT", "TTWO", "TXN", "TXT", "TYL", "UAL", "UDR", "UHS", "ULTA",
    "UNH", "UNP", "UPS", "URI", "USB", "V", "VFC", "VICI", "VLO", "VMC",
    "VNO", "VRSK", "VRSN", "VRTX", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA",
    "WBD", "WDC", "WEC", "WELL", "WFC", "WHR", "WM", "WMB", "WMT", "WRB",
    "WRK", "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XRAY", "XYL", "YUM",
    "ZBH", "ZBRA", "ZION", "ZTS",
]
