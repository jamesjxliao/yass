from __future__ import annotations

from datetime import date

import polars as pl

from screener.data.cache import CacheManager
from screener.data.fmp import _FIELD_MAP
from screener.data.pit import PITQuery
from screener.data.provider import DataProvider
from screener.data.universe import UniverseManager

# Derive PIT fields from the FMP field mapping (single source of truth)
PIT_FIELDS = list(_FIELD_MAP.values())


class PITDataServer:
    """Point-in-time data server for backtesting.

    Serves data as it would have been known on a given date,
    preventing lookahead bias.
    """

    def __init__(
        self,
        provider: DataProvider,
        cache: CacheManager,
        universe_manager: UniverseManager,
    ):
        self._provider = provider
        self._cache = cache
        self._universe = universe_manager
        self._pit = PITQuery(cache)
        self._has_pit_data: bool | None = None

    def _check_pit_data(self) -> bool:
        """Check whether pit_snapshots has data (memoized)."""
        if self._has_pit_data is None:
            result = self._cache.to_polars(
                "SELECT COUNT(*) AS cnt FROM pit_snapshots"
            )
            self._has_pit_data = result["cnt"][0] > 0
        return self._has_pit_data

    def _get_latest_prices(
        self, tickers: list[str], as_of_date: date
    ) -> pl.DataFrame:
        """Get close price and 20-day avg volume as of date from price_cache."""
        result = self._cache.to_polars(
            """WITH recent AS (
                   SELECT ticker, close, volume,
                          ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn,
                          AVG(volume) OVER (
                              PARTITION BY ticker ORDER BY date DESC
                              ROWS BETWEEN CURRENT ROW AND 19 FOLLOWING
                          ) AS avg_vol_20d
                   FROM price_cache
                   WHERE ticker IN (SELECT UNNEST(?::VARCHAR[]))
                     AND date <= ?::DATE
               )
               SELECT ticker, close, avg_vol_20d AS avg_volume_20d
               FROM recent WHERE rn = 1""",
            [tickers, str(as_of_date)],
        )
        return result

    def get_tradeable_tickers(self, as_of_date: date, lookback_days: int = 30) -> list[str]:
        """Return tickers that had price data near as_of_date.

        A stock is tradeable if it has at least one price record in the
        lookback window. This naturally excludes delisted/acquired stocks
        without needing historical index constituency data.
        """
        start = str(as_of_date - __import__("datetime").timedelta(days=lookback_days))
        result = self._cache.to_polars(
            """SELECT DISTINCT ticker FROM price_cache
               WHERE date >= ?::DATE AND date <= ?::DATE""",
            [start, str(as_of_date)],
        )
        if result.is_empty():
            return []
        return result["ticker"].to_list()

    def get_universe_as_of(self, index: str, as_of_date: date) -> list[str]:
        """Return tickers in universe at as_of_date (survivorship-bias-free).

        Intersects universe membership with tradeable tickers from price_cache.
        A stock must be both in the universe AND have recent price data.
        """
        members = set(self._universe.get_active(index, as_of_date))
        if not members:
            # No membership data — fall back to all tradeable
            return self.get_tradeable_tickers(as_of_date)
        tradeable = set(self.get_tradeable_tickers(as_of_date))
        return list(members & tradeable) if tradeable else list(members)

    def _get_sectors(self, tickers: list[str]) -> pl.DataFrame:
        """Get sectors from the sector_map table."""
        return self._cache.get_sectors(tickers)

    def get_screening_data(
        self, tickers: list[str], as_of_date: date
    ) -> pl.DataFrame:
        """Return fundamentals as known on as_of_date.

        Merges PIT fundamental snapshots with latest price data from cache.
        """
        if self._check_pit_data():
            df = self._pit.get_fundamentals_as_of(
                tickers, PIT_FIELDS, as_of_date
            )
            if not df.is_empty() and len(df.columns) > 1:
                # Drop stale close/volume from PIT fundamentals before joining
                # actual market prices (PIT close is from key-metrics, not market)
                drop_cols = [c for c in ("close", "avg_volume_20d") if c in df.columns]
                if drop_cols:
                    df = df.drop(drop_cols)
                # Merge in close price and volume from price_cache
                prices = self._get_latest_prices(tickers, as_of_date)
                if not prices.is_empty():
                    df = df.join(prices, on="ticker", how="left")
                # Merge in sector from cached profiles
                sectors = self._get_sectors(df["ticker"].to_list())
                if not sectors.is_empty():
                    df = df.join(sectors, on="ticker", how="left")
                return df

        # Fallback: use provider directly (current data, not PIT-correct)
        return self._provider.get_fundamentals(tickers)
