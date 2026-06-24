from __future__ import annotations

import logging
from datetime import date

from screener.data.cache import CacheManager
from screener.data.provider import DataProvider

logger = logging.getLogger(__name__)


class UniverseManager:
    """Manages universe membership over time for survivorship-bias-free backtesting."""

    def __init__(self, cache: CacheManager):
        self._cache = cache

    def sync(self, provider: DataProvider, index: str) -> int:
        """Fetch current constituency from provider and update membership table.

        Returns the number of active tickers.
        """
        tickers = provider.get_universe(index)
        today = date.today().isoformat()

        for ticker in tickers:
            existing = self._cache.to_polars(
                """SELECT ticker FROM universe_membership
                   WHERE ticker = ? AND index_name = ? AND removed_date IS NULL""",
                [ticker, index],
            )
            if existing.is_empty():
                self._cache._conn.execute(
                    """INSERT INTO universe_membership (ticker, index_name, added_date)
                       VALUES (?, ?, ?)""",
                    [ticker, index, today],
                )

        # Mark removed tickers
        current_set = set(tickers)
        all_active = self._cache.to_polars(
            """SELECT ticker FROM universe_membership
               WHERE index_name = ? AND removed_date IS NULL""",
            [index],
        )
        if not all_active.is_empty():
            for row in all_active.iter_rows(named=True):
                if row["ticker"] not in current_set:
                    self._cache._conn.execute(
                        """UPDATE universe_membership
                           SET removed_date = ?
                           WHERE ticker = ? AND index_name = ? AND removed_date IS NULL""",
                        [today, row["ticker"], index],
                    )

        logger.info("Synced universe '%s': %d active tickers", index, len(tickers))
        return len(tickers)

    def get_active(self, index: str, as_of_date: date | None = None) -> list[str]:
        """Return tickers active in the universe at the given date.

        Includes tickers that were later delisted (survivorship-bias-free).
        If as_of_date is None, returns currently active tickers.
        """
        if as_of_date is None:
            result = self._cache.to_polars(
                """SELECT ticker FROM universe_membership
                   WHERE index_name = ? AND removed_date IS NULL""",
                [index],
            )
        else:
            as_of = as_of_date.isoformat()
            result = self._cache.to_polars(
                # `removed_date <= added_date` marks a corrupt row from a ticker
                # that LEFT and RE-JOINED the index (the builder paired a later
                # add with an earlier remove). Such a row is unsatisfiable for
                # every as_of, silently dropping a real member (EQT, PCG, DD/DOW,
                # SNDK, FOX/FOXA...) for ALL dates — survivorship bias. Treat the
                # spurious removed_date as NULL (active from added_date onward).
                """SELECT ticker FROM universe_membership
                   WHERE index_name = ?
                     AND added_date <= ?
                     AND (removed_date IS NULL
                          OR removed_date <= added_date
                          OR removed_date > ?)""",
                [index, as_of, as_of],
            )

        if result.is_empty():
            return []
        return result["ticker"].to_list()

    def bulk_load_history(self, index: str, records: list[dict]) -> int:
        """Bulk-load historical membership records.

        Each record has: ticker, added_date, removed_date (or None).
        """
        if not records:
            return 0

        import polars as pl

        df = pl.DataFrame(records).with_columns(
            pl.lit(index).alias("index_name"),
            pl.lit(False).alias("is_delisted"),
        )
        for col in ["added_date", "removed_date"]:
            if col in df.columns:
                dtype = df[col].dtype
                if dtype == pl.String or dtype == pl.Utf8:
                    df = df.with_columns(
                        pl.col(col).str.to_date("%Y-%m-%d", strict=False)
                    )
                elif dtype == pl.Null:
                    df = df.with_columns(
                        pl.lit(None).cast(pl.Date).alias(col)
                    )

        self._cache._conn.execute(
            "DELETE FROM universe_membership WHERE index_name = ?", [index]
        )
        self._cache._conn.register("_membership_bulk", df.to_arrow())
        self._cache._conn.execute(
            """INSERT INTO universe_membership
               (ticker, index_name, added_date, removed_date, is_delisted)
               SELECT ticker, index_name, added_date, removed_date, is_delisted
               FROM _membership_bulk"""
        )
        self._cache._conn.unregister("_membership_bulk")
        logger.info("Bulk-loaded %d membership records for '%s'", len(records), index)
        return len(records)
