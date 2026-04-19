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
                """SELECT ticker FROM universe_membership
                   WHERE index_name = ?
                     AND added_date <= ?
                     AND (removed_date IS NULL OR removed_date > ?)""",
                [index, as_of, as_of],
            )

        if result.is_empty():
            return []
        return result["ticker"].to_list()
