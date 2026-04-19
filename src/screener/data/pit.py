from __future__ import annotations

from datetime import date

import polars as pl

from screener.data.cache import CacheManager


class PITQuery:
    """Point-in-time query logic. Returns data as it was known at a given date."""

    def __init__(self, cache: CacheManager):
        self._cache = cache

    def get_as_of(self, ticker: str, field: str, as_of_date: date) -> float | None:
        """Return the value of a field for a ticker as it was known on as_of_date."""
        result = self._cache.to_polars(
            """SELECT value FROM pit_snapshots
               WHERE ticker = ? AND field = ? AND observed_at <= ?
               ORDER BY observed_at DESC LIMIT 1""",
            [ticker, field, as_of_date.isoformat()],
        )
        if result.is_empty():
            return None
        return result["value"][0]

    def get_fundamentals_as_of(
        self, tickers: list[str], fields: list[str], as_of_date: date
    ) -> pl.DataFrame:
        """Return fundamentals for multiple tickers as known on as_of_date.

        Uses a single SQL query with window functions instead of N*M individual queries.
        Returns a DataFrame with columns: ticker, <field1>, <field2>, ...
        """
        if not tickers or not fields:
            return pl.DataFrame()

        as_of = as_of_date.isoformat()

        # Single query: get the latest value per (ticker, field) where observed_at <= as_of_date
        raw = self._cache.to_polars(
            """SELECT ticker, field, value,
                      ROW_NUMBER() OVER (
                          PARTITION BY ticker, field
                          ORDER BY observed_at DESC
                      ) AS rn
               FROM pit_snapshots
               WHERE ticker IN (SELECT UNNEST(?::VARCHAR[]))
                 AND field IN (SELECT UNNEST(?::VARCHAR[]))
                 AND observed_at <= ?""",
            [tickers, fields, as_of],
        )

        if raw.is_empty():
            # Return DataFrame with ticker column and null fields
            return pl.DataFrame({"ticker": tickers})

        # Keep only the most recent observation per (ticker, field)
        latest = raw.filter(pl.col("rn") == 1).select("ticker", "field", "value")

        # Pivot from long to wide format
        pivoted = latest.pivot(on="field", index="ticker", values="value")

        # Ensure all requested tickers are present (left join with full ticker list)
        all_tickers = pl.DataFrame({"ticker": tickers})
        result = all_tickers.join(pivoted, on="ticker", how="left")

        return result
