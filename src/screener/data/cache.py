from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import duckdb
import polars as pl

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS api_cache (
    cache_key   VARCHAR PRIMARY KEY,
    provider    VARCHAR NOT NULL,
    endpoint    VARCHAR NOT NULL,
    fetched_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at  TIMESTAMP NOT NULL,
    data_json   JSON NOT NULL
);

CREATE TABLE IF NOT EXISTS pit_snapshots (
    ticker      VARCHAR NOT NULL,
    field       VARCHAR NOT NULL,
    value       DOUBLE,
    report_date DATE NOT NULL,
    observed_at TIMESTAMP NOT NULL,
    source      VARCHAR NOT NULL,
    PRIMARY KEY (ticker, field, report_date, observed_at)
);

CREATE TABLE IF NOT EXISTS universe_membership (
    ticker       VARCHAR NOT NULL,
    index_name   VARCHAR NOT NULL,
    added_date   DATE,
    removed_date DATE,
    is_delisted  BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (ticker, index_name, added_date)
);

CREATE TABLE IF NOT EXISTS price_cache (
    ticker     VARCHAR NOT NULL,
    date       DATE NOT NULL,
    open       DOUBLE,
    high       DOUBLE,
    low        DOUBLE,
    close      DOUBLE,
    volume     DOUBLE,
    source     VARCHAR NOT NULL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS sector_map (
    ticker     VARCHAR PRIMARY KEY,
    sector     VARCHAR NOT NULL
);
"""


class CacheManager:
    def __init__(self, db_path: Path | str = ":memory:"):
        self._conn = duckdb.connect(str(db_path))
        self._conn.execute("SET TimeZone = 'UTC'")
        for statement in SCHEMA_SQL.strip().split(";"):
            statement = statement.strip()
            if statement:
                self._conn.execute(statement)

    def close(self):
        self._conn.close()

    @staticmethod
    def _make_key(provider: str, endpoint: str, params: dict | None = None) -> str:
        raw = f"{provider}:{endpoint}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get_or_fetch_if_cached(
        self, provider: str, endpoint: str, params: dict | None = None
    ) -> list[dict[str, Any]] | None:
        """Check cache only — return data if cached and not expired, else None."""
        key = self._make_key(provider, endpoint, params)
        result = self._conn.execute(
            """SELECT data_json FROM api_cache
               WHERE cache_key = ? AND expires_at > CURRENT_TIMESTAMP""",
            [key],
        ).fetchone()
        if result is not None:
            return json.loads(result[0])
        return None

    def get_or_fetch(
        self,
        provider: str,
        endpoint: str,
        fetch_fn: Callable[[], list[dict[str, Any]]],
        ttl_hours: int = 24,
        params: dict | None = None,
    ) -> list[dict[str, Any]]:
        cached = self.get_or_fetch_if_cached(provider, endpoint, params)
        if cached is not None:
            return cached

        logger.debug("Cache miss: %s/%s", provider, endpoint)
        key = self._make_key(provider, endpoint, params)
        data = fetch_fn()
        expires_at = datetime.now(UTC) + timedelta(hours=ttl_hours)
        self._conn.execute(
            """INSERT OR REPLACE INTO api_cache
               (cache_key, provider, endpoint, expires_at, data_json)
               VALUES (?, ?, ?, ?, ?)""",
            [key, provider, endpoint, expires_at, json.dumps(data)],
        )
        return data

    def record_pit_snapshot(
        self,
        ticker: str,
        field: str,
        value: float | None,
        report_date: str,
        source: str,
        observed_at: str | None = None,
    ) -> None:
        """Record a point-in-time observation. Only inserts if the value has changed.

        If observed_at is None, uses current timestamp.
        """
        latest = self._conn.execute(
            """SELECT value FROM pit_snapshots
               WHERE ticker = ? AND field = ? AND report_date = ?
               ORDER BY observed_at DESC LIMIT 1""",
            [ticker, field, report_date],
        ).fetchone()

        if latest is not None and latest[0] == value:
            return

        ts = observed_at or datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")
        self._conn.execute(
            """INSERT OR REPLACE INTO pit_snapshots
               (ticker, field, value, report_date, observed_at, source)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [ticker, field, value, report_date, ts, source],
        )

    def bulk_store_pit_snapshots(self, snapshots: list[tuple]) -> None:
        """Bulk-insert PIT snapshots via Arrow for performance.

        Each tuple is (ticker, field, value, report_date, source, observed_at).
        Uses INSERT OR REPLACE in a single transaction.
        """
        if not snapshots:
            return
        df = pl.DataFrame(
            snapshots,
            schema=["ticker", "field", "value", "report_date", "source", "observed_at"],
            orient="row",
        ).with_columns(
            pl.col("report_date").str.to_date("%Y-%m-%d"),
            pl.col("observed_at").str.to_datetime("%Y-%m-%d"),
        )
        self._conn.register("_pit_bulk", df.to_arrow())
        self._conn.execute(
            """INSERT OR REPLACE INTO pit_snapshots
               (ticker, field, value, report_date, observed_at, source)
               SELECT ticker, field, value, report_date, observed_at, source
               FROM _pit_bulk"""
        )
        self._conn.unregister("_pit_bulk")

    def get_cached_price_range(self, ticker: str) -> tuple[str, str] | None:
        """Return (min_date, max_date) of cached prices for a ticker, or None."""
        count = self._conn.execute(
            "SELECT COUNT(*) FROM price_cache WHERE ticker = ?", [ticker]
        ).fetchone()[0]
        if count == 0:
            return None
        result = self._conn.execute(
            "SELECT MIN(date), MAX(date) FROM price_cache WHERE ticker = ?",
            [ticker],
        ).fetchone()
        return (str(result[0]), str(result[1]))

    def store_prices(self, df: pl.DataFrame, source: str) -> int:
        """Store price data in price_cache. Returns number of rows inserted.

        Uses INSERT OR IGNORE to skip duplicates (historical prices are immutable).
        """
        if df.is_empty():
            return 0
        to_insert = df.with_columns(pl.lit(source).alias("source"))
        self._conn.register("_prices_to_insert", to_insert.to_arrow())
        self._conn.execute(
            """INSERT OR IGNORE INTO price_cache
               (ticker, date, open, high, low, close, volume, source)
               SELECT ticker, date, open, high, low, close, volume, source
               FROM _prices_to_insert"""
        )
        self._conn.unregister("_prices_to_insert")
        return len(to_insert)

    def get_prices(self, tickers: list[str], start: str, end: str) -> pl.DataFrame:
        """Read cached prices for given tickers and date range."""
        if not tickers:
            return pl.DataFrame()
        return self.to_polars(
            """SELECT ticker, date, open, high, low, close, volume
               FROM price_cache
               WHERE ticker IN (SELECT UNNEST(?::VARCHAR[]))
                 AND date >= ?::DATE AND date <= ?::DATE
               ORDER BY ticker, date""",
            [tickers, start, end],
        )

    def store_sector(self, ticker: str, sector: str) -> None:
        """Store ticker→sector mapping."""
        self._conn.execute(
            "INSERT OR REPLACE INTO sector_map (ticker, sector) VALUES (?, ?)",
            [ticker, sector],
        )

    def get_sectors(self, tickers: list[str]) -> pl.DataFrame:
        """Get sectors for given tickers."""
        if not tickers:
            return pl.DataFrame(schema={"ticker": pl.Utf8, "sector": pl.Utf8})
        return self.to_polars(
            """SELECT ticker, sector FROM sector_map
               WHERE ticker IN (SELECT UNNEST(?::VARCHAR[]))""",
            [tickers],
        )

    def to_polars(self, sql: str, params: list | None = None) -> pl.DataFrame:
        """Execute SQL and return result as a Polars DataFrame."""
        if params:
            result = self._conn.execute(sql, params)
        else:
            result = self._conn.execute(sql)
        return result.pl()
