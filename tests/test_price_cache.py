from __future__ import annotations

from datetime import date

import polars as pl
from screener.data.cache import CacheManager


def _make_price_df(ticker: str, start: date, days: int) -> pl.DataFrame:
    """Create a simple price DataFrame for testing."""
    from datetime import timedelta

    rows = []
    for i in range(days):
        d = start + timedelta(days=i)
        if d.weekday() < 5:  # skip weekends
            rows.append({
                "ticker": ticker,
                "date": d,
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 1_000_000.0,
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_store_and_retrieve_prices():
    cache = CacheManager(":memory:")
    df = _make_price_df("AAPL", date(2024, 1, 2), 10)
    stored = cache.store_prices(df, source="test")
    assert stored > 0

    result = cache.get_prices(["AAPL"], "2024-01-02", "2024-01-15")
    assert len(result) == len(df)
    assert result["ticker"][0] == "AAPL"
    cache.close()


def test_cached_price_range():
    cache = CacheManager(":memory:")
    df = _make_price_df("MSFT", date(2024, 3, 1), 20)
    cache.store_prices(df, source="test")

    r = cache.get_cached_price_range("MSFT")
    assert r is not None
    min_date, max_date = r
    assert min_date == str(df["date"].min())
    assert max_date == str(df["date"].max())

    # Non-existent ticker
    assert cache.get_cached_price_range("NOPE") is None
    cache.close()


def test_prices_no_duplicates():
    """Storing the same data twice should not create duplicates."""
    cache = CacheManager(":memory:")
    df = _make_price_df("AAPL", date(2024, 1, 2), 5)
    cache.store_prices(df, source="test")
    cache.store_prices(df, source="test")  # store again

    result = cache.get_prices(["AAPL"], "2024-01-01", "2024-01-31")
    assert len(result) == len(df)
    cache.close()


def test_prices_partial_range():
    """Requesting a subset of cached data should return only that subset."""
    cache = CacheManager(":memory:")
    df = _make_price_df("AAPL", date(2024, 1, 2), 30)
    cache.store_prices(df, source="test")

    # Request only first week
    result = cache.get_prices(["AAPL"], "2024-01-02", "2024-01-05")
    assert len(result) < len(df)
    assert all(d <= date(2024, 1, 5) for d in result["date"].to_list())
    cache.close()


def test_prices_multiple_tickers():
    cache = CacheManager(":memory:")
    df_aapl = _make_price_df("AAPL", date(2024, 1, 2), 10)
    df_msft = _make_price_df("MSFT", date(2024, 1, 2), 10)
    cache.store_prices(df_aapl, source="test")
    cache.store_prices(df_msft, source="test")

    result = cache.get_prices(["AAPL", "MSFT"], "2024-01-02", "2024-01-15")
    tickers = result["ticker"].unique().to_list()
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    cache.close()


def test_get_or_fetch_if_cached():
    cache = CacheManager(":memory:")

    # Should return None when not cached
    assert cache.get_or_fetch_if_cached("test", "endpoint") is None

    # Store something
    cache.get_or_fetch("test", "endpoint", lambda: [{"a": 1}], ttl_hours=1)

    # Should now return the cached data
    result = cache.get_or_fetch_if_cached("test", "endpoint")
    assert result == [{"a": 1}]
    cache.close()
