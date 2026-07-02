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


def test_invalidate_prices():
    cache = CacheManager(":memory:")
    df = _make_price_df("AAPL", date(2024, 1, 2), 10)
    cache.store_prices(df, source="test")
    assert len(cache.get_prices(["AAPL"], "2024-01-01", "2024-01-31")) > 0

    cache.invalidate_prices(["AAPL"])
    assert len(cache.get_prices(["AAPL"], "2024-01-01", "2024-01-31")) == 0
    assert cache.get_cached_price_range("AAPL") is None
    cache.close()


def test_split_refetch_failure_preserves_history():
    """A split-triggered re-fetch that returns empty (transient API failure) must
    NOT wipe the ticker's existing cached history."""
    from unittest.mock import MagicMock

    from screener.data.fmp import CachedFMPProvider, FMPProvider

    cache = CacheManager(":memory:")
    # Seed a full history whose last two closes look like a forward split
    # (>40% single-day drop) so detect_splits fires.
    df = _make_price_df("BKNG", date(2024, 1, 2), 20)
    df = df.with_columns(
        pl.when(pl.col("date") == df["date"].max())
        .then(pl.col("close") / 3.0)  # engineered split-day drop
        .otherwise(pl.col("close"))
        .alias("close")
    )
    cache.store_prices(df, source="test")
    n_before = len(cache.get_prices(["BKNG"], "2024-01-01", "2024-12-31"))
    assert n_before > 0

    fmp = FMPProvider(api_key="k")
    fmp.get_prices_single = MagicMock(return_value=pl.DataFrame())  # re-fetch fails
    provider = CachedFMPProvider(fmp, cache)

    result = provider.get_prices(["BKNG"], date(2024, 1, 2), date(2024, 1, 31))

    # Re-fetch was attempted, but the empty result must leave history intact.
    fmp.get_prices_single.assert_called()
    assert cache.get_cached_price_range("BKNG") is not None
    assert len(cache.get_prices(["BKNG"], "2024-01-01", "2024-12-31")) == n_before
    assert not result.is_empty()
    cache.close()


def test_detect_splits():
    from screener.data.splits import detect_splits

    normal = pl.DataFrame({
        "ticker": ["AAPL"] * 5,
        "date": [date(2024, 1, i) for i in range(1, 6)],
        "close": [100.0, 101.0, 99.5, 102.0, 100.5],
    })
    assert detect_splits(normal) == []

    # Forward split: price drops by half
    split = pl.DataFrame({
        "ticker": ["BKNG"] * 5,
        "date": [date(2024, 6, i) for i in range(24, 29)],
        "close": [5500.0, 5520.0, 5480.0, 110.0, 112.0],
    })
    assert detect_splits(split) == ["BKNG"]

    # Reverse split: price doubles
    reverse = pl.DataFrame({
        "ticker": ["XYZ"] * 4,
        "date": [date(2024, 3, i) for i in range(1, 5)],
        "close": [5.0, 5.1, 15.0, 15.2],
    })
    assert detect_splits(reverse) == ["XYZ"]

    # Mixed: one split, one normal
    mixed = pl.concat([normal, split])
    assert sorted(detect_splits(mixed)) == ["BKNG"]

    # Empty
    assert detect_splits(pl.DataFrame()) == []


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
