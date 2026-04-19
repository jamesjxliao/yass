from __future__ import annotations

from screener.data.cache import CacheManager


def test_cache_round_trip(cache: CacheManager):
    """Data stored in cache should be retrievable."""
    data = [{"ticker": "AAPL", "price": 150.0}, {"ticker": "MSFT", "price": 300.0}]

    result = cache.get_or_fetch(
        provider="test",
        endpoint="prices",
        fetch_fn=lambda: data,
        ttl_hours=1,
    )
    assert result == data

    # Second call should hit cache (fetch_fn would raise if called)
    result2 = cache.get_or_fetch(
        provider="test",
        endpoint="prices",
        fetch_fn=lambda: (_ for _ in ()).throw(RuntimeError("should not be called")),
        ttl_hours=1,
    )
    assert result2 == data


def test_cache_different_params(cache: CacheManager):
    """Different params should produce different cache entries."""
    data_a = [{"v": "a"}]
    data_b = [{"v": "b"}]

    result_a = cache.get_or_fetch("test", "ep", lambda: data_a, params={"x": 1})
    result_b = cache.get_or_fetch("test", "ep", lambda: data_b, params={"x": 2})

    assert result_a == data_a
    assert result_b == data_b


def test_pit_snapshot_records_and_deduplicates(cache: CacheManager):
    """PIT snapshots should deduplicate same values, store different observed_at."""
    cache.record_pit_snapshot(
        "AAPL", "pe_ratio", 25.0, "2024-03-31", "fmp", observed_at="2024-06-30"
    )
    cache.record_pit_snapshot(
        "AAPL", "pe_ratio", 25.0, "2024-03-31", "fmp", observed_at="2024-07-01"
    )  # same value, skipped
    cache.record_pit_snapshot(
        "AAPL", "pe_ratio", 28.0, "2024-03-31", "fmp", observed_at="2024-09-30"
    )  # new value

    result = cache.to_polars(
        "SELECT * FROM pit_snapshots WHERE ticker = 'AAPL' AND field = 'pe_ratio'"
    )
    assert len(result) == 2  # only 2 distinct values


def test_bulk_store_pit_snapshots(cache: CacheManager):
    """Bulk insert should store all snapshots and handle duplicates."""
    snapshots = [
        ("AAPL", "roe", 0.25, "2024-03-31", "fmp_qkm", "2024-05-15"),
        ("AAPL", "roic", 0.15, "2024-03-31", "fmp_qkm", "2024-05-15"),
        ("MSFT", "roe", 0.30, "2024-03-31", "fmp_qkm", "2024-05-15"),
    ]
    cache.bulk_store_pit_snapshots(snapshots)

    result = cache.to_polars("SELECT * FROM pit_snapshots ORDER BY ticker, field")
    assert len(result) == 3
    assert result["ticker"].to_list() == ["AAPL", "AAPL", "MSFT"]

    # Insert again with updated value — should replace
    updated = [("AAPL", "roe", 0.28, "2024-03-31", "fmp_qkm", "2024-05-15")]
    cache.bulk_store_pit_snapshots(updated)
    result2 = cache.to_polars(
        "SELECT value FROM pit_snapshots WHERE ticker='AAPL' AND field='roe'"
    )
    assert result2["value"][0] == 0.28


def test_bulk_store_pit_snapshots_empty(cache: CacheManager):
    """Bulk insert with empty list should be a no-op."""
    cache.bulk_store_pit_snapshots([])
    result = cache.to_polars("SELECT COUNT(*) as cnt FROM pit_snapshots")
    assert result["cnt"][0] == 0


def test_to_polars(cache: CacheManager):
    """SQL results should convert to Polars DataFrames."""
    cache._conn.execute(
        """INSERT INTO universe_membership (ticker, index_name, added_date)
           VALUES ('AAPL', 'sp500', '2024-01-01')"""
    )
    df = cache.to_polars("SELECT ticker FROM universe_membership WHERE index_name = 'sp500'")
    assert df["ticker"].to_list() == ["AAPL"]
