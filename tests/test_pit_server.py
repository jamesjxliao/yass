from __future__ import annotations

import datetime
from unittest.mock import MagicMock

import polars as pl
from screener.backtest.pit_server import PITDataServer
from screener.data.cache import CacheManager


def test_close_column_not_leaked_from_pit(cache: CacheManager):
    """PIT close from key-metrics should be dropped in favor of market close."""
    # Store a PIT snapshot with a stale 'close' value
    cache.record_pit_snapshot("AAPL", "close", 150.0, "2024-01-01", "fmp", "2024-04-01")
    cache.record_pit_snapshot("AAPL", "roe", 0.25, "2024-01-01", "fmp_qkm", "2024-02-15")

    # Store actual market price (different from PIT close)
    prices = pl.DataFrame({
        "ticker": ["AAPL"],
        "date": ["2024-03-15"],
        "open": [175.0],
        "high": [176.0],
        "low": [174.0],
        "close": [175.0],  # actual market close, NOT 150
        "volume": [1_000_000.0],
    }).with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
    cache.store_prices(prices, source="test")

    provider = MagicMock()
    um = MagicMock()
    pit_server = PITDataServer(provider, cache, um)

    df = pit_server.get_screening_data(["AAPL"], datetime.date(2024, 4, 1))

    assert "close" in df.columns
    # close should be 175.0 (market price), NOT 150.0 (PIT snapshot)
    assert df["close"][0] == 175.0


def test_in_memory_price_frame_matches_db(cache: CacheManager):
    """The backtest's in-memory price-frame path must return identical latest
    close / 20-day avg volume / tradeable tickers as the DB query path — this is
    what lets run_backtest skip re-scanning price_cache every rebalance."""
    rows = []
    for ticker, base in [("AAPL", 100.0), ("MSFT", 200.0)]:
        for i in range(25):
            d = datetime.date(2024, 3, 1) + datetime.timedelta(days=i)
            if d.weekday() < 5:
                rows.append({
                    "ticker": ticker, "date": d, "open": base, "high": base + 1,
                    "low": base - 1, "close": base + i,
                    "volume": 1_000_000.0 + i * 100_000.0,
                })
    prices = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
    cache.store_prices(prices, source="test")

    pit = PITDataServer(MagicMock(), cache, MagicMock())
    as_of = datetime.date(2024, 3, 30)
    tickers = ["AAPL", "MSFT"]

    db_prices = pit._get_latest_prices(tickers, as_of).sort("ticker")
    db_tradeable = sorted(pit.get_tradeable_tickers(as_of))

    pit.use_price_frame(prices)
    mem_prices = pit._get_latest_prices(tickers, as_of).sort("ticker")
    mem_tradeable = sorted(pit.get_tradeable_tickers(as_of))

    assert mem_tradeable == db_tradeable == ["AAPL", "MSFT"]
    assert mem_prices["ticker"].to_list() == db_prices["ticker"].to_list()
    for col in ("close", "avg_volume_20d"):
        for a, b in zip(mem_prices[col].to_list(), db_prices[col].to_list()):
            assert abs(a - b) < 1e-9, f"{col} diverged: {a} vs {b}"


def test_avg_volume_is_not_single_day(cache: CacheManager):
    """avg_volume_20d should be a 20-day average, not a single day's volume."""
    # Store 25 days of prices with varying volume
    rows = []
    for i in range(25):
        d = datetime.date(2024, 3, 1) + datetime.timedelta(days=i)
        if d.weekday() < 5:  # skip weekends
            rows.append({
                "ticker": "AAPL",
                "date": d,
                "open": 175.0,
                "high": 176.0,
                "low": 174.0,
                "close": 175.0,
                "volume": 1_000_000.0 + i * 100_000.0,
            })

    prices = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
    cache.store_prices(prices, source="test")

    # Store a PIT snapshot so PIT path is used
    cache.record_pit_snapshot("AAPL", "roe", 0.25, "2024-01-01", "fmp_qkm", "2024-02-15")

    provider = MagicMock()
    um = MagicMock()
    pit_server = PITDataServer(provider, cache, um)

    df = pit_server.get_screening_data(["AAPL"], datetime.date(2024, 3, 30))

    assert "avg_volume_20d" in df.columns
    vol = df["avg_volume_20d"][0]
    # Should be an average, not the last day's volume
    last_day_vol = max(r["volume"] for r in rows if r["date"].weekday() < 5)
    assert vol != last_day_vol, "avg_volume_20d should be averaged, not single-day"
