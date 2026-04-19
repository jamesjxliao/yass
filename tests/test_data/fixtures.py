"""Synthetic Polars DataFrames for testing."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl


def make_fundamentals(n: int = 10) -> pl.DataFrame:
    """Create a synthetic fundamentals DataFrame with known values."""
    tickers = [f"TEST{i:03d}" for i in range(n)]
    return pl.DataFrame({
        "ticker": tickers,
        "market_cap": [1e9 * (i + 1) for i in range(n)],
        "close": [100.0 + i * 10 for i in range(n)],
        "avg_volume_20d": [1_000_000.0 * (i + 1) for i in range(n)],
        "beta": [0.8 + i * 0.1 for i in range(n)],
        "pe_ratio": [10.0 + i * 3 for i in range(n)],
        "roe": [0.05 + i * 0.03 for i in range(n)],
        "debt_to_equity": [0.5 + i * 0.2 for i in range(n)],
        "fcf_yield": [0.02 + i * 0.01 for i in range(n)],
        "earnings_yield": [0.03 + i * 0.015 for i in range(n)],
        "book_to_price": [0.3 + i * 0.15 for i in range(n)],
        "dividend_yield": [0.01 + i * 0.005 for i in range(n)],
        "revenue_growth": [0.05 + i * 0.02 for i in range(n)],
        "earnings_stability": [0.5 + i * 0.05 for i in range(n)],
        "rev_growth_current": [0.05 + i * 0.03 for i in range(n)],
        "rev_growth_prior": [0.08 + i * 0.01 for i in range(n)],
        "eps_growth_current": [0.10 + i * 0.04 for i in range(n)],
        "eps_growth_prior": [0.12 + i * 0.02 for i in range(n)],
        "op_margin_current": [0.10 + i * 0.02 for i in range(n)],
        "op_margin_prior": [0.08 + i * 0.015 for i in range(n)],
        "gross_margin_current": [0.30 + i * 0.03 for i in range(n)],
        "gross_margin_prior": [0.28 + i * 0.025 for i in range(n)],
    })


def make_prices(tickers: list[str] | None = None, days: int = 252) -> pl.DataFrame:
    """Create synthetic daily price data."""
    tickers = tickers or ["TEST000", "TEST001", "TEST002"]
    rows = []
    start = date(2024, 1, 2)
    for ticker in tickers:
        price = 100.0
        for d in range(days):
            current = start + timedelta(days=d)
            if current.weekday() < 5:
                price *= 1.0005  # slight upward drift
                rows.append({
                    "ticker": ticker,
                    "date": current,
                    "open": price * 0.999,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1_000_000.0,
                })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
