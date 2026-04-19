from __future__ import annotations

import random
from datetime import date, timedelta

import polars as pl

# Stable seed for reproducible mock data
_RNG = random.Random(42)

# Realistic S&P 500 subset for development
MOCK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "JPM", "JNJ",
    "V", "UNH", "PG", "XOM", "MA", "HD", "CVX", "MRK", "ABBV", "LLY",
    "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO", "ACN", "TMO", "ABT",
    "DHR", "CRM", "NKE", "TXN", "NEE", "PM", "UPS", "RTX", "ORCL", "HON",
    "QCOM", "LOW", "MS", "GS", "BLK", "AMGN", "INTC", "AMD", "CAT", "BA",
]


class MockProvider:
    """Mock data provider with synthetic but realistic data for development and testing."""

    def get_fundamentals(self, tickers: list[str]) -> pl.DataFrame:
        rows = []
        for ticker in tickers:
            seed = hash(ticker) % 10000
            rng = random.Random(seed)
            rows.append({
                "ticker": ticker,
                "market_cap": rng.uniform(10e9, 3e12),
                "close": rng.uniform(20.0, 500.0),
                "avg_volume_20d": rng.uniform(500_000, 50_000_000),
                "beta": rng.uniform(0.5, 2.0),
                "pe_ratio": rng.uniform(5.0, 60.0),
                "roe": rng.uniform(-0.05, 0.50),
                "debt_to_equity": rng.uniform(0.0, 3.0),
                "fcf_yield": rng.uniform(-0.02, 0.15),
                "earnings_yield": rng.uniform(0.01, 0.20),
                "book_to_price": rng.uniform(0.1, 2.0),
                "dividend_yield": rng.uniform(0.0, 0.06),
                "revenue_growth": rng.uniform(-0.10, 0.40),
                "earnings_stability": rng.uniform(0.3, 1.0),
                "rev_growth_current": rng.uniform(-0.10, 0.40),
                "rev_growth_prior": rng.uniform(-0.10, 0.40),
                "eps_growth_current": rng.uniform(-0.30, 0.60),
                "eps_growth_prior": rng.uniform(-0.30, 0.60),
                "op_margin_current": rng.uniform(0.05, 0.35),
                "op_margin_prior": rng.uniform(0.05, 0.35),
                "gross_margin_current": rng.uniform(0.20, 0.70),
                "gross_margin_prior": rng.uniform(0.20, 0.70),
            })
        return pl.DataFrame(rows)

    def get_prices(self, tickers: list[str], start: date, end: date) -> pl.DataFrame:
        rows = []
        for ticker in tickers:
            seed = hash(ticker) % 10000
            rng = random.Random(seed)
            price = rng.uniform(20.0, 500.0)
            current = start
            while current <= end:
                # Skip weekends
                if current.weekday() < 5:
                    daily_return = rng.gauss(0.0003, 0.015)
                    price *= 1 + daily_return
                    high = price * (1 + abs(rng.gauss(0, 0.005)))
                    low = price * (1 - abs(rng.gauss(0, 0.005)))
                    rows.append({
                        "ticker": ticker,
                        "date": current,
                        "open": price * (1 + rng.gauss(0, 0.002)),
                        "high": high,
                        "low": low,
                        "close": price,
                        "volume": rng.uniform(500_000, 50_000_000),
                    })
                current += timedelta(days=1)
        return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))

    def get_universe(self, index: str) -> list[str]:
        if index == "sp500":
            return MOCK_TICKERS
        raise ValueError(f"Mock only supports 'sp500', got '{index}'")

    def get_delisted(self) -> list[str]:
        return []
