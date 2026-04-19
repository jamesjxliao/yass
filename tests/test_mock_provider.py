from __future__ import annotations

from datetime import date

from screener.data.mock import MockProvider


def test_mock_fundamentals():
    provider = MockProvider()
    df = provider.get_fundamentals(["AAPL", "MSFT", "GOOGL"])
    assert len(df) == 3
    assert "ticker" in df.columns
    assert "market_cap" in df.columns
    assert "pe_ratio" in df.columns
    # All values should be positive for these fields
    assert (df["market_cap"] > 0).all()


def test_mock_prices():
    provider = MockProvider()
    df = provider.get_prices(["AAPL"], start=date(2024, 1, 2), end=date(2024, 1, 31))
    assert len(df) > 0
    assert set(df.columns) == {"ticker", "date", "open", "high", "low", "close", "volume"}
    # Prices should be positive
    assert (df["close"] > 0).all()


def test_mock_universe():
    provider = MockProvider()
    tickers = provider.get_universe("sp500")
    assert len(tickers) == 50
    assert "AAPL" in tickers


def test_mock_deterministic():
    """Same ticker should produce same data across calls."""
    p1 = MockProvider()
    p2 = MockProvider()
    df1 = p1.get_fundamentals(["AAPL"])
    df2 = p2.get_fundamentals(["AAPL"])
    assert df1["market_cap"][0] == df2["market_cap"][0]
