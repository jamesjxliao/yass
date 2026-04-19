from __future__ import annotations

from datetime import date
from typing import Protocol

import polars as pl


class DataProvider(Protocol):
    """Interface for all data providers (FMP, mock, etc.)."""

    def get_fundamentals(self, tickers: list[str]) -> pl.DataFrame:
        """Return fundamental data (market_cap, pe_ratio, roe, etc.) for given tickers."""
        ...

    def get_prices(self, tickers: list[str], start: date, end: date) -> pl.DataFrame:
        """Return daily OHLCV price data for given tickers in date range."""
        ...

    def get_universe(self, index: str) -> list[str]:
        """Return constituent tickers for a named universe (e.g., 'sp500')."""
        ...

    def get_delisted(self) -> list[str]:
        """Return tickers that have been delisted."""
        ...
