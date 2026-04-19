from __future__ import annotations

from datetime import date

from screener.data.cache import CacheManager
from screener.data.mock import MockProvider
from screener.data.universe import UniverseManager


def test_sync_and_get_active(cache: CacheManager, mock_provider: MockProvider):
    um = UniverseManager(cache)
    count = um.sync(mock_provider, "sp500")
    assert count == 50

    active = um.get_active("sp500")
    assert len(active) == 50
    assert "AAPL" in active


def test_get_active_as_of_date(cache: CacheManager):
    """Tickers removed before as_of_date should not appear."""
    um = UniverseManager(cache)

    # Insert a ticker that was removed
    cache._conn.execute(
        """INSERT INTO universe_membership (ticker, index_name, added_date, removed_date)
           VALUES ('OLD', 'sp500', '2020-01-01', '2023-06-01')"""
    )
    # Insert a ticker that's still active
    cache._conn.execute(
        """INSERT INTO universe_membership (ticker, index_name, added_date)
           VALUES ('NEW', 'sp500', '2022-01-01')"""
    )

    # As of 2023-01-01, both should be active
    active = um.get_active("sp500", as_of_date=date(2023, 1, 1))
    assert "OLD" in active
    assert "NEW" in active

    # As of 2024-01-01, only NEW should be active
    active = um.get_active("sp500", as_of_date=date(2024, 1, 1))
    assert "OLD" not in active
    assert "NEW" in active
