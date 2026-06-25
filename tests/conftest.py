from __future__ import annotations

import pytest
from screener.data.cache import CacheManager
from screener.data.mock import MockProvider


@pytest.fixture(autouse=True)
def _no_broker_poll_sleep(monkeypatch):
    """No-op the broker fill/hold poll sleeps. The poll loops exit on the first
    iteration in tests (mocked clients return a terminal state immediately), so
    the only effect is dropping ~2s waits — the broker suite is ~85% of wall time.
    """
    monkeypatch.setattr("screener.trading.broker.time.sleep", lambda *_: None)


@pytest.fixture
def cache():
    """In-memory DuckDB cache for testing."""
    cm = CacheManager(":memory:")
    yield cm
    cm.close()


@pytest.fixture
def mock_provider():
    return MockProvider()
