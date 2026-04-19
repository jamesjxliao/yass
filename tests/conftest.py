from __future__ import annotations

import pytest
from screener.data.cache import CacheManager
from screener.data.mock import MockProvider


@pytest.fixture
def cache():
    """In-memory DuckDB cache for testing."""
    cm = CacheManager(":memory:")
    yield cm
    cm.close()


@pytest.fixture
def mock_provider():
    return MockProvider()
