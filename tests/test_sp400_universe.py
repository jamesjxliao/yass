"""Tests for S&P 400 universe support."""

from __future__ import annotations

from datetime import date

import pytest
from screener.data.cache import CacheManager
from screener.data.fmp import _SP400_TICKERS
from screener.data.universe import UniverseManager


@pytest.fixture
def cache():
    cm = CacheManager(":memory:")
    yield cm
    cm.close()


class TestBulkLoadHistory:
    def test_inserts_records(self, cache):
        um = UniverseManager(cache)
        records = [
            {"ticker": "AAL", "added_date": "2020-01-01", "removed_date": None},
            {"ticker": "CROX", "added_date": "2021-06-02", "removed_date": None},
            {"ticker": "XYZ", "added_date": "2019-01-01", "removed_date": "2020-06-22"},
        ]
        count = um.bulk_load_history("sp400", records)
        assert count == 3

    def test_pit_query_active(self, cache):
        um = UniverseManager(cache)
        records = [
            {"ticker": "AAL", "added_date": "2020-01-01", "removed_date": None},
            {"ticker": "XYZ", "added_date": "2019-01-01", "removed_date": "2020-06-22"},
        ]
        um.bulk_load_history("sp400", records)

        active = um.get_active("sp400", date(2020, 3, 1))
        assert "AAL" in active
        assert "XYZ" in active

    def test_pit_query_after_removal(self, cache):
        um = UniverseManager(cache)
        records = [
            {"ticker": "AAL", "added_date": "2020-01-01", "removed_date": None},
            {"ticker": "XYZ", "added_date": "2019-01-01", "removed_date": "2020-06-22"},
        ]
        um.bulk_load_history("sp400", records)

        active = um.get_active("sp400", date(2021, 1, 1))
        assert "AAL" in active
        assert "XYZ" not in active

    def test_pit_query_before_addition(self, cache):
        um = UniverseManager(cache)
        records = [
            {"ticker": "AAL", "added_date": "2020-01-01", "removed_date": None},
        ]
        um.bulk_load_history("sp400", records)

        active = um.get_active("sp400", date(2019, 6, 1))
        assert "AAL" not in active

    def test_replaces_existing_records(self, cache):
        um = UniverseManager(cache)
        um.bulk_load_history("sp400", [
            {"ticker": "AAL", "added_date": "2020-01-01", "removed_date": None},
        ])
        um.bulk_load_history("sp400", [
            {"ticker": "CROX", "added_date": "2021-01-01", "removed_date": None},
        ])
        active = um.get_active("sp400")
        assert "CROX" in active
        assert "AAL" not in active


class TestSP400Fallback:
    def test_sp400_tickers_list_has_400(self):
        assert len(_SP400_TICKERS) == 400

    def test_sp400_tickers_sorted(self):
        assert _SP400_TICKERS == sorted(_SP400_TICKERS)
