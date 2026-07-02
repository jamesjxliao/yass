"""get_universe must not silently cache a stale fallback on a transient failure.

A plan restriction (401/403/404) legitimately falls back to the hardcoded list;
a transient 5xx / network error must RAISE so the 30-day universe cache is not
poisoned and UniverseManager.sync never rewrites PIT membership from a stale list.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
from screener.data.fmp import _SP500_TICKERS, FMPProvider


def _status_error(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "https://example.test")
    resp = httpx.Response(code, request=req)
    return httpx.HTTPStatusError(f"HTTP {code}", request=req, response=resp)


def test_plan_restriction_falls_back_to_hardcoded_list():
    p = FMPProvider(api_key="k")
    p._get = MagicMock(side_effect=_status_error(403))
    assert p.get_universe("sp500") == _SP500_TICKERS


def test_transient_5xx_raises_instead_of_caching_fallback():
    p = FMPProvider(api_key="k")
    p._get = MagicMock(side_effect=_status_error(503))
    with pytest.raises(httpx.HTTPStatusError):
        p.get_universe("sp500")


def test_network_error_raises():
    p = FMPProvider(api_key="k")
    p._get = MagicMock(side_effect=httpx.ConnectTimeout("timeout"))
    with pytest.raises(httpx.HTTPError):
        p.get_universe("sp500")


def test_live_endpoint_success_returns_symbols():
    p = FMPProvider(api_key="k")
    p._get = MagicMock(return_value=[{"symbol": "AAPL"}, {"symbol": "MSFT"}])
    assert p.get_universe("sp500") == ["AAPL", "MSFT"]
