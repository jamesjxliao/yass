"""get_universe must not silently cache a stale fallback on a transient failure.

A plan restriction (401/403/404) legitimately falls back to the hardcoded list;
a transient 5xx / network error must RAISE so the 30-day universe cache is not
poisoned and UniverseManager.sync never rewrites PIT membership from a stale list.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
from screener.data.fmp import _SP500_TICKERS, FMPAuthError, FMPProvider


def _status_error(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "https://example.test")
    resp = httpx.Response(code, request=req)
    return httpx.HTTPStatusError(f"HTTP {code}", request=req, response=resp)


def test_401_raises_fmpautherror_not_swallowed():
    """A rejected key (401) must fail loudly through methods that normally
    swallow HTTP errors — else a cancelled subscription silently returns empty
    fundamentals for the whole universe and the screener trades on garbage."""
    p = FMPProvider(api_key="dead")
    p._client = MagicMock()
    p._client.get.return_value = httpx.Response(
        401, request=httpx.Request("GET", "https://example.test")
    )
    # get_key_metrics wraps _get in `except httpx...: return []` — the non-httpx
    # FMPAuthError must slip past that and propagate.
    with pytest.raises(FMPAuthError):
        p.get_key_metrics("AAPL")


def test_401_does_not_retry():
    """A rejected key is not a transient error — don't waste retries on it."""
    p = FMPProvider(api_key="dead")
    p._client = MagicMock()
    p._client.get.return_value = httpx.Response(
        401, request=httpx.Request("GET", "https://example.test")
    )
    with pytest.raises(FMPAuthError):
        p._get("key-metrics")
    assert p._client.get.call_count == 1  # no exponential-backoff retries


def test_universe_401_raises_not_stale_fallback():
    """A 401 on the constituent endpoint must NOT quietly return the ~2022 list."""
    p = FMPProvider(api_key="dead")
    p._client = MagicMock()
    p._client.get.return_value = httpx.Response(
        401, request=httpx.Request("GET", "https://example.test")
    )
    with pytest.raises(FMPAuthError):
        p.get_universe("sp500")


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
