"""Sharadar provider: field computation, PIT stamping, membership, caching."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import polars as pl
from screener.data.cache import CacheManager
from screener.data.sharadar import (
    CachedSharadarProvider,
    SharadarProvider,
    _growth,
    _snapshots_from_history,
    compute_arq_fields,
)


def _arq_row(**over):
    base = dict(ticker="TST", dimension="ARQ", calendardate="2025-03-31",
                datekey="2025-05-01", currentratio=1.5, debtusd=100.0,
                cashnequsd=40.0, ebitda=50.0, netinc=30.0, marketcap=1000.0,
                fcf=25.0, capex=-10.0, revenue=200.0, rnd=20.0, equity=300.0,
                assets=600.0, ebit=45.0, ebt=40.0, taxexp=8.0, invcap=400.0,
                sgna=30.0, sbcomp=5.0, ncfo=35.0, intangibles=60.0,
                receivables=50.0, inventory=40.0, payables=30.0, cor=120.0,
                ev=1060.0, eps=2.0)
    base.update(over)
    return base


def test_compute_arq_fields_conventions():
    f = compute_arq_fields(_arq_row())
    assert abs(f["roe"] - 30 / 300) < 1e-9
    assert abs(f["roa"] - 30 / 600) < 1e-9
    # roic = ebit*(1-tax)/invcap, tax = 8/40 = 0.20
    assert abs(f["roic"] - 45 * 0.8 / 400) < 1e-9
    assert abs(f["net_debt_to_ebitda"] - (100 - 40) / 50) < 1e-9
    assert abs(f["earnings_yield"] - 30 / 1000) < 1e-9
    assert abs(f["capex_to_revenue"] - 10 / 200) < 1e-9  # abs() of negative capex
    assert abs(f["income_quality"] - 35 / 30) < 1e-9
    # ccc = (dso + dio - dpo) * 91.25
    expect = (50 / 200 + 40 / 120 - 30 / 120) * 91.25
    assert abs(f["cash_conversion_cycle"] - expect) < 1e-6
    assert abs(f["ev_to_sales"] - 1060 / 800) < 1e-9


def test_compute_arq_fields_null_guards():
    f = compute_arq_fields(_arq_row(equity=0, ebitda=None, marketcap=None,
                                    ebt=None, cor=None))
    assert f["roe"] is None                 # zero denominator
    assert f["net_debt_to_ebitda"] is None  # null denominator
    assert f["earnings_yield"] is None
    assert "cash_conversion_cycle" not in f or f["cash_conversion_cycle"] is None
    # tax fallback 0.25 when ebt is null
    assert abs(f["roic"] - 45 * 0.75 / 400) < 1e-9


def test_growth_requires_positive_base():
    assert _growth(12.0, 10.0) is not None
    assert _growth(12.0, -1.0) is None   # negative year-ago EPS → undefined
    assert _growth(12.0, 0.0) is None
    assert _growth(None, 10.0) is None


def test_snapshots_pit_stamping_and_windowed_fields():
    rows = []
    for i, q in enumerate(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31",
                           "2024-03-31", "2024-06-30"]):
        rows.append(_arq_row(calendardate=q, datekey=f"{q[:4]}-{int(q[5:7]):02d}-28",
                             revenue=200.0 + i * 10, eps=2.0 + i * 0.1))
    arq = pl.DataFrame(rows, infer_schema_length=None)
    snaps = _snapshots_from_history(arq, pl.DataFrame())
    by = {(s[1], s[3]): s for s in snaps if s[0] == "TST"}

    # YoY growth appears only once 4 quarters of history exist
    assert ("rev_growth_current", "2023-12-31") not in by
    g = by[("rev_growth_current", "2024-03-31")]
    assert abs(g[2] - (240.0 / 200.0 - 1)) < 1e-9
    assert g[5] == "2024-03-28"  # observed_at = the CURRENT row's datekey
    # prior-quarter growth needs 5 quarters
    gp = by[("rev_growth_prior", "2024-06-30")]
    assert abs(gp[2] - (240.0 / 200.0 - 1)) < 1e-9


def test_snapshots_annual_margins():
    ary = pl.DataFrame([
        dict(ticker="TST", calendardate="2023-12-31", datekey="2024-02-20",
             grossmargin=0.40, opinc=30.0, revenue=100.0),
        dict(ticker="TST", calendardate="2024-12-31", datekey="2025-02-19",
             grossmargin=0.45, opinc=40.0, revenue=110.0),
    ], infer_schema_length=None)
    snaps = _snapshots_from_history(pl.DataFrame(), ary)
    by = {(s[1], s[3]): s for s in snaps}
    assert abs(by[("gross_margin_current", "2024-12-31")][2] - 0.45) < 1e-9
    assert abs(by[("gross_margin_prior", "2024-12-31")][2] - 0.40) < 1e-9
    assert abs(by[("op_margin_current", "2024-12-31")][2] - 40 / 110) < 1e-9
    # first FY has no prior
    assert ("gross_margin_prior", "2023-12-31") not in by


def test_universe_dedupes_share_classes():
    p = SharadarProvider(api_key="k")
    p._frame = MagicMock(return_value=pl.DataFrame(
        {"ticker": ["AAPL", "GOOGL", "GOOG", "FOXA", "FOX", "NWSA", "NWS"]}
    ))
    u = p.get_universe("sp500")
    assert "GOOGL" in u and "GOOG" not in u
    assert "FOXA" in u and "FOX" not in u
    assert "NWSA" in u and "NWS" not in u


def test_sync_membership_builds_intervals():
    cache = CacheManager(":memory:")
    api = SharadarProvider(api_key="k")
    api.get_membership_events = MagicMock(return_value=pl.DataFrame({
        "date": ["2010-01-04", "2015-06-01", "2018-03-19", "2026-01-02"],
        "action": ["added", "removed", "added", "current"],
        "ticker": ["ACME", "ACME", "BETA", "BETA"],
    }))
    prov = CachedSharadarProvider(api, cache)
    n = prov.sync_membership()
    assert n == 2
    rows = cache.to_polars(
        "SELECT ticker, added_date::VARCHAR a, removed_date::VARCHAR r "
        "FROM universe_membership ORDER BY ticker"
    ).to_dicts()
    assert rows[0] == {"ticker": "ACME", "a": "2010-01-04", "r": "2015-06-01"}
    assert rows[1]["ticker"] == "BETA" and rows[1]["r"] is None
    cache.close()


def test_get_prices_gap_fill_caches_and_batches():
    cache = CacheManager(":memory:")
    api = SharadarProvider(api_key="k")
    calls = []

    def fake_prices(tickers, start, end):
        calls.append((tuple(tickers), start, end))
        rows = [dict(ticker=t, date=date(2024, 1, 2 + i), open=1.0, high=1.0,
                     low=1.0, close=10.0 + i, volume=100.0)
                for t in tickers for i in range(3)]
        return pl.DataFrame(rows)

    api.get_prices = fake_prices
    prov = CachedSharadarProvider(api, cache)
    df = prov.get_prices(["AAA"], date(2024, 1, 1), date(2024, 1, 10))
    assert len(df) == 3 and len(calls) == 1
    # fully cached now — second call must not hit the API
    df2 = prov.get_prices(["AAA"], date(2024, 1, 2), date(2024, 1, 4))
    assert len(calls) == 1 and len(df2) == 3
    cache.close()


def test_fundamentals_cached_after_first_fetch():
    cache = CacheManager(":memory:")
    api = SharadarProvider(api_key="k")
    prov = CachedSharadarProvider(api, cache)
    prov._build_fundamentals_rows = MagicMock(return_value={
        "TST": {"ticker": "TST", "roe": 0.1, "market_cap": 5e9, "close": 10.0},
    })
    df1 = prov.get_fundamentals(["TST"])
    df2 = prov.get_fundamentals(["TST"])
    prov._build_fundamentals_rows.assert_called_once()  # second hit from cache
    assert df1["roe"][0] == df2["roe"][0] == 0.1
    cache.close()
