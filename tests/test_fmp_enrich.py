"""Regression test for the null-EPS crash guard in enrich_profile_row (#5).

A null latest-quarter EPS must not raise — that TypeError used to escape and
abort the entire fundamentals batch (monthly screen/trade → no picks).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from screener.data.fmp import FMPProvider


def _stub_provider():
    p = FMPProvider(api_key="k")
    p.get_key_metrics = MagicMock(return_value=[])
    p.get_quarterly_key_metrics = MagicMock(return_value=[])
    p.get_financial_growth = MagicMock(return_value=[])
    p.get_ratios = MagicMock(return_value=[])
    p.get_price_target_consensus = MagicMock(return_value=None)
    p.get_insider_stats = MagicMock(return_value=None)
    return p


def test_enrich_profile_row_null_eps_does_not_crash():
    p = _stub_provider()
    # 6 quarterly rows; the LATEST (row 0) has eps=None but revenue present —
    # the realistic "revenue posted, EPS pending" FMP state.
    inc = [{"eps": None, "revenue": 100.0}] + [{"eps": 5.0, "revenue": 90.0}] * 5
    p.get_income_statement = MagicMock(return_value=inc)

    row = p.enrich_profile_row({"symbol": "AAA"}, "AAA")  # must NOT raise

    assert row.get("eps_growth_current") is None  # skipped, not crashed
    assert row["rev_growth_current"] == 100.0 / 90.0 - 1  # revenue path still works
