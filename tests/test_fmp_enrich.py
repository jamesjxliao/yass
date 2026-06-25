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


def test_enrich_profile_row_negative_eps_base_skips_growth():
    """Regression: a NEGATIVE year-ago EPS base must not yield sign-flipped
    growth. eps0/eps4 - 1 with eps4 < 0 made a loss->profit recovery read as a
    large negative growth; the guard now requires a positive base (leaves null).
    """
    p = _stub_provider()
    # rows 0-3: this year, EPS +2.0; rows 4-5: a year ago, a LOSS (EPS -1.0).
    inc = (
        [{"eps": 2.0, "revenue": 110.0}] * 4
        + [{"eps": -1.0, "revenue": 100.0}] * 2
    )
    p.get_income_statement = MagicMock(return_value=inc)

    row = p.enrich_profile_row({"symbol": "AAA"}, "AAA")

    # Negative base → growth left null, NOT a misleading -3.0.
    assert row.get("eps_growth_current") is None
    assert row.get("eps_growth_prior") is None
    # Revenue (always positive) still computes normally.
    assert row["rev_growth_current"] == 110.0 / 100.0 - 1
