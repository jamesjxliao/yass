from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from screener.data.staleness import detect_stale


def _quarters(ticker: str, ends: list[str]) -> pl.DataFrame:
    return pl.DataFrame({
        "ticker": [ticker] * len(ends),
        "report_date": [date.fromisoformat(e) for e in ends],
    })


# A normal quarterly calendar ending at each calendar quarter-end.
_CURRENT = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31"]
# Same company but missing the latest (Mar 2026) quarter — the FFIV case.
_STALE = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]

AS_OF = date(2026, 6, 8)


def test_current_ticker_not_flagged():
    stale = detect_stale(_quarters("CUR", _CURRENT), as_of=AS_OF)
    assert stale == []


def test_stale_ticker_flagged_one_quarter_behind():
    stale = detect_stale(_quarters("FFIV", _STALE), as_of=AS_OF)
    assert len(stale) == 1
    s = stale[0]
    assert s.ticker == "FFIV"
    assert s.latest_quarter == date(2025, 12, 31)
    # Next quarter-end it should have (Dec 31 + ~91d ≈ Mar 31/Apr 1) is overdue.
    assert s.expected_quarter >= date(2026, 3, 30)
    assert s.quarters_behind == 1
    assert s.days_overdue > 0


def test_two_quarters_behind():
    rows = _quarters("OLD", ["2025-03-31", "2025-06-30", "2025-09-30"])
    stale = detect_stale(rows, as_of=AS_OF)
    assert len(stale) == 1
    assert stale[0].quarters_behind == 2


def test_offset_fiscal_calendar_not_flagged():
    """A Jan/Apr/Jul/Oct ender whose next quarter (Apr 30) simply hasn't filed
    yet must not be flagged — it's current for its own calendar."""
    offset = ["2025-04-30", "2025-07-31", "2025-10-31", "2026-01-31"]
    stale = detect_stale(_quarters("OFF", offset), as_of=AS_OF)
    assert stale == []


def test_saturday_dated_quarter_ends_not_flagged():
    """FMP sometimes labels quarter-ends on a nearby Saturday; cadence is still
    ~91d, so a current ticker shouldn't trip."""
    sat = ["2025-03-29", "2025-06-28", "2025-09-27", "2025-12-27", "2026-03-28"]
    stale = detect_stale(_quarters("SAT", sat), as_of=AS_OF)
    assert stale == []


def test_insufficient_history_skipped():
    stale = detect_stale(_quarters("NEW", ["2025-12-31"]), as_of=AS_OF)
    assert stale == []


def test_empty_frame():
    empty = pl.DataFrame(
        {"ticker": [], "report_date": []},
        schema={"ticker": pl.Utf8, "report_date": pl.Date},
    )
    assert detect_stale(empty, as_of=AS_OF) == []


def test_grace_period_boundary():
    """A quarter exactly at the grace boundary is not yet stale; one day past is."""
    ends = ["2025-09-30", "2025-12-31"]  # cadence 92, expected next ≈ 2026-04-02
    rows = _quarters("EDGE", ends)
    expected_next = date(2025, 12, 31) + timedelta(days=92)
    boundary = expected_next + timedelta(days=55)
    assert detect_stale(rows, grace_days=55, as_of=boundary) == []
    assert len(detect_stale(rows, grace_days=55, as_of=boundary + timedelta(days=1))) == 1


def test_results_sorted_by_severity():
    rows = pl.concat([
        _quarters("ONE", ["2025-06-30", "2025-09-30", "2025-12-31"]),       # 1 behind
        _quarters("THREE", ["2024-12-31", "2025-03-31", "2025-06-30"]),     # 3 behind
    ])
    stale = detect_stale(rows, as_of=AS_OF)
    assert [s.ticker for s in stale] == ["THREE", "ONE"]
    assert stale[0].quarters_behind > stale[1].quarters_behind
