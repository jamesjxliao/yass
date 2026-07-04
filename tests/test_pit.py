"""Point-in-time query guard tests.

PITQuery is the single no-look-ahead guard behind every backtest (via
pit_server.get_fundamentals_as_of). These lock the two properties a silent
regression could break with the rest of the suite still green:

  1. the ``observed_at <= as_of`` temporal cutoff (no future data leaks back), and
  2. the recency ordering — latest fiscal PERIOD first, then latest filing of
     that period, so a late amendment of an OLDER quarter never shadows a newer
     quarter (the ORDER BY report_date DESC, observed_at DESC fix).
"""

from datetime import date

from screener.data.cache import CacheManager
from screener.data.pit import PITQuery


def _snap(cache: CacheManager, ticker, field, value, report_date, observed_at):
    cache.record_pit_snapshot(ticker, field, value, report_date, "test", observed_at=observed_at)


def test_get_as_of_excludes_future_observation(cache: CacheManager):
    """A value observed AFTER as_of must be invisible (no look-ahead)."""
    _snap(cache, "AAPL", "roe", 0.10, "2024-03-31", "2024-05-01")
    _snap(cache, "AAPL", "roe", 0.20, "2024-06-30", "2024-08-01")  # future vs as_of below
    q = PITQuery(cache)
    assert q.get_as_of("AAPL", "roe", date(2024, 5, 15)) == 0.10


def test_get_as_of_includes_boundary_observed_at_equals_as_of(cache: CacheManager):
    """observed_at == as_of is available that day (the <= boundary, not <)."""
    _snap(cache, "AAPL", "roe", 0.10, "2024-03-31", "2024-05-01")
    q = PITQuery(cache)
    assert q.get_as_of("AAPL", "roe", date(2024, 5, 1)) == 0.10


def test_get_as_of_latest_amendment_of_same_period_wins(cache: CacheManager):
    """Two filings of the SAME quarter -> the later-observed amendment wins."""
    _snap(cache, "AAPL", "roe", 0.10, "2024-03-31", "2024-05-01")  # original
    _snap(cache, "AAPL", "roe", 0.13, "2024-03-31", "2024-06-15")  # restatement
    q = PITQuery(cache)
    assert q.get_as_of("AAPL", "roe", date(2024, 7, 1)) == 0.13


def test_get_as_of_late_amendment_does_not_shadow_newer_quarter(cache: CacheManager):
    """The regression the ORDER BY fix targets: a late amendment of an OLDER
    quarter (observed_at AFTER the newer quarter's original filing) must NOT be
    served in place of the newer quarter. Ordering by observed_at alone picked
    the restated Q1; ordering by report_date first picks Q2."""
    _snap(cache, "AAPL", "roe", 0.10, "2024-03-31", "2024-05-01")  # Q1 original
    _snap(cache, "AAPL", "roe", 0.20, "2024-06-30", "2024-08-01")  # Q2 original
    _snap(cache, "AAPL", "roe", 0.11, "2024-03-31", "2024-09-15")  # Q1 amendment (late)
    q = PITQuery(cache)
    # Only Q1 filed yet.
    assert q.get_as_of("AAPL", "roe", date(2024, 5, 15)) == 0.10
    # Q2 filed, Q1 amendment not yet -> latest period is Q2.
    assert q.get_as_of("AAPL", "roe", date(2024, 8, 15)) == 0.20
    # All three visible -> STILL Q2 (the late Q1 amendment must not shadow it).
    assert q.get_as_of("AAPL", "roe", date(2024, 10, 1)) == 0.20


def test_get_fundamentals_as_of_excludes_future_and_picks_latest_period(cache: CacheManager):
    """Vectorized path: same temporal cutoff + latest-period recency as the
    scalar path, across the same late-amendment scenario."""
    _snap(cache, "AAPL", "roe", 0.10, "2024-03-31", "2024-05-01")
    _snap(cache, "AAPL", "roe", 0.20, "2024-06-30", "2024-08-01")
    _snap(cache, "AAPL", "roe", 0.11, "2024-03-31", "2024-09-15")  # late Q1 amendment
    q = PITQuery(cache)

    # Future exclusion: as of 2024-05-15 only Q1 is visible.
    early = q.get_fundamentals_as_of(["AAPL"], ["roe"], date(2024, 5, 15))
    assert early.filter(early["ticker"] == "AAPL")["roe"][0] == 0.10

    # As of 2024-10-01 the latest PERIOD is Q2, not the later-observed Q1 amendment.
    late = q.get_fundamentals_as_of(["AAPL"], ["roe"], date(2024, 10, 1))
    assert late.filter(late["ticker"] == "AAPL")["roe"][0] == 0.20
