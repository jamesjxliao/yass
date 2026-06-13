"""Per-ticker PIT staleness detector.

`fetch-history` re-fetches the whole universe every run, but FMP's quarterly
`key-metrics` is a *derived* endpoint with variable per-ticker computation lag:
a company can file its 10-Q with the SEC weeks before FMP recomputes the ratio
bundle. When FMP returns "newest = last quarter," our `INSERT OR REPLACE` store
just keeps the old latest quarter — it never marks the ticker stale. So a ticker
whose metrics FMP computes after a given run silently stays a quarter behind
until the next run happens to land after FMP catches up.

The existing DB validation only checks the *global* max `report_date`, which
stays current as long as ANY ticker is up to date — so it never sees a single
straggler like FFIV sitting a quarter behind while the rest of the book is fresh.

This detector closes that gap. It is cohort-free: each ticker is judged against
its OWN reporting cadence, so offset fiscal calendars (Jan/Apr/Jul/Oct enders,
Saturday-dated quarter-ends) don't false-positive. A ticker is "stale" when its
own next quarter-end, plus a grace period for filing + FMP compute lag, is
already in the past but no newer quarter has landed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl

# Grace beyond the next quarter-end before we call a ticker stale. Our PIT
# convention marks quarterly data "knowable" at report_date + 45d (10-Q filing
# window); 55 leaves a small buffer for normal filing/compute variance so we
# alarm only on genuine lag, not on a quarter that simply hasn't filed yet.
DEFAULT_GRACE_DAYS = 55

_MIN_QUARTERS = 2          # need >=2 dates to estimate a cadence
_DEFAULT_CADENCE = 91      # fallback if cadence can't be measured
_CADENCE_LO, _CADENCE_HI = 45, 150  # plausible quarterly spacing (days)


@dataclass(frozen=True)
class StaleTicker:
    ticker: str
    latest_quarter: date       # newest report_date we hold
    expected_quarter: date     # latest_quarter + cadence (the one we're missing)
    cadence_days: int          # measured median spacing between quarters
    days_overdue: int          # how far past (expected_quarter + grace) we are
    quarters_behind: int       # number of missing quarters


def detect_stale(
    rows: pl.DataFrame,
    *,
    grace_days: int = DEFAULT_GRACE_DAYS,
    as_of: date | None = None,
) -> list[StaleTicker]:
    """Find stale tickers from a frame of distinct (ticker, report_date) rows.

    Pure function (no DB) so it's directly unit-testable. Each ticker is judged
    against its own measured cadence; tickers with fewer than `_MIN_QUARTERS`
    quarters, or a cadence outside the quarterly range, are skipped rather than
    guessed at.
    """
    if as_of is None:
        as_of = date.today()
    if rows.is_empty():
        return []

    per = rows.with_columns(pl.col("report_date").cast(pl.Date)).group_by("ticker").agg(
        pl.col("report_date").max().alias("latest"),
        pl.col("report_date").sort().diff().dt.total_days().median().alias("cadence"),
        pl.col("report_date").n_unique().alias("n"),
    )

    stale: list[StaleTicker] = []
    for r in per.iter_rows(named=True):
        if r["n"] < _MIN_QUARTERS:
            continue
        cadence = int(round(r["cadence"])) if r["cadence"] is not None else _DEFAULT_CADENCE
        if not (_CADENCE_LO <= cadence <= _CADENCE_HI):
            continue  # not a clean quarterly series — don't infer staleness

        latest: date = r["latest"]
        expected = latest + timedelta(days=cadence)
        available_by = expected + timedelta(days=grace_days)
        if as_of <= available_by:
            continue

        days_overdue = (as_of - available_by).days
        quarters_behind = (as_of - expected).days // cadence + 1
        stale.append(StaleTicker(
            ticker=r["ticker"],
            latest_quarter=latest,
            expected_quarter=expected,
            cadence_days=cadence,
            days_overdue=days_overdue,
            quarters_behind=quarters_behind,
        ))

    stale.sort(key=lambda s: (s.quarters_behind, s.days_overdue), reverse=True)
    return stale


def find_stale_quarters(
    cache,
    *,
    source: str = "fmp_qkm",
    grace_days: int = DEFAULT_GRACE_DAYS,
    as_of: date | None = None,
    active_within_days: int | None = 7,
) -> list[StaleTicker]:
    """Detect tickers whose latest `source` quarter is overdue for a refresh.

    Reads distinct (ticker, report_date) for the given PIT source and delegates
    to `detect_stale`. `source` defaults to the quarterly key-metrics bundle
    (`fmp_qkm`), the one subject to FMP's compute lag.

    `active_within_days` scopes the check to *currently trading* tickers — those
    whose latest cached price is within that many days of the newest price in the
    table. Without it, the PIT table's delisted/acquired tickers (PCP, BRCM, ...)
    dominate the result at 40+ quarters behind, burying live stragglers like a
    one-quarter-late FFIV. Pass `None` to check every ticker regardless.
    """
    if active_within_days is None:
        rows = cache.to_polars(
            "SELECT DISTINCT ticker, report_date FROM pit_snapshots WHERE source = ?",
            [source],
        )
    else:
        rows = cache.to_polars(
            """
            SELECT DISTINCT ps.ticker, ps.report_date
            FROM pit_snapshots ps
            JOIN (SELECT ticker, MAX(date) AS last_px FROM price_cache GROUP BY ticker) p
              ON p.ticker = ps.ticker
            WHERE ps.source = ?
              AND p.last_px >= (SELECT MAX(date) FROM price_cache) - ?
            """,
            [source, active_within_days],
        )
    return detect_stale(rows, grace_days=grace_days, as_of=as_of)
