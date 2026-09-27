"""Stale-split detection and healing for the price cache.

Providers serve retroactively split-adjusted closes (FMP adjusted prices,
Sharadar SEP), so the day a new split lands every previously cached row for
that ticker is one adjustment behind what a fresh fetch returns. The cache's
``store_prices`` is INSERT OR IGNORE — stale rows never heal on their own —
leaving a permanent fake >40% day-over-day discontinuity at the cache seam
that corrupts the derived price fields (``momentum_12m_return`` / ``sma_200``
/ ``realized_vol_20d``). The fix: detect the seam, re-fetch the ticker's FULL
cached span, then clear-and-replace.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING, Callable

import polars as pl

if TYPE_CHECKING:
    from screener.data.cache import CacheManager

logger = logging.getLogger(__name__)


def _seams(prices: pl.DataFrame) -> pl.DataFrame:
    """Rows whose close moved > 40% vs the prior cached day: ticker, prev_date, date."""
    if prices.is_empty():
        return pl.DataFrame(schema={"ticker": pl.String, "prev_date": pl.Date,
                                    "date": pl.Date})
    with_ratio = prices.sort(["ticker", "date"]).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ticker")).alias("_ratio"),
        pl.col("date").shift(1).over("ticker").alias("prev_date"),
    )
    return with_ratio.filter(
        _is_seam(pl.col("_ratio"))
    ).select("ticker", "prev_date", "date")


_LO, _HI = 0.6, 1.67  # > 40% day-over-day move either way


def _is_seam(ratio):
    return (ratio < _LO) | (ratio > _HI)


def detect_splits(prices: pl.DataFrame) -> list[str]:
    """Return tickers with day-over-day close changes > 40% (candidate stale splits).

    Candidates only: a genuine crash/squeeze or a spin-off (SEP ``close`` is not
    spin-off-adjusted) produces the same seam — ``heal_split_prices`` tells them
    apart with a probe re-fetch before rewriting anything.
    """
    return _seams(prices)["ticker"].unique().to_list()


def _seam_is_genuine(
    ticker: str, seam_rows: pl.DataFrame,
    fetch_single: Callable[[str, date, date], pl.DataFrame],
) -> bool | None:
    """Probe-fetch each seam's two days. True = the provider's CURRENT data shows
    the same jump (a real move / spin-off, nothing stale); False = the jump is
    gone in fresh data (stale split adjustment); None = probe inconclusive (empty
    or missing a day) — the caller then falls back to the full re-fetch."""
    for prev_d, d in seam_rows.select("prev_date", "date").iter_rows():
        probe = fetch_single(ticker, prev_d, d)
        if probe.is_empty():
            return None
        closes = dict(zip(probe["date"].to_list(), probe["close"].to_list()))
        c0, c1 = closes.get(prev_d), closes.get(d)
        if not c0 or not c1:
            return None
        if not _is_seam(c1 / c0):
            return False
    return True


def heal_split_prices(
    cache: CacheManager,
    result: pl.DataFrame,
    tickers: list[str],
    start: date,
    end: date,
    fetch_single: Callable[[str, date, date], pl.DataFrame],
    source: str,
) -> pl.DataFrame:
    """Re-fetch and replace cached prices for tickers showing a split seam.

    ``result`` is the cache read serving this request; it is returned
    unchanged when no discontinuity is found, else re-read after healing.
    ``fetch_single(ticker, start, end)`` is the provider's raw single-ticker
    price fetch; ``source`` tags the re-stored rows.
    """
    seams = _seams(result)
    if seams.is_empty():
        return result
    healed_any = False
    for ticker in seams["ticker"].unique().to_list():
        # Cheap probe first. Without it every GENUINE >40% move (CAR's 2026
        # squeeze, MRNA +177%, bank failures) and every spin-off seam re-fetched
        # the ticker's full multi-year history and rewrote it on EVERY cached
        # read covering the seam — forever, since fresh data keeps the same seam.
        genuine = _seam_is_genuine(
            ticker, seams.filter(pl.col("ticker") == ticker), fetch_single
        )
        if genuine:
            continue
        logger.info("Split detected for %s — re-fetching adjusted prices", ticker)
        healed_any = True
        # Re-fetch the FULL cached span, not just this call's window —
        # invalidate_prices deletes ALL rows for the ticker, so a window-
        # only re-fetch would truncate (e.g.) 10yr of history to ~400 days
        # and silently corrupt the price cache the backtest is tuned on.
        cr = cache.get_cached_price_range(ticker)
        ref_start = min(start, date.fromisoformat(cr[0])) if cr else start
        ref_end = max(end, date.fromisoformat(cr[1])) if cr else end
        # Fetch BEFORE invalidating. A failed fetch comes back empty (FMP
        # swallows API errors) or raises (Sharadar) — deleting first would
        # wipe the ticker's entire cached history with no replacement. Only
        # clear-and-replace once real data is in hand; otherwise keep the
        # (stale-split but non-empty) existing rows.
        df = fetch_single(ticker, ref_start, ref_end)
        if not df.is_empty():
            cache.invalidate_prices([ticker])
            cache.store_prices(df, source=source)
        else:
            logger.warning(
                "Split re-fetch for %s returned no data — keeping "
                "existing cache", ticker,
            )
    return cache.get_prices(tickers, str(start), str(end)) if healed_any else result
