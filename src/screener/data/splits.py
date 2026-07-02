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


def detect_splits(prices: pl.DataFrame) -> list[str]:
    """Return tickers with day-over-day close changes > 40%, indicating stale split data."""
    if prices.is_empty():
        return []
    sorted_p = prices.sort(["ticker", "date"])
    with_ratio = sorted_p.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ticker")).alias("_ratio")
    )
    suspect = with_ratio.filter(
        (pl.col("_ratio") < 0.6) | (pl.col("_ratio") > 1.67)
    )
    return suspect["ticker"].unique().to_list()


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
    split_tickers = detect_splits(result)
    if not split_tickers:
        return result
    logger.info(
        "Split detected for %s — re-fetching adjusted prices",
        ", ".join(split_tickers),
    )
    for ticker in split_tickers:
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
    return cache.get_prices(tickers, str(start), str(end))
