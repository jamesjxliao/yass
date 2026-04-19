from __future__ import annotations

import logging

import polars as pl

from screener.engine.ranking import compute_composite_score
from screener.plugins.base import Filter, Signal

logger = logging.getLogger(__name__)


def enrich_with_price_data(
    fundamentals: pl.DataFrame, prices: pl.DataFrame
) -> pl.DataFrame:
    """Add price-derived columns to fundamentals DataFrame.

    Computes:
    - momentum_12m_return: 12-month return excluding most recent month
    - sma_200: 200-day simple moving average of close price
    """
    if prices.is_empty() or fundamentals.is_empty():
        return fundamentals

    sorted_prices = prices.sort(["ticker", "date"])

    # SMA-200: mean of last 200 closing prices per ticker
    sma_data = sorted_prices.group_by("ticker").agg(
        pl.col("close").tail(200).mean().alias("sma_200"),
    )

    # Momentum: (price_1m_ago / price_12m_ago) - 1
    # ~22 trading days = 1 month, ~252 trading days = 12 months
    momentum_data = sorted_prices.group_by("ticker").agg(
        pl.col("close").tail(22).first().alias("price_1m_ago"),
        pl.col("close").head(1).first().alias("price_12m_ago"),
        pl.col("close").len().alias("n_days"),
    )
    momentum_data = momentum_data.with_columns(
        pl.when(pl.col("n_days") >= 252)
        .then((pl.col("price_1m_ago") / pl.col("price_12m_ago")) - 1.0)
        .otherwise(None)
        .alias("momentum_12m_return")
    ).select("ticker", "momentum_12m_return")

    result = fundamentals
    result = result.join(sma_data, on="ticker", how="left")
    result = result.join(momentum_data, on="ticker", how="left")

    return result


class ScreeningPipeline:
    def __init__(
        self,
        filters: list[Filter],
        signals_with_weights: list[tuple[Signal, float]],
        top_n: int = 20,
        max_per_sector: int = 0,
    ):
        self.filters = filters
        self.signals_with_weights = signals_with_weights
        self.top_n = top_n
        self.max_per_sector = max_per_sector

    def run(
        self, df: pl.DataFrame, hold_bonus_tickers: set[str] | None = None,
        hold_bonus: float = 0.0,
    ) -> pl.DataFrame:
        """Run the full screening pipeline: filter -> score -> rank.

        Args:
            df: DataFrame with fundamentals/price data for the universe.
            hold_bonus_tickers: Current holdings to boost (reduce turnover).
            hold_bonus: Z-score bonus to add for current holdings.

        Returns:
            Top N candidates sorted by composite_score descending.
        """
        logger.info("Pipeline start: %d stocks", len(df))

        # Apply filters sequentially
        filtered = df
        for f in self.filters:
            mask = f.apply(filtered)
            before = len(filtered)
            filtered = filtered.filter(mask)
            logger.info(
                "  Filter '%s': %d -> %d (removed %d)",
                f.name,
                before,
                len(filtered),
                before - len(filtered),
            )

        if filtered.is_empty():
            logger.warning("All stocks filtered out, returning empty DataFrame")
            return filtered

        # Compute composite scores
        scored = compute_composite_score(filtered, self.signals_with_weights)

        # Apply hold bonus to reduce turnover
        if hold_bonus > 0 and hold_bonus_tickers:
            scored = scored.with_columns(
                pl.when(pl.col("ticker").is_in(list(hold_bonus_tickers)))
                .then(pl.col("composite_score") + hold_bonus)
                .otherwise(pl.col("composite_score"))
                .alias("composite_score")
            )

        # Rank by score
        ranked = scored.sort("composite_score", descending=True)

        # Apply sector concentration limit if configured
        if self.max_per_sector > 0 and "sector" in ranked.columns:
            before = len(ranked)
            ranked = ranked.with_columns(
                pl.col("sector").fill_null("__unknown__")
            ).with_columns(
                pl.col("ticker").cum_count().over("sector").alias("_sector_rank")
            ).filter(
                pl.col("_sector_rank") <= self.max_per_sector
            ).drop("_sector_rank")
            logger.info("  Sector cap (%d/sector): %d -> %d candidates",
                        self.max_per_sector, before, len(ranked))

        ranked = ranked.head(self.top_n)
        logger.info("Pipeline complete: returning top %d of %d", len(ranked), len(scored))

        return ranked
