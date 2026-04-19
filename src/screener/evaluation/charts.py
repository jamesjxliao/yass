from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

BENCHMARK_COLORS = {
    "SPY": "#ef4444",
    "QQQ": "#f59e0b",
}


def _compute_monthly_returns_from_prices(
    price_data, ticker: str, start_date: date, end_date: date
) -> list[float]:
    """Compute monthly returns for a single ticker (ETF benchmark)."""
    import polars as pl
    from dateutil.relativedelta import relativedelta

    prices = price_data.filter(
        (pl.col("ticker") == ticker)
        & (pl.col("date") >= start_date)
        & (pl.col("date") <= end_date)
    ).sort("date")

    if prices.is_empty():
        return []

    returns = []
    current = date(start_date.year, start_date.month, 1)
    while current < end_date:
        next_month = current + relativedelta(months=1)
        period = prices.filter(
            (pl.col("date") >= current) & (pl.col("date") < next_month)
        )
        if not period.is_empty():
            first = float(period["close"][0])
            last = float(period["close"][-1])
            returns.append(last / first - 1)
        current = next_month
    return returns


def plot_equity_curve(
    strategy_returns: list[float],
    start_date: date,
    output_path: Path,
    price_data=None,
    benchmark_tickers: list[str] | None = None,
    strategy_label: str = "Strategy",
) -> None:
    """Plot cumulative equity curves for strategy vs benchmarks."""
    from dateutil.relativedelta import relativedelta

    # Strategy cumulative
    strat_cum = [1.0]
    for r in strategy_returns:
        strat_cum.append(strat_cum[-1] * (1 + r))

    # Date axis
    dates = []
    current = date(start_date.year, start_date.month, 1)
    for _ in range(len(strat_cum)):
        dates.append(current)
        current += relativedelta(months=1)

    end_date = dates[-1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    fig.suptitle("Signal Evaluation", fontsize=14, fontweight="bold")

    # Plot strategy
    ax1.plot(dates, strat_cum, label=strategy_label, color="#2563eb", linewidth=2)

    # Plot benchmarks
    benchmark_tickers = benchmark_tickers or []
    for ticker in benchmark_tickers:
        if price_data is None:
            continue
        bench_returns = _compute_monthly_returns_from_prices(
            price_data, ticker, start_date, end_date
        )
        if not bench_returns:
            continue
        bench_cum = [1.0]
        for r in bench_returns:
            bench_cum.append(bench_cum[-1] * (1 + r))
        # Align length with dates
        bench_dates = dates[: len(bench_cum)]
        color = BENCHMARK_COLORS.get(ticker, "#9ca3af")
        ax1.plot(
            bench_dates, bench_cum, label=ticker,
            color=color, linewidth=1.5, linestyle="--",
        )

    ax1.set_ylabel("Growth of $1")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(dates[0], dates[-1])

    # Drawdown chart
    strat_peak = np.maximum.accumulate(strat_cum)
    strat_dd = [(c - p) / p for c, p in zip(strat_cum, strat_peak)]
    ax2.fill_between(dates, strat_dd, 0, alpha=0.4, color="#ef4444")
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.set_xlim(dates[0], dates[-1])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_heatmap(
    monthly_returns: list[float],
    start_date: date,
    output_path: Path,
) -> None:
    """Plot monthly returns as a year x month heatmap."""
    from dateutil.relativedelta import relativedelta

    current = date(start_date.year, start_date.month, 1)
    data: dict[int, dict[int, float]] = {}
    for r in monthly_returns:
        data.setdefault(current.year, {})[current.month] = r
        current += relativedelta(months=1)

    years = sorted(data.keys())
    months = list(range(1, 13))
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    grid = np.full((len(years), 12), np.nan)
    for i, year in enumerate(years):
        for month in months:
            if month in data.get(year, {}):
                grid[i, month - 1] = data[year][month] * 100

    fig, ax = plt.subplots(figsize=(12, max(3, len(years) * 0.5 + 1)))
    vmax = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)), 5)
    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)

    for i in range(len(years)):
        for j in range(12):
            val = grid[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(
                    j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=8, color=color,
                )

    ax.set_title("Monthly Returns (%)", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="%", shrink=0.8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_holdings_timeline(
    holdings: list[dict],
    output_path: Path,
    top_n_tickers: int = 20,
) -> None:
    """Plot portfolio holdings over time as a heatmap with turnover."""
    from collections import Counter

    import matplotlib.colors as mcolors

    if not holdings:
        return

    counts = Counter()
    for h in holdings:
        counts.update(h["picks"])
    top_tickers = [t for t, _ in counts.most_common(top_n_tickers)]

    dates = [h["date"] for h in holdings]
    matrix = np.zeros((len(top_tickers), len(dates)))
    for j, h in enumerate(holdings):
        picks = set(h["picks"])
        for i, ticker in enumerate(top_tickers):
            if ticker in picks:
                matrix[i, j] = 1

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, max(6, len(top_tickers) * 0.35 + 3)),
        height_ratios=[3, 1], gridspec_kw={"hspace": 0.3},
    )
    fig.suptitle("Portfolio Holdings Over Time", fontsize=14, fontweight="bold")

    cmap = mcolors.ListedColormap(["#f0f0f0", "#2563eb"])
    ax1.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
    ax1.set_yticks(range(len(top_tickers)))
    ax1.set_yticklabels(top_tickers, fontsize=8)
    ax1.set_ylabel("Stock")

    year_pos = [j for j, d in enumerate(dates) if d.month == 1]
    year_labels = [str(dates[j].year) for j in year_pos]
    ax1.set_xticks(year_pos)
    ax1.set_xticklabels(year_labels)
    ax1.set_title("Blue = held in portfolio")

    turnovers = []
    prev = set()
    for h in holdings:
        curr = set(h["picks"])
        if prev:
            all_pos = prev | curr
            turnovers.append(
                len(prev.symmetric_difference(curr)) / len(all_pos)
                if all_pos else 0
            )
        else:
            turnovers.append(1.0)
        prev = curr

    ax2.bar(range(len(turnovers)), turnovers,
            color="#ef4444", alpha=0.6, width=1.0)
    ax2.set_ylabel("Turnover")
    ax2.set_xticks(year_pos)
    ax2.set_xticklabels(year_labels)
    ax2.set_xlim(-0.5, len(dates) - 0.5)
    ax2.set_ylim(0, 1.0)
    avg = sum(turnovers) / len(turnovers)
    ax2.axhline(y=avg, color="#333", linestyle="--", linewidth=1,
                label=f"Avg: {avg:.0%}")
    ax2.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
