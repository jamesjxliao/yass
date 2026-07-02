# Archived filters

Filters that were built and backtested but did **not** improve the strategy, kept
for reference. This directory is outside the plugin-discovery glob (`filters/*.py`,
non-recursive), so these are not instantiated on screen/backtest runs and don't
clutter `list-plugins`. To revive one, move it back up into `filters/`.

- `price_above_sma.py` — hard 200-day-SMA trend gate; drops Sharpe to 0.892 by
  discarding beaten-down quality that mean-reverts (the strategy is contrarian
  on price).
