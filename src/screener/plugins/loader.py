from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from screener.plugins.base import Filter, Signal
from screener.plugins.registry import discover_filters, discover_signals

logger = logging.getLogger(__name__)


def load_filters(
    config_filters: list[dict[str, Any]], filters_dir: Path
) -> list[Filter]:
    """Load and configure filters based on config.

    Config format:
        - name: market_cap_filter
          params: {min_cap: 1000000000}
    """
    registry = discover_filters(filters_dir)
    loaded = []

    for entry in config_filters:
        name = entry["name"]
        params = entry.get("params", {})

        if name not in registry:
            raise ValueError(
                f"Filter '{name}' not found in {filters_dir}. "
                f"Available filters: {sorted(registry)}"
            )

        plugin = registry[name]

        # Apply params if the plugin accepts them
        if params and hasattr(plugin, "configure"):
            plugin.configure(**params)
        elif params:
            # Set attributes directly. A param that matches no existing attribute
            # is a typo (e.g. "minvol" for "min_vol") — raise instead of silently
            # dropping it and running the filter at its default threshold, matching
            # the fail-fast policy for unknown filter/signal NAMES above.
            for k, v in params.items():
                if not hasattr(plugin, k):
                    raise ValueError(
                        f"Filter '{name}' has no parameter '{k}' "
                        f"(check config/ for a typo)."
                    )
                setattr(plugin, k, v)

        loaded.append(plugin)

    return loaded


def load_signals(
    config_signals: list[dict[str, Any]], signals_dir: Path
) -> list[tuple[Signal, float]]:
    """Load and configure signals with weights based on config.

    Config format:
        - name: momentum_12m
          weight: 0.4

    Returns list of (signal, weight) tuples. Weights are normalized to sum to 1.0.
    """
    registry = discover_signals(signals_dir)
    loaded = []

    for entry in config_signals:
        name = entry["name"]
        weight = entry.get("weight", 1.0)

        if name not in registry:
            raise ValueError(
                f"Signal '{name}' not found in {signals_dir}. "
                f"Available signals: {sorted(registry)}"
            )

        loaded.append((registry[name], weight))

    # Normalize weights
    if loaded:
        total_weight = sum(w for _, w in loaded)
        if total_weight <= 0:
            raise ValueError(
                f"Signal weights sum to {total_weight} (must be > 0); all-zero/"
                "negative weights yield an order-dependent, meaningless ranking."
            )
        loaded = [(s, w / total_weight) for s, w in loaded]

    return loaded
