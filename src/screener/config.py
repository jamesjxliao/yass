from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    fmp_api_key: str = ""
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True
    etoro_api_key: str = ""
    etoro_user_key: str = ""
    etoro_demo: bool = True
    nasdaq_data_link_api_key: str = ""  # Nasdaq Data Link (Sharadar)
    # Provider selection: "auto" prefers Sharadar when its key is present, then
    # FMP, then mock. Set DATA_PROVIDER=fmp in .env to force FMP (rollback path).
    data_provider: str = "auto"  # auto | sharadar | fmp | mock
    db_path: Path = Path("data/screener.duckdb")
    cache_ttl_hours: int = 24
    default_universe: str = "sp500"
    project_root: Path = Path(".")

    # extra="ignore": .env legitimately holds keys consumed by scripts outside
    # Settings; an unknown entry must not brick every CLI command (pydantic's
    # default extra="forbid" on env-file entries did exactly that).
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @field_validator("data_provider")
    @classmethod
    def _check_data_provider(cls, v: str) -> str:
        # Fail at Settings() load, same fail-fast treatment as the pipeline
        # config keys — a DATA_PROVIDER typo must not resolve to a fallback.
        v = v.strip().lower()
        if v not in {"auto", "sharadar", "fmp", "mock"}:
            raise ValueError(f"data_provider must be auto|sharadar|fmp|mock, got {v!r}")
        return v

    @property
    def filters_dir(self) -> Path:
        return self.project_root / "filters"

    @property
    def signals_dir(self) -> Path:
        return self.project_root / "signals"


# The rebalance cadences runner._generate_rebalance_dates actually distinguishes.
# Anything else silently falls into its monthly `else` branch, so validate against
# this set rather than letting a typo (e.g. "quartely") run at monthly cadence.
VALID_FREQUENCIES = frozenset({"weekly", "monthly", "quarterly"})

# Recognized top-level YAML keys. from_yaml raises on anything else so a typo
# (e.g. "weigthing", "hold_bonuss", "max_per_sectors") fails loudly instead of
# silently falling back to the default — the same fail-fast policy load_signals
# applies to unknown signal names.
_KNOWN_CONFIG_KEYS = frozenset({
    "filters", "signals", "universe", "top_n", "rebalance_frequency",
    "max_per_sector", "position_stop_loss", "hold_bonus", "weighting",
})


class PipelineConfig:
    """Loaded from a YAML config file defining which filters/signals to use."""

    def __init__(
        self,
        filters: list[dict[str, Any]],
        signals: list[dict[str, Any]],
        universe: str = "sp500",
        top_n: int = 20,
        rebalance_frequency: str = "monthly",
        max_per_sector: int = 0,
        position_stop_loss: float = 0.0,
        hold_bonus: float = 0.0,
        weighting: str = "equal",
    ):
        self.filters = filters
        self.signals = signals
        self.universe = universe
        self.top_n = top_n
        self.rebalance_frequency = rebalance_frequency
        self.max_per_sector = max_per_sector
        self.position_stop_loss = position_stop_loss
        self.hold_bonus = hold_bonus
        # Validate eagerly at load (mirrors load_signals' fail-fast policy) so a
        # typo'd mode raises here, not after a full screen/network round-trip.
        from screener.engine.weighting import VALID_MODES

        if weighting not in VALID_MODES:
            raise ValueError(
                f"Unknown weighting {weighting!r}. Valid: {VALID_MODES}"
            )
        self.weighting = weighting

        if rebalance_frequency not in VALID_FREQUENCIES:
            raise ValueError(
                f"Unknown rebalance_frequency {rebalance_frequency!r}. "
                f"Valid: {sorted(VALID_FREQUENCIES)}"
            )

    def backtest_kwargs(self) -> dict[str, Any]:
        """The per-config knobs threaded into run_backtest / run_full_evaluation.

        Bundled in one place so adding a new backtest knob is a single edit here
        rather than a synchronized change across every call site — the class of
        bug where `lab` once hardcoded frequency="monthly" and silently ignored
        the config. (`rebalance_frequency` maps to run_backtest's `frequency`.)
        """
        return {
            "frequency": self.rebalance_frequency,
            "position_stop_loss": self.position_stop_loss,
            "hold_bonus": self.hold_bonus,
            "weighting": self.weighting,
        }

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        unknown = set(data) - _KNOWN_CONFIG_KEYS
        if unknown:
            raise ValueError(
                f"Unknown config key(s) in {path}: {sorted(unknown)}. "
                f"Valid keys: {sorted(_KNOWN_CONFIG_KEYS)}"
            )
        return cls(
            filters=data.get("filters", []),
            signals=data.get("signals", []),
            universe=data.get("universe", "sp500"),
            # `or default` so a present-but-empty YAML scalar (`top_n:` → None)
            # falls back to the default instead of flowing None into the pipeline.
            top_n=data.get("top_n") or 20,
            rebalance_frequency=data.get("rebalance_frequency") or "monthly",
            max_per_sector=data.get("max_per_sector") or 0,
            position_stop_loss=data.get("position_stop_loss") or 0.0,
            hold_bonus=data.get("hold_bonus") if data.get("hold_bonus") is not None else 0.0,
            weighting=data.get("weighting") or "equal",
        )
