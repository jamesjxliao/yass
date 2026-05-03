from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    fmp_api_key: str = ""
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True
    etoro_api_key: str = ""
    etoro_user_key: str = ""
    etoro_demo: bool = True
    db_path: Path = Path("data/screener.duckdb")
    cache_ttl_hours: int = 24
    default_universe: str = "sp500"
    project_root: Path = Path(".")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def filters_dir(self) -> Path:
        return self.project_root / "filters"

    @property
    def signals_dir(self) -> Path:
        return self.project_root / "signals"


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
    ):
        self.filters = filters
        self.signals = signals
        self.universe = universe
        self.top_n = top_n
        self.rebalance_frequency = rebalance_frequency
        self.max_per_sector = max_per_sector
        self.position_stop_loss = position_stop_loss
        self.hold_bonus = hold_bonus

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            filters=data.get("filters", []),
            signals=data.get("signals", []),
            universe=data.get("universe", "sp500"),
            top_n=data.get("top_n", 20),
            rebalance_frequency=data.get("rebalance_frequency", "monthly"),
            max_per_sector=data.get("max_per_sector", 0),
            position_stop_loss=data.get("position_stop_loss", 0.0),
            hold_bonus=data.get("hold_bonus", 0.0),
        )
