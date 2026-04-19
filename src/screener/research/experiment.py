from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


@dataclass
class ExperimentVariation:
    name: str
    signals: list[dict[str, Any]]
    filters: list[dict[str, Any]] | None = None  # None = use base config filters


@dataclass
class ExperimentGuardrails:
    holdout_start: date = date(2024, 1, 1)
    max_experiments: int = 50
    min_sharpe_threshold: float = 0.5
    min_sample_size: int = 30


@dataclass
class ExperimentConfig:
    name: str
    hypothesis: str
    base_config: str
    variations: list[ExperimentVariation]
    backtest_start: date = date(2016, 1, 1)
    backtest_end: date = date(2026, 1, 1)
    rebalance_frequency: str = "monthly"
    top_n: int = 20
    walk_forward: bool = True
    guardrails: ExperimentGuardrails = field(default_factory=ExperimentGuardrails)

    @classmethod
    def from_json(cls, path: Path) -> ExperimentConfig:
        with open(path) as f:
            data = json.load(f)

        guardrails_data = data.get("guardrails", {})
        guardrails = ExperimentGuardrails(
            holdout_start=date.fromisoformat(guardrails_data.get("holdout_start", "2024-01-01")),
            max_experiments=guardrails_data.get("max_experiments", 50),
            min_sharpe_threshold=guardrails_data.get("min_sharpe_threshold", 0.5),
            min_sample_size=guardrails_data.get("min_sample_size", 30),
        )

        bt = data.get("backtest", {})

        variations = [
            ExperimentVariation(
                name=v["name"],
                signals=v.get("signals", []),
                filters=v.get("filters"),
            )
            for v in data.get("variations", [])
        ]

        return cls(
            name=data["name"],
            hypothesis=data.get("hypothesis", ""),
            base_config=data.get("base_config", "config/default.yaml"),
            variations=variations,
            backtest_start=date.fromisoformat(bt.get("start_date", "2016-01-01")),
            backtest_end=date.fromisoformat(bt.get("end_date", "2026-01-01")),
            rebalance_frequency=bt.get("rebalance_frequency", "monthly"),
            top_n=bt.get("top_n", 20),
            walk_forward=bt.get("walk_forward", True),
            guardrails=guardrails,
        )
