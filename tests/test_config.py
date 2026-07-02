"""Regression tests for PipelineConfig.from_yaml validation (#27, weighting)."""
from __future__ import annotations

import pytest
from screener.config import PipelineConfig


def test_empty_scalar_falls_back_to_default(tmp_path):
    """A present-but-empty YAML scalar (`top_n:` → None) must fall back to the
    default, not flow None into the pipeline (#27)."""
    p = tmp_path / "c.yaml"
    p.write_text("universe: sp500\ntop_n:\nhold_bonus:\nsignals: []\nfilters: []\n")
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.top_n == 20
    assert cfg.hold_bonus == 0.0
    assert cfg.weighting == "equal"


def test_unknown_weighting_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("weighting: made_up\nsignals: []\nfilters: []\n")
    with pytest.raises(ValueError):
        PipelineConfig.from_yaml(p)


def test_unknown_top_level_key_raises(tmp_path):
    """A misspelled top-level key (e.g. 'weigthing') must fail loudly, not
    silently fall back to the default."""
    p = tmp_path / "c.yaml"
    p.write_text("weigthing: inverse_vol\nsignals: []\nfilters: []\n")
    with pytest.raises(ValueError, match="Unknown config key"):
        PipelineConfig.from_yaml(p)


def test_unknown_rebalance_frequency_raises(tmp_path):
    """A typo'd frequency ('quartely') must raise, not silently run monthly."""
    p = tmp_path / "c.yaml"
    p.write_text("rebalance_frequency: quartely\nsignals: []\nfilters: []\n")
    with pytest.raises(ValueError, match="rebalance_frequency"):
        PipelineConfig.from_yaml(p)


def test_backtest_kwargs_bundle_maps_frequency():
    """backtest_kwargs() bundles the run_backtest knobs and maps the config's
    rebalance_frequency onto run_backtest's `frequency` parameter."""
    cfg = PipelineConfig(
        filters=[], signals=[], rebalance_frequency="quarterly",
        position_stop_loss=0.05, hold_bonus=1.0, weighting="equal",
    )
    assert cfg.backtest_kwargs() == {
        "frequency": "quarterly",
        "position_stop_loss": 0.05,
        "hold_bonus": 1.0,
        "weighting": "equal",
    }


def test_settings_data_provider_validated():
    """DATA_PROVIDER typos fail at Settings() load; case is normalized."""
    from pydantic import ValidationError
    from screener.config import Settings

    assert Settings(data_provider="FMP").data_provider == "fmp"
    assert Settings(data_provider=" auto ").data_provider == "auto"
    with pytest.raises(ValidationError):
        Settings(data_provider="frnp")
