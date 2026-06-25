"""The shared research harness must reproduce the backtest setup faithfully.

run_variation with no overrides reproduces `screener lab` exactly (the gate), so
it must thread every config knob into run_backtest unchanged; overrides win.
"""
from __future__ import annotations

from datetime import date

import polars as pl
from screener.research.harness import (
    PRICE_LOOKBACK_DAYS,
    ResearchContext,
    run_variation,
)


class _Cfg:
    universe = "sp500"
    rebalance_frequency = "monthly"
    position_stop_loss = 0.0
    hold_bonus = 1.0
    weighting = "equal"

    def backtest_kwargs(self):
        return {
            "frequency": self.rebalance_frequency,
            "position_stop_loss": self.position_stop_loss,
            "hold_bonus": self.hold_bonus,
            "weighting": self.weighting,
        }


def _ctx() -> ResearchContext:
    return ResearchContext(
        config=_Cfg(), settings=None, pipeline="PIPE", provider=None,
        cache=None, pit_server="PIT", price_data=pl.DataFrame(),
        start=date(2017, 1, 1), end=date(2026, 1, 1),
    )


def test_run_variation_threads_config_knobs(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "screener.backtest.runner.run_backtest",
        lambda **kw: captured.update(kw) or "METRICS",
    )
    out = run_variation(_ctx())

    assert out == "METRICS"
    assert captured["pipeline"] == "PIPE"
    assert captured["pit_server"] == "PIT"
    assert captured["universe_index"] == "sp500"
    assert captured["frequency"] == "monthly"
    assert captured["weighting"] == "equal"
    assert captured["hold_bonus"] == 1.0
    assert captured["position_stop_loss"] == 0.0
    assert captured["start_date"] == date(2017, 1, 1)
    assert captured["end_date"] == date(2026, 1, 1)


def test_run_variation_overrides_win(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "screener.backtest.runner.run_backtest", lambda **kw: captured.update(kw)
    )
    run_variation(_ctx(), pipeline="ALT", frequency="quarterly")

    assert captured["pipeline"] == "ALT"          # alternative signal set
    assert captured["frequency"] == "quarterly"   # single-knob override
    assert captured["weighting"] == "equal"       # untouched knob still threaded


def test_price_lookback_constant():
    # ~252 trading days of pad so momentum_12m / sma_200 aren't biased early.
    assert PRICE_LOOKBACK_DAYS == 400
