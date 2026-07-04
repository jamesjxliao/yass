"""The shared _write_trade_log keeps the on-disk schema identical across brokers."""
from __future__ import annotations

import json

import polars as pl
from screener.cli import _write_trade_log
from screener.trading.broker import RebalanceOrder


def _df() -> pl.DataFrame:
    return pl.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "composite_score": [1.2, 0.9],
        "sector": ["Tech", "Tech"],
        "z_quality_score": [0.5, 0.3],
    })


def test_write_trade_log_alpaca_shape(tmp_path):
    order = RebalanceOrder(ticker="AAPL", side="buy", notional=100.0)
    order.status = "submitted"
    f = _write_trade_log(
        broker_mode="paper", dry_run=True, account={"equity": 1000, "cash": 50},
        picks=["AAPL", "MSFT"], weighting="equal",
        target_weights={"AAPL": 0.5, "MSFT": 0.5},
        previous_positions=["AAPL"], orders=[order], extended_result=_df(),
        suffix="_alpaca", extra={"broker": "alpaca"}, trade_dir=tmp_path,
    )
    assert f.parent == tmp_path and f.stem.endswith("_alpaca")
    log = json.loads(f.read_text())
    assert log["broker"] == "alpaca"  # explicit since the jul-2026 trade-alpaca rename
    assert log["mode"] == "paper"
    assert log["picks"] == ["AAPL", "MSFT"]
    assert log["orders"][0] == {
        "ticker": "AAPL", "side": "buy", "notional": 100.0, "status": "submitted",
        "arrival_price": None, "fill_price": None,
    }
    assert len(log["screen_results"]) == 2
    assert "z_quality_score" in log["screen_results"][0]


def test_write_trade_log_carries_execution_prices(tmp_path):
    """A filled live order persists arrival/fill so the tracker can decompose."""
    order = RebalanceOrder(ticker="AAPL", side="buy", notional=100.0)
    order.status = "submitted"
    order.arrival_price = 189.50
    order.fill_price = 189.92
    f = _write_trade_log(
        broker_mode="live", dry_run=False, account={"equity": 1000, "cash": 50},
        picks=["AAPL"], weighting="equal", target_weights={"AAPL": 1.0},
        previous_positions=[], orders=[order], extended_result=_df(),
        trade_dir=tmp_path,
    )
    o = json.loads(f.read_text())["orders"][0]
    assert o["arrival_price"] == 189.50 and o["fill_price"] == 189.92


def test_write_trade_log_etoro_variant(tmp_path):
    f = _write_trade_log(
        broker_mode="demo", dry_run=False, account={}, picks=[], weighting="equal",
        target_weights={}, previous_positions=[], orders=[], extended_result=_df(),
        suffix="_etoro", extra={"broker": "etoro"}, trade_dir=tmp_path,
    )
    assert f.stem.endswith("_etoro")
    log = json.loads(f.read_text())
    # extra keys land just after `date` (matching the original eToro log order).
    assert list(log.keys())[:2] == ["date", "broker"]
    assert log["broker"] == "etoro"
    assert log["orders"] == []
