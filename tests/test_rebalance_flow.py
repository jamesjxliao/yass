"""The shared _run_rebalance flow (used by `trade` and `etoro-trade`).

Dry-run coverage of the money path: it computes picks/weights/orders, calls
execute_orders in dry-run, writes the trade log, and honors the injected
divergences (Alpaca reads positions directly; eToro drops unresolvable picks).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from screener.cli import _run_rebalance
from screener.trading.broker import AlpacaBroker, RebalanceOrder


class _Cfg:
    universe = "sp500"
    top_n = 2
    hold_bonus = 1.0
    weighting = "equal"
    position_stop_loss = 0.0


def _provider() -> MagicMock:
    p = MagicMock()
    p.get_universe.return_value = ["AAPL", "MSFT", "NVDA"]
    p.get_prices.return_value = pl.DataFrame()        # empty → enrich passthrough
    p.get_fundamentals.return_value = pl.DataFrame()  # empty
    return p


def _screen_df() -> pl.DataFrame:
    return pl.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "composite_score": [1.2, 0.9],
        "sector": ["Tech", "Tech"],
        "realized_vol_20d": [0.20, 0.25],
    })


def _pipeline() -> MagicMock:
    pipe = MagicMock()
    pipe.run.return_value = _screen_df()
    return pipe


def test_run_rebalance_alpaca_dry_run(tmp_path):
    broker = MagicMock()
    broker.mode = "paper"
    broker.get_account.return_value = {"equity": 10000, "cash": 5000}
    broker.get_positions.return_value = {"AAPL": 4000}
    order = RebalanceOrder(ticker="MSFT", side="buy", notional=1000.0)
    order.status = "dry_run"
    broker.compute_rebalance_orders.return_value = [order]
    broker.execute_orders.return_value = [order]
    cache = MagicMock()

    _run_rebalance(
        broker=broker, provider=_provider(), cache=cache, pipeline=_pipeline(),
        pipeline_config=_Cfg(), dry_run=True,
        get_holdings=lambda enriched: broker.get_positions(),
        is_live_real=False, live_warning="X", trade_dir=tmp_path,
    )

    broker.execute_orders.assert_called_once()
    assert broker.execute_orders.call_args.kwargs["dry_run"] is True
    cache.close.assert_called_once()
    log = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert log["picks"] == ["AAPL", "MSFT"]
    assert log["previous_positions"] == ["AAPL"]
    assert "broker" not in log


def test_run_rebalance_etoro_drops_unresolved(tmp_path):
    broker = MagicMock()
    broker.mode = "demo"
    broker.get_account.return_value = {"equity": 10000, "cash": 5000}
    broker.resolve_positions.return_value = {}
    broker.resolve_instrument_ids.return_value = {"AAPL": 111}  # MSFT unresolvable
    order = RebalanceOrder(ticker="AAPL", side="buy", notional=5000.0)
    order.status = "dry_run"
    broker.compute_rebalance_orders.return_value = [order]
    broker.execute_orders.return_value = [order]
    pipe = _pipeline()

    def get_holdings(enriched):
        initial = pipe.run(enriched)
        return {} if initial.is_empty() else broker.resolve_positions(
            initial.head(2)["ticker"].to_list()
        )

    def finalize_picks(result, picks):
        resolved = broker.resolve_instrument_ids(picks)
        return [t for t in picks if t in resolved]

    _run_rebalance(
        broker=broker, provider=_provider(), cache=MagicMock(), pipeline=pipe,
        pipeline_config=_Cfg(), dry_run=True,
        get_holdings=get_holdings, finalize_picks=finalize_picks,
        is_live_real=False, live_warning="X",
        log_suffix="_etoro", log_extra={"broker": "etoro"}, trade_dir=tmp_path,
    )

    # MSFT dropped before order computation
    assert broker.compute_rebalance_orders.call_args[0][0] == ["AAPL"]
    log = json.loads(next(tmp_path.glob("*_etoro.json")).read_text())
    assert log["broker"] == "etoro"
    assert log["picks"] == ["AAPL"]


class _InverseVolCfg:
    universe = "sp500"
    top_n = 2
    hold_bonus = 1.0
    weighting = "inverse_vol"
    position_stop_loss = 0.0


def _alpaca_account(equity, cash):
    acct = MagicMock()
    acct.equity = equity
    acct.cash = cash
    acct.buying_power = cash
    acct.portfolio_value = equity
    return acct


def _real_alpaca_broker(client) -> AlpacaBroker:
    """A real AlpacaBroker with only the network client stubbed."""
    with patch("screener.trading.broker.TradingClient") as mock_cls:
        mock_cls.return_value = client
        return AlpacaBroker("k", "s", paper=True)


def test_run_rebalance_real_broker_execution_uses_inverse_vol(tmp_path):
    """End-to-end through a REAL AlpacaBroker (only the API client stubbed).

    The other flow tests mock the broker, so they never exercise the real
    compute_rebalance_orders → execute_orders money math wired through
    _run_rebalance. This drives a live (non-dry-run) rebalance and asserts the
    inverse_vol target_weights computed in the flow actually size the *submitted*
    orders (75k / 25k), not equal weight (50k / 50k) — the exact regression
    where the duplicated live path silently equal-weighted.
    """
    client = MagicMock()
    client.cancel_orders.return_value = []
    client.get_orders.return_value = []           # orders "fill" immediately
    client.get_account.return_value = _alpaca_account(equity=100_000, cash=100_000)
    client.get_all_positions.return_value = []    # start flat
    broker = _real_alpaca_broker(client)

    pipe = MagicMock()
    pipe.run.return_value = pl.DataFrame({
        "ticker": ["LOWVOL", "HIGHVOL"],
        "composite_score": [1.2, 0.9],
        "sector": ["Tech", "Tech"],
        "realized_vol_20d": [0.10, 0.30],         # 1/vol → 0.75 / 0.25
    })

    _run_rebalance(
        broker=broker, provider=_provider(), cache=MagicMock(), pipeline=pipe,
        pipeline_config=_InverseVolCfg(), dry_run=False,
        get_holdings=lambda enriched: broker.get_positions(),
        is_live_real=False, live_warning="X", trade_dir=tmp_path,
    )

    submitted = [c[0][0] for c in client.submit_order.call_args_list]
    buys = {o.symbol: float(o.notional) for o in submitted if o.side.value == "buy"}
    assert buys["LOWVOL"] == pytest.approx(75_000, abs=1)
    assert buys["HIGHVOL"] == pytest.approx(25_000, abs=1)

    log = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert log["weighting"] == "inverse_vol"
    assert log["target_weights"]["LOWVOL"] == pytest.approx(0.75, abs=1e-6)
    assert log["target_weights"]["HIGHVOL"] == pytest.approx(0.25, abs=1e-6)
