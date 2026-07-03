from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from screener.trading.alpaca import AlpacaBroker
from screener.trading.broker import RebalanceOrder


def _make_broker(mock_client):
    """Create a broker with a mocked TradingClient."""
    with patch("screener.trading.alpaca.TradingClient") as mock_cls:
        mock_cls.return_value = mock_client
        return AlpacaBroker("fake_key", "fake_secret", paper=True)


def _mock_account(equity=100000, cash=50000):
    mock = MagicMock()
    mock.equity = equity
    mock.cash = cash
    mock.buying_power = cash
    mock.portfolio_value = equity
    return mock


def _mock_position(symbol, qty, market_value, current_price):
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = str(qty)
    pos.market_value = str(market_value)
    pos.current_price = str(current_price)
    return pos


def test_cancel_open_orders_before_execute():
    """execute_orders should cancel existing orders before placing new ones."""
    client = MagicMock()
    client.cancel_orders.return_value = [MagicMock()]
    client.get_account.return_value = _mock_account(cash=10000)
    client.get_positions.return_value = []
    client.get_orders.return_value = []  # no open orders (fills complete)
    broker = _make_broker(client)

    orders = [RebalanceOrder(ticker="AAPL", side="buy", notional=10000.0)]
    broker.execute_orders(orders, dry_run=False)

    client.cancel_orders.assert_called_once()


def test_cancel_not_called_on_dry_run():
    """Dry run should NOT cancel existing orders."""
    client = MagicMock()
    broker = _make_broker(client)

    orders = [RebalanceOrder(ticker="AAPL", side="buy", notional=10000.0)]
    broker.execute_orders(orders, dry_run=True)

    client.cancel_orders.assert_not_called()
    client.submit_order.assert_not_called()


def test_two_phase_sells_before_buys():
    """Sells should execute and wait for fills before buys are submitted."""
    client = MagicMock()
    client.cancel_orders.return_value = []
    client.get_orders.return_value = []  # sells filled immediately
    client.get_account.return_value = _mock_account(equity=100000, cash=30000)
    client.get_all_positions.return_value = [
        _mock_position("MSFT", 50, 35000, 700),
        _mock_position("GOOGL", 20, 35000, 1750),
    ]

    broker = _make_broker(client)
    orders = [
        RebalanceOrder(ticker="OLD", side="sell", notional=10000.0),
        RebalanceOrder(ticker="NEW", side="buy", notional=10000.0),
    ]

    broker.execute_orders(orders, dry_run=False)

    # Full exit sells use close_position, not submit_order
    client.close_position.assert_called_once()
    # Buy should also be submitted
    assert client.submit_order.call_count >= 1


def test_buy_amount_recomputed_from_equity():
    """Buy amounts should be recomputed based on actual equity, not pre-sell estimates."""
    client = MagicMock()
    client.cancel_orders.return_value = []
    client.get_orders.return_value = []
    # After sells fill: $100k equity, $30k cash, 2 held positions at $35k each
    client.get_account.return_value = _mock_account(equity=100000, cash=30000)
    # get_all_positions is called by both get_positions() and _place_stops_for_holdings()
    client.get_all_positions.return_value = [
        _mock_position("HELD1", 50, 35000, 700),
        _mock_position("HELD2", 20, 35000, 1750),
    ]

    broker = _make_broker(client)
    orders = [
        RebalanceOrder(ticker="NEW1", side="buy", notional=99999),  # original estimate
    ]

    broker.execute_orders(orders, dry_run=False)

    # Target per stock = 100000 / 3 (HELD1 + HELD2 + NEW1) = ~33333
    # NEW1 has no current position, so buy amount = 33333
    submitted = [c[0][0] for c in client.submit_order.call_args_list]
    buy_order = [o for o in submitted if o.side.value == "buy"][0]
    # Should be ~33333, NOT the original 99999
    assert buy_order.notional < 40000


def test_execute_orders_phase2_respects_target_weights():
    """Phase 2 buy recompute must size by target_weights, not equal 1/N."""
    client = MagicMock()
    client.cancel_orders.return_value = []
    client.get_orders.return_value = []
    client.get_account.return_value = _mock_account(equity=100000, cash=100000)
    client.get_all_positions.return_value = []  # all cash, no positions
    client.get_positions.return_value = []

    broker = _make_broker(client)
    orders = [
        RebalanceOrder(ticker="LOWVOL", side="buy", notional=1),
        RebalanceOrder(ticker="HIGHVOL", side="buy", notional=1),
    ]
    # 70/30 split — equal weight would give 50k/50k
    broker.execute_orders(
        orders, dry_run=False, target_weights={"LOWVOL": 0.7, "HIGHVOL": 0.3},
    )

    submitted = [c[0][0] for c in client.submit_order.call_args_list]
    by_ticker = {o.symbol: o for o in submitted if o.side.value == "buy"}
    assert by_ticker["LOWVOL"].notional == pytest.approx(70000, abs=1)
    assert by_ticker["HIGHVOL"].notional == pytest.approx(30000, abs=1)


def test_uncleared_sell_aborts_buys():
    """A full-exit sell that didn't clear (still held after the wait) must mark
    the sell failed and abort buys — else Phase-2 buys spend cash never freed."""
    client = MagicMock()
    client.cancel_orders.return_value = []
    client.get_orders.return_value = []  # _wait_for_fills sees no OPEN orders → "filled"
    client.get_account.return_value = _mock_account(equity=100000, cash=0)
    # OLD is STILL HELD after the wait → the sell was accepted but never cleared
    client.get_all_positions.return_value = [_mock_position("OLD", 10, 5000, 500)]

    broker = _make_broker(client)
    orders = [
        RebalanceOrder(ticker="OLD", side="sell", notional=5000),  # full exit
        RebalanceOrder(ticker="NEW", side="buy", notional=5000),
    ]
    result = broker.execute_orders(orders, dry_run=False)

    status = {o.ticker: o.status for o in result}
    assert status["NEW"] == "aborted"
    assert status["OLD"].startswith("error")
    # no buy order may have been submitted
    submitted_buys = [
        c[0][0] for c in client.submit_order.call_args_list
        if c[0][0].side.value == "buy"
    ]
    assert not submitted_buys


def test_phase2_buys_clamped_to_cash():
    """Buys must never exceed available cash, even if equity-based targets are
    larger (defends against buying on margin if a sell didn't free funds)."""
    client = MagicMock()
    client.cancel_orders.return_value = []
    client.get_orders.return_value = []
    # $100k equity but only $5k actually available
    client.get_account.return_value = _mock_account(equity=100000, cash=5000)
    client.get_all_positions.return_value = []
    client.get_positions.return_value = []

    broker = _make_broker(client)
    orders = [
        RebalanceOrder(ticker="A", side="buy", notional=1),
        RebalanceOrder(ticker="B", side="buy", notional=1),
    ]  # equal weight → each target ~$50k, far over the $5k cash
    broker.execute_orders(orders, dry_run=False)

    submitted = [c[0][0] for c in client.submit_order.call_args_list]
    total_buys = sum(o.notional for o in submitted if o.side.value == "buy")
    assert total_buys <= 5000 + 1  # clamped to available cash


def test_stops_placed_for_all_positions():
    """Stop-losses should be placed for every held position after rebalance."""
    client = MagicMock()
    client.cancel_orders.return_value = []
    client.get_orders.return_value = []
    client.get_account.return_value = _mock_account(equity=100000, cash=0)
    client.get_all_positions.return_value = [
        _mock_position("AAPL", 50, 10000, 200),
        _mock_position("MSFT", 30, 10000, 333),
    ]
    client.get_positions.return_value = []

    broker = _make_broker(client)
    broker.execute_orders([], dry_run=False, stop_loss_pct=0.15)

    # Should place 2 stop orders (one per position)
    stop_calls = [
        c for c in client.submit_order.call_args_list
        if hasattr(c[0][0], "stop_price")
    ]
    assert len(stop_calls) == 2


def test_stops_not_placed_on_dry_run():
    """Dry run should not place stop-loss orders."""
    client = MagicMock()
    broker = _make_broker(client)

    broker.execute_orders([], dry_run=True, stop_loss_pct=0.15)

    client.submit_order.assert_not_called()
    client.get_all_positions.assert_not_called()


def test_trim_sell_uses_notional_not_close_position():
    """Trim sells (overweight reduction) must use notional order, not close_position."""
    client = MagicMock()
    client.cancel_orders.return_value = []
    client.get_orders.return_value = []
    # After sells: 3 positions (APP, INCY, HELD) at $100k equity
    client.get_account.return_value = _mock_account(equity=100000, cash=5000)
    client.get_all_positions.return_value = [
        _mock_position("APP", 100, 45000, 450),
        _mock_position("INCY", 50, 40000, 800),
        _mock_position("HELD", 30, 10000, 333),
    ]

    broker = _make_broker(client)
    orders = [
        RebalanceOrder(ticker="APP", side="sell", notional=12000, trim=True),
        RebalanceOrder(ticker="INCY", side="sell", notional=7000, trim=True),
    ]

    broker.execute_orders(orders, dry_run=False)

    # Trims must NOT call close_position (that liquidates the entire holding)
    client.close_position.assert_not_called()
    # Trims use submit_order with notional amount
    assert client.submit_order.call_count == 2
    for call in client.submit_order.call_args_list:
        req = call[0][0]
        assert req.side.value == "sell"
        assert req.notional is not None
