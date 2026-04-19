from __future__ import annotations

from unittest.mock import MagicMock, patch

from screener.trading.broker import AlpacaBroker, RebalanceOrder


def _make_broker(mock_client):
    """Create a broker with a mocked TradingClient."""
    with patch("screener.trading.broker.TradingClient") as mock_cls:
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
