"""AlpacaBroker execution-quality capture: arrival_price + fill_price.

These exercise the live capture paths in ``execute_orders`` — arrival is the
latest trade price read at submit; fill is the order's ``filled_avg_price`` read
after it clears. The tracker's gap decomposition depends on both being populated.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from screener.trading.alpaca import AlpacaBroker
from screener.trading.broker import RebalanceOrder


def _broker() -> AlpacaBroker:
    with patch("screener.trading.alpaca.TradingClient"):
        b = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    b._client = MagicMock()
    b._wait_for_fills = MagicMock()  # no real polling in tests
    return b


def test_execute_captures_arrival_and_fill_for_buy_and_sell():
    b = _broker()
    b._get_last_price = MagicMock(side_effect=lambda t: {"AAPL": 100.0, "MSFT": 50.0}[t])
    b.get_account = MagicMock(return_value={
        "equity": 2000, "cash": 2000, "buying_power": 2000, "portfolio_value": 2000})
    b.get_positions = MagicMock(return_value={})  # MSFT cleared post-sell; nothing held in phase 2
    b.cancel_open_orders = MagicMock(return_value=0)

    b._client.submit_order = MagicMock(side_effect=lambda req: MagicMock(id=f"buy-{req.symbol}"))
    b._client.close_position = MagicMock(return_value=MagicMock(id="sell-MSFT"))
    fills = {"buy-AAPL": 101.0, "sell-MSFT": 49.5}
    b._client.get_order_by_id = MagicMock(
        side_effect=lambda oid: MagicMock(filled_avg_price=fills[oid]))

    orders = [RebalanceOrder(ticker="MSFT", side="sell", notional=500),
              RebalanceOrder(ticker="AAPL", side="buy", notional=1000)]
    b.execute_orders(orders, dry_run=False)

    buy = next(o for o in orders if o.ticker == "AAPL")
    sell = next(o for o in orders if o.ticker == "MSFT")
    assert (buy.arrival_price, buy.fill_price) == (100.0, 101.0)
    assert (sell.arrival_price, sell.fill_price) == (50.0, 49.5)


def test_unfilled_order_leaves_fill_none_without_crashing():
    b = _broker()
    b._get_last_price = MagicMock(return_value=100.0)
    b.get_account = MagicMock(return_value={
        "equity": 1000, "cash": 1000, "buying_power": 1000, "portfolio_value": 1000})
    b.get_positions = MagicMock(return_value={})
    b.cancel_open_orders = MagicMock(return_value=0)
    b._client.submit_order = MagicMock(return_value=MagicMock(id="buy-AAPL"))
    # filled_avg_price None (still pending) → fill_price stays None, no exception.
    b._client.get_order_by_id = MagicMock(return_value=MagicMock(filled_avg_price=None))

    orders = [RebalanceOrder(ticker="AAPL", side="buy", notional=500)]
    b.execute_orders(orders, dry_run=False)
    assert orders[0].arrival_price == 100.0
    assert orders[0].fill_price is None


def test_dry_run_leaves_prices_unpopulated():
    b = _broker()
    b._get_last_price = MagicMock(return_value=100.0)
    orders = [RebalanceOrder(ticker="AAPL", side="buy", notional=500)]
    b.execute_orders(orders, dry_run=True)
    assert orders[0].arrival_price is None and orders[0].fill_price is None
    b._get_last_price.assert_not_called()
