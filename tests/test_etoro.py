from unittest.mock import MagicMock

from screener.trading.etoro import EtoroBroker, compute_equity


def test_equity_positions_only():
    """Simple case: credit + positions with PnL, no mirrors or orders."""
    portfolio = {
        "credit": 5000,
        "positions": [
            {"amount": 1000, "pnL": 50},
            {"amount": 2000, "pnL": -100},
        ],
        "mirrors": [],
        "ordersForOpen": [],
        "orders": [],
    }
    result = compute_equity(portfolio)

    # Available cash = 5000 - 0 - 0 = 5000
    # Total invested = 3000 (positions) + 0 + 0 = 3000
    # Unrealized PnL = 50 + (-100) = -50
    # Equity = 5000 + 3000 + (-50) = 7950
    assert result["cash"] == 5000
    assert result["equity"] == 7950
    assert result["portfolio_value"] == 2950  # 3000 + (-50)


def test_equity_with_pending_orders():
    """Pending orders reduce available cash and count as invested."""
    portfolio = {
        "credit": 10000,
        "positions": [{"amount": 2000, "pnL": 200}],
        "mirrors": [],
        "ordersForOpen": [
            {"amount": 500, "mirrorID": 0, "totalExternalCosts": 10},
        ],
        "orders": [{"amount": 300}],
    }
    result = compute_equity(portfolio)

    # Available cash = 10000 - 500 - 300 = 9200
    # Total invested = 2000 + 0 + 0 + 500 + 300 + 10 = 2810
    # Unrealized PnL = 200
    # Equity = 9200 + 2810 + 200 = 12210
    assert result["cash"] == 9200
    assert result["equity"] == 12210


def test_equity_with_mirrors():
    """Mirror (copy trading) positions contribute to invested and PnL."""
    portfolio = {
        "credit": 8000,
        "positions": [{"amount": 1000, "pnL": 100}],
        "mirrors": [
            {
                "availableAmount": 2000,
                "closedPositionsNetProfit": 300,
                "positions": [
                    {"amount": 500, "pnL": 75},
                ],
            },
        ],
        "ordersForOpen": [],
        "orders": [],
    }
    result = compute_equity(portfolio)

    # Available cash = 8000
    # Positions invested = 1000
    # Mirror positions invested = 500
    # Mirror adjusted = 2000 - 300 = 1700
    # Total invested = 1000 + 500 + 1700 = 3200
    # Positions PnL = 100
    # Mirror PnL = 75
    # Closed profit = 300
    # Unrealized PnL = 100 + 75 + 300 = 475
    # Equity = 8000 + 3200 + 475 = 11675
    assert result["cash"] == 8000
    assert result["equity"] == 11675
    assert result["portfolio_value"] == 3675  # 3200 + 475


def test_equity_mirror_orders_excluded_from_cash():
    """Orders with mirrorID != 0 should NOT reduce available cash."""
    portfolio = {
        "credit": 5000,
        "positions": [],
        "mirrors": [],
        "ordersForOpen": [
            {"amount": 1000, "mirrorID": 0, "totalExternalCosts": 0},
            {"amount": 2000, "mirrorID": 42, "totalExternalCosts": 0},
        ],
        "orders": [],
    }
    result = compute_equity(portfolio)

    # Only mirrorID=0 order reduces cash
    assert result["cash"] == 4000  # 5000 - 1000


def test_equity_empty_portfolio():
    """Brand new account with just credit."""
    portfolio = {
        "credit": 100000,
        "positions": [],
        "mirrors": [],
        "ordersForOpen": [],
        "orders": [],
    }
    result = compute_equity(portfolio)

    assert result["equity"] == 100000
    assert result["cash"] == 100000
    assert result["portfolio_value"] == 0


def test_equity_camelcase_field_variants():
    """API returns inconsistent casing (pnL vs pnl, mirrorID vs mirrorId)."""
    portfolio = {
        "credit": 3000,
        "positions": [{"amount": 1000, "pnl": 50}],
        "mirrors": [],
        "ordersForOpen": [
            {"amount": 200, "mirrorId": 0, "totalExternalCosts": 5},
        ],
        "orders": [],
    }
    result = compute_equity(portfolio)

    assert result["cash"] == 2800  # 3000 - 200
    # Total invested = 1000 + 200 + 5 = 1205
    # PnL = 50
    # Equity = 2800 + 1205 + 50 = 4055
    assert result["equity"] == 4055


class TestExecuteOrdersStopLoss:
    """Stop-loss must be set on every buy or the buy is skipped."""

    def _make_broker(self):
        broker = EtoroBroker(api_key="k", user_key="u", demo=True)
        broker._instrument_cache = {"AAPL": 1129, "MSFT": 1130}
        return broker

    def test_buy_skipped_when_rate_unavailable(self):
        from screener.trading.broker import RebalanceOrder

        broker = self._make_broker()
        broker.get_account = MagicMock(return_value={
            "equity": 10000, "cash": 5000, "buying_power": 5000, "portfolio_value": 5000,
        })
        broker.get_positions = MagicMock(return_value={})
        broker.cancel_open_orders = MagicMock(return_value=0)
        broker.get_positions_detailed = MagicMock(return_value=[])
        broker._get_rate = MagicMock(return_value=None)

        orders = [RebalanceOrder(ticker="AAPL", side="buy", notional=1000)]
        results = broker.execute_orders(orders, dry_run=False, stop_loss_pct=0.15)

        assert results[0].status == "error: could not get rate for stop-loss"

    def test_buy_proceeds_when_rate_available(self):
        from screener.trading.broker import RebalanceOrder

        broker = self._make_broker()
        broker.get_account = MagicMock(return_value={
            "equity": 10000, "cash": 5000, "buying_power": 5000, "portfolio_value": 5000,
        })
        broker.get_positions = MagicMock(return_value={})
        broker.cancel_open_orders = MagicMock(return_value=0)
        broker.get_positions_detailed = MagicMock(return_value=[])
        broker._get_rate = MagicMock(return_value=150.0)
        broker._open_position = MagicMock(return_value={"orderForOpen": {}})

        orders = [RebalanceOrder(ticker="AAPL", side="buy", notional=1000)]
        results = broker.execute_orders(orders, dry_run=False, stop_loss_pct=0.15)

        assert results[0].status == "submitted"
        call_args = broker._open_position.call_args
        assert call_args[0][0] == 1129
        assert call_args[1]["stop_loss_rate"] == 150.0 * 0.85

    def test_buy_no_stop_loss_when_pct_zero(self):
        from screener.trading.broker import RebalanceOrder

        broker = self._make_broker()
        broker.get_account = MagicMock(return_value={
            "equity": 10000, "cash": 5000, "buying_power": 5000, "portfolio_value": 5000,
        })
        broker.get_positions = MagicMock(return_value={})
        broker.cancel_open_orders = MagicMock(return_value=0)
        broker.get_positions_detailed = MagicMock(return_value=[])
        broker._open_position = MagicMock(return_value={"orderForOpen": {}})

        orders = [RebalanceOrder(ticker="AAPL", side="buy", notional=1000)]
        results = broker.execute_orders(orders, dry_run=False, stop_loss_pct=0.0)

        assert results[0].status == "submitted"
        call_args = broker._open_position.call_args
        assert call_args[0][0] == 1129
        assert call_args[1]["stop_loss_rate"] is None
