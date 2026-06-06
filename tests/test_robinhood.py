from __future__ import annotations

from screener.trading.robinhood import RobinhoodBroker, RobinhoodPosition


class TestParsePortfolio:

    def test_basic_portfolio(self):
        data = {"equity": 50000.0, "buying_power": 10000.0}
        result = RobinhoodBroker.parse_portfolio(data)

        assert result["equity"] == 50000.0
        assert result["cash"] == 10000.0
        assert result["buying_power"] == 10000.0
        assert result["portfolio_value"] == 40000.0

    def test_empty_portfolio(self):
        data = {"equity": 0, "buying_power": 0}
        result = RobinhoodBroker.parse_portfolio(data)

        assert result["equity"] == 0
        assert result["cash"] == 0

    def test_missing_fields_default_zero(self):
        result = RobinhoodBroker.parse_portfolio({})

        assert result["equity"] == 0
        assert result["cash"] == 0


class TestParsePositions:

    def test_with_quotes(self):
        positions = [
            {"symbol": "AAPL", "quantity": "10", "average_buy_price": "150.00"},
            {"symbol": "MSFT", "quantity": "5", "average_buy_price": "400.00"},
        ]
        quotes = {
            "AAPL": {"last_trade_price": "200.00"},
            "MSFT": {"last_trade_price": "420.00"},
        }

        result = RobinhoodBroker.parse_positions(positions, quotes)

        assert len(result) == 2
        assert result["AAPL"].market_value == 2000.0
        assert result["MSFT"].market_value == 2100.0
        assert result["AAPL"].quantity == 10.0

    def test_without_quotes_uses_avg_cost(self):
        positions = [
            {"symbol": "AAPL", "quantity": "10", "average_buy_price": "150.00"},
        ]

        result = RobinhoodBroker.parse_positions(positions)

        assert result["AAPL"].market_value == 1500.0

    def test_zero_quantity_excluded(self):
        positions = [
            {"symbol": "AAPL", "quantity": "0", "average_buy_price": "150.00"},
            {"symbol": "MSFT", "quantity": "5", "average_buy_price": "400.00"},
        ]

        result = RobinhoodBroker.parse_positions(positions)

        assert "AAPL" not in result
        assert "MSFT" in result

    def test_average_cost_field_variant(self):
        positions = [
            {"symbol": "AAPL", "quantity": "10", "average_cost": "150.00"},
        ]

        result = RobinhoodBroker.parse_positions(positions)

        assert result["AAPL"].average_cost == 150.0

    def test_extended_hours_price(self):
        positions = [
            {"symbol": "AAPL", "quantity": "10", "average_buy_price": "150.00"},
        ]
        quotes = {
            "AAPL": {"last_extended_hours_trade_price": "205.00"},
        }

        result = RobinhoodBroker.parse_positions(positions, quotes)

        assert result["AAPL"].market_value == 2050.0


class TestComputeRebalanceOrders:

    def _make_broker(self):
        return RobinhoodBroker(account_number="123456789")

    def test_buy_new_positions(self):
        broker = self._make_broker()
        orders = broker.compute_rebalance_orders(
            target_tickers=["AAPL", "MSFT"],
            account={"equity": 10000},
            current={},
        )

        buys = [o for o in orders if o.side == "buy"]
        assert len(buys) == 2
        assert all(o.notional == 5000.0 for o in buys)

    def test_sell_removed_positions(self):
        broker = self._make_broker()
        orders = broker.compute_rebalance_orders(
            target_tickers=["MSFT"],
            account={"equity": 10000},
            current={"AAPL": 5000, "MSFT": 5000},
        )

        sells = [o for o in orders if o.side == "sell"]
        assert len(sells) == 1
        assert sells[0].ticker == "AAPL"
        assert sells[0].notional == 5000

    def test_trim_overweight(self):
        broker = self._make_broker()
        orders = broker.compute_rebalance_orders(
            target_tickers=["AAPL", "MSFT"],
            account={"equity": 10000},
            current={"AAPL": 8000, "MSFT": 2000},
        )

        sells = [o for o in orders if o.side == "sell"]
        buys = [o for o in orders if o.side == "buy"]
        assert len(sells) == 1
        assert sells[0].ticker == "AAPL"
        assert sells[0].trim is True
        assert sells[0].notional == 3000.0
        assert len(buys) == 1
        assert buys[0].ticker == "MSFT"

    def test_within_tolerance_no_orders(self):
        broker = self._make_broker()
        orders = broker.compute_rebalance_orders(
            target_tickers=["AAPL", "MSFT"],
            account={"equity": 10000},
            current={"AAPL": 5100, "MSFT": 4900},
        )

        assert len(orders) == 0

    def test_empty_targets_sells_everything(self):
        broker = self._make_broker()
        orders = broker.compute_rebalance_orders(
            target_tickers=[],
            account={"equity": 10000},
            current={"AAPL": 5000},
        )

        assert len(orders) == 1
        assert orders[0].side == "sell"
        assert orders[0].ticker == "AAPL"

    def test_no_positions_no_targets(self):
        broker = self._make_broker()
        orders = broker.compute_rebalance_orders(
            target_tickers=[],
            account={"equity": 10000},
            current={},
        )

        assert len(orders) == 0


class TestFormatMcpOrder:

    def _make_broker(self):
        return RobinhoodBroker(account_number="123456789")

    def test_buy_uses_dollar_amount(self):
        broker = self._make_broker()
        from screener.trading.broker import RebalanceOrder

        order = RebalanceOrder(ticker="AAPL", side="buy", notional=1500.50)
        params = broker.format_mcp_order(order)

        assert params["account_number"] == "123456789"
        assert params["symbol"] == "AAPL"
        assert params["side"] == "buy"
        assert params["type"] == "market"
        assert params["dollar_amount"] == "1500.5"
        assert "quantity" not in params

    def test_full_exit_uses_quantity(self):
        broker = self._make_broker()
        from screener.trading.broker import RebalanceOrder

        order = RebalanceOrder(ticker="AAPL", side="sell", notional=3000)
        positions = {
            "AAPL": RobinhoodPosition(
                symbol="AAPL", quantity=15.5,
                average_cost=150, market_value=3000,
            ),
        }
        params = broker.format_mcp_order(order, positions=positions)

        assert params["quantity"] == "15.5"
        assert "dollar_amount" not in params

    def test_trim_uses_dollar_amount(self):
        broker = self._make_broker()
        from screener.trading.broker import RebalanceOrder

        order = RebalanceOrder(
            ticker="AAPL", side="sell", notional=500, trim=True,
        )
        positions = {
            "AAPL": RobinhoodPosition(
                symbol="AAPL", quantity=15.5,
                average_cost=150, market_value=3000,
            ),
        }
        params = broker.format_mcp_order(order, positions=positions)

        assert params["dollar_amount"] == "500"
        assert "quantity" not in params

    def test_full_exit_no_positions_falls_back_to_dollar(self):
        broker = self._make_broker()
        from screener.trading.broker import RebalanceOrder

        order = RebalanceOrder(ticker="AAPL", side="sell", notional=3000)
        params = broker.format_mcp_order(order)

        assert params["dollar_amount"] == "3000"


class TestMode:

    def test_mode_is_live(self):
        broker = RobinhoodBroker(account_number="123")
        assert broker.mode == "LIVE"
