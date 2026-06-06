from __future__ import annotations

import logging
from dataclasses import dataclass

from screener.trading.broker import RebalanceOrder

logger = logging.getLogger(__name__)


@dataclass
class RobinhoodPosition:
    symbol: str
    quantity: float
    average_cost: float
    market_value: float


class RobinhoodBroker:
    """Robinhood broker. API calls are made by Claude via MCP tools;
    this class handles data parsing and order computation."""

    def __init__(self, account_number: str):
        self.account_number = account_number

    @property
    def mode(self) -> str:
        return "LIVE"

    @staticmethod
    def parse_portfolio(portfolio_data: dict) -> dict:
        """Parse get_portfolio MCP response into standard account dict."""
        equity = float(portfolio_data.get("equity", 0))
        buying_power = float(portfolio_data.get("buying_power", 0))
        return {
            "equity": equity,
            "cash": buying_power,
            "buying_power": buying_power,
            "portfolio_value": equity - buying_power,
        }

    @staticmethod
    def parse_positions(
        positions_data: list[dict],
        quotes_data: dict[str, dict] | None = None,
    ) -> dict[str, RobinhoodPosition]:
        """Parse get_equity_positions + get_equity_quotes MCP responses.

        positions_data: list of position dicts from get_equity_positions
        quotes_data: {symbol: quote_dict} from get_equity_quotes (optional,
                     improves market_value accuracy over avg cost estimate)
        """
        result: dict[str, RobinhoodPosition] = {}
        for pos in positions_data:
            symbol = pos.get("symbol", "")
            quantity = float(pos.get("quantity", 0))
            avg_cost = float(
                pos.get("average_buy_price", pos.get("average_cost", 0))
            )
            if quantity <= 0:
                continue

            market_value = quantity * avg_cost
            if quotes_data and symbol in quotes_data:
                q = quotes_data[symbol]
                price = float(
                    q.get("last_trade_price",
                          q.get("last_extended_hours_trade_price", avg_cost))
                )
                market_value = quantity * price

            result[symbol] = RobinhoodPosition(
                symbol=symbol,
                quantity=quantity,
                average_cost=avg_cost,
                market_value=market_value,
            )
        return result

    def compute_rebalance_orders(
        self,
        target_tickers: list[str],
        account: dict,
        current: dict[str, float],
    ) -> list[RebalanceOrder]:
        """Compute equal-weight rebalance orders with 5% tolerance band."""
        equity = account["equity"]
        target_weight = 1.0 / len(target_tickers) if target_tickers else 0
        target_value = equity * target_weight

        target_set = set(target_tickers)
        current_set = set(current.keys())

        orders: list[RebalanceOrder] = []

        for ticker in current_set - target_set:
            orders.append(RebalanceOrder(
                ticker=ticker, side="sell", notional=current[ticker],
            ))

        for ticker in current_set & target_set:
            diff = current[ticker] - target_value
            if diff > target_value * 0.05:
                orders.append(RebalanceOrder(
                    ticker=ticker, side="sell", notional=diff, trim=True,
                ))

        for ticker in target_set:
            current_value = current.get(ticker, 0)
            diff = target_value - current_value
            if diff > target_value * 0.05:
                orders.append(RebalanceOrder(
                    ticker=ticker, side="buy", notional=diff,
                ))

        return orders

    def format_mcp_order(
        self,
        order: RebalanceOrder,
        positions: dict[str, RobinhoodPosition] | None = None,
    ) -> dict:
        """Format a RebalanceOrder as MCP place_equity_order parameters."""
        params: dict = {
            "account_number": self.account_number,
            "symbol": order.ticker,
            "side": order.side,
            "type": "market",
        }

        if order.side == "sell" and not order.trim and positions:
            pos = positions.get(order.ticker)
            if pos:
                params["quantity"] = str(pos.quantity)
                return params

        params["dollar_amount"] = str(round(order.notional, 2))
        return params


__all__ = ["RobinhoodBroker", "RobinhoodPosition", "RebalanceOrder"]
