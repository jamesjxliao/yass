from __future__ import annotations

import logging
from dataclasses import dataclass

from screener.trading.broker import RebalanceOrder, compute_rebalance_orders

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
        """Parse get_portfolio MCP response into standard account dict.

        The MCP `get_portfolio` payload (the inner `data` object — this also
        unwraps a `{"data": {...}}` envelope if passed whole) carries:
          - total_value:  positions + cash (what we rebalance against)
          - equity_value: market value of equity positions only
          - cash:         settled cash
          - buying_power: nested dict, real spendable figure
        `equity` is set to **total_value** so newly-added cash deploys into the
        target weights — using equity_value alone would strand the cash.
        """
        d = portfolio_data.get("data", portfolio_data)

        total_value = float(d.get("total_value", 0) or 0)
        equity_value = float(d.get("equity_value", 0) or 0)
        cash = float(d.get("cash", 0) or 0)

        bp = d.get("buying_power", 0)
        if isinstance(bp, dict):
            bp = bp.get("buying_power", 0)
        buying_power = float(bp or 0)

        return {
            "equity": total_value,
            "cash": cash,
            "buying_power": buying_power,
            "portfolio_value": equity_value,
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
        return compute_rebalance_orders(target_tickers, account["equity"], current)

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
