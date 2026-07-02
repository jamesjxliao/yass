from __future__ import annotations

import logging
from dataclasses import dataclass

from screener.trading.broker import RebalanceOrder, compute_rebalance_orders


def _to_float(v) -> float | None:
    """Parse a possibly-null/empty quote field to float, else None."""
    try:
        return float(v) if v not in (None, "", "0", "0.0", "0.0000") else None
    except (TypeError, ValueError):
        return None

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
          - total_value:  positions + ALL cash (settled + unsettled)
          - equity_value: market value of equity positions only
          - cash:         total cash balance (settled + unsettled)
          - buying_power: nested dict, the real spendable (settled) figure
        `equity` (the sizing base) is the **deployable** capital = position value
        + settled buying power = ``total_value − unsettled_cash``. Unsettled cash
        (``cash − buying_power`` — e.g. sale proceeds still settling in a cash
        account) is excluded so the equal-weight target isn't inflated by money
        that can't be traded yet: sizing to full total_value over-buys the names
        filled first while the rest wait on settlement. Settled added cash still
        deploys. A caller's explicit hold-back is applied separately via the
        ``reserve`` arg of ``compute_rebalance_orders``.
        """
        d = portfolio_data.get("data", portfolio_data)

        total_value = float(d.get("total_value", 0) or 0)
        equity_value = float(d.get("equity_value", 0) or 0)
        cash = float(d.get("cash", 0) or 0)

        bp = d.get("buying_power", 0)
        if isinstance(bp, dict):
            bp = bp.get("buying_power", 0)
        buying_power = float(bp or 0)

        # Unsettled = cash the broker won't let us trade yet (sale proceeds in a
        # cash account). Clamp at 0 so a margin account's buying_power > cash
        # can't inflate the base — we never size into margin. Subtract from
        # total_value (not equity_value + buying_power: identical value, but this
        # form stays bit-exact = total_value when fully settled).
        unsettled_cash = max(cash - buying_power, 0.0)
        deployable = total_value - unsettled_cash

        return {
            "equity": deployable,
            "cash": cash,
            "buying_power": buying_power,
            "portfolio_value": equity_value,
            "total_value": total_value,
            "unsettled_cash": unsettled_cash,
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
            if not (quotes_data and symbol in quotes_data):
                # No live quote: market value falls back to avg cost, a STALE
                # basis. Rebalance targets are computed off true market equity,
                # so an appreciated winner valued at cost looks underweight and
                # gets over-bought. Surface it so the skill can pass quotes.
                logger.warning(
                    "No live quote for %s — valuing at avg cost ($%.2f); "
                    "rebalance sizing for this name may be off",
                    symbol, avg_cost,
                )
            if quotes_data and symbol in quotes_data:
                q = quotes_data[symbol]
                # Coalesce: dict.get returns a present-but-null/zero value (halted
                # / after-hours / data-gap quote), so a bare get-chain yields None
                # (→ float() TypeError) or 0 (→ market_value 0 → phantom re-buy).
                # Take the first positive of last/ext-hours/avg_cost.
                price = next(
                    (p for p in (
                        _to_float(q.get("last_trade_price")),
                        _to_float(q.get("last_extended_hours_trade_price")),
                        avg_cost,
                    ) if p and p > 0),
                    avg_cost,
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
        target_weights: dict[str, float] | None = None,
        reserve: float = 0.0,
    ) -> list[RebalanceOrder]:
        """``reserve`` is capital to explicitly hold back from deployment (the
        "don't deploy this" amount) — subtracted from the deployable sizing base
        (``account["equity"]``, which already excludes unsettled cash) before
        per-name targets are computed. Clamped so the base never goes negative;
        a reserve larger than settled cash therefore shrinks the target book and
        trims positions down to it, which is a valid "reduce my exposure" input.
        """
        base = max(account["equity"] - max(reserve, 0.0), 0.0)
        return compute_rebalance_orders(
            target_tickers, base, current, target_weights,
        )

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
