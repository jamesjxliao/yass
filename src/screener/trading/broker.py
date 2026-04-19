from __future__ import annotations

import logging
from dataclasses import dataclass

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

logger = logging.getLogger(__name__)


@dataclass
class RebalanceOrder:
    ticker: str
    side: str  # "buy" or "sell"
    notional: float  # dollar amount
    status: str = "pending"


class AlpacaBroker:
    """Alpaca trading client for paper and live trading."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self._api_key = api_key
        self._secret_key = secret_key
        self._client = TradingClient(api_key, secret_key, paper=paper)
        self._paper = paper
        self._data_client = None

    @property
    def mode(self) -> str:
        return "PAPER" if self._paper else "LIVE"

    def get_account(self) -> dict:
        """Get account info: equity, cash, buying power."""
        account = self._client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
        }

    def get_positions(self) -> dict[str, float]:
        """Get current positions as {ticker: market_value}."""
        positions = self._client.get_all_positions()
        return {
            p.symbol: float(p.market_value)
            for p in positions
        }

    def compute_rebalance_orders(
        self, target_tickers: list[str], account: dict | None = None
    ) -> list[RebalanceOrder]:
        """Compute orders needed to rebalance to equal-weight target portfolio.

        Returns list of sell orders first, then buy orders.
        """
        if account is None:
            account = self.get_account()

        current = self.get_positions()
        portfolio_value = account["equity"]
        target_weight = 1.0 / len(target_tickers) if target_tickers else 0
        target_value = portfolio_value * target_weight

        target_set = set(target_tickers)
        current_set = set(current.keys())

        orders = []

        # Sell positions not in target
        for ticker in current_set - target_set:
            orders.append(RebalanceOrder(
                ticker=ticker, side="sell",
                notional=current[ticker],
            ))

        # Sell excess from overweight positions
        for ticker in current_set & target_set:
            diff = current[ticker] - target_value
            if diff > target_value * 0.05:  # only if >5% overweight
                orders.append(RebalanceOrder(
                    ticker=ticker, side="sell", notional=diff,
                ))

        # Buy new positions and underweight positions
        for ticker in target_set:
            current_value = current.get(ticker, 0)
            diff = target_value - current_value
            if diff > target_value * 0.05:  # only if >5% underweight
                orders.append(RebalanceOrder(
                    ticker=ticker, side="buy", notional=diff,
                ))

        return orders

    def cancel_open_orders(self) -> int:
        """Cancel all open orders. Returns number cancelled."""
        cancelled = self._client.cancel_orders()
        count = len(cancelled) if cancelled else 0
        if count:
            logger.info("Cancelled %d open orders", count)
        return count

    def _get_last_price(self, ticker: str) -> float | None:
        """Get latest trade price for a ticker."""
        try:
            if self._data_client is None:
                from alpaca.data.historical import StockHistoricalDataClient
                self._data_client = StockHistoricalDataClient(
                    self._api_key, self._secret_key
                )
            from alpaca.data.requests import StockLatestTradeRequest
            req = StockLatestTradeRequest(symbol_or_symbols=ticker)
            trade = self._data_client.get_stock_latest_trade(req)
            return float(trade[ticker].price)
        except Exception:
            return None

    def _submit_order(self, order: RebalanceOrder, close_position: bool = False) -> None:
        """Submit a single market order."""
        if close_position and order.side == "sell":
            # Close entire position to avoid fractional qty rounding issues
            self._client.close_position(order.ticker)
            order.status = "submitted"
            logger.info("SELL %s (close position) — submitted", order.ticker)
            return

        side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=order.ticker,
            notional=round(order.notional, 2),
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        self._client.submit_order(req)
        order.status = "submitted"
        logger.info(
            "%s %s $%.2f — submitted",
            order.side.upper(), order.ticker, order.notional,
        )

    def _wait_for_fills(self, orders: list[RebalanceOrder], timeout: int = 60) -> None:
        """Wait for submitted orders to fill. Polls every 2 seconds."""
        import time

        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        pending = {o.ticker for o in orders if o.status == "submitted"}
        if not pending:
            return

        deadline = time.monotonic() + timeout
        while pending and time.monotonic() < deadline:
            time.sleep(2)
            open_orders = self._client.get_orders(
                filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
            open_tickers = {o.symbol for o in open_orders if o.type.value == "market"}
            pending = pending & open_tickers

        if pending:
            logger.warning("Timed out waiting for fills: %s", pending)

    def execute_orders(
        self,
        orders: list[RebalanceOrder],
        dry_run: bool = False,
        stop_loss_pct: float = 0.0,
    ) -> list[RebalanceOrder]:
        """Execute rebalance orders in two phases.

        Phase 1: Cancel open orders, execute sells, wait for fills.
        Phase 2: Recompute buy amounts from actual cash, then execute buys.
        Stops are placed for all positions after buys complete.
        """
        sells = [o for o in orders if o.side == "sell"]
        buys = [o for o in orders if o.side == "buy"]

        if dry_run:
            for order in sells + buys:
                order.status = "dry_run"
                logger.info(
                    "[DRY RUN] %s %s $%.2f",
                    order.side.upper(), order.ticker, order.notional,
                )
            if stop_loss_pct > 0:
                logger.info(
                    "[DRY RUN] Stop-losses at -%.0f%% for all positions",
                    stop_loss_pct * 100,
                )
            return orders

        # Phase 1: Cancel existing orders, execute sells
        self.cancel_open_orders()
        buy_tickers = {o.ticker for o in buys}
        for order in sells:
            try:
                # Full exit: close position. Trim: sell by notional.
                is_exit = order.ticker not in buy_tickers
                self._submit_order(order, close_position=is_exit)
            except Exception as e:
                order.status = f"error: {e}"
                logger.error("SELL %s $%.2f — FAILED: %s", order.ticker, order.notional, e)

        # Wait for sells to fill before buying
        if sells:
            logger.info("Waiting for %d sell orders to fill...", len(sells))
            self._wait_for_fills(sells)

        # Abort buys if any sells failed
        failed_sells = [o for o in sells if o.status.startswith("error")]
        if failed_sells:
            logger.error(
                "Aborting buys — %d sell(s) failed: %s",
                len(failed_sells),
                [o.ticker for o in failed_sells],
            )
            for order in buys:
                order.status = "aborted"
            # Still place stops for existing positions
            if stop_loss_pct > 0:
                self._place_stops_for_holdings(stop_loss_pct)
            return orders

        # Phase 2: Recompute buy amounts from actual equity and positions
        if buys:
            account = self.get_account()
            current = self.get_positions()
            buy_tickers = {o.ticker for o in buys}
            # current reflects actual post-sell state (failed sells stay)
            all_target_tickers = list(set(current.keys()) | buy_tickers)
            target_value = account["equity"] / len(all_target_tickers) if all_target_tickers else 0

            logger.info(
                "Cash available: $%.2f — target $%.2f per stock",
                account["cash"], target_value,
            )
            for order in buys:
                current_value = current.get(order.ticker, 0)
                order.notional = max(target_value - current_value, 0)
                if order.notional < 1:
                    order.status = "skipped"
                    continue
                try:
                    self._submit_order(order)
                except Exception as e:
                    order.status = f"error: {e}"
                    logger.error("BUY %s $%.2f — FAILED: %s", order.ticker, order.notional, e)

        # Wait for buys to fill before placing stops
        if buys:
            submitted_buys = [o for o in buys if o.status == "submitted"]
            if submitted_buys:
                logger.info("Waiting for %d buy orders to fill...", len(submitted_buys))
                self._wait_for_fills(submitted_buys)

        # Place stop-losses for all held positions
        if stop_loss_pct > 0:
            self._place_stops_for_holdings(stop_loss_pct)

        return orders

    def _place_stops_for_holdings(self, stop_loss_pct: float) -> None:
        """Place stop-loss orders for all current positions.

        Called after rebalance since cancel_open_orders() wipes existing stops.
        """
        from alpaca.trading.requests import StopOrderRequest

        positions = self._client.get_all_positions()
        for pos in positions:
            price = float(pos.current_price)
            # Use whole shares for GTC stop orders (fractional requires DAY)
            qty = int(float(pos.qty))
            if qty <= 0 or price <= 0:
                continue
            stop_price = round(price * (1 - stop_loss_pct), 2)
            try:
                req = StopOrderRequest(
                    symbol=pos.symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    stop_price=stop_price,
                )
                self._client.submit_order(req)
                logger.info(
                    "STOP %s %d shares — trigger at $%.2f (current $%.2f)",
                    pos.symbol, qty, stop_price, price,
                )
            except Exception as e:
                logger.error("STOP %s — FAILED: %s", pos.symbol, e)
