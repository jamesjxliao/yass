"""Alpaca trading client (paper and live)."""
from __future__ import annotations

import logging
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from screener.trading.broker import (
    RebalanceOrder,
    compute_rebalance_orders,
    target_value_for,
    weighting_label,
)

logger = logging.getLogger(__name__)


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
        self, target_tickers: list[str], account: dict | None = None,
        target_weights: dict[str, float] | None = None,
    ) -> list[RebalanceOrder]:
        if account is None:
            account = self.get_account()
        current = self.get_positions()
        return compute_rebalance_orders(
            target_tickers, account["equity"], current, target_weights,
        )

    def cancel_open_orders(self) -> int:
        """Cancel all open orders and wait for holds to clear."""
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        cancelled = self._client.cancel_orders()
        count = len(cancelled) if cancelled else 0
        if not count:
            return 0

        logger.info("Cancelled %d open orders, waiting for holds to clear...", count)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            time.sleep(2)
            remaining = self._client.get_orders(
                filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
            if not remaining:
                break
        else:
            logger.warning("Timed out waiting for cancellations to clear")

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

    def _submit_order(self, order: RebalanceOrder, close_position: bool = False):
        """Submit a single market order. Returns the Alpaca order object (for
        later fill-price capture), or None if the SDK returned nothing.

        Records ``order.arrival_price`` — the latest trade price captured the
        instant before submission — so the tracker can separate timing drift
        (arrival vs the model's close) from execution slippage (fill vs arrival).
        """
        order.arrival_price = self._get_last_price(order.ticker)
        if close_position and order.side == "sell":
            # Close entire position to avoid fractional qty rounding issues
            aorder = self._client.close_position(order.ticker)
            order.status = "submitted"
            logger.info("SELL %s (close position) — submitted", order.ticker)
            return aorder

        side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=order.ticker,
            notional=round(order.notional, 2),
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        aorder = self._client.submit_order(req)
        order.status = "submitted"
        logger.info(
            "%s %s $%.2f — submitted",
            order.side.upper(), order.ticker, order.notional,
        )
        return aorder

    def _capture_fill_prices(
        self, orders: list[RebalanceOrder], id_map: dict[str, str]
    ) -> None:
        """Stamp ``fill_price`` from each order's ``filled_avg_price`` once it has
        filled. Best-effort and never raises — instrumentation must not disturb
        the execution path."""
        for o in orders:
            oid = id_map.get(o.ticker)
            if not oid:
                continue
            try:
                ao = self._client.get_order_by_id(oid)
                fap = getattr(ao, "filled_avg_price", None)
                if fap is not None:
                    o.fill_price = float(fap)
            except Exception:  # noqa: BLE001 — telemetry only
                logger.debug("Could not fetch fill price for %s", o.ticker)

    def _wait_for_fills(self, orders: list[RebalanceOrder], timeout: int = 60) -> None:
        """Wait for submitted orders to fill. Polls every 2 seconds."""
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
        target_weights: dict[str, float] | None = None,
    ) -> list[RebalanceOrder]:
        """Execute rebalance orders in two phases.

        Phase 1: Cancel open orders, execute sells, wait for fills.
        Phase 2: Recompute buy amounts from actual cash, then execute buys.
        Stops are placed for all positions after buys complete.

        target_weights: optional {ticker: weight} for non-equal sizing; the
            Phase-2 recompute scales each buy by its weight (None ⇒ equal 1/N).
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
        alpaca_ids: dict[str, str] = {}  # ticker -> Alpaca order id, for fill capture
        target_tickers = {o.ticker for o in buys}
        for o in sells:
            if o.trim:
                target_tickers.add(o.ticker)
        for order in sells:
            try:
                is_exit = order.ticker not in target_tickers
                aorder = self._submit_order(order, close_position=is_exit)
                if aorder is not None and getattr(aorder, "id", None):
                    alpaca_ids[order.ticker] = str(aorder.id)
            except Exception as e:
                order.status = f"error: {e}"
                logger.error("SELL %s $%.2f — FAILED: %s", order.ticker, order.notional, e)

        # Wait for sells to fill before buying
        if sells:
            logger.info("Waiting for %d sell orders to fill...", len(sells))
            self._wait_for_fills(sells)
            self._capture_fill_prices(sells, alpaca_ids)
            # A broker can ACCEPT a sell then asynchronously reject/cancel it; it
            # leaves the OPEN set so _wait_for_fills treats it as done, but the
            # position never cleared. Re-query positions: any full-exit name still
            # held means the sell did NOT fill → mark failed so the abort fires
            # (otherwise Phase-2 buys submit against cash that was never freed).
            held = self.get_positions()
            for o in sells:
                is_exit = o.ticker not in target_tickers
                if is_exit and not o.status.startswith("error") and o.ticker in held:
                    o.status = "error: sell did not clear (still held)"
                    logger.error("SELL %s — did not clear after wait", o.ticker)

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
            all_target_tickers = list(set(current.keys()) | target_tickers)
            equity = account["equity"]
            n_equal = len(all_target_tickers)

            logger.info(
                "Cash available: $%.2f — sizing %d buys (%s)",
                account["cash"], len(buys),
                weighting_label(target_weights),
            )
            cash_remaining = account["cash"]
            for order in buys:
                current_value = current.get(order.ticker, 0)
                target_value = target_value_for(
                    order.ticker, equity, target_weights, n_equal
                )
                order.notional = max(target_value - current_value, 0)
                # Never submit beyond available cash — defends against buying on
                # margin/phantom cash if a sell silently failed to free funds.
                order.notional = min(order.notional, max(cash_remaining, 0))
                if order.notional < 1:
                    order.status = "skipped"
                    continue
                try:
                    aorder = self._submit_order(order)
                    if aorder is not None and getattr(aorder, "id", None):
                        alpaca_ids[order.ticker] = str(aorder.id)
                    cash_remaining -= order.notional
                except Exception as e:
                    order.status = f"error: {e}"
                    logger.error("BUY %s $%.2f — FAILED: %s", order.ticker, order.notional, e)

        # Wait for buys to fill before placing stops
        if buys:
            submitted_buys = [o for o in buys if o.status == "submitted"]
            if submitted_buys:
                logger.info("Waiting for %d buy orders to fill...", len(submitted_buys))
                self._wait_for_fills(submitted_buys)
                self._capture_fill_prices(submitted_buys, alpaca_ids)

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
