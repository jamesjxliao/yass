from __future__ import annotations

import logging
import time
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
    trim: bool = False  # True = partial sell (reduce to target), False = full exit


TOLERANCE = 0.05


def target_value_for(
    ticker: str,
    equity: float,
    target_weights: dict[str, float] | None,
    n_tickers: int,
) -> float:
    """Dollar target for one ticker.

    Weighted by ``target_weights[ticker]`` when provided; otherwise equal-weight
    ``equity / n_tickers``. Shared by ``compute_rebalance_orders`` (the order
    preview) and every broker's Phase-2 buy recompute, so the previewed sizing
    and the executed sizing can never diverge.
    """
    if target_weights is not None and ticker in target_weights:
        return equity * target_weights[ticker]
    return equity / n_tickers if n_tickers else 0.0


def weighting_label(target_weights: dict[str, float] | None) -> str:
    """Human-readable sizing mode for logs.

    NB: the production equal-weight path passes a *populated* {ticker: 1/N} dict
    (compute_weights(..., "equal")), NOT None — so a bare truthiness check on
    target_weights mislabels every equal-weight run as "inverse-vol". Detect the
    1/N signature instead.
    """
    if not target_weights:
        return "equal-weight"
    n = len(target_weights)
    if n and all(abs(w - 1.0 / n) < 1e-9 for w in target_weights.values()):
        return "equal-weight"
    return "inverse-vol"


def compute_rebalance_orders(
    target_tickers: list[str],
    equity: float,
    current: dict[str, float],
    target_weights: dict[str, float] | None = None,
) -> list[RebalanceOrder]:
    """Compute rebalance orders with a 5% tolerance band.

    Shared across all brokers. Broker-specific methods fetch equity/positions
    and delegate here.

    target_weights: optional {ticker: weight} (weights sum to ~1) for non-equal
        sizing (e.g. inverse-vol). When None, falls back to equal weight 1/N.
        The per-ticker target value and its tolerance band both scale with the
        ticker's own weight.
    """
    n = len(target_tickers)

    def target_value_of(ticker: str) -> float:
        return target_value_for(ticker, equity, target_weights, n)

    target_set = set(target_tickers)
    current_set = set(current.keys())

    orders: list[RebalanceOrder] = []

    for ticker in current_set - target_set:
        orders.append(RebalanceOrder(
            ticker=ticker, side="sell", notional=current[ticker],
        ))

    for ticker in current_set & target_set:
        target_value = target_value_of(ticker)
        diff = current[ticker] - target_value
        if diff > target_value * TOLERANCE:
            orders.append(RebalanceOrder(
                ticker=ticker, side="sell", notional=diff, trim=True,
            ))

    for ticker in target_set:
        target_value = target_value_of(ticker)
        current_value = current.get(ticker, 0)
        diff = target_value - current_value
        if diff > target_value * TOLERANCE:
            orders.append(RebalanceOrder(
                ticker=ticker, side="buy", notional=diff,
            ))

    return orders


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
        target_tickers = {o.ticker for o in buys}
        for o in sells:
            if o.trim:
                target_tickers.add(o.ticker)
        for order in sells:
            try:
                is_exit = order.ticker not in target_tickers
                self._submit_order(order, close_position=is_exit)
            except Exception as e:
                order.status = f"error: {e}"
                logger.error("SELL %s $%.2f — FAILED: %s", order.ticker, order.notional, e)

        # Wait for sells to fill before buying
        if sells:
            logger.info("Waiting for %d sell orders to fill...", len(sells))
            self._wait_for_fills(sells)
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
                    self._submit_order(order)
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
