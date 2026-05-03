from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import httpx

from screener.trading.broker import RebalanceOrder

logger = logging.getLogger(__name__)

BASE_URL = "https://public-api.etoro.com/api/v1"


@dataclass
class EtoroPosition:
    position_id: int
    instrument_id: int
    ticker: str
    amount: float  # invested amount in USD
    units: float
    open_rate: float
    pnl: float = 0.0


def compute_equity(portfolio: dict) -> dict:
    """Compute equity, cash, and portfolio value from an eToro PnL response.

    Follows the eToro equity formula:
    Equity = Available Cash + Total Invested + Unrealized PnL
    """
    credit = portfolio.get("credit", 0)
    positions = portfolio.get("positions", [])
    mirrors = portfolio.get("mirrors", [])
    orders_for_open = portfolio.get("ordersForOpen", [])
    orders = portfolio.get("orders", [])

    orders_for_open_amount = sum(
        o.get("amount", 0) for o in orders_for_open
        if o.get("mirrorID", o.get("mirrorId", 0)) == 0
    )
    orders_amount = sum(o.get("amount", 0) for o in orders)
    available_cash = credit - (orders_for_open_amount + orders_amount)

    positions_amount = sum(p.get("amount", 0) for p in positions)
    mirrors_positions_amount = sum(
        p.get("amount", 0)
        for m in mirrors for p in m.get("positions", [])
    )
    mirrors_adjusted = sum(
        m.get("availableAmount", 0) - m.get("closedPositionsNetProfit", 0)
        for m in mirrors
    )
    external_costs = sum(
        o.get("totalExternalCosts", 0) for o in orders_for_open
        if o.get("mirrorID", o.get("mirrorId", 0)) == 0
    )
    total_invested = (
        positions_amount + mirrors_positions_amount + mirrors_adjusted
        + orders_for_open_amount + orders_amount + external_costs
    )

    positions_pnl = sum(
        p.get("pnL", p.get("pnl", 0)) for p in positions
    )
    mirrors_pnl = sum(
        p.get("pnL", p.get("pnl", 0))
        for m in mirrors for p in m.get("positions", [])
    )
    closed_profit = sum(
        m.get("closedPositionsNetProfit", 0) for m in mirrors
    )
    unrealized_pnl = positions_pnl + mirrors_pnl + closed_profit

    equity = available_cash + total_invested + unrealized_pnl

    return {
        "equity": equity,
        "cash": available_cash,
        "buying_power": available_cash,
        "portfolio_value": total_invested + unrealized_pnl,
    }


class EtoroBroker:
    """eToro trading client for demo and real accounts."""

    def __init__(self, api_key: str, user_key: str, demo: bool = True):
        self._api_key = api_key
        self._user_key = user_key
        self._demo = demo
        self._client = httpx.Client(timeout=30)
        self._instrument_cache: dict[str, int] = {}
        self._reverse_instrument_cache: dict[int, str] = {}

    @property
    def mode(self) -> str:
        return "DEMO" if self._demo else "REAL"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "x-user-key": self._user_key,
            "x-request-id": str(uuid.uuid4()),
            "Content-Type": "application/json",
        }

    def _execution_url(self, path: str) -> str:
        prefix = "demo/" if self._demo else ""
        return f"{BASE_URL}/trading/execution/{prefix}{path}"

    def _info_url(self, path: str) -> str:
        env = "demo" if self._demo else "real"
        return f"{BASE_URL}/trading/info/{env}/{path}"

    def _portfolio_url(self) -> str:
        if self._demo:
            return f"{BASE_URL}/trading/info/demo/portfolio"
        return f"{BASE_URL}/trading/info/portfolio"

    def resolve_instrument_id(self, ticker: str) -> int | None:
        if ticker in self._instrument_cache:
            return self._instrument_cache[ticker]

        resp = self._client.get(
            f"{BASE_URL}/market-data/search",
            params={"internalSymbolFull": ticker},
            headers=self._headers(),
        )
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("items", []):
            if item.get("internalSymbolFull") == ticker:
                iid = item["instrumentId"]
                self._instrument_cache[ticker] = iid
                self._reverse_instrument_cache[iid] = ticker
                return iid

        logger.warning("Could not resolve instrument ID for %s", ticker)
        return None

    def resolve_instrument_ids(self, tickers: list[str]) -> dict[str, int]:
        result = {}
        for ticker in tickers:
            iid = self.resolve_instrument_id(ticker)
            if iid is not None:
                result[ticker] = iid
        return result

    def _ticker_for_instrument(self, instrument_id: int) -> str:
        if instrument_id in self._reverse_instrument_cache:
            return self._reverse_instrument_cache[instrument_id]
        return f"ID:{instrument_id}"

    def get_account(self) -> dict:
        """Get account info: equity, cash, portfolio value."""
        resp = self._client.get(self._info_url("pnl"), headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        portfolio = data.get("clientPortfolio", data)
        return compute_equity(portfolio)

    def get_positions(self) -> dict[str, float]:
        """Get current positions as {ticker: market_value}."""
        resp = self._client.get(self._portfolio_url(), headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        portfolio = data.get("clientPortfolio", data)

        result: dict[str, float] = {}
        for pos in portfolio.get("positions", []):
            iid = pos.get("instrumentID", pos.get("instrumentId"))
            amount = pos.get("amount", 0)
            pnl = pos.get("pnL", pos.get("pnl", 0))
            ticker = self._ticker_for_instrument(iid)
            result[ticker] = result.get(ticker, 0) + amount + pnl
        return result

    def get_positions_detailed(self) -> list[EtoroPosition]:
        """Get detailed position info including position IDs."""
        resp = self._client.get(self._portfolio_url(), headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        portfolio = data.get("clientPortfolio", data)

        positions = []
        for pos in portfolio.get("positions", []):
            iid = pos.get("instrumentID", pos.get("instrumentId"))
            positions.append(EtoroPosition(
                position_id=pos.get("positionID", pos.get("positionId")),
                instrument_id=iid,
                ticker=self._ticker_for_instrument(iid),
                amount=pos.get("amount", 0),
                units=pos.get("units", 0),
                open_rate=pos.get("openRate", 0),
                pnl=pos.get("pnL", pos.get("pnl", 0)),
            ))
        return positions

    def compute_rebalance_orders(
        self, target_tickers: list[str], account: dict | None = None
    ) -> list[RebalanceOrder]:
        if account is None:
            account = self.get_account()

        self.resolve_instrument_ids(target_tickers)

        current = self.get_positions()
        portfolio_value = account["equity"]
        target_weight = 1.0 / len(target_tickers) if target_tickers else 0
        target_value = portfolio_value * target_weight

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
                    ticker=ticker, side="sell", notional=diff,
                ))

        for ticker in target_set:
            current_value = current.get(ticker, 0)
            diff = target_value - current_value
            if diff > target_value * 0.05:
                orders.append(RebalanceOrder(
                    ticker=ticker, side="buy", notional=diff,
                ))

        return orders

    def _open_position(
        self, instrument_id: int, amount: float, stop_loss_rate: float | None = None,
    ) -> dict:
        body: dict = {
            "InstrumentID": instrument_id,
            "IsBuy": True,
            "Leverage": 1,
            "Amount": round(amount, 2),
            "IsNoTakeProfit": True,
        }
        if stop_loss_rate is not None:
            body["StopLossRate"] = round(stop_loss_rate, 4)
        else:
            body["IsNoStopLoss"] = True

        resp = self._client.post(
            self._execution_url("market-open-orders/by-amount"),
            json=body,
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    def _close_position(self, position_id: int, instrument_id: int) -> dict:
        resp = self._client.post(
            self._execution_url(f"market-close-orders/positions/{position_id}"),
            json={"InstrumentId": instrument_id},
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    def _cancel_open_order(self, order_id: int) -> dict:
        resp = self._client.delete(
            self._execution_url(f"market-open-orders/{order_id}"),
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    def cancel_open_orders(self) -> int:
        """Cancel all pending open orders. Returns number cancelled."""
        resp = self._client.get(self._portfolio_url(), headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        portfolio = data.get("clientPortfolio", data)

        cancelled = 0
        for order in portfolio.get("ordersForOpen", []):
            order_id = order.get("orderID", order.get("orderId"))
            if order_id:
                try:
                    self._cancel_open_order(order_id)
                    cancelled += 1
                except Exception as e:
                    logger.error("Failed to cancel order %s: %s", order_id, e)

        if cancelled:
            logger.info("Cancelled %d open orders", cancelled)
        return cancelled

    def _get_rate(self, instrument_id: int) -> float | None:
        try:
            resp = self._client.get(
                f"{BASE_URL}/market-data/instruments/rates",
                params={"instrumentIds": str(instrument_id)},
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            for rate in data.get("rates", []):
                if rate.get("instrumentID") == instrument_id:
                    return float(rate.get("lastExecution", rate.get("ask", 0)))
        except Exception:
            pass
        return None

    def execute_orders(
        self,
        orders: list[RebalanceOrder],
        dry_run: bool = False,
        stop_loss_pct: float = 0.0,
    ) -> list[RebalanceOrder]:
        """Execute rebalance: close sells first, then open buys."""
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

        self.cancel_open_orders()

        # Phase 1: close positions to sell
        detailed = self.get_positions_detailed()
        pos_by_ticker: dict[str, list[EtoroPosition]] = {}
        for p in detailed:
            pos_by_ticker.setdefault(p.ticker, []).append(p)

        for order in sells:
            positions_to_close = pos_by_ticker.get(order.ticker, [])
            if not positions_to_close:
                order.status = "error: no position found"
                logger.error("SELL %s — no position found", order.ticker)
                continue
            try:
                for pos in positions_to_close:
                    self._close_position(pos.position_id, pos.instrument_id)
                order.status = "submitted"
                logger.info("SELL %s $%.2f — submitted", order.ticker, order.notional)
            except Exception as e:
                order.status = f"error: {e}"
                logger.error("SELL %s $%.2f — FAILED: %s", order.ticker, order.notional, e)

        failed_sells = [o for o in sells if o.status.startswith("error")]
        if failed_sells:
            logger.error(
                "Aborting buys — %d sell(s) failed: %s",
                len(failed_sells),
                [o.ticker for o in failed_sells],
            )
            for order in buys:
                order.status = "aborted"
            return orders

        # Phase 2: open buy positions
        if buys:
            account = self.get_account()
            current = self.get_positions()
            buy_tickers = {o.ticker for o in buys}
            all_target = list(set(current.keys()) | buy_tickers)
            target_value = account["equity"] / len(all_target) if all_target else 0

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

                iid = self._instrument_cache.get(order.ticker)
                if iid is None:
                    order.status = "error: unknown instrument"
                    continue

                stop_rate = None
                if stop_loss_pct > 0:
                    rate = self._get_rate(iid)
                    if rate:
                        stop_rate = rate * (1 - stop_loss_pct)

                try:
                    self._open_position(iid, order.notional, stop_loss_rate=stop_rate)
                    order.status = "submitted"
                    logger.info("BUY %s $%.2f — submitted", order.ticker, order.notional)
                except Exception as e:
                    order.status = f"error: {e}"
                    logger.error("BUY %s $%.2f — FAILED: %s", order.ticker, order.notional, e)

        return orders


__all__ = ["EtoroBroker", "EtoroPosition", "RebalanceOrder", "compute_equity"]
