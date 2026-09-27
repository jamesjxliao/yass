from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass

import httpx

from screener.trading.broker import (
    RebalanceOrder,
    compute_rebalance_orders,
    target_value_for,
    weighting_label,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://public-api.etoro.com/api/v1"

# Post-order verification re-reads positions to confirm a close/open actually
# took effect. eToro's positions endpoint lags order execution — a fill can take
# tens of seconds (observed ~30-90s) to appear/disappear there — so a single
# quick re-read false-negatives on a perfectly good order. We instead POLL:
# re-read every `_CLOSE_SETTLE_SECONDS` up to `_SETTLE_TIMEOUT_SECONDS`, returning
# as soon as the expected state is observed. `_CLOSE_SETTLE_SECONDS` is the poll
# interval (and the settle wait before the first read); patched to 0 in tests so
# the poll collapses to a single deterministic re-read.
_CLOSE_SETTLE_SECONDS = 3.0
# Total window to wait for a fill to reflect in positions before declaring it
# unconfirmed. Generous because a false negative on a sell aborts all Phase-2
# buys; better to wait than to leave a swap half-done and re-trade it.
_SETTLE_TIMEOUT_SECONDS = 90.0

# A trim prefers HIFO sub-positions, but taking a fractional lot can strand a
# residue too small for any later whole-unit deduction. Accept the HIFO-richer
# allocation only while it still fills this fraction of the whole-units-only
# one: a rounding-level shortfall shouldn't forfeit the tax benefit, a
# materially smaller trim should. Fill stays the primary goal.
_MIN_TRIM_FILL_RATIO = 0.9


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

    def _extract_pnl(p: dict) -> float:
        upnl = p.get("unrealizedPnL")
        if isinstance(upnl, dict):
            return upnl.get("pnL", 0)
        return p.get("pnL", p.get("pnl", 0))

    positions_pnl = sum(_extract_pnl(p) for p in positions)
    mirrors_pnl = sum(
        _extract_pnl(p)
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


def _plan_units(plan: list[tuple[EtoroPosition, float, bool]]) -> float:
    """Total units a trim allocation would sell."""
    return float(sum(units for _pos, units, _full in plan))


def _whole(units: float) -> int:
    """Whole units in ``units``, tolerant of float division noise.

    A bare ``int()`` truncates 2.9999999999999996 — which is what
    ``notional / price_per_unit`` produces for an exact 3 units, since
    ``620 / 6.2 == 100.00000000000001`` — costing a whole unit of trim. The
    epsilon only moves values already within 1e-9 of an integer, so it can
    never round a genuinely fractional amount up into an oversell.
    """
    return int(units + 1e-9)


def _allocate_trim(
    by_basis: list[EtoroPosition],
    desired: float,
    *,
    allow_full_close: bool,
) -> list[tuple[EtoroPosition, float, bool]]:
    """Assign ``desired`` units across sub-positions, highest basis first.

    Returns ``(position, units, full_close)``. A sub-position is taken whole when
    it fits inside what is left (any size — a full close has no whole-unit
    rule); otherwise only whole units can be deducted from it. The flag is NOT
    derivable from ``units == pos.units``: the whole-units-only allocation can
    consume a sub-position entirely and still have to send it as a partial
    ``UnitsToDeduct``, which is a different API call.

    Never exceeds ``desired``: fractional remainders are dropped, never rounded
    up into an oversell.
    """
    picks: list[tuple[EtoroPosition, float, bool]] = []
    remaining = desired
    for pos in by_basis:
        if remaining <= 0:
            break
        if allow_full_close and 0 < pos.units <= remaining:
            picks.append((pos, pos.units, True))
            remaining -= pos.units
            continue
        take = _whole(min(remaining, pos.units))
        if take < 1:
            continue
        picks.append((pos, float(take), False))
        remaining -= take
    return picks


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

    def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        last_resp = None
        for attempt in range(4):
            if attempt > 0:
                wait = 2 ** (attempt - 1) * 3
                logger.warning("Rate limited (429), waiting %ds before retry", wait)
                time.sleep(wait)
            last_resp = getattr(self._client, method)(url, **kwargs)
            if last_resp.status_code != 429:
                break
        last_resp.raise_for_status()
        return last_resp

    def _search_instruments(self, **params) -> list[dict]:
        """Call the market-data search API and return matching items."""
        resp = self._request_with_retry(
            "get",
            f"{BASE_URL}/market-data/search",
            params=params,
            headers=self._headers(),
        )
        return resp.json().get("items", [])

    def _cache_instrument(self, ticker: str, instrument_id: int) -> None:
        self._instrument_cache[ticker] = instrument_id
        self._reverse_instrument_cache[instrument_id] = ticker

    def resolve_instrument_id(self, ticker: str) -> int | None:
        if ticker in self._instrument_cache:
            return self._instrument_cache[ticker]

        items = self._search_instruments(internalSymbolFull=ticker)

        # Prefer US stock over crypto/other when ticker collides
        # (e.g., STX = Stacks crypto vs STX.US = Seagate stock)
        stock_match = None
        exact_match = None
        for item in items:
            symbol = item.get("internalSymbolFull", "")
            if symbol == ticker:
                exact_match = item
            if symbol == f"{ticker}.US":
                stock_match = item

        match = stock_match or exact_match
        if match:
            iid = match["instrumentId"]
            self._cache_instrument(ticker, iid)
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

    def resolve_positions(self, candidate_tickers: list[str]) -> dict[str, float]:
        """Resolve instrument IDs for candidates, then return positions as market values."""
        self.resolve_instrument_ids(candidate_tickers)
        detailed = self.get_positions_detailed()

        held_iids = {p.instrument_id for p in detailed
                     if p.ticker.startswith("ID:")}
        for iid in held_iids:
            ticker = self._resolve_instrument_to_ticker(iid)
            if ticker:
                logger.info("Resolved instrument %d -> %s", iid, ticker)
            else:
                logger.warning("Could not resolve instrument ID %d", iid)

        result: dict[str, float] = {}
        for pos in detailed:
            ticker = self._ticker_for_instrument(pos.instrument_id)
            result[ticker] = result.get(ticker, 0) + pos.amount + pos.pnl
        return result

    def _resolve_instrument_to_ticker(self, instrument_id: int) -> str | None:
        """Reverse-lookup an instrument ID to a ticker symbol via search API."""
        if instrument_id in self._reverse_instrument_cache:
            return self._reverse_instrument_cache[instrument_id]
        items = self._search_instruments(instrumentId=instrument_id)
        if items:
            symbol = items[0].get("internalSymbolFull", "")
            ticker = symbol.removesuffix(".US") if symbol else None
            if ticker:
                # Populate ONLY the reverse cache. Writing the forward cache here
                # would let a held non-US instrument whose bare symbol collides with
                # a screener ticker (e.g. Stacks crypto "STX" vs Seagate "STX.US")
                # overwrite the correct forward mapping, so a later buy for that
                # ticker would open the wrong asset. The forward cache is populated
                # only by resolve_instrument_id, which prefers the "{ticker}.US" stock.
                self._reverse_instrument_cache[instrument_id] = ticker
            return ticker
        return None

    def _ticker_for_instrument(self, instrument_id: int) -> str:
        if instrument_id in self._reverse_instrument_cache:
            return self._reverse_instrument_cache[instrument_id]
        return f"ID:{instrument_id}"

    def get_account(self) -> dict:
        """Get account info: equity, cash, portfolio value."""
        resp = self._request_with_retry(
            "get", self._info_url("pnl"), headers=self._headers()
        )
        data = resp.json()
        portfolio = data.get("clientPortfolio", data)
        return compute_equity(portfolio)

    def _get_pnl_positions(self) -> list[dict]:
        """Fetch positions from the PnL endpoint (includes unrealizedPnL)."""
        resp = self._request_with_retry(
            "get", self._info_url("pnl"), headers=self._headers()
        )
        data = resp.json()
        portfolio = data.get("clientPortfolio", data)
        return portfolio.get("positions", [])

    def get_positions(self) -> dict[str, float]:
        """Get current positions as {ticker: market_value}."""
        result: dict[str, float] = {}
        for pos in self._get_pnl_positions():
            iid = pos.get("instrumentID", pos.get("instrumentId"))
            upnl = pos.get("unrealizedPnL", {})
            market_value = upnl.get("exposureInAccountCurrency", pos.get("amount", 0))
            ticker = self._ticker_for_instrument(iid)
            result[ticker] = result.get(ticker, 0) + market_value
        return result

    def get_positions_detailed(self) -> list[EtoroPosition]:
        """Get detailed position info including position IDs and PnL."""
        positions = []
        for pos in self._get_pnl_positions():
            iid = pos.get("instrumentID", pos.get("instrumentId"))
            upnl = pos.get("unrealizedPnL", {})
            pnl = upnl.get("pnL", 0)
            positions.append(EtoroPosition(
                position_id=pos.get("positionID", pos.get("positionId")),
                instrument_id=iid,
                ticker=self._ticker_for_instrument(iid),
                amount=pos.get("amount", 0),
                units=pos.get("units", 0),
                open_rate=pos.get("openRate", 0),
                pnl=pnl,
            ))
        return positions

    def compute_rebalance_orders(
        self, target_tickers: list[str], account: dict | None = None,
        target_weights: dict[str, float] | None = None,
    ) -> list[RebalanceOrder]:
        if account is None:
            account = self.get_account()
        current = self.resolve_positions(target_tickers)
        return compute_rebalance_orders(
            target_tickers, account["equity"], current, target_weights,
        )

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

        resp = self._request_with_retry(
            "post",
            self._execution_url("market-open-orders/by-amount"),
            json=body,
            headers=self._headers(),
        )
        return resp.json()

    def _close_position(
        self, position_id: int, instrument_id: int,
        units_to_deduct: float | None = None,
    ) -> dict:
        body: dict = {"InstrumentId": instrument_id}
        if units_to_deduct is not None:
            # eToro rejects a fractional UnitsToDeduct (errorCode 776, "only whole
            # units are allowed") even on instruments whose unitsQuantityType is
            # 'fractional' — that flag governs OPENING, not partial closes. Send an
            # int so the JSON carries no decimal part.
            body["UnitsToDeduct"] = int(units_to_deduct)
        resp = self._request_with_retry(
            "post",
            self._execution_url(f"market-close-orders/positions/{position_id}"),
            json=body,
            headers=self._headers(),
        )
        return resp.json()

    def _execute_trim(
        self, order: RebalanceOrder, positions: list[EtoroPosition],
    ) -> float:
        """Partially close positions to reduce a holding by order.notional dollars.

        Returns the number of units actually requested for deduction, so the
        caller can verify the trim took effect (see `_await_units_reduction`).

        Sub-positions are consumed HIFO — highest ``open_rate`` (cost basis)
        first — so a trim realizes the smallest gain (or largest loss) per unit
        sold. eToro closes are per-sub-position, so lot choice is ours to make;
        the old largest-``units``-first order was tax-blind and, because early
        lots are usually both the largest and the lowest-basis, tended to
        realize the *biggest* embedded gains — FIFO-style lot cycling that
        converts long-term gains into short-term ones in a taxable account.
        Ties fall back to largest-units-first to minimize close calls.

        **The whole-unit rule binds PARTIAL closes only.** eToro rejects a
        fractional ``UnitsToDeduct`` (errorCode 776), but a sub-position can
        always be closed in its ENTIRETY whatever its size — that is how full
        exits already sell fractional holdings. Treating both alike made every
        sub-one-unit lot unusable regardless of basis, which silently
        de-optimized HIFO exactly on high-priced names (SNDK ~$1,437/unit, where
        a routine monthly buy IS a fractional lot): the dearest lots were
        skipped and the trim fell through to the older, cheaper, whole-unit
        ones — the FIFO-ish behavior HIFO exists to avoid.

        Fill still outranks lot choice: both allocations are costed and the
        HIFO-richer one is taken only if it clears ``_MIN_TRIM_FILL_RATIO`` of
        the other — see that constant for why.
        """
        total_value = sum(p.amount + p.pnl for p in positions)
        total_units = sum(p.units for p in positions)
        if total_value <= 0 or total_units <= 0:
            raise ValueError(f"No value to trim for {order.ticker}")
        price_per_unit = total_value / total_units
        desired = order.notional / price_per_unit

        by_basis = sorted(
            positions, key=lambda p: (p.open_rate, p.units), reverse=True
        )
        mixed = _allocate_trim(by_basis, desired, allow_full_close=True)
        whole = _allocate_trim(by_basis, desired, allow_full_close=False)
        plan, plan_units = whole, _plan_units(whole)
        if _plan_units(mixed) >= plan_units * _MIN_TRIM_FILL_RATIO:
            plan, plan_units = mixed, _plan_units(mixed)

        if not plan:
            logger.info(
                "TRIM %s: $%.2f (~%.4f units at $%.2f) cannot be expressed — no "
                "sub-position is a whole unit or small enough to close outright; "
                "leaving position as-is",
                order.ticker, order.notional, desired, price_per_unit,
            )
            return 0.0

        for i, (pos, units, full_close) in enumerate(plan):
            if i > 0:
                time.sleep(0.5)
            resp = self._close_position(
                pos.position_id, pos.instrument_id,
                # None ⇒ close the sub-position outright (any size). A partial
                # deduction stays a whole int at the call site, so the 776
                # fractional rejection can't be reintroduced from here.
                units_to_deduct=None if full_close else int(units),
            )
            # Log the broker's order acknowledgement. eToro returns an
            # orderForClose object (orderID/statusID) on HTTP 200 even for
            # partial closes it later drops without executing, so recording the
            # id is what makes such a drop traceable after the fact.
            ack = (resp or {}).get("orderForClose") or {}
            logger.info(
                "TRIM %s pos=%d: %.6f units ($%.2f, %s) — orderID=%s statusID=%s",
                order.ticker, pos.position_id, units, units * price_per_unit,
                "full close" if full_close else "partial", ack.get("orderID"),
                ack.get("statusID"),
            )
        return plan_units

    def _cancel_open_order(self, order_id: int) -> dict:
        resp = self._client.delete(
            self._execution_url(f"market-open-orders/{order_id}"),
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    def cancel_open_orders(self) -> int:
        """Cancel all pending open orders. Returns number cancelled."""
        resp = self._request_with_retry(
            "get", self._portfolio_url(), headers=self._headers()
        )
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
            resp = self._request_with_retry(
                "get",
                f"{BASE_URL}/market-data/instruments/rates",
                params={"instrumentIds": str(instrument_id)},
                headers=self._headers(),
            )
            data = resp.json()
            for rate in data.get("rates", []):
                if rate.get("instrumentID") == instrument_id:
                    return float(rate.get("lastExecution", rate.get("ask", 0)))
            logger.warning("No rate found for instrument %d in response: %s", instrument_id, data)
        except Exception as e:
            logger.error("Failed to get rate for instrument %d: %s", instrument_id, e)
        return None

    def _await_position_state(
        self,
        tickers: set[str],
        *,
        want_present: bool,
        timeout: float = _SETTLE_TIMEOUT_SECONDS,
        interval: float | None = None,
    ) -> set[str]:
        """Poll positions until every ticker reaches the wanted presence state.

        `want_present=True` waits for each ticker to APPEAR (open confirmed);
        `False` waits for it to DISAPPEAR (close confirmed). Returns the set of
        tickers that never reached the wanted state within `timeout` — empty
        means all confirmed.

        eToro's positions endpoint lags fills, so a fill is real but not yet
        visible; polling re-reads until it shows up rather than failing on the
        first miss. When `interval <= 0` (tests patch `_CLOSE_SETTLE_SECONDS` to
        0) this collapses to a single deterministic re-read, preserving the exact
        call count existing tests assert on.
        """
        if interval is None:
            interval = _CLOSE_SETTLE_SECONDS
        pending = set(tickers)
        deadline = time.monotonic() + timeout
        while True:
            time.sleep(interval)  # settle before (re-)reading
            held = {p.ticker for p in self.get_positions_detailed()}
            pending = {t for t in pending if (t not in held) == want_present}
            if not pending or interval <= 0 or time.monotonic() >= deadline:
                return pending

    def _await_units_reduction(
        self,
        baseline: dict[str, tuple[float, float]],
        *,
        timeout: float = _SETTLE_TIMEOUT_SECONDS,
        interval: float | None = None,
    ) -> set[str]:
        """Poll until each trimmed ticker's unit count actually drops.

        `baseline` maps ticker -> (units_held_before, units_requested). Returns
        the tickers whose reduction never became visible — empty means all
        confirmed.

        Trims cannot be verified by presence the way full exits are (a trim
        leaves a residual position by design), so they were previously the one
        close path with NO verification at all. eToro accepts a partial close
        with HTTP 200 and a real orderID, then may silently never execute it —
        observed live 2026-08-03, where two trims were reported "submitted"
        while units and cash stayed unchanged for 7+ minutes and the order
        appeared in no queue. Checking units closes that gap.

        A trim counts as confirmed once at least half the requested reduction is
        visible, which tolerates a partial fill across sub-positions while still
        catching "nothing happened at all".
        """
        if interval is None:
            interval = _CLOSE_SETTLE_SECONDS
        pending = set(baseline)
        deadline = time.monotonic() + timeout
        while True:
            time.sleep(interval)  # settle before (re-)reading
            units_now: dict[str, float] = {}
            for p in self.get_positions_detailed():
                units_now[p.ticker] = units_now.get(p.ticker, 0.0) + p.units
            pending = {
                t for t in pending
                if (baseline[t][0] - units_now.get(t, 0.0)) < baseline[t][1] * 0.5
            }
            if not pending or interval <= 0 or time.monotonic() >= deadline:
                return pending

    def _await_sell_proceeds(
        self,
        starting_cash: float,
        expected_proceeds: float,
        *,
        timeout: float = _SETTLE_TIMEOUT_SECONDS,
        interval: float | None = None,
    ) -> float:
        """Poll until closed-position proceeds are credited, returning cash.

        eToro credits closes asynchronously. Reading cash immediately after
        submitting sells sees the PRE-sell balance, so every Phase-2 buy is
        sized against money that has not landed yet — observed live 2026-08-03,
        where buys were sized ~1s after the sells and saw $19.68 instead of the
        ~$863 the trims would have freed, undersizing one buy and skipping
        another entirely. Waits for most of the expected proceeds rather than a
        fixed sleep, and returns whatever cash is available at timeout so a
        slow credit degrades to the old behaviour instead of hanging.
        """
        if interval is None:
            interval = _CLOSE_SETTLE_SECONDS
        cash = starting_cash
        if expected_proceeds <= 0:
            return cash
        target = starting_cash + expected_proceeds * 0.9
        deadline = time.monotonic() + timeout
        while True:
            time.sleep(interval)
            cash = max(cash, self.get_account().get("cash", 0.0))
            if cash >= target or interval <= 0 or time.monotonic() >= deadline:
                return cash

    def execute_orders(
        self,
        orders: list[RebalanceOrder],
        dry_run: bool = False,
        stop_loss_pct: float = 0.0,
        target_weights: dict[str, float] | None = None,
    ) -> list[RebalanceOrder]:
        """Execute rebalance: close sells first, then open buys.

        target_weights: optional {ticker: weight} for non-equal sizing; the
            Phase-2 buy recompute scales each buy by its weight (None ⇒ equal).
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

        self.cancel_open_orders()

        # Phase 1: close positions to sell
        detailed = self.get_positions_detailed()
        pos_by_ticker: dict[str, list[EtoroPosition]] = {}
        for p in detailed:
            pos_by_ticker.setdefault(p.ticker, []).append(p)

        trim_baseline: dict[str, tuple[float, float]] = {}
        for order in sells:
            positions_to_close = pos_by_ticker.get(order.ticker, [])
            if not positions_to_close:
                order.status = "error: no position found"
                logger.error("SELL %s — no position found", order.ticker)
                continue
            try:
                if order.trim:
                    units_before = sum(p.units for p in positions_to_close)
                    requested = self._execute_trim(order, positions_to_close)
                    if requested <= 0:
                        # No sub-position was a whole unit or small enough to
                        # close outright, so nothing was submitted. Not a
                        # failure, and must not abort the buys. `_execute_trim`
                        # has already logged the specific reason.
                        order.status = "skipped: not expressible in tradeable units"
                        order.notional = 0.0
                        continue
                    trim_baseline[order.ticker] = (units_before, requested)
                else:
                    for i, pos in enumerate(positions_to_close):
                        if i > 0:
                            time.sleep(0.5)
                        self._close_position(pos.position_id, pos.instrument_id)
                order.status = "submitted"
                logger.info("SELL %s $%.2f — submitted", order.ticker, order.notional)
            except Exception as e:
                order.status = f"error: {e}"
                logger.error("SELL %s $%.2f — FAILED: %s", order.ticker, order.notional, e)

        # Verify full exits actually cleared. eToro's close endpoint returns HTTP
        # 200 even when the close does not go through, so a no-op close is
        # otherwise marked "submitted" and the abort guard below never fires —
        # leaving the name still held while Phase-2 buys spend cash that was never
        # freed. Poll positions (the endpoint lags the fill by tens of seconds)
        # and demote any full exit still held after the settle window to an error,
        # routing it through the same abort path as a raised close (mirrors
        # AlpacaBroker).
        full_exits = [o for o in sells if not o.trim and not o.status.startswith("error")]
        if full_exits:
            still_held = self._await_position_state(
                {o.ticker for o in full_exits}, want_present=False,
            )
            for order in full_exits:
                if order.ticker in still_held:
                    order.status = "error: close did not clear (still held)"
                    logger.error(
                        "SELL %s (full exit) — still held %.0fs after close; "
                        "treating as unconfirmed (settle lag or a genuine reject)",
                        order.ticker, _SETTLE_TIMEOUT_SECONDS,
                    )

        # Verify trims actually reduced the position. A trim leaves a residual by
        # design, so the presence check above cannot see it — units must be
        # compared instead. Without this, an accepted-but-never-executed partial
        # close is logged "submitted" and Phase-2 buys spend cash it never freed.
        trims = [o for o in sells
                 if o.trim and o.ticker in trim_baseline
                 and not o.status.startswith("error")]
        if trims:
            unreduced = self._await_units_reduction(trim_baseline)
            for order in trims:
                if order.ticker in unreduced:
                    order.status = "error: trim did not take effect (units unchanged)"
                    logger.error(
                        "SELL %s (trim) — units unchanged %.0fs after close; the "
                        "order was acknowledged but never executed",
                        order.ticker, _SETTLE_TIMEOUT_SECONDS,
                    )

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
            # Let the sell proceeds land before sizing anything: eToro credits
            # closes asynchronously, and sizing off the pre-sell balance
            # undersizes or skips buys that the preview showed as fundable.
            confirmed_sells = [o for o in sells if o.status == "submitted"]
            if confirmed_sells:
                pre_sell_cash = self.get_account().get("cash", 0.0)
                self._await_sell_proceeds(
                    pre_sell_cash, sum(o.notional for o in confirmed_sells),
                )
            account = self.get_account()
            buy_tickers = {o.ticker for o in buys}
            exited_tickers = {o.ticker for o in sells
                              if o.status == "submitted" and not o.trim}
            all_candidates = list(buy_tickers | (set(pos_by_ticker.keys()) - exited_tickers))
            current = self.resolve_positions(all_candidates)
            for t in exited_tickers:
                current.pop(t, None)
            all_target = list(set(current.keys()) | buy_tickers)
            equity = account["equity"]
            n_equal = len(all_target)

            # Fail loud if a buy is missing from a provided weights dict: it would
            # silently fall back to equal-weight 1/n_equal here while the preview
            # sized it differently, so the executed size would diverge.
            if target_weights is not None:
                missing = [o.ticker for o in buys if o.ticker not in target_weights]
                if missing:
                    logger.warning(
                        "Phase-2 sizing: %d buy(s) missing from target_weights "
                        "(%s) — falling back to equal-weight 1/%d; size may diverge "
                        "from preview.", len(missing), missing, n_equal,
                    )

            logger.info(
                "Cash available: $%.2f — sizing %d buys (%s)",
                account["cash"], len(buys),
                weighting_label(target_weights),
            )
            submitted_buys = 0
            # eToro closes are async and the freed cash may not have settled yet.
            # Clamp buys to available cash so a fully-invested swap can't
            # over-submit beyond what's actually spendable.
            cash_remaining = account.get("cash", 0.0)
            for order in buys:
                current_value = current.get(order.ticker, 0)
                target_value = target_value_for(
                    order.ticker, equity, target_weights, n_equal
                )
                order.notional = max(target_value - current_value, 0)
                order.notional = min(order.notional, max(cash_remaining, 0))
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
                    else:
                        order.status = "error: could not get rate for stop-loss"
                        logger.error(
                            "BUY %s — skipped, could not determine stop-loss rate",
                            order.ticker,
                        )
                        continue

                if submitted_buys > 0:
                    time.sleep(0.5)
                try:
                    self._open_position(iid, order.notional, stop_loss_rate=stop_rate)
                    order.status = "submitted"
                    submitted_buys += 1
                    cash_remaining -= order.notional
                    logger.info("BUY %s $%.2f — submitted", order.ticker, order.notional)
                except Exception as e:
                    order.status = f"error: {e}"
                    logger.error("BUY %s $%.2f — FAILED: %s", order.ticker, order.notional, e)

            # Verify opens cleared, mirroring the full-exit verification above.
            # _open_position returns HTTP 200 even when the open does not go
            # through, which both under-invests the book AND (since cash_remaining
            # was already decremented) shrinks every later buy in this loop. Poll
            # positions (the endpoint lags the fill) and surface any submitted buy
            # that produced no holding so a partial rebalance isn't silently logged
            # as fully "submitted". (A buy that adds to an already-held name can't
            # be confirmed by presence alone — this catches the common
            # swap-in-of-a-new-name case.)
            submitted = [o for o in buys if o.status == "submitted"]
            if submitted:
                not_held = self._await_position_state(
                    {o.ticker for o in submitted}, want_present=True,
                )
                for order in submitted:
                    if order.ticker in not_held:
                        order.status = "error: open did not clear (not held)"
                        logger.error(
                            "BUY %s — not held %.0fs after open; treating as "
                            "unconfirmed (settle lag or a genuine reject)",
                            order.ticker, _SETTLE_TIMEOUT_SECONDS,
                        )

            # Capture fill prices (telemetry): a freshly-opened position's
            # open_rate IS its execution price, so tracking can measure eToro's
            # spread/markup vs the model close. Best-effort, post-execution, must
            # never disturb the (already-complete) rebalance. Only confirmed
            # swap-in buys get a clean single-lot open_rate; sells aren't captured
            # (the position is gone after a close). arrival_price is left None —
            # for eToro the fill-vs-model-close gap already isolates the spread.
            confirmed = [o for o in submitted if o.status == "submitted"]
            if confirmed:
                try:
                    open_rates = {p.ticker: p.open_rate
                                  for p in self.get_positions_detailed() if p.open_rate}
                    for order in confirmed:
                        rate = open_rates.get(order.ticker)
                        if rate:
                            order.fill_price = float(rate)
                except Exception:  # noqa: BLE001 — telemetry only
                    logger.debug("Could not capture eToro fill prices")

        return orders


__all__ = ["EtoroBroker", "EtoroPosition", "RebalanceOrder", "compute_equity"]
