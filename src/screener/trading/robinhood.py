from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from zoneinfo import ZoneInfo

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

    @property
    def price(self) -> float:
        """The per-share price the rebalance sized against.

        Use this to convert a trim's notional back to shares — never a raw quote
        field. ``parse_positions`` coalesces last / extended-hours / average cost
        when a quote is null or zero, so re-deriving the price from the quote
        diverges from the one behind ``market_value`` in exactly the halted and
        after-hours cases that coalesce exists for, and the resulting quantity
        order would not match the dollar amount that was sized.
        """
        return self.market_value / self.quantity if self.quantity else 0.0


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
        unwraps a `{"data": {...}}` envelope if passed whole) carries
        ``total_value`` (positions + all cash), ``equity_value`` (positions
        only), ``cash``, and a nested ``buying_power``.

        ``equity`` — the sizing base — is the account's own capital; the target
        book is 100% of it and no more. **Never size off ``buying_power``**: the
        agentic account is a MARGIN account (aug-2026), so buying power exceeds
        account value by the margin multiple and deploying it would lever the
        book. It is returned for reporting and pre-trade checks only.

        ``unsettled_cash = max(cash − buying_power, 0)`` is a cash-account safety
        net — proceeds not yet tradable, which would otherwise over-buy the names
        filled first while the rest waited on settlement. It is 0 on margin, so
        ``equity == total_value``; kept because it costs nothing if the account
        ever reverts to cash. An explicit hold-back is separate — see
        ``compute_rebalance_orders(reserve=)``.
        """
        d = portfolio_data.get("data", portfolio_data)

        total_value = float(d.get("total_value", 0) or 0)
        equity_value = float(d.get("equity_value", 0) or 0)
        cash = float(d.get("cash", 0) or 0)

        bp = d.get("buying_power", 0)
        if isinstance(bp, dict):
            bp = bp.get("buying_power", 0)
        buying_power = float(bp or 0)

        # Clamp at 0 so the margin account's buying_power > cash can't inflate
        # the base — we never size into margin. Subtract from total_value (not
        # equity_value + buying_power: identical value, but this form stays
        # bit-exact = total_value when nothing is unsettled, i.e. always now).
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
        lot_plan: LotPlan | None = None,
    ) -> dict:
        """Format a RebalanceOrder as MCP place_equity_order parameters.

        ``lot_plan`` (from ``plan_trim_lots``) turns a TRIM into a specified-lot
        sell: the broker rejects ``tax_lots`` alongside ``dollar_amount``, so a
        lot-selected trim must be placed by quantity instead. A plan that fell
        back to FIFO (empty ``tax_lots``) drops through to the dollar path.

        Omitting ``lot_plan`` on a trim is warned about rather than treated as
        the same thing: a deliberate fallback carries a ``note`` explaining
        itself, whereas a forgotten planning step would otherwise be
        indistinguishable from correct FIFO behaviour, at execution time and in
        the trade log alike.
        """
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

        if order.side == "sell" and order.trim:
            if lot_plan and lot_plan.tax_lots:
                params["quantity"] = lot_plan.quantity
                params["tax_lots"] = lot_plan.tax_lots
                return params
            if lot_plan is None:
                logger.warning(
                    "Trim of %s placed with no lot plan — the broker's FIFO "
                    "default applies; call plan_trim_lots() to select lots",
                    order.ticker,
                )

        params["dollar_amount"] = str(round(order.notional, 2))
        return params


# --- HIFO tax-lot selection --------------------------------------------------
#
# Robinhood defaults to FIFO, which cycles the oldest/lowest-basis lots and
# converts long-term gains into short-term ones. `place_equity_order` accepts
# `tax_lots` (aug-2026) so a SELL can name its lots — the same HIFO ordering
# eToro's `_execute_trim` uses, worth +0.11/+0.21/+0.31pp/yr after-tax CAGR at
# the 24/15, 35/23.8, 46/33 ST/LT brackets in the `tax_impact.py` measurement.
#
# Only TRIMS benefit: a full exit realizes every lot whatever order you name.

MAX_TAX_LOTS = 30                  # broker cap on one specified-lot sell
_QTY = Decimal("0.000001")         # Robinhood accepts 6 decimal places


@dataclass
class LotPlan:
    """How a trim should be placed.

    ``tax_lots`` empty ⇒ **fall back to a plain FIFO sell**. Selecting lots is
    the secondary goal; trimming the right dollar amount is the primary one, so
    whenever the lots can't cover the trade the trade still goes out at the
    right size and ``note`` says why the tax benefit was skipped.
    """

    quantity: str                  # exact sum of the lot quantities
    tax_lots: list[dict]
    note: str


def _dec(v) -> Decimal | None:
    # NB: deliberately not `_to_float` — that maps "0"/"0.0000" to None, a
    # quote-field convention ("zero price means no quote") that would discard a
    # genuine zero-basis lot, and float can't hold the exact-sum guarantee.
    try:
        return Decimal(str(v))
    except (ArithmeticError, TypeError, ValueError):
        return None


def plan_trim_lots(
    notional: float,
    price: float,
    lots_mcp: dict | list[dict],
) -> LotPlan:
    """Pick the lots for a ``notional``-dollar trim, highest cost basis first.

    ``lots_mcp`` is a ``get_equity_tax_lots`` response for the ticker. Highest
    basis first realizes the smallest gain (or largest loss) per share sold.

    Exactness matters: the broker rejects a lot list whose quantities don't sum
    to the order quantity, so everything is Decimal at 6dp and the shares are
    floored — a trim never oversells to make the arithmetic land.

    Lots are skipped when ``is_selectable`` is false (still syncing — a lot
    bought the same session cannot be named) or the basis is unknown. If what
    survives can't cover the trim, or would need more than ``MAX_TAX_LOTS``, the
    plan falls back to FIFO rather than shrinking the trade.

    That response is PAGINATED. An unread page is reported in ``note`` rather
    than passed over silently: a long-held name accumulates more lots than one
    page holds, and truncation both narrows the HIFO choice and can force a
    fallback that would otherwise blame ``is_selectable``. Drain the cursor
    before calling for the full benefit.
    """
    if isinstance(lots_mcp, dict):
        body = lots_mcp.get("data", lots_mcp)
        rows, more_pages = body.get("tax_lots", []), bool(body.get("next"))
    else:
        rows, more_pages = lots_mcp, False

    dec_price = _dec(price)
    dec_notional = _dec(notional)
    if not dec_price or dec_price <= 0 or not dec_notional or dec_notional <= 0:
        return LotPlan("0", [], "no positive price/notional — cannot size a trim")

    shares = (dec_notional / dec_price).quantize(_QTY, rounding=ROUND_DOWN)
    if shares <= 0:
        return LotPlan("0", [], "trim rounds to zero shares")

    candidates: list[tuple[Decimal, Decimal, str]] = []
    unselectable = 0
    for lot in rows:
        lot_id = lot.get("open_lot_id")
        if not lot_id:
            continue
        if not lot.get("is_selectable"):
            unselectable += 1
            continue
        avail = _dec(lot.get("quantity_available"))
        if avail is not None:
            avail = avail.quantize(_QTY, rounding=ROUND_DOWN)
        basis = _dec(lot.get("cost_per_share"))
        if basis is None:
            # Basis can arrive only as a lot total (cost_per_share pending).
            total, qty = _dec(lot.get("tax_cost_basis")), _dec(lot.get("quantity"))
            basis = total / qty if total and qty and qty > 0 else None
        if avail is None or avail <= 0 or basis is None:
            continue
        candidates.append((basis, avail, lot_id))

    candidates.sort(key=lambda c: c[0], reverse=True)

    picked: list[dict] = []
    remaining = shares
    for _basis, avail, lot_id in candidates:
        if remaining <= 0 or len(picked) >= MAX_TAX_LOTS:
            break
        take = min(avail, remaining)
        picked.append({
            "open_lot_id": lot_id,
            "quantity": format(take, "f"),
        })
        remaining -= take

    truncated = "; more lot pages were not fetched" if more_pages else ""
    qty_str = format(shares, "f")
    if remaining > 0:
        covered = shares - remaining
        why = (
            f"only {format(covered, 'f')} of {qty_str} shares are in selectable "
            f"lots ({unselectable} lot(s) still syncing){truncated}"
            if len(picked) < MAX_TAX_LOTS else
            f"would need more than the {MAX_TAX_LOTS}-lot cap"
        )
        return LotPlan(qty_str, [], f"FIFO fallback — {why}; trim size preserved")

    return LotPlan(qty_str, picked, f"HIFO across {len(picked)} lot(s){truncated}")


# --- Execution helpers -------------------------------------------------------
#
# The agentic account is a MARGIN account (aug-2026), so same-day sale proceeds
# are immediately re-deployable and a rebalance finishes in one session. The
# cash-account machinery this once needed — capping buys at settled buying
# power, logging the remainder as `deferred_buys`, and an unattended next-day
# T+1 follow-up routine to deploy it — is gone; recover it from git history if
# the account ever reverts to cash.

_EASTERN = ZoneInfo("America/New_York")
_MARKET_STALE_SECONDS = 300        # no print in 5 min ⇒ not a live session


def eastern_now(now: datetime | None = None) -> datetime:
    """Now, in US/Eastern. Stamp trade logs from this: the VM clock is UTC, so a
    bare ``datetime.now()`` rolls over mid-evening ET and would misjudge which
    trading day it is — and a UTC filename beside an ET ``date`` field can
    disagree about the day."""
    return (now or datetime.now(tz=UTC)).astimezone(_EASTERN)


def _trade_time(quote: dict) -> datetime | None:
    raw = quote.get("venue_last_trade_time")
    if not raw:
        return None
    try:  # timestamps carry nanoseconds; fromisoformat wants ≤6 fractional digits
        head, _, frac = str(raw).rstrip("Z").partition(".")
        return datetime.fromisoformat(f"{head}.{frac[:6] or '0'}+00:00")
    except ValueError:
        return None


def market_is_open(
    quotes: dict[str, dict], now: datetime | None = None,
) -> tuple[bool, str]:
    """Is the US market open for regular-hours trading right now?

    Decided empirically from the freshest ``venue_last_trade_time`` across the
    quotes rather than from a holiday calendar: if any S&P name printed a trade
    seconds ago, the market is open, and no calendar can drift out of date. The
    ET clock window is a secondary sanity check.

    Check this before placing any Robinhood order: an out-of-hours **market
    order does not fail — it queues** for the next open, so a mistimed run
    leaves orders sitting overnight, executing at a price nobody reviewed.
    Robinhood's order review does not reliably alert on it.
    """
    now = now or datetime.now(tz=UTC)
    et = now.astimezone(_EASTERN)
    if et.weekday() >= 5:
        return False, f"{et:%a} — weekend"
    if not (et.replace(hour=9, minute=30, second=0, microsecond=0)
            <= et
            <= et.replace(hour=16, minute=0, second=0, microsecond=0)):
        return False, f"{et:%H:%M} ET is outside 09:30–16:00"

    stamps = [t for t in (_trade_time(q) for q in quotes.values()) if t]
    if not stamps:
        return False, "no trade timestamps in the quotes to confirm an open market"
    age = (now - max(stamps)).total_seconds()
    if age > _MARKET_STALE_SECONDS:
        return False, (
            f"last print was {age / 60:.0f} min ago — market appears closed "
            "(holiday?)"
        )
    return True, f"open ({et:%H:%M} ET, last print {age:.0f}s ago)"


def parse_order_fills(
    orders_mcp: dict | list[dict],
    arrival: dict[str, float] | None = None,
    trims: set[str] | None = None,
) -> list[dict]:
    """Normalize ``get_equity_orders`` responses into trade-log order rows.

    ``notional`` falls back to ``quantity × fill_price`` when the order carries
    no ``dollar_based_amount``. Full exits are placed BY QUANTITY, and
    ``execution_decomposition`` skips any row lacking a positive notional — so
    without the fallback the execution-quality table would silently measure only
    buys and trims, never the sell side of a swap.

    ``trims`` names the tickers whose sell was a partial trim (take it from the
    computed ``RebalanceOrder``s — the MCP response can't distinguish a trim
    from a full exit); every other sell is recorded as a full exit.
    """
    rows = orders_mcp
    if isinstance(rows, dict):
        rows = rows.get("data", rows).get("orders", [])

    out = []
    for o in rows:
        ticker = o.get("symbol", "")
        side = o.get("side", "buy")
        fill = _to_float(o.get("average_price"))
        quantity = _to_float(o.get("cumulative_quantity"))
        notional = _to_float((o.get("dollar_based_amount") or {}).get("amount"))
        if notional is None and fill and quantity:
            notional = round(fill * quantity, 2)
        out.append({
            "ticker": ticker,
            "side": side,
            "notional": notional,
            "status": o.get("state"),
            "trim": side == "sell" and ticker in (trims or set()),
            "arrival_price": (arrival or {}).get(ticker),
            "fill_price": fill,
            "quantity": quantity,
            "fees": _to_float(o.get("fees")) or 0.0,
        })
    return out


__all__ = [
    "MAX_TAX_LOTS",
    "LotPlan",
    "RobinhoodBroker",
    "RobinhoodPosition",
    "RebalanceOrder",
    "eastern_now",
    "market_is_open",
    "parse_order_fills",
    "plan_trim_lots",
]
