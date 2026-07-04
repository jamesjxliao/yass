"""Live-vs-backtest tracking record.

Builds the true out-of-sample record from rebalance trade logs
(``results/trades/*.json``): for each broker, the holdings held between
executed rebalances, the MODEL return of those holdings (equal-weight,
daily closes, the backtest runner's boundary convention), and the REALIZED
return from account-equity snapshots adjusted for cash flows (deposits /
withdrawals). The gap between the two series is the all-in implementation
cost (fills, fees, timing, partial syncs) — the number every in-sample
robustness check cannot provide.

Design notes:
- Trade-log schemas drift per broker session (``account`` may be a string,
  Robinhood uses ``account_before``/``account_after``, holdings may live in
  ``picks``/``holdings``/``final_holdings``). ``parse_trade_log`` normalizes
  defensively and returns None only for unreadable files.
- EVERY log (dry-run included) contributes an equity observation; only
  executed logs (dry_run false + orders/holdings evidence) define holding
  periods.
- Cash flows come from a user-editable ledger; ``detect_cash_flows`` seeds
  candidates from same-day equity jumps. Realized return over a period is
  eq_end / (eq_start + inflows) - 1 (deposit-at-start approximation).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import polars as pl

_TS_RE = re.compile(r"(\d{8})_(\d{6})")


@dataclass
class LogRecord:
    broker: str  # alpaca_paper | alpaca_live | etoro | robinhood
    ts: datetime
    log_date: date
    equity: float | None
    holdings: list[str]
    executed: bool
    file: str
    note: str = ""
    orders: list = field(default_factory=list)  # raw order dicts (for exec-cost decomposition)


@dataclass
class CashFlow:
    broker: str
    on: date
    amount: float
    note: str = ""
    inferred: bool = False
    ts: datetime | None = None  # intraday ordering vs same-day rebalance events

    def effective_ts(self) -> datetime:
        return self.ts or datetime(self.on.year, self.on.month, self.on.day, 12)


@dataclass
class Period:
    broker: str
    start: date
    end: date
    holdings: list[str]
    model_ret: float | None = None
    realized_ret: float | None = None
    spy_ret: float | None = None
    flows: float = 0.0
    start_equity: float | None = None
    end_equity: float | None = None
    flags: list[str] = field(default_factory=list)


def _first_dict(*candidates) -> dict:
    for c in candidates:
        if isinstance(c, dict):
            return c
    return {}


def _equity_of(d: dict) -> float | None:
    acct = _first_dict(d.get("account_after"), d.get("account"))
    for k in ("equity", "total_value", "portfolio_value"):
        v = acct.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _holdings_of(d: dict) -> list[str]:
    fh = d.get("final_holdings")
    if isinstance(fh, dict) and fh:
        return sorted(fh.keys())
    for k in ("final_holdings", "holdings", "picks"):
        v = d.get(k)
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return sorted(v)
    return []


def _executed_of(d: dict) -> bool:
    if d.get("dry_run") is not False:
        return False
    for k in ("orders", "orders_executed_today"):
        orders = d.get(k)
        if isinstance(orders, list):
            for o in orders:
                if isinstance(o, dict) and o.get("status") not in ("dry_run",):
                    return True
    # hand-written logs (deploy-cash sessions) may carry holdings but no
    # normalized orders list — treat a non-dry log with final holdings as
    # executed so the holdings timeline stays continuous
    return isinstance(d.get("final_holdings"), (dict, list)) or isinstance(
        d.get("holdings"), list)


def parse_trade_log(path: Path) -> LogRecord | None:
    try:
        d = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(d, dict):
        return None
    broker = d.get("broker", "alpaca")
    if broker == "alpaca":
        broker = "alpaca_paper" if d.get("mode") == "PAPER" else "alpaca_live"
    m = _TS_RE.search(path.name)
    if m:
        ts = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    else:
        try:
            ts = datetime.fromisoformat(d.get("date", "")).replace(hour=12)
        except ValueError:
            return None
    try:
        log_date = date.fromisoformat(d["date"])
    except (KeyError, ValueError):
        log_date = ts.date()
    raw_orders = d.get("orders")
    orders = [o for o in raw_orders if isinstance(o, dict)] if isinstance(raw_orders, list) else []
    return LogRecord(
        broker=broker, ts=ts, log_date=log_date, equity=_equity_of(d),
        holdings=_holdings_of(d), executed=_executed_of(d), file=path.name,
        note=str(d.get("note", ""))[:200], orders=orders,
    )


def load_history(trade_dir: Path) -> list[LogRecord]:
    recs = [r for p in sorted(trade_dir.glob("*.json"))
            if (r := parse_trade_log(p)) is not None]
    return sorted(recs, key=lambda r: (r.broker, r.ts))


def rebalance_events(recs: list[LogRecord]) -> dict[str, list[LogRecord]]:
    """Last executed log per (broker, day) — later same-day logs supersede
    earlier ones (fixups / corrective completions)."""
    out: dict[str, list[LogRecord]] = {}
    for r in recs:
        if not r.executed or not r.holdings:
            continue
        seq = out.setdefault(r.broker, [])
        if seq and seq[-1].ts.date() == r.ts.date():
            seq[-1] = r
        else:
            seq.append(r)
    return out


def detect_cash_flows(recs: list[LogRecord],
                      min_abs: float = 500.0,
                      min_frac: float = 0.02) -> list[CashFlow]:
    """Seed ledger candidates: equity jumps between consecutive same-day
    observations (market moves can't explain a 2%+/$500+ intraday step
    between two log snapshots minutes apart)."""
    flows: list[CashFlow] = []
    by_broker: dict[str, list[LogRecord]] = {}
    for r in recs:
        if r.equity is not None:
            by_broker.setdefault(r.broker, []).append(r)
    for broker, seq in by_broker.items():
        for a, b in zip(seq, seq[1:]):
            if a.ts.date() != b.ts.date():
                continue
            diff = b.equity - a.equity
            if abs(diff) >= min_abs and abs(diff) >= min_frac * max(a.equity, 1.0):
                flows.append(CashFlow(broker=broker, on=a.ts.date(),
                                      amount=round(diff, 2),
                                      note=f"same-day jump {a.file} -> {b.file}",
                                      inferred=True, ts=b.ts))
    return flows


def portfolio_return(tickers: list[str], start: date, end: date,
                     prices: pl.DataFrame) -> tuple[float | None, list[str]]:
    """Equal-weight return over [start, end): per ticker, first close on/after
    start to first close on/after end (falls back to last close before end —
    the runner's boundary-chaining convention). Returns (ret, flags)."""
    flags: list[str] = []
    rets: list[float] = []
    for t in tickers:
        px = prices.filter(pl.col("ticker") == t).sort("date")
        w = px.filter((pl.col("date") >= pl.lit(start)) & (pl.col("date") < pl.lit(end)))
        if w.is_empty():
            flags.append(f"no-price:{t}")
            continue
        sp = w["close"][0]
        after = px.filter(pl.col("date") >= pl.lit(end))
        ep = after["close"][0] if not after.is_empty() else w["close"][-1]
        rets.append(ep / sp - 1.0)
    if not rets:
        return None, flags
    if len(rets) < len(tickers):
        flags.append(f"partial:{len(rets)}/{len(tickers)}")
    return sum(rets) / len(rets), flags


def build_periods(events: dict[str, list[LogRecord]],
                  prices: pl.DataFrame,
                  ledger: list[CashFlow]) -> list[Period]:
    periods: list[Period] = []
    max_px_date = prices["date"].max() if not prices.is_empty() else None
    for broker, seq in events.items():
        for a, b in zip(seq, seq[1:]):
            start, end = a.ts.date(), b.ts.date()
            if start >= end:
                continue
            p = Period(broker=broker, start=start, end=end,
                       holdings=a.holdings,
                       start_equity=a.equity, end_equity=b.equity)
            p.model_ret, p.flags = portfolio_return(a.holdings, start, end, prices)
            p.spy_ret, _ = portfolio_return(["SPY"], start, end, prices)
            if max_px_date is not None and end > max_px_date:
                # the model side can't see the period's end yet — don't compare
                p.flags.append("pending-prices")
            p.flows = sum(f.amount for f in ledger
                          if f.broker == broker
                          and a.ts < f.effective_ts() <= b.ts)
            if a.equity and b.equity:
                base = a.equity + p.flows
                if base > 0:
                    p.realized_ret = b.equity / base - 1.0
                    if p.model_ret is not None and abs(
                            p.realized_ret - p.model_ret) > 0.05:
                        p.flags.append("gap>5% — missing cash flow?")
            periods.append(p)
    return periods


def _close_on_or_after(prices: pl.DataFrame, ticker: str, d: date) -> float | None:
    """The model's entry price: first close on/after ``d`` (runner convention)."""
    px = (prices.filter((pl.col("ticker") == ticker) & (pl.col("date") >= d))
          .sort("date"))
    return float(px["close"][0]) if not px.is_empty() else None


def execution_decomposition(events: dict[str, list[LogRecord]],
                            prices: pl.DataFrame) -> list[dict]:
    """Per executed rebalance, decompose the transaction cost the equity-gap
    can't attribute, using the per-order ``fill_price``/``arrival_price`` logged
    live. Cost is oriented so positive = transacted WORSE than the model close
    (paid more on a buy / received less on a sell).

      total   = fill      vs model close  (spread + timing; needs fill only)
      timing  = arrival   vs model close  (needs arrival_price too)
      slippage= fill       vs arrival     (pure execution; needs arrival too)

    Aggregated dollar-weighted to bp of traded notional and bp of book equity
    (book bp is what maps onto the period gap). Only orders that actually carry a
    fill contribute — pre-instrumentation logs yield an empty list.
    """
    out: list[dict] = []
    for broker, seq in events.items():
        for a in seq:
            rebal = a.ts.date()
            book = a.equity or 0.0
            tot = tim = slip = tot_notional = 0.0
            arrival_notional = 0.0  # notional of ONLY the arrival-instrumented fills
            n = 0
            for o in a.orders:
                fill = o.get("fill_price")
                notional = o.get("notional")
                if not isinstance(fill, (int, float)) or not isinstance(notional, (int, float)):
                    continue
                if fill <= 0 or notional <= 0:
                    continue
                mc = _close_on_or_after(prices, o.get("ticker", ""), rebal)
                if mc is None or mc <= 0:
                    continue
                buy = o.get("side") == "buy"
                tot += notional * ((fill - mc) / mc if buy else (mc - fill) / mc)
                tot_notional += notional
                n += 1
                arr = o.get("arrival_price")
                if isinstance(arr, (int, float)) and arr > 0:
                    arrival_notional += notional
                    slip += notional * ((fill - arr) / arr if buy else (arr - fill) / arr)
                    tim += notional * ((arr - mc) / mc if buy else (mc - arr) / mc)
            if n == 0:
                continue
            out.append({
                "broker": broker, "date": str(rebal), "n_fills": n,
                "traded_notional": tot_notional,
                "exec_cost_bp_traded": 10000 * tot / tot_notional if tot_notional else None,
                "exec_cost_bp_book": 10000 * tot / book if book else None,
                # Divide the timing/slippage numerators by the notional they were
                # summed over (arrival-instrumented fills only) — NOT tot_notional,
                # which also counts fills missing an arrival price and would dilute
                # the split whenever a rebalance mixes instrumented and
                # uninstrumented fills (e.g. a transient quote failure leaves one
                # order with no arrival_price).
                "timing_bp_traded": (10000 * tim / arrival_notional
                                     if arrival_notional else None),
                "slippage_bp_traded": (10000 * slip / arrival_notional
                                       if arrival_notional else None),
            })
    return out


def load_or_seed_ledger(recs: list[LogRecord], ledger_path: Path,
                        reseed: bool = False) -> list[CashFlow]:
    """Load the user-editable cash-flow ledger; seed it with detected
    candidates on first run (or when reseed is set)."""
    if ledger_path.exists() and not reseed:
        rows = json.loads(ledger_path.read_text())
        return [CashFlow(broker=r["broker"], on=date.fromisoformat(r["date"]),
                         amount=float(r["amount"]), note=r.get("note", ""),
                         inferred=bool(r.get("inferred", False)),
                         ts=datetime.fromisoformat(r["ts"]) if r.get("ts") else None)
                for r in rows]
    flows = detect_cash_flows(recs)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(
        [{"broker": f.broker, "date": str(f.on), "amount": f.amount,
          "ts": f.ts.isoformat() if f.ts else None,
          "note": f.note, "inferred": f.inferred} for f in flows], indent=2))
    print(f"[ledger] seeded {len(flows)} candidate cash flows -> {ledger_path}\n"
          f"[ledger] review/edit that file (amounts, missed deposits), then re-run.")
    return flows


def report(db_path: str, trade_dir: Path, track_dir: Path,
           reseed: bool = False) -> None:
    """Print the per-broker live-vs-backtest report and save it as JSON.

    Cumulative rows compound ONLY periods where both sides are measured and
    prices are final, so the comparison stays apples-to-apples.
    """
    from screener.data.cache import CacheManager

    recs = load_history(trade_dir)
    events = rebalance_events(recs)
    ledger = load_or_seed_ledger(recs, track_dir / "cash_flows.json", reseed)

    tickers = sorted({t for seq in events.values() for r in seq for t in r.holdings}
                     | {"SPY"})
    cache = CacheManager(db_path)
    ph = ",".join("?" * len(tickers))
    prices = cache.to_polars(
        f"SELECT ticker, date, close FROM price_cache "
        f"WHERE ticker IN ({ph}) AND date >= '2026-03-01'", tickers)
    cache.close()

    periods = build_periods(events, prices, ledger)

    def pct(x):
        return f"{100 * x:+7.2f}%" if x is not None else "      \u2014"

    rows = []
    for broker in sorted(events):
        ps = [p for p in periods if p.broker == broker]
        if not ps:
            continue
        print(f"\n=== {broker} ===")
        print(f"{'period':<25}{'n':>3}{'model':>9}{'realized':>10}{'SPY':>9}"
              f"{'gap(bp)':>9}  flags")
        cum_m = cum_r = cum_s = 1.0
        for p in ps:
            gap = (None if p.model_ret is None or p.realized_ret is None
                   else 10000 * (p.realized_ret - p.model_ret))
            if (p.model_ret is not None and p.realized_ret is not None
                    and "pending-prices" not in p.flags):
                cum_m *= 1 + p.model_ret
                cum_r *= 1 + p.realized_ret
                if p.spy_ret is not None:
                    cum_s *= 1 + p.spy_ret
            line = (f"{str(p.start)} - {str(p.end):<13}{len(p.holdings):>3}"
                    f"{pct(p.model_ret)}{pct(p.realized_ret)}{pct(p.spy_ret)}")
            line += f"{gap:>+9.0f}" if gap is not None else f"{'\u2014':>9}"
            if p.flags:
                line += f"  {';'.join(p.flags)}"
            print(line)
            rows.append({**asdict(p), "start": str(p.start), "end": str(p.end),
                         "gap_bps": gap})
        print(f"{'CUMULATIVE':<25}{'':>3}{pct(cum_m - 1)}{pct(cum_r - 1)}"
              f"{pct(cum_s - 1)}{10000 * (cum_r - cum_m):>+9.0f}")
        rows.append({"broker": broker, "cumulative": True,
                     "model": cum_m - 1, "realized": cum_r - 1,
                     "spy": cum_s - 1})

    decomp = execution_decomposition(events, prices)
    if decomp:
        def _bp(x):
            return f"{x:+.0f}" if x is not None else "n/a"
        print("\n=== execution quality: transaction cost vs model close "
              "(+bp = transacted worse) ===")
        print(f"{'broker':<14}{'date':<12}{'fills':>6}{'traded$':>12}"
              f"{'cost bp/trade':>14}{'bp/book':>9}{'timing':>8}{'slip':>7}")
        for r in decomp:
            print(f"{r['broker']:<14}{r['date']:<12}{r['n_fills']:>6}"
                  f"{r['traded_notional']:>12,.0f}"
                  f"{_bp(r['exec_cost_bp_traded']):>14}{_bp(r['exec_cost_bp_book']):>9}"
                  f"{_bp(r['timing_bp_traded']):>8}{_bp(r['slippage_bp_traded']):>7}")
        print("  timing = arrival vs model close; slip = fill vs arrival "
              "(n/a where arrival not instrumented, e.g. eToro / pre-instrumentation logs)")
        rows.append({"execution_decomposition": decomp})
    else:
        print("\n[exec] no filled orders carry price instrumentation yet — "
              "arrival/fill logging accrues from the next live rebalance.")

    track_dir.mkdir(parents=True, exist_ok=True)
    out = track_dir / "tracking_report.json"
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n[saved] {out}")
    inferred = [f for f in ledger if f.inferred]
    if inferred:
        print(f"[note] {len(inferred)} cash flows are auto-inferred \u2014 "
              f"verify them in {track_dir / 'cash_flows.json'}")
