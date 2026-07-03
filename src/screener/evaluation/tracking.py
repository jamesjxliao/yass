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
from dataclasses import dataclass, field
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
    return LogRecord(
        broker=broker, ts=ts, log_date=log_date, equity=_equity_of(d),
        holdings=_holdings_of(d), executed=_executed_of(d), file=path.name,
        note=str(d.get("note", ""))[:200],
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
