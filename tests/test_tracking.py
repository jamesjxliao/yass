"""Tests for the live-vs-backtest tracking core (evaluation/tracking.py)."""
from __future__ import annotations

import json
from datetime import date, datetime

import polars as pl
import pytest
from screener.evaluation.tracking import (
    CashFlow,
    LogRecord,
    build_periods,
    detect_cash_flows,
    execution_decomposition,
    load_history,
    parse_trade_log,
    portfolio_return,
    rebalance_events,
)


def _write(tmp_path, name, payload):
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


def test_parse_standard_alpaca_log(tmp_path):
    p = _write(tmp_path, "20260406_125947.json", {
        "date": "2026-04-06", "mode": "PAPER", "dry_run": False,
        "account": {"equity": 109789.96, "portfolio_value": 109789.96},
        "picks": ["MU", "WDC"], "orders": [
            {"ticker": "ALB", "side": "sell", "notional": 1.0,
             "status": "submitted"}],
    })
    r = parse_trade_log(p)
    assert r.broker == "alpaca_paper"
    assert r.executed is True
    assert r.equity == pytest.approx(109789.96)
    assert r.holdings == ["MU", "WDC"]


def test_parse_handwritten_robinhood_log(tmp_path):
    """Robinhood skill logs: account is a string, equity in account_after,
    holdings under final_holdings/holdings, no normalized orders list."""
    p = _write(tmp_path, "20260622_081859_robinhood.json", {
        "date": "2026-06-22", "broker": "robinhood", "mode": "LIVE",
        "dry_run": False, "account": "581816907 (Agentic, cash)",
        "account_after": {"total_value": 40021.29},
        "holdings": ["SNDK", "INCY"],
    })
    r = parse_trade_log(p)
    assert r.broker == "robinhood"
    assert r.equity == pytest.approx(40021.29)
    assert r.executed is True  # holdings evidence, no orders list
    assert r.holdings == ["INCY", "SNDK"]


def test_dry_run_is_equity_observation_not_event(tmp_path):
    p = _write(tmp_path, "20260405_031431.json", {
        "date": "2026-04-05", "mode": "PAPER", "dry_run": True,
        "account": {"equity": 108329.61}, "picks": ["MU"],
        "orders": [{"ticker": "X", "side": "buy", "notional": 1.0,
                    "status": "dry_run"}],
    })
    r = parse_trade_log(p)
    assert r.executed is False
    assert r.equity == pytest.approx(108329.61)


def test_rebalance_events_same_day_supersede(tmp_path):
    for name, picks in [("20260601_100000_etoro.json", ["A", "B"]),
                        ("20260601_110000_etoro.json", ["A", "C"])]:
        _write(tmp_path, name, {
            "date": "2026-06-01", "broker": "etoro", "mode": "REAL",
            "dry_run": False, "account": {"equity": 100.0}, "picks": picks,
            "orders": [{"ticker": "A", "side": "buy", "notional": 1.0,
                        "status": "submitted"}],
        })
    events = rebalance_events(load_history(tmp_path))
    assert len(events["etoro"]) == 1
    assert events["etoro"][0].holdings == ["A", "C"]  # later log wins


def _prices():
    rows = []
    for t, closes in {"AAA": [10.0, 11.0, 12.0, 13.0],
                      "BBB": [20.0, 20.0, 20.0, 30.0],
                      "SPY": [100.0, 101.0, 102.0, 103.0]}.items():
        for i, c in enumerate(closes):
            rows.append({"ticker": t, "date": date(2026, 6, 1 + 7 * i),
                         "close": c})
    return pl.DataFrame(rows)


def test_portfolio_return_boundary_convention():
    # [jun1, jun15): start = first close >= jun1, end = first close >= jun15
    r, flags = portfolio_return(["AAA", "BBB"], date(2026, 6, 1),
                                date(2026, 6, 15), _prices())
    assert r == pytest.approx(((12.0 / 10.0 - 1) + (20.0 / 20.0 - 1)) / 2)
    assert flags == []


def test_portfolio_return_missing_ticker_flagged():
    r, flags = portfolio_return(["AAA", "ZZZ"], date(2026, 6, 1),
                                date(2026, 6, 15), _prices())
    assert r == pytest.approx(12.0 / 10.0 - 1)
    assert any(f.startswith("no-price:ZZZ") for f in flags)
    assert any(f.startswith("partial:") for f in flags)


def test_build_periods_deposit_adjusted(tmp_path):
    for name, eq in [("20260601_100000_etoro.json", 1000.0),
                     ("20260615_100000_etoro.json", 2100.0)]:
        _write(tmp_path, name, {
            "date": name[:4] + "-" + name[4:6] + "-" + name[6:8],
            "broker": "etoro", "mode": "REAL", "dry_run": False,
            "account": {"equity": eq}, "picks": ["AAA", "BBB"],
            "orders": [{"ticker": "AAA", "side": "buy", "notional": 1.0,
                        "status": "submitted"}],
        })
    events = rebalance_events(load_history(tmp_path))
    ledger = [CashFlow(broker="etoro", on=date(2026, 6, 10), amount=1000.0)]
    periods = build_periods(events, _prices(), ledger)
    assert len(periods) == 1
    p = periods[0]
    # realized: 2100 / (1000 + 1000 deposit) - 1 = +5%
    assert p.realized_ret == pytest.approx(0.05)
    # model: AAA jun1->jun15 = 12/10-1 = 20%; BBB = 20/20-1 = 0% -> mean 10%
    assert p.model_ret == pytest.approx(0.10)
    assert p.flows == pytest.approx(1000.0)


def test_detect_cash_flows_same_day_jump(tmp_path):
    for name, eq in [("20260601_100000_etoro.json", 20128.79),
                     ("20260601_110000_etoro.json", 25128.79)]:
        _write(tmp_path, name, {
            "date": "2026-06-01", "broker": "etoro", "mode": "REAL",
            "dry_run": True, "account": {"equity": eq}, "picks": ["A"],
        })
    flows = detect_cash_flows(load_history(tmp_path))
    assert len(flows) == 1
    assert flows[0].amount == pytest.approx(5000.0)
    assert flows[0].inferred is True


def _rec_with_orders(orders, *, equity=10_000.0, ts=None):
    ts = ts or datetime(2026, 6, 1, 9, 35)
    return LogRecord(
        broker="alpaca_paper", ts=ts, log_date=ts.date(), equity=equity,
        holdings=[o["ticker"] for o in orders], executed=True,
        file="x.json", orders=orders,
    )


def _close_frame(rows):
    return pl.DataFrame(
        {"ticker": [t for t, _ in rows], "date": [date(2026, 6, 1)] * len(rows),
         "close": [c for _, c in rows]})


def test_execution_decomposition_splits_timing_and_slippage():
    # model close 100; arrived at 100.5 (timing +50bp); filled at 101 (slip ~+50bp);
    # total fill-vs-close = +100bp of the $1,000 trade => +10bp of the $10k book.
    rec = _rec_with_orders([
        {"ticker": "AAPL", "side": "buy", "notional": 1000.0,
         "arrival_price": 100.5, "fill_price": 101.0}])
    [row] = execution_decomposition({"alpaca_paper": [rec]}, _close_frame([("AAPL", 100.0)]))
    assert row["n_fills"] == 1
    assert row["exec_cost_bp_traded"] == pytest.approx(100.0, abs=0.5)
    assert row["exec_cost_bp_book"] == pytest.approx(10.0, abs=0.1)
    assert row["timing_bp_traded"] == pytest.approx(50.0, abs=0.5)
    assert row["slippage_bp_traded"] == pytest.approx(49.75, abs=0.5)


def test_execution_decomposition_sell_and_no_arrival():
    # eToro-style: fill only, no arrival. A sell filled BELOW the model close is a
    # cost (received less). timing/slippage stay None without arrival.
    rec = _rec_with_orders([
        {"ticker": "MSFT", "side": "sell", "notional": 500.0, "fill_price": 99.0}])
    [row] = execution_decomposition({"alpaca_paper": [rec]}, _close_frame([("MSFT", 100.0)]))
    assert row["exec_cost_bp_traded"] == pytest.approx(100.0, abs=0.5)  # (100-99)/100
    assert row["timing_bp_traded"] is None
    assert row["slippage_bp_traded"] is None


def test_execution_decomposition_ignores_uninstrumented_orders():
    # pre-instrumentation logs (no fill_price) contribute nothing.
    rec = _rec_with_orders([
        {"ticker": "AAPL", "side": "buy", "notional": 1000.0, "status": "submitted"}])
    assert execution_decomposition({"alpaca_paper": [rec]}, _close_frame([("AAPL", 100.0)])) == []


def test_execution_decomposition_mixed_arrival_scales_by_arrival_notional():
    # Two $1,000 buys, only the first arrival-instrumented (the second lost its
    # quote — e.g. Alpaca _get_last_price returned None). The timing/slippage
    # split must be scaled by the ARRIVAL notional ($1,000), NOT the full traded
    # notional ($2,000) — dividing by the full notional halves (dilutes) the split
    # and it no longer reconciles with the total exec cost.
    rec = _rec_with_orders([
        {"ticker": "AAPL", "side": "buy", "notional": 1000.0,
         "arrival_price": 100.5, "fill_price": 101.0},
        {"ticker": "MSFT", "side": "buy", "notional": 1000.0, "fill_price": 101.0},
    ])
    [row] = execution_decomposition(
        {"alpaca_paper": [rec]}, _close_frame([("AAPL", 100.0), ("MSFT", 100.0)]))
    assert row["n_fills"] == 2
    assert row["exec_cost_bp_traded"] == pytest.approx(100.0, abs=0.5)  # both fills, /2000
    # arrival-only notional in the denominator -> not diluted by the MSFT fill
    assert row["timing_bp_traded"] == pytest.approx(50.0, abs=0.5)
    assert row["slippage_bp_traded"] == pytest.approx(49.75, abs=0.5)
