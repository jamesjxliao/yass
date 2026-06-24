"""Regression tests for the backtest return-windowing fixes (#10 boundary-day,
#12 delisting) — locks the behavior the bug-review workflow verified.

run_backtest is driven with tiny fakes for the pipeline/pit_server and synthetic
prices with a KNOWN boundary-day jump, so the period returns are hand-checkable.
"""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from screener.backtest.runner import run_backtest


class _FakePipeline:
    """Always picks the same tickers, ignoring the screening data."""

    def __init__(self, tickers):
        self._tickers = tickers
        self.top_n = len(tickers)

    def run(self, screening_data, hold_bonus_tickers=None, hold_bonus=0.0):
        return pl.DataFrame({
            "ticker": self._tickers,
            "composite_score": [1.0] * len(self._tickers),
        })


class _FakePITServer:
    def __init__(self, tickers):
        self._tickers = tickers

    def get_universe_as_of(self, index, as_of):
        return list(self._tickers)

    def get_screening_data(self, tickers, as_of):
        return pl.DataFrame({"ticker": list(tickers)})


def _prices(rows):
    """rows: list of (ticker, 'YYYY-MM-DD', close)."""
    return pl.DataFrame({
        "ticker": [r[0] for r in rows],
        "date": [date.fromisoformat(r[1]) for r in rows],
        "close": [float(r[2]) for r in rows],
        "low": [float(r[2]) for r in rows],
    })


def _run(tickers, price_rows):
    return run_backtest(
        pipeline=_FakePipeline(tickers),
        pit_server=_FakePITServer(tickers),
        price_data=_prices(price_rows),
        universe_index="sp500",
        start_date=date(2020, 1, 1),
        end_date=date(2020, 3, 1),
        transaction_cost_bps=0.0,  # isolate the return math from cost
    )


def test_boundary_day_return_is_compounded():
    """Period i's end price chains to the next rebalance's first close, so the
    boundary-day return (Jan31 110 -> Feb3 121) lands in period 0 — not dropped."""
    rows = [
        ("AAA", "2020-01-02", 100),  # P0 start (first close >= Jan 1)
        ("AAA", "2020-01-31", 110),  # last close before Feb 1 (old end_price)
        ("AAA", "2020-02-03", 121),  # first close >= Feb 1 = P0 end == P1 start
        ("AAA", "2020-02-28", 130),  # last close before Mar 1
        ("AAA", "2020-03-02", 132),  # first close >= Mar 1 = P1 end
    ]
    m = _run(["AAA"], rows)
    # P0 = 121/100 - 1 = 0.21 (the OLD code dropped the 110->121 boundary day → 0.10)
    assert m.periodic_returns[0] == pytest.approx(0.21)
    # P1 = 132/121 - 1
    assert m.periodic_returns[1] == pytest.approx(132 / 121 - 1)


def test_delisted_pick_carries_to_last_trade():
    """A pick with no close on/after the next rebalance (delisted mid-period)
    carries to its last in-window trade instead of vanishing (#12)."""
    rows = [
        ("AAA", "2020-01-02", 100),
        ("AAA", "2020-01-31", 110),
        ("AAA", "2020-02-03", 121),  # P1 start
        ("AAA", "2020-02-20", 130),  # last trade — delists, no close >= Mar 1
    ]
    m = _run(["AAA"], rows)
    # P0 still chains to the Feb 3 boundary close
    assert m.periodic_returns[0] == pytest.approx(0.21)
    # P1: no boundary close → carry to last trade (130/121 - 1), not dropped to 0
    assert m.periodic_returns[1] == pytest.approx(130 / 121 - 1)
