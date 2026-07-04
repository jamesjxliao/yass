"""Regression tests for the backtest return-windowing fixes (#10 boundary-day,
#12 delisting) — locks the behavior the bug-review workflow verified.

run_backtest is driven with tiny fakes for the pipeline/pit_server and synthetic
prices with a KNOWN boundary-day jump, so the period returns are hand-checkable.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest
from screener.backtest.metrics import compute_alpha_beta
from screener.backtest.runner import benchmark_period_returns, run_backtest


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

    def use_price_frame(self, price_data):
        pass  # the fake serves screening data directly; no price-frame lookups

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


# --- benchmark alignment + CAPM alpha/beta ---------------------------------
def test_benchmark_period_returns_uses_rebalance_intervals():
    # +10% each month; rebalance dates are month-starts, so each holding-period
    # return must be exactly 0.10 — measured over [rebal_i, rebal_{i+1}].
    prices = pl.DataFrame({
        "date": [date(2020, m, 1) for m in (1, 2, 3, 4)],
        "close": [100.0, 110.0, 121.0, 133.1],
    })
    r = benchmark_period_returns(prices, date(2020, 1, 1), date(2020, 4, 1), "monthly")
    assert len(r) == 3
    assert all(x == pytest.approx(0.10) for x in r)


def test_benchmark_period_returns_uses_first_close_on_or_after_boundary():
    # When a rebalance date has no close ON that day (weekend/holiday), the
    # benchmark must price it at the FIRST close ON/AFTER — the same convention
    # the strategy uses for its own boundaries — not the last close on/before
    # (which would phase the benchmark 1-3 days early and bias the CAPM/IR fit).
    # No close on 2020-02-01; 105 sits just before it, 110 just after.
    prices = pl.DataFrame({
        "date": [date(2020, 1, 1), date(2020, 1, 29), date(2020, 2, 3),
                 date(2020, 3, 1), date(2020, 4, 1)],
        "close": [100.0, 105.0, 110.0, 120.0, 130.0],
    })
    r = benchmark_period_returns(prices, date(2020, 1, 1), date(2020, 4, 1), "monthly")
    assert len(r) == 3
    # Period 0 boundary at 2020-02-01 -> first close on/after = 110 (not 105).
    assert r[0] == pytest.approx(110.0 / 100.0 - 1)   # 0.10, would be 0.05 under old convention


def test_benchmark_period_returns_empty_frame():
    empty = pl.DataFrame({"date": [], "close": []})
    assert benchmark_period_returns(empty, date(2020, 1, 1), date(2020, 4, 1)) == []


def test_compute_alpha_beta_recovers_known_params():
    rng = np.random.default_rng(0)
    x = rng.normal(0.01, 0.04, 200)
    y = 0.008 + 1.3 * x + rng.normal(0, 0.001, 200)  # known alpha/beta + tiny noise
    ab = compute_alpha_beta(y, x, periods_per_year=12)
    assert ab is not None
    assert ab.beta == pytest.approx(1.3, abs=0.05)
    assert ab.alpha_period == pytest.approx(0.008, abs=0.002)
    assert ab.alpha_annual == pytest.approx(ab.alpha_period * 12)
    assert ab.r_squared > 0.95
    assert ab.t_alpha > 1.98  # alpha solidly positive


def test_compute_alpha_beta_guards():
    assert compute_alpha_beta([0.1, 0.2], [0.1, 0.2]) is None  # < 3 periods
    assert compute_alpha_beta([0.1, 0.2, 0.3], [0.05, 0.05, 0.05]) is None  # no variance
