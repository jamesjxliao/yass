from __future__ import annotations

import math

import polars as pl
import pytest
from screener.engine.weighting import compute_weights
from screener.trading.broker import compute_rebalance_orders, target_value_for


def _picks(tickers, vols):
    return pl.DataFrame({"ticker": tickers, "realized_vol_20d": vols})


def test_equal_weighting_is_uniform():
    w = compute_weights(_picks(["A", "B", "C", "D"], [0.1, 0.2, 0.3, 0.4]), "equal")
    assert all(abs(v - 0.25) < 1e-12 for v in w.values())
    assert abs(sum(w.values()) - 1.0) < 1e-12


def test_inverse_vol_sums_to_one_and_downweights_high_vol():
    w = compute_weights(_picks(["LOW", "HIGH"], [0.10, 0.40]), "inverse_vol")
    assert abs(sum(w.values()) - 1.0) < 1e-12
    # 1/0.10 vs 1/0.40 -> 10 vs 2.5 -> 0.8 vs 0.2
    assert w["LOW"] == pytest.approx(0.8)
    assert w["HIGH"] == pytest.approx(0.2)
    assert w["LOW"] > w["HIGH"]


def test_inverse_vol_missing_vol_gets_neutral_weight():
    # A name with null/zero vol gets the MEDIAN inverse-vol score of the usable
    # names — neither dominating (1/0) nor starved (a flat 1/N). With usable
    # vols 0.10 and 0.40 -> scores 10 and 2.5 -> median 6.25 for the missing one.
    w = compute_weights(_picks(["A", "B", "MISSING"], [0.10, 0.40, None]), "inverse_vol")
    assert abs(sum(w.values()) - 1.0) < 1e-12
    # raw scores: A=10, B=2.5, MISSING=median(10,2.5)=6.25; total=18.75
    assert w["A"] == pytest.approx(10 / 18.75)
    assert w["MISSING"] == pytest.approx(6.25 / 18.75)
    # the neutral name sits between the two real names, not near zero
    assert w["B"] < w["MISSING"] < w["A"]


def test_inverse_vol_all_missing_vol_is_equal():
    w = compute_weights(_picks(["A", "B", "C"], [None, 0.0, None]), "inverse_vol")
    assert w == pytest.approx({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})


def test_inverse_vol_without_vol_column_is_equal():
    df = pl.DataFrame({"ticker": ["A", "B"]})
    w = compute_weights(df, "inverse_vol")
    assert w == pytest.approx({"A": 0.5, "B": 0.5})


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        compute_weights(_picks(["A"], [0.1]), "made_up")


def test_empty_picks():
    assert compute_weights(pl.DataFrame({"ticker": []}), "inverse_vol") == {}


def test_target_value_for_weighted_and_equal():
    # weighted: equity * weight
    assert target_value_for("A", 10_000.0, {"A": 0.7, "B": 0.3}, 2) == pytest.approx(7_000)
    # ticker missing from weights -> equal fallback equity / n
    assert target_value_for("Z", 10_000.0, {"A": 1.0}, 4) == pytest.approx(2_500)
    # no weights at all -> equal
    assert target_value_for("A", 10_000.0, None, 5) == pytest.approx(2_000)
    # n=0 guard
    assert target_value_for("A", 10_000.0, None, 0) == 0.0


def test_rebalance_orders_use_target_weights():
    # From all cash: target 60/40 split of $10k -> buy $6k / $4k.
    orders = compute_rebalance_orders(
        target_tickers=["A", "B"],
        equity=10_000.0,
        current={},
        target_weights={"A": 0.6, "B": 0.4},
    )
    by_ticker = {o.ticker: o for o in orders}
    assert by_ticker["A"].side == "buy"
    assert math.isclose(by_ticker["A"].notional, 6_000.0, rel_tol=1e-9)
    assert math.isclose(by_ticker["B"].notional, 4_000.0, rel_tol=1e-9)


def test_rebalance_orders_none_weights_is_equal():
    orders = compute_rebalance_orders(
        target_tickers=["A", "B"], equity=10_000.0, current={}, target_weights=None,
    )
    for o in orders:
        assert math.isclose(o.notional, 5_000.0, rel_tol=1e-9)


def test_weighted_tolerance_band_scales_with_weight():
    # B is already at its 40% target ($4k); within tolerance -> no order for B.
    # A is at 0 vs 60% target -> buy.
    orders = compute_rebalance_orders(
        target_tickers=["A", "B"],
        equity=10_000.0,
        current={"B": 4_000.0},
        target_weights={"A": 0.6, "B": 0.4},
    )
    tickers = {o.ticker for o in orders}
    assert "A" in tickers
    assert "B" not in tickers
