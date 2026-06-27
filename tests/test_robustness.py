"""Unit tests for the trustworthiness statistics core (evaluation.robustness)."""
from __future__ import annotations

import math

import numpy as np
from screener.backtest.metrics import compute_psr
from screener.evaluation.robustness import (
    ann_to_period_sharpe,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    inv_norm_cdf,
    sample_skew_kurt,
    stationary_bootstrap_ci,
)


# --- inverse normal CDF ----------------------------------------------------
def test_inv_norm_cdf_known_quantiles():
    assert inv_norm_cdf(0.5) == 0.0 or abs(inv_norm_cdf(0.5)) < 1e-6
    assert math.isclose(inv_norm_cdf(0.975), 1.959964, abs_tol=1e-4)
    assert math.isclose(inv_norm_cdf(0.025), -1.959964, abs_tol=1e-4)
    assert math.isclose(inv_norm_cdf(0.84134), 1.0, abs_tol=1e-3)


def test_inv_norm_cdf_is_inverse_of_cdf():
    # Round-trip Phi(Phi^-1(p)) == p
    for p in (0.01, 0.2, 0.5, 0.8, 0.99):
        x = inv_norm_cdf(p)
        cdf = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        assert math.isclose(cdf, p, abs_tol=1e-4)


def test_inv_norm_cdf_boundaries():
    assert inv_norm_cdf(0.0) == -math.inf
    assert inv_norm_cdf(1.0) == math.inf


# --- skew / kurtosis -------------------------------------------------------
def test_sample_skew_kurt_symmetric():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200_000)
    skew, kurt = sample_skew_kurt(x)
    assert abs(skew) < 0.05
    assert abs(kurt - 3.0) < 0.1  # normal kurtosis ~ 3


def test_sample_skew_kurt_constant():
    skew, kurt = sample_skew_kurt(np.ones(10))
    assert skew == 0.0 and kurt == 3.0


# --- compute_psr benchmark parameter (the reconciliation) ------------------
def test_psr_benchmark_lowers_probability():
    # A positive benchmark must reduce P(true Sharpe > benchmark).
    base = compute_psr(0.33, 108, 0.0, 3.0, benchmark=0.0)
    deflated = compute_psr(0.33, 108, 0.0, 3.0, benchmark=0.1)
    assert deflated < base
    assert 0.0 <= deflated <= base <= 1.0


def test_psr_benchmark_default_is_zero():
    # Default benchmark keeps the legacy P(Sharpe > 0) behavior.
    assert compute_psr(0.33, 108, 0.0, 3.0) == compute_psr(0.33, 108, 0.0, 3.0, benchmark=0.0)


def test_psr_per_period_not_saturated():
    # The whole point of the fix: a per-period Sharpe gives an informative PSR,
    # not a saturated 1.0. (An annualized 1.146 would round to 1.0.)
    sr_p = ann_to_period_sharpe(1.146)
    psr = compute_psr(sr_p, 108, -0.26, 2.56)
    assert 0.95 < psr < 1.0


# --- expected max Sharpe (DSR benchmark) -----------------------------------
def test_expected_max_sharpe_monotonic_in_trials():
    sig = ann_to_period_sharpe(0.13)
    vals = [expected_max_sharpe(t, sig) for t in (2, 10, 50, 120, 250)]
    assert all(b > a for a, b in zip(vals, vals[1:]))  # strictly increasing


def test_expected_max_sharpe_monotonic_in_dispersion():
    a = expected_max_sharpe(120, ann_to_period_sharpe(0.10))
    b = expected_max_sharpe(120, ann_to_period_sharpe(0.16))
    assert b > a


def test_expected_max_sharpe_degenerate():
    assert expected_max_sharpe(1, 0.04) == 0.0
    assert expected_max_sharpe(120, 0.0) == 0.0


# --- deflated Sharpe ratio -------------------------------------------------
def test_dsr_below_undeflated_psr():
    sr_p = ann_to_period_sharpe(1.146)
    skew, kurt = -0.26, 2.56
    psr0 = compute_psr(sr_p, 108, skew, kurt, benchmark=0.0)
    dsr, sr0 = deflated_sharpe_ratio(sr_p, 108, skew, kurt, 120, ann_to_period_sharpe(0.13))
    assert sr0 > 0
    assert dsr < psr0  # deflation always lowers the probability


def test_dsr_decreases_with_more_trials():
    sr_p = ann_to_period_sharpe(1.146)
    sig = ann_to_period_sharpe(0.13)
    dsr_50, _ = deflated_sharpe_ratio(sr_p, 108, -0.26, 2.56, 50, sig)
    dsr_250, _ = deflated_sharpe_ratio(sr_p, 108, -0.26, 2.56, 250, sig)
    assert dsr_250 < dsr_50  # more trials -> higher hurdle -> lower DSR


# --- stationary bootstrap --------------------------------------------------
def test_bootstrap_ci_ordered_and_brackets_point():
    rng = np.random.default_rng(1)
    rets = rng.normal(0.012, 0.04, 108)  # ~ a real monthly equity book
    point = rets.mean() / rets.std() * math.sqrt(12)
    ci = stationary_bootstrap_ci(rets, n_reps=2000, seed=7)
    lo, mid, hi = ci["sharpe"]
    assert lo < mid < hi
    assert lo < point < hi  # the point estimate sits inside its own CI
    # CAGR and MaxDD bands are also ordered
    assert ci["cagr"][0] < ci["cagr"][2]
    assert ci["maxdd"][0] < ci["maxdd"][2]


def test_bootstrap_deterministic_with_seed():
    rets = np.random.default_rng(2).normal(0.01, 0.03, 60)
    a = stationary_bootstrap_ci(rets, n_reps=500, seed=11)
    b = stationary_bootstrap_ci(rets, n_reps=500, seed=11)
    assert a["sharpe"] == b["sharpe"]


def test_bootstrap_tiny_sample_safe():
    ci = stationary_bootstrap_ci(np.array([0.01]), n_reps=10)
    assert ci["sharpe"] == (0.0, 0.0, 0.0)
