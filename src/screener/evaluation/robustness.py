"""Trustworthiness statistics: confidence intervals and overfitting deflation.

Pure, dependency-light functions (the I/O-free half of the trustworthiness /
robustness audit) so the math is unit-tested rather than buried in a runner
script. The Probabilistic Sharpe Ratio itself lives in
``backtest.metrics`` (single source of truth); this module builds the Deflated
Sharpe Ratio and the block-bootstrap on top of it.

References: Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014);
Politis & Romano, "The Stationary Bootstrap" (1994).
"""
from __future__ import annotations

import math

import numpy as np

from screener.backtest.metrics import compute_psr

EULER_GAMMA = 0.5772156649015329


def inv_norm_cdf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation, err < 1.2e-9)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def sample_skew_kurt(rets: np.ndarray) -> tuple[float, float]:
    """Population skewness and full (non-excess, normal==3) kurtosis of ``rets``."""
    r = np.asarray(rets, dtype=float)
    sd = r.std()
    if sd == 0 or len(r) == 0:
        return 0.0, 3.0
    z = (r - r.mean()) / sd
    return float((z**3).mean()), float((z**4).mean())


def ann_to_period_sharpe(sr_ann: float, ppy: int = 12) -> float:
    """De-annualize a Sharpe ratio to per-period units (PSR/DSR expect per-period)."""
    return sr_ann / math.sqrt(ppy)


def expected_max_sharpe(n_trials: int, sigma_sr_period: float) -> float:
    """E[max Sharpe] across ``n_trials`` independent skill-less trials (per-period).

    The Deflated-Sharpe benchmark SR0: the best Sharpe you'd expect from
    ``n_trials`` strategies whose true Sharpe is zero and whose estimates have
    per-period dispersion ``sigma_sr_period``. Grows with both arguments.
    """
    if n_trials < 2 or sigma_sr_period <= 0:
        return 0.0
    return sigma_sr_period * (
        (1 - EULER_GAMMA) * inv_norm_cdf(1 - 1.0 / n_trials)
        + EULER_GAMMA * inv_norm_cdf(1 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe_ratio(
    sharpe_period: float,
    n: int,
    skew: float,
    kurt: float,
    n_trials: int,
    sigma_sr_period: float,
) -> tuple[float, float]:
    """Deflated Sharpe Ratio = P(true Sharpe > best-of-N-skill-less-trials).

    All Sharpe inputs are per-period. Returns ``(dsr, benchmark_sr0)`` where
    ``benchmark_sr0`` is the multiple-testing hurdle the strategy had to clear.
    """
    sr0 = expected_max_sharpe(n_trials, sigma_sr_period)
    dsr = compute_psr(sharpe_period, n, skew, kurt, benchmark=sr0)
    return dsr, sr0


def _max_drawdown(rets: np.ndarray) -> float:
    # Prepend the starting equity (1.0) so a drawdown in the opening period(s) is
    # captured; np.cumprod starts at 1+rets[0], hiding any decline before the
    # first new equity high (biases the bootstrap MaxDD CI optimistic).
    cum = np.concatenate([[1.0], np.cumprod(1 + rets)])
    peak = np.maximum.accumulate(cum)
    return float(np.min((cum - peak) / peak))


def _cagr(rets: np.ndarray, ppy: int = 12) -> float:
    total = float(np.prod(1 + rets) - 1)
    n_years = len(rets) / ppy
    if n_years <= 0 or total <= -1:
        return 0.0
    return (1 + total) ** (1 / n_years) - 1


def stationary_bootstrap_ci(
    rets: np.ndarray,
    n_reps: int = 5000,
    mean_block: float = 3.0,
    ppy: int = 12,
    seed: int = 42,
    lo: float = 5.0,
    hi: float = 95.0,
) -> dict[str, tuple[float, float, float]]:
    """Stationary (Politis-Romano) block bootstrap CI on Sharpe / CAGR / MaxDD.

    Geometric block lengths (mean ``mean_block``) preserve the serial dependence
    an i.i.d. resample would destroy. Returns {metric: (p_lo, p50, p_hi)}.
    """
    rets = np.asarray(rets, dtype=float)
    n = len(rets)
    if n < 2:
        z = (0.0, 0.0, 0.0)
        return {"sharpe": z, "cagr": z, "maxdd": z}
    rng = np.random.default_rng(seed)
    p_continue = 1.0 - 1.0 / mean_block
    sharpes, cagrs, dds = [], [], []
    for _ in range(n_reps):
        idx = np.empty(n, dtype=int)
        i = int(rng.integers(0, n))
        for t in range(n):
            idx[t] = i
            if rng.random() < p_continue:
                i = (i + 1) % n
            else:
                i = int(rng.integers(0, n))
        sample = rets[idx]
        sd = sample.std()
        sharpes.append(sample.mean() / sd * math.sqrt(ppy) if sd > 0 else 0.0)
        cagrs.append(_cagr(sample, ppy))
        dds.append(_max_drawdown(sample))
    out = {}
    for name, arr in [("sharpe", sharpes), ("cagr", cagrs), ("maxdd", dds)]:
        a = np.array(arr)
        out[name] = (float(np.percentile(a, lo)), float(np.percentile(a, 50)),
                     float(np.percentile(a, hi)))
    return out
