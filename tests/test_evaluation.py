from __future__ import annotations

from datetime import date

import polars as pl
from screener.evaluation.report import (
    MonteCarloResult,
    RegimeResult,
    _ols_with_intercept,
    _solve_linear,
    run_regime_analysis,
)


def test_ols_known_values():
    """OLS on y = 2 + 3*x should recover alpha=2, beta=3."""
    y = [2 + 3 * x for x in range(10)]
    X = [[float(x)] for x in range(10)]
    alpha, betas, r_sq = _ols_with_intercept(y, X)
    assert abs(alpha - 2.0) < 1e-6
    assert abs(betas[0] - 3.0) < 1e-6
    assert abs(r_sq - 1.0) < 1e-6


def test_ols_multiple_regressors():
    """OLS with 2 independent regressors recovers correct betas."""
    x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    x2 = [5.0, 3.0, 8.0, 1.0, 7.0, 2.0, 6.0, 4.0]
    # y = 2*x1 + 3*x2 (no intercept to avoid near-singularity)
    y = [2 * a + 3 * b for a, b in zip(x1, x2)]
    X = [[a, b] for a, b in zip(x1, x2)]
    alpha, betas, r_sq = _ols_with_intercept(y, X)
    assert abs(betas[0] - 2.0) < 0.1
    assert abs(betas[1] - 3.0) < 0.1
    assert r_sq > 0.99


def test_solve_linear():
    """Solve 2x2 system: x + y = 3, 2x + y = 5 -> x=2, y=1."""
    A = [[1.0, 1.0], [2.0, 1.0]]
    b = [3.0, 5.0]
    x = _solve_linear(A, b)
    assert abs(x[0] - 2.0) < 1e-10
    assert abs(x[1] - 1.0) < 1e-10


def test_monte_carlo_result_summary():
    mc = MonteCarloResult(
        strategy_sharpe=1.5,
        random_sharpes=[0.1, 0.2, 0.3, 0.4, 0.5],
        percentile=100.0,
    )
    summary = mc.summary()
    assert "SIGNIFICANT" in summary
    assert "100.0%" in summary


def test_monte_carlo_not_significant():
    mc = MonteCarloResult(
        strategy_sharpe=0.3,
        random_sharpes=[0.1, 0.2, 0.3, 0.4, 0.5],
        percentile=50.0,
    )
    summary = mc.summary()
    assert "NOT SIGNIFICANT" in summary


def test_regime_analysis():
    """Test regime classification with known returns."""
    # Create price data with clear bull/bear months
    rows = []
    months = [
        (date(2024, 1, 2), date(2024, 1, 31), 100, 105),  # bull +5%
        (date(2024, 2, 1), date(2024, 2, 29), 105, 100),  # bear -4.8%
        (date(2024, 3, 1), date(2024, 3, 29), 100, 101),  # flat +1%
    ]
    for start, end, p1, p2 in months:
        rows.append({"ticker": "TEST", "date": start, "close": float(p1),
                      "open": float(p1), "high": float(p1+1), "low": float(p1-1),
                      "volume": 1e6})
        rows.append({"ticker": "TEST", "date": end, "close": float(p2),
                      "open": float(p2), "high": float(p2+1), "low": float(p2-1),
                      "volume": 1e6})

    price_data = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
    strategy_returns = [0.03, -0.05, 0.01]

    result = run_regime_analysis(
        strategy_returns, price_data,
        start_date=date(2024, 1, 1), end_date=date(2024, 4, 1),
    )

    assert isinstance(result, RegimeResult)
    assert len(result.regimes) > 0


def test_regime_summary_format():
    result = RegimeResult(regimes={
        "Bull": {"n_months": 10, "avg_return": 0.05, "sharpe": 2.0},
        "Bear": {"n_months": 5, "avg_return": -0.04, "sharpe": -1.5},
    })
    summary = result.summary()
    assert "Bull" in summary
    assert "Bear" in summary
