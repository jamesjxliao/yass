from __future__ import annotations

import polars as pl

from signals._normalize import minmax


class EfficiencyAccelerationSignal:
    """Efficiency acceleration: companies becoming more productive over time.

    Captures the structural shift where companies produce more output per
    unit of input — historically driven by automation, offshoring, and
    process improvement; accelerated in the AI era.

    Three components:
    1. Revenue growth acceleration (rev_growth_current vs rev_growth_prior)
       — is top-line growth speeding up?
    2. Operating margin expansion while growing — converting more revenue
       to profit (operating leverage kicking in).
    3. EPS growth acceleration (eps_growth_current vs eps_growth_prior)
       — earnings improving faster than before.

    The combination identifies companies whose efficiency is *improving*,
    not just companies that are already efficient (which quality captures).
    """

    name = "efficiency_acceleration"
    description = "Revenue + EPS growth acceleration with margin expansion"
    higher_is_better = True

    def compute(self, df: pl.DataFrame) -> pl.Series:
        components = []

        # Component 1: Revenue growth acceleration
        # Positive delta = growth is speeding up
        if "rev_growth_current" in df.columns and "rev_growth_prior" in df.columns:
            curr = df["rev_growth_current"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            prior = df["rev_growth_prior"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            accel = (curr - prior).clip(-0.5, 0.5)
            components.append(minmax(accel))

        # Component 2: Operating margin expansion (same as margin_expansion
        # but only operating margin — we want operating leverage specifically)
        if "op_margin_current" in df.columns and "op_margin_prior" in df.columns:
            curr = df["op_margin_current"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            prior = df["op_margin_prior"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            expansion = (curr - prior).clip(-0.2, 0.2)
            components.append(minmax(expansion))

        # Component 3: EPS growth acceleration
        # Positive delta = earnings growth speeding up
        if "eps_growth_current" in df.columns and "eps_growth_prior" in df.columns:
            curr = df["eps_growth_current"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            prior = df["eps_growth_prior"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            accel = (curr - prior).clip(-1.0, 1.0)
            components.append(minmax(accel))

        # Component 4: SGA leverage (declining SGA-to-revenue = operating leverage)
        if "sga_to_revenue" in df.columns and "sga_to_revenue_prior" in df.columns:
            curr = df["sga_to_revenue"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            prior = df["sga_to_revenue_prior"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
            # prior - curr: positive = SGA ratio declining = good
            sga_delta = (prior - curr).clip(-0.15, 0.15)
            components.append(minmax(sga_delta))

        if not components:
            return pl.Series([0.5] * len(df))

        stacked = pl.DataFrame({f"c{i}": c for i, c in enumerate(components)})
        return stacked.mean_horizontal()
