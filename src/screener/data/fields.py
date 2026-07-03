"""Vendor-neutral screening field schema.

The canonical set of fundamental fields the screener consumes, by *our* column
names — independent of which provider sourced them. Each provider maps its own
raw columns onto these names (FMP via ``_RENAME_MAP`` in ``fmp.py``; Sharadar
computes them from raw SF1 components in ``sharadar.py``). ``pit_server`` serves
exactly ``PIT_FIELDS`` from ``pit_snapshots``.

This module holds no vendor-specific logic, so code that uses one provider
(e.g. Sharadar) needn't import another provider's module just to know the field
list — that decoupling is why it lives here rather than in ``fmp.py``.
"""

# Screening fields obtained by renaming a provider's raw column to our name.
# (Order matches fmp.py's ``_RENAME_MAP`` values; kept in sync by an assert there.)
RENAMED_FIELDS = [
    "market_cap",
    "close",
    "avg_volume_20d",
    "roe",
    "earnings_yield",
    "fcf_yield",
    "ev_to_sales",
    "roa",
    "roic",
    "current_ratio",
    "net_debt_to_ebitda",
]

# Fields injected during enrichment — already use our naming (no rename needed).
# Some are provider-specific in availability (e.g. analyst_target / insider_buy_ratio
# are FMP-only; Sharadar leaves them null) but the *names* are vendor-neutral.
PASSTHROUGH_FIELDS = [
    "beta",
    "gross_margin_current", "gross_margin_prior",
    "op_margin_current", "op_margin_prior",
    "eps_growth_current", "eps_growth_prior",
    "rev_growth_current", "rev_growth_prior",
    "analyst_target", "insider_buy_ratio",
    "sga_to_revenue", "sga_to_revenue_prior",
    "rd_to_revenue", "sbc_to_revenue",
    "income_quality", "capex_to_revenue",
    "intangibles_to_assets", "cash_conversion_cycle",
]

# All PIT-servable screening fields, by our column names.
PIT_FIELDS = RENAMED_FIELDS + PASSTHROUGH_FIELDS
