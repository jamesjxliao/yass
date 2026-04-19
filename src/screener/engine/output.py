from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def to_json(df: pl.DataFrame, path: Path) -> None:
    """Write ranked results to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = df.to_dicts()
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)


def to_csv(df: pl.DataFrame, path: Path) -> None:
    """Write ranked results to CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


def to_console(df: pl.DataFrame, top_n: int | None = None) -> str:
    """Format ranked results for console display."""
    display_df = df.head(top_n) if top_n else df

    # Select key columns for display
    display_cols = ["ticker", "composite_score"]
    z_cols = [c for c in display_df.columns if c.startswith("z_")]
    display_cols.extend(z_cols)

    available = [c for c in display_cols if c in display_df.columns]
    return str(display_df.select(available))
