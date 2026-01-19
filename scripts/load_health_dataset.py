"""Dataset loader for health-like tabular data."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def load_health_dataset(
    path: str,
    categorical_cols: Sequence[str],
    numeric_cols: Sequence[str],
) -> pd.DataFrame:
    """Load a dataset with explicit column typing."""

    df = pd.read_csv(path)
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
