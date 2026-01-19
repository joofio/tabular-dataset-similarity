"""Synthetic perturbations for dataset similarity experiments."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd


def apply_mean_variance_drift(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    mean_shift: float = 0.0,
    scale: float = 1.0,
) -> pd.DataFrame:
    """Shift and scale numeric columns."""

    out = df.copy()
    out[numeric_cols] = out[numeric_cols] * scale + mean_shift
    return out


def apply_noise_injection(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    std: float = 0.1,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Add Gaussian noise to numeric columns."""

    rng = np.random.default_rng(random_state)
    out = df.copy()
    noise = rng.normal(0.0, std, size=out[numeric_cols].shape)
    out[numeric_cols] = out[numeric_cols] + noise
    return out


def apply_category_collapse(
    df: pd.DataFrame,
    categorical_cols: Sequence[str],
    min_freq: float = 0.01,
    other_label: str = "__OTHER__",
) -> pd.DataFrame:
    """Collapse rare categories into a single bucket."""

    out = df.copy()
    for col in categorical_cols:
        freq = out[col].value_counts(normalize=True)
        rare = freq[freq < min_freq].index
        out[col] = out[col].where(~out[col].isin(rare), other_label)
    return out


def apply_correlation_mix(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    alpha: float = 0.5,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Attenuate correlations by mixing with a permuted copy."""

    rng = np.random.default_rng(random_state)
    out = df.copy()
    permuted = out[numeric_cols].sample(frac=1.0, random_state=random_state)
    out[numeric_cols] = alpha * out[numeric_cols] + (1.0 - alpha) * permuted.values
    return out


def apply_label_shift(
    df: pd.DataFrame,
    target_col: str,
    desired_proportions: Dict[str, float],
    random_state: int | None = None,
) -> pd.DataFrame:
    """Resample rows to match a new class distribution."""

    rng = np.random.default_rng(random_state)
    frames = []
    for label, proportion in desired_proportions.items():
        subset = df[df[target_col] == label]
        if subset.empty:
            continue
        sample_size = int(len(df) * proportion)
        frames.append(subset.sample(n=sample_size, replace=True, random_state=random_state))
    if not frames:
        return df.copy()
    out = pd.concat(frames).sample(frac=1.0, random_state=random_state)
    return out


def apply_missingness(
    df: pd.DataFrame,
    cols: Sequence[str],
    rate: float = 0.1,
    mechanism: str = "mcar",
    random_state: int | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Apply missingness under MCAR or simple MNAR rules."""

    rng = np.random.default_rng(random_state)
    out = df.copy()

    # Convert columns to float to allow NaN values (NaN is a float type)
    for col in cols:
        if out[col].dtype in (np.int64, np.int32, int):
            out[col] = out[col].astype(float)

    if mechanism == "mcar":
        mask = rng.uniform(size=out[cols].shape) < rate
        out.loc[:, cols] = out[cols].mask(mask)
        return out

    if mechanism == "mnar":
        if threshold is None:
            threshold = 0.8
        for col in cols:
            values = out[col].astype(float)
            cutoff = values.quantile(threshold)
            mask = values >= cutoff
            out.loc[mask, col] = np.nan
        return out

    raise ValueError("mechanism must be 'mcar' or 'mnar'")


def apply_interaction_rewire(
    df: pd.DataFrame,
    cols: Sequence[str],
    random_state: int | None = None,
) -> pd.DataFrame:
    """Break pairwise interactions by shuffling within quantile bins."""

    rng = np.random.default_rng(random_state)
    out = df.copy()
    if len(cols) < 2:
        return out

    base = cols[0]
    bins = pd.qcut(out[base].rank(method="first"), q=4, duplicates="drop")
    for col in cols[1:]:
        shuffled = out[col].copy()
        for level in bins.unique():
            idx = bins[bins == level].index
            shuffled.loc[idx] = rng.permutation(shuffled.loc[idx].values)
        out[col] = shuffled
    return out
