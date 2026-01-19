"""Reporting utilities for monotonicity and stability summaries."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import scipy.stats as stats


def monotonicity_score(levels: Sequence[float], values: Sequence[float]) -> float:
    """Spearman correlation between perturbation level and metric values."""

    corr, _ = stats.spearmanr(levels, values)
    return float(corr)


def stability_summary(values: Sequence[float]) -> dict:
    """Return standard deviation and IQR for a set of values."""

    values = np.asarray(values)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return {
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "iqr": float(q3 - q1),
    }
