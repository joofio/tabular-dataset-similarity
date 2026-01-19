"""Statistical utilities for uncertainty estimation and comparisons."""

from __future__ import annotations

from typing import Callable, Iterable, Tuple

import numpy as np


def bootstrap_ci(
    values: Iterable[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for a statistic.

    Returns (lower, upper, estimate).
    """

    rng = np.random.default_rng(random_state)
    values = np.asarray(list(values))
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")

    boot = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot.append(statistic(sample))
    boot = np.sort(boot)

    lower = np.quantile(boot, alpha / 2)
    upper = np.quantile(boot, 1 - alpha / 2)
    estimate = statistic(values)
    return float(lower), float(upper), float(estimate)


def mixed_effects_model(df, formula: str, group_col: str):
    """Fit a mixed-effects model using statsmodels if available."""

    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("statsmodels is required for mixed effects models") from exc

    model = smf.mixedlm(formula, df, groups=df[group_col])
    return model.fit()
