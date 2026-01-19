"""Feature importance utilities with held-out permutation importance."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def fit_and_importance(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scoring: str | None = None,
    n_repeats: int = 10,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> Dict[str, float]:
    """Fit the estimator and return feature importances.

    If the estimator exposes `feature_importances_`, those are used. Otherwise,
    permutation importance is computed on a held-out validation split to avoid
    leakage from the training data.
    """

    estimator.fit(X_train, y_train)
    if hasattr(estimator, "feature_importances_"):
        values = getattr(estimator, "feature_importances_")
        return {name: float(val) for name, val in zip(X_train.columns, values)}

    result = permutation_importance(
        estimator,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    return {name: float(val) for name, val in zip(X_val.columns, result.importances_mean)}
