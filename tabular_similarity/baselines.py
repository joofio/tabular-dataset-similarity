"""Baseline similarity metrics for tabular datasets."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .preprocessing import fit_transform_df, transform_df


def propensity_score_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    categorical_cols: Sequence[str],
    numeric_cols: Sequence[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, float]:
    """Two-sample test via propensity model; returns AUROC and Brier score."""

    combined = pd.concat([real_df, synth_df], axis=0)
    labels = np.concatenate(
        [np.zeros(len(real_df), dtype=int), np.ones(len(synth_df), dtype=int)]
    )

    train_idx, test_idx = train_test_split(
        np.arange(len(combined)),
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    X_train = combined.iloc[train_idx]
    X_test = combined.iloc[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    X_train_p, pre = fit_transform_df(X_train, categorical_cols, numeric_cols)
    X_test_p = transform_df(X_test, pre)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_p, y_train)
    proba = model.predict_proba(X_test_p)[:, 1]

    return {
        "propensity_auroc": float(roc_auc_score(y_test, proba)),
        "propensity_brier": float(brier_score_loss(y_test, proba)),
    }


def _histogram(data: np.ndarray, bins: int = 20) -> np.ndarray:
    hist, _ = np.histogram(data, bins=bins, density=True)
    hist = hist + 1e-12
    return hist / hist.sum()


def marginal_js_distance(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    categorical_cols: Sequence[str],
    numeric_cols: Sequence[str],
) -> float:
    """Average Jensen-Shannon distance across columns."""

    distances = []
    for col in categorical_cols:
        real_counts = real_df[col].astype(str).value_counts(normalize=True)
        synth_counts = synth_df[col].astype(str).value_counts(normalize=True)
        all_idx = real_counts.index.union(synth_counts.index)
        p = real_counts.reindex(all_idx, fill_value=0.0).values
        q = synth_counts.reindex(all_idx, fill_value=0.0).values
        distances.append(float(jensenshannon(p, q)))

    for col in numeric_cols:
        p = _histogram(real_df[col].dropna().values)
        q = _histogram(synth_df[col].dropna().values)
        distances.append(float(jensenshannon(p, q)))

    return float(np.mean(distances)) if distances else 0.0


def marginal_wasserstein_distance(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numeric_cols: Sequence[str],
) -> float:
    """Average Wasserstein distance across numeric columns."""

    distances = []
    for col in numeric_cols:
        distances.append(
            float(
                stats.wasserstein_distance(
                    real_df[col].dropna().values,
                    synth_df[col].dropna().values,
                )
            )
        )
    return float(np.mean(distances)) if distances else 0.0


def correlation_matrix_distance(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numeric_cols: Sequence[str],
) -> float:
    """Frobenius distance between numeric correlation matrices."""

    if len(numeric_cols) < 2:
        return 0.0
    real_corr = real_df[numeric_cols].corr().fillna(0.0).values
    synth_corr = synth_df[numeric_cols].corr().fillna(0.0).values
    diff = real_corr - synth_corr
    return float(np.linalg.norm(diff, ord="fro"))


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    """Maximum Mean Discrepancy with RBF kernel."""

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    def _rbf(a, b):
        sq = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return np.exp(-gamma * sq)

    k_xx = _rbf(X, X).mean()
    k_yy = _rbf(Y, Y).mean()
    k_xy = _rbf(X, Y).mean()
    return float(k_xx + k_yy - 2 * k_xy)


def energy_distance_score(X: np.ndarray, Y: np.ndarray) -> float:
    """Energy distance for multivariate samples."""

    return float(stats.energy_distance(X.ravel(), Y.ravel()))
