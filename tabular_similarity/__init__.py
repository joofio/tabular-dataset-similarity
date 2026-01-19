"""Utilities for tabular dataset similarity experiments."""

from .preprocessing import build_preprocessor, fit_transform_df, transform_df
from .splits import generate_repeated_splits, train_test_split_indices
from .importance import fit_and_importance
from .ranking_metrics import compare_rankings, rank_features
from .cross_classification import evaluate_cross_classification
from .baselines import (
    correlation_matrix_distance,
    energy_distance_score,
    marginal_js_distance,
    marginal_wasserstein_distance,
    mmd_rbf,
    propensity_score_metrics,
)
from .aggregation import aggregate_scores
from .stats import bootstrap_ci
from .reporting import monotonicity_score, stability_summary
