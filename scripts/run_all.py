"""Entry point to run the full experiment pipeline."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from configs.experiment import EXPERIMENT
from configs.perturbations import PERTURBATIONS
from tabular_similarity import baselines, perturbations
from tabular_similarity.preprocessing import fit_transform_df, transform_df
from tabular_similarity.ranking_scores import compute_ranking_scores, get_feature_importance
from tabular_similarity.aggregation import aggregate_scores
from tabular_similarity.stats import bootstrap_ci
from scripts.log_metadata import write_metadata


# Model mapping
MODELS = {
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
}


def _classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Compute classification metrics."""
    # Compute metrics - warnings will be shown if y_pred contains classes not in y_true
    # This is informative: it means train/test have different class distributions
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    metrics["auroc"] = None
    if y_proba is not None:
        # Need at least 2 classes in y_true for AUROC
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            return metrics

        try:
            if y_proba.ndim == 2 and y_proba.shape[1] > 2:
                metrics["auroc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
            elif y_proba.ndim == 2 and y_proba.shape[1] == 2:
                metrics["auroc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            elif y_proba.ndim == 1:
                metrics["auroc"] = float(roc_auc_score(y_true, y_proba))
        except (ValueError, IndexError):
            pass
    return metrics


def _regression_metrics(y_true, y_pred) -> dict:
    """Compute regression metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _compute_baseline_metrics(real_df, synth_df, cat_cols, num_cols) -> dict:
    """Compute all baseline similarity metrics."""
    metrics = {}

    # Propensity score metrics
    try:
        propensity = baselines.propensity_score_metrics(real_df, synth_df, cat_cols, num_cols)
        metrics["propensity_auroc"] = propensity["propensity_auroc"]
        metrics["propensity_brier"] = propensity["propensity_brier"]
    except Exception:
        metrics["propensity_auroc"] = None
        metrics["propensity_brier"] = None

    # Marginal JS distance
    try:
        metrics["marginal_js_distance"] = baselines.marginal_js_distance(
            real_df, synth_df, cat_cols, num_cols
        )
    except Exception:
        metrics["marginal_js_distance"] = None

    # Marginal Wasserstein distance (numeric only)
    if num_cols:
        try:
            metrics["marginal_wasserstein"] = baselines.marginal_wasserstein_distance(
                real_df, synth_df, num_cols
            )
        except Exception:
            metrics["marginal_wasserstein"] = None

    # Correlation matrix distance (numeric only, needs 2+ columns)
    if len(num_cols) >= 2:
        try:
            metrics["correlation_distance"] = baselines.correlation_matrix_distance(
                real_df, synth_df, num_cols
            )
        except Exception:
            metrics["correlation_distance"] = None

    # MMD and Energy distance (on preprocessed numeric data)
    if num_cols:
        try:
            real_num = real_df[num_cols].dropna().values
            synth_num = synth_df[num_cols].dropna().values
            if len(real_num) > 0 and len(synth_num) > 0:
                metrics["mmd_rbf"] = baselines.mmd_rbf(real_num, synth_num)
                metrics["energy_distance"] = baselines.energy_distance_score(real_num, synth_num)
        except Exception:
            metrics["mmd_rbf"] = None
            metrics["energy_distance"] = None

    return metrics


def run_all(output_dir: str = "results") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    seed = EXPERIMENT["random_seed"]
    test_size = EXPERIMENT["test_size"]
    reps = EXPERIMENT["repeats"]
    classification_models = EXPERIMENT["models"]["classification"]
    regression_models = EXPERIMENT["models"]["regression"]

    for dataset in EXPERIMENT["datasets"]:
        df = pd.read_csv(dataset["path"])
        cat_cols = dataset["categorical_cols"]
        num_cols = dataset["numeric_cols"]
        target_cols = dataset["target_cols"]
        dataset_name = dataset["name"]

        print(f"Processing {dataset_name}...")

        all_results = []

        for spec in PERTURBATIONS:
            perturbed = _apply_perturbation(df, spec, cat_cols, num_cols)
            perturbation_name = spec["name"]
            print(f"  Perturbation: {perturbation_name}")

            # Compute all baseline metrics
            baseline_metrics = _compute_baseline_metrics(df, perturbed, cat_cols, num_cols)

            # Full model-based metrics for each target column
            target_results = _run_target_experiments(
                df,
                perturbed,
                cat_cols,
                num_cols,
                target_cols,
                classification_models,
                regression_models,
                reps,
                seed,
                test_size,
            )

            # Aggregate results across targets
            aggregated = _aggregate_target_results(target_results)

            all_results.append({
                "perturbation": perturbation_name,
                "dataset": dataset_name,
                "baseline_metrics": baseline_metrics,
                "target_results": target_results,
                "aggregated": aggregated,
            })

        Path(output_dir, f"full_metrics_{dataset_name}.json").write_text(
            json.dumps(all_results, indent=2)
        )
        write_metadata(
            str(Path(output_dir, f"metadata_{dataset_name}.json")),
            {
                "random_seed": seed,
                "test_size": test_size,
                "repeats": reps,
                "perturbations": [spec["name"] for spec in PERTURBATIONS],
                "dataset": dataset_name,
            },
        )
        print(f"Saved results for {dataset_name}")


def _aggregate_target_results(target_results: dict) -> dict:
    """Aggregate metrics across all targets and repetitions."""
    all_metrics = {
        "trtr": [], "trts": [], "tsts": [], "tstr": [],
        "accuracy": [], "balanced_accuracy": [], "macro_f1": [],
        "mae": [], "rmse": [], "r2": [],
    }
    ranking_metrics = {
        "kendalltau": [], "weightedtau": [], "spearmanr": [],
        "ndcg_score": [], "rbo": [],
    }

    for target_col, results in target_results.items():
        for r in results:
            # Basic metrics - handle both old format (single value) and new format (dict)
            if isinstance(r.get("trtr"), dict):
                # New format with multiple metrics
                for key in ["accuracy", "balanced_accuracy", "macro_f1"]:
                    if key in r["trtr"]:
                        all_metrics[key].append(r["trtr"][key])
                for key in ["mae", "rmse", "r2"]:
                    if key in r["trtr"]:
                        all_metrics[key].append(r["trtr"][key])
            else:
                # Old format with single value
                for key in ["trtr", "trts", "tsts", "tstr"]:
                    if key in r and r[key] is not None:
                        all_metrics[key].append(r[key])

            # Ranking scores
            if "ranking_scores" in r and r["ranking_scores"]:
                for key in ranking_metrics:
                    if key in r["ranking_scores"] and r["ranking_scores"][key] is not None:
                        ranking_metrics[key].append(r["ranking_scores"][key])

    # Compute aggregates with bootstrap CI
    aggregated = {}
    for metric, values in {**all_metrics, **ranking_metrics}.items():
        if values:
            lower, upper, estimate = bootstrap_ci(values, statistic=np.median, n_boot=500)
            aggregated[metric] = {
                "median": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "n": len(values),
            }

    return aggregated


def _run_target_experiments(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    cat_cols: list,
    num_cols: list,
    target_cols: list,
    classification_models: list,
    regression_models: list,
    reps: int,
    seed: int,
    test_size: float,
) -> dict:
    """Run experiments for each target column with repetitions."""
    np.random.seed(seed)
    random.seed(seed)

    results = {}

    for target_col in target_cols:
        is_categorical = target_col in cat_cols
        model_names = classification_models if is_categorical else regression_models
        task_type = "classification" if is_categorical else "regression"

        # Features are all columns except target
        feature_cat_cols = [c for c in cat_cols if c != target_col]
        feature_num_cols = [c for c in num_cols if c != target_col]

        X_real = real_df.drop(target_col, axis=1)
        y_real = real_df[target_col].copy()
        X_synth = synth_df.drop(target_col, axis=1)
        y_synth = synth_df[target_col].copy()

        # Drop rows with NaN in target (can happen with missingness perturbation)
        real_valid = ~y_real.isna()
        synth_valid = ~y_synth.isna()
        X_real = X_real[real_valid]
        y_real = y_real[real_valid]
        X_synth = X_synth[synth_valid]
        y_synth = y_synth[synth_valid]

        # For categorical targets, ensure proper dtype after NaN removal
        # (missingness converts to float, we need to convert back for classification)
        if is_categorical:
            y_real = y_real.astype(str)
            y_synth = y_synth.astype(str)

        # Skip if too few samples remain
        if len(y_real) < 10 or len(y_synth) < 10:
            results[target_col] = []
            continue

        # Skip if insufficient class diversity for classification
        if is_categorical:
            if len(y_real.unique()) < 2 or len(y_synth.unique()) < 2:
                results[target_col] = []
                continue

        target_results = []

        for rep in range(reps):
            rep_seed = seed + rep

            # Split real data (stratified for classification to ensure class balance)
            stratify_real = y_real if is_categorical else None
            stratify_synth = y_synth if is_categorical else None

            try:
                X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
                    X_real, y_real, test_size=test_size, random_state=rep_seed,
                    stratify=stratify_real
                )
            except ValueError:
                # Stratification failed (e.g., class with only 1 sample), fall back to random
                X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
                    X_real, y_real, test_size=test_size, random_state=rep_seed
                )

            try:
                X_train_synth, X_test_synth, y_train_synth, y_test_synth = train_test_split(
                    X_synth, y_synth, test_size=test_size, random_state=rep_seed,
                    stratify=stratify_synth
                )
            except ValueError:
                # Stratification failed, fall back to random
                X_train_synth, X_test_synth, y_train_synth, y_test_synth = train_test_split(
                    X_synth, y_synth, test_size=test_size, random_state=rep_seed
                )

            # Preprocess real data
            X_train_real_p, pre_real = fit_transform_df(
                X_train_real, feature_cat_cols, feature_num_cols
            )
            X_test_real_p = transform_df(X_test_real, pre_real, feature_cat_cols)

            # Preprocess synthetic data
            X_train_synth_p, pre_synth = fit_transform_df(
                X_train_synth, feature_cat_cols, feature_num_cols
            )
            X_test_synth_p = transform_df(X_test_synth, pre_synth, feature_cat_cols)

            # Cross-transform for cross-testing
            X_test_synth_for_real = transform_df(X_test_synth, pre_real, feature_cat_cols)
            X_test_real_for_synth = transform_df(X_test_real, pre_synth, feature_cat_cols)

            for model_name in model_names:
                base_model = MODELS.get(model_name)
                if base_model is None:
                    continue

                # Train on real data
                model_real = clone(base_model)
                if hasattr(model_real, "random_state"):
                    model_real.set_params(random_state=rep_seed)
                model_real.fit(X_train_real_p, y_train_real)

                # Train on synthetic data
                model_synth = clone(base_model)
                if hasattr(model_synth, "random_state"):
                    model_synth.set_params(random_state=rep_seed)
                model_synth.fit(X_train_synth_p, y_train_synth)

                # Get predictions
                pred_trtr = model_real.predict(X_test_real_p)
                pred_trts = model_real.predict(X_test_synth_for_real)
                pred_tsts = model_synth.predict(X_test_synth_p)
                pred_tstr = model_synth.predict(X_test_real_for_synth)

                # Get probabilities for classification AUROC
                proba_trtr = proba_trts = proba_tsts = proba_tstr = None
                if task_type == "classification" and hasattr(model_real, "predict_proba"):
                    try:
                        proba_trtr = model_real.predict_proba(X_test_real_p)
                        proba_trts = model_real.predict_proba(X_test_synth_for_real)
                        proba_tsts = model_synth.predict_proba(X_test_synth_p)
                        proba_tstr = model_synth.predict_proba(X_test_real_for_synth)
                    except Exception:
                        pass

                # Compute metrics
                if task_type == "classification":
                    trtr = _classification_metrics(y_test_real, pred_trtr, proba_trtr)
                    trts = _classification_metrics(y_test_synth, pred_trts, proba_trts)
                    tsts = _classification_metrics(y_test_synth, pred_tsts, proba_tsts)
                    tstr = _classification_metrics(y_test_real, pred_tstr, proba_tstr)
                else:
                    trtr = _regression_metrics(y_test_real, pred_trtr)
                    trts = _regression_metrics(y_test_synth, pred_trts)
                    tsts = _regression_metrics(y_test_synth, pred_tsts)
                    tstr = _regression_metrics(y_test_real, pred_tstr)

                # Extract feature importance and compute ranking scores
                feat_real = get_feature_importance(
                    model_real, X_train_real_p, y_train_real, random_state=rep_seed
                )
                feat_synth = get_feature_importance(
                    model_synth, X_train_synth_p, y_train_synth, random_state=rep_seed
                )
                ranking_scores = compute_ranking_scores(feat_real, feat_synth)

                target_results.append({
                    "rep": rep,
                    "model": model_name,
                    "task_type": task_type,
                    "trtr": trtr,
                    "trts": trts,
                    "tsts": tsts,
                    "tstr": tstr,
                    "ranking_scores": ranking_scores,
                    "feature_importance_real": feat_real,
                    "feature_importance_synth": feat_synth,
                })

        results[target_col] = target_results

    return results


def _apply_perturbation(df, spec, cat_cols, num_cols):
    """Apply perturbation based on available column types."""
    name = spec["name"]
    params = spec.get("params", {})

    # Numeric perturbations - skip if no numeric columns
    if name == "mean_variance_drift":
        if not num_cols:
            return df.copy()
        return perturbations.apply_mean_variance_drift(df, num_cols, **params)
    if name == "noise_injection":
        if not num_cols:
            return df.copy()
        return perturbations.apply_noise_injection(df, num_cols, **params)
    if name == "correlation_mix":
        if not num_cols:
            return df.copy()
        return perturbations.apply_correlation_mix(df, num_cols, **params)

    # Categorical perturbations - skip if no categorical columns
    if name == "category_collapse":
        if not cat_cols:
            return df.copy()
        return perturbations.apply_category_collapse(df, cat_cols, **params)

    # Mixed perturbations
    if name == "missingness_mcar":
        all_cols = list(cat_cols) + list(num_cols)
        if not all_cols:
            return df.copy()
        return perturbations.apply_missingness(df, all_cols, **params)

    raise ValueError(f"Unknown perturbation: {name}")


if __name__ == "__main__":
    run_all()
