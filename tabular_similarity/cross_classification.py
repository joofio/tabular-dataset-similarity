"""Cross-classification (TRTR/TRTS/TSTR/TSTS) metrics."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from .preprocessing import fit_transform_df, transform_df
from .splits import train_test_split_indices


def _classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    if y_proba is not None:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        except ValueError:
            # AUROC not defined for a single class or invalid probability shape.
            metrics["auroc"] = float("nan")
    return metrics


def _regression_metrics(y_true, y_pred) -> Dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_cross_classification(
    X_real,
    y_real,
    X_synth,
    y_synth,
    categorical_cols,
    numeric_cols,
    model,
    task_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Evaluate TRTR/TRTS/TSTR/TSTS metrics with train-only preprocessing."""

    real_train_idx, real_test_idx = train_test_split_indices(
        len(X_real),
        y=y_real,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )
    synth_train_idx, synth_test_idx = train_test_split_indices(
        len(X_synth),
        y=y_synth,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )

    X_real_train = X_real.iloc[real_train_idx]
    y_real_train = y_real.iloc[real_train_idx]
    X_real_test_raw = X_real.iloc[real_test_idx]
    y_real_test = y_real.iloc[real_test_idx]

    X_synth_train = X_synth.iloc[synth_train_idx]
    y_synth_train = y_synth.iloc[synth_train_idx]
    X_synth_test_raw = X_synth.iloc[synth_test_idx]
    y_synth_test = y_synth.iloc[synth_test_idx]

    X_real_train_p, pre_real = fit_transform_df(
        X_real_train, categorical_cols, numeric_cols
    )
    X_synth_train_p, pre_synth = fit_transform_df(
        X_synth_train, categorical_cols, numeric_cols
    )

    X_real_test = transform_df(X_real_test_raw, pre_real)
    X_synth_test = transform_df(X_synth_test_raw, pre_real)
    X_real_test_for_synth = transform_df(X_real_test_raw, pre_synth)
    X_synth_test_for_synth = transform_df(X_synth_test_raw, pre_synth)

    tr_model = model
    ts_model = model.__class__(**model.get_params())

    tr_model.fit(X_real_train_p, y_real_train)
    ts_model.fit(X_synth_train_p, y_synth_train)

    if task_type == "classification":
        trtr_proba = (
            tr_model.predict_proba(X_real_test)
            if hasattr(tr_model, "predict_proba")
            else None
        )
        trts_proba = (
            tr_model.predict_proba(X_synth_test)
            if hasattr(tr_model, "predict_proba")
            else None
        )
        tsts_proba = (
            ts_model.predict_proba(X_real_test_for_synth)
            if hasattr(ts_model, "predict_proba")
            else None
        )
        tstr_proba = (
            ts_model.predict_proba(X_synth_test_for_synth)
            if hasattr(ts_model, "predict_proba")
            else None
        )
        trtr = _classification_metrics(
            y_real_test, tr_model.predict(X_real_test), trtr_proba
        )
        trts = _classification_metrics(
            y_synth_test, tr_model.predict(X_synth_test), trts_proba
        )
        tsts = _classification_metrics(
            y_real_test, ts_model.predict(X_real_test_for_synth), tsts_proba
        )
        tstr = _classification_metrics(
            y_synth_test, ts_model.predict(X_synth_test_for_synth), tstr_proba
        )
    else:
        trtr = _regression_metrics(y_real_test, tr_model.predict(X_real_test))
        trts = _regression_metrics(y_synth_test, tr_model.predict(X_synth_test))
        tsts = _regression_metrics(
            y_real_test, ts_model.predict(X_real_test_for_synth)
        )
        tstr = _regression_metrics(
            y_synth_test, ts_model.predict(X_synth_test_for_synth)
        )

    return {"trtr": trtr, "trts": trts, "tsts": tsts, "tstr": tstr}
