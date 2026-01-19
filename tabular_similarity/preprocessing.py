"""Preprocessing utilities with train-only fitting to avoid leakage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


@dataclass
class Preprocessor:
    """Container for a fitted preprocessing pipeline."""

    transformer: ColumnTransformer
    feature_names: List[str]


def build_preprocessor(
    categorical_cols: Sequence[str],
    numeric_cols: Sequence[str],
) -> ColumnTransformer:
    """Create a ColumnTransformer for ordinal-encoding categoricals and imputing values.

    The encoder uses a fixed unknown value to keep train/test handling consistent.
    """

    cat_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    num_pipeline = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])

    transformer = ColumnTransformer(
        transformers=[
            ("categorical", cat_pipeline, list(categorical_cols)),
            ("numeric", num_pipeline, list(numeric_cols)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return transformer


def fit_transform_df(
    X_train: pd.DataFrame,
    categorical_cols: Sequence[str],
    numeric_cols: Sequence[str],
) -> Tuple[pd.DataFrame, Preprocessor]:
    """Fit the preprocessor on training data and return the transformed DataFrame."""

    # Convert categorical columns to string to avoid mixed type errors
    X_train = X_train.copy()
    if categorical_cols:
        X_train[list(categorical_cols)] = X_train[list(categorical_cols)].astype(str)

    transformer = build_preprocessor(categorical_cols, numeric_cols)
    array = transformer.fit_transform(X_train)
    feature_names = list(transformer.get_feature_names_out())
    X_train_p = pd.DataFrame(array, columns=feature_names, index=X_train.index)
    return X_train_p, Preprocessor(transformer=transformer, feature_names=feature_names)


def transform_df(
    X: pd.DataFrame,
    preprocessor: Preprocessor,
    categorical_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Transform data using a fitted preprocessor."""

    # Convert categorical columns to string to avoid mixed type errors
    X = X.copy()
    if categorical_cols:
        X[list(categorical_cols)] = X[list(categorical_cols)].astype(str)

    array = preprocessor.transformer.transform(X)
    return pd.DataFrame(array, columns=preprocessor.feature_names, index=X.index)
