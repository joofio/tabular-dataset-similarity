"""Split utilities for reproducible train/test and repeated CV."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
    train_test_split,
)


def train_test_split_indices(
    n_samples: int,
    y: Sequence | None = None,
    test_size: float = 0.2,
    stratify: bool = False,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return train/test indices with optional stratification."""

    idx = np.arange(n_samples)
    stratify_y = y if stratify and y is not None else None
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        stratify=stratify_y,
        random_state=random_state,
    )
    return np.array(train_idx), np.array(test_idx)


def generate_repeated_splits(
    n_samples: int,
    y: Sequence | None = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int | None = None,
    stratify: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create repeated CV split indices, optionally stratified."""

    if stratify and y is not None:
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
    else:
        splitter = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )

    splits = []
    for train_idx, test_idx in splitter.split(np.arange(n_samples), y):
        splits.append((np.array(train_idx), np.array(test_idx)))
    return splits


def save_splits(path: str | Path, splits: Sequence[Tuple[np.ndarray, np.ndarray]]) -> None:
    """Persist splits to JSON for reproducibility."""

    path = Path(path)
    payload = [
        {"train_idx": train.tolist(), "test_idx": test.tolist()}
        for train, test in splits
    ]
    path.write_text(json.dumps(payload, indent=2))
