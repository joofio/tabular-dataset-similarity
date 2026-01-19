"""Aggregation utilities for dataset similarity scores."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


def aggregate_scores(metric_by_target: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    """Aggregate metric scores by taking the median across targets."""

    if not metric_by_target:
        return {}

    metric_names = next(iter(metric_by_target.values())).keys()
    aggregated = {}
    for metric in metric_names:
        values = [metrics[metric] for metrics in metric_by_target.values()]
        aggregated[metric] = float(np.median(values))
    return aggregated
