"""Plotting helpers for metric summaries."""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from tabular_similarity.reporting import monotonicity_score, stability_summary


def summarize_metrics(levels: Sequence[float], metric_values: Mapping[str, Sequence[float]]):
    records = []
    for name, values in metric_values.items():
        records.append(
            {
                "metric": name,
                "monotonicity": monotonicity_score(levels, values),
                "std": stability_summary(values)["std"],
                "iqr": stability_summary(values)["iqr"],
            }
        )
    return pd.DataFrame.from_records(records)
