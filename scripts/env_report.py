"""Collect environment versions for reproducibility."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn


def write_env_report(path: str) -> None:
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }
    Path(path).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    write_env_report("results/env_report.json")
