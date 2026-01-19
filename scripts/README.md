# Scripts

This folder contains small, documented entry points for running experiments and
reproducibility tasks. All scripts are plain Python modules.

## Overview

- `run_all.py`
  - Runs a minimal end-to-end baseline experiment using the configs in
    `configs/experiment.py` and `configs/perturbations.py`.
  - Writes `results/baseline_metrics.json` and `results/metadata.json`.

- `load_health_dataset.py`
  - Loads a CSV and applies explicit column typing for categorical and numeric
    columns. Useful for health-like tabular datasets.

- `plot_metrics.py`
  - Builds a summary table with monotonicity and stability stats for metrics.
  - Intended to be called from a notebook or a small driver script.

- `env_report.py`
  - Writes a minimal environment version report to `results/env_report.json`.

- `save_splits.py`
  - Generates repeated CV splits and saves them to `results/splits.json` for
    reproducibility.

- `synth_generation.py`
  - Helper to write a synthetic generator config and optionally call an R
    script via `Rscript`.

- `export_requirements.py`
  - Emits a pinned `requirements.txt` based on installed package versions.

- `log_metadata.py`
  - Writes experiment metadata (seeds, configs, split references) to
    `results/metadata.json`.

## Usage examples

Run the baseline experiment:

```bash
python scripts/run_all.py
```

Generate and save repeated splits:

```bash
python scripts/save_splits.py
```

Write a minimal environment report:

```bash
python scripts/env_report.py
```

Export a pinned requirements file:

```bash
python scripts/export_requirements.py
```

## Notes

- Paths are relative to the repository root.
- Update `configs/experiment.py` and `configs/perturbations.py` before running
  experiments.
- Scripts write outputs to the `results/` directory.
