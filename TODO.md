# TODO

## Data pipeline and leakage
- Fit encoders and imputers on train only, then apply to test/synthetic with explicit unknown handling; update `Final - Function Definition.ipynb`.
- Use stratified splits for classification and repeated CV; parameterize split ratio and seeds; update `Final - Function Definition.ipynb`.
- Standardize preprocessing across real/synthetic using a single `Pipeline` + `ColumnTransformer`; update `Final - Function Definition.ipynb`.
- Compute permutation importance on a held-out fold with explicit scoring and fixed seeds; update `Final - Function Definition.ipynb`.
- Replace random tie-breaking in ranks with deterministic tie handling; update `Final - Function Definition.ipynb`.

## Metrics and baselines
- Replace accuracy-only CC with balanced accuracy, macro-F1, AUROC (where valid) and add MAE/RMSE for regression; normalize robustly; update `Final - Function Definition.ipynb`.
- Add baseline similarity metrics: propensity-score classifier (two-sample), marginal JS/Wasserstein, correlation-matrix distance, MMD/energy; add `metrics/baselines.py` and call from `Final - Function Definition.ipynb`.
- Remove or quarantine string edit distances, or replace with ranking-aware metrics (Spearman footrule, Kendall top-weighted, Jaccard@k); update `Final - Function Definition.ipynb`.
- Expose RBO parameters (p, depth) and evaluate top-k sensitivity; update `Final - Function Definition.ipynb`.

## Perturbations and experiments
- Implement perturbations beyond column permutation: mean/variance drift, category collapse, noise injection, correlation attenuation/amplification, label shift, missingness; add `scripts/perturbations.py`.
- Add config-driven perturbation levels and seeds; add `configs/perturbations.yaml` and wire into `Final - Function Definition.ipynb`.
- Make synthetic generation reproducible with relative paths and parameter logging; update `generate_data.R`.
- Add a higher-fidelity health-like dataset loader and preprocessing; add `scripts/load_health_dataset.py` and document in `data/README.md`.

## Aggregation and statistics
- Define a primary aggregation rule (median across targets/models) and implement a single "RBO similarity score"; update `Final - Function Definition.ipynb`.
- Add bootstrap confidence intervals over seeds/splits/targets; add `scripts/stats.py`.
- Add mixed-effects analysis (dataset and target as random effects) for metric comparisons; add `scripts/stats.py`.
- Add monotonicity and stability summaries per metric; update `Final - Viz copy.ipynb` or add `scripts/plot_metrics.py`.

## Reproducibility and reporting
- Pin environment versions; add `requirements.txt` or `environment.yml`.
- Centralize experiment config (datasets, targets, models, metrics, seeds) in `configs/experiment.yaml` and read it from notebooks/scripts.
- Save train/test splits and seeds for each run; log to `results/metadata.json`.
- Add a single entry point to regenerate figures and tables; add `scripts/run_all.py` and update `README.md`.
