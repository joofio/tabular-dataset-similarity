# TODO

## Data pipeline and leakage
- [x] Fit encoders and imputers on train only, then apply to test/synthetic with explicit unknown handling
  - **Implemented in**: `tabular_similarity/preprocessing.py` - `fit_transform_df()` fits on train, `transform_df()` applies to test
  - OrdinalEncoder with `handle_unknown="use_encoded_value", unknown_value=-1`
  - SimpleImputer for both categorical (most_frequent) and numeric (median)
  - Categorical columns converted to string with `.astype(str)` to avoid mixed type errors
- [x] Use stratified splits for classification and repeated CV; parameterize split ratio and seeds
  - **Implemented in**: `tabular_similarity/splits.py` - `RepeatedStratifiedKFold` support, configurable `test_size` and `random_state`
  - **Config in**: `configs/experiment.py` - `test_size`, `random_seed`, `repeats` parameters
- [x] Standardize preprocessing across real/synthetic using a single `Pipeline` + `ColumnTransformer`
  - **Implemented in**: `tabular_similarity/preprocessing.py` - uses `ColumnTransformer` with separate pipelines for categorical and numeric
- [x] Compute permutation importance on a held-out fold with explicit scoring and fixed seeds
  - **Implemented in**: `tabular_similarity/importance.py` - `fit_and_importance()` uses validation set for permutation importance
  - **Also in**: `tabular_similarity/ranking_scores.py` - `get_feature_importance()` with configurable `random_state`
- [x] Replace random tie-breaking in ranks with deterministic tie handling
  - **Implemented in**: `tabular_similarity/ranking_metrics.py` - `rank_features()` uses alphabetical tie-breaking
  - **Fixed in**: `tabular_similarity/ranking_scores.py` - now uses deterministic sorting by (value desc, name asc)

## Metrics and baselines
- [x] Replace accuracy-only CC with balanced accuracy, macro-F1, AUROC (where valid) and add MAE/RMSE for regression; normalize robustly
  - **Implemented in**: `tabular_similarity/cross_classification.py`
    - Classification: accuracy, balanced_accuracy, macro_f1, auroc
    - Regression: mae, rmse, r2
- [x] Add baseline similarity metrics: propensity-score classifier (two-sample), marginal JS/Wasserstein, correlation-matrix distance, MMD/energy
  - **Implemented in**: `tabular_similarity/baselines.py`
    - `propensity_score_metrics()` - AUROC and Brier score
    - `marginal_js_distance()` - Jensen-Shannon for categorical and numeric
    - `marginal_wasserstein_distance()` - Wasserstein for numeric
    - `correlation_matrix_distance()` - Frobenius norm
    - `mmd_rbf()` - Maximum Mean Discrepancy with RBF kernel
    - `energy_distance_score()` - Energy distance
- [x] Remove or quarantine string edit distances, or replace with ranking-aware metrics (Spearman footrule, Kendall top-weighted, Jaccard@k)
  - **Implemented in**: `tabular_similarity/ranking_metrics.py`
    - `spearman_footrule()`, `jaccard_at_k()`, `compare_rankings()` with kendall_tau, weighted_tau
  - **Also in**: `tabular_similarity/ranking_scores.py` - comprehensive ranking metrics including:
    - ndcg_score, cohen_kappa_score, r2_score, kendalltau, weightedtau, spearmanr, rbo
    - Optional: levenshtein, jaro_winkler (requires packages)
- [x] Expose RBO parameters (p, depth) and evaluate top-k sensitivity
  - **Implemented in**: `tabular_similarity/ranking_metrics.py` - `compare_rankings()` accepts `rbo_p` and `jaccard_k` parameters

## Perturbations and experiments
- [x] Implement perturbations beyond column permutation: mean/variance drift, category collapse, noise injection, correlation attenuation/amplification, label shift, missingness
  - **Implemented in**: `tabular_similarity/perturbations.py`
    - `apply_mean_variance_drift()` - shift and scale numeric columns
    - `apply_noise_injection()` - Gaussian noise to numeric columns
    - `apply_category_collapse()` - collapse rare categories
    - `apply_correlation_mix()` - attenuate correlations
    - `apply_label_shift()` - resample to new class distribution
    - `apply_missingness()` - MCAR and MNAR missingness
    - `apply_interaction_rewire()` - break pairwise interactions
- [x] Add config-driven perturbation levels and seeds
  - **Implemented in**: `configs/perturbations.py` - list of perturbation specs with names and params
  - **Used by**: `scripts/run_all.py` - applies perturbations based on config
- [ ] Make synthetic generation reproducible with relative paths and parameter logging
  - **Partially implemented**: `scripts/synth_generation.py` - writes config JSON but doesn't actually run R script
  - **TODO**: Wire up to actual R generator, use relative paths
- [x] Add a higher-fidelity health-like dataset loader and preprocessing
  - **Implemented in**: `scripts/load_health_dataset.py` - explicit column typing
  - **Config in**: `configs/experiment.py` - 6 health datasets with column specifications

## Aggregation and statistics
- [x] Define a primary aggregation rule (median across targets/models) and implement a single "RBO similarity score"
  - **Implemented in**: `tabular_similarity/aggregation.py` - `aggregate_scores()` takes median across targets
- [x] Add bootstrap confidence intervals over seeds/splits/targets
  - **Implemented in**: `tabular_similarity/stats.py` - `bootstrap_ci()` with configurable statistic and alpha
- [x] Add mixed-effects analysis (dataset and target as random effects) for metric comparisons
  - **Implemented in**: `tabular_similarity/stats.py` - `mixed_effects_model()` using statsmodels
- [x] Add monotonicity and stability summaries per metric
  - **Implemented in**: `tabular_similarity/reporting.py` - `monotonicity_score()` and `stability_summary()`
  - **Used by**: `scripts/plot_metrics.py` - `summarize_metrics()` function

## Reproducibility and reporting
- [x] Pin environment versions
  - **Implemented**: `requirements.txt` with pinned versions for all dependencies
- [x] Centralize experiment config (datasets, targets, models, metrics, seeds) in config and read from notebooks/scripts
  - **Implemented in**: `configs/experiment.py` - datasets, models, seeds, test_size, repeats
  - **Implemented in**: `configs/perturbations.py` - perturbation specs
- [x] Save train/test splits and seeds for each run; log to results metadata
  - **Implemented in**: `scripts/save_splits.py` - saves splits to JSON per dataset
  - **Implemented in**: `scripts/log_metadata.py` - saves metadata per dataset
  - **Implemented in**: `scripts/run_all.py` - writes metadata with each experiment
- [x] Add a single entry point to regenerate figures and tables
  - **Implemented in**: `scripts/run_all.py` - full experiment pipeline
    - Loads datasets from config
    - Applies perturbations
    - Computes propensity metrics and baseline metrics
    - Runs TRTR/TRTS/TSTS/TSTR with repetitions
    - Extracts feature importance and ranking scores
    - Saves results to JSON

## Integration status between run_all.py and tabular_similarity modules

### Currently used in run_all.py

- `tabular_similarity/baselines.py` - **ALL metrics now used:**
  - `propensity_score_metrics()`
  - `marginal_js_distance()`, `marginal_wasserstein_distance()`
  - `correlation_matrix_distance()`, `mmd_rbf()`, `energy_distance_score()`
- `tabular_similarity/perturbations.py` - all perturbation functions
- `tabular_similarity/preprocessing.py` - `fit_transform_df()`, `transform_df()`
- `tabular_similarity/ranking_scores.py` - `compute_ranking_scores()`, `get_feature_importance()`
- `tabular_similarity/aggregation.py` - `aggregate_scores()` (via `_aggregate_target_results()`)
- `tabular_similarity/stats.py` - `bootstrap_ci()` for confidence intervals
- **Rich metrics integrated:**
  - Classification: accuracy, balanced_accuracy, macro_f1, auroc
  - Regression: mae, rmse, mse, r2

### Not yet integrated into run_all.py

- `tabular_similarity/cross_classification.py` - has `evaluate_cross_classification()` wrapper
  - **Current**: run_all.py implements the logic directly with richer metrics
  - **Status**: Functionality is equivalent, just not using the wrapper
- `tabular_similarity/importance.py` - has `fit_and_importance()` with validation set
  - **Current**: ranking_scores.py has similar implementation
- `tabular_similarity/ranking_metrics.py` - has alternative `rank_features()`, `compare_rankings()`
  - **Current**: ranking_scores.py provides overlapping functionality
- `tabular_similarity/reporting.py` - `monotonicity_score()`, `stability_summary()`
  - **Status**: Available for post-processing analysis, not needed in main pipeline

## Completed work summary

All major TODO items have been completed:

1. ✅ **Integrated all metrics into run_all.py**:
   - Implemented rich classification metrics (accuracy, balanced_accuracy, macro_f1, auroc)
   - Implemented rich regression metrics (mae, rmse, mse, r2)
   - Added all baseline metrics (marginal JS/Wasserstein, correlation distance, MMD, energy)
   - Integrated `bootstrap_ci()` for confidence intervals in aggregation
   - Using `aggregate_scores()` logic for summary statistics

2. ✅ **Fixed deterministic tie-breaking**:
   - Updated `ranking_scores.py` to use deterministic sorting by (value desc, name asc)
   - Removed random noise approach

3. ✅ **Added requirements.txt**:
   - Pinned all core dependencies (numpy, pandas, scikit-learn, scipy, statsmodels)
   - Included optional dependencies (rbo, textdistance, Levenshtein, jellyfish)

## Remaining optional work

1. **Wire up synthetic generation** (if needed):
   - Connect `synth_generation.py` to actual R script execution
   - Currently writes config but doesn't execute R generator

2. **Post-processing analysis tools** (already available):
   - `tabular_similarity/reporting.py` - monotonicity and stability analysis
   - `scripts/plot_metrics.py` - visualization helpers
   - These can be used for results analysis after experiments
