# Methods and Data

## 1. Introduction and Overview

This study presents a comprehensive framework for evaluating dataset similarity in tabular data contexts, with particular emphasis on comparing real and synthetic or perturbed datasets. The methodology integrates multiple complementary approaches: distributional distance metrics, discriminative classifiers, predictive model performance comparisons, and feature importance ranking analysis. This multi-faceted evaluation strategy provides robust assessment of dataset similarity across different dimensions of data fidelity.

## 2. Data Sources and Characteristics

### 2.1 Dataset Description

We evaluate our methodology on six health-related tabular datasets, each presenting distinct structural characteristics in terms of feature composition and dimensionality. Table 1 summarizes the key properties of each dataset.

| Dataset | Categorical Features | Numeric Features | Total Features |
|---------|---------------------|------------------|----------------|
| real_data_testing | 3 (sex, cp, Selector) | 5 (age, trestbps, chol, thalach, oldpeak) | 8 |
| real_data2_testing | 3 (sex, cp, Selector) | 5 (age, trestbps, chol, thalach, oldpeak) | 8 |
| real_data3_testing | 3 (sex, cp, Selector) | 5 (age, trestbps, chol, thalach, oldpeak) | 8 |
| real_data4_testing | 3 (sex, cp, Selector) | 5 (age, trestbps, chol, thalach, oldpeak) | 8 |
| real_data5_testing | 0 | 8 (age, trestbps, chol, thalach, oldpeak, sex, cp, Selector) | 8 |
| real_data6_testing | 3 (sex, cp, Selector) | 5 (age, trestbps, chol, thalach, oldpeak) | 8 |

The datasets encompass health-related attributes including demographic variables (age, sex), clinical measurements (resting blood pressure, serum cholesterol, maximum heart rate achieved, ST depression), and categorical indicators (chest pain type, selector variable). This composition reflects typical real-world health data scenarios where mixed data types necessitate specialized preprocessing strategies.

### 2.2 Feature Type Specification

Explicit declaration of feature types is fundamental to our methodology, as it governs:
1. **Preprocessing strategy selection**: Categorical features undergo ordinal encoding, while numeric features receive median imputation and passthrough transformation.
2. **Perturbation applicability**: Certain perturbations (e.g., noise injection, correlation mixing) apply exclusively to numeric features, while others (e.g., category collapse) target categorical features.
3. **Model selection**: Classification models are employed when predicting categorical targets, regression models for numeric targets.
4. **Metric computation**: Feature type determines appropriate distance metrics (Jensen-Shannon for categorical distributions, Wasserstein for numeric).

## 3. Preprocessing Pipeline

### 3.1 Design Principles

Our preprocessing pipeline adheres to several critical principles to ensure methodological rigor:

**Principle 1: Train-Only Fitting**
All transformations (encoding, imputation) are fitted exclusively on training data and subsequently applied to test/synthetic data. This prevents data leakage and ensures unbiased evaluation metrics.

**Principle 2: Consistent Transformation**
A unified `ColumnTransformer` architecture applies identical preprocessing logic across all data splits, ensuring comparability of results.

**Principle 3: Explicit Unknown Handling**
The pipeline explicitly handles previously unseen categories in test data through the `unknown_value=-1` parameter in `OrdinalEncoder`, preventing runtime failures during cross-dataset evaluation.

### 3.2 Implementation Architecture

The preprocessing pipeline employs scikit-learn's `ColumnTransformer` to orchestrate parallel processing of different feature types:

```
ColumnTransformer
├── Categorical Pipeline
│   ├── SimpleImputer(strategy="most_frequent")
│   └── OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
└── Numeric Pipeline
    ├── SimpleImputer(strategy="median")
    └── passthrough
```

**Categorical Processing**: Missing categorical values are imputed using the most frequent category observed in the training set. Subsequently, ordinal encoding transforms categories to integer representations. The `handle_unknown="use_encoded_value"` configuration assigns a dedicated code (-1) to categories encountered during transformation that were absent during fitting—a common scenario when applying real-data-fitted transformers to synthetic data.

**Numeric Processing**: Missing numeric values are imputed using the median, a robust central tendency measure less sensitive to outliers than the mean. Numeric features pass through without scaling, as tree-based models (our primary classifiers) are invariant to monotonic transformations.

### 3.3 Type Coercion

Prior to encoding, categorical columns undergo explicit string conversion (`astype(str)`) to prevent mixed-type errors that arise when categorical columns contain heterogeneous types (integers, strings, missing values). This coercion ensures uniform input to the ordinal encoder regardless of source data quality.

## 4. Perturbation Methodology

### 4.1 Rationale

To systematically evaluate similarity metrics under controlled conditions, we introduce synthetic perturbations that simulate common forms of distribution shift between datasets. Each perturbation type targets specific data characteristics, enabling assessment of metric sensitivity to different forms of divergence.

### 4.2 Perturbation Types

#### 4.2.1 Mean-Variance Drift
**Target**: Numeric features
**Mechanism**: Applies affine transformation $X' = \alpha X + \beta$ to numeric columns
**Parameters**: `mean_shift` ($\beta$), `scale` ($\alpha$)
**Rationale**: Simulates systematic measurement bias or calibration differences between data sources, common in multi-site clinical studies where measurement protocols may vary.

#### 4.2.2 Noise Injection
**Target**: Numeric features
**Mechanism**: Adds Gaussian noise $X' = X + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma^2)$
**Parameters**: `std` ($\sigma$), `random_state`
**Rationale**: Models measurement noise, transcription errors, or natural variation in repeated measurements. The Gaussian assumption reflects the central limit theorem's applicability to aggregated measurement errors.

#### 4.2.3 Category Collapse
**Target**: Categorical features
**Mechanism**: Maps rare categories (frequency < `min_freq`) to a common "OTHER" label
**Parameters**: `min_freq`, `other_label`
**Rationale**: Simulates coding scheme differences between datasets, where fine-grained categories in one source are aggregated in another. Common in administrative data where coding standards evolve over time.

#### 4.2.4 Correlation Mixing
**Target**: Numeric features
**Mechanism**: Interpolates between original and row-permuted data: $X' = \alpha X + (1-\alpha) X_{\text{permuted}}$
**Parameters**: `alpha` (mixing coefficient), `random_state`
**Rationale**: Attenuates inter-feature correlations while preserving marginal distributions. Tests whether similarity metrics capture multivariate structure beyond univariate statistics.

#### 4.2.5 Missingness (MCAR)
**Target**: All features
**Mechanism**: Randomly masks values with probability `rate` under Missing Completely At Random assumption
**Parameters**: `rate`, `random_state`
**Rationale**: Simulates data quality degradation from incomplete records. MCAR represents the simplest missing data mechanism, providing a baseline for missingness impact assessment.

#### 4.2.6 Missingness (MNAR)
**Target**: Numeric features
**Mechanism**: Masks values exceeding a quantile threshold, creating systematic missingness
**Parameters**: `threshold` (quantile), `rate`
**Rationale**: Models informative missingness where extreme values are preferentially missing, such as patients with severe symptoms dropping out of studies.

#### 4.2.7 Label Shift
**Target**: Classification target
**Mechanism**: Resamples data to achieve specified class proportions
**Parameters**: `desired_proportions`, `random_state`
**Rationale**: Simulates changes in outcome prevalence between populations, critical for transfer learning evaluation in epidemiological contexts.

#### 4.2.8 Interaction Rewiring
**Target**: Numeric feature pairs
**Mechanism**: Permutes feature values within quantile bins, breaking pairwise interactions while partially preserving marginals
**Parameters**: `cols` (feature subset), `random_state`
**Rationale**: Tests sensitivity to higher-order statistical dependencies beyond correlations.

### 4.3 Perturbation Configuration

Perturbations are applied through a configuration-driven system that specifies perturbation type and parameters:

```python
PERTURBATIONS = [
    {"name": "mean_variance_drift", "params": {"mean_shift": 0.5, "scale": 1.2}},
    {"name": "noise_injection", "params": {"std": 0.1}},
    {"name": "category_collapse", "params": {"min_freq": 0.05}},
    {"name": "correlation_mix", "params": {"alpha": 0.7}},
    {"name": "missingness_mcar", "params": {"rate": 0.1}},
]
```

This declarative approach ensures reproducibility and facilitates systematic exploration of perturbation parameter spaces.

## 5. Similarity Metrics

### 5.1 Distributional Distance Metrics

#### 5.1.1 Jensen-Shannon Distance
**Definition**: $JSD(P||Q) = \sqrt{\frac{1}{2}D_{KL}(P||M) + \frac{1}{2}D_{KL}(Q||M)}$, where $M = \frac{1}{2}(P+Q)$

**Application**: Computed for each categorical feature by comparing empirical category distributions between datasets. For numeric features, values are discretized into 20 histogram bins before computing JS distance.

**Interpretation**: Bounded in $[0, 1]$, with 0 indicating identical distributions. The symmetric, finite nature of JS distance makes it preferable to KL divergence for comparing potentially non-overlapping distributions.

**Aggregation**: Average JS distance across all features provides a summary measure of marginal distributional similarity.

#### 5.1.2 Wasserstein Distance
**Definition**: $W_1(P, Q) = \inf_{\gamma \in \Gamma(P,Q)} \mathbb{E}_{(x,y)\sim\gamma}[|x-y|]$

**Application**: Computed for each numeric feature using the 1-dimensional Earth Mover's Distance formulation.

**Interpretation**: Represents the minimum "work" required to transform one distribution into another, measured in the feature's native units. Unlike JS distance, Wasserstein distance accounts for the geometry of the feature space.

**Aggregation**: Average across numeric features.

#### 5.1.3 Correlation Matrix Distance
**Definition**: Frobenius norm of correlation matrix difference: $||\Sigma_1 - \Sigma_2||_F$

**Application**: Correlation matrices computed for numeric features only; requires minimum 2 numeric features.

**Interpretation**: Captures differences in linear dependency structure between datasets. Important for detecting perturbations that preserve marginals but alter multivariate relationships (e.g., correlation mixing).

### 5.2 Two-Sample Discriminative Metrics

#### 5.2.1 Propensity Score Classifier
**Methodology**:
1. Concatenate real and synthetic datasets with binary labels (0=real, 1=synthetic)
2. Train logistic regression classifier to discriminate between sources
3. Evaluate on held-out test set (20% stratified split)

**Metrics Reported**:
- **AUROC**: Area under receiver operating characteristic curve. Values near 0.5 indicate indistinguishable datasets; values approaching 1.0 indicate perfect discriminability.
- **Brier Score**: Mean squared error of probability predictions. Lower values indicate better calibrated predictions.

**Rationale**: A classifier's ability to distinguish real from synthetic data directly measures their statistical discriminability. This approach detects subtle multivariate differences that univariate metrics might miss.

#### 5.2.2 Maximum Mean Discrepancy (MMD)
**Definition**: $MMD^2(P, Q) = \mathbb{E}[k(X,X')] + \mathbb{E}[k(Y,Y')] - 2\mathbb{E}[k(X,Y)]$

**Implementation**: Uses RBF kernel with bandwidth $\gamma = 1/d$ where $d$ is the feature dimension.

**Interpretation**: A kernel-based two-sample test statistic. MMD equals zero if and only if the two distributions are identical in the reproducing kernel Hilbert space induced by the kernel.

#### 5.2.3 Energy Distance
**Definition**: $E(P, Q) = 2\mathbb{E}[||X-Y||] - \mathbb{E}[||X-X'||] - \mathbb{E}[||Y-Y'||]$

**Interpretation**: A distribution-free multivariate generalization of the two-sample Cramér-von Mises statistic. Zero if and only if distributions are identical.

### 5.3 Predictive Performance Metrics

#### 5.3.1 Cross-Classification Framework

The Train-Real/Train-Synthetic (TRTS) framework evaluates dataset similarity through downstream predictive task performance. For each target column, we train models on one dataset and evaluate on both:

| Abbreviation | Train Data | Test Data | Interpretation |
|--------------|-----------|-----------|----------------|
| TRTR | Real | Real | Baseline real-data performance |
| TRTS | Real | Synthetic | Real model's generalization to synthetic |
| TSTS | Synthetic | Synthetic | Baseline synthetic-data performance |
| TSTR | Synthetic | Real | Synthetic model's generalization to real |

**Similarity Indicator**: When TRTR ≈ TRTS and TSTS ≈ TSTR, the datasets are functionally interchangeable for predictive modeling purposes.

#### 5.3.2 Classification Metrics
For categorical target variables:
- **Accuracy**: Proportion of correct predictions
- **Balanced Accuracy**: Average of per-class recall, accounts for class imbalance
- **Macro F1**: Unweighted mean of per-class F1 scores
- **AUROC**: One-vs-rest AUROC for multi-class; standard AUROC for binary

#### 5.3.3 Regression Metrics
For numeric target variables:
- **MAE**: Mean Absolute Error, robust to outliers
- **RMSE**: Root Mean Squared Error, penalizes large errors
- **MSE**: Mean Squared Error
- **R²**: Coefficient of determination, proportion of variance explained

#### 5.3.4 Model Selection

We employ diverse model families to ensure robustness of conclusions:

**Classification Models**:
- Decision Tree Classifier: Non-parametric, interpretable, captures non-linear relationships
- Random Forest Classifier: Ensemble method, reduces variance, provides feature importances

**Regression Models**:
- Linear Regression: Parametric baseline, assumes linear relationships
- Random Forest Regressor: Non-parametric, captures non-linearities

Model diversity ensures that conclusions about dataset similarity are not artifacts of a particular model's inductive bias.

## 6. Feature Importance Analysis

### 6.1 Importance Extraction

Feature importance measures the contribution of each feature to model predictions. We employ two extraction methods depending on model type:

**Tree-Based Models**: Extract `feature_importances_` attribute, representing mean decrease in impurity across all splits involving each feature.

**Other Models**: Compute permutation importance on training data:
1. Establish baseline performance
2. For each feature, randomly shuffle its values and measure performance degradation
3. Repeat 10 times and average
4. Report mean importance across repetitions

### 6.2 Ranking Similarity Metrics

We compare feature importance rankings between models trained on real versus synthetic data using multiple complementary metrics:

#### 6.2.1 Rank Correlation Metrics

**Kendall's Tau (τ)**:
$$\tau = \frac{(\text{concordant pairs}) - (\text{discordant pairs})}{\binom{n}{2}}$$

Measures ordinal association between rankings. Ranges from -1 (perfect disagreement) to +1 (perfect agreement).

**Weighted Tau**: Assigns higher weight to disagreements among top-ranked features, reflecting that accurate ranking of important features matters more than ranking of unimportant ones.

**Spearman's ρ**: Pearson correlation applied to ranks. More sensitive to differences in the tails of rankings than Kendall's τ.

#### 6.2.2 Top-k Similarity Metrics

**NDCG (Normalized Discounted Cumulative Gain)**: Measures ranking quality with position-dependent discounting. Emphasizes agreement on highly-ranked features.

**Jaccard@k**: Overlap coefficient of top-k features: $J_k = \frac{|R_k \cap S_k|}{|R_k \cup S_k|}$

#### 6.2.3 Rank-Biased Overlap (RBO)

$$RBO_{p}(R, S) = (1-p) \sum_{d=1}^{\infty} p^{d-1} \cdot A_d$$

where $A_d$ is the overlap at depth $d$. The parameter $p$ determines weight concentration on top ranks. RBO handles rankings of different lengths and provides a tunable top-weightedness.

#### 6.2.4 Rank-Based Agreement Scores

**Cohen's Kappa**: Agreement coefficient correcting for chance, treating ranks as categorical labels.

**R² on Raw Importances**: Coefficient of determination between real and synthetic importance vectors, measuring linear agreement in importance magnitudes (not just orderings).

### 6.3 Deterministic Tie-Breaking

When features have identical importance values, we apply deterministic tie-breaking by feature name (alphabetical ordering). This ensures reproducibility—identical inputs always produce identical rankings—unlike random tie-breaking approaches that introduce spurious variance.

## 7. Experimental Design

### 7.1 Train-Test Split Strategy

Each experiment employs repeated random splits:
- **Test Size**: 20% of data reserved for evaluation
- **Repetitions**: 5 independent splits per configuration (configurable)
- **Seed Management**: Base seed + repetition index ensures reproducibility while providing variance estimates

### 7.2 Cross-Validation Considerations

For each target column $t$:
1. Remove $t$ from feature set
2. Adjust categorical/numeric feature lists accordingly
3. Split both real and synthetic data using identical random seeds
4. Fit separate preprocessors on real and synthetic training sets
5. Apply cross-transformations for TRTS/TSTR evaluation

### 7.3 Handling Missing Target Values

Perturbations introducing missingness may affect target columns. We handle this by:
1. Filtering rows with missing target values
2. Proceeding with experiment if ≥10 samples remain in both datasets
3. Recording filtered sample sizes in results metadata

## 8. Statistical Analysis

### 8.1 Aggregation Strategy

Results are aggregated hierarchically:
1. **Within-target**: Across repetitions and models for each target column
2. **Across-targets**: Median aggregation provides robust summary statistic

The median is preferred over the mean due to its robustness to outliers that may arise from numerical instabilities in edge cases.

### 8.2 Uncertainty Quantification

**Bootstrap Confidence Intervals**:
- 500 bootstrap resamples of metric values
- Report 2.5th and 97.5th percentiles for 95% CI
- Apply to median estimate for each aggregated metric

### 8.3 Monotonicity Assessment

For perturbation intensity studies, we compute Spearman correlation between perturbation level and metric response. High positive correlation indicates the metric reliably increases with perturbation intensity; high negative correlation indicates reliable decrease. Near-zero correlation suggests insensitivity to that perturbation type.

### 8.4 Stability Assessment

Metric stability is assessed via:
- **Standard Deviation**: Across repetitions/targets
- **Interquartile Range (IQR)**: Robust dispersion measure

Low dispersion indicates consistent, reliable metrics; high dispersion suggests sensitivity to random sampling or model initialization.

## 9. Implementation and Reproducibility

### 9.1 Software Environment

All experiments are implemented in Python 3.12+ with the following core dependencies:
- NumPy ≥1.24.0: Numerical computations
- Pandas ≥2.0.0: Data manipulation
- Scikit-learn ≥1.3.0: Machine learning models and preprocessing
- SciPy ≥1.10.0: Statistical functions and distance metrics
- Statsmodels ≥0.14.0: Mixed-effects models

Optional dependencies for ranking metrics:
- rbo ≥0.1.0: Rank-Biased Overlap computation
- python-Levenshtein, jellyfish: String similarity metrics

### 9.2 Configuration Management

All experiment parameters are centralized in configuration files:
- `configs/experiment.py`: Datasets, models, seeds, split parameters
- `configs/perturbations.py`: Perturbation specifications

This separation of configuration from code ensures experiments are fully specified by declarative parameter files.

### 9.3 Output Artifacts

Each experiment produces:
1. **Full metrics JSON**: Complete results for all perturbations, targets, repetitions
2. **Metadata JSON**: Experiment configuration, random seeds, timestamps
3. **Split files**: Train/test indices for exact reproduction

### 9.4 Execution

The complete experiment pipeline is executed via a single entry point:

```bash
python -B scripts/run_all.py
```

The `-B` flag prevents bytecode caching, ensuring configuration changes take immediate effect.

## 10. Limitations and Considerations

### 10.1 Methodological Limitations

1. **Feature Type Specification**: Manual declaration of feature types may introduce errors; automated type inference could improve usability but risks misclassification.

2. **Model Selection Bias**: Conclusions may be influenced by the specific model families employed. We mitigate this through model diversity but cannot guarantee generalization to all model types.

3. **Perturbation Coverage**: While our perturbation suite covers major distribution shift categories, real-world distribution shifts may exhibit combinations or forms not captured.

### 10.2 Computational Considerations

1. **Permutation Importance**: Computationally expensive for large feature sets or datasets. We use n_repeats=10 as a balance between precision and runtime.

2. **Bootstrap Confidence Intervals**: 500 resamples provide reasonable precision for 95% CI; more extreme quantiles would require more resamples.

3. **MMD/Energy Distance**: Quadratic complexity in sample size; may become prohibitive for very large datasets.

## 11. Conclusion

This methodology provides a comprehensive framework for evaluating dataset similarity through multiple complementary lenses: distributional distances, discriminative power, predictive equivalence, and feature importance alignment. The modular, configuration-driven implementation ensures reproducibility and facilitates extension to new perturbation types, metrics, or datasets.

The multi-metric approach recognizes that "similarity" is not a unitary concept—datasets may be similar in marginal distributions but differ in multivariate structure, or exhibit predictive equivalence despite distributional differences. By reporting diverse metrics with uncertainty quantification, we enable nuanced assessment of dataset similarity tailored to specific downstream use cases.
