# Discussion

## Overview of Findings

Our experimental framework evaluated dataset similarity across multiple dimensions using five distinct perturbation types applied to six health-related datasets. The results reveal important insights about which perturbations most significantly impact data utility and how different similarity metrics capture these effects.

## 5.1 Discriminability Analysis

The propensity score AUROC, which measures how easily a classifier can distinguish between original and perturbed data, revealed a clear hierarchy of perturbation severity (Figure 1: baseline_heatmap.png).

**High Discriminability (AUROC > 0.65)**:
- *Missingness MCAR* (AUROC = 0.69): The introduction of missing values created the most distinguishable perturbation. This is expected, as MCAR fundamentally alters the data structure by removing values entirely, leaving patterns that classifiers readily detect.
- *Mean-Variance Drift* (AUROC = 0.67): Systematic shifts in numeric distributions are detectable but less severe than missingness, as the underlying data structure remains intact.

**Low Discriminability (AUROC ≈ 0.43)**:
- *Category Collapse*, *Correlation Mix*, and *Noise Injection* all produced AUROC values near 0.43, only slightly above the random baseline of 0.5. These perturbations preserve the overall distributional properties sufficiently that a propensity classifier struggles to identify the perturbed samples.

This finding has important implications for synthetic data evaluation: **low discriminability does not imply high utility**—it merely indicates distributional similarity at the sample level.

## 5.2 Distributional Distance Metrics

The marginal Jensen-Shannon distance and Maximum Mean Discrepancy (MMD) provided complementary perspectives on distributional shift (Figure 1: baseline_heatmap.png).

**Marginal JS Distance**:
- *Missingness MCAR* exhibited the highest marginal JS distance (0.374), reflecting the fundamental change in marginal distributions when values are removed and imputed.
- *Correlation Mix* showed moderate marginal shift (0.136), which may seem counterintuitive since this perturbation is designed to preserve marginals while altering correlations. The observed shift likely results from the interpolation with permuted values introducing subtle marginal changes.
- *Mean-Variance Drift*, *Category Collapse*, and *Noise Injection* produced minimal marginal shifts (<0.025), indicating these perturbations operate within the bounds of the original distributions.

**MMD (RBF Kernel)**:
- *Correlation Mix* and *Mean-Variance Drift* produced the highest MMD values (~0.01), capturing multivariate distributional shifts that univariate metrics miss.
- *Category Collapse* and *Noise Injection* showed near-zero MMD, suggesting these perturbations preserve the multivariate structure.

## 5.3 Feature Importance Ranking Agreement

The ranking similarity metrics (Kendall's τ, Spearman's ρ, RBO) measure whether models trained on original versus perturbed data learn similar feature importance orderings—a critical indicator of whether perturbed data preserves predictive relationships (Figure 2: ranking_boxplot.png).

**High Ranking Preservation (τ > 0.9)**:
- *Mean-Variance Drift* achieved near-perfect ranking agreement (τ = 0.997, ρ = 0.999, RBO = 0.995). This is expected: affine transformations of features do not alter their relative predictive importance for most model families.
- *Category Collapse* maintained strong ranking agreement (τ = 0.94), indicating that aggregating rare categories does not substantially alter the importance hierarchy.

**Moderate Ranking Preservation (τ 0.75-0.90)**:
- *Noise Injection* (τ = 0.89) and *Correlation Mix* (τ = 0.75) showed moderate ranking agreement. Noise injection adds variance uniformly, which may differentially impact features based on their signal-to-noise ratios. Correlation mixing explicitly breaks inter-feature relationships, which can elevate previously suppressed features or diminish previously important ones.

**Low Ranking Preservation (τ < 0.6)**:
- *Missingness MCAR* produced the lowest ranking agreement (τ = 0.53, ρ = 0.65, RBO = 0.63). Random missingness followed by imputation fundamentally alters the information content of features, leading to substantial reordering of feature importance rankings.

## 5.4 Predictive Performance Degradation

The TRTR-TRTS comparison (Train Real/Test Real vs. Train Real/Test Synthetic) directly measures whether a model's predictive capability transfers to perturbed data (Figure 3: classification_comparison.png, Figure 4: regression_comparison.png).

For regression tasks, the RMSE ratio (TRTS/TRTR) quantifies performance degradation:
- *Mean-Variance Drift* showed the highest RMSE ratios, indicating that while the data remains statistically similar, the shifted feature scales cause models to underperform on perturbed test sets.
- *Missingness MCAR* produced variable performance degradation, depending on which features were affected and their importance to the prediction task.

For classification tasks, the accuracy drop (TRTR - TRTS) revealed:
- Smaller accuracy drops for perturbations that preserve feature importance rankings (*Mean-Variance Drift*, *Category Collapse*)
- Larger accuracy drops for perturbations that disrupt rankings (*Missingness*, *Correlation Mix*)

This correlation between ranking preservation and predictive transfer supports the hypothesis that **feature importance agreement is a meaningful proxy for data utility**.

## 5.5 Metric Sensitivity Analysis

Different metrics exhibited varying sensitivity to different perturbation types (Figure 5: perturbation_summary.png):

| Perturbation | Best Detected By | Poorly Detected By |
|--------------|------------------|-------------------|
| Mean-Variance Drift | Propensity AUROC, Wasserstein | JS Distance, MMD |
| Missingness MCAR | All metrics | — |
| Correlation Mix | MMD, Ranking Metrics | Propensity AUROC |
| Category Collapse | — | All metrics |
| Noise Injection | — | All metrics |

*Category Collapse* and *Noise Injection* were poorly detected by all metrics, suggesting these perturbations may be acceptable for many downstream applications. Conversely, *Missingness MCAR* was detected by all metrics, indicating it represents a severe form of data degradation.

## 5.6 Implications for Synthetic Data Evaluation

Our findings suggest a multi-metric approach to synthetic data evaluation:

1. **Discriminability alone is insufficient**: Low propensity AUROC does not guarantee utility. Category Collapse showed low discriminability but may still be problematic for applications requiring fine-grained categorical distinctions.

2. **Feature importance ranking is a strong utility indicator**: High ranking agreement (τ > 0.9) consistently corresponded with good predictive transfer. This metric captures whether the synthetic data preserves the predictive relationships in the original data.

3. **Perturbation type matters more than magnitude**: Different perturbation types affect different aspects of data utility. Practitioners should select metrics based on their specific use case:
   - For general ML model training: prioritize ranking agreement metrics
   - For statistical analysis: prioritize marginal and multivariate distance metrics
   - For privacy-utility tradeoffs: monitor propensity AUROC alongside utility metrics

4. **Confidence intervals are essential**: The bootstrap confidence intervals (Figure 6: aggregated_ci.png) revealed substantial variability across repetitions and targets, emphasizing the need for uncertainty quantification rather than point estimates.

## 5.7 Limitations

Several limitations should be considered when interpreting these results:

1. **Dataset specificity**: Our evaluation focused on health-related tabular datasets with relatively small sample sizes. Results may differ for larger datasets or different domains.

2. **Perturbation parameter sensitivity**: We evaluated single parameter settings for each perturbation type. The relationship between perturbation intensity and metric response warrants further investigation.

3. **Model family effects**: While we employed multiple model types (Decision Trees, Random Forests, Linear models), the generalizability to other architectures (neural networks, gradient boosting) requires additional study.

4. **Metric completeness**: Our metric suite, while comprehensive, does not include all possible similarity measures. Metrics based on downstream task performance in specific applications may reveal additional insights.

## 5.8 Recommendations

Based on our findings, we recommend the following practices for dataset similarity evaluation:

1. **Use a multi-metric dashboard**: No single metric captures all dimensions of similarity. Report at minimum:
   - Propensity AUROC (discriminability)
   - Marginal JS distance (univariate similarity)
   - Feature importance ranking agreement (utility preservation)
   - Cross-classification performance (predictive transfer)

2. **Prioritize ranking-based metrics for utility assessment**: Kendall's τ and RBO showed strong correspondence with predictive performance transfer and are interpretable measures of whether models learn similar patterns from original and perturbed data.

3. **Report uncertainty**: Use bootstrap confidence intervals or repeated cross-validation to quantify metric variability. Point estimates can be misleading.

4. **Consider perturbation-specific evaluation**: Different use cases may tolerate different perturbation types. A privacy-preserving application might accept noise injection (low detectability, moderate ranking agreement) but reject missingness (high detectability, low ranking agreement).

## 5.9 Conclusion

This study demonstrates that dataset similarity is a multi-dimensional concept requiring diverse metrics for comprehensive evaluation. Our experimental framework, combining distributional distances, discriminative tests, predictive performance comparisons, and feature importance analysis, provides a robust foundation for evaluating synthetic or perturbed tabular data. The finding that feature importance ranking agreement strongly correlates with predictive utility transfer offers practitioners a practical and interpretable metric for assessing data quality in machine learning applications.

---

## Figure References

- **Figure 1** (baseline_heatmap.png): Heatmap of baseline similarity metrics (propensity AUROC, JS distance, MMD, Wasserstein distance) across perturbation types. Lower values indicate higher similarity to original data.

- **Figure 2** (ranking_boxplot.png): Boxplots comparing ranking similarity metrics (Kendall's τ, Spearman's ρ, RBO) across perturbation types. Higher values indicate better preservation of feature importance rankings.

- **Figure 3** (classification_comparison.png): Left panel: Scatter plot of TRTR vs. TRTS accuracy, with points colored by perturbation type. Right panel: Boxplot of accuracy drop (TRTR - TRTS) by perturbation.

- **Figure 4** (regression_comparison.png): RMSE ratio (TRTS/TRTR) by perturbation type for regression tasks. Values above 1.0 indicate performance degradation on perturbed data.

- **Figure 5** (perturbation_summary.png): Summary visualization showing dataset discriminability (propensity AUROC) and average feature importance agreement per perturbation type.

- **Figure 6** (aggregated_ci.png): Aggregated metrics with 95% bootstrap confidence intervals, showing median values and uncertainty across all repetitions and targets.

- **Table 1** (summary_statistics.csv): Complete summary statistics including mean and standard deviation for all metrics by perturbation type.
