"""Transform experiment results to CSV and generate visualizations."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns


def load_results(results_dir: str = "results") -> dict:
    """Load all full_metrics JSON files from results directory."""
    results_path = Path(results_dir)
    all_results = {}

    for json_file in results_path.glob("full_metrics_*.json"):
        dataset_name = json_file.stem.replace("full_metrics_", "")
        with open(json_file) as f:
            all_results[dataset_name] = json.load(f)

    return all_results


def results_to_baseline_csv(all_results: dict) -> pd.DataFrame:
    """Convert baseline metrics to a flat CSV-friendly DataFrame."""
    rows = []

    for dataset_name, dataset_results in all_results.items():
        for result in dataset_results:
            row = {
                "dataset": dataset_name,
                "perturbation": result["perturbation"],
            }
            # Add baseline metrics
            if "baseline_metrics" in result:
                for metric, value in result["baseline_metrics"].items():
                    row[metric] = value
            # Legacy format support
            elif "propensity_auroc" in result:
                row["propensity_auroc"] = result.get("propensity_auroc")
                row["propensity_brier"] = result.get("propensity_brier")

            rows.append(row)

    return pd.DataFrame(rows)


def results_to_target_csv(all_results: dict) -> pd.DataFrame:
    """Convert target-level TRTR/TRTS/TSTS/TSTR metrics to DataFrame."""
    rows = []

    for dataset_name, dataset_results in all_results.items():
        for result in dataset_results:
            perturbation = result["perturbation"]
            target_results = result.get("target_results", {})

            for target_col, experiments in target_results.items():
                for exp in experiments:
                    row = {
                        "dataset": dataset_name,
                        "perturbation": perturbation,
                        "target": target_col,
                        "rep": exp.get("rep"),
                        "model": exp.get("model"),
                        "task_type": exp.get("task_type", "unknown"),
                    }

                    # Extract TRTR/TRTS/TSTS/TSTR metrics
                    for metric_type in ["trtr", "trts", "tsts", "tstr"]:
                        metric_val = exp.get(metric_type)
                        if isinstance(metric_val, dict):
                            for sub_metric, val in metric_val.items():
                                row[f"{metric_type}_{sub_metric}"] = val
                        else:
                            row[metric_type] = metric_val

                    rows.append(row)

    return pd.DataFrame(rows)


def results_to_ranking_csv(all_results: dict) -> pd.DataFrame:
    """Convert ranking similarity scores to DataFrame."""
    rows = []

    for dataset_name, dataset_results in all_results.items():
        for result in dataset_results:
            perturbation = result["perturbation"]
            target_results = result.get("target_results", {})

            for target_col, experiments in target_results.items():
                for exp in experiments:
                    ranking_scores = exp.get("ranking_scores", {})
                    if not ranking_scores:
                        continue

                    row = {
                        "dataset": dataset_name,
                        "perturbation": perturbation,
                        "target": target_col,
                        "rep": exp.get("rep"),
                        "model": exp.get("model"),
                    }
                    row.update(ranking_scores)
                    rows.append(row)

    return pd.DataFrame(rows)


def results_to_aggregated_csv(all_results: dict) -> pd.DataFrame:
    """Convert aggregated metrics with CI to DataFrame."""
    rows = []

    for dataset_name, dataset_results in all_results.items():
        for result in dataset_results:
            aggregated = result.get("aggregated", {})
            if not aggregated:
                continue

            for metric, stats in aggregated.items():
                if isinstance(stats, dict):
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "perturbation": result["perturbation"],
                            "metric": metric,
                            "median": stats.get("median"),
                            "ci_lower": stats.get("ci_lower"),
                            "ci_upper": stats.get("ci_upper"),
                            "n": stats.get("n"),
                        }
                    )

    return pd.DataFrame(rows)


def save_csvs(all_results: dict, output_dir: str = "results") -> dict:
    """Save all CSV files and return DataFrames."""
    output_path = Path(output_dir)

    # Create DataFrames
    df_baseline = results_to_baseline_csv(all_results)
    df_target = results_to_target_csv(all_results)
    df_ranking = results_to_ranking_csv(all_results)
    df_aggregated = results_to_aggregated_csv(all_results)

    # Save to CSV
    df_baseline.to_csv(output_path / "baseline_metrics.csv", index=False)
    df_target.to_csv(output_path / "target_metrics.csv", index=False)
    df_ranking.to_csv(output_path / "ranking_metrics.csv", index=False)
    df_aggregated.to_csv(output_path / "aggregated_metrics.csv", index=False)

    print(f"Saved CSVs to {output_path}:")
    print(f"  - baseline_metrics.csv ({len(df_baseline)} rows)")
    print(f"  - target_metrics.csv ({len(df_target)} rows)")
    print(f"  - ranking_metrics.csv ({len(df_ranking)} rows)")
    print(f"  - aggregated_metrics.csv ({len(df_aggregated)} rows)")

    return {
        "baseline": df_baseline,
        "target": df_target,
        "ranking": df_ranking,
        "aggregated": df_aggregated,
    }


def plot_baseline_heatmap(df: pd.DataFrame, output_dir: str = "results"):
    """Create heatmap of baseline metrics by perturbation and dataset."""
    output_path = Path(output_dir) / "figures"
    output_path.mkdir(exist_ok=True)

    # Select numeric columns for heatmap
    metric_cols = [c for c in df.columns if c not in ["dataset", "perturbation"]]

    if not metric_cols:
        print("No numeric metrics found for heatmap")
        return

    # Pivot for heatmap (average across datasets)
    df_mean = df.groupby("perturbation")[metric_cols].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df_mean, annot=True, fmt=".3f", cmap="RdYlGn_r", ax=ax)
    ax.set_title("Baseline Metrics by Perturbation (lower = more similar)")
    plt.tight_layout()
    plt.savefig(output_path / "baseline_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved: {output_path / 'baseline_heatmap.png'}")


def plot_ranking_comparison(df: pd.DataFrame, output_dir: str = "results"):
    """Create boxplots comparing ranking metrics across perturbations."""
    output_path = Path(output_dir) / "figures"
    output_path.mkdir(exist_ok=True)

    ranking_cols = [
        "kendalltau",
        "weightedtau",
        "spearmanr",
        "ndcg_score",
        "rbo",
        "jaccard_top_5",
        "jaccard_top_10",
    ]
    available_cols = [c for c in ranking_cols if c in df.columns]

    if not available_cols:
        print("No ranking metrics found for comparison")
        return

    # Melt for boxplot
    df_melt = df.melt(
        id_vars=["dataset", "perturbation", "target", "rep", "model"],
        value_vars=available_cols,
        var_name="metric",
        value_name="score",
    ).dropna()

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df_melt, x="perturbation", y="score", hue="metric", ax=ax)
    ax.set_title("Ranking Similarity Metrics by Perturbation")
    ax.set_ylabel("Score (higher = more similar)")
    ax.set_xlabel("Perturbation Type")
    plt.xticks(rotation=45, ha="right")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path / "ranking_boxplot.png", dpi=150)
    plt.close()
    print(f"Saved: {output_path / 'ranking_boxplot.png'}")


def plot_trts_comparison(df: pd.DataFrame, output_dir: str = "results"):
    """Compare TRTR vs TRTS performance (classification and regression separately)."""
    output_path = Path(output_dir) / "figures"
    output_path.mkdir(exist_ok=True)

    # Classification metrics
    if "trtr_accuracy" in df.columns and "trts_accuracy" in df.columns:
        df_class = df[df["task_type"] == "classification"].dropna(
            subset=["trtr_accuracy", "trts_accuracy"]
        )

        if len(df_class) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Accuracy comparison
            for perturbation in df_class["perturbation"].unique():
                subset = df_class[df_class["perturbation"] == perturbation]
                axes[0].scatter(
                    subset["trtr_accuracy"],
                    subset["trts_accuracy"],
                    label=perturbation,
                    alpha=0.6,
                )
            axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect similarity")
            axes[0].set_xlabel("TRTR Accuracy")
            axes[0].set_ylabel("TRTS Accuracy")
            axes[0].set_title("Train Real/Test Real vs Train Real/Test Synthetic")
            axes[0].legend(fontsize=8)

            # TRTR - TRTS difference by perturbation
            df_class["accuracy_diff"] = (
                df_class["trtr_accuracy"] - df_class["trts_accuracy"]
            )
            sns.boxplot(data=df_class, x="perturbation", y="accuracy_diff", ax=axes[1])
            axes[1].axhline(0, color="red", linestyle="--", alpha=0.5)
            axes[1].set_title("Accuracy Drop (TRTR - TRTS)")
            axes[1].set_ylabel("Accuracy Difference")
            plt.xticks(rotation=45, ha="right")

            plt.tight_layout()
            plt.savefig(output_path / "classification_comparison.png", dpi=150)
            plt.close()
            print(f"Saved: {output_path / 'classification_comparison.png'}")

    # Regression metrics
    if "trtr_rmse" in df.columns and "trts_rmse" in df.columns:
        df_reg = df[df["task_type"] == "regression"].dropna(
            subset=["trtr_rmse", "trts_rmse"]
        )

        if len(df_reg) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            df_reg["rmse_ratio"] = df_reg["trts_rmse"] / df_reg["trtr_rmse"].replace(
                0, np.nan
            )
            sns.boxplot(data=df_reg, x="perturbation", y="rmse_ratio", ax=ax)
            ax.axhline(
                1, color="red", linestyle="--", alpha=0.5, label="Equal performance"
            )
            ax.set_title("RMSE Ratio (TRTS/TRTR) - Higher = Worse on Perturbed Data")
            ax.set_ylabel("RMSE Ratio")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(output_path / "regression_comparison.png", dpi=150)
            plt.close()
            print(f"Saved: {output_path / 'regression_comparison.png'}")


def plot_tstr_comparison(df: pd.DataFrame, output_dir: str = "results"):
    """Compare TRTR vs TSTR performance (Utility)."""
    output_path = Path(output_dir) / "figures"
    output_path.mkdir(exist_ok=True)

    # Classification metrics
    if "trtr_accuracy" in df.columns and "tstr_accuracy" in df.columns:
        df_class = df[df["task_type"] == "classification"].dropna(
            subset=["trtr_accuracy", "tstr_accuracy"]
        )

        if len(df_class) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Accuracy comparison
            for perturbation in df_class["perturbation"].unique():
                subset = df_class[df_class["perturbation"] == perturbation]
                axes[0].scatter(
                    subset["trtr_accuracy"],
                    subset["tstr_accuracy"],
                    label=perturbation,
                    alpha=0.6,
                )
            axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect utility")
            axes[0].set_xlabel("TRTR Accuracy (Real Model)")
            axes[0].set_ylabel("TSTR Accuracy (Synthetic Model)")
            axes[0].set_title("Utility: Train Real vs Train Synthetic (Test Real)")
            axes[0].legend(fontsize=8)

            # TRTR - TSTR difference by perturbation
            df_class["utility_loss"] = (
                df_class["trtr_accuracy"] - df_class["tstr_accuracy"]
            )
            sns.boxplot(data=df_class, x="perturbation", y="utility_loss", ax=axes[1])
            axes[1].axhline(0, color="red", linestyle="--", alpha=0.5)
            axes[1].set_title("Utility Loss (TRTR - TSTR)")
            axes[1].set_ylabel("Accuracy Drop")
            plt.xticks(rotation=45, ha="right")

            plt.tight_layout()
            plt.savefig(output_path / "utility_classification_comparison.png", dpi=150)
            plt.close()
            print(f"Saved: {output_path / 'utility_classification_comparison.png'}")

    # Regression metrics
    if "trtr_rmse" in df.columns and "tstr_rmse" in df.columns:
        df_reg = df[df["task_type"] == "regression"].dropna(
            subset=["trtr_rmse", "tstr_rmse"]
        )

        if len(df_reg) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            df_reg["utility_rmse_ratio"] = df_reg["tstr_rmse"] / df_reg[
                "trtr_rmse"
            ].replace(0, np.nan)
            sns.boxplot(data=df_reg, x="perturbation", y="utility_rmse_ratio", ax=ax)
            ax.axhline(
                1, color="red", linestyle="--", alpha=0.5, label="Equal performance"
            )
            ax.set_title("Utility RMSE Ratio (TSTR/TRTR) - Higher = Worse Model")
            ax.set_ylabel("RMSE Ratio")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(output_path / "utility_regression_comparison.png", dpi=150)
            plt.close()
            print(f"Saved: {output_path / 'utility_regression_comparison.png'}")


def plot_fidelity_vs_utility(df: pd.DataFrame, output_dir: str = "results"):
    """Compare Fidelity (TRTR-TRTS) vs Utility (TRTR-TSTR) drops."""
    output_path = Path(output_dir) / "figures"
    output_path.mkdir(exist_ok=True)

    # Classification
    if (
        "trtr_accuracy" in df.columns
        and "trts_accuracy" in df.columns
        and "tstr_accuracy" in df.columns
    ):
        df_class = df[df["task_type"] == "classification"].copy()
        df_class = df_class.dropna(
            subset=["trtr_accuracy", "trts_accuracy", "tstr_accuracy"]
        )

        if len(df_class) > 0:
            df_class["fidelity_drop"] = (
                df_class["trtr_accuracy"] - df_class["trts_accuracy"]
            )
            df_class["utility_drop"] = (
                df_class["trtr_accuracy"] - df_class["tstr_accuracy"]
            )

            fig, ax = plt.subplots(figsize=(10, 6))

            for perturb in df_class["perturbation"].unique():
                subset = df_class[df_class["perturbation"] == perturb]
                ax.scatter(
                    subset["fidelity_drop"],
                    subset["utility_drop"],
                    label=perturb,
                    alpha=0.6,
                )

            # Add y=x line
            min_val = min(
                df_class["fidelity_drop"].min(), df_class["utility_drop"].min()
            )
            max_val = max(
                df_class["fidelity_drop"].max(), df_class["utility_drop"].max()
            )
            ax.plot(
                [min_val, max_val], [min_val, max_val], "k--", alpha=0.3, label="y=x"
            )

            # Calculate correlation
            if len(df_class) > 1:
                r, p = st.pearsonr(df_class["fidelity_drop"], df_class["utility_drop"])
                ax.text(
                    0.05,
                    0.95,
                    f"Pearson r={r:.3f} (p={p:.3f})",
                    transform=ax.transAxes,
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            ax.set_xlabel("Fidelity Loss (TRTR - TRTS)\n(Does it look real?)")
            ax.set_ylabel("Utility Loss (TRTR - TSTR)\n(Does it train well?)")
            ax.set_title("Fidelity vs Utility Correlation (Classification)")
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_path / "fidelity_vs_utility_classification.png", dpi=150)
            plt.close()
            print(f"Saved: {output_path / 'fidelity_vs_utility_classification.png'}")

    # Regression
    if (
        "trtr_rmse" in df.columns
        and "trts_rmse" in df.columns
        and "tstr_rmse" in df.columns
    ):
        df_reg = df[df["task_type"] == "regression"].copy()
        df_reg = df_reg.dropna(subset=["trtr_rmse", "trts_rmse", "tstr_rmse"])

        if len(df_reg) > 0:
            # Use ratios for regression
            df_reg["fidelity_ratio"] = df_reg["trts_rmse"] / df_reg[
                "trtr_rmse"
            ].replace(0, np.nan)
            df_reg["utility_ratio"] = df_reg["tstr_rmse"] / df_reg["trtr_rmse"].replace(
                0, np.nan
            )

            # Filter outliers for plotting
            df_reg = df_reg[
                (df_reg["fidelity_ratio"] < 10) & (df_reg["utility_ratio"] < 10)
            ]

            fig, ax = plt.subplots(figsize=(10, 6))

            for perturb in df_reg["perturbation"].unique():
                subset = df_reg[df_reg["perturbation"] == perturb]
                ax.scatter(
                    subset["fidelity_ratio"],
                    subset["utility_ratio"],
                    label=perturb,
                    alpha=0.6,
                )

            ax.set_xlabel("Fidelity RMSE Ratio (TRTS / TRTR)")
            ax.set_ylabel("Utility RMSE Ratio (TSTR / TRTR)")
            ax.set_title("Fidelity vs Utility Correlation (Regression)")
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_path / "fidelity_vs_utility_regression.png", dpi=150)
            plt.close()
            print(f"Saved: {output_path / 'fidelity_vs_utility_regression.png'}")


def plot_aggregated_with_ci(df: pd.DataFrame, output_dir: str = "results"):
    """Plot aggregated metrics with confidence intervals."""
    output_path = Path(output_dir) / "figures"
    output_path.mkdir(exist_ok=True)

    if df.empty:
        print("No aggregated data for CI plot")
        return

    # Filter to key metrics
    key_metrics = ["accuracy", "balanced_accuracy", "kendalltau", "rbo", "r2"]
    df_filtered = df[df["metric"].isin(key_metrics)]

    if df_filtered.empty:
        # Try with available metrics
        df_filtered = df.head(50)

    # Aggregate across datasets (mean of medians, propagate CI)
    df_agg = (
        df_filtered.groupby(["perturbation", "metric"])
        .agg({"median": "mean", "ci_lower": "mean", "ci_upper": "mean", "n": "sum"})
        .reset_index()
    )

    if df_agg.empty:
        print("No data after aggregation for CI plot")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    # Create grouped bar plot with error bars
    perturbations = df_agg["perturbation"].unique()
    metrics = df_agg["metric"].unique()
    x = np.arange(len(perturbations))
    width = 0.8 / max(len(metrics), 1)

    for i, metric in enumerate(metrics):
        metric_data = df_agg[df_agg["metric"] == metric].copy()
        # Create a mapping for this metric
        metric_dict = metric_data.set_index("perturbation").to_dict("index")

        medians = [metric_dict.get(p, {}).get("median", np.nan) for p in perturbations]
        ci_lowers = [
            metric_dict.get(p, {}).get("ci_lower", np.nan) for p in perturbations
        ]
        ci_uppers = [
            metric_dict.get(p, {}).get("ci_upper", np.nan) for p in perturbations
        ]

        yerr_lower = [
            m - l if not np.isnan(m) and not np.isnan(l) else 0
            for m, l in zip(medians, ci_lowers)
        ]
        yerr_upper = [
            u - m if not np.isnan(m) and not np.isnan(u) else 0
            for m, u in zip(medians, ci_uppers)
        ]

        ax.bar(
            x + i * width,
            medians,
            width,
            label=metric,
            yerr=[yerr_lower, yerr_upper],
            capsize=2,
        )

    ax.set_xlabel("Perturbation")
    ax.set_ylabel("Score")
    ax.set_title("Aggregated Metrics with 95% Bootstrap CI")
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(perturbations, rotation=45, ha="right")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path / "aggregated_ci.png", dpi=150)
    plt.close()
    print(f"Saved: {output_path / 'aggregated_ci.png'}")


def plot_perturbation_summary(
    df_baseline: pd.DataFrame, df_ranking: pd.DataFrame, output_dir: str = "results"
):
    """Create summary visualization combining baseline and ranking metrics."""
    output_path = Path(output_dir) / "figures"
    output_path.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Propensity AUROC (discriminability)
    if "propensity_auroc" in df_baseline.columns:
        prop_data = df_baseline.groupby("perturbation")["propensity_auroc"].agg(
            ["mean", "std"]
        )
        prop_data = prop_data.sort_values("mean")

        axes[0].barh(
            prop_data.index, prop_data["mean"], xerr=prop_data["std"], capsize=3
        )
        axes[0].axvline(
            0.5, color="green", linestyle="--", alpha=0.7, label="Indistinguishable"
        )
        axes[0].set_xlabel("Propensity AUROC")
        axes[0].set_title(
            "Dataset Discriminability\n(0.5 = identical, 1.0 = perfectly distinguishable)"
        )
        axes[0].legend()

    # Right: Average ranking similarity
    if not df_ranking.empty:
        ranking_cols = ["kendalltau", "spearmanr", "rbo"]
        available = [c for c in ranking_cols if c in df_ranking.columns]

        if available:
            rank_mean = (
                df_ranking.groupby("perturbation")[available].mean().mean(axis=1)
            )
            rank_mean = rank_mean.sort_values(ascending=False)

            axes[1].barh(rank_mean.index, rank_mean.values)
            axes[1].set_xlabel("Average Ranking Similarity")
            axes[1].set_title(
                "Feature Importance Agreement\n(1.0 = identical rankings)"
            )

    plt.tight_layout()
    plt.savefig(output_path / "perturbation_summary.png", dpi=150)
    plt.close()
    print(f"Saved: {output_path / 'perturbation_summary.png'}")


def generate_summary_table(dfs: dict, output_dir: str = "results"):
    """Generate a summary statistics table."""
    output_path = Path(output_dir)

    summary_rows = []

    df_baseline = dfs["baseline"]
    df_ranking = dfs["ranking"]

    for perturbation in df_baseline["perturbation"].unique():
        row = {"perturbation": perturbation}

        # Baseline metrics
        baseline_subset = df_baseline[df_baseline["perturbation"] == perturbation]
        for col in ["propensity_auroc", "marginal_js_distance", "marginal_wasserstein"]:
            if col in baseline_subset.columns:
                row[f"{col}_mean"] = baseline_subset[col].mean()
                row[f"{col}_std"] = baseline_subset[col].std()

        # Ranking metrics
        if not df_ranking.empty:
            ranking_subset = df_ranking[df_ranking["perturbation"] == perturbation]
            for col in ["kendalltau", "spearmanr", "rbo"]:
                if col in ranking_subset.columns:
                    row[f"{col}_mean"] = ranking_subset[col].mean()
                    row[f"{col}_std"] = ranking_subset[col].std()

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(output_path / "summary_statistics.csv", index=False)
    print(f"Saved: {output_path / 'summary_statistics.csv'}")

    return df_summary


def compute_ranking_utility_correlation(
    df_target: pd.DataFrame, df_ranking: pd.DataFrame, output_dir: str = "results"
):
    """
    Compute correlation between ranking similarity metrics and utility degradation.

    This is the KEY VALIDATION: Do ranking metrics (Kendall's tau, RBO, etc.)
    predict actual utility loss (TRTR - TSTR or TRTR - TRTS performance)?
    """
    output_path = Path(output_dir) / "figures"
    output_path.mkdir(exist_ok=True)

    if df_ranking.empty or df_target.empty:
        print("Insufficient data for ranking-utility correlation analysis")
        return None

    # Merge ranking and target metrics on common keys
    merge_keys = ["dataset", "perturbation", "target", "rep", "model"]
    available_keys = [
        k for k in merge_keys if k in df_ranking.columns and k in df_target.columns
    ]

    if len(available_keys) < 3:
        print(f"Insufficient merge keys for correlation analysis: {available_keys}")
        return None

    df_merged = df_ranking.merge(df_target, on=available_keys, how="inner")

    if df_merged.empty:
        print("No matching records after merge")
        return None

    # Calculate utility degradation metrics
    # Fidelity: accuracy drop = TRTR - TRTS (positive = worse on perturbed data)
    if "trtr_accuracy" in df_merged.columns and "trts_accuracy" in df_merged.columns:
        df_merged["fidelity_accuracy_drop"] = (
            df_merged["trtr_accuracy"] - df_merged["trts_accuracy"]
        )

    # Utility: accuracy drop = TRTR - TSTR (positive = worse model from perturbed data)
    if "trtr_accuracy" in df_merged.columns and "tstr_accuracy" in df_merged.columns:
        df_merged["utility_accuracy_drop"] = (
            df_merged["trtr_accuracy"] - df_merged["tstr_accuracy"]
        )

    if (
        "trtr_balanced_accuracy" in df_merged.columns
        and "trts_balanced_accuracy" in df_merged.columns
    ):
        df_merged["fidelity_balanced_accuracy_drop"] = (
            df_merged["trtr_balanced_accuracy"] - df_merged["trts_balanced_accuracy"]
        )
    if (
        "trtr_balanced_accuracy" in df_merged.columns
        and "tstr_balanced_accuracy" in df_merged.columns
    ):
        df_merged["utility_balanced_accuracy_drop"] = (
            df_merged["trtr_balanced_accuracy"] - df_merged["tstr_balanced_accuracy"]
        )

    # For regression: RMSE increase ratio
    # Fidelity: TRTS/TRTR
    if "trtr_rmse" in df_merged.columns and "trts_rmse" in df_merged.columns:
        df_merged["fidelity_rmse_ratio"] = df_merged["trts_rmse"] / df_merged[
            "trtr_rmse"
        ].replace(0, np.nan)

    # Utility: TSTR/TRTR
    if "trtr_rmse" in df_merged.columns and "tstr_rmse" in df_merged.columns:
        df_merged["utility_rmse_ratio"] = df_merged["tstr_rmse"] / df_merged[
            "trtr_rmse"
        ].replace(0, np.nan)

    # Ranking metrics to correlate
    ranking_cols = [
        "kendalltau",
        "weightedtau",
        "spearmanr",
        "rbo",
        "ndcg_score",
        "jaccard_top_5",
        "jaccard_top_10",
    ]
    available_ranking = [c for c in ranking_cols if c in df_merged.columns]

    # Utility metrics to correlate against
    utility_cols = [
        "fidelity_accuracy_drop",
        "utility_accuracy_drop",
        "fidelity_balanced_accuracy_drop",
        "utility_balanced_accuracy_drop",
        "fidelity_rmse_ratio",
        "utility_rmse_ratio",
    ]
    available_utility = [c for c in utility_cols if c in df_merged.columns]

    if not available_ranking or not available_utility:
        print(
            f"Insufficient metrics for correlation: ranking={available_ranking}, utility={available_utility}"
        )
        return None

    # Compute correlation matrix
    corr_results = []
    for rank_metric in available_ranking:
        for util_metric in available_utility:
            subset = df_merged[[rank_metric, util_metric]].dropna()
            if len(subset) < 10:
                continue

            # Pearson correlation
            pearson_r, pearson_p = st.pearsonr(subset[rank_metric], subset[util_metric])
            # Spearman correlation
            spearman_r, spearman_p = st.spearmanr(
                subset[rank_metric], subset[util_metric]
            )

            corr_results.append(
                {
                    "ranking_metric": rank_metric,
                    "utility_metric": util_metric,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "n": len(subset),
                }
            )

    df_corr = pd.DataFrame(corr_results)
    df_corr.to_csv(Path(output_dir) / "ranking_utility_correlations.csv", index=False)
    print(f"Saved: {Path(output_dir) / 'ranking_utility_correlations.csv'}")

    # Print summary
    print("\n=== RANKING-UTILITY CORRELATION ANALYSIS ===")
    print(
        "(Negative correlation = higher ranking similarity predicts lower utility loss)"
    )
    print(df_corr.to_string(index=False))

    return df_corr, df_merged


def plot_ranking_vs_utility(df_merged: pd.DataFrame, output_dir: str = "results"):
    """
    Create scatter plots showing ranking metrics vs utility degradation.

    This visualizes the key hypothesis: ranking similarity predicts utility preservation.
    """
    output_path = Path(output_dir) / "figures"
    output_path.mkdir(exist_ok=True)

    # Primary comparison: Kendall's tau vs Utility accuracy drop (TRTR - TSTR)
    ranking_cols = ["kendalltau", "rbo", "spearmanr"]
    available_ranking = [c for c in ranking_cols if c in df_merged.columns]

    # Classification utility
    # Prefer utility (TSTR), fall back to fidelity (TRTS) if needed.
    acc_candidates = [
        "utility_accuracy_drop",
        "utility_balanced_accuracy_drop",
        "fidelity_accuracy_drop",
        "fidelity_balanced_accuracy_drop",
    ]
    acc_col = next((col for col in acc_candidates if col in df_merged.columns), None)
    acc_label_map = {
        "utility_accuracy_drop": "Accuracy Drop (TRTR - TSTR)",
        "utility_balanced_accuracy_drop": "Balanced Accuracy Drop (TRTR - TSTR)",
        "fidelity_accuracy_drop": "Accuracy Drop (TRTR - TRTS)",
        "fidelity_balanced_accuracy_drop": "Balanced Accuracy Drop (TRTR - TRTS)",
    }
    acc_label = acc_label_map.get(acc_col, f"Accuracy Drop ({acc_col})")

    if acc_col and available_ranking:
        df_class = df_merged.dropna(subset=[acc_col] + available_ranking[:1])

        if len(df_class) > 10:
            fig, axes = plt.subplots(
                1,
                min(len(available_ranking), 3),
                figsize=(5 * min(len(available_ranking), 3), 5),
            )
            if len(available_ranking) == 1:
                axes = [axes]

            for ax, rank_col in zip(axes, available_ranking[:3]):
                for perturb in df_class["perturbation"].unique():
                    subset = df_class[df_class["perturbation"] == perturb]
                    ax.scatter(
                        subset[rank_col],
                        subset[acc_col],
                        label=perturb,
                        alpha=0.6,
                        s=30,
                    )

                # Add trend line
                valid = df_class[[rank_col, acc_col]].dropna()
                if len(valid) > 5:
                    z = np.polyfit(valid[rank_col], valid[acc_col], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(
                        valid[rank_col].min(), valid[rank_col].max(), 100
                    )
                    ax.plot(x_line, p(x_line), "k--", alpha=0.5, label="Trend")

                    # Add correlation coefficient
                    r, pval = st.pearsonr(valid[rank_col], valid[acc_col])
                    print(
                        f"  Correlation ({rank_col} vs {acc_col}): r={r:.3f}, p={pval:.3f}"
                    )
                    ax.text(
                        0.05,
                        0.95,
                        f"r={r:.3f}, p={pval:.3f}",
                        transform=ax.transAxes,
                        fontsize=10,
                        va="top",
                    )

                ax.axhline(0, color="green", linestyle="--", alpha=0.3)
                ax.set_xlabel(f"{rank_col} (higher = more similar)")
                ax.set_ylabel(acc_label)
                ax.set_title("Ranking Similarity vs Utility Loss")

            axes[0].legend(fontsize=8, loc="upper right")
            plt.tight_layout()
            plt.savefig(output_path / "ranking_vs_utility_classification.png", dpi=150)
            plt.close()
            print(f"Saved: {output_path / 'ranking_vs_utility_classification.png'}")

    # Regression utility
    rmse_col = (
        "utility_rmse_ratio"
        if "utility_rmse_ratio" in df_merged.columns
        else "fidelity_rmse_ratio"
    )

    if rmse_col in df_merged.columns and available_ranking:
        df_reg = df_merged.dropna(subset=[rmse_col] + available_ranking[:1])
        # Filter extreme ratios
        df_reg = df_reg[(df_reg[rmse_col] > 0.1) & (df_reg[rmse_col] < 10)]

        if len(df_reg) > 10:
            fig, axes = plt.subplots(
                1,
                min(len(available_ranking), 3),
                figsize=(5 * min(len(available_ranking), 3), 5),
            )
            if len(available_ranking) == 1:
                axes = [axes]

            for ax, rank_col in zip(axes, available_ranking[:3]):
                for perturb in df_reg["perturbation"].unique():
                    subset = df_reg[df_reg["perturbation"] == perturb]
                    ax.scatter(
                        subset[rank_col],
                        subset[rmse_col],
                        label=perturb,
                        alpha=0.6,
                        s=30,
                    )

                # Add trend line
                valid = df_reg[[rank_col, rmse_col]].dropna()
                if len(valid) > 5:
                    z = np.polyfit(valid[rank_col], valid[rmse_col], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(
                        valid[rank_col].min(), valid[rank_col].max(), 100
                    )
                    ax.plot(x_line, p(x_line), "k--", alpha=0.5, label="Trend")

                    r, pval = st.pearsonr(valid[rank_col], valid[rmse_col])
                    print(
                        f"  Correlation ({rank_col} vs {rmse_col}): r={r:.3f}, p={pval:.3f}"
                    )
                    ax.text(
                        0.05,
                        0.95,
                        f"r={r:.3f}, p={pval:.3f}",
                        transform=ax.transAxes,
                        fontsize=10,
                        va="top",
                    )

                ax.axhline(
                    1, color="green", linestyle="--", alpha=0.3, label="No degradation"
                )
                ax.set_xlabel(f"{rank_col} (higher = more similar)")
                ax.set_ylabel(f"RMSE Ratio ({rmse_col})")
                ax.set_title("Ranking Similarity vs Utility Loss")

            axes[0].legend(fontsize=8, loc="upper right")
            plt.tight_layout()
            plt.savefig(output_path / "ranking_vs_utility_regression.png", dpi=150)
            plt.close()
            print(f"Saved: {output_path / 'ranking_vs_utility_regression.png'}")


def _slugify(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return safe.lower() if safe else "group"


def plot_grouped_correlation_heatmaps(
    df_corr: pd.DataFrame,
    group_col: str,
    output_dir: str = "results",
    value_col: str = "spearman_r",
):
    """Plot per-group heatmaps of ranking vs utility correlations."""
    if df_corr is None or df_corr.empty:
        print("No grouped correlation data for heatmaps")
        return

    output_path = Path(output_dir) / "figures"
    output_path.mkdir(exist_ok=True)

    for group_value, group_df in df_corr.groupby(group_col):
        pivot = group_df.pivot(
            index="ranking_metric",
            columns="utility_metric",
            values=value_col,
        )
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title(f"{group_col}: {group_value} ({value_col})")
        ax.set_xlabel("Utility metric")
        ax.set_ylabel("Ranking metric")
        plt.tight_layout()

        filename = (
            f"ranking_utility_heatmap_{group_col}_{_slugify(group_value)}.png"
        )
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"Saved: {output_path / filename}")


def plot_grouped_correlation_bars(
    df_corr: pd.DataFrame,
    group_col: str,
    output_dir: str = "results",
    value_col: str = "spearman_r",
):
    """Plot faceted bars of ranking vs utility correlations by group."""
    if df_corr is None or df_corr.empty:
        print("No grouped correlation data for bar plots")
        return

    output_path = Path(output_dir) / "figures"
    output_path.mkdir(exist_ok=True)

    g = sns.catplot(
        data=df_corr,
        x="ranking_metric",
        y=value_col,
        hue="utility_metric",
        col=group_col,
        kind="bar",
        col_wrap=3,
        sharey=True,
        height=4,
        aspect=1.2,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("Ranking metric", value_col)
    for ax in g.axes.flatten():
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    filename = f"ranking_utility_bar_by_{group_col}.png"
    plt.savefig(output_path / filename, dpi=150)
    plt.close()
    print(f"Saved: {output_path / filename}")


def plot_grouped_scatter_by_group(
    df_merged: pd.DataFrame,
    group_col: str,
    output_dir: str = "results",
):
    """Plot per-group scatter of ranking similarity vs utility loss."""
    if df_merged.empty:
        print("No merged data for grouped scatter plots")
        return

    output_path = Path(output_dir) / "figures"
    output_path.mkdir(exist_ok=True)

    ranking_cols = ["kendalltau", "rbo", "spearmanr"]
    available_ranking = [c for c in ranking_cols if c in df_merged.columns]
    if not available_ranking:
        print("No ranking metrics available for grouped scatter plots")
        return

    acc_candidates = [
        "utility_accuracy_drop",
        "utility_balanced_accuracy_drop",
        "fidelity_accuracy_drop",
        "fidelity_balanced_accuracy_drop",
    ]
    acc_col = next((col for col in acc_candidates if col in df_merged.columns), None)
    acc_label_map = {
        "utility_accuracy_drop": "Accuracy Drop (TRTR - TSTR)",
        "utility_balanced_accuracy_drop": "Balanced Accuracy Drop (TRTR - TSTR)",
        "fidelity_accuracy_drop": "Accuracy Drop (TRTR - TRTS)",
        "fidelity_balanced_accuracy_drop": "Balanced Accuracy Drop (TRTR - TRTS)",
    }
    acc_label = acc_label_map.get(acc_col, f"Accuracy Drop ({acc_col})")

    rmse_col = (
        "utility_rmse_ratio"
        if "utility_rmse_ratio" in df_merged.columns
        else "fidelity_rmse_ratio"
    )

    color_col = "perturbation" if group_col == "dataset" else "dataset"
    has_color = color_col in df_merged.columns

    for group_value, group_df in df_merged.groupby(group_col):
        group_slug = _slugify(group_value)

        if acc_col:
            df_class = group_df.dropna(subset=[acc_col] + available_ranking[:1])
            if len(df_class) >= 10:
                fig, axes = plt.subplots(
                    1,
                    min(len(available_ranking), 3),
                    figsize=(5 * min(len(available_ranking), 3), 5),
                )
                if len(available_ranking) == 1:
                    axes = [axes]

                for ax, rank_col in zip(axes, available_ranking[:3]):
                    if has_color:
                        for label in df_class[color_col].unique():
                            subset = df_class[df_class[color_col] == label]
                            ax.scatter(
                                subset[rank_col],
                                subset[acc_col],
                                label=label,
                                alpha=0.6,
                                s=30,
                            )
                    else:
                        ax.scatter(
                            df_class[rank_col],
                            df_class[acc_col],
                            alpha=0.6,
                            s=30,
                        )

                    valid = df_class[[rank_col, acc_col]].dropna()
                    if len(valid) > 5:
                        z = np.polyfit(valid[rank_col], valid[acc_col], 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(
                            valid[rank_col].min(), valid[rank_col].max(), 100
                        )
                        ax.plot(x_line, p(x_line), "k--", alpha=0.5)

                    ax.axhline(0, color="green", linestyle="--", alpha=0.3)
                    ax.set_xlabel(f"{rank_col} (higher = more similar)")
                    ax.set_ylabel(acc_label)
                    ax.set_title(f"{group_col}: {group_value}")

                if has_color:
                    axes[0].legend(fontsize=8, loc="upper right")
                plt.tight_layout()
                filename = (
                    f"ranking_vs_utility_{group_col}_{group_slug}_classification.png"
                )
                plt.savefig(output_path / filename, dpi=150)
                plt.close()
                print(f"Saved: {output_path / filename}")

        if rmse_col and rmse_col in group_df.columns:
            df_reg = group_df.dropna(subset=[rmse_col] + available_ranking[:1])
            df_reg = df_reg[(df_reg[rmse_col] > 0.1) & (df_reg[rmse_col] < 10)]
            if len(df_reg) >= 10:
                fig, axes = plt.subplots(
                    1,
                    min(len(available_ranking), 3),
                    figsize=(5 * min(len(available_ranking), 3), 5),
                )
                if len(available_ranking) == 1:
                    axes = [axes]

                for ax, rank_col in zip(axes, available_ranking[:3]):
                    if has_color:
                        for label in df_reg[color_col].unique():
                            subset = df_reg[df_reg[color_col] == label]
                            ax.scatter(
                                subset[rank_col],
                                subset[rmse_col],
                                label=label,
                                alpha=0.6,
                                s=30,
                            )
                    else:
                        ax.scatter(
                            df_reg[rank_col],
                            df_reg[rmse_col],
                            alpha=0.6,
                            s=30,
                        )

                    valid = df_reg[[rank_col, rmse_col]].dropna()
                    if len(valid) > 5:
                        z = np.polyfit(valid[rank_col], valid[rmse_col], 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(
                            valid[rank_col].min(), valid[rank_col].max(), 100
                        )
                        ax.plot(x_line, p(x_line), "k--", alpha=0.5)

                    ax.axhline(1, color="green", linestyle="--", alpha=0.3)
                    ax.set_xlabel(f"{rank_col} (higher = more similar)")
                    ax.set_ylabel(f"RMSE Ratio ({rmse_col})")
                    ax.set_title(f"{group_col}: {group_value}")

                if has_color:
                    axes[0].legend(fontsize=8, loc="upper right")
                plt.tight_layout()
                filename = (
                    f"ranking_vs_utility_{group_col}_{group_slug}_regression.png"
                )
                plt.savefig(output_path / filename, dpi=150)
                plt.close()
                print(f"Saved: {output_path / filename}")


def compute_grouped_ranking_utility_correlations(
    df_merged: pd.DataFrame,
    group_cols: list[str],
    min_n: int = 10,
    output_dir: str = "results",
):
    """Compute ranking-utility correlations within groups (e.g., per dataset)."""
    if df_merged.empty:
        print("No merged data for grouped correlation analysis")
        return None

    ranking_cols = [
        "kendalltau",
        "weightedtau",
        "spearmanr",
        "rbo",
        "ndcg_score",
        "jaccard_top_5",
        "jaccard_top_10",
    ]
    available_ranking = [c for c in ranking_cols if c in df_merged.columns]

    utility_cols = [
        "fidelity_accuracy_drop",
        "utility_accuracy_drop",
        "fidelity_balanced_accuracy_drop",
        "utility_balanced_accuracy_drop",
        "fidelity_rmse_ratio",
        "utility_rmse_ratio",
    ]
    available_utility = [c for c in utility_cols if c in df_merged.columns]

    if not available_ranking or not available_utility:
        print(
            "Insufficient metrics for grouped correlation: "
            f"ranking={available_ranking}, utility={available_utility}"
        )
        return None

    corr_results = []
    for group_vals, group_df in df_merged.groupby(group_cols):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        group_info = dict(zip(group_cols, group_vals))

        for rank_metric in available_ranking:
            for util_metric in available_utility:
                subset = group_df[[rank_metric, util_metric]].dropna()
                if len(subset) < min_n:
                    continue

                pearson_r, pearson_p = st.pearsonr(
                    subset[rank_metric], subset[util_metric]
                )
                spearman_r, spearman_p = st.spearmanr(
                    subset[rank_metric], subset[util_metric]
                )

                corr_results.append(
                    {
                        **group_info,
                        "ranking_metric": rank_metric,
                        "utility_metric": util_metric,
                        "pearson_r": pearson_r,
                        "pearson_p": pearson_p,
                        "spearman_r": spearman_r,
                        "spearman_p": spearman_p,
                        "n": len(subset),
                    }
                )

    df_corr = pd.DataFrame(corr_results)
    if df_corr.empty:
        print(f"No grouped correlations met min_n={min_n} for {group_cols}")
        return None

    group_suffix = "_".join(group_cols)
    output_path = Path(output_dir) / f"ranking_utility_correlations_by_{group_suffix}.csv"
    df_corr.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return df_corr


def compare_metrics_as_utility_predictors(
    df_merged: pd.DataFrame, df_baseline: pd.DataFrame, output_dir: str = "results"
):
    """
    Compare which metrics best predict utility degradation.

    This answers: Are ranking metrics better predictors of utility than baseline metrics?
    """
    output_path = Path(output_dir)

    # Merge baseline metrics with merged data
    merge_keys = ["dataset", "perturbation"]
    if not all(k in df_merged.columns and k in df_baseline.columns for k in merge_keys):
        return None

    df_all = df_merged.merge(
        df_baseline, on=merge_keys, how="left", suffixes=("", "_baseline")
    )

    # All candidate predictor metrics
    ranking_predictors = [
        "kendalltau",
        "rbo",
        "spearmanr",
        "weightedtau",
        "jaccard_top_5",
        "jaccard_top_10",
    ]
    baseline_predictors = ["propensity_auroc", "marginal_js_distance", "mmd_rbf"]

    available_predictors = [
        c for c in ranking_predictors + baseline_predictors if c in df_all.columns
    ]

    utility_metrics = [
        "fidelity_accuracy_drop",
        "utility_accuracy_drop",
        "fidelity_balanced_accuracy_drop",
        "utility_balanced_accuracy_drop",
        "fidelity_rmse_ratio",
        "utility_rmse_ratio",
    ]
    available_utility = [c for c in utility_metrics if c in df_all.columns]

    if not available_predictors or not available_utility:
        return None

    # Compute absolute correlations for each predictor
    predictor_scores = []
    for pred in available_predictors:
        pred_type = "ranking" if pred in ranking_predictors else "baseline"
        for util in available_utility:
            subset = df_all[[pred, util]].dropna()
            if len(subset) < 10:
                continue
            r, p = st.spearmanr(subset[pred], subset[util])
            predictor_scores.append(
                {
                    "predictor": pred,
                    "type": pred_type,
                    "utility_metric": util,
                    "abs_spearman": abs(r),
                    "spearman_r": r,
                    "p_value": p,
                    "n": len(subset),
                }
            )

    df_scores = pd.DataFrame(predictor_scores)
    if df_scores.empty:
        return None

    # Aggregate by predictor (mean across utility metrics)
    df_agg = (
        df_scores.groupby(["predictor", "type"])
        .agg({"abs_spearman": "mean", "n": "sum"})
        .reset_index()
        .sort_values("abs_spearman", ascending=False)
    )

    df_scores.to_csv(output_path / "predictor_comparison.csv", index=False)
    print(f"\nSaved: {output_path / 'predictor_comparison.csv'}")

    print("\n=== METRIC COMPARISON AS UTILITY PREDICTORS ===")
    print("(Higher |Spearman| = better predictor of utility degradation)")
    print(df_agg.to_string(index=False))

    return df_scores


def main(results_dir: str = "results"):
    """Main entry point for analysis."""
    print("Loading results...")
    all_results = load_results(results_dir)

    if not all_results:
        print(f"No results found in {results_dir}/")
        return

    print(f"Found results for {len(all_results)} datasets: {list(all_results.keys())}")

    # Save CSVs
    print("\n--- Generating CSVs ---")
    dfs = save_csvs(all_results, results_dir)

    # Generate visualizations
    print("\n--- Generating Visualizations ---")
    plot_baseline_heatmap(dfs["baseline"], results_dir)
    plot_ranking_comparison(dfs["ranking"], results_dir)
    plot_trts_comparison(dfs["target"], results_dir)
    plot_tstr_comparison(dfs["target"], results_dir)
    plot_fidelity_vs_utility(dfs["target"], results_dir)
    plot_aggregated_with_ci(dfs["aggregated"], results_dir)
    plot_perturbation_summary(dfs["baseline"], dfs["ranking"], results_dir)

    # Generate summary table
    print("\n--- Generating Summary ---")
    summary = generate_summary_table(dfs, results_dir)
    print("\nSummary Statistics:")
    print(summary.to_string(index=False))

    # Ranking vs utility correlation analysis
    print("\n--- Ranking vs Utility Analysis ---")
    corr_output = compute_ranking_utility_correlation(
        dfs["target"], dfs["ranking"], results_dir
    )
    if corr_output:
        _, df_merged = corr_output
        plot_ranking_vs_utility(df_merged, results_dir)
        compare_metrics_as_utility_predictors(df_merged, dfs["baseline"], results_dir)
        df_corr_dataset = compute_grouped_ranking_utility_correlations(
            df_merged, ["dataset"], min_n=10, output_dir=results_dir
        )
        if df_corr_dataset is not None:
            plot_grouped_correlation_heatmaps(
                df_corr_dataset, "dataset", results_dir
            )
            plot_grouped_correlation_bars(
                df_corr_dataset, "dataset", results_dir
            )
            plot_grouped_scatter_by_group(df_merged, "dataset", results_dir)

        df_corr_perturb = compute_grouped_ranking_utility_correlations(
            df_merged, ["perturbation"], min_n=10, output_dir=results_dir
        )
        if df_corr_perturb is not None:
            plot_grouped_correlation_heatmaps(
                df_corr_perturb, "perturbation", results_dir
            )
            plot_grouped_correlation_bars(
                df_corr_perturb, "perturbation", results_dir
            )
            plot_grouped_scatter_by_group(df_merged, "perturbation", results_dir)

    print(
        f"\n✓ Analysis complete! Check {results_dir}/ for CSVs and {results_dir}/figures/ for plots."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    args = parser.parse_args()

    main(args.results_dir)
