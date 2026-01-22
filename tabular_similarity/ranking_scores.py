"""Ranking similarity metrics for feature importance comparison."""

from __future__ import annotations

from typing import Dict

import numpy as np
import scipy.stats as st
from sklearn.metrics import cohen_kappa_score, ndcg_score, r2_score

try:
    import rbo
    HAS_RBO = True
except ImportError:
    HAS_RBO = False

try:
    import Levenshtein as levenshtein
    from jellyfish import damerau_levenshtein_distance, jaro_winkler_similarity, hamming_distance
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False


def compute_ranking_scores(feat_importance_real: Dict[str, float], feat_importance_synth: Dict[str, float]) -> Dict[str, float]:
    """
    Compare feature importance rankings between real and synthetic models.

    Based on create_scores_v3 from the notebook.

    Args:
        feat_importance_real: Feature name -> importance from model trained on real data
        feat_importance_synth: Feature name -> importance from model trained on synthetic data

    Returns:
        Dictionary with various ranking similarity metrics
    """
    if not feat_importance_real or not feat_importance_synth:
        return {}

    # Get common features
    ftkeys = list(feat_importance_real.keys())

    # Rank features (negative to get descending order - higher importance = lower rank number)
    # Use deterministic tie-breaks by feature name.
    real_items = sorted(feat_importance_real.items(), key=lambda x: (-x[1], x[0]))
    x1_rank_dict = {}
    for rank, (name, _) in enumerate(real_items, start=1):
        x1_rank_dict[name] = rank

    # Same deterministic approach for synthetic features
    synth_items = sorted(feat_importance_synth.items(), key=lambda x: (-x[1], x[0]))
    x2_rank_dict = {}
    for rank, (name, _) in enumerate(synth_items, start=1):
        x2_rank_dict[name] = rank

    real_ranked = [name for name, _ in real_items]
    synth_ranked = [name for name, _ in synth_items]

    true_score = []
    model_score = []
    true_score_rank = []
    model_score_rank = []

    for key in ftkeys:
        if key in x1_rank_dict and key in x2_rank_dict:
            true_score_rank.append(x1_rank_dict[key])
            model_score_rank.append(x2_rank_dict[key])
            true_score.append(feat_importance_real[key])
            model_score.append(feat_importance_synth[key])

    if len(true_score_rank) < 2:
        return {}

    sc = {}

    # NDCG score
    try:
        sc["ndcg_score"] = float(ndcg_score([true_score_rank], [model_score_rank]))
    except Exception:
        sc["ndcg_score"] = None

    # Cohen's Kappa
    try:
        sc["cohen_kappa_score"] = float(cohen_kappa_score(true_score_rank, model_score_rank))
    except Exception:
        sc["cohen_kappa_score"] = None

    # R2 score on raw importance values
    try:
        sc["r2_score"] = float(r2_score(true_score, model_score))
    except Exception:
        sc["r2_score"] = None

    # Kendall's Tau
    try:
        sc["kendalltau"] = float(st.kendalltau(true_score_rank, model_score_rank)[0])
    except Exception:
        sc["kendalltau"] = None

    # Weighted Tau
    try:
        sc["weightedtau"] = float(st.weightedtau(true_score_rank, model_score_rank)[0])
    except Exception:
        sc["weightedtau"] = None

    # Spearman correlation
    try:
        sc["spearmanr"] = float(st.spearmanr(true_score_rank, model_score_rank)[0])
    except Exception:
        sc["spearmanr"] = None

    # RBO (Rank-Biased Overlap)
    if HAS_RBO:
        try:
            sc["rbo"] = float(rbo.RankingSimilarity(true_score_rank, model_score_rank).rbo())
        except Exception:
            sc["rbo"] = None

    # Top-k overlap (Jaccard) on ranked feature names
    def _jaccard_top_k(rank_a, rank_b, k):
        k = min(k, len(rank_a), len(rank_b))
        if k <= 0:
            return None
        set_a = set(rank_a[:k])
        set_b = set(rank_b[:k])
        if not set_a and not set_b:
            return 1.0
        return len(set_a & set_b) / len(set_a | set_b)

    sc["jaccard_top_5"] = _jaccard_top_k(real_ranked, synth_ranked, 5)
    sc["jaccard_top_10"] = _jaccard_top_k(real_ranked, synth_ranked, 10)

    # String similarity metrics on ranks
    if HAS_LEVENSHTEIN:
        try:
            # Convert ranks to strings for Levenshtein
            str_true = "".join(str(int(r)) for r in true_score_rank)
            str_model = "".join(str(int(r)) for r in model_score_rank)

            max_len = max(len(str_true), len(str_model))
            if max_len > 0:
                lev_dist = levenshtein.distance(str_true, str_model)
                sc["levenshtein_normalized_similarity"] = float(1 - lev_dist / max_len)

            sc["jaro_winkler_similarity"] = float(jaro_winkler_similarity(str_true, str_model))
        except Exception:
            pass

    return sc


def get_feature_importance(model, X_train, y_train, random_state: int = 42) -> Dict[str, float]:
    """
    Extract feature importance from a fitted model.

    Uses feature_importances_ if available (tree-based models),
    otherwise falls back to permutation importance.

    Args:
        model: Fitted sklearn model
        X_train: Training features (DataFrame with column names)
        y_train: Training target
        random_state: Random state for permutation importance

    Returns:
        Dictionary mapping feature names to importance values
    """
    from sklearn.inspection import permutation_importance

    feature_names = list(X_train.columns)

    if hasattr(model, "feature_importances_"):
        # Tree-based models
        importances = model.feature_importances_
        return {name: float(imp) for name, imp in zip(feature_names, importances)}
    else:
        # Use permutation importance for other models
        try:
            result = permutation_importance(
                model, X_train, y_train,
                n_repeats=10,
                random_state=random_state,
                n_jobs=-1
            )
            return {name: float(imp) for name, imp in zip(feature_names, result.importances_mean)}
        except Exception:
            return {}
