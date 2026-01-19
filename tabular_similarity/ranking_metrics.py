"""Ranking utilities and similarity metrics for feature importance lists."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import scipy.stats as stats
import rbo


def rank_features(
    importances: Mapping[str, float],
    tie_breaker: str = "name",
) -> List[str]:
    """Return a deterministically ordered list of feature names.

    Ties are broken by feature name to keep ranking stable across runs.
    """

    if tie_breaker != "name":
        raise ValueError("Only tie_breaker='name' is supported for determinism.")

    sorted_items = sorted(importances.items(), key=lambda item: (-item[1], item[0]))
    return [name for name, _ in sorted_items]


def spearman_footrule(rank_a: Sequence[str], rank_b: Sequence[str]) -> float:
    """Compute a normalized Spearman footrule distance in [0, 1]."""

    rank_a_pos = {name: idx for idx, name in enumerate(rank_a, start=1)}
    rank_b_pos = {name: idx for idx, name in enumerate(rank_b, start=1)}

    total = 0.0
    for name in rank_a_pos:
        total += abs(rank_a_pos[name] - rank_b_pos.get(name, len(rank_b) + 1))

    max_total = len(rank_a) ** 2
    return 1.0 - (total / max_total)


def jaccard_at_k(rank_a: Sequence[str], rank_b: Sequence[str], k: int) -> float:
    """Jaccard similarity of top-k elements."""

    set_a = set(rank_a[:k])
    set_b = set(rank_b[:k])
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def compare_rankings(
    rank_a: Sequence[str],
    rank_b: Sequence[str],
    rbo_p: float = 0.9,
    jaccard_k: int = 5,
) -> Dict[str, float]:
    """Return a dictionary of ranking similarity metrics."""

    rank_b_pos = {name: idx for idx, name in enumerate(rank_b, start=1)}
    positions = [rank_b_pos[name] for name in rank_a]
    base = list(range(1, len(rank_a) + 1))

    kendall_tau = stats.kendalltau(base, positions).correlation
    weighted_tau = stats.weightedtau(base, positions).correlation
    rbo_score = rbo.RankingSimilarity(rank_a, rank_b).rbo(p=rbo_p)

    return {
        "kendall_tau": float(kendall_tau) if kendall_tau is not None else float("nan"),
        "weighted_tau": float(weighted_tau) if weighted_tau is not None else float("nan"),
        "spearman_footrule": float(spearman_footrule(rank_a, rank_b)),
        "jaccard_at_k": float(jaccard_at_k(rank_a, rank_b, jaccard_k)),
        "rbo": float(rbo_score),
    }
