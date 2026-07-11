"""Corpus Importance Score computation.

CIS = α × access_score + β × uniqueness_score + γ × coverage_score

All three components are normalised to [0, 1].
"""
import math
import numpy as np


def compute_access_score(hit_counts: dict[str, int]) -> dict[str, float]:
    """Log-normalised retrieval frequency. Returns scores in [0, 1]."""
    if not hit_counts:
        return {}
    max_count = max(hit_counts.values()) if hit_counts else 1
    log_max = math.log1p(max_count) or 1.0
    return {cid: math.log1p(cnt) / log_max for cid, cnt in hit_counts.items()}


def compute_uniqueness_score(embeddings: dict[str, np.ndarray]) -> dict[str, float]:
    """1 − max cosine similarity to any other chunk.

    High score = far from all neighbours = irreplaceable.
    Low score = near-duplicate exists = redundant candidate.
    """
    ids   = list(embeddings.keys())
    vecs  = np.stack([embeddings[i] for i in ids], axis=0).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-8)
    vecs_n = vecs / norms

    sims = vecs_n @ vecs_n.T   # [N, N] cosine similarities
    np.fill_diagonal(sims, -1)  # exclude self-similarity

    max_sim = sims.max(axis=1).clip(0, 1)  # [N]
    scores  = (1.0 - max_sim).clip(0, 1)
    return {cid: float(s) for cid, s in zip(ids, scores)}


def compute_coverage_score(
    faq_results: dict[str, list[str]],
    top_k: int = 5,
) -> dict[str, float]:
    """Fraction of FAQ topics this chunk appears in top-K for.

    faq_results: {faq_id: [chunk_id_rank1, chunk_id_rank2, ...]}
    Returns scores in [0, 1].
    """
    if not faq_results:
        return {}
    n_faqs     = len(faq_results)
    topic_hits: dict[str, int] = {}
    for ranked in faq_results.values():
        for cid in ranked[:top_k]:
            topic_hits[cid] = topic_hits.get(cid, 0) + 1
    return {cid: cnt / n_faqs for cid, cnt in topic_hits.items()}


def compute_cis(
    access_scores:   dict[str, float],
    unique_scores:   dict[str, float],
    coverage_scores: dict[str, float],
    alpha: float = 0.33,
    beta:  float = 0.33,
    gamma: float = 0.34,
) -> dict[str, float]:
    """Combine three signals into CIS = α×access + β×uniqueness + γ×coverage."""
    all_ids = set(access_scores) | set(unique_scores) | set(coverage_scores)
    return {
        cid: (
            alpha * access_scores.get(cid, 0.0)
            + beta  * unique_scores.get(cid, 0.0)
            + gamma * coverage_scores.get(cid, 0.0)
        )
        for cid in all_ids
    }
