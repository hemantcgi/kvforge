"""Stratified corpus subsampling for absorption-curve experiments.

Preserves topic distribution when creating size tiers by clustering on
embeddings and sampling proportionally from each cluster.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def stratified_subsample(
    chunks: list[dict],
    n: int,
    embeddings: np.ndarray,
    n_clusters: int = 10,
    seed: int = 42,
) -> list[dict]:
    """Return *n* chunks preserving topic distribution via k-means clustering.

    Args:
        chunks: List of chunk dicts, each with at least ``chunk_id`` and ``text``.
        n: Target number of chunks.
        embeddings: [len(chunks), dim] array of chunk embeddings.
        n_clusters: Number of k-means clusters for stratification.
        seed: Random seed for reproducible subsampling.

    Returns:
        List of *n* (or fewer if corpus is smaller) chunk dicts.
    """
    total = len(chunks)
    if n >= total:
        return list(chunks)

    k = min(n_clusters, total)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)

    selected_indices: list[int] = []
    rng = np.random.RandomState(seed)
    for cluster_id in range(k):
        cluster_members = np.where(labels == cluster_id)[0]
        n_from_cluster = max(1, round(n * len(cluster_members) / total))
        n_from_cluster = min(n_from_cluster, len(cluster_members))
        chosen = rng.choice(cluster_members, size=n_from_cluster, replace=False)
        selected_indices.extend(chosen.tolist())

    if len(selected_indices) > n:
        selected_indices = rng.choice(selected_indices, size=n, replace=False).tolist()
    elif len(selected_indices) < n:
        remaining = set(range(total)) - set(selected_indices)
        extra = rng.choice(list(remaining), size=n - len(selected_indices), replace=False)
        selected_indices.extend(extra.tolist())

    return [chunks[i] for i in sorted(selected_indices)]


def filter_questions_by_chunks(
    questions: list[dict],
    chunk_ids: set,
    id_field: str = "source_chunk_ids",
) -> list[dict]:
    """Return only questions whose answer chunks are all present in *chunk_ids*."""
    result = []
    for q in questions:
        q_chunk_ids = q.get(id_field, [])
        if q_chunk_ids and all(cid in chunk_ids for cid in q_chunk_ids):
            result.append(q)
    return result


def create_size_tiers(
    chunks: list[dict],
    embeddings: np.ndarray,
    tiers: list[int],
    n_clusters: int = 10,
    seed: int = 42,
) -> dict[int, list[dict]]:
    """Create nested size tiers where smaller tiers are subsets of larger ones.

    Generates the largest tier first, then recursively subsamples to create
    smaller tiers. This guarantees that tier_{smaller} ⊆ tier_{larger} for
    all size pairs, making dose-response curves causally interpretable and
    removing generator variance as a confound.

    Args:
        chunks: Full corpus.
        embeddings: Corresponding embeddings.
        tiers: List of target sizes, e.g. [500, 1000, 2000, 4000, 6000].
        n_clusters: K-means clusters for stratification.
        seed: Random seed.

    Returns:
        ``{N: [subsampled chunks]}`` for each tier size N, with subset guarantees.
    """
    sorted_tiers = sorted(tiers, reverse=True)
    result: dict[int, list[dict]] = {}
    current_chunks = list(chunks)
    current_embs = embeddings
    for n in sorted_tiers:
        tier = stratified_subsample(current_chunks, n, current_embs, n_clusters, seed)
        result[n] = tier
        current_chunks = tier
        tier_ids = {c["chunk_id"] for c in tier}
        idxs = [i for i, c in enumerate(chunks) if c["chunk_id"] in tier_ids]
        current_embs = embeddings[idxs]
    return result
