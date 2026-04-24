"""K-means clustering, silhouette-based K selection, centroid persistence,
and nearest-cluster lookup for KVForge dynamic PRS.

All heavy lifting is pure NumPy — no sklearn dependency.

Public API
----------
* ``select_k(embeddings, k_range)`` — silhouette-score search for best K.
* ``cluster_embeddings(embeddings, k_range)`` → ``(centroids, labels)``.
* ``save_clusters(path, centroids, labels, ...)`` — atomic JSON write.
* ``load_clusters(path)`` — returns dict with ``centroids`` as ndarray.
* ``nearest_cluster(query_embedding, centroids)`` → cluster index (int).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _kmeans(
    embeddings: np.ndarray, k: int, max_iter: int = 100, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Vanilla K-means with random initialisation.

    Args:
        embeddings: ``(n, dim)`` float array.
        k: Number of clusters.
        max_iter: Maximum iteration count.
        seed: RNG seed for reproducibility.

    Returns:
        ``(centroids, labels)`` — shapes ``(k, dim)`` and ``(n,)``.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(embeddings), k, replace=False)
    centroids = embeddings[idx].copy().astype(float)
    labels = np.zeros(len(embeddings), dtype=int)
    for _ in range(max_iter):
        dists = np.linalg.norm(embeddings[:, None] - centroids[None], axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = embeddings[mask].mean(axis=0)
    return centroids, labels


def _silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient for a clustering.

    Returns 0.0 when there are fewer than 4 points or only one cluster.

    Args:
        embeddings: ``(n, dim)`` float array.
        labels: ``(n,)`` integer cluster assignments.

    Returns:
        Mean silhouette score in [-1, 1].
    """
    n = len(embeddings)
    if n < 4:
        return 0.0
    scores: list[float] = []
    unique = np.unique(labels)
    for i in range(n):
        same_mask = labels == labels[i]
        same_mask[i] = False
        other_clusters = [c for c in unique if c != labels[i]]
        if not other_clusters or same_mask.sum() == 0:
            continue
        a = np.linalg.norm(embeddings[i] - embeddings[same_mask], axis=1).mean()
        b = min(
            np.linalg.norm(embeddings[i] - embeddings[labels == c], axis=1).mean()
            for c in other_clusters
        )
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_k(embeddings: np.ndarray, k_range: tuple[int, int] = (3, 20)) -> int:
    """Choose the number of clusters K by maximising the silhouette score.

    Args:
        embeddings: ``(n, dim)`` float array.
        k_range: ``(min_k, max_k)`` inclusive range to search.

    Returns:
        Best K found.
    """
    best_k, best_score = k_range[0], -1.0
    for k in range(k_range[0], min(k_range[1] + 1, len(embeddings))):
        _, labels = _kmeans(embeddings, k)
        score = _silhouette(embeddings, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def cluster_embeddings(
    embeddings: np.ndarray, k_range: tuple[int, int] = (3, 20)
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster *embeddings* using silhouette-selected K-means.

    Args:
        embeddings: ``(n, dim)`` float array.
        k_range: ``(min_k, max_k)`` range for K selection.

    Returns:
        ``(centroids, labels)`` — shapes ``(k, dim)`` and ``(n,)``.
    """
    k = select_k(embeddings, k_range)
    return _kmeans(embeddings, k)


def save_clusters(
    path: str,
    centroids: np.ndarray,
    labels: np.ndarray,
    cluster_labels: Optional[list[str]] = None,
    lora_version: int = 0,
) -> None:
    """Persist cluster data to a JSON file atomically (temp-file rename).

    Args:
        path: Destination file path.
        centroids: ``(k, dim)`` centroid matrix.
        labels: ``(n,)`` integer per-point cluster assignments.
        cluster_labels: Optional list of human-readable cluster names.
        lora_version: LoRA round that produced these clusters.
    """
    data = {
        "k": len(centroids),
        "centroids": centroids.tolist(),
        "labels": labels.tolist(),
        "cluster_labels": cluster_labels or [f"cluster_{i}" for i in range(len(centroids))],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lora_version": lora_version,
    }
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def load_clusters(path: str) -> dict:
    """Load cluster data from a JSON file.

    The ``'centroids'`` key is converted from a nested list back to a NumPy
    array for immediate use with :func:`nearest_cluster`.

    Args:
        path: Path to the JSON file produced by :func:`save_clusters`.

    Returns:
        Dict with keys ``k``, ``centroids`` (ndarray), ``labels``,
        ``cluster_labels``, ``created_at``, ``lora_version``.
    """
    with open(path) as f:
        data = json.load(f)
    data["centroids"] = np.array(data["centroids"])
    return data


def nearest_cluster(query_embedding: np.ndarray, centroids: np.ndarray) -> int:
    """Return the index of the centroid most similar to *query_embedding*.

    Similarity is measured by cosine similarity (L2-normalised dot product).

    Args:
        query_embedding: ``(dim,)`` query vector.
        centroids: ``(k, dim)`` centroid matrix.

    Returns:
        Integer cluster index in ``[0, k)``.
    """
    norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9
    normed = centroids / norms
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
    return int((normed @ q_norm).argmax())
