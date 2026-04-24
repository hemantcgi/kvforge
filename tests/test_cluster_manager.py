"""Tests for core/cluster_manager.py — K-means, silhouette K-selection, routing."""

import numpy as np
import pytest
from core.cluster_manager import (
    select_k,
    cluster_embeddings,
    save_clusters,
    load_clusters,
    nearest_cluster,
)


def _make_blobs(n_per_cluster=20, n_clusters=3, dim=8, seed=0):
    """Create well-separated Gaussian blobs for testing."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)) * 5
    parts = [rng.standard_normal((n_per_cluster, dim)) + centers[i] for i in range(n_clusters)]
    return np.vstack(parts)


def test_select_k_finds_true_k():
    emb = _make_blobs(n_per_cluster=20, n_clusters=3)
    k = select_k(emb, k_range=(2, 6))
    assert k == 3


def test_cluster_embeddings_returns_correct_shapes():
    emb = _make_blobs(n_per_cluster=15, n_clusters=4)
    centroids, labels = cluster_embeddings(emb, k_range=(2, 6))
    assert centroids.shape[1] == emb.shape[1]
    assert labels.shape == (len(emb),)
    assert set(labels) == set(range(len(centroids)))


def test_cluster_embeddings_label_count_matches_centroids():
    emb = _make_blobs(n_per_cluster=10, n_clusters=3)
    centroids, labels = cluster_embeddings(emb, k_range=(2, 5))
    assert len(centroids) == len(set(labels))


def test_save_and_load_clusters(tmp_path):
    emb = _make_blobs()
    centroids, labels = cluster_embeddings(emb, k_range=(2, 5))
    path = str(tmp_path / "clusters.json")
    save_clusters(path, centroids, labels, lora_version=1)
    data = load_clusters(path)
    assert data["k"] == len(centroids)
    assert np.allclose(data["centroids"], centroids)
    assert data["lora_version"] == 1
    assert len(data["cluster_labels"]) == len(centroids)


def test_nearest_cluster_returns_correct_index():
    centroids = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    query = np.array([0.9, 0.1])
    idx = nearest_cluster(query, centroids)
    assert idx == 0


def test_nearest_cluster_second():
    centroids = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    query = np.array([0.1, 0.9])
    idx = nearest_cluster(query, centroids)
    assert idx == 1
