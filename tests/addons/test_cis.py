import pytest
import numpy as np
from addons.corpus_intelligence.cis import (
    compute_access_score,
    compute_uniqueness_score,
    compute_coverage_score,
    compute_cis,
)


def test_access_score_normalisation():
    counts = {"a": 100, "b": 50, "c": 0, "d": 10}
    scores = compute_access_score(counts)
    assert scores["a"] == pytest.approx(1.0, abs=0.01)
    assert scores["c"] == 0.0
    assert 0 < scores["b"] < scores["a"]


def test_access_score_log_scaled():
    counts = {"a": 1000, "b": 10}
    scores = compute_access_score(counts)
    assert scores["b"] < scores["a"]
    assert scores["a"] / scores["b"] < 5.0


def test_uniqueness_score_identical_chunks():
    embeddings = {
        "a": np.array([1.0, 0.0, 0.0]),
        "b": np.array([1.0, 0.0, 0.0]),
        "c": np.array([0.0, 1.0, 0.0]),
    }
    scores = compute_uniqueness_score(embeddings)
    assert scores["a"] == pytest.approx(0.0, abs=0.01)
    assert scores["b"] == pytest.approx(0.0, abs=0.01)
    assert scores["c"] == pytest.approx(1.0, abs=0.01)


def test_uniqueness_score_range():
    rng = np.random.default_rng(42)
    embeddings = {f"c{i}": rng.standard_normal(128) for i in range(20)}
    scores = compute_uniqueness_score(embeddings)
    for v in scores.values():
        assert 0.0 <= v <= 1.0


def test_coverage_score():
    faq_results = {
        "faq1": ["a", "b", "c", "d", "e"],
        "faq2": ["c", "d", "a", "x", "y"],
        "faq3": ["x", "y", "z", "m", "n"],
    }
    scores = compute_coverage_score(faq_results, top_k=5)
    assert scores["a"] == pytest.approx(2/3, abs=0.01)
    assert scores["c"] == pytest.approx(2/3, abs=0.01)
    assert scores["x"] == pytest.approx(2/3, abs=0.01)
    assert scores.get("z", 0.0) == pytest.approx(1/3, abs=0.01)


def test_compute_cis_combines_signals():
    access   = {"a": 0.9, "b": 0.2, "c": 0.5}
    unique   = {"a": 0.1, "b": 0.9, "c": 0.5}
    coverage = {"a": 0.8, "b": 0.3, "c": 0.5}
    cis = compute_cis(access, unique, coverage, alpha=0.33, beta=0.33, gamma=0.34)
    for score in cis.values():
        assert 0.0 <= score <= 1.0
    assert abs(cis["a"] - cis["b"]) < 0.4


def test_compute_cis_weights_sum_to_one():
    access   = {"x": 1.0}
    unique   = {"x": 1.0}
    coverage = {"x": 1.0}
    cis = compute_cis(access, unique, coverage, alpha=0.5, beta=0.3, gamma=0.2)
    assert cis["x"] == pytest.approx(1.0, abs=0.01)
