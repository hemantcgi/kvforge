"""Tests for core/difficulty_estimators.py — DifficultyEstimator protocol and built-ins."""

import numpy as np
import pytest
from core.difficulty_estimators import (
    DifficultyEstimator,
    IntraClusterDistance,
    VocabComplexity,
    EntityDensity,
    LengthVariance,
    get_estimator,
    register_estimator,
)


def test_protocol_structural():
    class Custom:
        def score(self, chunks, embeddings=None):
            return 0.5

    assert isinstance(Custom(), DifficultyEstimator)


def test_intra_cluster_distance_single_chunk():
    est = IntraClusterDistance()
    emb = np.array([[1.0, 0.0]])
    assert est.score(["hello"], emb) == 0.5


def test_intra_cluster_distance_orthogonal():
    est = IntraClusterDistance()
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    score = est.score(["a", "b"], emb)
    assert 0.9 < score <= 1.0  # orthogonal → distance ~1.0


def test_intra_cluster_distance_identical():
    est = IntraClusterDistance()
    emb = np.array([[1.0, 0.0], [1.0, 0.0]])
    score = est.score(["a", "b"], emb)
    assert score < 0.1  # identical → distance ~0.0


def test_vocab_complexity_high():
    est = VocabComplexity()
    chunks = ["phosphorylation methylation ubiquitination proteomics transcriptomics"]
    score = est.score(chunks)
    assert score > 0.3


def test_vocab_complexity_low():
    est = VocabComplexity()
    chunks = ["the cat sat on the mat and the dog ran fast"]
    score = est.score(chunks)
    assert score < 0.3


def test_entity_density_returns_float_in_range():
    est = EntityDensity()
    chunks = ["Apple Inc reported that Tim Cook met with Google CEO Sundar Pichai."]
    score = est.score(chunks)
    assert 0.0 <= score <= 1.0


def test_length_variance_uniform():
    est = LengthVariance()
    chunks = ["one two three", "four five six", "seven eight nine"]
    score = est.score(chunks)
    assert score < 0.1


def test_length_variance_mixed():
    est = LengthVariance()
    chunks = ["one", "one two three four five six seven eight nine ten eleven twelve"]
    score = est.score(chunks)
    assert score > 0.5


def test_get_estimator_known():
    est = get_estimator("intra_cluster_distance")
    assert isinstance(est, DifficultyEstimator)


def test_get_estimator_unknown():
    with pytest.raises(ValueError, match="Unknown difficulty estimator"):
        get_estimator("nonexistent_estimator")


def test_register_custom_estimator():
    class MyEst:
        def score(self, chunks, embeddings=None):
            return 0.42

    register_estimator("my_est", MyEst())
    est = get_estimator("my_est")
    assert est.score([]) == 0.42


def test_register_invalid_estimator():
    class BadEst:
        pass  # missing score()

    with pytest.raises(TypeError):
        register_estimator("bad", BadEst())
