"""Pluggable difficulty estimators for KVForge dynamic PRS.

Each estimator implements the ``DifficultyEstimator`` Protocol, which requires a
single ``score(chunks, embeddings=None) -> float`` method.  Scores are in [0, 1]
where higher means harder / more semantically diverse.

Built-in estimators
-------------------
* ``IntraClusterDistance`` — mean pairwise cosine distance between chunk embeddings.
* ``VocabComplexity`` — fraction of tokens that are not in a common vocabulary.
* ``EntityDensity`` — fraction of mid-sentence capitalised words (proxy for named entities).
* ``LengthVariance`` — coefficient of variation of chunk word lengths.

Use ``get_estimator(name)`` to retrieve a registered estimator by name, and
``register_estimator(name, estimator)`` to add custom ones.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DifficultyEstimator(Protocol):
    """Structural protocol for difficulty estimators.

    Any class that defines ``score(chunks, embeddings=None) -> float`` is a
    valid implementation — no inheritance required.
    """

    def score(self, chunks: list[str], embeddings: np.ndarray | None = None) -> float:
        """Estimate difficulty for a cluster of text chunks.

        Args:
            chunks: List of raw text strings belonging to the cluster.
            embeddings: Optional ``(n, dim)`` float array of chunk embeddings.

        Returns:
            A float in [0, 1] where 1.0 is maximally difficult.
        """
        ...


class IntraClusterDistance:
    """Mean pairwise cosine distance between chunk embeddings.

    Single chunk (or no embeddings) → returns 0.5 as a neutral estimate.
    Two identical embeddings → near 0.0.
    Two orthogonal embeddings → near 1.0.
    """

    def score(self, chunks: list[str], embeddings: np.ndarray | None = None) -> float:
        if embeddings is None or len(embeddings) < 2:
            return 0.5
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        normed = embeddings / norms
        sims = normed @ normed.T
        n = len(embeddings)
        distances = [1.0 - float(sims[i, j]) for i in range(n) for j in range(i + 1, n)]
        return float(np.mean(distances)) if distances else 0.5


class VocabComplexity:
    """Fraction of tokens that appear to be technical or rare.

    Uses word length as a proxy: words longer than ``rare_length`` characters
    tend to be domain-specific or technical (e.g. "phosphorylation", "ubiquitination").
    Short common words ("the", "is", "at") score near zero; jargon-heavy text
    scores close to 1.0.
    """

    def __init__(self, rare_length: int = 8) -> None:
        self._rare_length = rare_length

    def score(self, chunks: list[str], embeddings: np.ndarray | None = None) -> float:
        tokens = [t for chunk in chunks for t in re.findall(r"\b\w+\b", chunk.lower())]
        if not tokens:
            return 0.5
        rare = sum(1 for t in tokens if len(t) > self._rare_length)
        return rare / len(tokens)


class EntityDensity:
    """Proxy for named-entity density using mid-sentence capitalisation.

    Splits each chunk into sentences and counts words that start with an upper-case
    letter but are not the first word.  Normalised by total word count and scaled
    so that a density of 10 % maps to 1.0.
    """

    def score(self, chunks: list[str], embeddings: np.ndarray | None = None) -> float:
        total = entity = 0
        for chunk in chunks:
            for sent in re.split(r"[.!?]", chunk):
                words = sent.split()
                entity += sum(1 for i, w in enumerate(words) if i > 0 and w and w[0].isupper())
                total += len(words)
        if total == 0:
            return 0.5
        return min(entity / total * 10, 1.0)


class LengthVariance:
    """Coefficient of variation (std / mean) of chunk word counts, clamped to [0, 1].

    Uniform-length chunks → near 0.0.
    Highly mixed-length chunks → near 1.0.
    Single chunk → returns 0.5 as a neutral estimate.
    """

    def score(self, chunks: list[str], embeddings: np.ndarray | None = None) -> float:
        lengths = [len(c.split()) for c in chunks]
        if len(lengths) < 2:
            return 0.5
        mean = np.mean(lengths)
        if mean == 0:
            return 0.0
        return min(float(np.std(lengths) / mean), 1.0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, DifficultyEstimator] = {
    "intra_cluster_distance": IntraClusterDistance(),
    "vocab_complexity": VocabComplexity(),
    "entity_density": EntityDensity(),
    "length_variance": LengthVariance(),
}


def get_estimator(name: str) -> DifficultyEstimator:
    """Return the registered estimator for *name*.

    Args:
        name: Key in the registry (e.g. ``'intra_cluster_distance'``).

    Raises:
        ValueError: If *name* is not in the registry.
    """
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown difficulty estimator '{name}'. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]


def register_estimator(name: str, estimator: DifficultyEstimator) -> None:
    """Register a custom estimator under *name*.

    Args:
        name: Key to store in the registry.
        estimator: Object implementing the ``DifficultyEstimator`` protocol.

    Raises:
        TypeError: If *estimator* does not implement the protocol.
    """
    if not isinstance(estimator, DifficultyEstimator):
        raise TypeError(f"{estimator!r} does not implement DifficultyEstimator protocol")
    REGISTRY[name] = estimator
