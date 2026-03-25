"""Structural protocol for text-embedding backends.

All concrete embedders (FastEmbed, SentenceTransformers, OpenAI) must satisfy
the ``Embedder`` protocol defined here.  Code importing from this module can
remain backend-agnostic.
"""
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Protocol for objects that convert text into dense float vectors.

    All SmartQdrant embedding backends must implement ``encode`` and expose
    the ``dim`` property.  The protocol is ``runtime_checkable`` so
    ``isinstance`` checks work at runtime.
    """

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings into dense float vectors.

        Args:
            texts: List of strings to embed.  All items are embedded in a
                single call to minimise latency.

        Returns:
            List of embedding vectors, one per input string, each of length
            ``self.dim``.
        """
        ...

    @property
    def dim(self) -> int:
        """Dimensionality of the vectors produced by this embedder."""
        ...
