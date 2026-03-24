"""embeddings/base.py — Embedder Protocol."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings; returns list of float vectors."""
        ...

    @property
    def dim(self) -> int:
        """The dimensionality of produced vectors."""
        ...
