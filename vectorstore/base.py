"""Shared dataclasses and the VectorStore structural protocol.

This module defines the common data types (``Point``, ``ScoredPoint``) and the
``VectorStore`` protocol that all backend implementations (Qdrant, Chroma, FAISS)
must satisfy.  Code that imports from this module can remain backend-agnostic.
"""
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Point:
    """A vector with an identifier and an optional metadata payload.

    Attributes:
        id: Unique identifier for the point (integer or string).
        vector: Dense embedding vector as a list of floats.
        payload: Arbitrary key-value metadata stored alongside the vector.
    """

    id: int | str
    vector: list[float]
    payload: dict = field(default_factory=dict)


@dataclass
class ScoredPoint:
    """A retrieved point together with its similarity score.

    Attributes:
        id: Identifier of the matched point.
        score: Similarity score (higher is more similar; cosine range 0–1).
        payload: Metadata payload of the matched point.
    """

    id: int | str
    score: float
    payload: dict = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    """Structural protocol for vector store backends.

    All SmartQdrant backend implementations (``QdrantStore``, ``ChromaStore``,
    ``FAISSStore``) must satisfy this interface.  The protocol is
    ``runtime_checkable``, so ``isinstance(obj, VectorStore)`` works at runtime.
    """

    def create_collection(self, name: str, dim: int) -> None:
        """Create a new collection with cosine-similarity vectors of size *dim*.

        Args:
            name: Collection name.
            dim: Dimensionality of the embedding vectors.
        """
        ...

    def collection_exists(self, name: str) -> bool:
        """Return ``True`` if *name* already exists in the backend.

        Args:
            name: Collection name to check.
        """
        ...

    def delete_collection(self, name: str) -> None:
        """Permanently delete a collection and all its data.

        Args:
            name: Collection name to delete.
        """
        ...

    def upsert(self, collection: str, points: list[Point]) -> None:
        """Insert or update points in *collection*.

        Args:
            collection: Target collection name.
            points: Points to upsert.
        """
        ...

    def query(self, collection: str, vector: list[float], top_k: int,
               score_threshold: float | None = None) -> list[ScoredPoint]:
        """Return the *top_k* nearest neighbours to *vector*.

        Args:
            collection: Collection to search.
            vector: Query embedding vector.
            top_k: Maximum number of results to return.
            score_threshold: If set, discard results with score below this value.

        Returns:
            List of ``ScoredPoint`` objects ordered by descending similarity.
        """
        ...

    def scroll(self, collection: str, limit: int = 100,
                with_payload: bool = True, with_vectors: bool = False,
                offset: Any = None, scroll_filter: Any = None) -> tuple[list, Any]:
        """Page through all points in the collection.

        Args:
            collection: Collection to scroll.
            limit: Maximum number of points per page.
            with_payload: Include payload in returned points.
            with_vectors: Include raw vectors in returned points.
            offset: Opaque cursor returned by the previous call; ``None`` starts
                from the beginning.
            scroll_filter: Backend-specific filter object (e.g. a Qdrant
                ``Filter``); ``None`` returns all points.

        Returns:
            A ``(points, next_offset)`` tuple.  ``next_offset`` is ``None``
            when there are no more results.
        """
        ...

    def set_payload(self, collection: str, point_id: int | str, payload: dict) -> None:
        """Merge *payload* into the stored payload for *point_id*.

        Args:
            collection: Collection that owns the point.
            point_id: Identifier of the point to update.
            payload: Key-value pairs to merge into the existing payload.
        """
        ...

    def count(self, collection: str) -> int:
        """Return the total number of points stored in *collection*.

        Args:
            collection: Collection name.
        """
        ...
