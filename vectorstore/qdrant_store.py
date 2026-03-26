"""Qdrant-backed implementation of the VectorStore protocol.

Wraps ``qdrant-client`` to provide the standard KVForge ``VectorStore``
interface.  All collections use cosine-distance vectors.  Requires a running
Qdrant server (Docker or Qdrant Cloud).
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from vectorstore.base import Point, ScoredPoint


class QdrantStore:
    """VectorStore backed by a Qdrant server.

    Communicates with Qdrant over gRPC/HTTP using the official
    ``qdrant-client`` library.  All collections are created with
    ``Distance.COSINE`` similarity.

    Args:
        host: Hostname of the Qdrant server.
        port: REST/gRPC port of the Qdrant server.

    Attributes:
        native_client: The underlying ``QdrantClient`` instance, exposed for
            Qdrant-specific operations that are not part of the protocol.
    """

    def __init__(self, host: str = "localhost", port: int = 6333):
        self._client = QdrantClient(host=host, port=port)

    def create_collection(self, name: str, dim: int) -> None:
        """Create a new Qdrant collection with cosine-distance vectors.

        Args:
            name: Collection name.
            dim: Vector dimensionality.
        """
        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def collection_exists(self, name: str) -> bool:
        """Return ``True`` if the named collection exists on the server.

        Args:
            name: Collection name to check.
        """
        return self._client.collection_exists(name)

    def delete_collection(self, name: str) -> None:
        """Delete a collection and all its data from the server.

        Args:
            name: Collection name to delete.
        """
        self._client.delete_collection(name)

    def upsert(self, collection: str, points: list[Point]) -> None:
        """Upsert a batch of points into *collection*.

        Args:
            collection: Target collection name.
            points: Points to insert or update.
        """
        qdrant_points = [
            PointStruct(id=p.id, vector=p.vector, payload=p.payload)
            for p in points
        ]
        self._client.upsert(collection_name=collection, points=qdrant_points)

    def query(self, collection: str, vector: list[float], top_k: int,
               score_threshold: float | None = None) -> list[ScoredPoint]:
        """Return the top-K nearest neighbours to *vector* in *collection*.

        Args:
            collection: Collection to search.
            vector: Query embedding vector.
            top_k: Maximum number of results to return.
            score_threshold: Optional minimum cosine score; lower-scoring
                results are excluded.

        Returns:
            List of ``ScoredPoint`` objects ordered by descending similarity.
        """
        kwargs = dict(collection_name=collection, query=vector,
                       limit=top_k, with_payload=True)
        if score_threshold is not None:
            kwargs["score_threshold"] = score_threshold
        resp = self._client.query_points(**kwargs)
        return [ScoredPoint(id=h.id, score=h.score, payload=h.payload or {})
                for h in resp.points]

    def scroll(self, collection: str, limit: int = 100,
                with_payload: bool = True, with_vectors: bool = False,
                offset=None, scroll_filter=None):
        """Page through all points in *collection*.

        Args:
            collection: Collection to scroll.
            limit: Maximum points per page.
            with_payload: Include payload in results.
            with_vectors: Include raw vectors in results.
            offset: Opaque cursor from the previous call; ``None`` to start
                from the beginning.
            scroll_filter: Qdrant ``Filter`` object to narrow results.

        Returns:
            A ``(points, next_offset)`` tuple; ``next_offset`` is ``None``
            when all pages have been consumed.
        """
        from typing import Any
        kwargs = dict(collection_name=collection, limit=limit,
                       with_payload=with_payload, with_vectors=with_vectors)
        if offset is not None:
            kwargs["offset"] = offset
        if scroll_filter is not None:
            kwargs["scroll_filter"] = scroll_filter
        results, next_offset = self._client.scroll(**kwargs)
        wrapped = [ScoredPoint(id=r.id, score=0.0, payload=r.payload or {})
                   for r in results]
        return wrapped, next_offset

    def set_payload(self, collection: str, point_id: int | str,
                     payload: dict) -> None:
        """Overwrite or extend the payload of a single point.

        Args:
            collection: Collection that owns the point.
            point_id: Identifier of the point to update.
            payload: Key-value pairs to set on the point.
        """
        self._client.set_payload(
            collection_name=collection,
            payload=payload,
            points=[point_id],
        )

    def count(self, collection: str) -> int:
        """Return the total number of points in *collection*.

        Args:
            collection: Collection name.
        """
        return self._client.count(collection_name=collection).count

    @property
    def native_client(self) -> QdrantClient:
        """Escape hatch for Qdrant-specific operations not in the protocol."""
        return self._client
