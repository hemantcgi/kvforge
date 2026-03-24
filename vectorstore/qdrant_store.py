"""vectorstore/qdrant_store.py — Qdrant implementation of VectorStore."""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from vectorstore.base import Point, ScoredPoint


class QdrantStore:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self._client = QdrantClient(host=host, port=port)

    def create_collection(self, name: str, dim: int) -> None:
        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def collection_exists(self, name: str) -> bool:
        return self._client.collection_exists(name)

    def delete_collection(self, name: str) -> None:
        self._client.delete_collection(name)

    def upsert(self, collection: str, points: list[Point]) -> None:
        qdrant_points = [
            PointStruct(id=p.id, vector=p.vector, payload=p.payload)
            for p in points
        ]
        self._client.upsert(collection_name=collection, points=qdrant_points)

    def query(self, collection: str, vector: list[float], top_k: int,
               score_threshold: float | None = None) -> list[ScoredPoint]:
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
        self._client.set_payload(
            collection_name=collection,
            payload=payload,
            points=[point_id],
        )

    def count(self, collection: str) -> int:
        return self._client.count(collection_name=collection).count

    @property
    def native_client(self) -> QdrantClient:
        """Escape hatch for Qdrant-specific operations not in the protocol."""
        return self._client
