# vectorstore/pinecone_store.py
"""Pinecone serverless backend for KVForge."""
from typing import Any
from vectorstore.base import Point, ScoredPoint

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    Pinecone = None  # type: ignore
    ServerlessSpec = None  # type: ignore


class PineconeStore:
    """VectorStore backed by Pinecone serverless.

    Each KVForge collection maps to a Pinecone index. Vectors use cosine metric.
    Integer IDs are stored as strings (Pinecone requires string IDs).

    Args:
        cfg: Datasource config dict with pinecone_api_key, pinecone_cloud,
             pinecone_region, collection, vector_dim.

    Raises:
        ImportError: If pinecone package is not installed.
    """

    def __init__(self, cfg: dict) -> None:
        if Pinecone is None:
            raise ImportError("PineconeStore requires: pip install pinecone")
        self._pc = Pinecone(api_key=cfg["pinecone_api_key"])
        self._cloud = cfg.get("pinecone_cloud", "aws")
        self._region = cfg.get("pinecone_region", "us-east-1")
        self._dim = cfg.get("vector_dim", 384)
        self._indexes: dict[str, Any] = {}

    def _idx(self, name: str) -> Any:
        if name not in self._indexes:
            self._indexes[name] = self._pc.Index(name)
        return self._indexes[name]

    def create_collection(self, name: str, dim: int) -> None:
        if not self.collection_exists(name):
            self._pc.create_index(
                name=name, dimension=dim, metric="cosine",
                spec=ServerlessSpec(cloud=self._cloud, region=self._region),
            )
        self._indexes.pop(name, None)  # force re-fetch after creation

    def collection_exists(self, name: str) -> bool:
        return name in self._pc.list_indexes().names()

    def delete_collection(self, name: str) -> None:
        self._pc.delete_index(name)
        self._indexes.pop(name, None)

    def upsert(self, collection: str, points: list[Point]) -> None:
        idx = self._idx(collection)
        vectors = [
            {"id": str(p.id), "values": p.vector, "metadata": p.payload}
            for p in points
        ]
        # Pinecone recommends batches of ≤100
        for i in range(0, len(vectors), 100):
            idx.upsert(vectors=vectors[i:i + 100])

    def query(self, collection: str, vector: list[float], top_k: int,
              score_threshold: float | None = None) -> list[ScoredPoint]:
        idx = self._idx(collection)
        resp = idx.query(vector=vector, top_k=top_k, include_metadata=True)
        results = []
        for m in resp.matches:
            if score_threshold is not None and m.score < score_threshold:
                continue
            raw_id = m.id
            try:
                point_id: int | str = int(raw_id)
            except (ValueError, TypeError):
                point_id = raw_id
            results.append(ScoredPoint(id=point_id, score=m.score,
                                       payload=m.metadata or {}))
        return results

    def scroll(self, collection: str, limit: int = 100,
               with_payload: bool = True, with_vectors: bool = False,
               offset: Any = None, scroll_filter: Any = None) -> tuple[list, Any]:
        idx = self._idx(collection)
        resp = idx.list(prefix="", limit=limit,
                        pagination_token=offset if offset else None)
        ids = resp.ids
        next_offset = getattr(resp, "next_pagination_token", None)
        if not ids:
            return [], None
        fetch_resp = idx.fetch(ids=ids)
        points = []
        for id_str, vec in fetch_resp.vectors.items():
            try:
                point_id: int | str = int(id_str)
            except (ValueError, TypeError):
                point_id = id_str
            points.append(ScoredPoint(id=point_id, score=0.0,
                                      payload=vec.metadata if with_payload else {}))
        return points, next_offset or None

    def set_payload(self, collection: str, point_id: int | str,
                    payload: dict) -> None:
        self._idx(collection).update(id=str(point_id), set_metadata=payload)

    def count(self, collection: str) -> int:
        stats = self._idx(collection).describe_index_stats()
        return stats.total_vector_count
