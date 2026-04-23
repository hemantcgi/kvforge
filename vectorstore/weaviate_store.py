# vectorstore/weaviate_store.py
"""Weaviate v4 backend for KVForge."""
import uuid
from typing import Any
from vectorstore.base import Point, ScoredPoint

try:
    import weaviate
    import weaviate.classes as wvc
except ImportError:
    weaviate = None  # type: ignore
    wvc = None  # type: ignore


def _connect(cfg: dict) -> Any:
    """Create and return a Weaviate v4 client."""
    if weaviate is None:
        raise ImportError("WeaviateStore requires: pip install weaviate-client")
    url = cfg.get("weaviate_url", "http://localhost:8080")
    api_key = cfg.get("weaviate_api_key", "")
    if api_key:
        return weaviate.connect_to_custom(
            http_host=url.split("://")[-1].split(":")[0],
            http_port=int(url.split(":")[-1]) if ":" in url.split("://")[-1] else 8080,
            http_secure="https" in url,
            auth_credentials=weaviate.auth.AuthApiKey(api_key),
        )
    return weaviate.connect_to_local(url=url)


def _to_uuid(point_id: int | str) -> str:
    """Convert a KVForge integer ID to a deterministic UUID5 string."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, str(point_id)))


def _class_name(collection: str) -> str:
    """Weaviate class names must start with uppercase."""
    return collection[0].upper() + collection[1:] if collection else "Kvforge"


class WeaviateStore:
    """VectorStore backed by Weaviate v4.

    KVForge integer IDs are stored as deterministic UUID5 values.
    The original integer is also stored in the payload under '_kvforge_id'.

    Args:
        cfg: Datasource config dict with weaviate_url, weaviate_api_key,
             collection, vector_dim.

    Raises:
        ImportError: If weaviate-client is not installed.
    """

    def __init__(self, cfg: dict) -> None:
        self._client = _connect(cfg)
        self._dim = cfg.get("vector_dim", 384)

    def _col(self, name: str):
        return self._client.collections.get(_class_name(name))

    def create_collection(self, name: str, dim: int) -> None:
        cls = _class_name(name)
        if not self._client.collections.exists(cls):
            self._client.collections.create(
                name=cls,
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=wvc.config.VectorDistances.COSINE
                ),
            )

    def collection_exists(self, name: str) -> bool:
        return self._client.collections.exists(_class_name(name))

    def delete_collection(self, name: str) -> None:
        self._client.collections.delete(_class_name(name))

    def upsert(self, collection: str, points: list[Point]) -> None:
        col = self._col(collection)
        for p in points:
            props = dict(p.payload)
            props["_kvforge_id"] = int(p.id) if isinstance(p.id, (int, str)) else p.id
            col.data.insert(
                properties=props,
                vector=p.vector,
                uuid=_to_uuid(p.id),
            )

    def query(self, collection: str, vector: list[float], top_k: int,
              score_threshold: float | None = None) -> list[ScoredPoint]:
        col = self._col(collection)
        resp = col.query.near_vector(
            near_vector=vector,
            limit=top_k,
            return_metadata=wvc.query.MetadataQuery(score=True),
        )
        results = []
        for obj in resp.objects:
            score = obj.metadata.score or 0.0
            if score_threshold is not None and score < score_threshold:
                continue
            props = dict(obj.properties)
            raw_id = props.pop("_kvforge_id", obj.uuid)
            try:
                point_id: int | str = int(raw_id)
            except (ValueError, TypeError):
                point_id = str(raw_id)
            results.append(ScoredPoint(id=point_id, score=score, payload=props))
        return results

    def scroll(self, collection: str, limit: int = 100,
               with_payload: bool = True, with_vectors: bool = False,
               offset: Any = None, scroll_filter: Any = None) -> tuple[list, Any]:
        col = self._col(collection)
        it = col.iterator(return_properties=None if with_payload else [])
        points = []
        for obj in it:
            props = dict(obj.properties) if with_payload else {}
            raw_id = props.pop("_kvforge_id", obj.uuid)
            try:
                point_id: int | str = int(raw_id)
            except (ValueError, TypeError):
                point_id = str(raw_id)
            points.append(ScoredPoint(id=point_id, score=0.0, payload=props))
            if len(points) >= limit:
                break
        next_offset = None  # Weaviate iterator handles cursor internally
        return points, next_offset

    def set_payload(self, collection: str, point_id: int | str,
                    payload: dict) -> None:
        col = self._col(collection)
        col.data.update(uuid=_to_uuid(point_id), properties=payload)

    def count(self, collection: str) -> int:
        col = self._col(collection)
        agg = col.aggregate.over_all(total_count=True)
        return agg.total_count or 0
