# vectorstore/milvus_store.py
"""Milvus / Zilliz Cloud backend for KVForge."""
import json
from typing import Any
from vectorstore.base import Point, ScoredPoint

try:
    from pymilvus import (
        connections, utility, Collection,
        CollectionSchema, FieldSchema, DataType,
    )
except ImportError:
    connections = None  # type: ignore
    utility = None  # type: ignore
    Collection = None  # type: ignore
    CollectionSchema = None  # type: ignore
    FieldSchema = None  # type: ignore
    DataType = None  # type: ignore


class MilvusStore:
    """VectorStore backed by Milvus / Zilliz Cloud.

    set_payload fetches the existing record, merges payload, then upserts
    the full merged record — Milvus has no native partial-update API.

    Args:
        cfg: Datasource config dict with milvus_uri, milvus_token,
             collection, vector_dim.

    Raises:
        ImportError: If pymilvus is not installed.
    """

    def __init__(self, cfg: dict) -> None:
        if connections is None:
            raise ImportError("MilvusStore requires: pip install pymilvus")
        connections.connect(
            uri=cfg.get("milvus_uri", "http://localhost:19530"),
            token=cfg.get("milvus_token", "") or None,
        )
        self._dim = cfg.get("vector_dim", 384)
        self._collections: dict[str, Any] = {}

    def _col(self, name: str) -> Any:
        if name not in self._collections:
            self._collections[name] = Collection(name)
            self._collections[name].load()
        return self._collections[name]

    def create_collection(self, name: str, dim: int) -> None:
        if utility.has_collection(name):
            return
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64,
                        is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,
                        dim=dim),
            FieldSchema(name="payload", dtype=DataType.VARCHAR,
                        max_length=65535),
        ]
        schema = CollectionSchema(fields=fields)
        col = Collection(name=name, schema=schema)
        col.create_index(
            field_name="embedding",
            index_params={"metric_type": "COSINE", "index_type": "IVF_FLAT",
                          "params": {"nlist": 128}},
        )
        col.load()
        self._collections[name] = col

    def collection_exists(self, name: str) -> bool:
        return utility.has_collection(name)

    def delete_collection(self, name: str) -> None:
        utility.drop_collection(name)
        self._collections.pop(name, None)

    def upsert(self, collection: str, points: list[Point]) -> None:
        col = self._col(collection)
        data = [
            {"id": int(p.id), "embedding": p.vector,
             "payload": json.dumps(p.payload)}
            for p in points
        ]
        col.upsert(data)

    def query(self, collection: str, vector: list[float], top_k: int,
              score_threshold: float | None = None) -> list[ScoredPoint]:
        col = self._col(collection)
        results_raw = col.search(
            data=[vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["payload"],
        )
        results = []
        for hit in results_raw[0]:
            score = hit.score
            if score_threshold is not None and score < score_threshold:
                continue
            payload = json.loads(hit.entity.get("payload") or "{}")
            results.append(ScoredPoint(id=hit.id, score=score, payload=payload))
        return results

    def scroll(self, collection: str, limit: int = 100,
               with_payload: bool = True, with_vectors: bool = False,
               offset: Any = None, scroll_filter: Any = None) -> tuple[list, Any]:
        col = self._col(collection)
        off = int(offset) if offset is not None else 0
        output_fields = ["id", "payload"] if with_payload else ["id"]
        rows = col.query(
            expr="id >= 0",
            output_fields=output_fields,
            offset=off,
            limit=limit,
        )
        points = []
        for row in rows:
            payload = json.loads(row["payload"]) if with_payload else {}
            points.append(ScoredPoint(id=row["id"], score=0.0, payload=payload))
        next_offset = off + len(rows) if len(rows) == limit else None
        return points, next_offset

    def set_payload(self, collection: str, point_id: int | str,
                    payload: dict) -> None:
        col = self._col(collection)
        existing = col.query(
            expr=f"id == {int(point_id)}",
            output_fields=["id", "embedding", "payload"],
        )
        if not existing:
            return
        record = existing[0]
        current = json.loads(record["payload"] or "{}")
        current.update(payload)
        col.upsert([{
            "id": record["id"],
            "embedding": record["embedding"],
            "payload": json.dumps(current),
        }])

    def count(self, collection: str) -> int:
        col = self._col(collection)
        return col.num_entities
