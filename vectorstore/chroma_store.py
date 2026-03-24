"""vectorstore/chroma_store.py — ChromaDB implementation of VectorStore."""
from typing import Any
from vectorstore.base import Point, ScoredPoint


class ChromaStore:
    """Local in-process ChromaDB — good for development without Docker."""

    def __init__(self, persist_dir: str = ".chroma"):
        try:
            import chromadb
        except ImportError:
            raise ImportError("ChromaStore requires: pip install chromadb")
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collections: dict = {}

    def _get_col(self, name: str):
        if name not in self._collections:
            self._collections[name] = self._client.get_collection(name)
        return self._collections[name]

    def create_collection(self, name: str, dim: int) -> None:
        col = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
        self._collections[name] = col

    def collection_exists(self, name: str) -> bool:
        return any(c.name == name for c in self._client.list_collections())

    def delete_collection(self, name: str) -> None:
        self._client.delete_collection(name)
        self._collections.pop(name, None)

    def upsert(self, collection: str, points: list[Point]) -> None:
        col = self._get_col(collection)
        col.upsert(
            ids=[str(p.id) for p in points],
            embeddings=[p.vector for p in points],
            documents=[p.payload.get("text", "") for p in points],
            metadatas=[{k: v for k, v in p.payload.items() if k != "text"}
                       for p in points],
        )

    def query(self, collection: str, vector: list[float], top_k: int,
               score_threshold: float | None = None) -> list[ScoredPoint]:
        col = self._get_col(collection)
        results = col.query(
            query_embeddings=[vector], n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        out = []
        for id_, doc, meta, dist in zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0]
        ):
            score = 1.0 - dist
            if score_threshold is not None and score < score_threshold:
                continue
            out.append(ScoredPoint(id=id_, score=score,
                                    payload={"text": doc, **(meta or {})}))
        return out

    def scroll(self, collection: str, limit: int = 100,
                with_payload: bool = True, with_vectors: bool = False,
                offset: Any = None, scroll_filter: Any = None) -> tuple[list, Any]:
        col = self._get_col(collection)
        offset_int = offset or 0
        results = col.get(limit=limit, offset=offset_int,
                           include=["documents", "metadatas"])
        out = [
            ScoredPoint(id=id_, score=0.0,
                         payload={"text": doc, **(meta or {})})
            for id_, doc, meta in zip(
                results["ids"], results["documents"], results["metadatas"]
            )
        ]
        next_offset = offset_int + len(out) if len(out) == limit else None
        return out, next_offset

    def set_payload(self, collection: str, point_id: int | str,
                     payload: dict) -> None:
        col = self._get_col(collection)
        existing = col.get(ids=[str(point_id)],
                            include=["metadatas", "documents", "embeddings"])
        if not existing["ids"]:
            return
        meta = existing["metadatas"][0] or {}
        meta.update({k: v for k, v in payload.items() if k != "text"})
        col.update(ids=[str(point_id)], metadatas=[meta])

    def count(self, collection: str) -> int:
        return self._get_col(collection).count()
