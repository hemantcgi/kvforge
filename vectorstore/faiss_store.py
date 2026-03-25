"""vectorstore/faiss_store.py — FAISS in-process vector store backend."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from vectorstore.base import Point, ScoredPoint


class FAISSStore:
    """
    FAISS flat in-process vector store for SmartQdrant.

    Vectors are normalised to unit length so inner-product search equals
    cosine similarity.  The index and metadata (ID map + payloads) are
    persisted to ``persist_dir`` as two files per collection:

    * ``<name>.index``    — FAISS binary index
    * ``<name>.meta.pkl`` — dict with keys ``id_map`` and ``payloads``

    Args:
        persist_dir: Directory for index and metadata files.
            Created automatically on first use.
    """

    def __init__(self, persist_dir: str = ".faiss") -> None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            raise ImportError(
                "FAISSStore requires faiss-cpu (or faiss-gpu):\n"
                "  pip install faiss-cpu"
            )
        self._root = Path(persist_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._indexes: dict[str, Any] = {}
        self._payloads: dict[str, dict[str, dict]] = {}
        self._id_map: dict[str, list[str]] = {}

    # ── internal helpers ────────────────────────────────────────────────────

    def _paths(self, name: str) -> tuple[Path, Path]:
        return (self._root / f"{name}.index",
                self._root / f"{name}.meta.pkl")

    def _load(self, name: str) -> None:
        """Load index + metadata from disk into memory."""
        import faiss
        idx_path, meta_path = self._paths(name)
        if idx_path.exists() and meta_path.exists():
            self._indexes[name] = faiss.read_index(str(idx_path))
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self._payloads[name] = meta["payloads"]
            self._id_map[name] = meta["id_map"]

    def _save(self, name: str) -> None:
        """Persist index + metadata to disk."""
        import faiss
        idx_path, meta_path = self._paths(name)
        faiss.write_index(self._indexes[name], str(idx_path))
        with open(meta_path, "wb") as f:
            pickle.dump(
                {"payloads": self._payloads[name], "id_map": self._id_map[name]},
                f,
            )

    def _ensure_loaded(self, name: str) -> None:
        if name not in self._indexes:
            self._load(name)

    # ── VectorStore protocol ────────────────────────────────────────────────

    def create_collection(self, name: str, dim: int) -> None:
        """Create a new FlatIP index (cosine on normalised vectors)."""
        import faiss
        self._indexes[name] = faiss.IndexFlatIP(dim)
        self._payloads[name] = {}
        self._id_map[name] = []
        self._save(name)

    def collection_exists(self, name: str) -> bool:
        if name in self._indexes:
            return True
        idx_path, _ = self._paths(name)
        return idx_path.exists()

    def delete_collection(self, name: str) -> None:
        for path in self._paths(name):
            path.unlink(missing_ok=True)
        self._indexes.pop(name, None)
        self._payloads.pop(name, None)
        self._id_map.pop(name, None)

    def upsert(self, collection: str, points: list[Point]) -> None:
        """
        Add or update points.

        FAISS FlatIP does not support in-place updates, so existing IDs have
        their payload updated but a new vector appended.  For the common case
        of a fresh index (first ``index`` run) this is not an issue.
        """
        import faiss
        self._ensure_loaded(collection)
        index = self._indexes[collection]
        for p in points:
            sid = str(p.id)
            vec = np.array([p.vector], dtype="float32")
            faiss.normalize_L2(vec)
            if sid not in self._payloads[collection]:
                index.add(vec)
                self._id_map[collection].append(sid)
            self._payloads[collection][sid] = p.payload
        self._save(collection)

    def query(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        score_threshold: float | None = None,
    ) -> list[ScoredPoint]:
        import faiss
        self._ensure_loaded(collection)
        vec = np.array([vector], dtype="float32")
        faiss.normalize_L2(vec)
        scores, indices = self._indexes[collection].search(vec, top_k)
        id_map = self._id_map[collection]
        results: list[ScoredPoint] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if score_threshold is not None and float(score) < score_threshold:
                continue
            sid = id_map[int(idx)]
            results.append(
                ScoredPoint(
                    id=sid,
                    score=float(score),
                    payload=self._payloads[collection].get(sid, {}),
                )
            )
        return results

    def scroll(
        self,
        collection: str,
        limit: int = 100,
        with_payload: bool = True,
        with_vectors: bool = False,
        offset: Any = None,
        scroll_filter: Any = None,
    ) -> tuple[list[ScoredPoint], Any]:
        """Page through all points in insertion order."""
        self._ensure_loaded(collection)
        id_map = self._id_map[collection]
        start = 0
        if offset is not None:
            try:
                start = id_map.index(str(offset)) + 1
            except ValueError:
                start = 0
        batch = id_map[start : start + limit]
        points = [
            ScoredPoint(
                id=sid,
                score=0.0,
                payload=self._payloads[collection].get(sid, {}),
            )
            for sid in batch
        ]
        next_offset = id_map[start + limit] if start + limit < len(id_map) else None
        return points, next_offset

    def set_payload(
        self, collection: str, point_id: int | str, payload: dict
    ) -> None:
        """Merge *payload* into the existing payload for *point_id*."""
        self._ensure_loaded(collection)
        sid = str(point_id)
        self._payloads[collection].setdefault(sid, {}).update(payload)
        self._save(collection)

    def count(self, collection: str) -> int:
        self._ensure_loaded(collection)
        return self._indexes[collection].ntotal
