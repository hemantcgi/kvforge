"""ChromaDB-backed implementation of the VectorStore protocol.

Wraps ``chromadb.PersistentClient`` for an in-process, on-disk vector store.
No Docker or external server is required, making this backend convenient for
local development.  Requires ``pip install chromadb``.
"""
from typing import Any
from vectorstore.base import Point, ScoredPoint


class ChromaStore:
    """VectorStore backed by a local ChromaDB persistent instance.

    All collections are created with cosine-distance HNSW indexing.  Chroma
    stores point IDs as strings internally; numeric IDs are automatically
    cast with ``str()``.

    Args:
        persist_dir: Directory where ChromaDB persists its SQLite database and
            index files.  Created if it does not already exist.

    Raises:
        ImportError: If ``chromadb`` is not installed.
    """

    def __init__(self, persist_dir: str = ".chroma"):
        try:
            import chromadb
        except ImportError:
            raise ImportError("ChromaStore requires: pip install chromadb")
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collections: dict = {}

    def _get_col(self, name: str):
        """Return a cached Chroma collection handle, loading it on first access.

        Args:
            name: Collection name.
        """
        if name not in self._collections:
            self._collections[name] = self._client.get_collection(name)
        return self._collections[name]

    def create_collection(self, name: str, dim: int) -> None:
        """Create or open an existing collection configured for cosine distance.

        The ``dim`` parameter is accepted for interface compatibility but is not
        used by Chroma, which infers dimensionality from the first inserted vectors.

        Args:
            name: Collection name.
            dim: Vector dimensionality (unused by Chroma; kept for protocol compatibility).
        """
        col = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
        self._collections[name] = col

    def collection_exists(self, name: str) -> bool:
        """Return ``True`` if *name* is a known collection in this Chroma client.

        Args:
            name: Collection name to check.
        """
        return any(c.name == name for c in self._client.list_collections())

    def delete_collection(self, name: str) -> None:
        """Delete the collection from Chroma and clear the local cache entry.

        Args:
            name: Collection name to delete.
        """
        self._client.delete_collection(name)
        self._collections.pop(name, None)

    def upsert(self, collection: str, points: list[Point]) -> None:
        """Upsert points into *collection*.

        The ``"text"`` key of each point's payload is stored as the Chroma
        ``document``; all other payload keys become Chroma ``metadata``.

        Args:
            collection: Target collection name.
            points: Points to insert or update.
        """
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
        """Return the top-K nearest neighbours to *vector*.

        Chroma returns L2 distances for cosine space; scores are converted to
        ``1 - distance`` so that higher scores indicate higher similarity.

        Args:
            collection: Collection to search.
            vector: Query embedding vector.
            top_k: Maximum number of results to return.
            score_threshold: Optional minimum score; results below this are
                discarded after distance conversion.

        Returns:
            List of ``ScoredPoint`` objects ordered by descending similarity.
        """
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
        """Page through all points in *collection* using an integer offset.

        The ``scroll_filter`` argument is accepted for interface compatibility
        but is not applied — Chroma's ``get()`` API does not support server-side
        field filtering in the same way as Qdrant.

        Args:
            collection: Collection to scroll.
            limit: Maximum points per page.
            with_payload: Included for protocol compatibility; payload is always
                returned by this implementation.
            with_vectors: Included for protocol compatibility; vectors are not
                returned by this implementation.
            offset: Integer offset into the collection (``None`` means start at 0).
            scroll_filter: Ignored; present for protocol compatibility.

        Returns:
            A ``(points, next_offset)`` tuple; ``next_offset`` is ``None`` when
            the last page has been reached.
        """
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
        """Merge *payload* into the metadata of an existing point.

        The ``"text"`` key, if present in *payload*, is silently ignored because
        Chroma stores the document text separately and does not allow updating
        it via ``update()``.

        Args:
            collection: Collection that owns the point.
            point_id: Identifier of the point to update.
            payload: Key-value pairs to merge into the existing metadata.
        """
        col = self._get_col(collection)
        existing = col.get(ids=[str(point_id)],
                            include=["metadatas", "documents", "embeddings"])
        if not existing["ids"]:
            return
        meta = existing["metadatas"][0] or {}
        meta.update({k: v for k, v in payload.items() if k != "text"})
        col.update(ids=[str(point_id)], metadatas=[meta])

    def count(self, collection: str) -> int:
        """Return the total number of points stored in *collection*.

        Args:
            collection: Collection name.
        """
        return self._get_col(collection).count()
