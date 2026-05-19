"""ChunkStreamer — streams batches of VectorStore points for compute jobs.

Works with any VectorStore backend (Qdrant, Chroma, FAISS) using client-side
filtering so no backend-specific query objects are needed.
"""
from __future__ import annotations

from typing import Any, Generator


class ChunkStreamer:
    """Scrolls a VectorStore collection and yields filtered batches.

    Args:
        store: Any object implementing the VectorStore protocol's scroll() method.
        scroll_page_size: Points fetched per scroll call. Larger = fewer round-trips.
    """

    def __init__(self, store: Any, scroll_page_size: int = 50) -> None:
        self._store = store
        self._page_size = scroll_page_size

    def stream(
        self,
        collection: str,
        filter_type: str,
        filter_value: Any,
        batch_size: int = 16,
    ) -> Generator[list, None, None]:
        """Yield batches of points matching the filter.

        Args:
            collection: VectorStore collection name.
            filter_type: One of "null" (kv_version is absent/None),
                "stale" (kv_version < filter_value), "source" (source_file == filter_value),
                or "all" (no filtering).
            filter_value: Threshold for "stale", filename for "source"; ignored otherwise.
            batch_size: Points per yielded batch.

        Yields:
            Non-empty lists of points, each list <= batch_size.
        """
        offset = None
        batch: list = []

        while True:
            results, offset = self._store.scroll(
                collection,
                limit=self._page_size,
                with_payload=True,
                offset=offset,
            )
            if not results:
                break

            for point in results:
                if self._skip(point, filter_type, filter_value):
                    continue
                batch.append(point)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            if offset is None:
                break

        if batch:
            yield batch

    @staticmethod
    def _skip(point: Any, filter_type: str, filter_value: Any) -> bool:
        """Return True if this point should be excluded by the filter."""
        if filter_type == "null":
            return point.payload.get("kv_version") is not None
        if filter_type == "stale":
            ver = point.payload.get("kv_version")
            if ver is None:
                return False  # null kv_version is always stale
            try:
                return int(ver) >= int(filter_value)
            except (TypeError, ValueError):
                return False  # unparseable version is treated as stale
        if filter_type == "source":
            return point.payload.get("source_file") != filter_value
        return False  # "all" — include everything
