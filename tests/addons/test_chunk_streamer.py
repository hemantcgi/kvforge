"""Tests for ChunkStreamer — client-side filtered pagination."""
import types
import pytest

from addons.compute.chunk_streamer import ChunkStreamer


def _pt(id_, payload):
    """Create a SimpleNamespace point with .id and .payload."""
    return types.SimpleNamespace(id=id_, payload=payload)


def make_store(points, page_size=None):
    """Return a mock store whose scroll() paginates through points.

    offset=None → start from beginning; offset=N → start from index N.
    Returns (results, next_offset_or_None).
    """
    def scroll(collection, limit, with_payload, offset=None):
        start = 0 if offset is None else offset
        batch = points[start : start + limit]
        next_offset = start + limit if start + limit < len(points) else None
        return batch, next_offset

    store = types.SimpleNamespace(scroll=scroll)
    return store


# ── Basic stream tests ────────────────────────────────────────────────────────

def test_stream_all():
    pts = [_pt(i, {"text": f"chunk {i}"}) for i in range(5)]
    store = make_store(pts)
    streamer = ChunkStreamer(store, scroll_page_size=3)
    batches = list(streamer.stream("col", "all", None, batch_size=3))
    assert sum(len(b) for b in batches) == 5
    all_ids = [p.id for b in batches for p in b]
    assert all_ids == list(range(5))


def test_stream_null_filter():
    pts = [
        _pt(0, {"kv_version": None}),   # included (kv_version explicitly None)
        _pt(1, {}),                      # included (no kv_version key)
        _pt(2, {"kv_version": 1}),       # excluded
        _pt(3, {"kv_version": 2}),       # excluded
        _pt(4, {}),                      # included
    ]
    store = make_store(pts)
    streamer = ChunkStreamer(store, scroll_page_size=10)
    batches = list(streamer.stream("col", "null", None, batch_size=10))
    ids = [p.id for b in batches for p in b]
    assert set(ids) == {0, 1, 4}


def test_stream_stale_filter():
    """kv_version < filter_value=2 are stale; None is always stale."""
    pts = [
        _pt(0, {"kv_version": 0}),    # stale (0 < 2)
        _pt(1, {"kv_version": 1}),    # stale (1 < 2)
        _pt(2, {"kv_version": 2}),    # not stale (2 >= 2)
        _pt(3, {"kv_version": 3}),    # not stale
        _pt(4, {}),                   # null → always stale
    ]
    store = make_store(pts)
    streamer = ChunkStreamer(store, scroll_page_size=10)
    batches = list(streamer.stream("col", "stale", 2, batch_size=10))
    ids = [p.id for b in batches for p in b]
    assert set(ids) == {0, 1, 4}


def test_stream_source_filter():
    pts = [
        _pt(0, {"source_file": "a.pdf"}),
        _pt(1, {"source_file": "b.pdf"}),
        _pt(2, {"source_file": "a.pdf"}),
        _pt(3, {"source_file": "c.pdf"}),
    ]
    store = make_store(pts)
    streamer = ChunkStreamer(store, scroll_page_size=10)
    batches = list(streamer.stream("col", "source", "a.pdf", batch_size=10))
    ids = [p.id for b in batches for p in b]
    assert set(ids) == {0, 2}


def test_stream_empty_collection():
    store = make_store([])
    streamer = ChunkStreamer(store, scroll_page_size=10)
    batches = list(streamer.stream("col", "all", None, batch_size=5))
    assert batches == []


def test_batch_size_respected():
    pts = [_pt(i, {}) for i in range(10)]
    store = make_store(pts)
    streamer = ChunkStreamer(store, scroll_page_size=10)
    batches = list(streamer.stream("col", "all", None, batch_size=3))
    sizes = [len(b) for b in batches]
    assert sizes == [3, 3, 3, 1]


def test_stale_null_kv_version_is_stale():
    pts = [_pt(0, {"kv_version": None})]
    store = make_store(pts)
    streamer = ChunkStreamer(store, scroll_page_size=10)
    batches = list(streamer.stream("col", "stale", 5, batch_size=10))
    assert len(batches) == 1
    assert batches[0][0].id == 0


def test_stream_qdrant_chroma_faiss_scroll_shapes():
    """Verify streamer works with differently shaped mock point objects."""
    # Qdrant-style: SimpleNamespace with .id and .payload
    qdrant_pt = types.SimpleNamespace(id="q1", payload={"source_file": "doc.pdf"})
    # Chroma-style: dict wrapped in SimpleNamespace (streamer accesses .payload)
    chroma_pt = types.SimpleNamespace(id="c1", payload={"source_file": "doc.pdf"})
    # FAISS-style: ScoredPoint with .id, .payload, .score
    faiss_pt = types.SimpleNamespace(id="f1", payload={"source_file": "doc.pdf"}, score=0.95)

    for pt in [qdrant_pt, chroma_pt, faiss_pt]:
        store = make_store([pt])
        streamer = ChunkStreamer(store, scroll_page_size=10)
        batches = list(streamer.stream("col", "all", None, batch_size=10))
        assert len(batches) == 1
        assert batches[0][0] is pt
