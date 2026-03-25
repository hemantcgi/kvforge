"""Tests for VectorStore abstraction."""
import pytest
from unittest.mock import MagicMock, patch


def test_point_and_scored_point_dataclasses():
    from vectorstore.base import Point, ScoredPoint
    p = Point(id=1, vector=[0.1, 0.2], payload={"text": "hello"})
    assert p.id == 1
    assert p.payload["text"] == "hello"
    sp = ScoredPoint(id=2, score=0.9, payload={"text": "result"})
    assert sp.score == 0.9


def test_qdrant_store_create_collection():
    from vectorstore.qdrant_store import QdrantStore
    with patch("vectorstore.qdrant_store.QdrantClient") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.collection_exists.return_value = False
        store = QdrantStore(host="localhost", port=6333)
        store.create_collection("test-col", dim=384)
        mock_client.create_collection.assert_called_once()


def test_qdrant_store_upsert():
    from vectorstore.qdrant_store import QdrantStore
    from vectorstore.base import Point
    with patch("vectorstore.qdrant_store.QdrantClient") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        store = QdrantStore(host="localhost", port=6333)
        points = [Point(id=0, vector=[0.1, 0.2], payload={"text": "hello"})]
        store.upsert("col", points)
        mock_client.upsert.assert_called_once()


def test_qdrant_store_query_returns_scored_points():
    from vectorstore.qdrant_store import QdrantStore
    from vectorstore.base import ScoredPoint
    with patch("vectorstore.qdrant_store.QdrantClient") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_hit = MagicMock()
        mock_hit.id = 1
        mock_hit.score = 0.9
        mock_hit.payload = {"text": "result"}
        mock_client.query_points.return_value.points = [mock_hit]
        store = QdrantStore(host="localhost", port=6333)
        results = store.query("col", [0.1, 0.2], top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], ScoredPoint)
        assert results[0].score == 0.9


def test_registry_returns_qdrant_store_by_default():
    from vectorstore.registry import get_store
    from vectorstore.qdrant_store import QdrantStore
    with patch("vectorstore.qdrant_store.QdrantClient"):
        store = get_store({"vector_store": "qdrant", "qdrant_host": "localhost", "qdrant_port": 6333})
    assert isinstance(store, QdrantStore)


def test_faiss_store_create_and_query(tmp_path):
    """FAISSStore creates a collection and returns scored results."""
    from vectorstore.faiss_store import FAISSStore
    store = FAISSStore(persist_dir=str(tmp_path / "faiss"))
    store.create_collection("test", 4)
    from vectorstore.base import Point
    store.upsert("test", [
        Point(id=0, vector=[1.0, 0.0, 0.0, 0.0], payload={"text": "alpha"}),
        Point(id=1, vector=[0.0, 1.0, 0.0, 0.0], payload={"text": "beta"}),
    ])
    results = store.query("test", [1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].payload["text"] == "alpha"
    assert results[0].score > 0.9


def test_faiss_store_scroll_and_set_payload(tmp_path):
    """FAISSStore scroll returns all points; set_payload merges correctly."""
    from vectorstore.faiss_store import FAISSStore
    from vectorstore.base import Point
    store = FAISSStore(persist_dir=str(tmp_path / "faiss2"))
    store.create_collection("col", 2)
    store.upsert("col", [
        Point(id=0, vector=[1.0, 0.0], payload={"text": "x"}),
        Point(id=1, vector=[0.0, 1.0], payload={"text": "y"}),
    ])
    points, next_off = store.scroll("col", limit=10)
    assert len(points) == 2
    assert next_off is None
    store.set_payload("col", 0, {"kv_cache": "abc"})
    points2, _ = store.scroll("col", limit=10)
    p0 = next(p for p in points2 if str(p.id) == "0")
    assert p0.payload["kv_cache"] == "abc"
    assert p0.payload["text"] == "x"  # original field preserved
