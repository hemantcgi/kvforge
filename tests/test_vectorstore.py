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
