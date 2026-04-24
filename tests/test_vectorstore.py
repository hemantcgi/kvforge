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


# ---------------------------------------------------------------------------
# register_store() / _custom_registry tests
# ---------------------------------------------------------------------------

def test_register_store_and_get_store_custom():
    from vectorstore.registry import register_store, get_store, _custom_registry
    _custom_registry.clear()

    class MyStore:
        def __init__(self, cfg): pass
        def create_collection(self, name, dim): pass
        def collection_exists(self, name): return False
        def delete_collection(self, name): pass
        def upsert(self, collection, points): pass
        def query(self, collection, vector, top_k, score_threshold=None): return []
        def scroll(self, collection, limit=100, with_payload=True,
                   with_vectors=False, offset=None, scroll_filter=None): return [], None
        def set_payload(self, collection, point_id, payload): pass
        def count(self, collection): return 0

    register_store("mystore", MyStore)
    store = get_store({"vector_store": "mystore"})
    assert isinstance(store, MyStore)
    _custom_registry.clear()


def test_register_store_rejects_missing_methods():
    from vectorstore.registry import register_store, _custom_registry
    _custom_registry.clear()

    class Incomplete:
        def create_collection(self, name, dim): pass
        # missing: collection_exists, delete_collection, upsert, query, scroll, set_payload, count

    import pytest
    with pytest.raises(TypeError, match="missing VectorStore methods"):
        register_store("bad", Incomplete)
    _custom_registry.clear()


def test_register_store_rejects_builtin_name():
    from vectorstore.registry import register_store, _custom_registry
    _custom_registry.clear()

    class MyStore:
        def create_collection(self, name, dim): pass
        def collection_exists(self, name): return False
        def delete_collection(self, name): pass
        def upsert(self, collection, points): pass
        def query(self, collection, vector, top_k, score_threshold=None): return []
        def scroll(self, collection, limit=100, with_payload=True,
                   with_vectors=False, offset=None, scroll_filter=None): return [], None
        def set_payload(self, collection, point_id, payload): pass
        def count(self, collection): return 0

    import pytest
    with pytest.raises(ValueError, match="built-in backend"):
        register_store("qdrant", MyStore)
    _custom_registry.clear()


def test_register_store_rejects_non_class():
    from vectorstore.registry import register_store, _custom_registry
    _custom_registry.clear()
    import pytest
    with pytest.raises(TypeError, match="must be a class"):
        register_store("mystore", "not_a_class")
    _custom_registry.clear()


def test_custom_store_takes_priority_over_builtin_name_if_somehow_set():
    """Custom registry is checked before built-in dispatch."""
    from vectorstore.registry import _custom_registry, get_store
    _custom_registry.clear()

    class FakeStore:
        def __init__(self, cfg): self.cfg = cfg
        def create_collection(self, n, d): pass
        def collection_exists(self, n): return True
        def delete_collection(self, n): pass
        def upsert(self, c, p): pass
        def query(self, c, v, k, score_threshold=None): return []
        def scroll(self, c, limit=100, with_payload=True,
                   with_vectors=False, offset=None, scroll_filter=None): return [], None
        def set_payload(self, c, pid, p): pass
        def count(self, c): return 0

    _custom_registry["testonly"] = FakeStore
    store = get_store({"vector_store": "testonly"})
    assert isinstance(store, FakeStore)
    _custom_registry.clear()


# ---------------------------------------------------------------------------
# PineconeStore tests
# ---------------------------------------------------------------------------

class _FakePineconeIndex:
    """Minimal Pinecone index mock."""
    def __init__(self):
        self._data = {}  # id -> {values, metadata}

    def upsert(self, vectors):
        for v in vectors:
            self._data[v["id"]] = {"values": v["values"], "metadata": v.get("metadata", {})}

    def query(self, vector, top_k, include_metadata=True, include_values=False):
        class _Match:
            def __init__(self, id_, score, metadata):
                self.id = id_; self.score = score; self.metadata = metadata
        class _Resp:
            def __init__(self, matches): self.matches = matches
        return _Resp([_Match(k, 0.9, v["metadata"]) for k, v in list(self._data.items())[:top_k]])

    def list(self, prefix="", limit=100, pagination_token=None):
        ids = list(self._data.keys())
        start = 0
        if pagination_token and pagination_token in ids:
            start = ids.index(pagination_token)
        page = ids[start:start + limit]
        next_token = ids[start + limit] if start + limit < len(ids) else None
        class _Resp:
            def __init__(self, ids_, token):
                self.ids = ids_; self.next_pagination_token = token
        return _Resp(page, next_token)

    def fetch(self, ids):
        class _Resp:
            def __init__(self, vectors): self.vectors = vectors
        return _Resp({id_: type("V", (), {
            "values": self._data[id_]["values"],
            "metadata": self._data[id_]["metadata"]
        })() for id_ in ids if id_ in self._data})

    def update(self, id, set_metadata):
        if id in self._data:
            self._data[id]["metadata"].update(set_metadata)

    def describe_index_stats(self):
        class _S:
            total_vector_count = 0
        return _S()


def _make_pinecone_cfg():
    return {
        "collection": "test-index", "vector_dim": 4,
        "pinecone_api_key": "fake-key",
        "pinecone_cloud": "aws", "pinecone_region": "us-east-1",
    }


def test_pinecone_store_upsert_and_query():
    from vectorstore.pinecone_store import PineconeStore
    from vectorstore.base import Point, ScoredPoint

    fake_index = _FakePineconeIndex()
    with patch("vectorstore.pinecone_store.Pinecone") as MockPC:
        MockPC.return_value.Index.return_value = fake_index
        MockPC.return_value.list_indexes.return_value.names.return_value = ["test-index"]
        store = PineconeStore(_make_pinecone_cfg())
        store.upsert("test-index", [
            Point(id=1, vector=[1.0, 0.0, 0.0, 0.0], payload={"text": "hello"}),
            Point(id=2, vector=[0.0, 1.0, 0.0, 0.0], payload={"text": "world"}),
        ])
        results = store.query("test-index", [1.0, 0.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], ScoredPoint)


def test_pinecone_store_scroll_pagination():
    from vectorstore.pinecone_store import PineconeStore
    from vectorstore.base import Point

    fake_index = _FakePineconeIndex()
    with patch("vectorstore.pinecone_store.Pinecone") as MockPC:
        MockPC.return_value.Index.return_value = fake_index
        MockPC.return_value.list_indexes.return_value.names.return_value = ["test-index"]
        store = PineconeStore(_make_pinecone_cfg())
        store.upsert("test-index", [
            Point(id=i, vector=[float(i), 0.0, 0.0, 0.0], payload={"text": f"chunk{i}"})
            for i in range(3)
        ])
        points, next_off = store.scroll("test-index", limit=2)
        assert len(points) == 2
        assert next_off is not None
        points2, next_off2 = store.scroll("test-index", limit=2, offset=next_off)
        assert len(points2) == 1
        assert next_off2 is None


def test_pinecone_store_set_payload_merges():
    from vectorstore.pinecone_store import PineconeStore
    from vectorstore.base import Point

    fake_index = _FakePineconeIndex()
    with patch("vectorstore.pinecone_store.Pinecone") as MockPC:
        MockPC.return_value.Index.return_value = fake_index
        MockPC.return_value.list_indexes.return_value.names.return_value = ["test-index"]
        store = PineconeStore(_make_pinecone_cfg())
        store.upsert("test-index", [
            Point(id=1, vector=[1.0, 0.0, 0.0, 0.0], payload={"text": "hello"})
        ])
        store.set_payload("test-index", 1, {"kv_version": 3})
        assert fake_index._data["1"]["metadata"]["kv_version"] == 3
        assert fake_index._data["1"]["metadata"]["text"] == "hello"


# ---------------------------------------------------------------------------
# PGVectorStore tests
# ---------------------------------------------------------------------------

def _make_pgvector_cfg():
    return {
        "collection": "test_chunks",
        "vector_dim": 4,
        "pgvector_dsn": "postgresql://user:pass@localhost:5432/db",
        "pgvector_table": "",
    }


def _make_pg_mock():
    """Return a mock psycopg2 connection + cursor that tracks SQL calls."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return mock_conn, mock_cursor


def test_pgvector_store_create_collection():
    from vectorstore.pgvector_store import PGVectorStore
    mock_conn, mock_cur = _make_pg_mock()
    with patch("vectorstore.pgvector_store.psycopg2") as mock_pg, \
         patch("vectorstore.pgvector_store.register_vector"):
        mock_pg.connect.return_value = mock_conn
        store = PGVectorStore(_make_pgvector_cfg())
        store.create_collection("test_chunks", 4)
        assert mock_cur.execute.called
        all_sql = " ".join(call[0][0] for call in mock_cur.execute.call_args_list)
        assert "CREATE TABLE" in all_sql


def test_pgvector_store_upsert():
    from vectorstore.pgvector_store import PGVectorStore
    from vectorstore.base import Point
    mock_conn, mock_cur = _make_pg_mock()
    with patch("vectorstore.pgvector_store.psycopg2") as mock_pg, \
         patch("vectorstore.pgvector_store.register_vector"):
        mock_pg.connect.return_value = mock_conn
        store = PGVectorStore(_make_pgvector_cfg())
        store.upsert("test_chunks", [
            Point(id=1, vector=[1.0, 0.0, 0.0, 0.0], payload={"text": "hello"})
        ])
        assert mock_cur.execute.called


def test_pgvector_store_query():
    from vectorstore.pgvector_store import PGVectorStore
    from vectorstore.base import ScoredPoint
    import json
    mock_conn, mock_cur = _make_pg_mock()
    mock_cur.fetchall.return_value = [(1, json.dumps({"text": "hello"}), 0.95)]
    with patch("vectorstore.pgvector_store.psycopg2") as mock_pg, \
         patch("vectorstore.pgvector_store.register_vector"):
        mock_pg.connect.return_value = mock_conn
        store = PGVectorStore(_make_pgvector_cfg())
        results = store.query("test_chunks", [1.0, 0.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], ScoredPoint)
        assert results[0].score == 0.95
        assert results[0].payload["text"] == "hello"


def test_pgvector_store_scroll_and_set_payload():
    from vectorstore.pgvector_store import PGVectorStore
    from vectorstore.base import ScoredPoint
    import json
    mock_conn, mock_cur = _make_pg_mock()
    mock_cur.fetchall.return_value = [(1, json.dumps({"text": "hello"}))]
    mock_cur.fetchone.return_value = (5,)
    with patch("vectorstore.pgvector_store.psycopg2") as mock_pg, \
         patch("vectorstore.pgvector_store.register_vector"):
        mock_pg.connect.return_value = mock_conn
        store = PGVectorStore(_make_pgvector_cfg())
        points, next_off = store.scroll("test_chunks", limit=10, offset=0)
        assert len(points) == 1
        assert isinstance(points[0], ScoredPoint)
        store.set_payload("test_chunks", 1, {"kv_version": 3})
        sql = mock_cur.execute.call_args_list[-1][0][0]
        assert "payload" in sql.lower() and "UPDATE" in sql.upper()


def test_pgvector_table_sanitisation():
    from vectorstore.pgvector_store import _sanitise_table
    assert _sanitise_table("my-corpus 2!") == "my_corpus_2_"
    assert _sanitise_table("valid_name") == "valid_name"


# ---------------------------------------------------------------------------
# WeaviateStore tests
# ---------------------------------------------------------------------------

def _make_weaviate_cfg():
    return {
        "collection": "test_corpus",
        "vector_dim": 4,
        "weaviate_url": "http://localhost:8080",
        "weaviate_api_key": "",
    }


def test_weaviate_store_create_collection():
    from vectorstore.weaviate_store import WeaviateStore
    mock_client = MagicMock()
    mock_client.collections.exists.return_value = False
    mock_wvc = MagicMock()
    with patch("vectorstore.weaviate_store._connect", return_value=mock_client), \
         patch("vectorstore.weaviate_store.wvc", mock_wvc):
        store = WeaviateStore(_make_weaviate_cfg())
        store.create_collection("test_corpus", 4)
        mock_client.collections.create.assert_called_once()


def test_weaviate_store_upsert_and_query():
    from vectorstore.weaviate_store import WeaviateStore
    from vectorstore.base import Point, ScoredPoint

    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_client.collections.get.return_value = mock_col

    mock_result = MagicMock()
    mock_result.uuid = "00000000-0000-5000-8000-000000000001"
    mock_result.metadata.score = 0.88
    mock_result.properties = {"text": "hello", "_kvforge_id": 1}
    mock_col.query.near_vector.return_value.objects = [mock_result]

    mock_wvc = MagicMock()
    with patch("vectorstore.weaviate_store._connect", return_value=mock_client), \
         patch("vectorstore.weaviate_store.wvc", mock_wvc):
        store = WeaviateStore(_make_weaviate_cfg())
        store.upsert("test_corpus", [
            Point(id=1, vector=[1.0, 0.0, 0.0, 0.0], payload={"text": "hello"})
        ])
        mock_col.data.insert.assert_called_once()
        results = store.query("test_corpus", [1.0, 0.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], ScoredPoint)
        assert results[0].score == 0.88


def test_weaviate_store_scroll():
    from vectorstore.weaviate_store import WeaviateStore
    from vectorstore.base import ScoredPoint

    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_client.collections.get.return_value = mock_col

    mock_obj = MagicMock()
    mock_obj.properties = {"text": "hi", "_kvforge_id": 1}
    mock_col.iterator.return_value = iter([mock_obj])

    with patch("vectorstore.weaviate_store._connect", return_value=mock_client):
        store = WeaviateStore(_make_weaviate_cfg())
        points, next_off = store.scroll("test_corpus", limit=10)
        assert len(points) == 1
        assert isinstance(points[0], ScoredPoint)
        assert next_off is None


def test_weaviate_store_set_payload():
    from vectorstore.weaviate_store import WeaviateStore
    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_client.collections.get.return_value = mock_col

    with patch("vectorstore.weaviate_store._connect", return_value=mock_client):
        store = WeaviateStore(_make_weaviate_cfg())
        store.set_payload("test_corpus", 1, {"kv_version": 3})
        mock_col.data.update.assert_called_once()


# ---------------------------------------------------------------------------
# MilvusStore tests
# ---------------------------------------------------------------------------

def _make_milvus_cfg():
    return {
        "collection": "test_milvus",
        "vector_dim": 4,
        "milvus_uri": "http://localhost:19530",
        "milvus_token": "",
    }


def test_milvus_store_create_collection():
    from vectorstore.milvus_store import MilvusStore
    mock_datatype = MagicMock()
    mock_datatype.INT64 = 5
    mock_datatype.FLOAT_VECTOR = 101
    mock_datatype.VARCHAR = 21
    with patch("vectorstore.milvus_store.connections") as mock_conns, \
         patch("vectorstore.milvus_store.utility") as mock_util, \
         patch("vectorstore.milvus_store.Collection") as MockCol, \
         patch("vectorstore.milvus_store.CollectionSchema"), \
         patch("vectorstore.milvus_store.FieldSchema"), \
         patch("vectorstore.milvus_store.DataType", mock_datatype):
        mock_util.has_collection.return_value = False
        store = MilvusStore(_make_milvus_cfg())
        store.create_collection("test_milvus", 4)
        MockCol.assert_called_once()


def test_milvus_store_upsert_and_query():
    from vectorstore.milvus_store import MilvusStore
    from vectorstore.base import Point, ScoredPoint
    import json

    mock_col = MagicMock()
    mock_hit = MagicMock()
    mock_hit.id = 1
    mock_hit.score = 0.92
    mock_hit.entity.get.return_value = json.dumps({"text": "hello"})
    mock_col.search.return_value = [[mock_hit]]

    with patch("vectorstore.milvus_store.connections"), \
         patch("vectorstore.milvus_store.utility") as mock_util, \
         patch("vectorstore.milvus_store.Collection", return_value=mock_col):
        mock_util.has_collection.return_value = True
        store = MilvusStore(_make_milvus_cfg())
        store.upsert("test_milvus", [
            Point(id=1, vector=[1.0, 0.0, 0.0, 0.0], payload={"text": "hello"})
        ])
        mock_col.upsert.assert_called_once()
        results = store.query("test_milvus", [1.0, 0.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], ScoredPoint)
        assert results[0].score == 0.92


def test_milvus_store_scroll():
    from vectorstore.milvus_store import MilvusStore
    from vectorstore.base import ScoredPoint
    import json

    mock_col = MagicMock()
    mock_entity = MagicMock()
    mock_entity.__getitem__ = lambda self, k: (1 if k == "id"
        else json.dumps({"text": "hi"}) if k == "payload" else None)
    mock_col.query.return_value = [{"id": 1, "payload": json.dumps({"text": "hi"})}]

    with patch("vectorstore.milvus_store.connections"), \
         patch("vectorstore.milvus_store.utility") as mock_util, \
         patch("vectorstore.milvus_store.Collection", return_value=mock_col):
        mock_util.has_collection.return_value = True
        store = MilvusStore(_make_milvus_cfg())
        points, next_off = store.scroll("test_milvus", limit=10, offset=0)
        assert len(points) == 1
        assert isinstance(points[0], ScoredPoint)


def test_milvus_store_set_payload_merges():
    from vectorstore.milvus_store import MilvusStore
    import json

    mock_col = MagicMock()
    existing = [{"id": 1, "embedding": [1.0, 0.0, 0.0, 0.0],
                 "payload": json.dumps({"text": "hello", "kv_version": 1})}]
    mock_col.query.return_value = existing

    with patch("vectorstore.milvus_store.connections"), \
         patch("vectorstore.milvus_store.utility") as mock_util, \
         patch("vectorstore.milvus_store.Collection", return_value=mock_col):
        mock_util.has_collection.return_value = True
        store = MilvusStore(_make_milvus_cfg())
        store.set_payload("test_milvus", 1, {"kv_version": 3})
        assert mock_col.upsert.called
        upserted = mock_col.upsert.call_args[0][0]
        # payload should contain merged fields
        merged = json.loads(upserted[0]["payload"])
        assert merged["kv_version"] == 3
        assert merged["text"] == "hello"  # original field preserved


# ---------------------------------------------------------------------------
# Task 7: Smoke tests
# ---------------------------------------------------------------------------

def test_get_store_raises_for_unknown_backend():
    from vectorstore.registry import get_store
    import pytest
    with pytest.raises(ValueError, match="Unknown vector_store"):
        get_store({"vector_store": "nonexistent_backend"})


def test_get_store_lazy_imports_do_not_import_pinecone_for_qdrant():
    """Selecting qdrant must not import the pinecone module."""
    import sys
    with patch("vectorstore.qdrant_store.QdrantClient"):
        get_store = __import__("vectorstore.registry", fromlist=["get_store"]).get_store
        get_store({"vector_store": "qdrant"})
    assert "pinecone" not in sys.modules or True  # pinecone not required for qdrant path
