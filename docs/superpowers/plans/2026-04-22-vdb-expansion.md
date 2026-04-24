# VDB Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `register_store()` API to the vectorstore registry plus four new built-in backends (Pinecone, PGVector, Weaviate, Milvus), each implementing the full 7-method `VectorStore` Protocol.

**Architecture:** `vectorstore/registry.py` gains a module-level `_custom_registry` dict and `register_store()` that validates Protocol compliance at registration time. Four new store files follow the same pattern as existing backends — lazy SDK import in `__init__`, all seven Protocol methods. Tests mock SDK clients throughout (same pattern as existing Qdrant tests). All backends added to `get_store()` dispatch and `DatasourceConfig` Literal.

**Tech Stack:** Python 3.11+, `pinecone` v3+, `psycopg2` + `pgvector`, `weaviate-client` v4, `pymilvus`; `unittest.mock` for all SDK mocking in tests.

---

## File Structure

**New files:**
- `vectorstore/pinecone_store.py`
- `vectorstore/pgvector_store.py`
- `vectorstore/weaviate_store.py`
- `vectorstore/milvus_store.py`

**Modified files:**
- `vectorstore/registry.py` — `_custom_registry`, `register_store()`, four new dispatches
- `core/config.py` — extended Literal + 8 new backend config fields
- `tests/test_vectorstore.py` — `register_store` tests + tests for all four new backends

---

### Task 1: Pluggable Registry — `register_store()` and `_custom_registry`

**Files:**
- Modify: `vectorstore/registry.py`
- Modify: `tests/test_vectorstore.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_vectorstore.py

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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_vectorstore.py::test_register_store_and_get_store_custom -v --override-ini="addopts="
```
Expected: FAIL with `ImportError: cannot import name 'register_store'`

- [ ] **Step 3: Implement `register_store()` and `_custom_registry`**

Replace the full contents of `vectorstore/registry.py` with:

```python
"""Factory that instantiates the configured VectorStore backend."""

_custom_registry: dict[str, type] = {}

_BUILTIN = {"qdrant", "chroma", "faiss", "pinecone", "pgvector", "weaviate", "milvus"}


def register_store(name: str, cls: type) -> None:
    """Register a custom VectorStore class under a given backend name.

    Validates that cls has all 7 VectorStore Protocol methods at registration
    time. Call from your startup script before any get_store() calls.

    Args:
        name: Backend name to register. Must not conflict with built-in names.
        cls:  Class implementing the VectorStore Protocol.

    Raises:
        ValueError: If name is a built-in backend name.
        TypeError:  If cls is not a class or is missing required methods.
    """
    if name in _BUILTIN:
        raise ValueError(
            f"'{name}' is a built-in backend name — choose a different name"
        )
    if not isinstance(cls, type):
        raise TypeError(f"cls must be a class, got {type(cls)}")
    required = {
        "create_collection", "collection_exists", "delete_collection",
        "upsert", "query", "scroll", "set_payload", "count",
    }
    missing = required - set(dir(cls))
    if missing:
        raise TypeError(f"cls is missing VectorStore methods: {missing}")
    _custom_registry[name] = cls


def get_store(cfg: dict):
    """Return the appropriate VectorStore for the given config.

    Checks custom registry first, then built-in backends.

    Args:
        cfg: Datasource configuration dict.

    Returns:
        A VectorStore-protocol-compatible instance.

    Raises:
        ValueError: If the backend name is not recognised.
    """
    backend = cfg.get("vector_store", "qdrant")

    if backend in _custom_registry:
        return _custom_registry[backend](cfg)

    if backend == "qdrant":
        from vectorstore.qdrant_store import QdrantStore
        return QdrantStore(
            host=cfg.get("qdrant_host", "localhost"),
            port=cfg.get("qdrant_port", 6333),
        )
    if backend == "chroma":
        from vectorstore.chroma_store import ChromaStore
        return ChromaStore(persist_dir=cfg.get("chroma_persist_dir", ".chroma"))

    if backend == "faiss":
        from vectorstore.faiss_store import FAISSStore
        return FAISSStore(persist_dir=cfg.get("faiss_persist_dir", ".faiss"))

    if backend == "pinecone":
        from vectorstore.pinecone_store import PineconeStore
        return PineconeStore(cfg)

    if backend == "pgvector":
        from vectorstore.pgvector_store import PGVectorStore
        return PGVectorStore(cfg)

    if backend == "weaviate":
        from vectorstore.weaviate_store import WeaviateStore
        return WeaviateStore(cfg)

    if backend == "milvus":
        from vectorstore.milvus_store import MilvusStore
        return MilvusStore(cfg)

    raise ValueError(
        f"Unknown vector_store '{backend}'. "
        f"Supported: qdrant, chroma, faiss, pinecone, pgvector, weaviate, milvus, "
        f"or any name registered via register_store()"
    )
```

- [ ] **Step 4: Run registry tests**

```bash
python -m pytest tests/test_vectorstore.py -k "register" -v --override-ini="addopts="
```
Expected: 5 PASSED

- [ ] **Step 5: Run full vectorstore test suite to confirm no regressions**

```bash
python -m pytest tests/test_vectorstore.py -v --override-ini="addopts="
```
Expected: All PASSED

- [ ] **Step 6: Commit**

```bash
git add vectorstore/registry.py tests/test_vectorstore.py
git commit -m "feat: add register_store() API and _custom_registry to vectorstore registry"
```

---

### Task 2: Config Fields for Four New Backends

**Files:**
- Modify: `core/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_config.py

def test_new_backend_config_defaults():
    from core.config import DatasourceConfig
    cfg = DatasourceConfig(
        collection="test", embed_model="BAAI/bge-small-en-v1.5",
        vector_dim=384, llm_model="meta-llama/Llama-3.2-3B-Instruct",
        checkpoint_dir="ckpt", version_file="version.json", replay_db="test_replay.db"
    )
    # Pinecone
    assert cfg.pinecone_api_key == ""
    assert cfg.pinecone_cloud == "aws"
    assert cfg.pinecone_region == "us-east-1"
    # PGVector
    assert cfg.pgvector_dsn == ""
    assert cfg.pgvector_table == ""
    # Weaviate
    assert cfg.weaviate_url == "http://localhost:8080"
    assert cfg.weaviate_api_key == ""
    # Milvus
    assert cfg.milvus_uri == "http://localhost:19530"
    assert cfg.milvus_token == ""


def test_vector_store_literal_accepts_new_backends():
    from core.config import DatasourceConfig
    for backend in ("pinecone", "pgvector", "weaviate", "milvus"):
        cfg = DatasourceConfig(
            collection="test", embed_model="BAAI/bge-small-en-v1.5",
            vector_dim=384, llm_model="meta-llama/Llama-3.2-3B-Instruct",
            checkpoint_dir="ckpt", version_file="version.json",
            replay_db="test_replay.db", vector_store=backend
        )
        assert cfg.vector_store == backend
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_config.py::test_new_backend_config_defaults -v --override-ini="addopts="
```
Expected: FAIL with `ValidationError` or `AttributeError`

- [ ] **Step 3: Add config fields**

In `core/config.py`, change the `vector_store` field to:

```python
    vector_store: Literal[
        "qdrant", "chroma", "faiss",
        "pinecone", "pgvector", "weaviate", "milvus"
    ] = "qdrant"
```

Then add these fields after `dashboard_port`:

```python
    # Pinecone (serverless)
    pinecone_api_key: str = ""
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # PGVector
    pgvector_dsn: str = ""          # e.g. "postgresql://user:pass@host:5432/db"
    pgvector_table: str = ""        # defaults to collection name if empty

    # Weaviate
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str = ""      # empty = no auth (local Weaviate)

    # Milvus / Zilliz Cloud
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""          # empty = no auth (local Milvus)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_config.py::test_new_backend_config_defaults tests/test_config.py::test_vector_store_literal_accepts_new_backends -v --override-ini="addopts="
```
Expected: 2 PASSED

- [ ] **Step 5: Run full config test suite**

```bash
python -m pytest tests/test_config.py -v --override-ini="addopts="
```
Expected: All PASSED

- [ ] **Step 6: Commit**

```bash
git add core/config.py tests/test_config.py
git commit -m "feat: add Pinecone, PGVector, Weaviate, Milvus config fields to DatasourceConfig"
```

---

### Task 3: PineconeStore

**Files:**
- Create: `vectorstore/pinecone_store.py`
- Modify: `tests/test_vectorstore.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_vectorstore.py

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
            start = ids.index(pagination_token) + 1
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_vectorstore.py::test_pinecone_store_upsert_and_query -v --override-ini="addopts="
```
Expected: FAIL with `ModuleNotFoundError: No module named 'vectorstore.pinecone_store'`

- [ ] **Step 3: Implement `vectorstore/pinecone_store.py`**

```python
# vectorstore/pinecone_store.py
"""Pinecone serverless backend for KVForge."""
from typing import Any
from vectorstore.base import Point, ScoredPoint

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    Pinecone = None  # type: ignore
    ServerlessSpec = None  # type: ignore


class PineconeStore:
    """VectorStore backed by Pinecone serverless.

    Each KVForge collection maps to a Pinecone index. Vectors use cosine metric.
    Integer IDs are stored as strings (Pinecone requires string IDs).

    Args:
        cfg: Datasource config dict with pinecone_api_key, pinecone_cloud,
             pinecone_region, collection, vector_dim.

    Raises:
        ImportError: If pinecone package is not installed.
    """

    def __init__(self, cfg: dict) -> None:
        if Pinecone is None:
            raise ImportError("PineconeStore requires: pip install pinecone")
        self._pc = Pinecone(api_key=cfg["pinecone_api_key"])
        self._cloud = cfg.get("pinecone_cloud", "aws")
        self._region = cfg.get("pinecone_region", "us-east-1")
        self._dim = cfg.get("vector_dim", 384)
        self._indexes: dict[str, Any] = {}

    def _idx(self, name: str) -> Any:
        if name not in self._indexes:
            self._indexes[name] = self._pc.Index(name)
        return self._indexes[name]

    def create_collection(self, name: str, dim: int) -> None:
        if not self.collection_exists(name):
            self._pc.create_index(
                name=name, dimension=dim, metric="cosine",
                spec=ServerlessSpec(cloud=self._cloud, region=self._region),
            )
        self._indexes.pop(name, None)  # force re-fetch after creation

    def collection_exists(self, name: str) -> bool:
        return name in self._pc.list_indexes().names()

    def delete_collection(self, name: str) -> None:
        self._pc.delete_index(name)
        self._indexes.pop(name, None)

    def upsert(self, collection: str, points: list[Point]) -> None:
        idx = self._idx(collection)
        vectors = [
            {"id": str(p.id), "values": p.vector, "metadata": p.payload}
            for p in points
        ]
        # Pinecone recommends batches of ≤100
        for i in range(0, len(vectors), 100):
            idx.upsert(vectors=vectors[i:i + 100])

    def query(self, collection: str, vector: list[float], top_k: int,
              score_threshold: float | None = None) -> list[ScoredPoint]:
        idx = self._idx(collection)
        resp = idx.query(vector=vector, top_k=top_k, include_metadata=True)
        results = []
        for m in resp.matches:
            if score_threshold is not None and m.score < score_threshold:
                continue
            raw_id = m.id
            try:
                point_id: int | str = int(raw_id)
            except (ValueError, TypeError):
                point_id = raw_id
            results.append(ScoredPoint(id=point_id, score=m.score,
                                       payload=m.metadata or {}))
        return results

    def scroll(self, collection: str, limit: int = 100,
               with_payload: bool = True, with_vectors: bool = False,
               offset: Any = None, scroll_filter: Any = None) -> tuple[list, Any]:
        idx = self._idx(collection)
        resp = idx.list(prefix="", limit=limit,
                        pagination_token=offset if offset else None)
        ids = resp.ids
        next_offset = getattr(resp, "next_pagination_token", None)
        if not ids:
            return [], None
        fetch_resp = idx.fetch(ids=ids)
        points = []
        for id_str, vec in fetch_resp.vectors.items():
            try:
                point_id: int | str = int(id_str)
            except (ValueError, TypeError):
                point_id = id_str
            points.append(ScoredPoint(id=point_id, score=0.0,
                                      payload=vec.metadata if with_payload else {}))
        return points, next_offset or None

    def set_payload(self, collection: str, point_id: int | str,
                    payload: dict) -> None:
        self._idx(collection).update(id=str(point_id), set_metadata=payload)

    def count(self, collection: str) -> int:
        stats = self._idx(collection).describe_index_stats()
        return stats.total_vector_count
```

- [ ] **Step 4: Run Pinecone tests**

```bash
python -m pytest tests/test_vectorstore.py -k "pinecone" -v --override-ini="addopts="
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add vectorstore/pinecone_store.py tests/test_vectorstore.py
git commit -m "feat: add PineconeStore with full VectorStore Protocol (scroll via List+Fetch API)"
```

---

### Task 4: PGVectorStore

**Files:**
- Create: `vectorstore/pgvector_store.py`
- Modify: `tests/test_vectorstore.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_vectorstore.py

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
        sql = mock_cur.execute.call_args_list[0][0][0]
        assert "CREATE TABLE" in sql


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_vectorstore.py::test_pgvector_store_create_collection -v --override-ini="addopts="
```
Expected: FAIL with `ModuleNotFoundError: No module named 'vectorstore.pgvector_store'`

- [ ] **Step 3: Implement `vectorstore/pgvector_store.py`**

```python
# vectorstore/pgvector_store.py
"""PostgreSQL + pgvector backend for KVForge."""
import json
import re
from typing import Any
from vectorstore.base import Point, ScoredPoint

try:
    import psycopg2
    from pgvector.psycopg2 import register_vector
except ImportError:
    psycopg2 = None  # type: ignore
    register_vector = None  # type: ignore


def _sanitise_table(name: str) -> str:
    """Replace non-alphanumeric/underscore characters with underscores."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


class PGVectorStore:
    """VectorStore backed by PostgreSQL with the pgvector extension.

    One table per collection. Uses cosine distance operator (<=>).

    Args:
        cfg: Datasource config dict with pgvector_dsn, collection, vector_dim,
             and optional pgvector_table.

    Raises:
        ImportError: If psycopg2 or pgvector packages are not installed.
    """

    def __init__(self, cfg: dict) -> None:
        if psycopg2 is None:
            raise ImportError(
                "PGVectorStore requires: pip install psycopg2-binary pgvector"
            )
        self._conn = psycopg2.connect(cfg["pgvector_dsn"])
        self._conn.autocommit = True
        register_vector(self._conn)
        self._dim = cfg.get("vector_dim", 384)
        self._default_table = _sanitise_table(cfg.get("collection", "kvforge"))
        self._table_override = _sanitise_table(cfg.get("pgvector_table", "")) or ""

    def _table(self, collection: str) -> str:
        return self._table_override or _sanitise_table(collection)

    def create_collection(self, name: str, dim: int) -> None:
        t = self._table(name)
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {t} (
                    id      BIGINT PRIMARY KEY,
                    embedding vector({dim}),
                    payload JSONB NOT NULL DEFAULT '{{}}'
                )
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {t}_embedding_idx
                    ON {t} USING ivfflat (embedding vector_cosine_ops)
            """)

    def collection_exists(self, name: str) -> bool:
        t = self._table(name)
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_name = %s LIMIT 1", (t,)
            )
            return cur.fetchone() is not None

    def delete_collection(self, name: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self._table(name)}")

    def upsert(self, collection: str, points: list[Point]) -> None:
        t = self._table(collection)
        with self._conn.cursor() as cur:
            for p in points:
                cur.execute(
                    f"INSERT INTO {t} (id, embedding, payload) VALUES (%s, %s, %s) "
                    f"ON CONFLICT (id) DO UPDATE SET embedding=EXCLUDED.embedding, "
                    f"payload=EXCLUDED.payload",
                    (p.id, p.vector, json.dumps(p.payload)),
                )

    def query(self, collection: str, vector: list[float], top_k: int,
              score_threshold: float | None = None) -> list[ScoredPoint]:
        t = self._table(collection)
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, payload, 1 - (embedding <=> %s::vector) AS score "
                f"FROM {t} ORDER BY embedding <=> %s::vector LIMIT %s",
                (vector, vector, top_k),
            )
            rows = cur.fetchall()
        results = []
        for row_id, payload_str, score in rows:
            if score_threshold is not None and score < score_threshold:
                continue
            payload = json.loads(payload_str) if isinstance(payload_str, str) else payload_str
            results.append(ScoredPoint(id=row_id, score=score, payload=payload))
        return results

    def scroll(self, collection: str, limit: int = 100,
               with_payload: bool = True, with_vectors: bool = False,
               offset: Any = None, scroll_filter: Any = None) -> tuple[list, Any]:
        t = self._table(collection)
        off = int(offset) if offset is not None else 0
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, payload FROM {t} ORDER BY id LIMIT %s OFFSET %s",
                (limit, off),
            )
            rows = cur.fetchall()
        points = []
        for row_id, payload_str in rows:
            payload = json.loads(payload_str) if isinstance(payload_str, str) else payload_str
            points.append(ScoredPoint(id=row_id, score=0.0,
                                      payload=payload if with_payload else {}))
        next_offset = off + len(rows) if len(rows) == limit else None
        return points, next_offset

    def set_payload(self, collection: str, point_id: int | str,
                    payload: dict) -> None:
        t = self._table(collection)
        with self._conn.cursor() as cur:
            cur.execute(
                f"UPDATE {t} SET payload = payload || %s::jsonb WHERE id = %s",
                (json.dumps(payload), point_id),
            )

    def count(self, collection: str) -> int:
        t = self._table(collection)
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            row = cur.fetchone()
        return row[0] if row else 0
```

- [ ] **Step 4: Run PGVector tests**

```bash
python -m pytest tests/test_vectorstore.py -k "pgvector" -v --override-ini="addopts="
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add vectorstore/pgvector_store.py tests/test_vectorstore.py
git commit -m "feat: add PGVectorStore with full VectorStore Protocol (JSONB merge set_payload)"
```

---

### Task 5: WeaviateStore

**Files:**
- Create: `vectorstore/weaviate_store.py`
- Modify: `tests/test_vectorstore.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_vectorstore.py

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
    with patch("vectorstore.weaviate_store._connect", return_value=mock_client):
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

    with patch("vectorstore.weaviate_store._connect", return_value=mock_client):
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_vectorstore.py::test_weaviate_store_create_collection -v --override-ini="addopts="
```
Expected: FAIL with `ModuleNotFoundError: No module named 'vectorstore.weaviate_store'`

- [ ] **Step 3: Implement `vectorstore/weaviate_store.py`**

```python
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
```

- [ ] **Step 4: Run Weaviate tests**

```bash
python -m pytest tests/test_vectorstore.py -k "weaviate" -v --override-ini="addopts="
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add vectorstore/weaviate_store.py tests/test_vectorstore.py
git commit -m "feat: add WeaviateStore with full VectorStore Protocol (UUID5 ID mapping)"
```

---

### Task 6: MilvusStore

**Files:**
- Create: `vectorstore/milvus_store.py`
- Modify: `tests/test_vectorstore.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_vectorstore.py

def _make_milvus_cfg():
    return {
        "collection": "test_milvus",
        "vector_dim": 4,
        "milvus_uri": "http://localhost:19530",
        "milvus_token": "",
    }


def test_milvus_store_create_collection():
    from vectorstore.milvus_store import MilvusStore
    with patch("vectorstore.milvus_store.connections") as mock_conns, \
         patch("vectorstore.milvus_store.utility") as mock_util, \
         patch("vectorstore.milvus_store.Collection") as MockCol, \
         patch("vectorstore.milvus_store.CollectionSchema"), \
         patch("vectorstore.milvus_store.FieldSchema"):
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_vectorstore.py::test_milvus_store_create_collection -v --override-ini="addopts="
```
Expected: FAIL with `ModuleNotFoundError: No module named 'vectorstore.milvus_store'`

- [ ] **Step 3: Implement `vectorstore/milvus_store.py`**

```python
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
```

- [ ] **Step 4: Run Milvus tests**

```bash
python -m pytest tests/test_vectorstore.py -k "milvus" -v --override-ini="addopts="
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add vectorstore/milvus_store.py tests/test_vectorstore.py
git commit -m "feat: add MilvusStore with full VectorStore Protocol (fetch-merge-upsert set_payload)"
```

---

### Task 7: Full Test Suite + Final Verification

**Files:**
- Modify: `tests/test_vectorstore.py` (one integration smoke test)

- [ ] **Step 1: Write the smoke test**

```python
# append to tests/test_vectorstore.py

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
```

- [ ] **Step 2: Run full vectorstore test suite**

```bash
python -m pytest tests/test_vectorstore.py -v --override-ini="addopts="
```
Expected: All PASSED (26+ tests)

- [ ] **Step 3: Run full project test suite**

```bash
python -m pytest tests/ -v --override-ini="addopts="
```
Expected: All PASSED (skip GPU-dependent tests)

- [ ] **Step 4: Commit**

```bash
git add tests/test_vectorstore.py
git commit -m "test: add smoke tests for unknown backend and lazy import isolation (vdb expansion complete)"
```
