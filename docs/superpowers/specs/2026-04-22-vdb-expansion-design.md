# VDB Expansion: Pluggable Registry + Four New Backends

**Date:** 2026-04-22
**Status:** Approved for implementation

---

## Problem

KVForge currently supports three vector store backends (Qdrant, Chroma, FAISS) via a hard-coded if-ladder in `vectorstore/registry.py`. Adding a new backend requires modifying three files: the store implementation, the registry dispatch, and `DatasourceConfig`'s `vector_store` Literal. There is no way for an operator to register a custom backend without editing KVForge source code. This limits deployment flexibility — teams running Pinecone, Weaviate, Milvus, or PostgreSQL with pgvector must either wrap their backend in one of the existing adapters or maintain a fork.

---

## Solution

Two changes in one feature:

1. **Pluggable registry** — `register_store(name, cls)` API that lets operators register any Protocol-compliant class before the first `get_store()` call, validated at registration time.
2. **Four new built-in backends** — Pinecone (serverless), PGVector, Weaviate (v4), Milvus — each implementing the full 7-method `VectorStore` Protocol.

---

## Architecture

### Registry Enhancement

`vectorstore/registry.py` gains a module-level `_custom_registry: dict[str, type] = {}` and a `register_store()` function. `get_store()` checks the custom registry before the built-in if-ladder.

```python
_custom_registry: dict[str, type] = {}

def register_store(name: str, cls: type) -> None:
    _BUILTIN = {"qdrant", "chroma", "faiss", "pinecone", "pgvector", "weaviate", "milvus"}
    if name in _BUILTIN:
        raise ValueError(f"'{name}' is a built-in backend name — choose a different name")
    if not isinstance(cls, type):
        raise TypeError(f"cls must be a class, got {type(cls)}")
    required = {"create_collection", "collection_exists", "delete_collection",
                "upsert", "query", "scroll", "set_payload", "count"}
    missing = required - set(dir(cls))
    if missing:
        raise TypeError(f"cls is missing VectorStore methods: {missing}")
    _custom_registry[name] = cls

def get_store(cfg: dict):
    backend = cfg.get("vector_store", "qdrant")
    if backend in _custom_registry:
        return _custom_registry[backend](cfg)
    # ... existing built-in dispatch + four new backends
```

Validation happens at `register_store()` call time, not at first use. This catches protocol violations during application startup rather than mid-query.

---

### Config Fields

New fields in `DatasourceConfig`. All are optional with defaults. The `vector_store` Literal is extended.

```python
vector_store: Literal[
    "qdrant", "chroma", "faiss",
    "pinecone", "pgvector", "weaviate", "milvus"
] = "qdrant"

# Pinecone (serverless)
pinecone_api_key: str = ""
pinecone_cloud: str = "aws"         # "aws" | "gcp" | "azure"
pinecone_region: str = "us-east-1"

# PGVector
pgvector_dsn: str = ""              # e.g. "postgresql://user:pass@host:5432/db"
pgvector_table: str = ""            # defaults to collection name if empty

# Weaviate
weaviate_url: str = "http://localhost:8080"
weaviate_api_key: str = ""          # empty = no auth (local Weaviate)

# Milvus / Zilliz Cloud
milvus_uri: str = "http://localhost:19530"
milvus_token: str = ""              # empty = no auth (local Milvus)
```

Each backend reads only its own fields from cfg. `get_store()` passes `cfg` directly to each constructor.

---

### Backend Implementations

#### Pinecone (`vectorstore/pinecone_store.py`)

SDK: `pinecone` v3+ (serverless). Each KVForge collection maps to a Pinecone index. Vectors stored with cosine metric.

**Constructor:** Creates a `Pinecone` client from `cfg["pinecone_api_key"]`. Index is created with `ServerlessSpec(cloud, region)`.

**`scroll`:** Pinecone's `index.list(prefix="", limit=100, pagination_token=offset)` returns ID pages. IDs are then fetched in batches of 100 via `index.fetch(ids)` to retrieve metadata. Returns `(points, next_pagination_token)`. When exhausted, `next_pagination_token` is `None`.

**`set_payload`:** `index.update(id=str(point_id), set_metadata=payload)` — Pinecone merges metadata fields cleanly.

**`query`:** `index.query(vector=vector, top_k=top_k, include_metadata=True)`.

**`delete_collection`:** `pc.delete_index(name)`.

**ID handling:** KVForge integer IDs are stored as strings (`str(id)`) in Pinecone's string-ID namespace. Converted back to int on retrieval where possible.

---

#### PGVector (`vectorstore/pgvector_store.py`)

SDK: `psycopg2` + `pgvector` Python extension. One table per collection:

```sql
CREATE TABLE IF NOT EXISTS {table} (
    id      BIGINT PRIMARY KEY,
    embedding vector({dim}),
    payload JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS {table}_embedding_idx
    ON {table} USING ivfflat (embedding vector_cosine_ops);
```

**`scroll`:** `SELECT id, payload FROM {table} ORDER BY id LIMIT %s OFFSET %s`. Offset is an integer cursor; `next_offset = offset + len(results)`, `None` when `len(results) < limit`.

**`set_payload`:** `UPDATE {table} SET payload = payload || %s::jsonb WHERE id = %s` — PostgreSQL JSONB merge operator (`||`) merges fields without overwriting unrelated keys.

**`query`:** `SELECT id, payload, 1 - (embedding <=> %s::vector) AS score FROM {table} ORDER BY embedding <=> %s::vector LIMIT %s` — pgvector cosine distance operator `<=>`.

**`pgvector_table`:** Defaults to `cfg["collection"]` if empty. Sanitised (only alphanumeric and underscores) before use in SQL to prevent injection.

**Connection:** Single `psycopg2` connection per `PGVectorStore` instance, created in `__init__`. `autocommit=True`.

---

#### Weaviate (`vectorstore/weaviate_store.py`)

SDK: `weaviate-client` v4. KVForge collections map to Weaviate classes (capitalised: `"my_corpus"` → `"My_corpus"`).

**`scroll`:** Weaviate v4's `collection.iterator()` returns a native Python iterator with built-in cursor support. Wrapped to produce `(List[ScoredPoint], next_offset)` where `next_offset` is the iterator state (an opaque object) or `None` when exhausted.

**`set_payload`:** `collection.data.update(uuid=_to_uuid(point_id), properties=payload)` — Weaviate merges properties.

**`query`:** `collection.query.near_vector(near_vector=vector, limit=top_k, return_metadata=MetadataQuery(score=True))`.

**ID mapping:** KVForge integer IDs are stored as deterministic UUID5 (namespace = `uuid.NAMESPACE_OID`, name = `str(id)`) so the same integer always maps to the same UUID. The integer is also stored in the `payload` under `_kvforge_id` for reverse lookup.

**Auth:** `weaviate.connect_to_custom(http_host, api_key=AuthApiKey(key))` if `weaviate_api_key` is set; `weaviate.connect_to_local(url)` otherwise.

---

#### Milvus (`vectorstore/milvus_store.py`)

SDK: `pymilvus`. Collections use `IVF_FLAT` index with `COSINE` metric.

**Schema:** Fields: `id` (INT64, primary, auto_id=False), `embedding` (FLOAT_VECTOR, dim), `payload` (VARCHAR, max_length=65535 — JSON-serialised).

**`scroll`:** `collection.query(expr="id >= 0", output_fields=["id", "payload"], offset=offset, limit=limit)`. Offset is an integer. `next_offset = offset + len(results)`, `None` when `len(results) < limit`.

**`set_payload`:** Milvus has no partial update API. Implementation: (1) `collection.query(expr=f"id == {point_id}", output_fields=["*"])` to fetch existing record including embedding, (2) deserialise existing payload JSON, (3) merge new payload fields, (4) `collection.upsert([full_record])`. Correct but heavier than other backends; acceptable for KV recomputation workloads.

**`query`:** `collection.search(data=[vector], anns_field="embedding", param={"metric_type": "COSINE"}, limit=top_k, output_fields=["payload"])`.

**Connection:** `connections.connect(uri=cfg["milvus_uri"], token=cfg["milvus_token"])` in `__init__`. Token empty string = no auth.

---

### `get_store()` Dispatch (Updated)

```python
def get_store(cfg: dict):
    backend = cfg.get("vector_store", "qdrant")

    if backend in _custom_registry:
        return _custom_registry[backend](cfg)

    if backend == "qdrant":
        from vectorstore.qdrant_store import QdrantStore
        return QdrantStore(host=cfg.get("qdrant_host", "localhost"),
                           port=cfg.get("qdrant_port", 6333))
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

---

## What Does Not Change

- `vectorstore/base.py` — Protocol and dataclasses unchanged; no new methods
- `vectorstore/qdrant_store.py`, `chroma_store.py`, `faiss_store.py` — untouched
- All pipeline modules (`kv_indexer.py`, `kv_inference.py`, etc.) — already backend-agnostic via `get_store(cfg)`
- All existing tests — new backends are purely additive

---

## New Files

| File | Purpose |
|---|---|
| `vectorstore/pinecone_store.py` | Pinecone serverless backend |
| `vectorstore/pgvector_store.py` | PostgreSQL + pgvector backend |
| `vectorstore/weaviate_store.py` | Weaviate v4 backend |
| `vectorstore/milvus_store.py` | Milvus / Zilliz Cloud backend |

## Modified Files

| File | Change |
|---|---|
| `vectorstore/registry.py` | `_custom_registry`, `register_store()`, four new backend dispatches |
| `core/config.py` | Extended `vector_store` Literal + 8 backend-specific fields |

---

## Success Criteria

1. `register_store("mystore", MyClass)` raises `TypeError` immediately if `MyClass` is missing any of the 7 Protocol methods
2. `register_store("qdrant", MyClass)` raises `ValueError` — cannot shadow built-in names
3. All four new backends pass the existing `tests/test_vectorstore.py` parametrized suite with mocked SDK clients
4. A datasource config with `"vector_store": "pinecone"` resolves correctly through `get_store()` without importing pinecone when the backend is not selected
5. `set_payload` on Milvus correctly merges new fields with existing payload without losing unrelated keys
