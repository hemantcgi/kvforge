# SmartQdrant — Frequently Asked Questions

---

## Table of Contents

**Vector Stores**
- [How do I use SmartQdrant with ChromaDB instead of Qdrant?](#how-do-i-use-smartqdrant-with-chromadb-instead-of-qdrant)
- [How do I add support for Pinecone, Weaviate, or another vector database?](#how-do-i-add-support-for-pinecone-weaviate-or-another-vector-database)
- [How do I use SmartQdrant with pgvector (PostgreSQL)?](#how-do-i-use-smartqdrant-with-pgvector-postgresql)
- [How do I use SmartQdrant with FAISS?](#how-do-i-use-smartqdrant-with-faiss)
- [How do I use SmartQdrant with Milvus or Zilliz Cloud?](#how-do-i-use-smartqdrant-with-milvus-or-zilliz-cloud)
- [How do I use SmartQdrant with LanceDB?](#how-do-i-use-smartqdrant-with-lancedb)
- [How do I use SmartQdrant with Redis (RedisSearch)?](#how-do-i-use-smartqdrant-with-redis-redissearch)
- [How do I use SmartQdrant with Elasticsearch or OpenSearch?](#how-do-i-use-smartqdrant-with-elasticsearch-or-opensearch)
- [How do I use SmartQdrant with MongoDB Atlas Vector Search?](#how-do-i-use-smartqdrant-with-mongodb-atlas-vector-search)
- [Can I use an existing Qdrant collection I already have?](#can-i-use-an-existing-qdrant-collection-i-already-have)

**Language Models**
- [How do I use my own LLM for KV computation?](#how-do-i-use-my-own-llm-for-kv-computation)
- [How do I use a gated model like Llama 3 that requires a HuggingFace token?](#how-do-i-use-a-gated-model-like-llama-3-that-requires-a-huggingface-token)
- [Can I use an API-hosted LLM (OpenAI, Anthropic, Gemini)?](#can-i-use-an-api-hosted-llm-openai-anthropic-gemini)
- [Can I run this without a GPU?](#can-i-run-this-without-a-gpu)

**Embedding Models**
- [How do I use OpenAI embeddings instead of FastEmbed?](#how-do-i-use-openai-embeddings-instead-of-fastembed)
- [How do I use sentence-transformers embeddings?](#how-do-i-use-sentence-transformers-embeddings)
- [How do I add a custom embedding model?](#how-do-i-add-a-custom-embedding-model)
- [Can I use different embedding models for different collections?](#can-i-use-different-embedding-models-for-different-collections)

**Document Ingestion**
- [How do I index Markdown documentation?](#how-do-i-index-markdown-documentation)
- [How do I index a JSONL dataset?](#how-do-i-index-a-jsonl-dataset)
- [How do I index HTML pages or web content?](#how-do-i-index-html-pages-or-web-content)
- [How do I index an entire directory of mixed file types?](#how-do-i-index-an-entire-directory-of-mixed-file-types)
- [How do I add support for a custom document format?](#how-do-i-add-support-for-a-custom-document-format)

**KV Cache & Phases**
- [What exactly is stored in the KV cache payload?](#what-exactly-is-stored-in-the-kv-cache-payload)
- [How does KV injection work under the hood?](#how-does-kv-injection-work-under-the-hood)
- [Why do some queries fall back to text-in-context even in Phase 2?](#why-do-some-queries-fall-back-to-text-in-context-even-in-phase-2)
- [How do I manually advance or roll back the phase?](#how-do-i-manually-advance-or-roll-back-the-phase)

**Training & PRS**
- [How do I tune the PRS threshold?](#how-do-i-tune-the-prs-threshold)
- [My PRS is not improving across training rounds — what do I do?](#my-prs-is-not-improving-across-training-rounds--what-do-i-do)
- [How do I bring my own FAQs for PRS evaluation?](#how-do-i-bring-my-own-faqs-for-prs-evaluation)
- [How do I change the PRS scoring weights?](#how-do-i-change-the-prs-scoring-weights)

**Multi-Corpus & Production**
- [Can I run multiple independent corpora on the same instance?](#can-i-run-multiple-independent-corpora-on-the-same-instance)
- [How do I keep KV tensors fresh when I update my documents?](#how-do-i-keep-kv-tensors-fresh-when-i-update-my-documents)
- [How do I monitor what is happening at runtime?](#how-do-i-monitor-what-is-happening-at-runtime)
- [How do I reset everything and start over?](#how-do-i-reset-everything-and-start-over)
- [What are the GPU memory requirements?](#what-are-the-gpu-memory-requirements)

---

## Vector Stores

### How do I use SmartQdrant with ChromaDB instead of Qdrant?

ChromaDB is a lightweight, in-process vector database with no Docker dependency. It is a good choice for local development, laptops, and single-machine deployments where you want to avoid running a separate Qdrant service.

#### Step 1 — Install dependencies

```bash
# Core SmartQdrant dependencies
pip install fastembed pypdf fastapi uvicorn httpx pydantic pytest

# ChromaDB (replaces qdrant-client for storage)
pip install chromadb
```

> You do **not** need `qdrant-client` installed if you are using ChromaDB exclusively. The two backends are independent. However, `qdrant-client` is still needed if you want to run the full test suite (some tests mock it directly).

#### Step 2 — Create a datasource config pointing to ChromaDB

Run `init` to scaffold a new config file:

```bash
python smartqdrant.py init --name my-corpus
```

This creates **`datasource_my-corpus.json`** in your current directory with default Qdrant settings. Open that file and make the following changes — set `vector_store` to `"chroma"` and add `chroma_persist_dir`:

Two fields require changes from the Qdrant defaults, and one new field must be added:

| Field | Default (Qdrant) | Change to (ChromaDB) | Notes |
|-------|-----------------|----------------------|-------|
| `vector_store` | `"qdrant"` | `"chroma"` | Selects the ChromaDB backend |
| `qdrant_host` / `qdrant_port` | present | remove or leave | Ignored when `vector_store` is `"chroma"` |
| `chroma_persist_dir` | not present | **add** `".chroma/my-corpus"` | Directory where ChromaDB writes its SQLite + parquet files |

Complete `datasource_my-corpus.json` after editing:

```json
{
  "collection":         "my-corpus",
  "vector_store":       "chroma",
  "chroma_persist_dir": ".chroma/my-corpus",

  "embed_model":        "BAAI/bge-small-en-v1.5",
  "embedder_backend":   "fastembed",
  "vector_dim":         384,

  "llm_model":          "meta-llama/Llama-3.2-3B-Instruct",
  "chunk_size":         600,
  "chunk_overlap":      60,
  "top_k":              5,

  "checkpoint_dir":     "lora_checkpoints/my-corpus/",
  "version_file":       "my-corpus_version.json",
  "replay_db":          "my-corpus_replay.db"
}
```

`chroma_persist_dir` is where ChromaDB writes its SQLite and parquet files. It is created automatically on first use. You can set it to any path. Multiple corpora should use separate directories (e.g. `.chroma/legal`, `.chroma/hr`) to avoid collection name collisions.

#### Step 3 — Index your documents

```bash
# Index a PDF
python smartqdrant.py index \
  --config datasource_my-corpus.json \
  --source ./my_document.pdf

# Index a directory of markdown files
python smartqdrant.py index \
  --config datasource_my-corpus.json \
  --source ./docs/
```

You will see output like:

```
Loading documents from ./my_document.pdf...
Loaded 142 chunks
Embedding 142 chunks...
Indexed 142 points into 'my-corpus'
```

#### Step 4 — Search

```bash
python smartqdrant.py search \
  --config datasource_my-corpus.json \
  "What is the maximum retention period?"
```

#### Step 5 — Verify the collection on disk

```python
import chromadb

client = chromadb.PersistentClient(path=".chroma/my-corpus")
col = client.get_collection("my-corpus")
print(f"Collection has {col.count()} documents")
print(col.peek(5))   # show first 5 entries
```

#### Step 6 — Use ChromaDB in Python code

```python
import json
from embeddings.registry import get_embedder
from vectorstore.registry import get_store

with open("datasource_my-corpus.json") as f:
    cfg = json.load(f)

store = get_store(cfg)   # returns ChromaStore
embedder = get_embedder(cfg)

query_vec = embedder.encode(["How do I reset my password?"])[0]
results = store.query("my-corpus", query_vec, top_k=5)

for r in results:
    print(f"score={r.score:.4f}  {r.payload['text'][:120]}")
```

#### Limitations compared to Qdrant

| Feature | Qdrant | ChromaDB |
|---------|--------|----------|
| Docker required | Yes | No |
| Payload filtering | Full (used by `compute-kv`) | Limited |
| Horizontal scaling | Yes | No |
| Remote access | Yes (REST + gRPC) | In-process only |
| Production workloads | Recommended | Development / single-machine |
| `kv_indexer.py compute-kv` filtering | Efficient (index filter) | Full scan (slower) |

> **KV computation note:** `kv_indexer.py compute-kv` uses Qdrant-specific `IsNullCondition` scroll filters to find chunks without KV tensors. With ChromaDB, it falls back to a full collection scan and filters in Python. For corpora under ~5,000 chunks this is fine. For larger corpora, switch to Qdrant for KV computation, even if you keep ChromaDB for search.

#### Troubleshooting

**`ModuleNotFoundError: No module named 'chromadb'`**
```bash
pip install chromadb
```

**`Collection 'my-corpus' does not exist`**
The collection is created on first `index`. You cannot `search` an empty collection. Run `index` first.

**`sqlite3.OperationalError: database is locked`**
Another process has the `.chroma` directory open. ChromaDB uses SQLite which allows only one writer at a time. Stop the other process or use separate `chroma_persist_dir` paths.

**Collection not found after restart**
Make sure `chroma_persist_dir` in your config matches the path used when indexing. If the directory does not exist or points elsewhere, ChromaDB will create a new empty collection.

---

### How do I add support for Pinecone, Weaviate, or another vector database?

SmartQdrant's `VectorStore` protocol (`vectorstore/base.py`) requires eight methods. Implement all eight in a new file, register it in the factory, and set the backend name in your config. No other files need to change.

#### The VectorStore protocol

```python
# vectorstore/base.py  (already in the repo — shown here for reference)
class VectorStore(Protocol):
    def create_collection(self, name: str, dim: int) -> None: ...
    def collection_exists(self, name: str) -> bool: ...
    def delete_collection(self, name: str) -> None: ...
    def upsert(self, collection: str, points: list[Point]) -> None: ...
    def query(self, collection: str, vector: list[float], top_k: int,
               score_threshold: float | None = None) -> list[ScoredPoint]: ...
    def scroll(self, collection: str, limit: int = 100,
                with_payload: bool = True, with_vectors: bool = False,
                offset=None, scroll_filter=None) -> tuple[list, Any]: ...
    def set_payload(self, collection: str, point_id: int | str,
                     payload: dict) -> None: ...
    def count(self, collection: str) -> int: ...
```

#### Example: Pinecone

```bash
pip install pinecone-client
```

Create a Pinecone index in the [Pinecone console](https://app.pinecone.io) with dimension matching your `vector_dim` and metric `cosine`. Then:

```python
# vectorstore/pinecone_store.py
from vectorstore.base import Point, ScoredPoint


class PineconeStore:
    """Pinecone serverless vector store backend."""

    def __init__(self, api_key: str, index_name: str):
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError("PineconeStore requires: pip install pinecone-client")
        pc = Pinecone(api_key=api_key)
        self._index = pc.Index(index_name)
        self._index_name = index_name

    def create_collection(self, name: str, dim: int) -> None:
        # Pinecone indexes are managed in the console / API — this is a no-op.
        # Raise if the index does not exist so misconfiguration is caught early.
        stats = self._index.describe_index_stats()
        if stats.total_vector_count == 0:
            print(f"[pinecone] Index '{self._index_name}' is empty and ready.")

    def collection_exists(self, name: str) -> bool:
        return True  # Pinecone index existence is verified in __init__

    def delete_collection(self, name: str) -> None:
        self._index.delete(delete_all=True)

    def upsert(self, collection: str, points: list[Point]) -> None:
        vectors = [
            {"id": str(p.id), "values": p.vector, "metadata": p.payload}
            for p in points
        ]
        self._index.upsert(vectors=vectors)

    def query(self, collection: str, vector: list[float], top_k: int,
               score_threshold: float | None = None) -> list[ScoredPoint]:
        resp = self._index.query(vector=vector, top_k=top_k, include_metadata=True)
        results = []
        for m in resp.matches:
            if score_threshold is not None and m.score < score_threshold:
                continue
            results.append(ScoredPoint(id=m.id, score=m.score,
                                        payload=m.metadata or {}))
        return results

    def scroll(self, collection: str, limit: int = 100,
                with_payload: bool = True, with_vectors: bool = False,
                offset=None, scroll_filter=None) -> tuple[list, None]:
        # Pinecone does not support full-collection iteration.
        # KV backfill via compute-kv is not available with Pinecone.
        return [], None

    def set_payload(self, collection: str, point_id: int | str,
                     payload: dict) -> None:
        self._index.update(id=str(point_id), set_metadata=payload)

    def count(self, collection: str) -> int:
        return self._index.describe_index_stats().total_vector_count
```

**Register in `vectorstore/registry.py`:**

Open `vectorstore/registry.py` and add an `elif` block for Pinecone before the final `raise ValueError`. The complete file should look like this:

```python
# vectorstore/registry.py
def get_store(cfg: dict):
    backend = cfg.get("vector_store", "qdrant")
    if backend == "qdrant":
        from vectorstore.qdrant_store import QdrantStore
        return QdrantStore(
            host=cfg.get("qdrant_host", "localhost"),
            port=cfg.get("qdrant_port", 6333),
        )
    if backend == "chroma":
        from vectorstore.chroma_store import ChromaStore
        return ChromaStore(persist_dir=cfg.get("chroma_persist_dir", ".chroma"))
    elif backend == "pinecone":          # ← add this block
        from vectorstore.pinecone_store import PineconeStore
        return PineconeStore(
            api_key=cfg["pinecone_api_key"],
            index_name=cfg.get("pinecone_index_name", cfg["collection"]),
        )
    raise ValueError(
        f"Unknown vector_store '{backend}'. Choose: qdrant, chroma, pinecone"
    )
```

**Datasource config** — run `python smartqdrant.py init --name my-corpus` to create `datasource_my-corpus.json`, then edit it to add the Pinecone fields. The minimum required changes:

```json
{
  "collection":          "my-corpus",
  "vector_store":        "pinecone",
  "pinecone_api_key":    "pcsk_...",
  "pinecone_index_name": "my-corpus",
  "vector_dim":          384,

  "embed_model":         "BAAI/bge-small-en-v1.5",
  "embedder_backend":    "fastembed",
  "llm_model":           "meta-llama/Llama-3.2-3B-Instruct",
  "chunk_size":          600,
  "chunk_overlap":       60,
  "top_k":               5,
  "checkpoint_dir":      "lora_checkpoints/my-corpus/",
  "version_file":        "my-corpus_version.json",
  "replay_db":           "my-corpus_replay.db"
}
```

#### Example: Weaviate

```bash
pip install weaviate-client
```

```python
# vectorstore/weaviate_store.py
from vectorstore.base import Point, ScoredPoint


class WeaviateStore:
    """Weaviate vector store backend."""

    def __init__(self, url: str = "http://localhost:8080", api_key: str | None = None):
        try:
            import weaviate
        except ImportError:
            raise ImportError("WeaviateStore requires: pip install weaviate-client")
        auth = weaviate.auth.AuthApiKey(api_key) if api_key else None
        self._client = weaviate.connect_to_custom(http_host=url, auth_credentials=auth)

    def _class_name(self, name: str) -> str:
        # Weaviate class names must start with uppercase
        return name[0].upper() + name[1:]

    def create_collection(self, name: str, dim: int) -> None:
        from weaviate.classes.config import Configure, Property, DataType
        cls = self._class_name(name)
        if not self._client.collections.exists(cls):
            self._client.collections.create(
                name=cls,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[Property(name="text", data_type=DataType.TEXT)],
            )

    def collection_exists(self, name: str) -> bool:
        return self._client.collections.exists(self._class_name(name))

    def delete_collection(self, name: str) -> None:
        self._client.collections.delete(self._class_name(name))

    def upsert(self, collection: str, points: list[Point]) -> None:
        col = self._client.collections.get(self._class_name(collection))
        with col.batch.dynamic() as batch:
            for p in points:
                batch.add_object(properties=p.payload, vector=p.vector,
                                  uuid=str(p.id))

    def query(self, collection: str, vector: list[float], top_k: int,
               score_threshold: float | None = None) -> list[ScoredPoint]:
        from weaviate.classes.query import MetadataQuery
        col = self._client.collections.get(self._class_name(collection))
        resp = col.query.near_vector(near_vector=vector, limit=top_k,
                                      return_metadata=MetadataQuery(distance=True))
        results = []
        for obj in resp.objects:
            score = 1.0 - obj.metadata.distance
            if score_threshold is not None and score < score_threshold:
                continue
            results.append(ScoredPoint(id=str(obj.uuid), score=score,
                                        payload=obj.properties))
        return results

    def scroll(self, collection: str, limit: int = 100,
                with_payload: bool = True, with_vectors: bool = False,
                offset=None, scroll_filter=None) -> tuple[list, any]:
        col = self._client.collections.get(self._class_name(collection))
        resp = col.query.fetch_objects(limit=limit, offset=offset or 0)
        out = [ScoredPoint(id=str(o.uuid), score=0.0, payload=o.properties)
               for o in resp.objects]
        next_offset = (offset or 0) + len(out) if len(out) == limit else None
        return out, next_offset

    def set_payload(self, collection: str, point_id: int | str,
                     payload: dict) -> None:
        col = self._client.collections.get(self._class_name(collection))
        col.data.update(uuid=str(point_id), properties=payload)

    def count(self, collection: str) -> int:
        col = self._client.collections.get(self._class_name(collection))
        return col.aggregate.over_all(total_count=True).total_count
```

**Register in `vectorstore/registry.py`:**

Open `vectorstore/registry.py` and add a Weaviate `elif` block. Add it after the last existing backend and before the `raise ValueError`:

```python
# vectorstore/registry.py
def get_store(cfg: dict):
    backend = cfg.get("vector_store", "qdrant")
    if backend == "qdrant":
        from vectorstore.qdrant_store import QdrantStore
        return QdrantStore(
            host=cfg.get("qdrant_host", "localhost"),
            port=cfg.get("qdrant_port", 6333),
        )
    if backend == "chroma":
        from vectorstore.chroma_store import ChromaStore
        return ChromaStore(persist_dir=cfg.get("chroma_persist_dir", ".chroma"))
    elif backend == "weaviate":          # ← add this block
        from vectorstore.weaviate_store import WeaviateStore
        return WeaviateStore(
            url=cfg.get("weaviate_url", "http://localhost:8080"),
            api_key=cfg.get("weaviate_api_key"),
        )
    raise ValueError(
        f"Unknown vector_store '{backend}'. Choose: qdrant, chroma, weaviate"
    )
```

**Datasource config** — run `python smartqdrant.py init --name my-corpus` to create `datasource_my-corpus.json`, then add/change these fields:

```json
{
  "collection":       "my-corpus",
  "vector_store":     "weaviate",
  "weaviate_url":     "http://localhost:8080",
  "weaviate_api_key": "",

  "embed_model":      "BAAI/bge-small-en-v1.5",
  "embedder_backend": "fastembed",
  "vector_dim":       384,
  "llm_model":        "meta-llama/Llama-3.2-3B-Instruct",
  "chunk_size":       600,
  "chunk_overlap":    60,
  "top_k":            5,
  "checkpoint_dir":   "lora_checkpoints/my-corpus/",
  "version_file":     "my-corpus_version.json",
  "replay_db":        "my-corpus_replay.db"
}
```

#### Checklist for any new backend

- [ ] `scroll()` returns a next-page cursor (or `None` when exhausted) — this drives `compute-kv` backfill
- [ ] `set_payload()` merges into existing payload without overwriting unrelated fields
- [ ] `query()` returns `ScoredPoint` objects with `payload["text"]` populated — used by text-in-context fallback
- [ ] IDs are stable across upsert calls — `compute-kv` looks up chunks by the same ID used at index time

---

### How do I use SmartQdrant with pgvector (PostgreSQL)?

pgvector adds vector similarity search to PostgreSQL. If you are already running Postgres, this is the lowest-friction path to adding vector search — no extra Docker container, no new service.

#### Step 1a — Start PostgreSQL with pgvector

If you already have PostgreSQL 13+ running, skip ahead to Step 1b.

To start a new instance using Docker:

```bash
docker run -d \
  --name pgvector \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=smartqdrant \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

Wait a few seconds for the container to start, then verify it is running:

```bash
docker ps | grep pgvector
```

#### Step 1b — Enable the pgvector extension

Run this once per database. The `pgvector/pgvector:pg16` Docker image ships with the extension pre-installed — you just need to activate it:

```bash
docker exec -it pgvector psql -U postgres -d smartqdrant \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

If you are using an existing PostgreSQL server (not Docker), connect to your target database and run:

```sql
-- Requires superuser or the CREATE privilege on the database
CREATE EXTENSION IF NOT EXISTS vector;
```

#### Step 1c — Install Python dependencies

```bash
pip install psycopg2-binary pgvector
```

#### Step 2 — Create the backend file

Create a new file at `vectorstore/pgvector_store.py` in the SmartQdrant repository root:

```python
# vectorstore/pgvector_store.py
from __future__ import annotations
from typing import Any
from vectorstore.base import Point, ScoredPoint


class PgvectorStore:
    """PostgreSQL pgvector backend for SmartQdrant."""

    def __init__(self, dsn: str):
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
        except ImportError:
            raise ImportError("PgvectorStore requires: pip install psycopg2-binary pgvector")
        self._dsn = dsn
        self._conn = psycopg2.connect(dsn)
        register_vector(self._conn)
        self._conn.autocommit = True

    def _table(self, name: str) -> str:
        # Sanitize collection name to a safe table name
        return "sq_" + name.replace("-", "_").replace(".", "_")

    def create_collection(self, name: str, dim: int) -> None:
        table = self._table(name)
        with self._conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id      TEXT PRIMARY KEY,
                    vector  vector({dim}),
                    payload JSONB DEFAULT '{{}}'::jsonb
                )
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {table}_vec_idx
                ON {table} USING ivfflat (vector vector_cosine_ops)
                WITH (lists = 100)
            """)

    def collection_exists(self, name: str) -> bool:
        table = self._table(name)
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                "WHERE table_name = %s)", (table,)
            )
            return cur.fetchone()[0]

    def delete_collection(self, name: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self._table(name)}")

    def upsert(self, collection: str, points: list[Point]) -> None:
        import json
        table = self._table(collection)
        with self._conn.cursor() as cur:
            for p in points:
                cur.execute(
                    f"INSERT INTO {table} (id, vector, payload) VALUES (%s, %s, %s) "
                    "ON CONFLICT (id) DO UPDATE SET vector=EXCLUDED.vector, payload=EXCLUDED.payload",
                    (str(p.id), p.vector, json.dumps(p.payload))
                )

    def query(self, collection: str, vector: list[float], top_k: int,
               score_threshold: float | None = None) -> list[ScoredPoint]:
        import json
        table = self._table(collection)
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, payload, 1 - (vector <=> %s::vector) AS score "
                f"FROM {table} ORDER BY vector <=> %s::vector LIMIT %s",
                (vector, vector, top_k)
            )
            rows = cur.fetchall()
        results = []
        for row in rows:
            score = float(row[2])
            if score_threshold and score < score_threshold:
                continue
            payload = row[1] if isinstance(row[1], dict) else json.loads(row[1])
            results.append(ScoredPoint(id=row[0], score=score, payload=payload))
        return results

    def scroll(self, collection: str, limit: int = 100, with_payload: bool = True,
                with_vectors: bool = False, offset=None, scroll_filter=None
               ) -> tuple[list[ScoredPoint], Any]:
        import json
        table = self._table(collection)
        where = f"WHERE id > %s" if offset else ""
        params = [offset] if offset else []
        params.append(limit + 1)
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT id, payload FROM {table} {where} ORDER BY id LIMIT %s", params)
            rows = cur.fetchall()
        has_more = len(rows) > limit
        rows = rows[:limit]
        points = [ScoredPoint(id=r[0], score=0.0,
                               payload=r[1] if isinstance(r[1], dict) else json.loads(r[1]))
                  for r in rows]
        next_offset = rows[-1][0] if has_more else None
        return points, next_offset

    def set_payload(self, collection: str, point_id: int | str, payload: dict) -> None:
        import json
        table = self._table(collection)
        with self._conn.cursor() as cur:
            cur.execute(
                f"UPDATE {table} SET payload = payload || %s::jsonb WHERE id = %s",
                (json.dumps(payload), str(point_id))
            )

    def count(self, collection: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table(collection)}")
            return cur.fetchone()[0]
```

#### Step 3 — Register in the factory

Open `vectorstore/registry.py` and add a `pgvector` elif block. The complete file should look like this after editing:

```python
# vectorstore/registry.py
def get_store(cfg: dict):
    backend = cfg.get("vector_store", "qdrant")
    if backend == "qdrant":
        from vectorstore.qdrant_store import QdrantStore
        return QdrantStore(
            host=cfg.get("qdrant_host", "localhost"),
            port=cfg.get("qdrant_port", 6333),
        )
    if backend == "chroma":
        from vectorstore.chroma_store import ChromaStore
        return ChromaStore(persist_dir=cfg.get("chroma_persist_dir", ".chroma"))
    elif backend == "pgvector":          # ← add this block
        from vectorstore.pgvector_store import PgvectorStore
        return PgvectorStore(dsn=cfg["pgvector_dsn"])
    raise ValueError(
        f"Unknown vector_store '{backend}'. Choose: qdrant, chroma, pgvector"
    )
```

#### Step 4 — Create a datasource config

Run `init` to create `datasource_my-corpus.json`:

```bash
python smartqdrant.py init --name my-corpus
```

This generates a default Qdrant config. Open **`datasource_my-corpus.json`** and make these changes:

| Field | Action | Value |
|-------|--------|-------|
| `vector_store` | change from `"qdrant"` | `"pgvector"` |
| `pgvector_dsn` | **add** (not in default) | `"postgresql://postgres:secret@localhost:5432/smartqdrant"` |
| `qdrant_host`, `qdrant_port` | leave or remove | Ignored when using pgvector |

The DSN format is: `postgresql://[user]:[password]@[host]:[port]/[database]`

Complete `datasource_my-corpus.json` after editing:

```json
{
  "collection":      "my-corpus",
  "vector_store":    "pgvector",
  "pgvector_dsn":    "postgresql://postgres:secret@localhost:5432/smartqdrant",

  "embed_model":     "BAAI/bge-small-en-v1.5",
  "embedder_backend":"fastembed",
  "vector_dim":      384,

  "llm_model":       "meta-llama/Llama-3.2-3B-Instruct",
  "chunk_size":      600,
  "chunk_overlap":   60,
  "top_k":           5,

  "checkpoint_dir":  "lora_checkpoints/my-corpus/",
  "version_file":    "my-corpus_version.json",
  "replay_db":       "my-corpus_replay.db"
}
```

#### Step 5 — Index and search

```bash
python smartqdrant.py index --config datasource_my-corpus.json --source ./docs/
python smartqdrant.py search --config datasource_my-corpus.json "What is the return policy?"
```

#### Step 6 — Verify the collection in PostgreSQL

```bash
docker exec -it pgvector psql -U postgres -d smartqdrant \
  -c "SELECT COUNT(*) FROM sq_my_corpus;"
```

Or from Python:

```python
import psycopg2
conn = psycopg2.connect("postgresql://postgres:secret@localhost:5432/smartqdrant")
cur = conn.cursor()
cur.execute("SELECT id, payload->>'text' FROM sq_my_corpus LIMIT 3")
for row in cur.fetchall():
    print(row[0], row[1][:80])
```

#### Limitations compared to Qdrant

| Feature | Qdrant | pgvector |
|---------|--------|----------|
| Docker required | Yes | Yes (or managed Postgres) |
| Native vector index | HNSW (fast) | IVFFlat / HNSW (slower for very large) |
| Payload filtering | Full | Full (SQL WHERE) |
| Horizontal scaling | Yes | Postgres replication |
| Production workloads | Recommended | Good for ≤10M vectors |
| Existing Postgres | No | Yes — zero new infra |
| `compute-kv` scroll | Efficient | Cursor-based (efficient) |

#### Troubleshooting

**`ERROR: type "vector" does not exist`**
The extension is not installed. Run `CREATE EXTENSION vector;` as a superuser in your database.

**`IVFFlat index requires at least one row`**
The IVFFlat index cannot be created on an empty table. It is created as `IF NOT EXISTS` so it will be built on next upsert that triggers a re-index. For very small datasets this is not an issue.

**`psycopg2.OperationalError: could not connect to server`**
Check your `pgvector_dsn` connection string and ensure PostgreSQL is running.

**`UnicodeDecodeError` on payload**
Ensure your documents use UTF-8 encoding. Pass `client_encoding='UTF8'` in the DSN if needed: `postgresql://...?client_encoding=UTF8`.

---

### How do I use SmartQdrant with FAISS?

FAISS (Facebook AI Similarity Search) is an in-process library — no server, no Docker, no network. It stores everything in memory and optionally saves/loads from disk. It is ideal for offline batch workflows, laptop development, and datasets up to a few million vectors.

#### Step 1 — Install dependencies

```bash
# CPU-only (works everywhere)
pip install faiss-cpu

# GPU version (CUDA required)
pip install faiss-gpu
```

#### Step 2 — Create the backend file

Create `vectorstore/faiss_store.py`:

```python
# vectorstore/faiss_store.py
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Any
from vectorstore.base import Point, ScoredPoint


class FAISSStore:
    """FAISS in-process vector store backend for SmartQdrant."""

    def __init__(self, persist_dir: str = ".faiss"):
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            raise ImportError("FAISSStore requires: pip install faiss-cpu")
        self._root = Path(persist_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._indexes: dict[str, Any] = {}
        self._payloads: dict[str, dict[str, dict]] = {}  # collection -> id -> payload
        self._id_map: dict[str, list[str]] = {}          # collection -> ordered IDs

    def _paths(self, name: str):
        return (self._root / f"{name}.index",
                self._root / f"{name}.meta.pkl")

    def _load(self, name: str):
        idx_path, meta_path = self._paths(name)
        if idx_path.exists():
            self._indexes[name] = self._faiss.read_index(str(idx_path))
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self._payloads[name] = meta["payloads"]
            self._id_map[name] = meta["id_map"]

    def _save(self, name: str):
        idx_path, meta_path = self._paths(name)
        self._faiss.write_index(self._indexes[name], str(idx_path))
        with open(meta_path, "wb") as f:
            pickle.dump({"payloads": self._payloads[name],
                         "id_map": self._id_map[name]}, f)

    def create_collection(self, name: str, dim: int) -> None:
        index = self._faiss.IndexFlatIP(dim)   # inner product ≈ cosine on normalized vecs
        self._indexes[name] = index
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
        import numpy as np
        if collection not in self._indexes:
            self._load(collection)
        index = self._indexes[collection]
        for p in points:
            sid = str(p.id)
            vec = np.array([p.vector], dtype="float32")
            # normalize for cosine similarity
            self._faiss.normalize_L2(vec)
            if sid not in self._payloads[collection]:
                index.add(vec)
                self._id_map[collection].append(sid)
            self._payloads[collection][sid] = p.payload
        self._save(collection)

    def query(self, collection: str, vector: list[float], top_k: int,
               score_threshold: float | None = None) -> list[ScoredPoint]:
        import numpy as np
        if collection not in self._indexes:
            self._load(collection)
        vec = np.array([vector], dtype="float32")
        self._faiss.normalize_L2(vec)
        scores, indices = self._indexes[collection].search(vec, top_k)
        results = []
        id_map = self._id_map[collection]
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if score_threshold and float(score) < score_threshold:
                continue
            sid = id_map[idx]
            results.append(ScoredPoint(id=sid, score=float(score),
                                        payload=self._payloads[collection].get(sid, {})))
        return results

    def scroll(self, collection: str, limit: int = 100, with_payload: bool = True,
                with_vectors: bool = False, offset=None, scroll_filter=None
               ) -> tuple[list[ScoredPoint], Any]:
        if collection not in self._indexes:
            self._load(collection)
        id_map = self._id_map[collection]
        start = 0
        if offset is not None:
            try:
                start = id_map.index(str(offset)) + 1
            except ValueError:
                start = 0
        batch = id_map[start:start + limit]
        points = [ScoredPoint(id=sid, score=0.0,
                               payload=self._payloads[collection].get(sid, {}))
                  for sid in batch]
        next_offset = id_map[start + limit] if start + limit < len(id_map) else None
        return points, next_offset

    def set_payload(self, collection: str, point_id: int | str, payload: dict) -> None:
        if collection not in self._payloads:
            self._load(collection)
        sid = str(point_id)
        self._payloads[collection].setdefault(sid, {}).update(payload)
        self._save(collection)

    def count(self, collection: str) -> int:
        if collection not in self._indexes:
            self._load(collection)
        return self._indexes[collection].ntotal
```

#### Step 3 — Register in the factory

Open `vectorstore/registry.py` and add a `faiss` elif block. The complete file after editing:

```python
# vectorstore/registry.py
def get_store(cfg: dict):
    backend = cfg.get("vector_store", "qdrant")
    if backend == "qdrant":
        from vectorstore.qdrant_store import QdrantStore
        return QdrantStore(
            host=cfg.get("qdrant_host", "localhost"),
            port=cfg.get("qdrant_port", 6333),
        )
    if backend == "chroma":
        from vectorstore.chroma_store import ChromaStore
        return ChromaStore(persist_dir=cfg.get("chroma_persist_dir", ".chroma"))
    elif backend == "faiss":             # ← add this block
        from vectorstore.faiss_store import FAISSStore
        return FAISSStore(persist_dir=cfg.get("faiss_persist_dir", ".faiss"))
    raise ValueError(
        f"Unknown vector_store '{backend}'. Choose: qdrant, chroma, faiss"
    )
```

#### Step 4 — Create a datasource config

Run `python smartqdrant.py init --name my-corpus` to create **`datasource_my-corpus.json`**. Then open it and make the following changes:

| Field | Action | Value |
|-------|--------|-------|
| `vector_store` | change | `"faiss"` |
| `faiss_persist_dir` | **add** (not in default) | `".faiss/my-corpus"` |

Complete config after editing:

```json
{
  "collection":       "my-corpus",
  "vector_store":     "faiss",
  "faiss_persist_dir":".faiss/my-corpus",

  "embed_model":      "BAAI/bge-small-en-v1.5",
  "embedder_backend": "fastembed",
  "vector_dim":       384,

  "llm_model":        "meta-llama/Llama-3.2-3B-Instruct",
  "chunk_size":       600,
  "chunk_overlap":    60,
  "top_k":            5,

  "checkpoint_dir":   "lora_checkpoints/my-corpus/",
  "version_file":     "my-corpus_version.json",
  "replay_db":        "my-corpus_replay.db"
}
```

#### Step 5 — Index and search

```bash
python smartqdrant.py index --config datasource_my-corpus.json --source ./docs/
python smartqdrant.py search --config datasource_my-corpus.json "explain the refund process"
```

After indexing, two files are written per collection:
- `.faiss/my-corpus/my-corpus.index` — the FAISS binary index
- `.faiss/my-corpus/my-corpus.meta.pkl` — ID map and payloads

#### Step 6 — Verify

```python
import faiss, pickle

index = faiss.read_index(".faiss/my-corpus/my-corpus.index")
print(f"Vectors stored: {index.ntotal}")

with open(".faiss/my-corpus/my-corpus.meta.pkl", "rb") as f:
    meta = pickle.load(f)
print(f"IDs: {meta['id_map'][:5]}")
```

#### Limitations compared to Qdrant

| Feature | Qdrant | FAISS |
|---------|--------|-------|
| Server required | Yes | No |
| Payload filtering | Full | None (post-filter in Python) |
| Concurrent writers | Yes | No (single process) |
| Memory usage | Low (disk-backed) | All vectors in RAM |
| KV `compute-kv` | Efficient | Full scan (in-memory, fast) |
| Index update | In-place | Rebuild on upsert |
| Production | Yes | Batch/offline/dev only |

#### Troubleshooting

**`ModuleNotFoundError: No module named 'faiss'`**
```bash
pip install faiss-cpu   # or faiss-gpu
```

**`AssertionError` from FAISS on upsert**
Vectors must be `float32`. The store normalizes them automatically, but if you are calling the FAISS API directly ensure `dtype="float32"`.

**Index grows but scores drop**
FAISS `IndexFlatIP` supports adding vectors but not deleting or updating them. The `upsert` implementation above adds a new vector for new IDs but cannot replace existing ones. For a dataset that changes frequently, delete the collection and re-index.

**High memory usage**
Each `float32` vector of dimension 384 uses 1.5 KB. 1 million vectors = ~1.5 GB RAM. Use `IndexIVFFlat` (requires training) for larger datasets.

---

### How do I use SmartQdrant with Milvus or Zilliz Cloud?

Milvus is a production-grade, horizontally scalable vector database. Zilliz Cloud is the fully managed version. Both use the same Python SDK (`pymilvus`).

#### Step 1 — Install dependencies

```bash
pip install pymilvus
```

**Local Milvus (Docker):**

```bash
# Milvus standalone (single node)
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -v $(pwd)/milvus_data:/var/lib/milvus \
  milvusdb/milvus:v2.4.0 \
  milvus run standalone
```

**Zilliz Cloud:** Create a free cluster at [cloud.zilliz.com](https://cloud.zilliz.com) and copy the endpoint + API key.

#### Step 2 — Create the backend file

Create `vectorstore/milvus_store.py`:

```python
# vectorstore/milvus_store.py
from __future__ import annotations
from typing import Any
from vectorstore.base import Point, ScoredPoint


class MilvusStore:
    """Milvus / Zilliz Cloud vector store backend."""

    def __init__(self, uri: str, token: str = "", db_name: str = "default"):
        try:
            from pymilvus import MilvusClient
        except ImportError:
            raise ImportError("MilvusStore requires: pip install pymilvus")
        self._client = MilvusClient(uri=uri, token=token, db_name=db_name)

    def create_collection(self, name: str, dim: int) -> None:
        from pymilvus import DataType
        if self._client.has_collection(name):
            return
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id",     DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        index_params = self._client.prepare_index_params()
        index_params.add_index("vector", metric_type="COSINE", index_type="HNSW",
                                params={"M": 16, "efConstruction": 200})
        self._client.create_collection(name, schema=schema, index_params=index_params)

    def collection_exists(self, name: str) -> bool:
        return self._client.has_collection(name)

    def delete_collection(self, name: str) -> None:
        if self._client.has_collection(name):
            self._client.drop_collection(name)

    def upsert(self, collection: str, points: list[Point]) -> None:
        import json
        data = [{"id": str(p.id), "vector": p.vector,
                 "payload": json.dumps(p.payload)} for p in points]
        self._client.upsert(collection_name=collection, data=data)

    def query(self, collection: str, vector: list[float], top_k: int,
               score_threshold: float | None = None) -> list[ScoredPoint]:
        import json
        results = self._client.search(
            collection_name=collection,
            data=[vector],
            limit=top_k,
            output_fields=["payload"],
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        )
        out = []
        for hit in results[0]:
            score = float(hit["distance"])
            if score_threshold and score < score_threshold:
                continue
            payload = json.loads(hit["entity"].get("payload", "{}"))
            out.append(ScoredPoint(id=hit["id"], score=score, payload=payload))
        return out

    def scroll(self, collection: str, limit: int = 100, with_payload: bool = True,
                with_vectors: bool = False, offset=None, scroll_filter=None
               ) -> tuple[list[ScoredPoint], Any]:
        import json
        expr = f'id > "{offset}"' if offset else ""
        rows = self._client.query(collection_name=collection, filter=expr,
                                   output_fields=["id", "payload"], limit=limit + 1)
        has_more = len(rows) > limit
        rows = rows[:limit]
        points = [ScoredPoint(id=r["id"], score=0.0,
                               payload=json.loads(r.get("payload", "{}")))
                  for r in rows]
        next_offset = rows[-1]["id"] if has_more else None
        return points, next_offset

    def set_payload(self, collection: str, point_id: int | str, payload: dict) -> None:
        import json
        existing = self._client.query(collection_name=collection,
                                       filter=f'id == "{point_id}"',
                                       output_fields=["payload"])
        current = json.loads(existing[0]["payload"]) if existing else {}
        current.update(payload)
        self._client.upsert(collection_name=collection,
                             data=[{"id": str(point_id), "payload": json.dumps(current)}])

    def count(self, collection: str) -> int:
        return self._client.get_collection_stats(collection)["row_count"]
```

#### Step 3 — Register in the factory

Open `vectorstore/registry.py` and add a `milvus` elif block. The complete file after editing:

```python
# vectorstore/registry.py
def get_store(cfg: dict):
    backend = cfg.get("vector_store", "qdrant")
    if backend == "qdrant":
        from vectorstore.qdrant_store import QdrantStore
        return QdrantStore(
            host=cfg.get("qdrant_host", "localhost"),
            port=cfg.get("qdrant_port", 6333),
        )
    if backend == "chroma":
        from vectorstore.chroma_store import ChromaStore
        return ChromaStore(persist_dir=cfg.get("chroma_persist_dir", ".chroma"))
    elif backend == "milvus":            # ← add this block
        from vectorstore.milvus_store import MilvusStore
        return MilvusStore(
            uri=cfg.get("milvus_uri", "http://localhost:19530"),
            token=cfg.get("milvus_token", ""),
            db_name=cfg.get("milvus_db", "default"),
        )
    raise ValueError(
        f"Unknown vector_store '{backend}'. Choose: qdrant, chroma, milvus"
    )
```

#### Step 4 — Create a datasource config

Run `python smartqdrant.py init --name my-corpus` to create **`datasource_my-corpus.json`**. Then open it and make changes per your deployment:

| Field | Action | Value |
|-------|--------|-------|
| `vector_store` | change | `"milvus"` |
| `milvus_uri` | **add** | `"http://localhost:19530"` (local) or Zilliz Cloud endpoint |
| `milvus_token` | **add** | `""` (local) or Zilliz API key |

**Local Milvus — complete config:**

```json
{
  "collection":    "my-corpus",
  "vector_store":  "milvus",
  "milvus_uri":    "http://localhost:19530",

  "embed_model":   "BAAI/bge-small-en-v1.5",
  "embedder_backend":"fastembed",
  "vector_dim":    384,
  "llm_model":     "meta-llama/Llama-3.2-3B-Instruct",
  "top_k":         5,
  "chunk_size":    600,
  "chunk_overlap": 60,
  "checkpoint_dir":"lora_checkpoints/my-corpus/",
  "version_file":  "my-corpus_version.json",
  "replay_db":     "my-corpus_replay.db"
}
```

**Zilliz Cloud — complete config** (find your endpoint and API key in the Zilliz Cloud console under **Clusters → Connect**):

```json
{
  "collection":    "my-corpus",
  "vector_store":  "milvus",
  "milvus_uri":    "https://in03-xxxx.api.gcp-us-west1.zillizcloud.com",
  "milvus_token":  "your-zilliz-api-key",

  "embed_model":   "BAAI/bge-small-en-v1.5",
  "embedder_backend":"fastembed",
  "vector_dim":    384,
  "llm_model":     "meta-llama/Llama-3.2-3B-Instruct",
  "top_k":         5,
  "chunk_size":    600,
  "chunk_overlap": 60,
  "checkpoint_dir":"lora_checkpoints/my-corpus/",
  "version_file":  "my-corpus_version.json",
  "replay_db":     "my-corpus_replay.db"
}
```

#### Step 5 — Index and search

```bash
python smartqdrant.py index --config datasource_my-corpus.json --source ./docs/
python smartqdrant.py search --config datasource_my-corpus.json "how do I cancel my subscription"
```

#### Step 6 — Verify

```python
from pymilvus import MilvusClient
client = MilvusClient(uri="http://localhost:19530")
stats = client.get_collection_stats("my-corpus")
print(f"Row count: {stats['row_count']}")
```

#### Limitations compared to Qdrant

| Feature | Qdrant | Milvus |
|---------|--------|--------|
| Docker image size | ~150 MB | ~800 MB |
| HNSW index | Yes | Yes |
| Payload filtering | Full | Full (expr filter) |
| Horizontal scaling | Yes | Yes (better at scale) |
| Cloud managed | Qdrant Cloud | Zilliz Cloud |
| Setup complexity | Low | Medium |

#### Troubleshooting

**`MilvusException: collection not found`**
Run `index` before `search`. Collection is created on first `index`.

**`MilvusException: rate limit exceeded` (Zilliz free tier)**
Free Zilliz clusters have QPS limits. Add a short sleep between requests or upgrade to a paid tier.

**`grpc._channel._InactiveRpcError`**
Milvus is not reachable. Check that the container is running: `docker ps | grep milvus`.

---

### How do I use SmartQdrant with LanceDB?

LanceDB is a serverless, columnar vector database that stores data as Lance files on disk (or S3/GCS). It requires no Docker and is faster than ChromaDB for large datasets due to its columnar format.

#### Step 1 — Install dependencies

```bash
pip install lancedb pyarrow
```

#### Step 2 — Create the backend file

Create `vectorstore/lancedb_store.py`:

```python
# vectorstore/lancedb_store.py
from __future__ import annotations
from typing import Any
from vectorstore.base import Point, ScoredPoint


class LanceDBStore:
    """LanceDB serverless columnar vector store backend."""

    def __init__(self, uri: str = ".lancedb"):
        try:
            import lancedb
        except ImportError:
            raise ImportError("LanceDBStore requires: pip install lancedb pyarrow")
        self._db = lancedb.connect(uri)
        self._tables: dict[str, Any] = {}

    def _tbl(self, name: str):
        if name not in self._tables:
            if name in self._db.table_names():
                self._tables[name] = self._db.open_table(name)
        return self._tables.get(name)

    def create_collection(self, name: str, dim: int) -> None:
        import pyarrow as pa
        schema = pa.schema([
            pa.field("id",      pa.utf8()),
            pa.field("vector",  pa.list_(pa.float32(), dim)),
            pa.field("payload", pa.utf8()),
        ])
        tbl = self._db.create_table(name, schema=schema, exist_ok=True)
        self._tables[name] = tbl

    def collection_exists(self, name: str) -> bool:
        return name in self._db.table_names()

    def delete_collection(self, name: str) -> None:
        self._db.drop_table(name, ignore_missing=True)
        self._tables.pop(name, None)

    def upsert(self, collection: str, points: list[Point]) -> None:
        import json
        tbl = self._tbl(collection)
        data = [{"id": str(p.id), "vector": p.vector,
                 "payload": json.dumps(p.payload)} for p in points]
        tbl.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(data)

    def query(self, collection: str, vector: list[float], top_k: int,
               score_threshold: float | None = None) -> list[ScoredPoint]:
        import json
        tbl = self._tbl(collection)
        rows = (tbl.search(vector)
                   .metric("cosine")
                   .limit(top_k)
                   .to_list())
        results = []
        for row in rows:
            score = 1.0 - float(row.get("_distance", 0.0))
            if score_threshold and score < score_threshold:
                continue
            payload = json.loads(row.get("payload", "{}"))
            results.append(ScoredPoint(id=row["id"], score=score, payload=payload))
        return results

    def scroll(self, collection: str, limit: int = 100, with_payload: bool = True,
                with_vectors: bool = False, offset=None, scroll_filter=None
               ) -> tuple[list[ScoredPoint], Any]:
        import json
        tbl = self._tbl(collection)
        filt = f'id > "{offset}"' if offset else None
        scanner = tbl.to_arrow(filter=filt) if filt else tbl.to_arrow()
        rows = scanner.to_pylist()
        has_more = len(rows) > limit
        rows = rows[:limit]
        points = [ScoredPoint(id=r["id"], score=0.0,
                               payload=json.loads(r.get("payload", "{}")))
                  for r in rows]
        next_offset = rows[-1]["id"] if has_more else None
        return points, next_offset

    def set_payload(self, collection: str, point_id: int | str, payload: dict) -> None:
        import json
        tbl = self._tbl(collection)
        existing = tbl.search().where(f'id = "{point_id}"').to_list()
        current = json.loads(existing[0]["payload"]) if existing else {}
        current.update(payload)
        tbl.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(
            [{"id": str(point_id), "payload": json.dumps(current)}]
        )

    def count(self, collection: str) -> int:
        return self._tbl(collection).count_rows()
```

#### Step 3 — Register in the factory

Open `vectorstore/registry.py` and add a `lancedb` elif block. The complete file after editing:

```python
# vectorstore/registry.py
def get_store(cfg: dict):
    backend = cfg.get("vector_store", "qdrant")
    if backend == "qdrant":
        from vectorstore.qdrant_store import QdrantStore
        return QdrantStore(
            host=cfg.get("qdrant_host", "localhost"),
            port=cfg.get("qdrant_port", 6333),
        )
    if backend == "chroma":
        from vectorstore.chroma_store import ChromaStore
        return ChromaStore(persist_dir=cfg.get("chroma_persist_dir", ".chroma"))
    elif backend == "lancedb":           # ← add this block
        from vectorstore.lancedb_store import LanceDBStore
        return LanceDBStore(uri=cfg.get("lancedb_uri", ".lancedb"))
    raise ValueError(
        f"Unknown vector_store '{backend}'. Choose: qdrant, chroma, lancedb"
    )
```

#### Step 4 — Create a datasource config

Run `python smartqdrant.py init --name my-corpus` to create **`datasource_my-corpus.json`**. Then open it and make the following changes:

| Field | Action | Value |
|-------|--------|-------|
| `vector_store` | change | `"lancedb"` |
| `lancedb_uri` | **add** (not in default) | `".lancedb/my-corpus"` |

Complete config after editing:

```json
{
  "collection":    "my-corpus",
  "vector_store":  "lancedb",
  "lancedb_uri":   ".lancedb/my-corpus",

  "embed_model":   "BAAI/bge-small-en-v1.5",
  "embedder_backend":"fastembed",
  "vector_dim":    384,
  "llm_model":     "meta-llama/Llama-3.2-3B-Instruct",
  "top_k":         5,
  "chunk_size":    600,
  "chunk_overlap": 60,
  "checkpoint_dir":"lora_checkpoints/my-corpus/",
  "version_file":  "my-corpus_version.json",
  "replay_db":     "my-corpus_replay.db"
}
```

#### Step 5 — Index and search

```bash
python smartqdrant.py index --config datasource_my-corpus.json --source ./docs/
python smartqdrant.py search --config datasource_my-corpus.json "what is the warranty period"
```

#### Step 6 — Verify

```python
import lancedb
db = lancedb.connect(".lancedb/my-corpus")
tbl = db.open_table("my-corpus")
print(f"Rows: {tbl.count_rows()}")
print(tbl.to_pandas().head(3)[["id", "payload"]])
```

#### Limitations compared to Qdrant

| Feature | Qdrant | LanceDB |
|---------|--------|---------|
| Server required | Yes | No |
| Storage format | Custom | Apache Lance (columnar) |
| Cloud object storage | No | S3 / GCS / Azure Blob |
| Filtering | Full | SQL-like (fast) |
| Multi-process writes | Yes | No (single writer) |
| Production | Yes | Yes (read-heavy workloads) |

#### Troubleshooting

**`ModuleNotFoundError: No module named 'lancedb'`**
```bash
pip install lancedb pyarrow
```

**`ArrowInvalid: Schema for batch (id: utf8, ...) does not match table schema`**
Ensure `vector_dim` in your config matches the dimension used when the table was created. Delete the `.lancedb` directory and re-index if you changed models.

**`lancedb.exceptions.TableNotFoundError`**
Run `index` before `search`. The table is created on first index.

---

### How do I use SmartQdrant with Redis (RedisSearch)?

Redis with the `RedisSearch` module (included in Redis Stack) supports vector similarity search. If Redis is already in your stack (caching, queues), this adds vector search with no new services.

#### Step 1 — Install dependencies

```bash
pip install redis[hiredis]
```

**Start Redis Stack (includes RedisSearch):**

```bash
docker run -d --name redis-stack \
  -p 6379:6379 \
  -p 8001:8001 \
  redis/redis-stack:latest
```

Port 8001 is the RedisInsight UI (optional).

#### Step 2 — Create the backend file

Create `vectorstore/redis_store.py`:

```python
# vectorstore/redis_store.py
from __future__ import annotations
import json
import struct
from typing import Any
from vectorstore.base import Point, ScoredPoint


class RedisStore:
    """Redis (RedisSearch) vector store backend for SmartQdrant."""

    def __init__(self, host: str = "localhost", port: int = 6379,
                 password: str = "", db: int = 0):
        try:
            import redis
        except ImportError:
            raise ImportError("RedisStore requires: pip install redis[hiredis]")
        self._r = redis.Redis(host=host, port=port, password=password or None,
                               db=db, decode_responses=False)

    def _idx(self, name: str) -> str:
        return f"sq:{name}:idx"

    def _key(self, name: str, point_id) -> str:
        return f"sq:{name}:pt:{point_id}"

    def create_collection(self, name: str, dim: int) -> None:
        from redis.commands.search.field import VectorField, TextField, TagField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        try:
            self._r.ft(self._idx(name)).info()
            return   # already exists
        except Exception:
            pass
        schema = (
            TextField("$.text",    as_name="text"),
            TagField("$.id",       as_name="id"),
            VectorField("$.vector", "HNSW", {
                "TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"
            }, as_name="vector"),
        )
        self._r.ft(self._idx(name)).create_index(
            schema,
            definition=IndexDefinition(prefix=[f"sq:{name}:pt:"], index_type=IndexType.JSON),
        )

    def collection_exists(self, name: str) -> bool:
        try:
            self._r.ft(self._idx(name)).info()
            return True
        except Exception:
            return False

    def delete_collection(self, name: str) -> None:
        try:
            self._r.ft(self._idx(name)).dropindex(delete_documents=True)
        except Exception:
            pass

    def upsert(self, collection: str, points: list[Point]) -> None:
        pipe = self._r.pipeline(transaction=False)
        for p in points:
            doc = {"id": str(p.id), "vector": p.vector, **p.payload}
            pipe.json().set(self._key(collection, p.id), "$", doc)
        pipe.execute()

    def query(self, collection: str, vector: list[float], top_k: int,
               score_threshold: float | None = None) -> list[ScoredPoint]:
        from redis.commands.search.query import Query
        vec_bytes = struct.pack(f"{len(vector)}f", *vector)
        q = (Query(f"*=>[KNN {top_k} @vector $vec AS score]")
             .sort_by("score")
             .return_fields("score", "$")
             .dialect(2))
        res = self._r.ft(self._idx(collection)).search(q, query_params={"vec": vec_bytes})
        results = []
        for doc in res.docs:
            score = 1.0 - float(doc.score)
            if score_threshold and score < score_threshold:
                continue
            payload = json.loads(doc["$"]) if isinstance(doc["$"], (str, bytes)) else {}
            results.append(ScoredPoint(id=doc.id.split(":")[-1], score=score, payload=payload))
        return results

    def scroll(self, collection: str, limit: int = 100, with_payload: bool = True,
                with_vectors: bool = False, offset=None, scroll_filter=None
               ) -> tuple[list[ScoredPoint], Any]:
        pattern = f"sq:{collection}:pt:*"
        cursor = int(offset) if offset is not None else 0
        cursor, keys = self._r.scan(cursor=cursor, match=pattern, count=limit)
        points = []
        for key in keys[:limit]:
            data = self._r.json().get(key, "$")
            if data:
                doc = data[0] if isinstance(data, list) else data
                points.append(ScoredPoint(id=str(doc.get("id", key)), score=0.0, payload=doc))
        next_offset = cursor if cursor != 0 else None
        return points, next_offset

    def set_payload(self, collection: str, point_id: int | str, payload: dict) -> None:
        key = self._key(collection, point_id)
        for field, value in payload.items():
            self._r.json().set(key, f"$.{field}", value)

    def count(self, collection: str) -> int:
        try:
            info = self._r.ft(self._idx(collection)).info()
            return int(info.get("num_docs", 0))
        except Exception:
            return 0
```

#### Step 3 — Register in the factory

Open `vectorstore/registry.py` and add a `redis` elif block. The complete file after editing:

```python
# vectorstore/registry.py
def get_store(cfg: dict):
    backend = cfg.get("vector_store", "qdrant")
    if backend == "qdrant":
        from vectorstore.qdrant_store import QdrantStore
        return QdrantStore(
            host=cfg.get("qdrant_host", "localhost"),
            port=cfg.get("qdrant_port", 6333),
        )
    if backend == "chroma":
        from vectorstore.chroma_store import ChromaStore
        return ChromaStore(persist_dir=cfg.get("chroma_persist_dir", ".chroma"))
    elif backend == "redis":             # ← add this block
        from vectorstore.redis_store import RedisStore
        return RedisStore(
            host=cfg.get("redis_host", "localhost"),
            port=cfg.get("redis_port", 6379),
            password=cfg.get("redis_password", ""),
            db=cfg.get("redis_db", 0),
        )
    raise ValueError(
        f"Unknown vector_store '{backend}'. Choose: qdrant, chroma, redis"
    )
```

#### Step 4 — Create a datasource config

Run `python smartqdrant.py init --name my-corpus` to create **`datasource_my-corpus.json`**. Then open it and make the following changes:

| Field | Action | Value |
|-------|--------|-------|
| `vector_store` | change | `"redis"` |
| `redis_host` | **add** | `"localhost"` |
| `redis_port` | **add** | `6379` |
| `redis_password` | **add** | `""` (empty for local) |

Complete config after editing:

```json
{
  "collection":    "my-corpus",
  "vector_store":  "redis",
  "redis_host":    "localhost",
  "redis_port":    6379,
  "redis_password":"",

  "embed_model":   "BAAI/bge-small-en-v1.5",
  "embedder_backend":"fastembed",
  "vector_dim":    384,
  "llm_model":     "meta-llama/Llama-3.2-3B-Instruct",
  "top_k":         5,
  "chunk_size":    600,
  "chunk_overlap": 60,
  "checkpoint_dir":"lora_checkpoints/my-corpus/",
  "version_file":  "my-corpus_version.json",
  "replay_db":     "my-corpus_replay.db"
}
```

#### Step 5 — Index and search

```bash
python smartqdrant.py index --config datasource_my-corpus.json --source ./docs/
python smartqdrant.py search --config datasource_my-corpus.json "billing cycle question"
```

#### Step 6 — Verify

```python
import redis
r = redis.Redis(host="localhost", port=6379, decode_responses=True)
print(f"Keys: {r.dbsize()}")
info = r.ft("sq:my-corpus:idx").info()
print(f"Indexed docs: {info['num_docs']}")
```

Or use RedisInsight at [http://localhost:8001](http://localhost:8001) to browse documents visually.

#### Limitations compared to Qdrant

| Feature | Qdrant | Redis |
|---------|--------|-------|
| Primary use case | Vector search | Cache / queues + vector |
| Persistence | Disk-native | RDB/AOF snapshots |
| Memory overhead | Low | High (all data in RAM) |
| Payload filtering | Full | Limited (tag/text fields) |
| KV `compute-kv` scroll | Efficient | SCAN-based (slower) |
| Production | Yes | Yes (if already using Redis) |

#### Troubleshooting

**`redis.exceptions.ModuleNotFoundError: ERR unknown command 'FT.CREATE'`**
You are running plain Redis, not Redis Stack. Switch to `redis/redis-stack` Docker image.

**`ResponseError: Wrong number of arguments`**
Check your `redis` Python package version. Use `pip install "redis[hiredis]>=4.6"`.

**High memory usage**
All vector data lives in RAM. For 1 million 384-dim vectors: ~1.5 GB. Monitor with `redis-cli INFO memory`.

---

### How do I use SmartQdrant with Elasticsearch or OpenSearch?

Elasticsearch (8.x+) and OpenSearch (2.x+) have built-in dense vector (`knn_vector`) support. If your team already runs an ES/OpenSearch cluster for full-text search, this adds vector similarity with zero new infrastructure.

Both use the same Python client API pattern; only the import and auth differ.

#### Step 1 — Install dependencies

**Elasticsearch:**
```bash
pip install elasticsearch
```

**OpenSearch:**
```bash
pip install opensearch-py
```

**Docker (Elasticsearch 8):**
```bash
docker run -d --name elasticsearch \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -p 9200:9200 \
  docker.elastic.co/elasticsearch/elasticsearch:8.13.0
```

**Docker (OpenSearch 2):**
```bash
docker run -d --name opensearch \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  -p 9200:9200 \
  opensearchproject/opensearch:2.13.0
```

#### Step 2 — Create the backend file

Create `vectorstore/elastic_store.py` (works for both ES and OpenSearch):

```python
# vectorstore/elastic_store.py
from __future__ import annotations
import json
from typing import Any
from vectorstore.base import Point, ScoredPoint


class ElasticStore:
    """Elasticsearch / OpenSearch vector store backend."""

    def __init__(self, hosts: list[str], api_key: str = "",
                 backend: str = "elasticsearch"):
        if backend == "opensearch":
            try:
                from opensearchpy import OpenSearch
                self._es = OpenSearch(hosts=hosts)
            except ImportError:
                raise ImportError("ElasticStore (OpenSearch) requires: pip install opensearch-py")
        else:
            try:
                from elasticsearch import Elasticsearch
                kwargs = {"api_key": api_key} if api_key else {}
                self._es = Elasticsearch(hosts, **kwargs)
            except ImportError:
                raise ImportError("ElasticStore requires: pip install elasticsearch")
        self._backend = backend

    def create_collection(self, name: str, dim: int) -> None:
        if self._es.indices.exists(index=name):
            return
        mapping = {
            "mappings": {
                "properties": {
                    "vector":  {"type": "dense_vector", "dims": dim,
                                 "index": True, "similarity": "cosine"},
                    "payload": {"type": "object", "enabled": True},
                }
            }
        }
        self._es.indices.create(index=name, body=mapping)

    def collection_exists(self, name: str) -> bool:
        return bool(self._es.indices.exists(index=name))

    def delete_collection(self, name: str) -> None:
        if self._es.indices.exists(index=name):
            self._es.indices.delete(index=name)

    def upsert(self, collection: str, points: list[Point]) -> None:
        from elasticsearch import helpers
        actions = [
            {"_index": collection, "_id": str(p.id),
             "_source": {"vector": p.vector, "payload": p.payload}}
            for p in points
        ]
        helpers.bulk(self._es, actions)
        self._es.indices.refresh(index=collection)

    def query(self, collection: str, vector: list[float], top_k: int,
               score_threshold: float | None = None) -> list[ScoredPoint]:
        body = {
            "knn": {"field": "vector", "query_vector": vector,
                    "k": top_k, "num_candidates": top_k * 5},
            "_source": ["payload"],
        }
        res = self._es.search(index=collection, body=body)
        results = []
        for hit in res["hits"]["hits"]:
            score = float(hit["_score"])
            if score_threshold and score < score_threshold:
                continue
            payload = hit["_source"].get("payload", {})
            results.append(ScoredPoint(id=hit["_id"], score=score, payload=payload))
        return results

    def scroll(self, collection: str, limit: int = 100, with_payload: bool = True,
                with_vectors: bool = False, offset=None, scroll_filter=None
               ) -> tuple[list[ScoredPoint], Any]:
        body: dict = {"query": {"match_all": {}}, "size": limit}
        if offset:
            body["search_after"] = [offset]
            body["sort"] = [{"_id": "asc"}]
        res = self._es.search(index=collection, body=body, sort="_id:asc")
        hits = res["hits"]["hits"]
        points = [ScoredPoint(id=h["_id"], score=0.0,
                               payload=h["_source"].get("payload", {}))
                  for h in hits]
        next_offset = hits[-1]["sort"][0] if len(hits) == limit else None
        return points, next_offset

    def set_payload(self, collection: str, point_id: int | str, payload: dict) -> None:
        script = {"source": "ctx._source.payload.putAll(params.payload)",
                  "params": {"payload": payload}}
        self._es.update(index=collection, id=str(point_id), body={"script": script})

    def count(self, collection: str) -> int:
        return self._es.count(index=collection)["count"]
```

#### Step 3 — Register in the factory

Open `vectorstore/registry.py` and add an `elasticsearch`/`opensearch` elif block. The complete file after editing:

```python
# vectorstore/registry.py
def get_store(cfg: dict):
    backend = cfg.get("vector_store", "qdrant")
    if backend == "qdrant":
        from vectorstore.qdrant_store import QdrantStore
        return QdrantStore(
            host=cfg.get("qdrant_host", "localhost"),
            port=cfg.get("qdrant_port", 6333),
        )
    if backend == "chroma":
        from vectorstore.chroma_store import ChromaStore
        return ChromaStore(persist_dir=cfg.get("chroma_persist_dir", ".chroma"))
    elif backend in ("elasticsearch", "opensearch"):   # ← add this block
        from vectorstore.elastic_store import ElasticStore
        return ElasticStore(
            hosts=cfg.get("elastic_hosts", ["http://localhost:9200"]),
            api_key=cfg.get("elastic_api_key", ""),
            backend=backend,
        )
    raise ValueError(
        f"Unknown vector_store '{backend}'. "
        f"Choose: qdrant, chroma, elasticsearch, opensearch"
    )
```

#### Step 4 — Create a datasource config

Run `python smartqdrant.py init --name my-corpus` to create **`datasource_my-corpus.json`**. Then open it and make the following changes:

| Field | Action | Value |
|-------|--------|-------|
| `vector_store` | change | `"elasticsearch"` or `"opensearch"` |
| `elastic_hosts` | **add** | `["http://localhost:9200"]` |
| `elastic_api_key` | **add** | `""` (empty for local without auth) |

**Elasticsearch — complete config:**

```json
{
  "collection":       "my-corpus",
  "vector_store":     "elasticsearch",
  "elastic_hosts":    ["http://localhost:9200"],
  "elastic_api_key":  "",

  "embed_model":      "BAAI/bge-small-en-v1.5",
  "embedder_backend": "fastembed",
  "vector_dim":       384,
  "llm_model":        "meta-llama/Llama-3.2-3B-Instruct",
  "top_k":            5,
  "chunk_size":       600,
  "chunk_overlap":    60,
  "checkpoint_dir":   "lora_checkpoints/my-corpus/",
  "version_file":     "my-corpus_version.json",
  "replay_db":        "my-corpus_replay.db"
}
```

**OpenSearch — complete config** (only two fields differ from the Elasticsearch config above):

```json
{
  "collection":       "my-corpus",
  "vector_store":     "opensearch",
  "elastic_hosts":    ["http://localhost:9200"],
  "elastic_api_key":  "",

  "embed_model":      "BAAI/bge-small-en-v1.5",
  "embedder_backend": "fastembed",
  "vector_dim":       384,
  "llm_model":        "meta-llama/Llama-3.2-3B-Instruct",
  "top_k":            5,
  "chunk_size":       600,
  "chunk_overlap":    60,
  "checkpoint_dir":   "lora_checkpoints/my-corpus/",
  "version_file":     "my-corpus_version.json",
  "replay_db":        "my-corpus_replay.db"
}
```

#### Step 5 — Index and search

```bash
python smartqdrant.py index --config datasource_my-corpus.json --source ./docs/
python smartqdrant.py search --config datasource_my-corpus.json "SLA definition"
```

#### Step 6 — Verify

```bash
curl -s http://localhost:9200/my-corpus/_count | python3 -m json.tool
```

Or from Python:

```python
from elasticsearch import Elasticsearch
es = Elasticsearch(["http://localhost:9200"])
print(es.count(index="my-corpus"))
print(es.search(index="my-corpus", body={"query": {"match_all": {}}, "size": 3}))
```

#### Limitations compared to Qdrant

| Feature | Qdrant | Elasticsearch / OpenSearch |
|---------|--------|---------------------------|
| Primary use | Vector search | Full-text + vector (hybrid) |
| Hybrid BM25 + vector | No | Yes |
| HNSW index | Yes | Yes (8.0+) |
| Resource usage | Low | High (JVM heap) |
| Payload filtering | Full | Full (ES query DSL) |
| Production | Yes | Yes (at scale) |
| License | Apache 2 | SSPL (ES) / Apache 2 (OS) |

> **Hybrid search tip:** Elasticsearch's native BM25 + vector fusion (`rrf` ranker) can improve recall compared to vector-only search. This is a key advantage over Qdrant for document corpora where keyword matches matter.

#### Troubleshooting

**`ConnectionError: [Errno 111] Connection refused`**
Ensure the container is running: `docker ps | grep elastic`.

**`RequestError: mapper [vector] of different type`**
Index was created with a different dimension. Delete the index and re-index: `es.indices.delete(index="my-corpus")`.

**`elasticsearch.AuthenticationException: HTTP/1.1 401 Unauthorized`**
Pass your API key: `"elastic_api_key": "base64encodedkey"`. For local dev, disable security: `-e "xpack.security.enabled=false"`.

---

### How do I use SmartQdrant with MongoDB Atlas Vector Search?

MongoDB Atlas Vector Search adds vector similarity to your existing MongoDB collections. If you already store your source documents in MongoDB, this eliminates a separate vector DB entirely.

> **Important:** MongoDB Atlas Vector Search requires an Atlas cluster (M10 or higher, or serverless). It does **not** work with a self-hosted `mongod` instance. Local MongoDB does not support `$vectorSearch`.

#### Step 1 — Prerequisites (complete these before writing any code)

1. **Create a MongoDB Atlas account** at [cloud.mongodb.com](https://cloud.mongodb.com) if you do not have one.
2. **Create a cluster**: In Atlas, click **Build a Database** → select **M10** (or higher) or **Serverless**.
3. **Get your connection string**: In Atlas, go to your cluster → **Connect** → **Drivers** → select Python → copy the connection string. It looks like:
   ```
   mongodb+srv://myuser:mypassword@cluster0.xxxxx.mongodb.net/
   ```
4. **Whitelist your IP**: In Atlas, go to **Network Access** → **Add IP Address** → add your current IP (or `0.0.0.0/0` for development).
5. **Install the Python driver**:
   ```bash
   pip install pymongo
   ```

#### Step 2 — Create the Vector Search Index in Atlas

**This step must be completed before running `smartqdrant.py index`.** The index cannot be created by SmartQdrant — it must be created in the Atlas UI.

In the Atlas UI:
1. Click your cluster → **Browse Collections**
2. Select (or create) your database and collection
3. Click the **Search Indexes** tab → **Create Search Index**
4. Select **JSON Editor** and paste:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "vector",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}
```

5. Set **Index Name** to `vector_index`
6. Click **Create Search Index** and wait for status to show **Active** (1–2 minutes)

> If your `vector_dim` is not 384, change `numDimensions` to match your `vector_dim` value.

#### Step 3 — Create the backend file

Create a new file at `vectorstore/mongodb_store.py` in the SmartQdrant repository root:

```python
# vectorstore/mongodb_store.py
from __future__ import annotations
from typing import Any
from vectorstore.base import Point, ScoredPoint


class MongoDBStore:
    """MongoDB Atlas Vector Search backend for SmartQdrant."""

    def __init__(self, uri: str, database: str = "smartqdrant"):
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError("MongoDBStore requires: pip install pymongo")
        self._client = MongoClient(uri)
        self._db = self._client[database]

    def _col(self, name: str):
        return self._db[name]

    def create_collection(self, name: str, dim: int) -> None:
        # MongoDB creates collections implicitly on first insert.
        # Ensure the collection exists.
        if name not in self._db.list_collection_names():
            self._db.create_collection(name)
        # Note: The Vector Search index must be created in Atlas UI (see FAQ above).

    def collection_exists(self, name: str) -> bool:
        return name in self._db.list_collection_names()

    def delete_collection(self, name: str) -> None:
        self._db.drop_collection(name)

    def upsert(self, collection: str, points: list[Point]) -> None:
        from pymongo import ReplaceOne
        col = self._col(collection)
        ops = [
            ReplaceOne({"_id": str(p.id)},
                        {"_id": str(p.id), "vector": p.vector, **p.payload},
                        upsert=True)
            for p in points
        ]
        col.bulk_write(ops, ordered=False)

    def query(self, collection: str, vector: list[float], top_k: int,
               score_threshold: float | None = None) -> list[ScoredPoint]:
        col = self._col(collection)
        pipeline = [
            {"$vectorSearch": {
                "index":         "vector_index",
                "path":          "vector",
                "queryVector":   vector,
                "numCandidates": top_k * 10,
                "limit":         top_k,
            }},
            {"$project": {
                "score": {"$meta": "vectorSearchScore"},
                "text": 1, "_id": 1,
            }},
        ]
        results = []
        for doc in col.aggregate(pipeline):
            score = float(doc.get("score", 0.0))
            if score_threshold and score < score_threshold:
                continue
            payload = {k: v for k, v in doc.items() if k not in ("_id", "score", "vector")}
            results.append(ScoredPoint(id=doc["_id"], score=score, payload=payload))
        return results

    def scroll(self, collection: str, limit: int = 100, with_payload: bool = True,
                with_vectors: bool = False, offset=None, scroll_filter=None
               ) -> tuple[list[ScoredPoint], Any]:
        col = self._col(collection)
        query = {"_id": {"$gt": offset}} if offset else {}
        cursor = col.find(query, {"vector": 0}).limit(limit + 1).sort("_id", 1)
        docs = list(cursor)
        has_more = len(docs) > limit
        docs = docs[:limit]
        points = [ScoredPoint(id=d["_id"], score=0.0,
                               payload={k: v for k, v in d.items() if k != "_id"})
                  for d in docs]
        next_offset = docs[-1]["_id"] if has_more else None
        return points, next_offset

    def set_payload(self, collection: str, point_id: int | str, payload: dict) -> None:
        self._col(collection).update_one(
            {"_id": str(point_id)},
            {"$set": payload},
            upsert=True,
        )

    def count(self, collection: str) -> int:
        return self._col(collection).estimated_document_count()
```

#### Step 4 — Register in the factory

Open `vectorstore/registry.py` and add a `mongodb` elif block. The complete file after editing:

```python
# vectorstore/registry.py
def get_store(cfg: dict):
    backend = cfg.get("vector_store", "qdrant")
    if backend == "qdrant":
        from vectorstore.qdrant_store import QdrantStore
        return QdrantStore(
            host=cfg.get("qdrant_host", "localhost"),
            port=cfg.get("qdrant_port", 6333),
        )
    if backend == "chroma":
        from vectorstore.chroma_store import ChromaStore
        return ChromaStore(persist_dir=cfg.get("chroma_persist_dir", ".chroma"))
    elif backend == "mongodb":           # ← add this block
        from vectorstore.mongodb_store import MongoDBStore
        return MongoDBStore(
            uri=cfg["mongodb_uri"],
            database=cfg.get("mongodb_database", "smartqdrant"),
        )
    raise ValueError(
        f"Unknown vector_store '{backend}'. Choose: qdrant, chroma, mongodb"
    )
```

#### Step 5 — Create a datasource config

Run `python smartqdrant.py init --name my-corpus` to create **`datasource_my-corpus.json`**. Then open it and make the following changes:

| Field | Action | Value |
|-------|--------|-------|
| `vector_store` | change | `"mongodb"` |
| `mongodb_uri` | **add** | your Atlas connection string (from Step 1 prerequisite) |
| `mongodb_database` | **add** | `"smartqdrant"` (or any database name) |

Complete config after editing:

```json
{
  "collection":        "my-corpus",
  "vector_store":      "mongodb",
  "mongodb_uri":       "mongodb+srv://user:pass@cluster0.xxxxx.mongodb.net/",
  "mongodb_database":  "smartqdrant",

  "embed_model":       "BAAI/bge-small-en-v1.5",
  "embedder_backend":  "fastembed",
  "vector_dim":        384,
  "llm_model":         "meta-llama/Llama-3.2-3B-Instruct",
  "top_k":             5,
  "chunk_size":        600,
  "chunk_overlap":     60,
  "checkpoint_dir":    "lora_checkpoints/my-corpus/",
  "version_file":      "my-corpus_version.json",
  "replay_db":         "my-corpus_replay.db"
}
```

#### Step 6 — Index and search

```bash
python smartqdrant.py index --config datasource_my-corpus.json --source ./docs/
python smartqdrant.py search --config datasource_my-corpus.json "data retention policy"
```

#### Step 7 — Verify

```python
from pymongo import MongoClient
client = MongoClient("mongodb+srv://user:pass@cluster0.xxxxx.mongodb.net/")
db = client["smartqdrant"]
col = db["my-corpus"]
print(f"Documents: {col.estimated_document_count()}")
print(col.find_one({}, {"_id": 1, "text": 1}))
```

#### Limitations compared to Qdrant

| Feature | Qdrant | MongoDB Atlas |
|---------|--------|---------------|
| Primary use | Vector search | Document DB + vector |
| Vector index | HNSW | HNSW (Atlas) |
| Payload filtering | Full | Full (MongoDB query) |
| Local dev (free) | Docker | Atlas free tier (512 MB) |
| KV `compute-kv` | Efficient | Collection scan (efficient) |
| Cost | Self-hosted free | Atlas pricing |

#### Troubleshooting

**`OperationFailure: $vectorSearch is not allowed`**
Vector Search requires Atlas M10+ or Serverless. It does not work on local MongoDB (`mongod`).

**`IndexNotFound: index vector_index not found`**
Create the Vector Search index in Atlas UI first (see Step 2).

**`ServerSelectionTimeoutError`**
Check that your Atlas cluster IP allowlist includes your current IP address, or set it to `0.0.0.0/0` for development.

**Slow search results**
Increase `numCandidates` in the `$vectorSearch` pipeline stage (currently `top_k * 10`). Higher values improve recall at the cost of latency.

---

### Can I use an existing Qdrant collection I already have?

Yes. SmartQdrant is additive — it reads and writes extra payload fields on your existing points without touching your vectors or any of your existing payload fields.

#### Step 1 — Identify your collection's embedding model and dimension

If you are unsure which model was used:

```python
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)
info = client.get_collection("your-collection")
print(info.config.params.vectors)   # shows size and distance metric
```

The `size` value is your `vector_dim`.

#### Step 2 — Create a config for the existing collection

```bash
python smartqdrant.py init --name existing --llm-model meta-llama/Llama-3.2-3B-Instruct
```

Edit `datasource_existing.json` to match your existing collection:

```json
{
  "collection":       "your-existing-collection",
  "qdrant_host":      "your-qdrant-host",
  "qdrant_port":      6333,

  "embed_model":      "BAAI/bge-small-en-v1.5",
  "embedder_backend": "fastembed",
  "vector_dim":       384,

  "llm_model":        "meta-llama/Llama-3.2-3B-Instruct",
  "hf_token":         "hf_...",

  "checkpoint_dir":   "lora_checkpoints/existing/",
  "version_file":     "existing_version.json",
  "replay_db":        "existing_replay.db"
}
```

#### Step 3 — Verify that your existing points have a `text` field

SmartQdrant uses `payload["text"]` for text-in-context fallback and KV computation. Check:

```python
from qdrant_client import QdrantClient

client = QdrantClient("your-qdrant-host", port=6333)
results, _ = client.scroll("your-existing-collection", limit=3, with_payload=True)
for r in results:
    print(r.payload.keys())   # should include "text"
```

If your field is named differently (e.g. `"content"` or `"body"`), you need to either rename it or add an alias. The simplest approach is a one-time migration:

```python
results, offset = client.scroll("your-collection", limit=100, with_payload=True)
while results:
    for r in results:
        if "content" in r.payload and "text" not in r.payload:
            client.set_payload("your-collection",
                                payload={"text": r.payload["content"]},
                                points=[r.id])
    results, offset = client.scroll("your-collection", limit=100,
                                     with_payload=True, offset=offset)
    if offset is None:
        break
```

#### Step 4 — Backfill KV tensors (GPU required)

```bash
# Compute KV tensors for all points that don't have kv_cache yet
python kv_indexer.py --config datasource_existing.json compute-kv
```

This loops through every point, runs a forward pass, and writes `kv_cache` and `kv_version` into the payload. Your vectors are untouched. Depending on corpus size this takes 2–10 minutes per 1,000 chunks on a single A10G GPU.

Progress is resumable — if interrupted, re-run the same command. Only points with `kv_version = null` are processed.

#### Step 5 — Generate FAQs and run training

```bash
python tools/generate_faqs.py \
  --config datasource_existing.json \
  --output existing_faqs.json \
  --n 50

python index_and_train.py dummy.pdf \
  --config datasource_existing.json \
  --faqs existing_faqs.json \
  --skip-index   # skip re-indexing, jump straight to training
```

> The `--skip-index` flag is available in `index_and_train.py` to avoid re-indexing an existing collection.

---

## Language Models

### How do I use my own LLM for KV computation?

SmartQdrant uses HuggingFace `AutoModelForCausalLM` for KV computation and LoRA training. Any decoder-only transformer hosted on HuggingFace Hub (or locally) works.

#### Step 1 — Set `llm_model` in your config

```json
{
  "llm_model": "mistralai/Mistral-7B-Instruct-v0.3"
}
```

Other tested models:

```json
{ "llm_model": "google/gemma-2-2b-it" }
{ "llm_model": "Qwen/Qwen2.5-3B-Instruct" }
{ "llm_model": "microsoft/phi-3-mini-4k-instruct" }
{ "llm_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0" }
{ "llm_model": "mistralai/Mixtral-8x7B-Instruct-v0.1" }
```

#### Step 2 — Verify KV shape auto-discovery

SmartQdrant reads `num_hidden_layers`, `num_key_value_heads`, and `hidden_size` / `head_dim` from the HuggingFace model config automatically. Verify before running the full pipeline:

```python
import model_loader

cfg = {"llm_model": "mistralai/Mistral-7B-Instruct-v0.3"}
model_loader.init(cfg)
num_layers, num_kv_heads, head_dim = model_loader.get_kv_shape(cfg)
print(f"KV shape: layers={num_layers}, kv_heads={num_kv_heads}, head_dim={head_dim}")
# Expected for Mistral-7B: layers=32, kv_heads=8, head_dim=128
```

#### Step 3 — Confirm LoRA target modules

SmartQdrant auto-detects which attention projection module names exist in your model and warns if none of the configured names are found:

```python
import model_loader, warnings

cfg = {"llm_model": "mistralai/Mistral-7B-Instruct-v0.3",
       "lora_target_modules": ["q_proj", "k_proj", "v_proj"]}
model_loader.init(cfg)
model, _ = model_loader.load(None)
matched = model_loader.detect_lora_targets(model, cfg["lora_target_modules"])
print("LoRA targets found:", matched)
```

Common LoRA target patterns by model family:

| Model family | Typical `lora_target_modules` |
|-------------|-------------------------------|
| Llama / Mistral / Phi | `["q_proj", "k_proj", "v_proj"]` |
| Falcon | `["query_key_value"]` |
| GPT-2 / GPT-J | `["c_attn"]` |
| BLOOM | `["query_key_value", "dense"]` |
| Gemma | `["q_proj", "k_proj", "v_proj", "o_proj"]` |
| Qwen2 | `["q_proj", "k_proj", "v_proj", "o_proj"]` |

For an unknown architecture, list all module names and pick attention projections:

```python
for name, module in model.named_modules():
    if "proj" in name or "attn" in name:
        print(name)
```

#### Step 4 — Use a locally saved model

If your model is saved to disk rather than hosted on HuggingFace:

```json
{
  "llm_model": "/home/ubuntu/models/my-fine-tuned-llama"
}
```

`model_loader.py` passes `llm_model` directly to `AutoModelForCausalLM.from_pretrained()`, which accepts both Hub IDs and local paths.

#### Step 5 — Run the pipeline

```bash
python index_and_train.py my_document.pdf \
  --config datasource_my-corpus.json \
  --faqs my_faqs.json
```

---

### How do I use a gated model like Llama 3 that requires a HuggingFace token?

Models like `meta-llama/Llama-3.2-3B-Instruct` require you to:
1. Create a HuggingFace account at [huggingface.co](https://huggingface.co)
2. Visit the model card and click **Agree and access repository**
3. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with **Read** permission

#### Option A — Config file (recommended for servers)

```json
{
  "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
  "hf_token":  "hf_xxxxxxxxxxxxxxxxxxxx"
}
```

SmartQdrant sets `HF_TOKEN` in the environment before calling `from_pretrained()`.

> **Security note:** Do not commit `datasource_*.json` files containing tokens to version control. Add `datasource_*.json` to `.gitignore` or use the environment variable approach instead.

#### Option B — Environment variable (recommended for CI/CD)

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
python index_and_train.py document.pdf --config my_config.json
```

Or in a shell script:

```bash
#!/bin/bash
export HF_TOKEN=$(cat ~/.hf_token)   # read from a file not in version control
python index_and_train.py "$@"
```

#### Option C — HuggingFace CLI login (interactive)

```bash
pip install huggingface_hub
huggingface-cli login
# Enter your token when prompted — it is saved to ~/.cache/huggingface/token
```

After login, no token is needed in the config or environment.

#### Verifying access

```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B-Instruct",
                                     token="hf_xxxx")
print(config.model_type)   # should print: llama
```

If you see `401 Client Error: Unauthorized`, your token is invalid or you have not accepted the license agreement.

---

### Can I use an API-hosted LLM (OpenAI, Anthropic, Gemini)?

#### What requires a local model

KV cache computation and LoRA fine-tuning both require direct access to the model's internal tensors. Specifically:

- **KV computation** needs `outputs.past_key_values` — the raw attention tensors produced during a forward pass. No API exposes this.
- **LoRA training** needs gradient flow through the model's weight matrices. No API exposes this.

These operations cannot be performed against API-hosted models. A local HuggingFace model is required for Phases 2 and 3.

#### What can use an API LLM

Phase 1 text-in-context fallback is just prompt engineering — retrieved chunks are placed into a prompt and the model generates an answer. You can replace the local generation in `kv_inference.generate_text_in_context()` with an API call:

```python
# kv_inference.py — replace generate_text_in_context() for API usage
import openai

def generate_text_in_context_openai(query: str, chunks: list[dict],
                                     api_key: str,
                                     model: str = "gpt-4o-mini") -> str:
    context = "\n\n---\n\n".join(
        f"[page {c['page']}, score {c['score']}]\n{c['text']}"
        for c in chunks
    )
    prompt = (
        f"Using only the context below, answer the question in 2-4 sentences. "
        f"Cite page numbers.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()
```

This gives you a fully functional **Phase 1 RAG system** with no GPU required. KV injection (Phase 2) and parametric answering (Phase 3) remain unavailable with API models.

#### Practical hybrid architecture

If you want the best of both worlds:

```
Retrieval:   OpenAI text-embedding-3-small  (high-quality embeddings, no GPU)
Answer:      OpenAI gpt-4o-mini             (Phase 1, no GPU)
KV compute:  local Llama 3.2 3B             (Phase 2+, GPU required)
LoRA train:  local Llama 3.2 3B             (Phase 2+, GPU required)
```

Use OpenAI embeddings for the highest retrieval quality, and the local small model only for the KV/LoRA work that requires tensor access.

---

### Can I run this without a GPU?

The table below shows which operations are CPU-compatible:

| Operation | CPU | GPU | Notes |
|-----------|:---:|:---:|-------|
| `smartqdrant.py init` | ✅ | ✅ | Config scaffolding only |
| `smartqdrant.py index` | ✅ | ✅ | Embedding runs on CPU with FastEmbed |
| `smartqdrant.py search` | ✅ | ✅ | Embedding + vector search |
| `python -m pytest tests/` | ✅ | ✅ | All 76 tests mock GPU modules |
| `monitoring_dashboard.py` | ✅ | ✅ | FastAPI dashboard |
| `prs_evaluator.py` (evaluation only) | ✅ | ✅ | If using API LLM for generation |
| KV tensor computation | ❌ | ✅ | LLM forward pass required |
| LoRA training | ❌ | ✅ | Gradient computation required |
| KV injection at query time | ❌ | ✅ | Tensor operations on model device |

#### Running the test suite on CPU

```bash
# All 76 tests pass on CPU — GPU modules are mocked
python -m pytest tests/ -v --override-ini="addopts="
```

Expected time: ~15–30 seconds on a modern laptop.

#### Local development workflow without a GPU

1. Use `smartqdrant.py init / index / search` for all ingestion and retrieval work
2. Use the dashboard to verify indexing output
3. When ready to train, push the config and data to a GPU server:

```bash
rsync -avz --exclude='venv/' --exclude='__pycache__/' \
  -e "ssh -i your-key.pem" \
  ./ ubuntu@<gpu-server>:~/smartqdrant/

ssh -i your-key.pem ubuntu@<gpu-server>
cd ~/smartqdrant
python index_and_train.py document.pdf --config datasource_my-corpus.json --faqs faqs.json
```

---

## Embedding Models

### How do I use OpenAI embeddings instead of FastEmbed?

#### Step 1 — Install the OpenAI SDK

```bash
pip install openai
```

#### Step 2 — Update your datasource config

```json
{
  "embedder_backend": "openai",
  "embed_model":      "text-embedding-3-small",
  "vector_dim":       1536,
  "openai_api_key":   "sk-..."
}
```

Or use the environment variable (preferred — keeps secrets out of config files):

```bash
export OPENAI_API_KEY=sk-...
```

OpenAI embedding model reference:

| Model | `vector_dim` | Cost (per 1M tokens) | Notes |
|-------|:------------:|:--------------------:|-------|
| `text-embedding-3-small` | 1536 | ~$0.02 | Best value — strong performance |
| `text-embedding-3-large` | 3072 | ~$0.13 | Highest quality |
| `text-embedding-ada-002` | 1536 | ~$0.10 | Legacy; prefer 3-small |

> **Important:** The `vector_dim` in your config must exactly match the model's output. A mismatch causes an error at index time. SmartQdrant validates this with `validate_embed_dim()` before writing any data.

#### Step 3 — Index and search as normal

```bash
python smartqdrant.py index --config datasource_my-corpus.json --source ./docs/
python smartqdrant.py search --config datasource_my-corpus.json "your query"
```

#### Step 4 — Estimate cost before indexing

```python
# Rough cost estimate for a corpus
num_chunks = 5000
avg_words_per_chunk = 120
avg_tokens_per_word = 1.3
total_tokens = num_chunks * avg_words_per_chunk * avg_tokens_per_word
cost_usd = total_tokens / 1_000_000 * 0.02   # text-embedding-3-small rate
print(f"Estimated indexing cost: ${cost_usd:.4f}")
# 5000 chunks → ~$0.016
```

---

### How do I use sentence-transformers embeddings?

#### Step 1 — Install sentence-transformers

```bash
pip install sentence-transformers
```

#### Step 2 — Update your config

```json
{
  "embedder_backend": "sentence_transformers",
  "embed_model":      "BAAI/bge-base-en-v1.5",
  "vector_dim":       768
}
```

#### Step 3 — Verify the model dimension before indexing

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
test_vec = model.encode(["hello world"])
print(f"Dimension: {len(test_vec[0])}")   # must match vector_dim in config
```

SmartQdrant will raise a `ValueError` at index time if the actual dimension does not match `vector_dim`, so misconfiguration is caught before any data is written.

#### Common sentence-transformers models

| Model | `vector_dim` | Speed | Quality | Notes |
|-------|:------------:|:-----:|:-------:|-------|
| `BAAI/bge-small-en-v1.5` | 384 | Fast | Good | Default; best for resource-limited setups |
| `BAAI/bge-base-en-v1.5` | 768 | Medium | Better | Good general-purpose choice |
| `BAAI/bge-large-en-v1.5` | 1024 | Slow | Best BAAI | Use when retrieval quality matters most |
| `intfloat/e5-large-v2` | 1024 | Slow | Excellent | Strong on MTEB benchmarks |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Medium | Good | Well-tested, widely used |
| `thenlper/gte-large` | 1024 | Slow | Excellent | Strong on passage retrieval tasks |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | Slow | Excellent | Top performer on many benchmarks |

> **Multilingual corpora:** Use `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (dim 768) or `intfloat/multilingual-e5-large` (dim 1024).

---

### How do I add a custom embedding model?

The `Embedder` protocol requires exactly two things: an `encode(texts)` method that returns `list[list[float]]`, and a `dim` property that returns the vector dimension.

#### Step 1 — Create the embedder class

```python
# embeddings/cohere_embedder.py
import os


class CohereEmbedder:
    """Cohere Embed API backend."""

    def __init__(self, model_name: str = "embed-english-v3.0",
                 dim: int = 1024,
                 api_key: str | None = None,
                 input_type: str = "search_document"):
        try:
            import cohere
        except ImportError:
            raise ImportError("CohereEmbedder requires: pip install cohere")
        self._client = cohere.Client(api_key or os.environ["COHERE_API_KEY"])
        self._model_name = model_name
        self._dim = dim
        self._input_type = input_type

    def encode(self, texts: list[str]) -> list[list[float]]:
        # Cohere has a batch limit of 96 texts per call
        all_vectors = []
        batch_size = 96
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self._client.embed(
                texts=batch,
                model=self._model_name,
                input_type=self._input_type,
            )
            all_vectors.extend(resp.embeddings)
        return all_vectors

    @property
    def dim(self) -> int:
        return self._dim
```

#### Step 2 — Register in the factory

Open `embeddings/registry.py` and add before the final `raise ValueError`:

```python
if backend == "cohere":
    from embeddings.cohere_embedder import CohereEmbedder
    return CohereEmbedder(
        model_name=model_name,
        dim=dim,
        api_key=cfg.get("cohere_api_key"),
        input_type=cfg.get("cohere_input_type", "search_document"),
    )
```

#### Step 3 — Update your config

```json
{
  "embedder_backend": "cohere",
  "embed_model":      "embed-english-v3.0",
  "vector_dim":       1024,
  "cohere_api_key":   "..."
}
```

#### Step 4 — Validate before indexing

```python
from embeddings.registry import get_embedder

cfg = {"embedder_backend": "cohere", "embed_model": "embed-english-v3.0",
       "vector_dim": 1024, "cohere_api_key": "..."}
embedder = get_embedder(cfg)
vecs = embedder.encode(["test sentence"])
assert len(vecs) == 1
assert len(vecs[0]) == embedder.dim == 1024
print("Embedder OK")
```

---

### Can I use different embedding models for different collections?

Yes — each datasource config is completely independent. The embedding model is baked into the collection at index time: the vector dimension stored in Qdrant must match the model output. You cannot mix embedding models within a single collection.

```
datasource_legal.json
  embedder_backend: sentence_transformers
  embed_model:      legal-bert-base-uncased
  vector_dim:       768
  collection:       legal-docs

datasource_code.json
  embedder_backend: openai
  embed_model:      text-embedding-3-small
  vector_dim:       1536
  collection:       code-docs

datasource_general.json
  embedder_backend: fastembed
  embed_model:      BAAI/bge-small-en-v1.5
  vector_dim:       384
  collection:       general-docs
```

Each corpus runs completely independently — separate collections, separate version files, separate replay databases, separate LoRA checkpoints. They can share the same Qdrant instance and the same GPU server.

---

## Document Ingestion

### How do I index Markdown documentation?

#### Setup

```bash
# No extra dependencies needed for Markdown
python smartqdrant.py init --name docs --loader markdown
```

#### Index a single file

```bash
python smartqdrant.py index \
  --config datasource_docs.json \
  --source ./README.md
```

#### Index a directory of Markdown files

Change the loader to `directory` so mixed `.md`/`.pdf`/`.html` files are handled automatically:

```bash
python smartqdrant.py init --name docs --loader directory
python smartqdrant.py index \
  --config datasource_docs.json \
  --source ./docs/
```

#### How the Markdown loader splits content

The loader splits on `#`, `##`, and `###` headings using a regex split. Each heading becomes a new chunk. The heading text itself is preserved as the first line of the chunk body. Sections with fewer than 10 words (the `min_chunk_words` threshold) are silently skipped.

Example — given this file:

```markdown
# Installation

Run `pip install smartqdrant` to install.

## Configuration

Copy `datasource_template.json` and edit the fields.

## Troubleshooting

Check logs in `kv_background.log`.
```

The loader produces three chunks: one per `##` section.

#### Tuning for long documents

If your Markdown files contain very long sections without headings, consider switching to `loader: directory` and adding a post-processing step to further split large chunks. Alternatively, pre-process your Markdown files to add intermediate `##` headings.

---

### How do I index a JSONL dataset?

JSONL (JSON Lines) is common for datasets, evaluation sets, and structured knowledge bases.

#### Setup

No extra dependencies required.

```bash
python smartqdrant.py init --name knowledge-base --loader jsonl
```

#### Config

```json
{
  "loader":         "jsonl",
  "jsonl_text_key": "content"
}
```

`jsonl_text_key` (default `"text"`) names the field SmartQdrant reads as the chunk text. All other fields in each JSON object are stored in metadata and available in search results via `payload`.

#### Example JSONL formats

Standard format (default config, `jsonl_text_key: "text"`):
```jsonl
{"text": "Retrieval-Augmented Generation combines dense retrieval with generative models.", "source": "paper_1", "year": 2020}
{"text": "LoRA enables parameter-efficient fine-tuning by decomposing weight updates.", "source": "paper_2", "year": 2022}
```

Custom field name (`jsonl_text_key: "content"`):
```jsonl
{"id": "doc-001", "content": "SmartQdrant stores KV tensors in Qdrant payload fields.", "category": "architecture"}
{"id": "doc-002", "content": "Phase 2 activates after PRS exceeds the configured threshold.", "category": "phases"}
```

HuggingFace datasets export format (`jsonl_text_key: "passage"`):
```jsonl
{"passage": "The Earth is approximately 4.5 billion years old.", "title": "Earth", "source": "wiki"}
```

#### Indexing

```bash
python smartqdrant.py index \
  --config datasource_knowledge-base.json \
  --source ./my_dataset.jsonl
```

#### Accessing metadata in search results

```python
results = store.query("knowledge-base", query_vec, top_k=5)
for r in results:
    print(r.score, r.payload["source"], r.payload["text"][:100])
```

---

### How do I index HTML pages or web content?

#### Setup

```bash
pip install beautifulsoup4
python smartqdrant.py init --name web-corpus --loader html
```

#### Index a single HTML file

```bash
python smartqdrant.py index \
  --config datasource_web-corpus.json \
  --source ./pages/article.html
```

#### How the HTML loader works

1. Reads the file with UTF-8 encoding
2. Parses with `BeautifulSoup(html, "html.parser")`
3. Extracts all visible text with `soup.get_text(separator=" ", strip=True)`
4. Splits the cleaned text into overlapping word-level chunks (using `chunk_size` and `chunk_overlap` from config)
5. Skips chunks with fewer than `min_chunk_words` (default 10) words

Script tags, style tags, and all HTML markup are stripped. Only visible text content is kept.

#### Downloading pages from the web before indexing

SmartQdrant does not crawl URLs directly. Download HTML first:

```bash
# Single page
curl -L https://example.com/docs/page > ./pages/page.html

# Multiple pages with wget
wget -r -l 2 -A .html -P ./pages/ https://example.com/docs/

# Python — download a list of URLs
python - <<'EOF'
import httpx
from pathlib import Path

urls = [
    "https://example.com/docs/intro",
    "https://example.com/docs/api",
]
Path("pages").mkdir(exist_ok=True)
for i, url in enumerate(urls):
    resp = httpx.get(url, follow_redirects=True)
    Path(f"pages/page_{i}.html").write_bytes(resp.content)
    print(f"Downloaded {url}")
EOF
```

Then index the downloaded directory:

```bash
python smartqdrant.py init --name web-corpus --loader directory
python smartqdrant.py index \
  --config datasource_web-corpus.json \
  --source ./pages/
```

---

### How do I index an entire directory of mixed file types?

```bash
python smartqdrant.py init --name mixed-corpus --loader directory
python smartqdrant.py index \
  --config datasource_mixed-corpus.json \
  --source ./corpus/
```

The directory loader walks the directory recursively and dispatches each file by extension:

| Extension | Loader used | Extra dependency |
|-----------|-------------|:----------------:|
| `.pdf` | PDFLoader | `pypdf` |
| `.md`, `.markdown` | MarkdownLoader | none |
| `.jsonl` | JSONLLoader | none |
| `.html`, `.htm` | HTMLLoader | `beautifulsoup4` |
| anything else | Skipped | — |

Files that are skipped are logged to stdout. If you need to index `.txt` or `.rst` files, add a plain-text loader (see [How do I add support for a custom document format?](#how-do-i-add-support-for-a-custom-document-format)).

#### Chunk sizing across formats

All loaders respect the same `chunk_size` and `chunk_overlap` config values. The PDF and HTML loaders use word-level chunking. The Markdown loader splits on headings rather than word count. The JSONL loader treats each JSON object as one chunk (no further splitting).

If you are mixing PDFs (which need word-level splitting) with Markdown (which splits by heading), be aware that Markdown chunks may be smaller than `chunk_size`. This is usually fine — the model handles variable-length chunks well.

---

### How do I add support for a custom document format?

#### Step 1 — Implement the DocumentLoader protocol

The protocol is in `ingestion/base.py` and requires a single method: `load(source: str) -> list[dict]`. Each returned dict must have a `"text"` key (string) and a `"metadata"` key (dict containing at least `"chunk_id"` and `"source"`).

```python
# ingestion/csv_loader.py
import csv
from pathlib import Path


class CSVLoader:
    """Load rows from a CSV file. Each row becomes one chunk."""

    def __init__(self, text_column: str = "text", min_words: int = 5):
        self.text_column = text_column
        self.min_words = min_words

    def load(self, source: str) -> list[dict]:
        docs = []
        with open(source, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                text = row.get(self.text_column, "").strip()
                if len(text.split()) < self.min_words:
                    continue
                # All columns except the text column go into metadata
                meta = {k: v for k, v in row.items() if k != self.text_column}
                meta.update({"source": Path(source).name, "chunk_id": i})
                docs.append({"text": text, "metadata": meta})
        return docs
```

#### Step 2 — Register in the loader factory

Open `ingestion/registry.py` and add before the final `raise ValueError`:

```python
if loader_type == "csv":
    from ingestion.csv_loader import CSVLoader
    return CSVLoader(
        text_column=cfg.get("csv_text_column", "text"),
        min_words=cfg.get("csv_min_words", 5),
    )
```

#### Step 3 — Add a Literal type to the config validator

Open `config.py` and update the `loader` field:

```python
loader: Literal["pdf", "markdown", "jsonl", "html", "directory", "csv"] = "pdf"
```

Also add any new config keys:

```python
csv_text_column: str = "text"
csv_min_words: int = 5
```

#### Step 4 — Use in your datasource config

```json
{
  "loader":           "csv",
  "csv_text_column":  "body",
  "csv_min_words":    10
}
```

#### Step 5 — Write a test

```python
# tests/test_csv_loader.py
def test_csv_loader_reads_rows(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("body,category\nHello world from a CSV file with words,A\nShort,B\n")

    from ingestion.csv_loader import CSVLoader
    loader = CSVLoader(text_column="body", min_words=3)
    docs = loader.load(str(csv_file))

    assert len(docs) == 1             # "Short" row skipped (< 3 words)
    assert "Hello world" in docs[0]["text"]
    assert docs[0]["metadata"]["category"] == "A"
    assert docs[0]["metadata"]["chunk_id"] == 0
```

---

## KV Cache & Phases

### What exactly is stored in the KV cache payload?

#### What gets stored

For each chunk, SmartQdrant runs one LLM forward pass (`model(**inputs, use_cache=True)`) and stores the resulting attention key-value tensors. Before storage, the per-token tensors are mean-pooled over the sequence length dimension, collapsing the variable-length token axis into a fixed-size representation.

```
Raw forward pass output:    [num_layers, 2, num_kv_heads, seq_len, head_dim]
After mean pool over seq:   [num_layers, 2, num_kv_heads, head_dim]
Cast to float16, base64:    stored in Qdrant payload as "kv_cache" string
```

#### Size calculation for common models

| Model | Layers | KV heads | Head dim | Size per chunk |
|-------|:------:|:--------:|:--------:|:--------------:|
| TinyLlama-1.1B | 22 | 4 | 64 | ~45 KB |
| Llama-3.2-3B | 28 | 8 | 128 | ~115 KB |
| Mistral-7B | 32 | 8 | 128 | ~131 KB |
| Llama-3.1-8B | 32 | 8 | 128 | ~131 KB |

Formula: `num_layers × 2 × num_kv_heads × head_dim × 2 bytes`

For 2,520 chunks with Llama 3.2 3B: `2520 × 115 KB ≈ 290 MB` of additional payload data in Qdrant.

#### All payload fields written by SmartQdrant

| Field | Type | Written by | Description |
|-------|------|-----------|-------------|
| `kv_cache` | string (base64) | `kv_indexer.py` | Mean-pooled KV tensor |
| `kv_version` | int or null | `kv_indexer.py` | LoRA version used to compute the tensor |
| `access_count` | int | `kv_background.py` | Total times retrieved by a query |
| `last_accessed_ts` | int | `kv_background.py` | Unix timestamp of most recent retrieval |
| `avg_retrieval_rank` | float | `kv_background.py` | Mean rank position across all retrievals |
| `parametric_hit_count` | int | `kv_background.py` | Times answered directly from model weights (Phase 3) |
| `tier` | string | `access_tracker.py` | `hot` / `warm` / `cold` / `frozen` |
| `text` | string | `kv_indexer.py` | Chunk text (used for KV computation and text-in-context fallback) |
| `page` | int | `kv_indexer.py` | Source page number |
| `source_file` | string | `kv_indexer.py` | Originating filename |
| `indexed_at` | int | `kv_indexer.py` | Unix timestamp of indexing |

---

### How does KV injection work under the hood?

#### The problem with standard RAG

In standard RAG, every query requires the LLM to process both the query and all retrieved chunks through the full attention stack:

```
Prompt = [system] + [chunk 1 text] + [chunk 2 text] + ... + [query]
model.generate(tokenize(Prompt))
  → attention over all tokens → generation
```

For a 5-chunk retrieval with 500 tokens per chunk, the model processes ~2,500 context tokens on every single query. This is slow and scales linearly with the number of retrieved chunks.

#### How KV injection avoids this

Instead of including chunk text in the prompt, SmartQdrant pre-computes what the LLM would have produced when attending over those chunks — the key-value tensors — and injects them directly as `past_key_values`:

```python
# Simplified from kv_inference.py
chunk_kvs = [kv_utils.deserialize_kv(c["kv_cache"], shape=kv_shape) for c in chunks]
past_kv = kv_utils.stack_past_key_values(chunk_kvs, ...)

output = model.generate(
    **tokenize(f"Answer: {query}"),
    past_key_values=past_kv,   # ← pre-loaded context; not re-processed
)
```

The model attends over the pre-loaded KV state when processing the query tokens, but never re-tokenizes or re-encodes the chunk text. The effective prompt for the LLM at query time is just the question — the "context" is already baked into the KV cache.

#### Performance comparison

| Method | Tokens processed at query time | Relative latency |
|--------|:------------------------------:|:----------------:|
| Text-in-context (Phase 1) | query + all chunk text (~2,500 tokens) | 1× (baseline) |
| KV injection (Phase 2) | query only (~20 tokens) | ~5–10× faster |
| Parametric (Phase 3) | query only, no retrieval | ~15–20× faster |

#### Why KV staleness matters

The KV tensors are computed using a specific set of model weights (identified by `kv_version`). After LoRA training, the model weights change. KV tensors computed with the old weights are incompatible with the new model — injecting them causes the model to attend over stale representations and produce worse answers. This is why SmartQdrant tracks `kv_version` and falls back to text-in-context for stale chunks.

---

### Why do some queries fall back to text-in-context even in Phase 2?

SmartQdrant checks every retrieved chunk before deciding on the inference path. A single stale or missing KV tensor causes the entire query to fall back to text-in-context.

#### Diagnosing the cause

```python
import json
import version as ver
from qdrant_client import QdrantClient

with open("datasource_my-corpus.json") as f:
    cfg = json.load(f)

ver.init(cfg)
current_ver = ver.get_lora_version()
print(f"Current LoRA version: {current_ver}")

client = QdrantClient(cfg["qdrant_host"], port=cfg["qdrant_port"])

# Count chunks with null kv_version (never had KV computed)
from qdrant_client.models import Filter, IsNullCondition
null_results, _ = client.scroll(
    collection_name=cfg["collection"],
    scroll_filter=Filter(must=[IsNullCondition(is_null={"key": "kv_version"})]),
    limit=1,
)
print(f"Chunks with kv_version=null: needs backfill")

# Count chunks with stale kv_version
from qdrant_client.models import FieldCondition, Range
stale_results, _ = client.scroll(
    collection_name=cfg["collection"],
    scroll_filter=Filter(must=[
        FieldCondition(key="kv_version", range=Range(lt=current_ver))
    ]),
    limit=1,
)
print(f"Chunks with kv_version < {current_ver}: stale, healing in background")
```

#### Fixing each cause

**`kv_version` is null (chunks never had KV computed)**

These were indexed using `smartqdrant.py index` or `bedrock_rag.py index` which only embed vectors, not KV tensors. Fix by running the KV indexer:

```bash
python kv_indexer.py --config datasource_my-corpus.json compute-kv
```

**`kv_version` < current LoRA version (stale after training)**

Background workers heal these automatically after first retrieval. If you want to pre-heal before the next query:

```bash
python kv_indexer.py --config datasource_my-corpus.json compute-kv \
  --stale-version <current_lora_version>
```

**`kv_cache` field missing entirely**

Chunks were inserted directly into Qdrant (not through SmartQdrant). Same fix — run `compute-kv`.

**Phase is actually 1, not 2**

```python
import json, version as ver
with open("datasource_my-corpus.json") as f: cfg = json.load(f)
ver.init(cfg)
print("Phase:", ver.get_phase())   # if this prints 1, KV injection is disabled
```

---

### How do I manually advance or roll back the phase?

SmartQdrant advances phases automatically when PRS thresholds are met during `index_and_train.py`. For manual control:

```python
import json
import version as ver

with open("datasource_my-corpus.json") as f:
    cfg = json.load(f)
ver.init(cfg)

# Inspect current state
state = ver.load()
print("Phase:", state.get("phase", 1))
print("LoRA version:", state.get("current_lora_version", 0))
print("PRS history:", state.get("prs_history", []))

# Advance to Phase 2 (enables KV injection)
ver.activate_phase_2()
print("Activated Phase 2")

# Advance to Phase 3 (enables confidence gate)
ver.activate_phase_3()
print("Activated Phase 3")

# Roll back to Phase 1 (disables KV injection and confidence gate)
ver.set_phase(1)
print("Rolled back to Phase 1")
```

Phase rollbacks are appropriate when:
- You observe answer quality degradation after a phase transition
- You want to A/B test response quality between phases
- A training run produced a high PRS but the model actually overfit to the FAQ set

To roll back from the command line without writing Python:

```bash
# Edit the version JSON file directly
python -c "
import json
with open('my-corpus_version.json') as f:
    v = json.load(f)
v['phase'] = 1
with open('my-corpus_version.json', 'w') as f:
    json.dump(v, f, indent=2)
print('Rolled back to Phase 1')
"
```

---

## Training & PRS

### How do I tune the PRS threshold?

PRS (Parametric Readiness Score) gates phase transitions. The threshold defaults to `0.75` — the model must score ≥ 0.75 before Phase 2 activates.

#### Setting the threshold

```json
{
  "prs_threshold": 0.70
}
```

#### Interpreting PRS values

| PRS range | Interpretation | Recommended action |
|-----------|---------------|-------------------|
| < 0.50 | Model has not learned the corpus | Check chunk quality, increase epochs, use better base model |
| 0.50–0.65 | Partial learning | More FAQs for evaluation, higher `lora_rank`, more epochs |
| 0.65–0.75 | Below default threshold | Try lowering threshold to 0.70 or run another training round |
| 0.75–0.85 | Good | Default threshold met; Phase 2 reliable |
| 0.85–0.92 | Excellent | Consider activating Phase 3 |
| > 0.92 | Near-optimal | Phase 3 appropriate; monitor for overfit |

#### Separate thresholds for Phase 2 and Phase 3

Both phase transitions use the same `prs_threshold`. If you want Phase 2 to activate easily but require a higher bar for Phase 3, run the evaluator manually and check the PRS score before manually calling `ver.activate_phase_3()`:

```python
# Check PRS and only activate Phase 3 if score is high enough
prs = run_prs_evaluation(cfg, faqs)
if prs >= 0.88:
    ver.activate_phase_3()
    print(f"Phase 3 activated (PRS={prs:.4f})")
else:
    print(f"PRS={prs:.4f} — staying in Phase 2")
```

---

### My PRS is not improving across training rounds — what do I do?

Work through this checklist in order:

#### 1. Verify FAQ quality first

Auto-generated FAQs can contain hallucinations. Inspect the output:

```bash
python tools/generate_faqs.py \
  --config datasource_my-corpus.json \
  --output faqs_review.json \
  --n 20
cat faqs_review.json | python -m json.tool | head -60
```

Check that question–answer pairs are factually grounded in your corpus. Delete any that are not. Hallucinated FAQs cause `accuracy` to be measured against incorrect ground truths, making PRS look artificially low.

#### 2. Increase the number of FAQs

PRS is averaged across all FAQ pairs. With fewer than 20 FAQs, a single bad answer swings the score significantly. Generate more:

```bash
python tools/generate_faqs.py \
  --config datasource_my-corpus.json \
  --output my_faqs.json \
  --n 100
```

#### 3. Increase training epochs

```json
{ "lora_epochs": 6 }
```

#### 4. Increase LoRA capacity

```json
{
  "lora_rank":  32,
  "lora_alpha": 64
}
```

Higher rank = more trainable parameters = higher capacity, at the cost of ~2× more VRAM for the LoRA adapter and slower training.

#### 5. Check chunk size and overlap

Very short chunks (< 50 words) give the model too little context to answer from weights. Very long chunks (> 1000 words) may confuse training. Optimal range is 150–600 words:

```json
{
  "chunk_size":    400,
  "chunk_overlap": 80
}
```

You will need to re-index if you change chunk sizes (existing chunks in the vector store were built with the old parameters).

#### 6. Use a larger base model

Smaller models have less memorization capacity. If you are using TinyLlama-1.1B on a large corpus, the model may not have the capacity to achieve high PRS regardless of training duration. Try Llama-3.2-3B or Mistral-7B.

#### 7. Check for catastrophic forgetting

If PRS was high in round N but drops in round N+1, the model is forgetting previously learned knowledge. Increase the replay buffer diversity:

```json
{
  "lora_epochs": 3,
  "lora_rank":   16
}
```

Lower learning rates also help: `"lora_lr": 0.0001` (default is `0.0002`).

---

### How do I bring my own FAQs for PRS evaluation?

Any JSON array where each object contains a question field and an answer field works.

#### Standard format

```json
[
  {
    "question": "What is the maximum file upload size?",
    "answer":   "100 MB per file, 1 GB per day"
  },
  {
    "question": "How do I reset my API key?",
    "answer":   "Go to Settings → API Keys → Revoke and regenerate"
  }
]
```

#### Custom field names

If your dataset uses different field names (common for HuggingFace QA datasets):

```json
{
  "faq_question_key": "query",
  "faq_answer_key":   "ground_truth"
}
```

Then your FAQ file can use:

```json
[
  {"query": "What year was the treaty signed?", "ground_truth": "1847"},
  {"query": "Who was the first president?",     "ground_truth": "George Washington"}
]
```

Common HuggingFace QA dataset schemas and the config keys they require:

| Dataset | Question field | Answer field | Config keys |
|---------|---------------|-------------|-------------|
| SQuAD | `question` | `answers.text[0]` | (standard — preprocess to flat) |
| Natural Questions | `question` | `annotations.short_answers` | (preprocess) |
| TriviaQA | `question` | `answer.value` | (preprocess) |
| Custom RAGAs format | `question` | `ground_truth` | `faq_question_key: question, faq_answer_key: ground_truth` |
| Custom Q&A CSV | any | any | set both keys accordingly |

#### Running evaluation

```bash
python prs_evaluator.py \
  --config datasource_my-corpus.json \
  --faqs my_faqs.json
```

Output:

```
PRS Evaluation
  Accuracy:    0.82  (41/50 questions answered correctly)
  Calibration: 0.74  (stated confidence correlates with correctness)
  Consistency: 0.81  (answers agree across 2 independent samples)
  ─────────────────────────────────────────────────────
  PRS Score:   0.79  ✅ Above threshold (0.75) — Phase 2 eligible
```

---

### How do I change the PRS scoring weights?

The default formula weights accuracy most heavily:

```
PRS = 0.5 × accuracy + 0.3 × calibration + 0.2 × consistency
```

Adjust in your config:

```json
{
  "prs_weights": {
    "accuracy":    0.7,
    "calibration": 0.2,
    "consistency": 0.1
  }
}
```

Weights must sum to 1.0.

#### When to change weights

**Emphasize accuracy** — you care most about factual correctness, less about confidence calibration:
```json
{ "prs_weights": { "accuracy": 0.8, "calibration": 0.1, "consistency": 0.1 } }
```

**Emphasize calibration** — your use case requires the model to know what it doesn't know (e.g. medical, legal):
```json
{ "prs_weights": { "accuracy": 0.4, "calibration": 0.5, "consistency": 0.1 } }
```

**Disable consistency** — you are running evaluation quickly and want to skip the second sampling pass (which doubles evaluation time):
```json
{ "prs_weights": { "accuracy": 0.6, "calibration": 0.4, "consistency": 0.0 } }
```

---

## Multi-Corpus & Production

### Can I run multiple independent corpora on the same instance?

Yes. Each corpus is fully isolated by its datasource config. The only shared resource is the Qdrant instance (or ChromaDB process) and the GPU.

#### Typical multi-corpus layout

```
project/
├── datasource_legal.json          ← legal document corpus
├── datasource_hr.json             ← HR policy corpus
├── datasource_engineering.json    ← technical documentation
│
├── legal_version.json             ← phase/PRS state for legal
├── hr_version.json
├── engineering_version.json
│
├── legal_replay.db                ← LoRA training replay buffer
├── hr_replay.db
├── engineering_replay.db
│
└── lora_checkpoints/
    ├── legal/                     ← LoRA adapter for legal
    ├── hr/
    └── engineering/
```

#### Running operations per corpus

```bash
# Index each corpus independently
python smartqdrant.py index --config datasource_legal.json --source ./legal_docs/
python smartqdrant.py index --config datasource_hr.json    --source ./hr_policies/

# Train each corpus — these are sequential (one GPU)
python index_and_train.py dummy.pdf --config datasource_legal.json       --faqs legal_faqs.json --skip-index
python index_and_train.py dummy.pdf --config datasource_engineering.json --faqs eng_faqs.json   --skip-index

# Query from the right corpus
python smartqdrant.py search --config datasource_legal.json       "What is the arbitration clause?"
python smartqdrant.py search --config datasource_engineering.json "How do I configure OAuth?"
```

#### Shared base model, separate LoRA adapters

`model_loader.py` caches the model by checkpoint path. Different LoRA adapters hot-swap on top of the same base model weights. The base model (e.g. Llama 3.2 3B) is loaded once per process; each corpus's adapter is merged in when that corpus's pipeline runs.

---

### How do I keep KV tensors fresh when I update my documents?

KV tensors are tied to both the document content and the LoRA adapter version. Changes to either require recomputation.

#### When you add new documents

```bash
# 1. Index the new documents (adds chunks, no KV yet)
python smartqdrant.py index --config datasource_my-corpus.json --source ./new_docs/

# 2. Compute KV tensors for the new chunks (they have kv_version=null)
python kv_indexer.py --config datasource_my-corpus.json compute-kv
```

#### When you update existing documents

There is no partial update — re-index the changed file (which replaces its chunks) then recompute:

```bash
# Re-index deletes old chunks and creates new ones
python smartqdrant.py index --config datasource_my-corpus.json --source ./updated_file.pdf

# Compute KV for the new chunks
python kv_indexer.py --config datasource_my-corpus.json compute-kv
```

#### After a LoRA training round

All existing KV tensors become stale because the model weights changed. Background workers heal them lazily (each chunk is recomputed the first time it is retrieved), but you can pre-heal the entire collection:

```bash
# Get the current LoRA version number
python -c "import json, version as ver; ver.init(json.load(open('datasource_my-corpus.json'))); print(ver.get_lora_version())"

# Recompute KV for all chunks with an outdated version
python kv_indexer.py --config datasource_my-corpus.json compute-kv \
  --stale-version <current_lora_version>
```

#### Continuous update strategy

For corpora that are updated frequently (e.g. daily news ingestion), run this after each update batch:

```bash
#!/bin/bash
# daily_update.sh
python smartqdrant.py index --config $CONFIG --source $NEW_DOCS_DIR
python kv_indexer.py --config $CONFIG compute-kv
echo "Update complete at $(date)"
```

---

### How do I monitor what is happening at runtime?

#### Live dashboard (recommended)

```bash
python monitoring_dashboard.py --config datasource_my-corpus.json
```

Open [http://localhost:8080](http://localhost:8080). The dashboard shows:

- **Phase and LoRA version** — current system state
- **Tier distribution** — how many chunks are hot / warm / cold / frozen
- **Top 10 chunks by access count** — which knowledge is used most
- **PRS history** — score trend across training rounds
- **Query rate** — approximate queries per minute

#### Version file inspection

```bash
cat my-corpus_version.json
```

```json
{
  "phase": 2,
  "current_lora_version": 3,
  "checkpoint_path": "lora_checkpoints/my-corpus/v3/",
  "prs_history": [
    {"round": 1, "prs": 0.71, "timestamp": 1742847000},
    {"round": 2, "prs": 0.79, "timestamp": 1742933400},
    {"round": 3, "prs": 0.84, "timestamp": 1743019800}
  ]
}
```

#### Background worker logs

```bash
# Run background workers and capture logs
nohup python kv_background.py --config datasource_my-corpus.json \
  > logs/kv_background.log 2>&1 &

# Tail the log
tail -f logs/kv_background.log
```

Log lines look like:
```
✅ kv_background workers started
[kv_background] Healed chunk 142 → kv_version=3
[kv_background] Access flush: 23 chunks updated
[kv_background] KV recompute error for chunk 99: CUDA out of memory
```

#### Checking collection health

```python
from vectorstore.registry import get_store
import json

with open("datasource_my-corpus.json") as f:
    cfg = json.load(f)

store = get_store(cfg)
total = store.count(cfg["collection"])
print(f"Total chunks: {total}")

# For Qdrant — count chunks per KV status
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, IsNullCondition
client = QdrantClient(cfg["qdrant_host"], port=cfg["qdrant_port"])
null_kv, _ = client.scroll(cfg["collection"],
    scroll_filter=Filter(must=[IsNullCondition(is_null={"key": "kv_version"})]),
    limit=1)
print(f"Chunks without KV: {len(null_kv)}")
```

---

### How do I reset everything and start over?

#### Full reset

```bash
# 1. Delete the collection from the vector store
python -c "
import json
from vectorstore.registry import get_store
with open('datasource_my-corpus.json') as f:
    cfg = json.load(f)
store = get_store(cfg)
if store.collection_exists(cfg['collection']):
    store.delete_collection(cfg['collection'])
    print(f'Deleted collection: {cfg[\"collection\"]}')
"

# 2. Remove phase/PRS state
rm -f my-corpus_version.json

# 3. Remove LoRA training state
rm -f my-corpus_replay.db
rm -rf lora_checkpoints/my-corpus/

# 4. For ChromaDB — also remove the persist directory
rm -rf .chroma/my-corpus/

# 5. Re-index from scratch
python smartqdrant.py index \
  --config datasource_my-corpus.json \
  --source ./my_document.pdf
```

#### Partial reset — keep vectors, remove KV state

If you want to keep your indexed vectors but start KV computation and training over (e.g. after switching to a different LLM):

```bash
# Remove only training state — keep the collection intact
rm -f my-corpus_version.json my-corpus_replay.db
rm -rf lora_checkpoints/my-corpus/

# Recompute KV with the new model (overwrites existing kv_cache fields)
python kv_indexer.py --config datasource_my-corpus.json compute-kv
```

---

### What are the GPU memory requirements?

#### Memory by model size

| Model | Parameters | VRAM for KV compute | VRAM for LoRA training | Recommended GPU |
|-------|:----------:|:-------------------:|:----------------------:|----------------|
| TinyLlama-1.1B | 1.1B | ~2 GB | ~4 GB | Any GPU with ≥ 6 GB VRAM |
| Llama-3.2-3B | 3B | ~6 GB | ~10 GB | RTX 3080, A10G, RTX 4080 |
| Mistral-7B | 7B | ~14 GB | ~20 GB | A10G (24 GB), RTX 3090, RTX 4090 |
| Llama-3.1-8B | 8B | ~16 GB | ~24 GB | A10G (24 GB), RTX 4090 |
| Mistral-22B | 22B | ~44 GB | OOM | A100 80GB or multi-GPU |

All sizes assume float16 precision. Batch size 1 during KV computation, batch size 4–8 during LoRA training.

#### Reducing memory usage

**4-bit quantization** (reduces inference VRAM by ~60%):

```bash
pip install bitsandbytes
```

```json
{
  "load_in_4bit": true
}
```

This is supported by `model_loader.py` via the `bitsandbytes` library. Quantized models cannot be used for gradient computation, so LoRA training falls back to a QLoRA approach (`prepare_model_for_kbit_training` from PEFT).

**Gradient checkpointing** (reduces training VRAM by ~30% at the cost of ~20% slower training):

```json
{
  "gradient_checkpointing": true
}
```

**Reducing LoRA rank** for smaller VRAM:

```json
{
  "lora_rank":  8,
  "lora_alpha": 16
}
```

#### AWS instance guide

| Instance | GPU | VRAM | Max model (training) | Monthly cost (on-demand) |
|----------|-----|:----:|:--------------------:|:------------------------:|
| g4dn.xlarge | T4 | 16 GB | Llama-3.2-3B | ~$0.53/hr |
| g5.xlarge | A10G | 24 GB | Mistral-7B | ~$1.01/hr |
| g5.2xlarge | A10G | 24 GB | Mistral-7B (more RAM) | ~$1.21/hr |
| p3.2xlarge | V100 | 16 GB | Llama-3.2-3B | ~$3.06/hr |
| p4d.24xlarge | 8× A100 | 8×40 GB | 70B models | ~$32.77/hr |

SmartQdrant was benchmarked on `g5.xlarge` with Llama 3.2 3B — this is the recommended starting point.

---

*Have a question not covered here? Open an issue at [github.com/hemantcgi/smartqdrant/issues](https://github.com/hemantcgi/smartqdrant/issues).*
