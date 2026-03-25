# Vector Stores

← [Back to FAQ index](../../FAQ.md)

---

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

← [Back to FAQ index](../../FAQ.md)
