# VectorStore API Reference

## Protocol

All vector store backends implement the `VectorStore` runtime-checkable Protocol from `vectorstore/base.py`.

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `create_collection` | `(name: str, dim: int) → None` | Create a new collection with given vector dimension |
| `collection_exists` | `(name: str) → bool` | Return True if collection exists |
| `delete_collection` | `(name: str) → None` | Delete collection and all its data |
| `upsert` | `(collection: str, points: list[Point]) → None` | Insert or update points |
| `query` | `(collection: str, vector: list[float], top_k: int) → list[ScoredPoint]` | Nearest-neighbour search |
| `scroll` | `(collection: str, limit: int, offset: int) → list[ScoredPoint]` | Page through all points |
| `set_payload` | `(collection: str, point_id: int, payload: dict) → None` | Merge payload fields for a point |
| `count` | `(collection: str) → int` | Return total point count |

### Data Classes

```python
@dataclass
class Point:
    id: int
    vector: list[float]
    payload: dict

@dataclass
class ScoredPoint:
    id: int
    score: float
    payload: dict
```

## Backends

### Qdrant (`vector_store: "qdrant"`)

- **Install:** `pip install qdrant-client`
- **Requires:** Qdrant server (Docker: `docker run -p 6333:6333 qdrant/qdrant`)
- **Config fields:** `qdrant_host` (default: `"localhost"`), `qdrant_port` (default: `6333`)
- **Best for:** Production, multi-corpus, advanced filtering, cloud deployment

### ChromaDB (`vector_store: "chroma"`)

- **Install:** `pip install chromadb`
- **Requires:** Nothing (in-process, persistent to disk)
- **Config fields:** `chroma_persist_dir` (default: `".chroma"`)
- **Best for:** Local development, single-machine, no Docker

### FAISS (`vector_store: "faiss"`)

- **Install:** `pip install faiss-cpu` (or `faiss-gpu` for GPU)
- **Requires:** Nothing (fully offline)
- **Config fields:** `faiss_persist_dir` (default: `".faiss"`)
- **Persistence:** Two files per collection: `<name>.index` (FAISS binary) + `<name>.meta.pkl` (payloads)
- **Best for:** Fully offline, air-gapped environments

## Comparison

| Feature | Qdrant | ChromaDB | FAISS |
|---------|--------|---------|-------|
| Docker needed | Yes | No | No |
| Persistent | Yes | Yes | Yes |
| Filtering | Rich | Basic | None |
| Multi-tenant | Yes | Yes | Manual |
| GPU support | No | No | Optional |
| Production-ready | Yes | Dev/test | Research |
