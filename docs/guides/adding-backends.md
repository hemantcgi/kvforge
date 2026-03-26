# Adding New Backends

KVForge uses a protocol + registry pattern for all three pluggable subsystems.
Adding a new backend is always the same three steps:

1. Implement the protocol in a new file
2. Register it in the registry
3. Add any new config fields to `DatasourceConfig`

---

## Adding a VectorStore backend

### Step 1: Implement `VectorStore` protocol

Create `vectorstore/<name>_store.py`. Implement all 8 methods:

```python
# vectorstore/mystore_store.py
from vectorstore.base import VectorStore, Point, ScoredPoint

class MyStore:
    def __init__(self, **kwargs):
        # initialise connection
        pass

    def create_collection(self, name: str, dim: int) -> None: ...
    def collection_exists(self, name: str) -> bool: ...
    def delete_collection(self, name: str) -> None: ...
    def upsert(self, collection: str, points: list[Point]) -> None: ...
    def query(self, collection: str, vector: list[float], top_k: int = 5) -> list[ScoredPoint]: ...
    def scroll(self, collection: str, limit: int = 100, offset: int = 0) -> list[ScoredPoint]: ...
    def set_payload(self, collection: str, point_id: int, payload: dict) -> None: ...
    def count(self, collection: str) -> int: ...
```

### Step 2: Register in `vectorstore/registry.py`

Add an `elif` branch **before** the final `raise ValueError`:

```python
    if backend == "mystore":
        from vectorstore.mystore_store import MyStore
        return MyStore(
            host=cfg.get("mystore_host", "localhost"),
            port=cfg.get("mystore_port", 1234),
        )

    raise ValueError(
        f"Unknown vector_store '{backend}'. "
        f"Supported: qdrant, chroma, faiss, mystore"
    )
```

### Step 3: Add config fields to `config.py`

In `config.py`, update `DatasourceConfig`:

```python
vector_store: Literal["qdrant", "chroma", "faiss", "mystore"] = "qdrant"
mystore_host: str = "localhost"
mystore_port: int = 1234
```

---

## Adding an Embedder backend

### Step 1: Implement `Embedder` protocol

Create `embeddings/<name>_embedder.py`:

```python
# embeddings/myembedder_embedder.py
class MyEmbedder:
    def __init__(self, model_name: str, **kwargs):
        # load model
        self.dim = 768  # set actual dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        # return list of embedding vectors
        ...
```

### Step 2: Register in `embeddings/registry.py`

```python
    if backend == "myembedder":
        from embeddings.myembedder_embedder import MyEmbedder
        return MyEmbedder(model_name=cfg.get("embed_model", "my-default-model"))
```

---

## Adding a DocumentLoader backend

### Step 1: Implement `DocumentLoader` protocol

Create `ingestion/<name>_loader.py`:

```python
# ingestion/myformat_loader.py
class MyFormatLoader:
    def load(self, source: str) -> list[dict]:
        # Return list of {"text": str, "metadata": dict} dicts
        # metadata must include at least {"source": str}
        ...
```

### Step 2: Register in `ingestion/registry.py`

```python
    if loader_type == "myformat":
        from ingestion.myformat_loader import MyFormatLoader
        return MyFormatLoader(
            chunk_size=cfg.get("chunk_size", 600),
            chunk_overlap=cfg.get("chunk_overlap", 60),
        )
```

### Step 3: Add to `kvforge.py` init choices

```python
    p_init.add_argument("--loader", default="pdf",
                         choices=["pdf", "markdown", "jsonl", "html", "directory", "myformat"])
```
