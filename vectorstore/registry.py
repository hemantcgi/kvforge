"""vectorstore/registry.py — Factory for VectorStore implementations."""


def get_store(cfg: dict):
    """Return the appropriate VectorStore for the given config.

    Dispatches on cfg['vector_store'] (default: 'qdrant').
    """
    backend = cfg.get("vector_store", "qdrant")
    if backend == "qdrant":
        from vectorstore.qdrant_store import QdrantStore
        return QdrantStore(
            host=cfg.get("qdrant_host", "localhost"),
            port=cfg.get("qdrant_port", 6333),
        )
    if backend == "chroma":
        persist_dir = cfg.get("chroma_persist_dir", ".chroma")
        from vectorstore.chroma_store import ChromaStore
        return ChromaStore(persist_dir=persist_dir)
    raise ValueError(
        f"Unknown vector_store '{backend}'. Choose: qdrant, chroma"
    )
