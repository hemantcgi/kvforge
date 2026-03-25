"""vectorstore/registry.py — Factory for VectorStore implementations."""


def get_store(cfg: dict):
    """Return the appropriate VectorStore for the given config.

    Dispatches on ``cfg['vector_store']`` (default: ``'qdrant'``).

    Supported backends
    ------------------
    ``qdrant``  — Qdrant server (Docker or cloud).  Requires ``qdrant-client``.
    ``chroma``  — ChromaDB in-process persistent store.  Requires ``chromadb``.
    ``faiss``   — FAISS flat index persisted to disk.  Requires ``faiss-cpu``.
    """
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

    if backend == "faiss":
        from vectorstore.faiss_store import FAISSStore
        return FAISSStore(persist_dir=cfg.get("faiss_persist_dir", ".faiss"))

    raise ValueError(
        f"Unknown vector_store '{backend}'. "
        f"Supported: qdrant, chroma, faiss"
    )
