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
