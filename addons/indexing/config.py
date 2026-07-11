from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class IndexingConfig(BaseModel):
    """Configuration for the KV indexing pipeline addon."""

    # Document loading
    loader: Literal["pdf", "markdown", "jsonl", "html", "directory",
                    "docx", "pptx", "xlsx", "zip"] = "pdf"
    chunk_size: int = 600
    chunk_overlap: int = 60
    embed_batch: int = 64
    upsert_batch: int = 128
    jsonl_text_key: str = "text"
    rows_per_chunk: int = 50  # xlsx loader

    # Embedding
    embedder_backend: Literal["fastembed", "sentence_transformers", "openai", "clip"] = "fastembed"
    embed_model: str
    vector_dim: int
    openai_api_key: str = ""
    clip_model: str = "openai/clip-vit-base-patch32"

    # Vector store selection
    vector_store: Literal["qdrant", "chroma", "faiss", "pinecone",
                           "pgvector", "weaviate", "milvus"] = "qdrant"

    # Qdrant connection
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # ChromaDB
    chroma_persist_dir: str = ".chroma"

    # FAISS
    faiss_persist_dir: str = ".faiss"

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # pgvector
    pgvector_dsn: str = ""
    pgvector_table: str = ""

    # Weaviate
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str = ""

    # Milvus
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""

    # KV tensor shape (overrides auto-detection per model)
    model_library: dict = Field(default_factory=dict)

    # Dynamic PRS clustering
    cluster_k_range: list[int] = Field(default_factory=lambda: [3, 20])
