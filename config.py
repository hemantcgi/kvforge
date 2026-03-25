"""config.py — Pydantic model for SmartQdrant datasource configuration."""
import json
from typing import Literal
from pydantic import BaseModel, Field


class DatasourceConfig(BaseModel):
    # Vector store connection
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    vector_store: Literal["qdrant", "chroma", "faiss"] = "qdrant"

    # ChromaDB (in-process)
    chroma_persist_dir: str = ".chroma"

    # FAISS (in-process)
    faiss_persist_dir: str = ".faiss"

    # Collection & ingestion
    collection: str
    loader: Literal["pdf", "markdown", "jsonl", "html", "directory"] = "pdf"
    chunk_size: int = 600
    chunk_overlap: int = 60
    embed_batch: int = 64
    upsert_batch: int = 128
    top_k: int = 5
    jsonl_text_key: str = "text"

    # Embedding
    embedder_backend: Literal["fastembed", "sentence_transformers", "openai"] = "fastembed"
    embed_model: str
    vector_dim: int

    # Language model
    llm_model: str
    hf_token: str | None = None
    max_new_tokens: int = 256
    model_library: dict = Field(default_factory=dict)

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj"]
    )
    lora_dropout: float = 0.05
    lora_epochs: int = 3
    lora_lr: float = 0.0002

    # State files
    checkpoint_dir: str
    version_file: str
    replay_db: str

    # Phase gating
    gate_threshold: float = 0.75
    prs_threshold: float = 0.75
    prs_weights: dict = Field(
        default_factory=lambda: {"accuracy": 0.5, "calibration": 0.3, "consistency": 0.2}
    )

    # FAQ schema
    faq_question_key: str = "question"
    faq_answer_key: str = "answer"

    # Dashboard
    access_flush_seconds: int = 300
    access_flush_queries: int = 50
    dashboard_port: int = 8080


def load_config(path: str) -> DatasourceConfig:
    """Load and validate a datasource config from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    # Strip _comment and _*_options keys (template annotations, not fields)
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    return DatasourceConfig(**data)
