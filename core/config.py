"""Pydantic configuration model for a KVForge datasource.

A single ``DatasourceConfig`` object captures every tunable parameter for one
corpus: which vector store to use, which embedder, which LLM, LoRA hyper-
parameters, phase-gating thresholds, and monitoring settings.  Instances are
normally created by calling ``load_config`` with a JSON file path.
"""
import json
from typing import Literal
from pydantic import BaseModel, Field


class DatasourceConfig(BaseModel):
    """All runtime parameters for a single KVForge datasource.

    Fields are grouped by concern below.  Required fields (those with no
    default) must be supplied in the JSON config file.

    Attributes:
        qdrant_host: Hostname of the Qdrant server (used when
            ``vector_store='qdrant'``).
        qdrant_port: Port of the Qdrant server.
        vector_store: Which vector store backend to use.  One of
            ``'qdrant'``, ``'chroma'``, ``'faiss'``.
        chroma_persist_dir: Local directory for ChromaDB data files
            (used when ``vector_store='chroma'``).
        faiss_persist_dir: Local directory for FAISS index files
            (used when ``vector_store='faiss'``).
        collection: Name of the vector store collection for this datasource.
            **Required.**
        loader: Document loader format.  One of ``'pdf'``, ``'markdown'``,
            ``'jsonl'``, ``'html'``, ``'directory'``.
        chunk_size: Target chunk size in words for PDF and directory loaders.
        chunk_overlap: Word overlap between consecutive chunks.
        embed_batch: Number of texts to embed in a single batch.
        upsert_batch: Number of points to upsert per batch call.
        top_k: Default number of nearest neighbours to retrieve.
        jsonl_text_key: Field name that holds the text in each JSONL object.
        embedder_backend: Embedding backend.  One of ``'fastembed'``,
            ``'sentence_transformers'``, ``'openai'``.
        embed_model: Model name or identifier for the embedder backend.
            **Required.**
        vector_dim: Dimensionality of the embedding vectors.  **Required.**
        llm_model: HuggingFace model ID for the language model.  **Required.**
        hf_token: HuggingFace access token for gated models (optional).
        max_new_tokens: Maximum tokens the LLM generates per response.
        model_library: Optional registry mapping model IDs to their KV shape
            (``kv_num_layers``, ``kv_num_heads``, ``kv_head_dim``).  Overrides
            auto-detection when present.
        lora_rank: LoRA rank ``r`` used in ``LoraConfig``.
        lora_alpha: LoRA scaling factor ``alpha``.
        lora_target_modules: List of module name suffixes to apply LoRA to
            (e.g. ``['q_proj', 'k_proj', 'v_proj']``).
        lora_dropout: Dropout probability applied to LoRA layers.
        lora_epochs: Number of training epochs per LoRA round.
        lora_lr: Learning rate for LoRA fine-tuning.
        checkpoint_dir: Directory where LoRA checkpoint sub-folders are saved.
            **Required.**
        version_file: Path to the JSON file tracking LoRA version and phase.
            **Required.**
        replay_db: Path to the SQLite replay-buffer database.  **Required.**
        gate_threshold: Confidence-gate threshold; queries above this score
            are answered parametrically (Phase 3 only).
        prs_threshold: Minimum PRS required for phase transitions.
        prs_weights: Weights for the three PRS components: ``'accuracy'``,
            ``'calibration'``, ``'consistency'``.
        faq_question_key: Key name for the question field in FAQ JSON files.
        faq_answer_key: Key name for the answer field in FAQ JSON files.
        access_flush_seconds: Background access-tracker flush interval in
            seconds.
        access_flush_queries: Background access-tracker flush query-count
            trigger.
        dashboard_port: Port for the monitoring dashboard FastAPI server.
    """

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
    """Load and validate a datasource config from a JSON file.

    Template annotation keys that begin with ``_`` (e.g. ``_comment``,
    ``_vector_store_options``) are stripped before validation so that
    annotated template files can be used directly.

    Args:
        path: Path to the JSON config file.

    Returns:
        A validated ``DatasourceConfig`` instance.

    Raises:
        pydantic.ValidationError: If any required field is missing or a value
            fails type/constraint validation.
    """
    with open(path) as f:
        data = json.load(f)
    # Strip _comment and _*_options keys (template annotations, not fields)
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    return DatasourceConfig(**data)
