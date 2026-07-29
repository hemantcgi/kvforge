# DatasourceConfig Reference

All configuration is stored in a `datasource_<name>.json` file created by `python kvforge.py init`.
The file is validated by the `DatasourceConfig` Pydantic model on load.

## Core Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `collection` | `str` | _(required)_ | Vector store collection name |
| `vector_store` | `"qdrant"\|"chroma"\|"faiss"` | `"qdrant"` | Vector store backend |
| `loader` | `"pdf"\|"markdown"\|"jsonl"\|"html"\|"directory"` | `"pdf"` | Document loader type |
| `embed_model` | `str` | `"BAAI/bge-small-en-v1.5"` | Embedding model identifier |
| `embedder_backend` | `"fastembed"\|"sentence_transformers"\|"openai"` | `"fastembed"` | Embedder backend |
| `vector_dim` | `int` | `384` | Embedding vector dimension (must match embed_model) |
| `llm_model` | `str` | `"meta-llama/Llama-3.2-3B-Instruct"` | HuggingFace LLM model ID |

## Vector Store Fields

| Field | Type | Default | Used when |
|-------|------|---------|-----------|
| `qdrant_host` | `str` | `"localhost"` | `vector_store="qdrant"` |
| `qdrant_port` | `int` | `6333` | `vector_store="qdrant"` |
| `chroma_persist_dir` | `str` | `".chroma"` | `vector_store="chroma"` |
| `faiss_persist_dir` | `str` | `".faiss"` | `vector_store="faiss"` |

## Chunking Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chunk_size` | `int` | `600` | Target chunk size in words |
| `chunk_overlap` | `int` | `60` | Overlap between consecutive chunks in words |
| `embed_batch` | `int` | `64` | Batch size for embedding computation |
| `upsert_batch` | `int` | `128` | Batch size for vector store upserts |
| `top_k` | `int` | `5` | Number of chunks to retrieve per query |

## LoRA Training Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lora_rank` | `int` | `16` | LoRA rank (higher = more parameters, more capacity) |
| `lora_alpha` | `int` | `32` | LoRA alpha (scaling factor, typically 2× rank) |
| `lora_target_modules` | `list[str]` | `["q_proj","k_proj","v_proj"]` | Attention layers to apply LoRA |
| `lora_dropout` | `float` | `0.05` | LoRA dropout rate |
| `lora_epochs` | `int` | `3` | Training epochs per LoRA round |
| `lora_lr` | `float` | `0.0002` | Learning rate |
| `checkpoint_dir` | `str` | `"lora_checkpoints/<name>/"` | Directory for LoRA adapter weights |

## PRS / Phase Transition Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prs_threshold` | `float` | `0.75` | PRS score required to advance to Phase 3 |
| `gate_threshold` | `float` | `0.75` | Confidence gate threshold within Phase 3 |
| `prs_weights` | `dict` | `{"accuracy":0.5,"calibration":0.3,"consistency":0.2}` | PRS component weights |

## FAQ / Evaluation Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `faq_question_key` | `str` | `"question"` | JSON key for FAQ question field |
| `faq_answer_key` | `str` | `"answer"` | JSON key for FAQ answer field |

## Operational Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `version_file` | `str` | `"<name>_version.json"` | Phase state file |
| `replay_db` | `str` | `"<name>_replay.db"` | SQLite replay buffer database |
| `access_flush_seconds` | `int` | `300` | Access counter flush interval (seconds) |
| `access_flush_queries` | `int` | `50` | Access counter flush interval (queries) |
| `dashboard_port` | `int` | `8080` | Monitoring dashboard port |
| `model_library` | `dict` | `{}` | Reserved for multi-model configurations |
