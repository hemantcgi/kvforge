# SmartQdrant API Reference

## Module Overview

| Module | Location | Purpose |
|--------|----------|---------|
| `config` | `config.py` | `DatasourceConfig` Pydantic model — all tunable parameters |
| `kv_utils` | `kv_utils.py` | KV tensor serialization, mean pooling, cache format conversion |
| `model_loader` | `model_loader.py` | Thread-safe singleton LLM + tokenizer loader |
| `version` | `version.py` | Atomic read/write of `version.json` phase state |
| `confidence_gate` | `confidence_gate.py` | Phase 3 gate: entropy + hedging + similarity scoring |
| `replay_buffer` | `replay_buffer.py` | SQLite-backed weighted training sampler |
| `access_tracker` | `access_tracker.py` | Thread-safe tier classification (hot/warm/cold/frozen) |
| `pipeline.kv_indexer` | `pipeline/kv_indexer.py` | Chunk + embed + KV tensor computation CLI |
| `pipeline.kv_inference` | `pipeline/kv_inference.py` | Phase 1/2/3 query-time inference |
| `pipeline.kv_background` | `pipeline/kv_background.py` | Background KV healing and access flush daemon |
| `pipeline.lora_trainer` | `pipeline/lora_trainer.py` | LoRA fine-tuning CLI |
| `pipeline.prs_evaluator` | `pipeline/prs_evaluator.py` | Parametric Readiness Score evaluation CLI |
| `pipeline.monitoring_dashboard` | `pipeline/monitoring_dashboard.py` | FastAPI monitoring server |
| `pipeline.index_and_train` | `pipeline/index_and_train.py` | Subprocess-based pipeline orchestrator |
| `vectorstore` | `vectorstore/` | VectorStore protocol + backends |
| `embeddings` | `embeddings/` | Embedder protocol + backends |
| `ingestion` | `ingestion/` | DocumentLoader protocol + backends |

## Detail Pages

- [config.md](config.md) — `DatasourceConfig` fields
- [pipeline.md](pipeline.md) — Pipeline module CLI and API
- [vectorstore.md](vectorstore.md) — VectorStore protocol and backends
- [embeddings.md](embeddings.md) — Embedder protocol and backends
- [ingestion.md](ingestion.md) — DocumentLoader protocol and backends

## Auto-generated API docs

Run `./scripts/generate_docs.sh` to generate full HTML API documentation from docstrings
into `docs/api/generated/` (not committed to git).
