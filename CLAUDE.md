# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**KVForge** is a progressive system that transitions deployed question-answering systems from Retrieval-Augmented Generation (RAG) to **fully parametric answering** — a fine-tuned small language model answering directly from its weights with no retrieval. The central empirical result: a **2B-parameter Gemma-4-E2B-it**, LoRA-fine-tuned on cloud-LLM-generated QA pairs, **matches or exceeds text-in-context RAG on factual accuracy across three of four enterprise corpora** (Customer Support +71%, SQuAD +38%, Bedrock +44%, PubMedQA +64% with extended training).

Three phases of increasing autonomy:

- **Phase 1:** Standard text-in-context RAG
- **Phase 2:** KV cache injection — pre-computed KV tensors are injected at query time, bypassing chunk re-encoding (a latency optimization, not a quality win)
- **Phase 3:** Corpus-wide confidence-gated parametric answering — the confidence gate (entropy + hedging + query similarity) runs for every query

The four evaluation datasets are UC1 (Bitext customer support), UC2 (PubMedQA), UC3 (SQuAD 2.0), and UC4 (Amazon Bedrock User Guide).

Despite the repo folder being named `qdrant`, this is a **pure Python project** (no Rust, no Cargo).

## Common Commands

### Testing
```bash
# Run all tests (no GPU required)
python -m pytest tests/ -v --override-ini="addopts="

# Run a single test file
python -m pytest tests/test_kvforge.py -v --override-ini="addopts="

# Run a single test by name
python -m pytest tests/test_kvforge.py::test_init -v --override-ini="addopts="
```

### CLI Entry Points
```bash
# Initialize a datasource config
python kvforge.py init --name my-corpus

# Index documents
./scripts/index.sh datasource_my-corpus.json ./docs/

# Single-shot query
./scripts/ask.sh datasource_my-corpus.json "Your question"

# Full pipeline (Phase 1 → 2 → 3)
./scripts/run_pipeline.sh datasource_my-corpus.json ./docs/ faqs.json

# Individual pipeline steps
python -m pipeline.kv_indexer --config cfg.json index <source>
python -m pipeline.lora_trainer --config cfg.json --faqs faqs.json
python -m pipeline.prs_evaluator --config cfg.json --faqs faqs.json

# Generate FAQs via cloud LLM (sleep-time pre-computation)
python -m pipeline.sleep_faq_generator --config cfg.json --output faqs.json --count 50

# Start monitoring dashboard (per datasource)
./scripts/dashboard.sh datasource_my-corpus.json 8081

# Start KVForge Studio (multi-use-case web UI on port 8080)
python kvforge_portal.py --port 8080
```

## Architecture

### Module Map

```
core/
├── config.py           — Pydantic DatasourceConfig (all tunable parameters)
├── model_loader.py     — Thread-safe singleton LLM loader with optional LoRA
├── kv_utils.py         — KV tensor serialization/deserialization (base64 ↔ torch)
├── version.py          — Atomic state management (phase, LoRA version, PRS history)
├── confidence_gate.py  — Phase 3 decision logic
├── replay_buffer.py    — SQLite-backed weighted training sampler (tier system)
└── access_tracker.py   — Thread-safe hit counter and tier classification

pipeline/
├── kv_indexer.py       — Chunk → embed → KV tensors → upsert to vector store
├── kv_inference.py     — Query-time: inject KV tensors or fallback to text
├── kv_background.py    — Daemon: background KV recomputation + access flushing
├── lora_trainer.py     — LoRA fine-tuning with tier-weighted replay buffer
├── prs_evaluator.py    — Parametric Readiness Score (accuracy + calibration + consistency)
├── sleep_faq_generator.py — Offline FAQ generation via Gemini/Claude/OpenAI
├── monitoring_dashboard.py — FastAPI dashboard with tier stats and PRS history
└── index_and_train.py  — Orchestrator: spawns pipeline steps as subprocesses

studio/
├── kvforge_portal.py   — Main FastAPI app (port 8080) for multi-UC management
├── routes.py           — FastAPI router; HTML + SSE streams
├── api.py              — REST endpoints for use-case management
├── pipeline_runner.py  — Spawns pipeline steps; streams stdout/stderr via SSE
├── job_manager.py      — In-memory job tracking
└── gpu_monitor.py      — Free GPU detection

Pluggable backends (Protocol-based, no ABC/inheritance):
├── vectorstore/        — qdrant_store, chroma_store, faiss_store
├── embeddings/         — fastembed, sentence_transformer, openai
└── ingestion/          — pdf, markdown, jsonl, html, directory loaders
```

### Key Patterns

**Plugin Architecture:** All backends implement Python `@runtime_checkable` Protocol classes (structural typing only — no base classes). New backends are added by implementing the protocol interface without modifying existing code. See `docs/guides/adding-backends.md`.

**State Files (per datasource):**
- `version.json` — atomically updated via temp-file rename; tracks current phase (1/2/3), LoRA version, PRS history
- `<datasource>_replay.db` — SQLite replay buffer for tier-weighted LoRA training
- `lora_checkpoints/<datasource>/` — LoRA adapter checkpoints

**KV Tensor Storage:** Shape `[num_layers, 2, num_kv_heads, head_dim]`, stored as base64 in the vector store payload alongside the embedding vector. Tensors are versioned by LoRA round — stale tensors trigger background recomputation.

**Tier System (training prioritization):**
- **hot:** top 15% by access count, accessed < 7 days → weight 8
- **warm:** next 50%, accessed < 30 days → weight 4
- **cold:** remaining accessed chunks → weight 2
- **frozen:** never accessed → weight 1

**Enhanced Tier (full per-token KV cache):**
- Controlled by `enable_enhanced_tier` in `core/config.py` / datasource config.
- When `true`, hot chunks may be promoted to full per-token KV storage during background KV recompute.
- Default is `false` pending Component 6 empirical validation to prove the extra storage/compute cost is justified.

**KDS/fKDS-driven KV-injection eligibility:**
- `kds_threshold` (float or `None`) gates whether retrieved chunks are eligible for KV-injection using the legacy consistency-only KDS.
- `fkds_threshold` (float or `None`) gates eligibility using factual KDS (fKDS), a blend of consistency KDS and factual answer accuracy against FAQ ground truth. fKDS is preferred when both thresholds are configured; it was added because Component 6 validation found consistency-only KDS does not correlate with KV-injection quality.
- Either threshold is calibrated empirically in Component 6; a value of `None` (or absent key) for the active threshold disables KV injection and keeps the pipeline in `text_rag`-only mode (fail-closed).

**Subprocess Orchestration:** `index_and_train.py` and `studio/pipeline_runner.py` spawn pipeline steps as subprocesses. Studio streams stdout/stderr to the browser via SSE.

### Data Flow

**Index time:** Documents → DocumentLoader → chunking → Embedder → KV tensor computation (one LLM forward pass per chunk) → VectorStore upsert (embedding + base64 KV tensor + metadata)

**Query time:** Query → embed → retrieve top-K → if all chunks have fresh KV tensors: inject tensors (skip text encoding); else: text-in-context fallback + enqueue stale chunks for background recomputation

**Training time:** FAQ generation → LoRA fine-tune (tier-weighted) → KV recomputation with new weights → PRS evaluation → auto-advance phase if thresholds met

## Deployment Context

Designed for AWS EC2 g5.xlarge (NVIDIA A10G, 24GB VRAM). Multi-GPU setups run one use-case per GPU with per-UC isolation via `uc_config.json` and `CUDA_VISIBLE_DEVICES`. KVForge Studio on port 8080 manages all use-cases; per-UC monitoring dashboards run on configurable ports.
