# SmartQdrant — Qdrant-LLM Coupled Attention System

A production-ready RAG system that pre-computes and stores LLM KV-cache tensors directly in Qdrant, enabling faster inference by skipping re-encoding of retrieved chunks. Includes LoRA fine-tuning, tier-based access tracking, a confidence gate for Phase 3 parametric inference, and a live monitoring dashboard.

---

## How It Works

```
PDF → chunk + embed → Qdrant
                          ↓
               KV tensors computed per chunk (LLM forward pass)
               stored in Qdrant payload as base64 float16
                          ↓
Query → embed → vector search → retrieve top-K chunks
                          ↓
           All KV fresh?  ─── yes ──→  KV injection (fast path)
                │ no
                ↓
        text-in-context fallback + enqueue stale chunks for background heal
                          ↓
                     LLM generates answer
```

### Phase State Machine

| Phase | Behaviour |
|-------|-----------|
| 1 | Standard RAG — text-in-context only |
| 2 | KV injection enabled — fresh chunks skip re-encoding |
| 3 | Confidence gate — high-confidence queries answered from model weights directly |

Phase advances automatically: Phase 2 activates after first successful LoRA training cycle; Phase 3 after PRS ≥ 0.80 for two consecutive rounds.

---

## Architecture

| File | Responsibility |
|------|---------------|
| `bedrock_rag.py` | PDF → chunk → embed → Qdrant (SP1 baseline) |
| `kv_indexer.py` | Extended indexer: chunk + embed + KV compute + upsert |
| `kv_utils.py` | KV tensor ops: mean_pool, serialize, deserialize, stack |
| `model_loader.py` | Singleton LLM + LoRA loader (GPU) |
| `version.py` | Atomic version.json I/O, phase transitions |
| `replay_buffer.py` | SQLite-backed tier-weighted chunk sampler |
| `lora_trainer.py` | LoRA fine-tuning with replay buffer |
| `prs_evaluator.py` | Parametric Readiness Score: accuracy + calibration + self-consistency |
| `index_and_train.py` | Orchestrator: index → train → KV refresh → PRS → phase advance |
| `kv_background.py` | Daemon threads: KV recompute queue + access tracker flush |
| `kv_inference.py` | Query-time: KV inject or text fallback + stale chunk healing |
| `confidence_gate.py` | Phase 3: entropy + hedging + query-similarity gate |
| `access_tracker.py` | Thread-safe hit counter, tier classification, weekly report |
| `monitoring_dashboard.py` | FastAPI dashboard at :8080 |

---

## Requirements

### Local (CPU — indexing, tests, dashboard)
```bash
pip install qdrant-client fastembed pymupdf fastapi uvicorn httpx pytest pytest-xdist numpy
```

### GPU server (LoRA training, KV compute, inference)
```bash
pip install -r requirements_gpu.txt
# torch transformers peft bitsandbytes accelerate datasets fastapi uvicorn httpx
```

---

## Quick Start

### 1. Start Qdrant
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v ~/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### 2. Create a datasource config

Copy `datasource_template.json` and fill in your values:
```bash
cp datasource_template.json my_datasource.json
```

Key fields:
```json
{
  "qdrant_host": "localhost",
  "qdrant_port": 6333,
  "collection": "my-collection",
  "embed_model": "BAAI/bge-small-en-v1.5",
  "llm_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "checkpoint_dir": "lora_checkpoints/my-datasource/",
  "version_file": "my_datasource_version.json",
  "replay_db": "my_datasource_replay.db"
}
```

### 3. Create the Qdrant collection
```bash
curl -X PUT http://localhost:6333/collections/my-collection \
  -H "Content-Type: application/json" \
  -d '{"vectors": {"size": 384, "distance": "Cosine"}}'
```

### 4. Run the full pipeline (GPU required)
```bash
python3 index_and_train.py my_document.pdf --config my_datasource.json
```

This runs in sequence:
1. Chunk + embed + upsert to Qdrant
2. Compute KV tensors for all chunks
3. LoRA fine-tuning with tier-weighted replay buffer
4. Recompute KV tensors with updated LoRA weights
5. PRS evaluation (accuracy / calibration / self-consistency)
6. Activate Phase 2 if PRS threshold met

### 5. Query
```bash
python3 -c "
import json, sys
sys.path.insert(0, '.')
import kv_background, kv_inference
with open('my_datasource.json') as f:
    cfg = json.load(f)
kv_background.start(cfg)
print(kv_inference.answer_with_retrieval('Your question here', cfg))
"
```

### 6. Start the monitoring dashboard
```bash
python3 monitoring_dashboard.py
# Open http://localhost:8080
```

Dashboard shows:
- Current phase and LoRA version
- Tier distribution (hot / warm / cold / frozen)
- Top 10 chunks by access count
- PRS history across training rounds

---

## Using SmartQdrant with an Existing Qdrant Instance

SmartQdrant is a **middleware addon** — it does not modify Qdrant. It uses the standard `qdrant-client` to read and write extra payload fields on your existing points. Your Qdrant instance, collections, REST API, and built-in dashboard all continue to work unchanged.

```
┌─────────────────────────────────────┐
│         Your Application            │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         SmartQdrant Layer           │
│  kv_inference.py / confidence_gate  │
│  lora_trainer / access_tracker      │
│  monitoring_dashboard (:8080)       │
└──────────────┬──────────────────────┘
               │  standard qdrant-client
┌──────────────▼──────────────────────┐
│   Your Existing Qdrant (unchanged)  │
│   REST :6333 / dashboard :6333/dashboard │
└─────────────────────────────────────┘
```

### Step 1 — Point the config at your existing collection

Create a config file for your collection:

```bash
cp datasource_template.json my_existing.json
```

Edit `my_existing.json` to match your setup:

```json
{
  "qdrant_host": "your-qdrant-host",
  "qdrant_port": 6333,
  "collection": "your-existing-collection",
  "embed_model": "BAAI/bge-small-en-v1.5",
  "llm_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "top_k": 5,
  "checkpoint_dir": "lora_checkpoints/my_existing/",
  "version_file": "my_existing_version.json",
  "replay_db": "my_existing_replay.db"
}
```

> The `embed_model` must match the model used to embed your existing vectors (same dimensionality).

### Step 2 — Backfill KV tensors on existing points (GPU required)

SmartQdrant stores a `kv_cache` tensor in each point's payload. Run this once to compute them for all existing chunks:

```bash
# Backfill all points that have no kv_cache yet
python3 kv_indexer.py --config my_existing.json compute-kv

# Or backfill only a specific source file
python3 kv_indexer.py --config my_existing.json compute-kv --source-file mydoc.pdf
```

This reads each point's `text` field, runs it through the LLM forward pass, and writes the `kv_cache` + `kv_version` payload fields back to Qdrant. Your vectors and other payload fields are untouched.

### Step 3 — Run LoRA training on your data

```bash
python3 lora_trainer.py --config my_existing.json
```

### Step 4 — Evaluate and activate Phase 2

```bash
python3 prs_evaluator.py --config my_existing.json --faqs your_faqs.json
```

If PRS ≥ 0.75, activate Phase 2 manually:

```bash
python3 -c "
import json, version as ver
with open('my_existing.json') as f: cfg = json.load(f)
ver.init(cfg)
ver.activate_phase_2()
print('Phase 2 active:', ver.get_phase())
"
```

### Step 5 — Route queries through SmartQdrant

Replace your direct Qdrant search calls with:

```python
import json, sys
sys.path.insert(0, '/path/to/smartqdrant')
import kv_background, kv_inference

with open('my_existing.json') as f:
    cfg = json.load(f)

kv_background.start(cfg)   # start background workers (call once)

answer = kv_inference.answer_with_retrieval("Your question", cfg)
print(answer)
```

### Step 6 — Start the monitoring dashboard

```bash
python3 monitoring_dashboard.py --config my_existing.json
# Open http://localhost:8080
```

### What SmartQdrant adds to your existing points

When backfilling, these fields are written to each point's payload:

| Field | Type | Description |
|-------|------|-------------|
| `kv_cache` | string (base64) | Mean-pooled KV tensor for this chunk |
| `kv_version` | int | LoRA version used to compute the cache |
| `access_count` | int | Query hit count (updated at runtime) |
| `last_accessed_ts` | int | Unix timestamp of last access |
| `avg_retrieval_rank` | float | Average rank when retrieved |
| `parametric_hit_count` | int | Times answered from weights (Phase 3) |
| `tier` | string | hot / warm / cold / frozen |

All other fields in your existing payload are preserved.

### Requirements for existing collections

- Points must have a `text` field containing the chunk text (used for KV computation and text-in-context fallback)
- Points must have a `page` field (int) for source tracking — add it if missing
- Vector dimension must match `embed_model` output (384 for `BAAI/bge-small-en-v1.5`)

---

## Running Tests

All 31 tests run locally without a GPU:

```bash
pytest tests/test_kv_utils.py tests/test_kv_indexer.py \
       tests/test_lora_trainer.py tests/test_kv_inference.py \
       tests/test_confidence_gate.py tests/test_access_tracker.py \
       tests/test_dashboard.py tests/test_integration_smoke.py -v
```

---

## Tier System

Chunks are classified based on access frequency and recency:

| Tier | Condition | Replay weight |
|------|-----------|---------------|
| hot | Top 15% by access count, last accessed < 7 days | 8 |
| warm | Next 50%, last accessed < 30 days | 4 |
| cold | Remaining accessed chunks | 2 |
| frozen | Never accessed | 1 |

Tier weights control how often each chunk appears in LoRA training batches, preventing catastrophic forgetting of frequently-used knowledge.

---

## Per-Datasource Isolation

Each datasource gets its own isolated state:

| Config key | Purpose |
|------------|---------|
| `version_file` | Phase, PRS history, known-good queries |
| `replay_db` | SQLite replay buffer |
| `checkpoint_dir` | LoRA checkpoints |

This allows multiple independent document collections on the same Qdrant instance.

---

## Confidence Gate (Phase 3)

When Phase 3 is active, each query is evaluated on three signals before deciding whether to retrieve:

| Signal | Weight | Meaning |
|--------|--------|---------|
| Token entropy | 0.4 | Low entropy = model is confident |
| Hedging score | 0.3 | Absence of hedging phrases = confident |
| Query similarity | 0.3 | Similarity to known-good queries |

If `P(no_retrieval) >= gate_threshold` (default 0.75), the model answers directly from weights. Otherwise it falls back to KV injection / text-in-context retrieval.

---

## EC2 Deployment (GPU)

Tested on AWS g5.xlarge (NVIDIA A10G, 24 GB VRAM).

```bash
# Sync code
rsync -avz --exclude='venv/' --exclude='__pycache__/' \
  -e "ssh -i your-key.pem" \
  ./ ubuntu@<ec2-ip>:~/smartqdrant/

# SSH in and install deps
ssh -i your-key.pem ubuntu@<ec2-ip>
cd ~/smartqdrant
python3 -m venv venv
venv/bin/pip install -r requirements_gpu.txt
venv/bin/pip install qdrant-client fastembed pymupdf

# Run pipeline
venv/bin/python3 index_and_train.py document.pdf --config datasource_bedrock.json
```

Verified results on the Amazon Bedrock Dataset (2520 chunks):
- KV compute: ~498s
- LoRA training: 3 epochs, 474 steps
- **PRS = 0.8512** — Phase 2 activated automatically

---

## License

MIT
