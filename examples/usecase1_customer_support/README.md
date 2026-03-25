# Use Case 1: Customer Support Q&A with Qdrant

This example demonstrates SmartQdrant on a real customer-support dataset,
using **Qdrant** as the vector store — the recommended backend for production
deployments that need efficient KV backfill, horizontal scaling, and remote access.

## Dataset

**[Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)**

| Property | Value |
|----------|-------|
| Source | HuggingFace — `bitext/Bitext-customer-support-llm-chatbot-training-dataset` |
| Domain | E-commerce customer support |
| Size used | 2 000 Q&A pairs (full dataset: ~26 000) |
| Format | Instruction / Response pairs across 27 intents |
| Why this dataset | Real conversational Q&A pairs with intent labels — perfect for PRS evaluation out of the box; no preprocessing required |

## Architecture

```
HuggingFace Dataset
        │
        ▼
  setup.py (download + format)
        │
        ▼ JSONL corpus
  smartqdrant.py index ──────────────► Qdrant (localhost:6333)
        │                                      │
        ▼                                      │
  pipeline/kv_indexer.py compute-kv ◄─── read chunks ──┘
        │                                      │
        ▼ KV tensors                           │ stored back
  ──────────────────────────────────────────────┘
        │
        ▼
  pipeline/lora_trainer.py ──── FAQs ──────────► LoRA checkpoint
        │
        ▼
  pipeline/kv_indexer.py compute-kv (re-run with updated weights)
        │
        ▼
  pipeline/prs_evaluator.py ──────────────────► PRS score (target ≥ 0.75)
        │
        ▼
  ask.py / pipeline/monitoring_dashboard.py
```

## Prerequisites

### 1. Start Qdrant

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

Verify: `curl http://localhost:6333/healthz` → `{"title":"qdrant","version":"..."}`

### 2. Install dependencies

```bash
pip install -r requirements_gpu.txt
pip install datasets          # HuggingFace dataset loader
pip install faiss-cpu         # required for PRS evaluator embedding
```

### 3. Set HuggingFace token (Llama is gated)

```bash
export HF_TOKEN=hf_your_token_here
# or add "hf_token": "hf_..." to config.json
```

### 4. GPU

KV computation and LoRA training require a GPU. On CPU they will run but
take hours. Minimum: 16 GB VRAM (A10G / T4 / RTX 3090).

## Quickstart

```bash
# From repo root
bash examples/usecase1_customer_support/run_pipeline.sh
```

Or step by step:

```bash
# 0. Download dataset
python examples/usecase1_customer_support/setup.py

# 1. Index into Qdrant
python smartqdrant.py index \
  --config examples/usecase1_customer_support/config.json \
  --source examples/usecase1_customer_support/data/corpus.jsonl

# 2. Compute KV tensors
python -m pipeline.kv_indexer \
  --config examples/usecase1_customer_support/config.json \
  compute-kv

# 3. Fine-tune LoRA
python -m pipeline.lora_trainer \
  --config examples/usecase1_customer_support/config.json \
  --faqs examples/usecase1_customer_support/faqs.json

# 4. Recompute KV with updated weights
python -m pipeline.kv_indexer \
  --config examples/usecase1_customer_support/config.json \
  compute-kv

# 5. Evaluate
python -m pipeline.prs_evaluator \
  --config examples/usecase1_customer_support/config.json \
  --faqs examples/usecase1_customer_support/faqs.json \
  --sample 30

# 6. Ask a question
python ask.py \
  --config examples/usecase1_customer_support/config.json \
  "How do I cancel my subscription?"
```

## Configuration

Key fields in `config.json`:

| Field | Value | Notes |
|-------|-------|-------|
| `vector_store` | `"qdrant"` | Uses local Qdrant on Docker |
| `qdrant_host` | `"localhost"` | Change to remote host for cloud deployment |
| `collection` | `"customer-support"` | Qdrant collection name |
| `loader` | `"jsonl"` | JSONL format with `text` field |
| `embed_model` | `BAAI/bge-small-en-v1.5` | 384-dim FastEmbed model |
| `llm_model` | `meta-llama/Llama-3.2-3B-Instruct` | Base LLM for KV and LoRA |
| `lora_epochs` | `3` | Increase to 5–10 for better PRS |
| `prs_threshold` | `0.75` | Phase 2→3 gate |

## Expected Results

After the full pipeline on a GPU:

| Metric | Expected range |
|--------|---------------|
| Indexed chunks | ~2 000 |
| KV coverage | 100 % |
| PRS after 1 round | 0.60–0.75 |
| PRS after 3 rounds | 0.75–0.90 |
| Phase 3 activation | After PRS ≥ 0.75 |

## Monitoring

```bash
python -m pipeline.monitoring_dashboard \
  --config examples/usecase1_customer_support/config.json
# Open http://localhost:8081
```

## Why Qdrant for Customer Support?

- **Production-ready**: Supports cloud deployment (Qdrant Cloud), horizontal sharding, and replication
- **Efficient KV backfill**: `compute-kv` uses `IsNullCondition` payload filters to find only un-computed chunks — critical when adding new support tickets incrementally
- **Payload filtering**: Future work can filter by `intent` or `category` at query time
- **Dashboard**: Full monitoring with access-count tracking per chunk
