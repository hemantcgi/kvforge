# Use Case 3: General Knowledge Q&A with FAISS

This example demonstrates SmartQdrant on SQuAD v2 (Stanford Question Answering
Dataset), using **FAISS** as the vector store — completely offline, no server,
no Docker. Everything runs in a single process.

## Why this combination is interesting

- **Fully offline**: FAISS stores the index and metadata as local files
  (`.faiss/squad-qa/squad-qa.index` and `.faiss/squad-qa/squad-qa.meta.pkl`).
  No network calls after dataset download. Suitable for air-gapped environments,
  CI pipelines, and laptop demos.
- **Unanswerable questions**: SQuAD v2 introduces ~50 000 unanswerable questions.
  The confidence gate (Phase 3) must correctly abstain on these — they appear
  in the corpus but have no gold answer. This stresses the calibration component
  of PRS more than the other use cases.
- **General knowledge**: Llama 3.2-3B already knows much of this content
  parametrically, so PRS starts higher than the biomedical use case. Fine-tuning
  shows diminishing returns, demonstrating the system's ability to decide when
  Phase 3 is safe to activate.

## Dataset

**[SQuAD v2](https://huggingface.co/datasets/rajpurkar/squad_v2)**

| Property | Value |
|----------|-------|
| Source | HuggingFace — `rajpurkar/squad_v2` |
| Domain | Wikipedia (general knowledge — history, science, culture, geography) |
| Size used | 2 000 unique passage contexts as corpus; 50 answerable pairs as FAQs |
| Format | Wikipedia passages + reading comprehension Q&A |
| Why this dataset | Gold-standard NLP benchmark; large, diverse, includes unanswerable questions |

## Architecture

```
HuggingFace SQuAD v2
        │
        ▼
  setup.py (deduplicate passages, extract answerable Q&A)
        │
        ▼ JSONL corpus (unique Wikipedia passages)
  smartqdrant.py index ──────────────► FAISS (.faiss/squad-qa/)
        │                               (IndexFlatIP, L2-normalised)
        ▼
  kv_indexer.py compute-kv ◄─── scroll (in-memory scan)
        │
        ▼ KV tensors stored in .meta.pkl payload
  lora_trainer.py ──── faqs.json ──► LoRA checkpoint
        │
        ▼
  kv_indexer.py compute-kv (re-run)
        │
        ▼
  prs_evaluator.py ──────────────────► PRS score
```

## Prerequisites

### 1. Install dependencies (no Docker or cloud account needed)

```bash
pip install -r requirements_gpu.txt
pip install datasets faiss-cpu
```

For GPU-accelerated FAISS (optional):
```bash
pip install faiss-gpu   # CUDA required
```

### 2. Set HuggingFace token

```bash
export HF_TOKEN=hf_your_token_here
```

### 3. GPU

Required for KV computation and LoRA training. FAISS itself runs on CPU.

## Quickstart

```bash
# From repo root
bash examples/usecase3_squad/run_pipeline.sh
```

Or step by step:

```bash
# 0. Download dataset
python examples/usecase3_squad/setup.py

# 1. Index into FAISS
python smartqdrant.py index \
  --config examples/usecase3_squad/config.json \
  --source examples/usecase3_squad/data/corpus.jsonl

# 2. Compute KV tensors
python -m pipeline.kv_indexer \
  --config examples/usecase3_squad/config.json \
  compute-kv

# 3. Fine-tune LoRA
python -m pipeline.lora_trainer \
  --config examples/usecase3_squad/config.json \
  --faqs examples/usecase3_squad/faqs.json

# 4. Recompute KV
python -m pipeline.kv_indexer \
  --config examples/usecase3_squad/config.json \
  compute-kv

# 5. Evaluate
python -m pipeline.prs_evaluator \
  --config examples/usecase3_squad/config.json \
  --faqs examples/usecase3_squad/faqs.json \
  --sample 30

# 6. Ask a question
python ask.py \
  --config examples/usecase3_squad/config.json \
  "Who invented the telephone?"
```

## Configuration

| Field | Value | Notes |
|-------|-------|-------|
| `vector_store` | `"faiss"` | In-process, fully offline |
| `faiss_persist_dir` | `.faiss/squad-qa` | Index file + metadata PKL stored here |
| `collection` | `"squad-qa"` | Logical name for the index |
| `loader` | `"jsonl"` | JSONL with `text` field |
| `embed_model` | `BAAI/bge-small-en-v1.5` | 384-dim, CPU-friendly |

## Expected Results

| Metric | Expected range |
|--------|---------------|
| Indexed chunks | ~2 000 passages |
| KV coverage | 100 % |
| PRS (base, no training) | 0.55–0.70 (general knowledge — Llama already knows much of this) |
| PRS after 3 LoRA rounds | 0.70–0.85 |
| Phase 3 activation | Likely after round 1–2 |

This is the highest starting PRS of the three use cases, reflecting the
model's strong Wikipedia pre-training. The SQuAD confidence-gate results are
particularly interesting: compare Phase 2 (KV injection) vs Phase 3
(parametric) answer quality on the same questions.

## FAISS internals

The `FAISSStore` uses `IndexFlatIP` (exact inner-product search on L2-normalised
vectors = cosine similarity). Files written:

```
examples/usecase3_squad/.faiss/squad-qa/
├── squad-qa.index     # FAISS binary index
└── squad-qa.meta.pkl  # {id_map: [...], payloads: {...}}
```

Inspect the index:

```python
import faiss, pickle

index = faiss.read_index("examples/usecase3_squad/.faiss/squad-qa/squad-qa.index")
print(f"Vectors stored: {index.ntotal}, dimension: {index.d}")

with open("examples/usecase3_squad/.faiss/squad-qa/squad-qa.meta.pkl", "rb") as f:
    meta = pickle.load(f)
print(f"IDs: {meta['id_map'][:5]}")
print(f"Sample payload: {list(meta['payloads'].values())[0]}")
```

## Limitations of FAISS vs Qdrant

| Feature | FAISS | Qdrant |
|---------|-------|--------|
| Server required | No | Yes (Docker) |
| Concurrent writers | No (single process) | Yes |
| Incremental KV backfill filter | Full scan | Index filter (fast) |
| Production scale | Up to ~5M vectors | Unlimited (sharding) |
| Best for | Offline batch, CI, laptops | Production APIs |

## Monitoring

```bash
python -m pipeline.monitoring_dashboard \
  --config examples/usecase3_squad/config.json
# Open http://localhost:8083
```
