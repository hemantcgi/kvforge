# Use Case 2: Biomedical Q&A with ChromaDB

This example demonstrates SmartQdrant on the PubMedQA biomedical benchmark,
using **ChromaDB** as the vector store — no Docker, no external services.
ChromaDB runs in-process and persists data as SQLite + parquet files on disk.

## Why this combination is interesting

- **Domain mismatch**: Llama 3.2-3B has limited biomedical knowledge. LoRA
  fine-tuning on PubMedQA Q&A pairs materially improves PRS because the base
  model cannot answer biomedical questions parametrically — retrieval is
  essential until fine-tuning brings PRS above the threshold.
- **No infrastructure**: ChromaDB needs no Docker, no network config, no cloud
  account. The entire pipeline runs on a single machine.

## Dataset

**[PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA)**

| Property | Value |
|----------|-------|
| Source | HuggingFace — `qiaojin/PubMedQA` (pqa_labeled split) |
| Domain | Biomedical research (PubMed abstracts) |
| Size used | 1 000 abstracts → ~3 000–5 000 paragraph chunks |
| Format | Abstract contexts + yes/no/maybe final decision + long answer |
| Why this dataset | Real peer-reviewed Q&A with gold answers; tests domain adaptation |

## Architecture

```
HuggingFace PubMedQA
        │
        ▼
  setup.py (download + flatten paragraphs)
        │
        ▼ JSONL corpus
  smartqdrant.py index ──────────────► ChromaDB (.chroma/pubmedqa/)
        │                                        │
        ▼                                        │
  kv_indexer.py compute-kv ◄─── scroll all ─────┘
        │                           chunks
        ▼ KV tensors stored in ChromaDB payload
  lora_trainer.py ──── faqs.json ──► LoRA checkpoint
        │
        ▼
  kv_indexer.py compute-kv (re-run)
        │
        ▼
  prs_evaluator.py ──────────────────► PRS score
        │
        ▼
  ask.py / monitoring_dashboard.py
```

> **ChromaDB scroll note**: `compute-kv` uses full collection scan (no
> server-side filtering). For 1 000 abstracts (~4 000 chunks) this takes
> ~30 seconds on CPU. Use Qdrant for corpora > 50 000 chunks.

## Prerequisites

### 1. Install dependencies (no Docker needed)

```bash
pip install -r requirements_gpu.txt
pip install datasets chromadb
```

### 2. Set HuggingFace token

```bash
export HF_TOKEN=hf_your_token_here
```

### 3. GPU

Strongly recommended: KV computation for biomedical text (longer sequences)
is significantly slower on CPU. Minimum: 16 GB VRAM.

## Quickstart

```bash
# From repo root
bash examples/usecase2_pubmedqa/run_pipeline.sh
```

Or step by step:

```bash
# 0. Download dataset
python examples/usecase2_pubmedqa/setup.py

# 1. Index into ChromaDB
python smartqdrant.py index \
  --config examples/usecase2_pubmedqa/config.json \
  --source examples/usecase2_pubmedqa/data/corpus.jsonl

# 2. Compute KV tensors
python -m pipeline.kv_indexer \
  --config examples/usecase2_pubmedqa/config.json \
  compute-kv

# 3. Fine-tune LoRA
python -m pipeline.lora_trainer \
  --config examples/usecase2_pubmedqa/config.json \
  --faqs examples/usecase2_pubmedqa/faqs.json

# 4. Recompute KV
python -m pipeline.kv_indexer \
  --config examples/usecase2_pubmedqa/config.json \
  compute-kv

# 5. Evaluate
python -m pipeline.prs_evaluator \
  --config examples/usecase2_pubmedqa/config.json \
  --faqs examples/usecase2_pubmedqa/faqs.json \
  --sample 30

# 6. Ask a question
python ask.py \
  --config examples/usecase2_pubmedqa/config.json \
  "Does aspirin prevent colorectal cancer?"
```

## Configuration

| Field | Value | Notes |
|-------|-------|-------|
| `vector_store` | `"chroma"` | In-process, no server required |
| `chroma_persist_dir` | `.chroma/pubmedqa` | SQLite + parquet persisted here |
| `collection` | `"pubmedqa"` | ChromaDB collection name |
| `loader` | `"jsonl"` | JSONL with `text` field |
| `embed_model` | `BAAI/bge-small-en-v1.5` | Domain-agnostic; swap for `pritamdeka/S-PubMedBert-MS-MARCO` for better biomedical retrieval |

## Expected Results

| Metric | Expected range |
|--------|---------------|
| Indexed chunks | ~3 000–5 000 |
| KV coverage | 100 % |
| PRS (base, no training) | 0.30–0.50 (biomedical domain is hard for base Llama) |
| PRS after 3 LoRA rounds | 0.60–0.80 |

The large gap between base PRS and post-training PRS illustrates exactly
why LoRA fine-tuning matters most for specialized domains.

## Verifying ChromaDB data

```python
import chromadb
client = chromadb.PersistentClient(path="examples/usecase2_pubmedqa/.chroma/pubmedqa")
col = client.get_collection("pubmedqa")
print(f"Documents: {col.count()}")
print(col.peek(3))
```

## Monitoring

```bash
python -m pipeline.monitoring_dashboard \
  --config examples/usecase2_pubmedqa/config.json
# Open http://localhost:8082
```
