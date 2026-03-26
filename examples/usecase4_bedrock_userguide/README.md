# Use Case 4: Amazon Bedrock User Guide with Qdrant

This example demonstrates KVForge on a real technical documentation corpus — the
**Amazon Bedrock User Guide** (PDF, ~500 pages) — using **Qdrant** as the vector store
and a higher-dimension embedding model (`mxbai-embed-large-v1`, 1024-dim) for better
retrieval accuracy on dense technical text.

## Dataset

**[Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/)**

| Property | Value |
|----------|-------|
| Source | Amazon AWS official documentation (PDF) |
| Domain | Cloud AI services — foundation models, agents, fine-tuning, RAG |
| Size | ~500 pages, ~1 200–1 500 chunks at chunk_size=600 |
| Format | PDF (text-extracted, no OCR needed) |
| FAQs included | 50 pre-generated Q&A pairs (`faqs.json`) |
| Why this dataset | Dense technical documentation with precise terminology — ideal for testing parametric recall on specialist knowledge |

## Architecture

```
data/amazon-bedrock-user-guide.pdf
        │
        ▼
  kvforge.py index (PDF loader) ──────────► Qdrant (localhost:6333)
        │                                               │
        ▼                                               │
  pipeline/kv_indexer.py compute-kv ◄──── read chunks ─┘
        │                                               │
        ▼ KV tensors                                    │ stored back
  ──────────────────────────────────────────────────────┘
        │
        ▼
  pipeline/lora_trainer.py ──── faqs.json ──────────► LoRA checkpoint
        │
        ▼
  pipeline/kv_indexer.py compute-kv (re-run with updated weights)
        │
        ▼
  pipeline/prs_evaluator.py ──────────────────────► PRS score (target ≥ 0.75)
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
```

### 3. Set HuggingFace token (Llama is gated)

```bash
export HF_TOKEN=hf_your_token_here
# or add "hf_token": "hf_..." to config.json
```

### 4. GPU

KV computation and LoRA training require a GPU. Minimum: 16 GB VRAM (A10G / T4 / RTX 3090).
The `mxbai-embed-large-v1` embedding model (1024-dim) requires more memory than the default
`bge-small` model — ensure at least 4 GB VRAM free for embedding.

## Quickstart

```bash
# From repo root
bash examples/usecase4_bedrock_userguide/run_pipeline.sh
```

Or step by step:

```bash
CONFIG=examples/usecase4_bedrock_userguide/config.json
FAQS=examples/usecase4_bedrock_userguide/faqs.json
PDF=examples/usecase4_bedrock_userguide/data/amazon-bedrock-user-guide.pdf

# 1. Index the PDF
python kvforge.py index --config "$CONFIG" --source "$PDF"

# 2. Compute KV tensors
python -m pipeline.kv_indexer --config "$CONFIG" compute-kv

# 3. Fine-tune LoRA (uses pre-generated faqs.json)
python -m pipeline.lora_trainer --config "$CONFIG" --faqs "$FAQS"

# 4. Recompute KV with updated weights
python -m pipeline.kv_indexer --config "$CONFIG" compute-kv

# 5. Evaluate PRS
python -m pipeline.prs_evaluator --config "$CONFIG" --faqs "$FAQS" --sample 30

# 6. Ask a question
python ask.py --config "$CONFIG" "What is Amazon Bedrock?"
```

## Configuration

Key fields in `config.json`:

| Field | Value | Notes |
|-------|-------|-------|
| `vector_store` | `"qdrant"` | Uses local Qdrant on Docker |
| `collection` | `"bedrock-user-guide"` | Qdrant collection name |
| `loader` | `"pdf"` | PDF text extraction via pypdf |
| `embed_model` | `mixedbread-ai/mxbai-embed-large-v1` | 1024-dim — higher accuracy on technical text |
| `vector_dim` | `1024` | Must match embed_model output dimension |
| `llm_model` | `meta-llama/Llama-3.2-3B-Instruct` | Base LLM for KV and LoRA |
| `lora_epochs` | `3` | Increase to 5 for better PRS on dense technical content |
| `prs_threshold` | `0.75` | Phase 2→3 gate |
| `dashboard_port` | `8084` | Monitoring dashboard (UC1=8081, UC2=8082, UC3=8083) |

## Expected Results

After the full pipeline on a GPU:

| Metric | Expected range |
|--------|---------------|
| Indexed chunks | ~1 200–1 500 |
| KV coverage | 100 % |
| PRS after 1 round | 0.55–0.70 (dense technical content is harder) |
| PRS after 3 rounds | 0.70–0.85 |
| Phase 3 activation | After PRS ≥ 0.75 |

> **Note:** Technical documentation with precise terminology (API names, service limits,
> ARN formats) is harder for a 3B model to memorise than conversational Q&A. If PRS
> plateaus below 0.75, try increasing `lora_epochs` to 5 or using a larger base model.

## Monitoring

```bash
python -m pipeline.monitoring_dashboard \
  --config examples/usecase4_bedrock_userguide/config.json
# Open http://localhost:8084
```

## Example Queries

```bash
CONFIG=examples/usecase4_bedrock_userguide/config.json

python ask.py --config "$CONFIG" "What is Amazon Bedrock?"
python ask.py --config "$CONFIG" "Which foundation models are available in Bedrock?"
python ask.py --config "$CONFIG" "How do I fine-tune a model in Amazon Bedrock?"
python ask.py --config "$CONFIG" "What are Bedrock Agents and how do they work?"
python ask.py --config "$CONFIG" "How does Amazon Bedrock handle data privacy?"
python ask.py --config "$CONFIG" "What is the difference between on-demand and provisioned throughput?"
```

## A/B Evaluation (KVForge vs Gemini)

`eval_ab.py` runs every question in `faqs.json` through the dashboard's `/api/query`
endpoint, computes metrics against the ground-truth answers, and prints a comparison report.

**Metrics computed:**
- `semantic_sim` — cosine similarity between answer and ground-truth embeddings (fastembed, CPU)
- `token_f1` — token-level F1 between predicted and ground-truth answer
- `latency_ms` — wall-clock latency reported by the dashboard
- `ragas_similarity` / `ragas_correctness` — optional, requires `--ragas` flag

**Prerequisites:** Start the dashboard first:
```bash
python -m pipeline.monitoring_dashboard \
  --config examples/usecase4_bedrock_userguide/config.json
```

**Run:**
```bash
# Basic — against local dashboard on port 8084
python3 examples/usecase4_bedrock_userguide/eval_ab.py \
    --faq examples/usecase4_bedrock_userguide/faqs.json

# Save results to JSON
python3 examples/usecase4_bedrock_userguide/eval_ab.py \
    --faq examples/usecase4_bedrock_userguide/faqs.json \
    --out ab_eval_results.json

# Against EC2 dashboard + RAGAS evaluation with Gemini judge
python3 examples/usecase4_bedrock_userguide/eval_ab.py \
    --faq examples/usecase4_bedrock_userguide/faqs.json \
    --dashboard http://YOUR_EC2_HOST:8084 \
    --ragas --gemini-key YOUR_KEY
```

## Why Qdrant for Technical Documentation?

- **1024-dim vector support**: Qdrant handles large embedding dimensions efficiently — ChromaDB and FAISS work too, but Qdrant's HNSW index performs best at high dimensions
- **Payload filtering**: Filter by page number or section for targeted retrieval
- **Incremental updates**: When the Bedrock User Guide is updated, re-index only changed pages with `compute-kv --source-file`
- **Production-ready**: Deploy to Qdrant Cloud for a persistent, scalable knowledge base

## Difference from Other Use Cases

| | UC1 Customer Support | UC4 Bedrock User Guide |
|---|---|---|
| **Source format** | JSONL | PDF |
| **Vector store** | Qdrant | Qdrant |
| **Embed model** | bge-small (384-dim) | mxbai-embed-large (1024-dim) |
| **Domain** | Conversational Q&A | Technical documentation |
| **FAQs** | Generated from corpus | Pre-generated (50 pairs included) |
| **PRS difficulty** | Easier (conversational) | Harder (dense terminology) |
