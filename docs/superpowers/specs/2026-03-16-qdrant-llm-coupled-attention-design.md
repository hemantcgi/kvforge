# Qdrant-Coupled Attention System — Design Spec

**Date:** 2026-03-16
**Status:** Approved for implementation
**Authors:** Hemant (product), Claude (design)

---

## 1. Overview

This system evolves the existing Bedrock RAG pipeline from a static
retrieval-augmented generation (RAG) system into a continuously self-improving,
domain-adapting inference engine. It achieves two compounding goals:

1. **Faster inference** — pre-computed KV (key-value) tensors stored in Qdrant
   are injected directly into the LLM's attention layers, eliminating the cost
   of re-encoding retrieved context on every query.

2. **Domain adaptation** — LoRA fine-tuning of the LLM's attention projection
   matrices (W_Q, W_K, W_V) is triggered automatically after every new data
   source is indexed. Over successive rounds, the model internalises domain
   knowledge and increasingly answers queries from its own weights.

The system is built in six sub-projects (SP1–SP6), each delivering standalone
value and enabling the next.

---

## 2. Goals

| # | Goal |
|---|------|
| G1 | Every new document indexed into Qdrant also triggers a LoRA weight update |
| G2 | Pre-computed KV tensors stored per chunk enable injection at inference time |
| G3 | Stale KV pairs fall back to text-in-context; background worker heals them |
| G4 | Access analytics (hot/warm/cold/frozen) inform training priority and KV scheduling |
| G5 | Parametric Readiness Score (PRS) measures model self-sufficiency over time |
| G6 | Confidence gate activates when PRS ≥ 0.80 for two consecutive LoRA rounds |
| G7 | Monitoring dashboard shows PRS trend, training status, and access analytics |

---

## 3. Non-Goals (v1)

- Multi-GPU or distributed LoRA training
- Per-token KV storage (only mean-pooled KV)
- Support for models other than Llama 3.2 3B
- Real-time streaming KV injection during generation
- Elastic Weight Consolidation (EWC) or other continual-learning strategies
- Retraining from base model on full corpus each round

---

## 4. Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Vector DB | Qdrant 1.17 (local Docker) | Extended payload schema |
| Embedding | fastembed + BAAI/bge-small-en-v1.5 | Unchanged from current pipeline |
| LLM (inference + KV) | Llama 3.2 3B via HuggingFace transformers | Replaces Ollama for main inference |
| LoRA fine-tuning | HuggingFace PEFT | rank=16, α=32, target: q/k/v_proj |
| Quantisation | bitsandbytes (4-bit NF4) | For training; fp16 for inference |
| Dashboard | FastAPI + single-page HTML/JS | localhost:8080 |
| Python env | venv (Python 3.13, existing) | fastembed already installed |
| Hardware | AWS EC2 g5.xlarge (A10G 24GB VRAM, 16GB RAM) | |
| OS | Ubuntu 22.04 (EC2 AMI) | |

---

## 5. System Architecture

### 5.1 Component Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                        index_and_train.py                           │
│                    (orchestrator — one call per new source)         │
└────────┬────────────────────┬───────────────────────────────────────┘
         │                    │
         ▼                    ▼
┌────────────────┐   ┌────────────────────────────────────────────────┐
│  SP1           │   │  SP2                                           │
│  kv_indexer.py │   │  lora_trainer.py + replay_buffer.py            │
│                │   │                                                │
│  chunk + embed │   │  LoRA fine-tune q/k/v_proj                     │
│  → Qdrant      │   │  replay buffer (weighted by tier)              │
│  → LLM forward │   │  lora_version += 1                             │
│    → kv_cache  │   │  → compute KV for new chunks (model_loader.py) │
│  kv_version=N  │   │                                                │
└────────┬───────┘   └───────────────────┬────────────────────────────┘
         │                               │
         │         ┌─────────────────────┘
         │         │
         ▼         ▼
┌──────────────────────────────────────────────────────┐
│                   Qdrant Collection                   │
│  vector: [384-dim bge-small]                         │
│  payload:                                            │
│    text, page                    (existing)          │
│    kv_cache: [57344 fp16 floats] (SP1)               │
│    kv_version: null | int        (SP1/SP2)           │
│    access_count: int             (SP5)               │
│    last_accessed_ts: int|null    (SP5)               │
│    avg_retrieval_rank: float     (SP5)               │
│    parametric_hit_count: int     (SP5)               │
│    tier: hot|warm|cold|frozen    (SP5)               │
└──────────────────────────┬───────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
┌───────────────────────┐     ┌─────────────────────────────────┐
│  SP3                  │     │  SP4                            │
│  kv_inference.py      │     │  confidence_gate.py             │
│  + kv_background.py   │     │  + prs_evaluator.py             │
│                       │     │                                 │
│  query → search       │     │  (active when PRS ≥ 0.80)       │
│  → version check      │     │  try model directly first       │
│  → KV inject (fresh)  │     │  → score entropy + hedging      │
│  → text fallback +    │     │  → gate: direct or retrieve     │
│    bg KV update       │     │                                 │
└───────────────────────┘     └─────────────────────────────────┘
            │                             │
            └──────────────┬──────────────┘
                           │
                           ▼
            ┌──────────────────────────┐
            │  SP5                     │
            │  access_tracker.py       │
            │                          │
            │  in-memory counter dict  │
            │  async batch flush       │
            │  tier recomputation      │
            │  weekly access_report    │
            └──────────────────────────┘
                           │
                           ▼
            ┌──────────────────────────┐
            │  SP6                     │
            │  monitoring_dashboard.py │
            │  FastAPI + HTML/JS       │
            │  localhost:8080          │
            └──────────────────────────┘
```

### 5.2 Qdrant Payload Schema (full)

```json
{
  "vector": [384 floats],
  "payload": {
    "text":                 "…chunk text…",
    "page":                 42,
    "source_file":          "Amazon Bedrock Dataset.pdf",
    "indexed_at":           1773723034,

    "kv_cache":             [57344 fp16 floats as base64 string],
    "kv_version":           null,

    "access_count":         0,
    "last_accessed_ts":     null,
    "avg_retrieval_rank":   null,
    "parametric_hit_count": 0,
    "tier":                 "frozen"
  }
}
```

**KV cache layout** (Llama 3.2 3B, mean-pooled over token dimension):
- Dimensions: `[num_layers=28, 2, num_kv_heads=8, head_dim=128]` = 57,344 floats
- dtype: float16, serialised as base64 string in payload
- Size per chunk: ~112 KB
- Total for 2,587 chunks: ~280 MB

### 5.3 Shared Files

```
model_loader.py     — load Llama 3.2 3B + active LoRA adapter; singleton pattern
kv_utils.py         — mean_pool_kv(), serialize_kv(), deserialize_kv(), fp16 conversions
version.json        — {current_lora_version, checkpoint_path, prs_history[], phase}
```

### 5.4 Phase State Machine

```
Phase 1 (RAG only)
    │  condition: before SP1 is deployed
    │  PRS = 0 (not computed yet)
    ▼
Phase 2 (RAG + KV injection)
    │  condition: SP1 + SP3 deployed; at least one LoRA round complete
    │  KV tensors available; version checking active
    ▼
Phase 3 (Confidence-gated)
    │  condition: PRS ≥ 0.80 for 2 consecutive LoRA rounds
    │  SP4 activates; confidence_gate.py intercepts queries
    ▼
Phase 3b (Converged)
       condition: parametric_answer_rate plateaus (< 1% change over
                  2 consecutive weeks); model has fully internalised corpus
                  → Qdrant retrieval becomes a fallback only
```

---

## 6. Flow Diagrams

### 6.1 First-Time Document Indexing
*(collection does not yet exist)*

```
user: python3 index_and_train.py new_doc.pdf --first-run
         │
         ▼
[index_and_train.py]
  ├─ check Qdrant: collection "bedrock-user-guide" exists? → NO
  ├─ create collection (384-dim cosine, extended payload schema)
  │
  ├─ ── SP1: kv_indexer.py ──────────────────────────────────────
  │   ├─ read_pdf() → pages
  │   ├─ chunk_pages() → 2587 chunks
  │   ├─ fastembed.embed() → 384-dim vectors
  │   ├─ model_loader.load() → Llama 3.2 3B (fp16, A10G)
  │   ├─ for each chunk:
  │   │   ├─ LLM tokenize(text)
  │   │   ├─ LLM forward_pass() → extract K,V per layer
  │   │   ├─ kv_utils.mean_pool_kv() → [28,2,8,128]
  │   │   └─ kv_utils.serialize_kv() → base64 string
  │   └─ qdrant.upsert(vector, payload{kv_cache, kv_version=null})
  │                                ↑ not yet versioned (training not done)
  │
  ├─ ── SP2: lora_trainer.py ────────────────────────────────────
  │   ├─ no replay buffer yet (first run)
  │   ├─ dataset = all 2587 chunks (next-token prediction)
  │   ├─ PEFT LoraConfig(r=16, α=32, target=[q_proj,k_proj,v_proj])
  │   ├─ train ~15–30 min on A10G
  │   ├─ save adapter → lora_checkpoints/v1/
  │   ├─ version.json → {current_lora_version: 1, checkpoint_path: …}
  │   │
  │   └─ compute KV with updated weights:
  │       ├─ model_loader.reload(lora_v1)
  │       ├─ for each chunk: re-run forward pass → new kv_cache
  │       └─ qdrant.update_payload(kv_version=1)
  │
  ├─ ── SP2: prs_evaluator.py ───────────────────────────────────
  │   ├─ sample 50 FAQs from bedrock_50_faqs.json
  │   ├─ run parametric mode (no retrieval) → answers_param
  │   ├─ run RAG mode (with retrieval) → answers_rag
  │   ├─ compute PRS = 0.5*accuracy_ratio + 0.3*calibration + 0.2*consistency
  │   └─ version.json → {prs_history: [{round:1, prs:0.42}]}
  │
  └─ DONE
      Qdrant: 2587 chunks searchable, kv_version=1
      Model: lora_v1 active
      PRS: 0.42 (baseline)
      Phase: 2 (KV injection available)
```

### 6.2 Subsequent Document Indexing
*(collection exists, previous LoRA rounds complete)*

```
user: python3 index_and_train.py ec2_guide.pdf
         │
         ▼
[index_and_train.py]
  ├─ check Qdrant: collection exists? → YES (lora_version=6, prs=0.74)
  │
  ├─ ── SP1: kv_indexer.py (new chunks only) ────────────────────
  │   ├─ chunk ec2_guide.pdf → 384 new chunks
  │   ├─ fastembed.embed() → 384-dim vectors
  │   ├─ model_loader.load(lora_v6) → existing weights
  │   ├─ for each new chunk: forward pass → kv_cache
  │   └─ qdrant.upsert(new chunks, kv_version=null)
  │                              ↑ null: training not done yet
  │                              → IMMEDIATELY searchable by vector
  │
  ├─ ── SP2: lora_trainer.py ────────────────────────────────────
  │   ├─ new_chunks = 384 chunks from ec2_guide.pdf
  │   ├─ replay_buffer.sample(n=96, weight_by_tier=True)
  │   │     hot chunks: 40%, warm: 40%, cold: 15%, frozen: 5%
  │   ├─ dataset = new_chunks + replay_sample (480 total)
  │   ├─ fine-tune from lora_v6 → lora_v7 (~15 min)
  │   ├─ save adapter → lora_checkpoints/v7/
  │   ├─ version.json → {current_lora_version: 7}
  │   │
  │   └─ compute KV for NEW chunks only with lora_v7:
  │       ├─ model_loader.reload(lora_v7)
  │       ├─ for each of 384 new chunks: forward pass → kv_cache
  │       └─ qdrant.update_payload(new chunks, kv_version=7)
  │         (old 2587 chunks remain at kv_version=6 → lazily updated)
  │
  ├─ ── prs_evaluator.py ────────────────────────────────────────
  │   ├─ sample 50 FAQs, run dual-mode eval
  │   ├─ PRS = 0.77 (up from 0.74)
  │   └─ version.json → {prs_history: [..., {round:7, prs:0.77}]}
  │       PRS < 0.80 → Phase 3 NOT yet active
  │
  └─ DONE
      Qdrant: 2971 chunks (2587 old + 384 new)
      Old chunks: kv_version=6 (one round stale, lazily updated)
      New chunks: kv_version=7 (fresh)
      Model: lora_v7 active
```

### 6.3 Query Received — Phase 2, All KV Fresh

```
user: "What is Amazon Bedrock?"
         │
         ▼
[kv_inference.py]
  ├─ read version.json → current_lora_version = 7, phase = 2
  ├─ fastembed.embed(query) → 384-dim vector
  ├─ qdrant.query_points(top_k=5) → 5 chunks
  │     chunk A: kv_version=7 ✓
  │     chunk B: kv_version=7 ✓
  │     chunk C: kv_version=7 ✓
  │     chunk D: kv_version=7 ✓
  │     chunk E: kv_version=7 ✓
  │
  ├─ ALL chunks fresh → KV INJECTION PATH
  │   ├─ kv_utils.deserialize_kv(chunk.kv_cache) × 5
  │   │     each: np.ndarray [28, 2, 8, 128] fp16
  │   ├─ kv_utils.stack_past_key_values(list_of_5_kv_arrays)
  │   │     → HuggingFace past_key_values format:
  │   │       tuple of 28 tuples, each (K, V) with shape [1, 8, 5, 128]
  │   │       (5 = one mean-pooled token per chunk, concatenated along seq dim)
  │   │     (see Q1 in §10 for stacking strategy rationale)
  │   ├─ model.generate(prompt_only, past_key_values=retrieved_kv)
  │   │     model skips re-encoding chunk text entirely ⚡
  │   └─ return answer with citations
  │
  ├─ SP5: access_tracker.record(chunk_ids=[A,B,C,D,E], ranks=[1,2,3,4,5])
  │       in-memory update, no Qdrant write yet
  │
  └─ latency: ~0.9s total (vs ~3.5s text-in-context)
```

### 6.4 Query Received — Phase 2, Some KV Stale

```
user: "What EC2 instance types work best with Bedrock?"
         │  (new topic — some retrieved chunks are from ec2_guide.pdf)
         ▼
[kv_inference.py]
  ├─ fastembed.embed(query) → search → 5 chunks
  │     chunk A: kv_version=7 ✓  (old doc, already updated)
  │     chunk B: kv_version=7 ✓
  │     chunk C: kv_version=null ✗  (ec2_guide.pdf, training running)
  │     chunk D: kv_version=7 ✓
  │     chunk E: kv_version=null ✗
  │
  ├─ STALE CHUNKS DETECTED → TEXT-IN-CONTEXT FALLBACK
  │   ├─ build prompt with all 5 chunk texts (same as ollama_answer.py)
  │   ├─ model.generate(prompt_with_context_text)
  │   └─ return answer — same quality as current RAG, no degradation
  │
  ├─ SP5: access_tracker.record(chunk_ids=[A,B,C,D,E], ranks=[1..5])
  │
  └─ [kv_background.py — non-blocking, separate thread]
      ├─ queue.put(chunk_ids=[C, E])
      │     background worker picks up immediately
      ├─ model_loader.load(lora_v7)
      ├─ for chunk_id in [C, E]:
      │   ├─ fetch chunk text from Qdrant
      │   ├─ LLM forward_pass(text) → kv_cache
      │   └─ qdrant.update_payload(chunk_id, {kv_cache:…, kv_version:7})
      └─ next query hitting C or E gets KV injection ✓
```

### 6.5 Query Received — Phase 3, High Confidence (Direct Answer)

```
[Phase 3 active: PRS crossed 0.80 for 2 consecutive rounds]

user: "What is Amazon Bedrock?"
         │
         ▼
[confidence_gate.py]
  ├─ step 1: model.generate(query, max_new_tokens=20, greedy)
  │           → draft: "Amazon Bedrock is a fully managed service…"
  │
  ├─ step 2: score 3 signals
  │   ├─ token_entropy(draft) = 0.18  → LOW (concept, model certain)
  │   ├─ hedging_score(draft) = 0.0   → no "I think", "approximately" etc.
  │   └─ query_similarity_to_known_good = 0.91
  │         (cosine sim to queries that scored well in PRS offline eval)
  │
  ├─ P(no_retrieval) = 0.91 ≥ threshold 0.75
  │
  ├─ DIRECT ANSWER PATH ⚡
  │   ├─ model.generate(query, full response)
  │   └─ return answer (no Qdrant query at all)
  │
  ├─ SP5: access_tracker.record_parametric_hit(
  │         query=query,
  │         would_have_retrieved=[A,B,C,D,E]  ← top-5 by similarity
  │       )
  │       parametric_hit_count++ for those chunks
  │
  └─ latency: ~0.4s (no vector search, no KV deserialise)
```

### 6.6 Query Received — Phase 3, Low Confidence (Falls Back to Retrieval)

```
user: "What is the default on-demand quota for Anthropic Claude 3.5 Sonnet?"
         │
         ▼
[confidence_gate.py]
  ├─ step 1: draft answer
  │           → "The default quota is… approximately 50… or maybe 100…"
  │
  ├─ step 2: score signals
  │   ├─ token_entropy(draft) = 0.74  → HIGH (specific number, uncertain)
  │   ├─ hedging_score(draft) = 0.6   → "approximately", "or maybe"
  │   └─ query_similarity_to_known_good = 0.31  → not a memorised topic
  │
  ├─ P(no_retrieval) = 0.28 < threshold 0.75
  │
  ├─ RETRIEVAL FALLBACK
  │   └─ hand off to kv_inference.py (full SP3 path)
  │       ├─ qdrant.query_points(top_k=5)
  │       ├─ version check → KV injection or text fallback
  │       └─ return cited answer
  │
  └─ latency: ~3.5s (draft + retrieval + generation)
              draft overhead: ~0.15s (20 tokens, negligible)
```

### 6.7 Access Tracker Batch Flush

```
[kv_background.py — runs continuously in background thread]

every 50 queries OR every 5 minutes (whichever first):
         │
         ▼
[access_tracker.flush()]
  ├─ snapshot = copy of in-memory counter dict
  ├─ clear in-memory dict (new queries accumulate fresh)
  │
  ├─ for each chunk_id in snapshot:
  │   ├─ delta = {access_count: +N, last_accessed_ts: T,
  │   │           avg_retrieval_rank: rolling_avg}
  │   └─ qdrant.update_payload(chunk_id, delta)
  │
  ├─ recompute tier labels (first-match-wins order):
  │   ├─ frozen = access_count == 0
  │   ├─ hot    = top 15% of non-frozen AND last_accessed_ts > now-7d
  │   ├─ warm   = next 50% of non-frozen AND last_accessed_ts > now-30d
  │   └─ cold   = all remaining non-frozen
  │
  ├─ batch update tier field for all changed chunks
  │
  └─ every Sunday 00:00 UTC: write access_report.json
      {summary: {hot, warm, cold, frozen counts},
       frozen_chunk_ids: [...],
       hot_topics: [...],
       parametric_answer_rate: 0.61,
       most_accessed_pages: [38, 7, 42, 15, 93]}
```

### 6.8 LoRA Training Mid-Run (Inference Continues Uninterrupted)

```
[lora_trainer.py running — round 7 in progress]

         training process (GPU, separate)          inference process
         ──────────────────────────────────         ─────────────────
         loading lora_v6 weights                   serving queries using
         building dataset (new + replay)           lora_v6 (unchanged)
         forward + backward pass                   kv_inference.py reads
         gradient accumulation                     version.json → v6
         saving checkpoint                         all KV checks against v6
         version.json updated → v7                 ← only after this write
                                                   do new queries see v7
         compute KV for new chunks                 stale chunks (v6) still
         update Qdrant payload kv_version=7        serve text-in-context
         prs_evaluator.py runs                     until bg worker heals them
```

### 6.9 Phase 2 → Phase 3 Transition

```
[prs_evaluator.py — runs at end of every lora_trainer.py round]

  ├─ compute PRS for this round → prs = 0.81
  ├─ append to version.json: prs_history = [..., {round:8, prs:0.81}]
  │
  ├─ check transition condition:
  │   last_two = prs_history[-2:]  → [{round:7, prs:0.80}, {round:8, prs:0.81}]
  │   all(r["prs"] >= 0.80) → TRUE
  │
  ├─ PHASE TRANSITION: 2 → 3
  │   ├─ version.json["phase"] = 3        ← atomic write
  │   ├─ log: "Phase 3 activated at round 8, PRS=0.81"
  │   │
  │   └─ build known-good query index:
  │       ├─ for each of 50 eval FAQs scored in this round:
  │       │   if param_accuracy_ratio ≥ 0.85:
  │       │       add query embedding to known_good_queries.json
  │       └─ confidence_gate.py reads this index on next startup
  │
  ├─ in-flight queries (currently being processed by kv_inference.py):
  │   NOT affected — they complete against Phase 2 path normally
  │   Phase 3 intercepts only queries arriving AFTER version.json is written
  │
  └─ next query arrives:
      ├─ confidence_gate.py reads version.json → phase=3
      ├─ attempts direct answer first (SP4 path)
      └─ falls back to kv_inference.py (SP3 path) if P(no_retrieval) < 0.75
```

---

## 7. Sub-Project Specifications

### SP1 — Extended KV Indexer

**File:** `kv_indexer.py`
**Depends on:** `model_loader.py`, `kv_utils.py`, existing `bedrock_rag.py`

**Responsibilities:**
- Extend the `index` command to also compute mean-pooled KV tensors per chunk
- Store `kv_cache` (base64 fp16) and `kv_version=null` in Qdrant payload
- Add `source_file` and `indexed_at` fields to payload
- Expose `compute-kv` subcommand: compute KV for chunks matching a filter
  without re-embedding or re-chunking. Supports two staleness cases:
  - `kv_version=null` — chunks never computed (new documents)
  - `kv_version<N` — chunks computed under an older LoRA version (post-training)
- After each LoRA round, `lora_trainer.py` calls
  `kv_indexer.py compute-kv --stale-version N` to proactively heal all
  integer-versioned stale chunks (not just null). This prevents unbounded
  staleness for chunks never queried.

**Interface:**
```bash
python3 kv_indexer.py index new_doc.pdf [--config my_config.json]
python3 kv_indexer.py compute-kv --filter kv_version=null
python3 kv_indexer.py compute-kv --source-file ec2_guide.pdf
python3 kv_indexer.py compute-kv --stale-version 6  # heal all kv_version < 6
```

**KV computation detail:**
```python
def compute_kv_for_chunk(text: str, model, tokenizer) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=False,
                        use_cache=True)
    # HuggingFace past_key_values: tuple[tuple[Tensor, Tensor]] per layer
    # Each K or V tensor shape: [batch=1, num_kv_heads=8, seq_len, head_dim=128]
    # We mean-pool over the seq_len dimension (dim=2 before squeeze, dim=1 after)
    # to produce a fixed-size representation independent of chunk length.
    kv_stacked = []
    for layer_kv in outputs.past_key_values:
        k, v = layer_kv        # each: [1, num_kv_heads=8, seq_len, head_dim=128]
        k = k.squeeze(0)       # → [8, seq_len, 128]
        v = v.squeeze(0)       # → [8, seq_len, 128]
        k_pooled = k.mean(dim=1)  # mean over seq_len → [8, 128]
        v_pooled = v.mean(dim=1)  # mean over seq_len → [8, 128]
        kv_stacked.append(torch.stack([k_pooled, v_pooled]))  # [2, 8, 128]
    result = torch.stack(kv_stacked).cpu().to(torch.float16).numpy()
    # final shape: [num_layers=28, 2, num_kv_heads=8, head_dim=128]
    # = 28 × 2 × 8 × 128 = 57,344 float16 values ≈ 112 KB per chunk
    return result
```

---

### SP2 — LoRA Training Pipeline

**Files:** `lora_trainer.py`, `replay_buffer.py`
**Depends on:** `model_loader.py`, `kv_utils.py`, `version.json`

**Responsibilities:**
- Fine-tune Llama 3.2 3B attention projections on new document chunks
- Sample replay buffer weighted by tier (hot 40%, warm 40%, cold 15%, frozen 5%)
- Save LoRA adapter checkpoint, increment `lora_version`
- After training: re-compute KV for new chunks with updated weights
- Run `prs_evaluator.py` after each round; update `version.json`

**LoRA config:**
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

**Training objective:** next-token prediction (causal LM loss) on chunk texts

**Replay buffer (`replay_buffer.py`):**
- Maintains index of all previously indexed chunk IDs with tier labels
- `sample(n, weight_by_tier)`: returns chunk texts weighted by tier
- Backed by SQLite (`replay_buffer.db`) for persistence across runs

**Interface:**
```bash
python3 lora_trainer.py \
  --new-chunks-filter source_file=ec2_guide.pdf \
  --replay-ratio 0.2 \
  --epochs 3 \
  --output lora_checkpoints/v7/
```

---

### SP3 — KV-Injected Inference Engine

**Files:** `kv_inference.py`, `kv_background.py`
**Depends on:** `model_loader.py`, `kv_utils.py`, `access_tracker.py`
**Replaces:** `ollama_answer.py` (kept for fallback/comparison)

**Responsibilities:**
- Accept query, retrieve top-K chunks from Qdrant
- Version-check each chunk's `kv_version` against `current_lora_version`
- If ALL fresh → KV injection path (inject `past_key_values`)
- If ANY stale → text-in-context fallback (same as current `ollama_answer.py`)
- Either path → push stale chunk IDs to background queue
- Background worker (`kv_background.py`) computes fresh KV and updates Qdrant

**KV injection:**
```python
def generate_with_kv(query: str, chunks: list[dict], model, tokenizer) -> str:
    # Deserialise and stack KV tensors
    past_kvs = []
    for chunk in chunks:
        kv = kv_utils.deserialize_kv(chunk["kv_cache"])  # [28,2,8,128]
        past_kvs.append(kv)
    combined_past = kv_utils.stack_past_key_values(past_kvs)
    # Build query-only prompt (no chunk text in context)
    prompt = f"Answer based on your knowledge: {query}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            past_key_values=combined_past,
            max_new_tokens=512,
            do_sample=False
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)
```

**Interface (pipe-compatible, drop-in for `ollama_answer.py`):**
```bash
python3 bedrock_rag.py search "query" | python3 kv_inference.py
python3 kv_inference.py --query "what is bedrock?"
```

---

### SP4 — Confidence Gate + PRS Evaluator

**Files:** `confidence_gate.py`, `prs_evaluator.py`
**Depends on:** `kv_inference.py`, `model_loader.py`, `version.json`
**Active when:** `version.json.phase == 3` (PRS ≥ 0.80 for 2 consecutive rounds)

**Confidence gate signals:**
| Signal | Computation | Weight |
|--------|------------|--------|
| token_entropy | mean entropy of draft answer tokens | 0.4 |
| hedging_score | density of uncertainty markers | 0.3 |
| query_similarity | cosine sim to known-good query index | 0.3 |

**Known-good query index:**
- Small in-memory (or tiny Qdrant collection) of queries that scored
  `accuracy_ratio ≥ 0.85` in the most recent PRS offline evaluation
- Updated after every PRS run

**PRS computation (`prs_evaluator.py`):**
```
accuracy_ratio    = min(
                      mean(cosine_sim(param_answers, ground_truths))
                      / mean(cosine_sim(rag_answers, ground_truths)),
                      1.0            ← capped: ratio can exceed 1.0 when
                    )                  parametric outperforms RAG; cap keeps
                                       PRS in [0, 1] and threshold stable

calibration_score = 1 - mean(|self_confidence - actual_accuracy|)
                    where self_confidence is extracted via structured prompt:
                      "On a scale of 0–100, how confident are you in your
                       answer above? Reply with a single integer only."
                    The integer / 100 is used as self_confidence ∈ [0, 1].
                    actual_accuracy = cosine_sim(param_answer, ground_truth)

self_consistency  = mean(cosine_sim between 3 sampled answers per question,
                         temperature=0.7)

PRS = 0.5 * accuracy_ratio + 0.3 * calibration_score + 0.2 * self_consistency
    ∈ [0, 1]  (all three components bounded to [0, 1])
```

**Phase transition logic (runs after every PRS evaluation):**
```python
if len(prs_history) >= 2:
    last_two = prs_history[-2:]
    if all(r["prs"] >= 0.80 for r in last_two):
        version_json["phase"] = 3  # activate confidence gate
```

---

### SP5 — Access Tracker

**File:** `access_tracker.py`
**Depends on:** Qdrant client
**Used by:** SP3 (kv_inference.py), SP4 (confidence_gate.py)

**In-memory counter (thread-safe):**
```python
@dataclass
class ChunkAccess:
    count: int = 0
    rank_sum: float = 0.0
    last_ts: int = 0
    parametric_hits: int = 0
```

**Tier classification rules (evaluated in order; first match wins):**
```
frozen : access_count == 0
           → never retrieved; skip KV; minimal replay weight
hot    : access_count in top 15% of non-frozen chunks
         AND last_accessed_ts > now - 7d
           → highest priority for KV recompute and replay
warm   : access_count in next 50% of non-frozen chunks
         AND last_accessed_ts > now - 30d
           → standard treatment
cold   : all remaining non-frozen chunks
         (low access_count OR not accessed in > 30 days)
           → lowest priority; flagged for review after 90 days
```
Rules are mutually exclusive: frozen is tested first; hot/warm/cold apply
only to chunks with access_count > 0.

**Batch flush triggers:** every 50 queries OR every 5 minutes

**Weekly report** → `access_report.json`

---

### SP6 — Monitoring Dashboard

**File:** `monitoring_dashboard.py`
**Port:** 8080
**Framework:** FastAPI + single-page HTML/JS (auto-refresh every 30s)

**Data sources:**
| Panel | Source |
|-------|--------|
| PRS trend | `version.json` prs_history |
| Fine-tuning status | `version.json` + training lock file |
| Chunk tier distribution | Qdrant payload aggregation |
| Top accessed chunks | Qdrant payload sort by access_count |
| Retrieval vs direct trend | `access_report.json` daily snapshots |
| Confidence distribution | in-memory stats from confidence_gate.py |

**API endpoints:**
```
GET /api/stats          → full stats JSON (all panels)
GET /api/version        → version.json contents
GET /api/access-report  → latest access_report.json
GET /api/health         → system health check
```

---

## 8. Configuration (`my_config.json` — extended)

```json
{
  "collection":          "bedrock-user-guide",
  "qdrant_host":         "localhost",
  "qdrant_port":         6333,
  "embed_model":         "BAAI/bge-small-en-v1.5",
  "vector_dim":          384,
  "chunk_size":          600,
  "chunk_overlap":       60,
  "embed_batch":         64,
  "upsert_batch":        128,
  "top_k":               5,

  "llm_model":           "meta-llama/Llama-3.2-3B-Instruct",
  "llm_device":          "cuda",
  "llm_dtype":           "float16",
  "lora_rank":           16,
  "lora_alpha":          32,
  "lora_dropout":        0.05,
  "lora_target_modules": ["q_proj", "k_proj", "v_proj"],
  "lora_epochs":         3,
  "lora_lr":             2e-4,
  "replay_ratio":        0.2,
  "kv_num_layers":       28,
  "kv_num_heads":        8,
  "kv_head_dim":         128,

  "prs_threshold":       0.80,
  "prs_consecutive":     2,
  "prs_eval_sample":     50,
  "gate_threshold":      0.75,
  "access_flush_queries":50,
  "access_flush_seconds":300,
  "dashboard_port":      8080,
  "checkpoint_dir":      "lora_checkpoints/"
}
```

---

## 9. Deployment (AWS EC2 g5.xlarge)

```
Instance:   g5.xlarge (A10G 24GB VRAM, 4 vCPU, 16GB RAM)
AMI:        Deep Learning AMI GPU PyTorch 2.x (Ubuntu 22.04)
Storage:    100GB gp3 EBS (Qdrant data + checkpoints + replay buffer)

Ports open: 6333 (Qdrant REST), 6334 (Qdrant gRPC), 8080 (dashboard)

Process layout:
  [systemd] qdrant server          → always running
  [systemd] monitoring_dashboard   → always running (port 8080)
  [systemd] kv_background.py       → always running; manages two internal
                                     threads: (1) background KV recompute
                                     queue, (2) access_tracker flush loop
                                     (every 50 queries or 5 min via
                                     threading.Timer — NOT a cron job;
                                     flush reads the in-process counter dict)
  [manual]  index_and_train.py     → triggered per new document

Spot instance strategy:
  Use on-demand for inference (always-on at ~$1/hr)
  Spot interruption-tolerant for training runs (save checkpoint on SIGTERM)
  Estimated cost: ~$25/day on-demand, ~$10/day with spot for training
```

---

## 10. Open Questions (resolve before SP3 implementation)

| # | Question | Default assumption |
|---|----------|-------------------|
| Q1 | How to stack KV from multiple chunks into `past_key_values`? | Concatenate along seq dim. Each chunk contributes one mean-pooled "token" position. 5 chunks → seq_len=5 in past_key_values. HuggingFace format: tuple of 28 (K,V) pairs each shaped [1, 8, 5, 128]. Validate with unit test before SP3 integration. |
| Q2 | Should text-in-context fallback use the full `ollama_answer.py` system prompt or the HuggingFace model's own system prompt? | HuggingFace system prompt (consistent with KV path) |
| Q3 | Phase 3 gate threshold (0.75): should it be per-query-type or global? | Global initially, per-query-type in v2 |
| Q4 | Replay buffer: cap total size or use all historical chunks? | Cap at 5,000 chunks; evict by oldest + lowest tier |
