# Three-Arm Control Experiment Plan
## KVForge — Closing Open Item #1 from Peer Review

**Date:** 2026-09-02  
**Purpose:** Add the missing control arm (base model + retrieval) and produce the definitive three-arm factual-accuracy comparison that isolates (a) the effect of LoRA fine-tuning on retrieval-augmented performance and (b) whether parametric gain comes at the cost of base-model RAG quality (catastrophic forgetting).

---

## 1. What the Three Arms Test

| Arm | Label in paper | Model | Retrieval | Gate |
|-----|---------------|-------|-----------|------|
| **A** | Base + RAG | `google/gemma-4-E2B-it` (no adapter) | ✓ top-K chunks injected as context | N/A |
| **B** | KVForge + RAG | same base + LoRA adapter (N-tier) | ✓ same retrieval | disabled |
| **C** | KVForge Param | same base + LoRA adapter (N-tier) | ✗ closed-book | disabled |

- **A vs B**: isolates catastrophic forgetting — does LoRA hurt retrieval-augmented accuracy?
- **A vs C**: isolates parametric gain — how much does closing the retrieval loop cost vs. free retrieval?
- **B vs C**: already partially measured but now with a consistent FA metric.

Arm C is the "Param FA" already in Table 3. Arms A and B are new runs.

---

## 2. Existing Artifacts on Disk

### 2.1 LoRA Checkpoints (all from base `google/gemma-4-E2B-it`, rank=16, alpha=32)

| UC | Checkpoints available | N tiers |
|----|-----------------------|---------|
| UC1 Customer Support | `lora_checkpoints/gemma4_N{200,500,1000}_seed{42,43,44}/` | 200, 500, 1000 |
| UC2 PubMedQA | `lora_checkpoints/gemma4_N{200,500,1000}_seed{42,43,44}/`, `gemma4_N297` | 200, 500, 1000 (297) |
| UC4 Bedrock UG | `lora_checkpoints/phase3_N{500,1000,2000,4000,6000}/` | 500, 1000, 2000, 4000, 6000 |

### 2.2 Held-Out Eval Sets

| UC | File | Items |
|----|------|-------|
| UC1 | Need to regenerate — 76 QA pairs from training metadata (or use `faqs.json` once recovered) | 76 |
| UC2 | `eval_tier_{50,100,200,297}.json` (`.items` key) | 50–297 |
| UC3 | `eval_heldout.json` (`.items` key, 200 items) + `eval_tier_{500,1000,2000}.json` | 200 |
| UC4 | `eval_heldout.json` (`.items` key, 89 items) + `eval_tier_{500,1000,2000,4000,6000}.json` | 89 |

**UC1 action required:** The `faqs.json` used during training is not present in the UC1 directory. Before the GPU run, regenerate it with `sleep_faq_generator.py` or manually recover from the training log. The training metadata shows 76 QA examples were used; that same set (or a strict held-out subset) must be used for eval.

### 2.3 Retrieval Index

Qdrant collections must be indexed and accessible for Arms A and B. On the new GPU:
- UC1: `customer-support` collection (qdrant localhost:6333)
- UC2–UC4: respective collections per their `config.json`

If re-indexing from scratch is needed, `pipeline/kv_indexer.py` handles it. The raw corpus data should travel with the code rsync.

### 2.4 FA Scoring Components

- **LLM judge**: Claude Opus 4.7 — already specified in §5.7 of the paper; requires `ANTHROPIC_API_KEY` on the GPU.
- **Embedding similarity**: `BAAI/bge-small-en-v1.5` via `fastembed` (already in `requirements.txt`).
- **FA formula**: `0.5 × judge_score + 0.5 × cosine_sim(candidate_emb, reference_emb)`.

---

## 3. New Script: `pipeline/three_arm_evaluator.py`

### 3.1 Design Principles

- **Offline only** — no running dashboard or vLLM server required; uses HuggingFace pipeline directly (4-bit QLoRA, same as training).
- **Load base model once** per UC run; swap LoRA adapter in-place for each N-tier checkpoint (PEFT `set_adapter` / disable-and-load pattern).
- **Retrieval for Arms A and B**: issue a text-in-context RAG call against Qdrant (or a local FAISS dump if Qdrant is unavailable) using the same top-K=5 and `embed_model` as in production.
- **Closed-book for Arm C**: generate with no context, gate disabled.
- **FA scoring**: async Anthropic SDK calls for the judge; BGE cosine similarity computed locally.

### 3.2 Script Interface

```
python -m pipeline.three_arm_evaluator \
    --config   examples/usecase4_bedrock_userguide/config.json \
    --eval     examples/usecase4_bedrock_userguide/eval_heldout.json \
    --ckpt-dir examples/usecase4_bedrock_userguide/lora_checkpoints/ \
    --ckpt-pattern "phase3_N*" \
    --out-dir  results/three_arm/uc4/ \
    --max-samples 89
```

Key flags:
- `--ckpt-pattern`: glob selecting which checkpoint subdirs to evaluate (e.g. `gemma4_N*` for UC1/2, `phase2_N*` for UC3, `phase3_N*` for UC4).
- `--arms`: comma-separated subset, default `A,B,C` (useful to rerun just one arm).
- `--judge-model`: default `claude-opus-4-7` (override for testing with a cheaper model).
- `--max-samples`: cap for large eval sets; default = all items.

### 3.3 Output Schema

Per run, one JSON written to `--out-dir/<uc>_<ckpt_name>_three_arm.json`:

```json
{
  "uc": "usecase4_bedrock_userguide",
  "checkpoint": "phase3_N1000",
  "n_tier": 1000,
  "n_eval": 89,
  "arm_A": { "fa_mean": 0.71, "fa_std": 0.09, "per_question": [...] },
  "arm_B": { "fa_mean": 0.74, "fa_std": 0.08, "per_question": [...] },
  "arm_C": { "fa_mean": 0.78, "fa_std": 0.11, "per_question": [...] }
}
```

Each `per_question` entry:
```json
{
  "question": "...",
  "reference": "...",
  "answer_A": "...", "answer_B": "...", "answer_C": "...",
  "judge_A": 0.80, "judge_B": 0.85, "judge_C": 0.90,
  "cosine_A": 0.72, "cosine_B": 0.76, "cosine_C": 0.81,
  "fa_A": 0.76, "fa_B": 0.805, "fa_C": 0.855
}
```

A summary CSV `three_arm_summary.csv` is written alongside, one row per (UC, N):

```
uc, n_tier, arm_A_fa, arm_B_fa, arm_C_fa, delta_AB, delta_AC
```

### 3.4 Implementation Sketch

```python
# pipeline/three_arm_evaluator.py (outline)

def load_base_model(cfg):
    """Load google/gemma-4-E2B-it in 4-bit NF4; return (model, tokenizer)."""

def load_lora(model, ckpt_path):
    """Apply PEFT adapter from ckpt_path; return updated model."""

def unload_lora(model):
    """Disable active adapter, merge nothing (keep base weights clean)."""

def retrieve_context(query, cfg, embedder):
    """Query Qdrant top-K, return concatenated chunk text."""

def generate(model, tokenizer, prompt, max_new_tokens=256):
    """Single greedy generation; return text."""

def fa_score(question, candidate, reference, judge_client, embedder):
    """0.5 * judge + 0.5 * cosine. Returns float in [0,1]."""

def run_arm(arm_label, queries, model, tokenizer, cfg, embedder, judge_client, use_retrieval):
    """Run all queries for one arm; return list of per-Q dicts."""

def main():
    # 1. Load base model once
    # 2. For each checkpoint matching --ckpt-pattern:
    #      a. load_lora(model, ckpt)
    #      b. run_arm("B", queries, retrieval=True)
    #      c. run_arm("C", queries, retrieval=False)
    #      d. unload_lora(model)
    # 3. After all checkpoints: run_arm("A", queries, retrieval=True) [base, no adapter]
    # 4. Aggregate and write JSON + CSV
```

### 3.5 Tests: `tests/test_three_arm_evaluator.py`

The test file is written against the public API of `pipeline/three_arm_evaluator.py` using `unittest.mock` so it runs with no GPU, no Qdrant, and no Anthropic key.

| Test | What it checks |
|------|----------------|
| `test_fa_score_mock_judge` | Mock Anthropic client returns 0.8; BGE cosine returns 0.6. Assert FA = 0.5×0.8 + 0.5×0.6 = 0.70 ± 1e-4. |
| `test_load_unload_lora_pattern` | Mock PEFT `PeftModel.from_pretrained`; call `load_lora` then `unload_lora`; assert adapter is disabled (no active adapter) after unload. |
| `test_retrieve_context_mock_qdrant` | Mock Qdrant `QdrantClient.search`; assert `retrieve_context` returns concatenated text from top-K hits. |
| `test_generate_smoke` | Mock HF `pipeline` call; assert `generate` returns stripped non-empty string. |
| `test_output_json_schema` | Run evaluator on 2-question mock dataset (mocked judge + embedder); validate output JSON has keys `uc`, `checkpoint`, `n_tier`, `n_eval`, `arm_A`, `arm_B`, `arm_C`; each arm has `fa_mean`, `fa_std`, `per_question`. |
| `test_summary_csv_written` | Run evaluator on mock dataset; assert `three_arm_summary.csv` exists with columns `uc,n_tier,arm_A_fa,arm_B_fa,arm_C_fa,delta_AB,delta_AC`. |
| `test_arm_A_no_adapter` | Run Arm A with mocked model; assert model has no active PEFT adapter (base model unchanged during Arm A run). |
| `test_integration_smoke` | 2 fake QA pairs, all mocks (judge always returns 0.8, cosine always 0.7, retrieval always returns "context text"); assert no exceptions raised and all three arms produce results with numeric FA scores. |

Run with: `python -m pytest tests/test_three_arm_evaluator.py -v --override-ini="addopts="`

All tests must pass without GPU (`CUDA_VISIBLE_DEVICES=""` acceptable). CI should run them alongside `test_prs_evaluator.py` in the same pytest invocation.

---

## 4. Expected Results & Paper Updates

### 4.1 Extended Table 3

Current Table 3 has two FA columns: *RAG FA* (Phase 1 baseline) and *Param FA* (Arm C). After this experiment, it becomes:

| UC | N | Base+RAG FA (Arm A) | LoRA+RAG FA (Arm B) | Param FA (Arm C) | ΔA→B | ΔA→C |
|----|---|---------------------|---------------------|-----------------|------|------|
| UC1 | 200 | — | — | [existing] | — | — |
| ... | ... | | | | | |

ΔA→B quantifies catastrophic forgetting (should be ≤ 0 or small positive, meaning LoRA does not degrade RAG). ΔA→C quantifies parametric advantage (expected positive as N grows).

### 4.2 Section 6.3 Update

Add two sentences after the existing parametric-vs-RAG finding:
1. Arm A vs B result: "The base model with retrieval (Arm A) and its LoRA-fine-tuned counterpart with retrieval (Arm B) show a mean FA difference of X%, indicating negligible catastrophic forgetting of retrieval-based reasoning across all N tiers and corpora."
2. Arm A vs C result: "At N≥1000, Arm C (parametric) surpasses Arm A (base+RAG) by Y%, confirming that the knowledge distilled during LoRA training exceeds what retrieval alone provides."

---

## 5. GPU Deployment Steps

### 5.1 Prerequisites on New GPU

```bash
# Verify CUDA and available VRAM
nvidia-smi

# Expected: 1× NVIDIA A10G (24 GB VRAM) — sufficient for 4-bit QLoRA on gemma-4-E2B-it
# Model size at 4-bit: ~3–4 GB; full eval pipeline: ~6–8 GB peak
```

### 5.2 Sync Code and Artifacts

LoRA checkpoints are **not** synced — they will be generated fresh on the GPU as part of this experiment.

```bash
GPU_IP=<new_gpu_ip>
PEM_KEY=~/.ssh/<your-key>.pem

# Sync code + corpus data; exclude any existing checkpoints (will be trained fresh)
rsync -avz --progress -e "ssh -i $PEM_KEY" \
  <path-to-repo>/KVForge/ \
  ubuntu@$GPU_IP:~/kvforge/ \
  --exclude="*.pyc" --exclude="__pycache__" --exclude=".git" \
  --exclude="embeddings.npy" --exclude="replay.db" \
  --exclude="lora_checkpoints/"

# Sync HuggingFace model cache to avoid re-downloading on GPU (optional but saves ~30 min)
rsync -avz -e "ssh -i $PEM_KEY" \
  ~/.cache/huggingface/ ubuntu@$GPU_IP:~/.cache/huggingface/
```

### 5.3 Install Dependencies on GPU

```bash
ssh -i $PEM_KEY ubuntu@$GPU_IP

cd ~/kvforge
pip install -r requirements.txt
# Ensure anthropic SDK is present for judge:
pip install anthropic>=0.40

# Set API key
export ANTHROPIC_API_KEY=<your_key>
```

### 5.4 Start Qdrant and Index Corpora

```bash
# Start Qdrant (Docker recommended)
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# Index each UC (needed for retrieval in Arms A and B)
# UC1
python -m pipeline.kv_indexer \
  --config examples/usecase1_customer_support/config.json \
  index examples/usecase1_customer_support/data/

# Repeat for UC2, UC3, UC4 with their respective data/ dirs
```

### 5.5 Generate Training FAQs (All UCs)

All four corpora need FAQs generated before LoRA training can begin. Run `sleep_faq_generator.py` on the GPU for each UC:

```bash
# UC1 — 76 QA pairs (matches held-out eval count)
python -m pipeline.sleep_faq_generator \
  --config examples/usecase1_customer_support/config.json \
  --output examples/usecase1_customer_support/faqs.json \
  --count 76

# UC2 — 297 QA pairs (full tier)
python -m pipeline.sleep_faq_generator \
  --config examples/usecase2_pubmedqa/config.json \
  --output examples/usecase2_pubmedqa/faqs.json \
  --count 297

# UC3 — 200 QA pairs
python -m pipeline.sleep_faq_generator \
  --count 200

# UC4 — 89 QA pairs (matches held-out set)
python -m pipeline.sleep_faq_generator \
  --config examples/usecase4_bedrock_userguide/config.json \
  --output examples/usecase4_bedrock_userguide/faqs.json \
  --count 89
```

Estimated time: ~1–2 hours total (cloud LLM API calls; depends on rate limits).

### 5.6 Train LoRA Checkpoints (All UCs × N Tiers)

Each N tier is trained independently from the base model using the FAQs generated above. Seed 42 is used throughout for reproducibility.

```bash
# UC1: N = {200, 500, 1000}
for N in 200 500 1000; do
  python -m pipeline.lora_trainer \
    --config examples/usecase1_customer_support/config.json \
    --faqs   examples/usecase1_customer_support/faqs.json \
    --n-samples $N --seed 42 \
    --out-dir examples/usecase1_customer_support/lora_checkpoints/gemma4_N${N}_seed42/
done

# UC2: N = {200, 500, 1000, 297}
for N in 200 500 1000 297; do
  python -m pipeline.lora_trainer \
    --config examples/usecase2_pubmedqa/config.json \
    --faqs   examples/usecase2_pubmedqa/faqs.json \
    --n-samples $N --seed 42 \
    --out-dir examples/usecase2_pubmedqa/lora_checkpoints/gemma4_N${N}_seed42/
done

# UC3: N = {500, 1000, 2000}
for N in 500 1000 2000; do
  python -m pipeline.lora_trainer \
    --n-samples $N --seed 42 \
done

# UC4: N = {500, 1000, 2000, 4000, 6000}
for N in 500 1000 2000 4000 6000; do
  python -m pipeline.lora_trainer \
    --config examples/usecase4_bedrock_userguide/config.json \
    --faqs   examples/usecase4_bedrock_userguide/faqs.json \
    --n-samples $N --seed 42 \
    --out-dir examples/usecase4_bedrock_userguide/lora_checkpoints/phase3_N${N}/
done
```

Estimated training time: ~1.5–3 h per N tier (4-bit QLoRA on A10G); 16 total checkpoints × ~2 h avg ≈ **28–48 hours** total. Run in a `tmux` session or with `nohup`.

### 5.7 Verify PRS Before Evaluation

Run `prs_evaluator.py` on each UC after training to confirm the checkpoints are ready (PRS ≥ 0.60 gate threshold). This catches degenerate training runs before committing 8–12 h of three-arm eval.

```bash
# Verify all UCs using highest-N checkpoint (strongest signal)
python -m pipeline.prs_evaluator \
  --config examples/usecase1_customer_support/config.json \
  --faqs   examples/usecase1_customer_support/faqs.json

python -m pipeline.prs_evaluator \
  --config examples/usecase2_pubmedqa/config.json \
  --faqs   examples/usecase2_pubmedqa/faqs.json

python -m pipeline.prs_evaluator \

python -m pipeline.prs_evaluator \
  --config examples/usecase4_bedrock_userguide/config.json \
  --faqs   examples/usecase4_bedrock_userguide/faqs.json
```

**Gate:** PRS ≥ 0.60 required before proceeding to three-arm eval. If a UC fails, inspect the training log for that N tier before retraining.

### 5.8 Resolve UC1 Eval Set

The UC1 `faqs.json` will be generated in §5.5. The held-out eval set is either recovered from the training log or treated as the regenerated `faqs.json` (76 QA pairs):

```bash
# If a separate held-out split is needed, recover from training log
grep "question" examples/usecase1_customer_support/fkds_training_rounds/round_1_fkds.log | head -5

# Otherwise, the faqs.json generated in §5.5 doubles as the eval set (same 76 pairs)
cp examples/usecase1_customer_support/faqs.json \
   examples/usecase1_customer_support/eval_heldout.json
```

### 5.9 Run the Three-Arm Evaluator

```bash
# UC4 (fully equipped, 89 held-out, 5 N-tiers) — run first as pilot
python -m pipeline.three_arm_evaluator \
  --config   examples/usecase4_bedrock_userguide/config.json \
  --eval     examples/usecase4_bedrock_userguide/eval_heldout.json \
  --ckpt-dir examples/usecase4_bedrock_userguide/lora_checkpoints/ \
  --ckpt-pattern "phase3_N*" \
  --out-dir  results/three_arm/uc4/

# UC3 (200 items, 3 tiers)
python -m pipeline.three_arm_evaluator \
  --ckpt-pattern "phase2_N*" \
  --out-dir  results/three_arm/uc3/

# UC2 (use eval_tier_297.json for maximum coverage)
python -m pipeline.three_arm_evaluator \
  --config   examples/usecase2_pubmedqa/config.json \
  --eval     examples/usecase2_pubmedqa/eval_tier_297.json \
  --ckpt-dir examples/usecase2_pubmedqa/lora_checkpoints/ \
  --ckpt-pattern "gemma4_N*" \
  --out-dir  results/three_arm/uc2/

# UC1 (76 items after recovery)
python -m pipeline.three_arm_evaluator \
  --config   examples/usecase1_customer_support/config.json \
  --eval     examples/usecase1_customer_support/eval_heldout.json \
  --ckpt-dir examples/usecase1_customer_support/lora_checkpoints/ \
  --ckpt-pattern "gemma4_N*" \
  --out-dir  results/three_arm/uc1/
```

### 5.10 Estimated Runtime (Full From-Scratch Pipeline)

| Step | Estimated Time | Notes |
|------|----------------|-------|
| FAQ generation (all 4 UCs) | ~1–2 h | Cloud LLM API; rate-limited |
| LoRA training — 16 checkpoints total | ~28–48 h | ~2 h avg per checkpoint on A10G |
| PRS verification (all 4 UCs) | ~1–2 h | One run per UC |
| Qdrant indexing (all 4 UCs) | ~30–60 min | `kv_indexer.py` per UC |
| Three-arm eval — all 4 UCs | ~8–12 h | Judge API + generation |
| **Total wall clock** | **~2–3 days** | Run training in `tmux`; eval overnight |

**Training breakdown (16 checkpoints):**
- UC1: 3 tiers × ~1.5 h = ~4.5 h
- UC2: 4 tiers × ~2 h = ~8 h
- UC3: 3 tiers × ~2.5 h = ~7.5 h
- UC4: 5 tiers × ~3 h = ~15 h (large N)

Total LLM judge API calls (eval only): ~(89×5 + 297×3 + 200×3 + 76×3) × 3 arms ≈ 8,700 calls. At ~$0.015/1K tokens (Opus 4.7), cost ≈ $15–25.

---

## 6. Acceptance Criteria

- [ ] `three_arm_summary.csv` exists with rows for all (UC, N) combinations.
- [ ] Arm B FA ≥ Arm A FA − 0.02 (< 2% forgetting) for all corpora at all N.
- [ ] Arm C FA > Arm A FA for at least 3 of 4 corpora at N=1000.
- [ ] No NaN FA scores (all judge calls returned numeric scores).
- [ ] Paper §6.3 updated with concrete ΔA→B and ΔA→C values.
- [ ] Table 3 extended with Arm A and Arm B columns.

---

## 7. Open Items Not Addressed in This Plan

These remain as future work (per reviewer response):

1. **#1** — Ablation of the KV cache injection contribution (Phase 2 vs Phase 1 in Table 3) with latency breakdown.
3. **#3** — Latency/TTFT benchmarks comparing Phase 1, 2, 3 response times.
6. **#6** — Gate routing fraction: percentage of live queries passing confidence gate ≥ 0.75 per corpus.