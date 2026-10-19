# KVForge Experiment Catalog

> **Sections 0–6 below cover July–August 2026 only.** They were last refreshed 2026-08-09 and
> predate the September "parametric consolidation" reframe — see **Phase 7** at the end of this
> document for that work. Phase 6 cells still reading "(empty)" have since been filled.
>
> Compiled from ~188 summary.json files across EC2-A (A10G) and EC2-B (RTX PRO 4500).
> Model: **Google Gemma-4-E2B-it** (4-bit quantized, Llama-3.2-3B-Instruct for Phase 1 only)
> Vector store: **Qdrant** (ChromaDB for old PubMedQA only)
> Embedding: **bge-small-en-v1.5** (384d) — MXBAI-large ablated separately
> Judge: **DeepSeek V4 Flash** (Fireworks) with **Option C (CORRECT/PARTIAL/INCORRECT)** partial scoring — see `eval/metrics.py`

---

## Datasets

| ID | Name | Chunks | Chunk | VectorDB | Embedding | Domain |
|:---:|---|---|---:|---|---|---|
| **UC1** | Customer Support Q&A | ~500 | 600/60 | Qdrant | bge-small (384d) | Dialogue |
| **UC2** | PubMedQA Biomedical Q&A | 2,918 | 600/60 | Qdrant → ChromaDB | bge-small (384d) | Medical lit |
| **UC4** | AWS Bedrock User Guide | ~1,200 | default | Qdrant | bge-small (384d) | Technical docs |
| **HotpotQA** | Multi-Hop QA | ~5,000 | 600/60 | Qdrant | bge-small (384d) | Multi-hop |
| **2WikiMQA** | 2Wiki Multi-Hop QA | ~5,000 | 600/60 | Qdrant | bge-small (384d) | Multi-hop |

---

## Experiment Organization

Results are stored in two parallel directory trees:

```
results/
├── pubmedqa_crossover/          ← Old Llama-3.2-3B (invalid, stale KV cache)
├── pubmedqa_gemma4_crossover/   ← PubMedQA Gemma4 (text_rag + parametric only)
├── hotpotqa_textrag_20260807_0218/
│
├── absorption/                  ← Systematic "absorption" experiment series
│   ├── phase1/                  ← UC4, Llama-3.2-3B, text_rag + parametric
│   ├── phase1_gemma4/           ← UC4, Gemma4, text_rag + parametric
│   ├── phase1b/                 ← UC4, Gemma4 (v2 training)
│   ├── phase3/                  ← UC4, 3-mode (text_rag, kv_meanpool, parametric)
│   ├── phase4/                  ← Embedding comparison (bge-small vs mxbai-large)
│   ├── phase5/                  ← UC4, 3858 FAQs (large-scale)
│   └── uc1/                     ← Customer Support Gemma4
│
├── chat_4mode_tiers_*/          ← UC4, chat template, 4-mode × 5 tiers
├── phase3_4mode*/               ← UC4, system prompt, 4-mode
├── phase3_tiers*/               ← UC4, system prompt, tier sweep
├── fulltoken_sweep_*/           ← UC4, kv_fulltoken only sweep
├── cacheblend_sweep_*/          ← CacheBlend ratio sweep (UC4)
├── cacheblend_chat_*/           ← CacheBlend with chat template
├── ablation_prompt_*/           ← Prompt format ablation
├── ape_eval_*/                  ← APE calibration eval
├── mxbai_eval_*/                ← MXBAI embedding sweep (4-mode)
├── uc4_tier1000_v2/             ← UC4, 4-mode at N1000
├── uc4_tier2000_v2/             ← UC4, 4-mode at N2000
│
└── pathb_diversity/             ← Path B diversity training (Sprint 1)
    ├── uc4/                     ← UC4, 1x/5x/10x diversity variations
    └── 2wikimqa/                ← 2WikiMQA, 1x/5x/10x
```

---

## PHASE 0: Baseline (Llama-3.2-3B, deprecated)

### UC4 Bedrock — Phase 1 Absorption (Llama)

Tiers N500–N6000, 3 seeds, 2 modes. **All deprecated in favor of Gemma4.**

| Tier | Seeds | text_rag (avg FA) | parametric (avg FA) |
|:---:|:---:|:---:|:---:|
| 500 | 42–44 | 0.079 | 0.050 |
| 1000 | 42–44 | 0.084 | 0.051 |
| 2000 | 42–44 | 0.085 | 0.042 |
| 4000 | 42–44 | 0.076 | 0.040 |
| 6000 | 42–44 | 0.073 | 0.041 |
| **Overall** | | **~0.079** | **~0.045** |

Logs: `absorption_prompt_gpu1.log`, `ablation_system_prompt.log`

### PubMedQA Crossover (Llama — INVALID)

Stale KV caches from old Llama model. kv_meanpool result (FA≈0.003) is **invalid**.

| Tier | text_rag | kv_meanpool | parametric |
|:---:|:---:|:---:|:---:|
| 50 | **0.291** | 0.003 ❌ | 0.112 |
| 100 | **0.274** | 0.002 ❌ | 0.119 |
| 200 | **0.277** | 0.003 ❌ | 0.123 |

Logs: `pubmedqa_priority1.log`, `phase1_prep.log`

---

## PHASE 1: UC4 Bedrock — Core 4-Mode Comparison (Gemma4)

### 1a. System Prompt — 4 Mode × 5 Tiers

Full 4-mode (kv_fulltoken, kv_meanpool, text_rag, parametric) across FAQ tiers N500–N6000.
*FA scores use Option C partial judge — kv_meanpool and kv_fulltoken records not re-judged (unchanged).*

| Tier | kv_fulltoken | kv_meanpool | text_rag | parametric |
|:---:|:---:|:---:|:---:|:---:|
| 500 | 0.045 | 0.123 | **0.219** | **0.234** ◀ |
| 1000 | 0.045 | **0.123** | 0.198 | **0.228** ◀ |
| 2000 | 0.045 | 0.115 | 0.136 | **0.234** ◀ |
| 4000 | 0.045 | 0.117 | 0.116 | **0.234** ◀ |
| 6000 | 0.050 | 0.122 | 0.109 | **0.234** ◀ |
| **Avg** | **0.046** | **0.120** | **0.156** | **0.233** |

Logs: `phase3_4mode_tiers_gpu0.log`, `phase3_tiers_gpu0.log`

### 1b. System Prompt — 4 Mode v2 (higher kv_meanpool)

| Tier | kv_fulltoken | kv_meanpool | text_rag | parametric |
|:---:|:---:|:---:|:---:|:---:|
| 2000 | 0.045 | **0.151** | 0.106 | 0.113 |
| 4000 | 0.045 | 0.117 | 0.115 | 0.107 |
| 6000 | 0.050 | **0.127** | 0.122 | 0.113 |

Logs: `phase3_4mode_gpu0.log`

### 1c. Chat Template — 4 Mode × 5 Tiers

| Tier | kv_fulltoken | kv_meanpool | text_rag | parametric |
|:---:|:---:|:---:|:---:|:---:|
| 500 | 0.060 | 0.109 | **0.139** | 0.107 |
| 1000 | 0.060 | **0.118** | 0.112 | 0.107 |
| 2000 | 0.055 | 0.108 | 0.114 | 0.113 |
| 4000 | 0.055 | 0.119 | **0.154** | 0.113 |
| 6000 | 0.055 | 0.103 | **0.117** | 0.102 |
| **Avg** | **0.057** | **0.111** | **0.127** | **0.108** |

Logs: `chat_4mode_tiers_gpu1.log`, `chat_4mode_gpu0.log`

### 1d. Prompt Format Ablation (kv_fulltoken only, N500)

| Format | kv_fulltoken FA |
|:---|---:|
| system_prompt | 0.039 |
| chat_template | 0.049–0.060 |
| simple | 0.056 |

Logs: `ablation_gpu1.log`, `ablation_chat_template.log`, `ablation_simple.log`, `ablation_system_prompt.log`

---

## PHASE 2: Multi-Dataset Crossover (Gemma4)



### 2c. PubMedQA Gemma4 Crossover (text_rag + parametric only — NO kv_meanpool)

| Tier | text_rag | parametric |
|:---:|:---:|:---:|
| 50 | **0.229** | 0.201 |
| 100 | 0.191 | **0.207** ◀ |
| 200 | 0.187 | **0.203** ◀ |
| 297 | **0.195** | 0.190 |
| **Avg** | **0.200** | **0.200** |

Logs: `pubmedqa_crossover_tiers.log`, `pubmedqa_gemma4_v10.log`, `pubmedqa_indexer_v*.log`

### 2d. UC1 Customer Support (Gemma4 — text_rag + parametric only, NO kv_meanpool)

| Tier | Seeds | text_rag (avg) | parametric (avg) |
|:---:|:---:|:---:|:---:|
| 200 | 42–44 | 0.215 | **0.360** |
| 500 | 42–46 | 0.198 | **0.358** |
| 1000 | 42–44 | 0.192 | **0.356** |

Logs: `uc1_queue.log`, `uc1_seeds_45_46.log`

### 2e. HotpotQA (text_rag only)

| n | text_rag |
|:---:|:---:|
| 200 | 0.020 |

Logs: `hotpotqa_ec2a.log`, `hotpotqa_missing.log`

---

## PHASE 3: Embedding Ablation (UC4 Bedrock)

| Embedding | text_rag FA |
|:---|---:|
| **bge-small-en-v1.5** | **0.120** |
| mxbai-embed-large-v1 | 0.115 |

**MXBAI 4-mode sweep** (N500, n=89):

| Mode | FA |
|:---|---:|
| kv_fulltoken | 0.039 |
| **kv_meanpool** | **0.123** |
| text_rag | 0.094 |
| parametric | 0.096 |

Logs: `mxbai_gpu0.log`, `mxbai_v2_gpu1.log`, `mxbai_v3_gpu1.log`

---

## PHASE 4: Large-Scale FAQ (UC4, 3858 FAQs, Gemma4)

| Tier | Seeds | text_rag (avg) | parametric (avg) |
|:---:|:---:|:---:|:---:|
| 500 | 42–44 | 0.148 | 0.062 |
| 1000 | 42–44 | 0.144 | 0.060 |
| 2000 | 42–44 | 0.123 | 0.065 |
| 4000 | 42–44 | 0.141 | 0.057 |
| 6000 | 42–44 | 0.109 | 0.062 |

---

## PHASE 5: kv_fulltoken Deep-Dives (UC4)

### 5a. Fulltoken Tier Sweep

| Tier | kv_fulltoken FA |
|:---:|:---:|
| 500 | 0.045 |
| 1000 | 0.045 |
| 2000 | 0.045 |
| 4000 | 0.039 |
| 6000 | 0.050 |

### 5b. CacheBlend Ratio Sweep (system prompt)

| Ratio | kv_fulltoken FA |
|:---:|:---:|
| 0.0 | 0.045 |
| 0.1 | 0.039 |
| 0.25 | 0.045 |
| 0.5 | 0.045 |
| 0.75 | 0.045 |
| 1.0 | 0.045 |

### 5c. CacheBlend Ratio Sweep (chat template)

| Ratio | kv_fulltoken FA |
|:---:|:---:|
| 0.0 | 0.066 |
| 0.1 | 0.060 |
| 0.25 | 0.060 |
| 0.5 | 0.055 |
| 0.75 | 0.055 |
| 1.0 | 0.055 |

Logs: `cacheblend_gpu1.log`, `cacheblend_chat_gpu1.log`, `fulltoken_sweep*.log`

### 5d. APE Calibration

| Mode | FA |
|:---|---:|
| kv_fulltoken (APE) | 0.066 |

Logs: `ape_eval_gpu1.log`, `calibrate_ape_gpu0.log`

---

## PHASE 6: Path B Diversity Training (Sprint 1 — Llama 3.2 3B)

Path B = parametric-only evaluation with diversity-augmented training.

### UC4 Diversity

| Variation | FA |
|:---:|:---:|
| Path A baseline (text_rag) | 0.086 |
| Path B 1x | 0.056 (65.4% of A) |
| Path B 5x | 0.062 (71.9% of A) |
| Path B 10x | 0.062 (71.9% of A) |
| Gemma Path B 1x | (empty) |
| Gemma Path B 5x | (empty) |

### 2WikiMQA Diversity

| Variation | FA |
|:---:|:---:|
| Path B 1x | (empty) |
| Path B 5x | (empty) |
| Path B 10x | (empty) |
| Gemma Path B 1x | (empty) |
| Gemma Path B 5x | (empty) |

> **Conclusion (from log.md):** Diversity helps (+6.5pp 1x→5x) but plateaus at 5x. 3B parameter count is the bottleneck.

---

## PHASE 2: Multi-Dataset Absorption (Llama — deprecated)


### HotpotQA Absorption (parametric only)

| Tier | Seeds | parametric (avg) |
|:---:|:---:|:---:|
| 500 | 42–44 | 0.139 |
| 1000 | 42–44 | 0.141 |
| 1722 | 42–44 | 0.140 |

### 2WikiMQA Absorption (text_rag + parametric)

| Tier | Seeds | text_rag (avg) | parametric (avg) |
|:---:|:---:|:---:|:---:|
| 500 | 42–44 | 0.045 | 0.033 |
| 1000 | 42–44 | 0.039 | 0.029 |
| 1986 | 42–44 | 0.040 | 0.029 |

---

## Log File Index

### EC2-A (A10G) Logs

| Log | Experiment | Date | Size |
|:---|---|:---:|---:|
| `pubmedqa_priority1.log` | PubMedQA crossover priority 1 | Aug 7 | 290 KB |
| `pubmedqa_crossover_tiers.log` | PubMedQA Gemma4 tiers | Aug 9 | 267 KB |
| `pubmedqa_gemma4_v10.log` | PubMedQA Gemma4 v10 | Aug 8 | 53 KB |
| `pubmedqa_indexer_v*.log` | PubMedQA re-index (v4–v8) | Aug 8 | 5–20 KB |
| `hotpotqa_ec2a.log` | HotpotQA text_rag | Aug 6 | 45 KB |
| `phase3_4mode_tiers_gpu0.log` | UC4 Phase 3 tiers | Aug 6 | 216 KB |
| `phase3_tiers_gpu0.log` | UC4 Phase 3 tiers v2 | Aug 6 | 263 KB |
| `chat_4mode_tiers_gpu1.log` | UC4 chat 4-mode tiers | Aug 6 | 263 KB |
| `chat_4mode_gpu0.log` | UC4 chat 4-mode single | Aug 6 | 66 KB |
| `cacheblend_chat_gpu1.log` | CacheBlend chat template | Aug 6 | 106 KB |
| `cacheblend_gpu1.log` | CacheBlend system prompt | Aug 6 | 106 KB |
| `ablation_gpu1.log` | Prompt format ablation | Aug 6 | 53 KB |
| `fulltoken_sweep*.log` | kv_fulltoken tier sweep | Aug 6 | 66 KB |
| `mxbai_gpu0.log` | MXBAI embedding eval | Aug 6 | 60 KB |
| `uc1_queue.log` | UC1 Customer Support | Aug 6 | 4 KB |
| `uc1_seeds_45_46.log` | UC1 extra seeds | Aug 6 | 58 KB |
| `ape_eval_gpu1.log` | APE calibration | Aug 6 | 17 KB |
| `r4_e5_gpu0.log` | Rerotation E5 | Aug 6 | 1 KB |

### EC2-B (RTX PRO 4500) Logs — Released

| Log | Experiment | Date | Size |
|:---|---|:---:|---:|
| `uc2_queue.log` | UC2 PubMedQA Gemma4 (train OK, eval crashed) | Aug 5 | 4 KB |
| `phase2_setup.log` | Phase 2 dataset setup | Jul 28 | — |
| `phase2_sweep.log` | Phase 2 param sweep | Jul 28 | — |

---

## Key Findings Summary

1. **Partial judge (Option C)** was adopted — CORRECT(1.0)/PARTIAL(0.5)/INCORRECT(0.0) lifts FA scores by ~0.05–0.12 across most datasets. The PARTIAL category captures previously-zeroed answers.
5. **PubMedQA Gemma4** — with partial judge, parametric now marginally beats text_rag at N100–N200 (FA≈0.20). First time parametric crosses on PubMedQA.
6. **UC4 Bedrock** crossover shifted from N4000 (binary judge) → **N500** (partial judge) — the PARTIAL credit helped parametric more than text_rag.
7. **HotpotQA** and **2WikiMQA** remain near-floor (FA≈0.02–0.08) — multi-hop reasoning cannot be absorbed parametrically at this scale.
8. **Chat template** helps text_rag on UC4 but hurts kv_meanpool — prompt format matters for non-parametric modes.
9. **Path B diversity** showed diminishing returns: 1x→5x helps (+6.5pp), 5x→10x plateaus. 3B model confirmed as primary bottleneck.

---

# PHASE 7: Parametric Consolidation (September 2026)

The research reframed from "KV-cache latency" to **"an SLM progressively replaces RAG"**.
Full narrative and lever ranking: `docs/parametric_consolidation_roadmap.md`.
LoRA hyper-parameter study: `docs/sweep_findings/report.md`.

**Judge caveat for this whole phase:** `deepseek-v4-flash` scored ~2× more leniently than
`deepseek-v4-lite`. Numbers below are within-judge comparable only. Re-judge saved answers
with one consistent judge before comparing across sub-phases.

## 7a. LoRA rank & epoch sweep — closed-book QA (2026-09-07)

Model `google/gemma-4-E2B-it` (≈2B, 4-bit NF4 QLoRA), 337 QA pairs, metric `token-F1`.
Artifacts: `examples/usecase4_bedrock_userguide/lora_checkpoints/qa_*`.

| Arm | Rank | Epochs | exact F1 | para F1 | exact ref. |
|-----|-----:|-------:|---------:|--------:|-----------:|
| `qa_r16_ep3` | 16 | 3 | 0.159 | 0.155 | 0.31 |
| `qa_r16_ep6` | 16 | 6 | 0.167 | 0.159 | 0.31 |
| `qa_r16_ep9` | 16 | 9 | 0.167 | 0.164 | 0.31 |
| `qa_r16_ep12` | 16 | 12 | 0.169 | 0.163 | 0.33 |
| `qa_r32_ep3` | 32 | 3 | 0.162 | 0.167 | 0.31 |
| `qa_r64_ep3` | 64 | 3 | **0.170** | **0.167** | 0.28 |
| `qa_r128_ep3` | 128 | 3 | 0.170 | 0.163 | 0.33 |

**Findings:** ceiling ≈ token-F1 0.17. Epochs plateau after ~6 and overfit by 12. **r=64 is
the optimum** — r=128 matches on exact but loses paraphrase and raises refusals. Refusals
stuck at 0.28–0.33 regardless of rank/epochs (a separate, unsolved problem).

## 7b. Idea #1 — diverse rewrites + QA (2026-09-09 → 09-14)

The dominant lever. Training was *already* closed-book (question → answer, no context); the
fix was to diversify the *inputs*: ~10 QA + ~10 rewrites per chunk.

| Run | Recipe | FA |
|---|---|:---:|
| E2B r16/1ep (baseline) | QA only, n=89 | 0.147 |
| E4B r64/3ep | QA only, n=89 | 0.235 |
| **E4B r64, full corpus** | 60,209 QA + 61,554 rewrites (121,763 ex) | **0.235** (n=89) |
| E4B + diverse, ckpt sweep 9000→15222 | n=500 | 0.278 / 0.275 / 0.288 / 0.280 / 0.302 / 0.295 |
| E2B r64 @ 60k QA + rewrites | n=500 | 0.249 ± 0.012 |
| E4B @ 60k QA, no rewrites | n=500 | 0.244 ± 0.011 |
| Llama-3.2-3B-Instruct r64, same recipe | n=500 | 0.254 ± 0.012 |

**Findings:**
- Idea #1 gain is the **combination** of QA + rewrites — QA *volume* is inert (337 → 60k QA
  moved nothing: 0.235 → 0.244, within noise).
- The lever is **architecture-agnostic**: a smaller (3.2B) non-Gemma model matches the larger
  E2B (5.1B) on the same recipe. This overturned the earlier "architecture is the primary
  determinant" reading from the QA-only crossover data.
- **Rank-confound correction:** the "+0.088 FA scale-up" claim was confounded by adapter rank.
  At matched r=64 the true E2B→E4B capacity gap is **+0.025 (no rewrites) / +0.037 (with
  rewrites)**. Capacity is real but marginal.

## 7c. RAG baselines on the 500-Q set (2026-09-14)

| Arm | FA |
|---|:---:|
| E2B + RAG | 0.175 |
| E4B + RAG | 0.161 |

Trained parametric strictly dominates retrieval on both accuracy **and** latency on UC4.

## 7d. Generalization replications (2026-09-15 → 09-16)

Full Idea #1 recipe (~10 QA + ~10 rewrites/chunk → E2B r64 → 500-Q holdout) on:
- **PubMedQA** — 2,918 chunks (~29k QA + ~29k rewrites)

Known confound: the rewrite generator is `deepseekv4p`, not Claude Haiku 4.5 as in the UC4
runs (Anthropic key exhausted; CodeVector gateway has no Haiku). Rewrite-generator quality is
an open axis.

## 7e. PubMedQA yes/no/maybe — E2B finetuning vs RAG (2026-09-17 → 09-18)

**The headline negative result.** Corrected evaluation: all 1,000 `pqa_labeled` records,
exact string match on `final_decision`, **no LLM judge**. Artifacts on EC2:
`results/e2b_vs_rag/`.

| metric | E2B finetuning | RAG (text_rag) |
|---|---:|---:|
| n | 1000 | 1000 |
| **accuracy** | **0.3650** | **0.6800** |
| macro-acc | 0.3447 | 0.6374 |
| acc(yes) | 0.0725 | 0.7482 |
| acc(no) | 0.9615 | 0.6095 |
| acc(maybe) | 0.0000 | 0.5545 |
| latency mean | 0.319 | 0.396 |
| latency median | 0.313 | 0.390 |
| latency p95 | 0.330 | 0.457 |

**RAG wins on accuracy.** Finetuning wins only on latency (no retrieval round-trip). The
finetuned model collapses to a near-reflexive "no" (96 % on `no`, 7 % on `yes`, 0 % on
`maybe`). A rationale pilot (`logs/pilot_rationale.log`, n=50) confirmed this: judge accuracy
0.16, with the judge overriding the model on 29/50 records but correcting only 13.8 % of those.

**Implication:** consistent with the roadmap's honest floor — full "replace RAG" is
contradicted at this scale. The defensible claim is *progressive consolidation with retrieval
as a shrinking fallback*. Do not overstate the crossover.

## 7f. Also present on EC2 (not yet fully catalogued locally)

`results/three_arm/`, `results/tiered_crossover_test/`, `results/pubmedqa_gemma4_r4_e15/`,
`results/e2b_rejudge_lite/`, and the `eval500_*` / `idea1_*` / `rag500_*` families.
~30 result directories exist on EC2 that are **not** in the local working copy.

## 7g. Recorded but NOT started

Two experiments testing rewrite-generator quality by having the small models generate their
own rewrites: **E2B self-rewrites** and **E4B self-rewrites**. Status: recorded only — see
§5 of `docs/parametric_consolidation_roadmap.md`. Do not start without the user's go-ahead.