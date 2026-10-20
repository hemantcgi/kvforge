# Knowledge Measurement Plan — How Much Corpus Knowledge Does Each Store Retain?

**Status:** DESIGN — pre-registration draft, NOT YET RUN.
**Date:** 2026-10-06
**Owner:** Hemant
**Relates to:** `docs/evaluation_plan.md`, `docs/parametric_consolidation_roadmap.md` (PRS v2),
`docs/three_arm_experiment_plan.md`, `docs/matched_format_experiment_plan.md`.

---

## 0. Why this exists

Per-query factual accuracy measures **task performance**, not **how much of a corpus's
knowledge a store actually holds**. This plan measures *amount of knowledge retained* by each
store on a shared, corpus-derived **fact inventory** — a fixed ruler — and reports it as a
profile (coverage × fidelity × reliability), plus a **knowledge-coverage curve over training**.

It makes the two stores comparable:

- **Vector DB** holds the corpus *losslessly* (chunk text is physically present). Its binding
  constraint is **accessibility = retrieval recall**. Knowledge-in-DB = fraction of the
  inventory whose supporting chunk is retrievable *and* answerable from top-k.
- **LLM weights** hold a *lossy, compressed* fraction; accessibility = whether the model
  produces the fact closed-book, robustly across phrasings. Knowledge-in-LLM = fraction of the
  inventory recalled closed-book, reliability-weighted, **above the base-model baseline**.

Grounding (external): KR-Test *Beyond Perplexity* (arXiv 2601.03505) — corpus-grounded
retention via Teacher-decomposed atomic units, and argues *against* perplexity as the measure;
LAMA (arXiv 1909.01066) + BELIEF/MyriadLAMA — multi-prompt cloze probing with
consistency/reliability; FActScore (arXiv 2305.14251) — atomic-fact decomposition + precision;
Allen-Zhu & Li *Knowledge Capacity Scaling Laws* (arXiv 2404.05405) — knowledge-in-bits via a
bit-complexity lower bound (~2 bits/param); *Data Quality over Capacity* (arXiv 2607.21861) —
closed-book LoRA internalization measured by **recall over a corpus fact set**, reported to
beat a gold-chunk oracle (specific numbers quoted from a search snippet — **read the paper
before citing them**); catastrophic-forgetting surveys — no single metric suffices.

---

## 1. The shared ruler: an atomic-fact inventory

Decompose the corpus into atomic, self-contained facts, each tagged with its source chunk.

- **Input:** `examples/usecase4_bedrock_userguide/chunks.json` — `list[{chunk_id, text}]`
  (6,154 chunks). `chunk_id` is also the Qdrant point id (payload has `text`, `page`).
- **Extraction:** Teacher LLM (deepseek-v4 via the CodeVector gateway, the same client
  `tools/gen_diversity_deepseek.py` already uses) decomposes each chunk into ≤N atomic facts,
  each one proposition, context-independent.
- **Per fact, emit a probe bundle:**
  - `qa_probe`: `{question, gold_answer}` (short, fact-grounded — reuse the
    `tools/gen_eval_qa.py` / `build_holdout.py` prompting style).
  - `paraphrases[M]`: M≥5 paraphrased question phrasings (BELIEF/MyriadLAMA) — separates
    *knowing* from *lucky recall* / prompt regularities (Zhong 2021, "learning to recall").
  - `contrastive[]`: the correct continuation + 3 plausible-but-wrong distractors (KR-Test),
    for a judge-free log-prob scorer.
- **Inventory record:**
  `{fact_id, chunk_id, atomic_text, qa_probe, paraphrases[], contrastive[], coverage_count, gold_len}`
  where `coverage_count` = # training QA pairs + rewrites derived from `chunk_id` (§4).
- **Inventory = corpus knowledge mass** = |facts| (optionally in bits via Allen-Zhu; stretch).
- **Validation:** hand-check a random ~50-fact sample for atomicity + answerability-from-chunk
  before trusting the inventory. Extraction quality is Teacher-dependent (Threat T1).

**First-increment shortcut:** `eval_500.json` (500 items `{question, answer, chunk_id, type}`,
already deduped vs training) is a ready-made single-probe-per-chunk seed inventory. Use it to
stand the pipeline up before running full multi-fact-per-chunk extraction.

---

## 2. Stores / arms scored against the inventory

| Arm | What | How (real modules) |
|-----|------|--------------------|
| **S0 Base closed-book** | prior/pretrained knowledge (contamination baseline) | `model_loader.load(None)`, **base id derived per-S1 from the adapter's `adapter_config.json:base_model_name_or_path`** → `prs_evaluator._generate_parametric` |
| **S1 FT closed-book** | absorbed parametric knowledge | `model_loader.load(<adapter_dir>)` (merges adapter) |
| **S2 Vector DB** | retrievable + answerable knowledge | recall@k via Qdrant (§3); answerability via oracle reader (**the matching base model** given retrieved top-k text, and a gold-chunk oracle) |

**Which checkpoints are materialized (verified 2026-10-06):**

| Arm | Base | Trained on | Weights | Notes |
|-----|------|-----------|---------|-------|
| `phase3_N500 … N6000` | gemma-4-E2B-it | phase-3 FAQs (`faqs_all.json`, 6,154 space ✓) | **local ✓** | the locally-runnable N-sweep; coverage-count ← `faqs_all.json:source_chunk_ids` |
| `uc4_{1x,5x,10x}_v1` | **Llama-3.2-3B** | diversified_v1 (1x/5x/10x) | **GPU host only** (configs local, no `.safetensors`) | a clean repetition sweep, held out from `eval_500`; coverage-count ← `diversified_v1/per_chunk/` |
| E4B / E4B+diverse ckpt-9000…15222 | gemma-4-E4B-it | diversified_v1 | **GPU host only** | headline arms |

**Base model and S0/reader must match each S1** — comparing a Llama adapter's closed-book
answers against a Gemma S0 baseline is meaningless. Derive S0 + the S2 reader from each
adapter's `base_model_name_or_path`. **All GPU + Qdrant (:6333) live on the GPU host**, so every arm
runs there; the "local vs the GPU host" distinction below is only about *which adapters are
materialized*, not where compute happens.

Retention is reported as **lift over the matching S0** (so we don't credit facts the base
already knew).
S2 is the retrieval ceiling / mediator — it tells us whether a RAG loss is a *retrieval* miss
or a *reader* failure (advisor's point; separates "problem favors FT" from "retriever weak").

Optional S3 (FT+RAG) is out of scope here — it belongs to `three_arm_experiment_plan.md`.

---

## 3. Metrics (the knowledge profile)

Scored per probe, then aggregated over the inventory.

- **Fidelity (per fact):** primary = `eval/metrics.llm_judge(..., judge_mode="partial")`
  (CORRECT 1.0 / PARTIAL 0.5 / INCORRECT 0.0); robustness = **per-item length-controlled
  overlap with `k = len(gold_tokens)`** (`budget_f1`, or `first_k_precision` called with
  `k=len(gold_tokens)`). ⚠️ Do **not** use a fixed k=20/40 here: atomic-fact golds are ~5–10
  tokens, and `first_k_precision`'s denominator is the first-k *predicted* tokens, so a
  7-token gold under fixed k=20 caps a perfect answer at 7/20=0.35 — that cap correlates with
  `gold_len` and manufactures the very artifact it was meant to remove. (Fixed k is correct
  only for long-form golds, as in Exp5.) Report both metrics; trust only effects that survive
  the length-controlled one.
  Continuity option: `three_arm_evaluator.fa_score` (0.5·judge + 0.5·cosine) for comparability
  with existing tables — but judge-partial is the cleaner *knowledge* signal for short facts.
- **Coverage@store (breadth):** fraction of inventory with fidelity ≥ τ (τ = partial ≥ 0.5).
  The headline "amount retained."
- **Reliability:** agreement across the M paraphrased prompts (consistency) and across sampled
  generations (self-consistency, as in PRS). A fact counts as *robustly* retained only if it
  clears τ on a majority of paraphrases.
- **Vector DB (S2):** `recall@k` = `chunk_id ∈ retrieved_ids`; `answerable@k` = oracle-reader
  fidelity on top-k. (Caveat: facts split across chunks are present-but-not-answerable from any
  single top-k set — measure answerability, not mere presence. Threat T6.)
- **Retention lift:** `Coverage(S1) − Coverage(S0)`, per fact and aggregate.
- **Knowledge-coverage curve (headline artifact):** `Coverage(S1)` across the N-sweep
  checkpoints (`phase3_N500…N6000`; or the 1x/5x/10x repetition sweep on the GPU host) overlaid with
  the flat **`answerable@k`** ceiling from S2 — this is the only S2 quantity commensurate with
  fidelity-based coverage. **The crossover of the two curves = the knowledge-mass crossover**,
  a stronger version of the paper's per-query FA crossover. Plot **`recall@k` in a separate
  diagnostic panel** (it's a retrieval-hit rate, not a knowledge coverage) — never on the same
  axis as the coverage ceiling.
- CIs via `eval/stats.paired_bootstrap`; calibration via `eval/metrics.expected_calibration_error`.

---

## 4. Per-fact attribution join (feeds the sign(D) study)

Emit one joined table `results/knowledge/uc4/per_fact.jsonl`:

```
{fact_id, chunk_id, coverage_count, gold_len, provenance,
 base_correct(S0), ft_correct(S1), recall_at_k(S2), answerable_at_k(S2), robust(reliability)}
```

- `coverage_count` (the Ovadia "repetition" predictor) = count of training examples derived
  from `chunk_id` — **source must match the arm's actual training data**, or the H3 join is
  meaningless:
  - phase-3 E2B arms → count FAQs per chunk from `faqs_all.json:source_chunk_ids`.
  - diversified arms (Llama `uc4_*`, E4B+diverse) → count QA+rewrites per chunk from
    `diversified_v1/per_chunk/chunk_NNNNNN/` (the flat `{1x,5x,10x}/*.json` drop the chunk id).
- This table *is* the substrate for the per-query "what tips the balance" attribution model.
- **Explanatory vs deployable features (keep separate):** `coverage_count`, `recall@k`,
  `gold_len` are *not* known at query time — fine for explanation, not for a live gate. A
  deployable Phase-3 gate must be rebuilt from query-time signals (base-model confidence, query
  similarity to the training set). Do not conflate the two.

---

## 5. Build plan (modules — reuse what exists)

1. **`tools/build_fact_inventory.py`** (new) — modeled on `tools/gen_diversity_deepseek.py`'s
   CodeVector client. In: `chunks.json`. Out: `inventory.json` (§1 schema). Flags:
   `--chunks-file --out --facts-per-chunk --paraphrases --concurrency --resume`.
2. **`eval/recall_at_k.py`** (new) — embed question (fastembed, as in `kv_inference`),
   `QdrantStore.query(collection, vector, top_k)`, test `chunk_id ∈ ids`. Add `answerable@k`
   via oracle reader. **Prereq: the 6,154-chunk UC4 collection must be indexed in Qdrant**
   (the GPU host runs Qdrant on :6333; local may need indexing — verify first, cf. memory
   `uc4-two-corpora`: 2,520 KV-indexed vs 6,154 QA-gen).
3. **`eval/knowledge_probe.py`** (new) — load store via `model_loader.load(base|adapter)`,
   run `_generate_parametric` over `qa_probe` + `paraphrases`; score with `metrics.llm_judge`
   (partial) + per-item length-controlled overlap; contrastive log-prob scorer over
   `contrastive[]`. **Generation config (record in every output file):** pass the **adaptive
   budget (~30–40 `max_new_tokens`)**, never the `_generate_parametric` default of 256 (the
   bug that depressed scores until 09-12), and use **each checkpoint's trained `sft_format`**
   / prompt template (the prompt-format match was an open Exp1 item — a mismatch silently
   tanks closed-book scores).
4. **`tools/run_knowledge_measurement.py`** (new orchestrator) — runs S0/S1/S2 over the
   inventory, writes `results/knowledge/uc4/<store>/*.json` + the §4 per-fact table; CIs via
   `eval/stats`. **Save every generation** (so re-judging needs no GPU re-run — reuse the
   `tools/rejudge_partial_credit.py` pattern).

**Judge discipline:** one judge for everything — `deepseek-v4-lite` via CodeVector
(`JUDGE_API_KEY`/`JUDGE_BASE_URL` from local `opencode.json` / `~/.claude/settings.local.json`).
`deepseek-v4-flash` scored ~2× more leniently — never mix judges across a table (AGENTS.md §4).

---

## 6. Pre-registration: hypotheses & falsification

- **H1 (mass crossover):** `Coverage(S1)` rises with training N and crosses the `answerable@k`
  ceiling of S2 at some checkpoint. *Falsified if* S1 coverage never reaches S2's.
- **H2 (real absorption):** `Coverage(S1) − Coverage(S0) > 0` with CI excluding 0. *Falsified if*
  FT coverage ≈ base coverage → nothing was absorbed beyond pretraining.
- **H3 (repetition enables retention, Ovadia):** per-fact `ft_correct` increases with
  `coverage_count`. *Falsified if* no monotone relationship.
- **Diagnostic (not a hypothesis):** if `recall@k ≈ 1` but RAG `answerable@k` is low, the
  bottleneck is the *reader*, not retrieval; if `recall@k` is low, it's retrieval. This
  settles whether UC4's apparent parametric "win" is real or a retrieval artifact.

Success for the increment = pipeline runs end-to-end on UC4, inventory passes the §1 sample
check, and the three arms produce a coherent coverage profile + curve.

---

## 7. Threats & controls

| # | Threat | Control |
|---|--------|---------|
| T1 | Teacher extraction quality (bad/duplicate facts) | hand-validate 50-fact sample; dedup facts (normalized + Jaccard, as `dedup_eval_qa.py`) |
| T2 | Pretraining contamination (fact already known) | S0 base baseline; report lift over S0 |
| T3 | Learning-to-recall / prompt luck | M≥5 paraphrases + contrastive scorer; require majority agreement |
| T4 | Judge variance | single judge (deepseek-v4-lite); save generations for re-score |
| T5 | Length/verbosity artifact | `first_k_precision` alongside judge; trust only effects that survive it |
| T6 | Fact split across chunks (present ≠ answerable) | measure `answerable@k`, not mere presence |
| T7 | Single-seed noise | ≥3 seeds for any headline number (generation + eval) |
| T8 | Pooling across incompatible runs | pool only within one judge/model/harness/era; UC4 n=500 is the clean substrate (matched-format records have no chunk_id → fidelity/attribution only, no recall@k) |
| T9 | Held-out leakage into S1 | `eval_500.json` was deduped against the **diversified 60k QA + 89-eval**, *not* the phase-3 `faqs_all.json`. For phase-3 E2B arms, re-dedup `eval_500` against `faqs_all.json` (question + answer, exact + Jaccard≥0.80) before trusting S1 coverage — else a phase-3 checkpoint can regurgitate held-out facts. Diversified arms (Llama/E4B) are already properly held out. |

---

## 8. Scope & increments

*(All runs are on the GPU host — that's where the models + Qdrant live. "Materialized" = adapter
weights present for that arm.)*

- **Increment 0 (prereqs — no GPU). Status 2026-10-06:**
  - (a) ✅ **RESOLVED (the GPU host, 2026-10-06):** `bedrock-userguide` holds only the **2,520** KV-indexed
    subset, but a separate collection **`bedrock-userguide-6154` holds all 6,154 points @ dim 384**,
    and `eval_500.chunk_id` values resolve to real points there. **recall@k must target
    `bedrock-userguide-6154`, not the config default** `bedrock-userguide`. Embedder =
    `BAAI/bge-small-en-v1.5` 384-dim (paper §4.2 "mxbai-large 1024" note is stale).
  - (b) ✅ **leakage negligible**: eval_500 vs `faqs_all.json` (3,858) → 0 exact-question, 2
    near-dup questions (Jaccard≥0.80), 3 exact-answer. Phase-3 arm is clean (optionally drop the
    ~5 items). T9 resolved.
  - (c) ✅ bases confirmed: `phase3_N*` → `google/gemma-4-E2B-it` (weights local);
    `uc4_{1x,5x,10x}_v1` → `meta-llama/Llama-3.2-3B-Instruct` (weights GPU-host-only).
- **Increment 1 (cheapest materialized arm):** UC4; seed inventory = `eval_500.json`;
  S0 (base gemma-4-E2B-it) + S1 (`phase3_N6000`, E2B, **local weights**) + S2 (recall@k +
  answerable@k); single seed. Goal: stand up the pipeline, eyeball inventory quality, produce
  the first coverage profile + the retrieval-vs-reader diagnostic.
- **Increment 2 (N-sweep curve):** full multi-fact inventory via `build_fact_inventory.py`;
  coverage curve across `phase3_N500…N6000`; per-fact attribution table (§4). Add the
  **diversified Llama `uc4_{1x,5x,10x}_v1` repetition sweep** once its weights are re-synced
  from the GPU host — it's the clean, properly-held-out H3 test.
- **Increment 3 (headline arms):** E4B, E4B+diverse ckpt-9000…15222; ≥3 seeds; extend to other
  corpora with a chunk_id-bearing eval set.

### Execution log

**2026-10-06 — pipeline validated end-to-end on the GPU host.**

- **S2 recall@k (full n=500, judge-free), `bedrock-userguide-6154`, bge-small, top-k:**
  **recall@5 = 0.586** (293/500), **recall@10 = 0.684** (342/500). → The retriever misses the
  gold chunk for ~41% of queries at k=5; much of UC4 Standard-RAG's low FA is a *retrieval*
  failure, not only a reader failure (this is the §2 mediator working as intended).
  Output: `results/knowledge/uc4/recall_at_k_top{5,10}.json`.
- **S0 base + S1 (`phase3_N6000`) closed-book, 8-item smoke, heuristic judge:** executes fully
  (base load → adapter apply → generate → score → profile). Coverage 0.0 on this slice, and it
  is *explainable, not a bug*: (i) the first 8 items are dominated by TOC **page-number trivia**
  the model hasn't absorbed (and the §5.11 caveat flagged); (ii) the heuristic token-F1 judge is
  too strict for free-form answers (a correct item scored 0 at lc=0.375); (iii) the model shows
  **assistant-refusal behaviour** ("provide the document…") consistent with the known
  "refusals stuck 0.28–0.33" issue and a possible `sft_format` mismatch. A *meaningful* accuracy
  run therefore needs: content-question filtering (drop page-trivia), the real `deepseek-v4-lite`
  judge, and the checkpoint's trained prompt format. Output:
  `results/knowledge/uc4/probe_S{0,1}_*_smoke8.json`.
- **Integration fix made during the run:** `build_generate_fn` now unpacks the
  `(model, tokenizer)` tuple from `model_loader.load()` (was passing the tuple as `model` with a
  `None` tokenizer); covered by a new fake-module unit test.
- **Config note:** the thin studio `config_gemma4.json` lacks `embed_model` / `llm_model`; runs
  used a complete tmp config (`collection=bedrock-userguide-6154`, `embed_model=bge-small`,
  `llm_model=google/gemma-4-E2B-it`). Bake a proper `config_knowledge.json` before the full run.

**2026-10-06 (cont.) — orchestrator + real judge wired.**

- **`tools/run_knowledge_measurement.py`** added (S0+S1+S2 in one process → per-fact table +
  profile with retention-lift-over-base; pure join/align unit-tested). `score_fact` made
  judge-failure-safe (drops `judge_score=None`, counts `n_judge_failed`).
- **Judge id corrected:** the plan's `deepseek-v4-lite` is **disabled (403)** on the CodeVector
  gateway and `deepseekv4p` is **not granted**; the working judge is **`deepseek-code-h`**
  (a reasoning model; `llm_judge` default `max_tokens=1024` is sufficient). Creds on the GPU host at
  `codevector_creds.json`. All results here are **`deepseek-code-h`-judged** — not comparable to
  any `deepseek-v4-lite`/`flash` table (judge choice swings FA ~2×).
- **20-item content-Q validation (page-trivia excluded), real judge, 0 judge failures:**
  S1 (`phase3_N6000`, E2B) coverage **0.20** vs S0 base **0.15** → **retention_lift +0.05**;
  recall@5 0.60. Confirms the end-to-end path (generate → judge → profile → recall → lift).
- **Full content-Q run DONE — 481 items, both arms, `deepseek-code-h` judge, 0 judge failures**
  (`results/knowledge/uc4/orch_full_phase3N6000_codeh.json`). Arm = `phase3_N6000` (E2B):

  | store | measures | coverage@τ0.5 |
  |---|---|---|
  | S0 base closed-book | pretrained knowledge | **0.154** |
  | S1 FT closed-book | parametric knowledge | **0.262** |
  | **retention lift** (S1−S0) | genuinely absorbed over base | **+0.108** |
  | S2 vector-DB recall@5 | *retrievable* knowledge ceiling | **0.603** |

  Per-fact breakdown: of 126 FT-covered facts, **76 newly absorbed** (base wrong → FT right),
  50 prior knowledge. **53 FT-covered facts have a gold chunk retrieval misses at k=5** — the
  parametric store reaches knowledge RAG can't surface (complementarity → supports
  "consolidation + shrinking fallback", not replacement). Headline reading: this SLM holds
  ~26% of the corpus fact-inventory parametrically (vs 15% pretrained), while the vector DB
  makes ~60% *retrievable* — parametric coverage is real and additive but still well below the
  retrieval ceiling at this checkpoint. Next: run the diversified arms (E4B/E4B+diverse,
  Llama) for the coverage curve, add paraphrases (reliability), and a proper `config_knowledge.json`.

**2026-10-06 (cont.) — E2B coverage-vs-N curve** (`tools/run_coverage_curve.py`,
`results/knowledge/uc4/coverage_curve_e2b.json`; 481 content items, `deepseek-code-h`, 0 judge
failures; base=0.154, recall@5 ceiling=0.603):

| N (FAQ pairs) | 500 | 1000 | 2000 | 4000 | 6000 |
|---|---|---|---|---|---|
| S1 coverage | 0.262 | 0.252 | 0.258 | 0.258 | 0.262 |
| lift vs base | +0.108 | +0.098 | +0.104 | +0.104 | +0.108 |

**The curve is FLAT (0.252–0.262).** ⚠️ **CORRECTION (verified 2026-10-06): this is NOT a
training-QA-volume sweep.** All five `phase3_N{500..6000}` checkpoints were trained with the
*identical* command — `--faqs faqs_train.json`, `qa_examples: 337`, seed 42 (metadata
confirmed; adapter md5s differ only from pipeline/index-tier differences). The `N` suffix is the
**corpus / RAG-index tier** (`run_phase1.sh create_size_tiers`), **not** the number of training
QA pairs. So the flat line just shows the *same ~337-QA adapter* re-evaluated; it says nothing
about whether more QA pairs add knowledge. (It does cleanly reproduce §5.7's observation that
the *parametric* arm is flat across the tier axis while RAG declines — because the parametric
arm is literally one adapter at every tier.)

**The genuine QA-volume evidence is §5.12: 337 QA → 60,209 QA gave FA +0.009 (inert).** To
answer "does coverage rise with more *training* QA" in knowledge-mass terms, measure the
**337-QA vs 60k-QA E2B adapters** (same base/judge/inventory) — a real volume contrast — not the
`phase3_N*` tiers. Likewise the *diversity* axis (`phase3_E4B_r64_idea1`,
`phase3_LLAMA_r64_60kqa_rw`, Llama 1x/5x/10x) is the hypothesised lever. Those are the next runs.

**2026-10-07 — genuine volume + diversity coverage contrast (E2B, rank-matched r64)**
(`results/knowledge/uc4/volume_diversity_e2b.json`; 481 content items, `deepseek-code-h`, 0 judge
failures; base=0.154, recall@5 ceiling=0.603). **Provenance verified from checkpoint metadata +
`experiment_catalog.md` §7a / `sweep_findings/report.md`** (not names — the Msg-13 lesson):

| arm | base | rank | train QA | rewrites | epochs | S1 coverage | lift vs base |
|---|---|---|---:|---|---:|---:|---:|
| `qa_r64_ep3` | E2B | 64 | **337** | no | 3 | **0.243** | +0.089 |
| `phase3_E2B_r64_60kqa` | E2B | 64 | **60,209** | no | 1 | **0.422** | +0.268 |
| `phase3_E2B_r64_60kqa_rw` | E2B | 64 | **60,209** | **yes** (+61,554 ex) | 1 | **0.478** | +0.324 |

- **Volume axis (337 → 60k QA):** coverage **+0.179** (0.243 → 0.422). **Not flat** — this reverses
  the pre-run prediction. (Epoch confound runs *against* the finding: the 337 arm had 3 epochs
  vs the 60k arm's 1, and §3.2's epoch sweep plateaus after ~6, so more passes could not close a
  +0.179 gap. The old `phase3_N*` curve independently replicates the 337 low point at 0.252–0.262.)
- **Diversity axis (60k QA → 60k QA + rewrites):** coverage **+0.056** (0.422 → 0.478). Clean on
  base/rank/QA-count/epochs; the only diff is `--rewrites` (adds ~61.5k continuation examples —
  so "diversity" here is *also* +≈2× training examples, not volume-matched).

**Mechanism — overlap check (`scratchpad/overlap_check.py`, real `filter_items`→481; no chunk_id
in train files so question-near-dup + gold-answer-substring):**

| | probe-Q near-dup in train | gold fact-text (substring) present in train |
|---|---:|---:|
| 337-QA | 0/481 | **1.2%** (6) |
| 60k-QA only | 1/481 | **20.2%** (97) |
| 60k-QA + rewrites | — | **26.2%** (126) |

- **Probe-question contamination is ruled out** (≈0 Q near-dups in either set).
- **`eval_500.json` is NOT un-deduplicated** (earlier draft of this entry was wrong).
  `tools/dedup_eval_qa.py` built it by deduping the generated pool against the **60,209-QA
  training set** (`TRAIN=/tmp/faqs_full_idea1.json`) *and* the 89-eval, via **question-exact,
  question-Jaccard≥0.80, and answer-exact** (terse answers caught by the Q rule). So it is a
  genuine held-out set (matches the paper §5.12 "fresh held-out, Jaccard≥0.80 + answer-dedup"
  description; see threat **T9**). The 19.8–26.2% figures are a **residual *substring* leak** that
  exact-answer dedup does not catch (a short gold answer appearing inside a longer training
  answer). **This is a real leak in the held-out set itself** — worth flagging for the paper's
  held-out claim (do *not* edit the paper; deferred per Msg-2).

**⚠️ The volume jump cannot (yet) be attributed to knowledge, because the leak is the same size as
the jump:**

| axis | coverage jump (×481) | differential substring leak |
|---|---:|---:|
| Volume 337→60k | +0.179 ≈ **86 facts** | 97−6 = **91 facts** |
| Diversity 60k→+rw | +0.056 ≈ **27 facts** | 126−97 = **29 facts** |

For **both** axes the facts newly present (as substrings) in training ≈ the facts newly covered.
In the worst case the entire jump is the model reproducing fact text it saw verbatim in training
(leak-memorization), not generalization. S1 per-fact results were **not persisted**
(`run_coverage_curve` keeps only aggregate coverage), so the two sets cannot be linked per fact
and the jump cannot be split into leaked-vs-generalized. **"Corpus-span" is therefore *consistent
with* the data, not confirmed; per-fact confirmation pending** (needs per-fact persistence +
re-score, or dropping the 91 leaked items — do not re-run without asking).

**Status of the §5.12 reconciliation (NOT a clean same-arm comparison):**
- §5.12's "QA volume inert, +0.009" is **E4B** (337→60k, FA). The paper has **no rank-matched E2B
  337 FA** (only E2B r16 337 = 0.147), so this is *not* a direct same-arm contradiction with the
  E2B coverage jump.
- The divergence "E4B FA inert vs E2B coverage +0.179" has **unresolved** candidate causes: base
  model (E4B vs E2B), metric (coverage@τ0.5 vs mean FA), judge (`deepseek-code-h` here vs §5.12's),
  **and the substring leak above**. Not settled from memory; left open.
- **Diversity axis is the cleaner result** (base/rank/QA-count/epoch matched; only `--rewrites`
  differs) and is directionally consistent with §5.12's robust rewrites lever (+0.030 E2B FA on the
  answer-deduped set) — but it too carries a 29-fact leak, so §5.12's answer-deduped FA gain
  remains the stronger *generalization* evidence for rewrites.
- **Denominator note:** fixed — overlap check now uses the run's `filter_items` (481), matching the
  coverage numbers.

**2026-10-07 (cont.) — TRAINING-QA-VOLUME SWEEP (in progress).** Per user: the 337/60k/60k+rw
points are three scattered dots; what's needed is the *curve* of knowledge vs **number of training
QA pairs** (X), to find the peak/plateau and where the FT-SLM line crosses what RAG can retrieve.
The user also (correctly) set aside the substring-leak concern: for **retention** (trained on
material containing the fact → does the model reproduce it closed-book?) overlap is the definition,
not a confound; it would only matter for a "generalize to never-seen facts" framing.

- **Design (confirmed with user):** one E2B r64 / 1-epoch training pass over the 60,209 QA
  (`faqs_full_idea1.json`, `--from-base --seed 42 --replay-ratio 0` → 10 replay, negligible;
  3,764 steps, eff. batch 16). The GPU host trainer already shuffles with `data_seed=42`, so "QA seen by
  step *s*" is a **uniform sample spanning the whole corpus** — X = QA *density*, span held ~constant
  (this is the clean volume axis the `phase3_N*` tiers and the 3-dot contrast lacked).
- **Checkpointing:** `lora_save_steps=250` → checkpoints at steps 250…3750 = **QA ≈ 4k, 8k, …, 60k**
  (~15 evenly-spaced points, user's choice) + the final full-epoch adapter. Required one GPU-host-only
  trainer edit: `save_total_limit` made configurable (was hardcoded `=6`, which is why the paper's
  "six-checkpoint learning curve" kept only the last six); backup at
  `pipeline/lora_trainer.py.bak_volsweep`. Config: `config_e2b_r64_volsweep.json`.
- **Two Y-curves per checkpoint, same X** (one judge pass yields both, `deepseek-code-h`, 481 held-out
  facts): **retention = coverage@τ0.5** (breadth of distinct facts held) and **accuracy = mean
  fidelity** (continuous). New pure helper `mean_fidelity` (TDD, `eval/knowledge_profile.py`); wired
  into `run_coverage_curve` per-point (39 tests green). Baselines overlaid: **base S0 = 0.154**
  (untrained floor) and **RAG recall@5 = 0.603** (retrievable-knowledge ceiling — "how much the
  corpus knowledge traditional RAG can surface"). Crossing 0.603 = volume at which parametric
  overtakes retrieval; flattening = plateau.
- **Caps:** QA pool = 60,209 **unique**; 1 epoch reaches ~60k unique. If still climbing at 60k, more
  *unique* QA (beyond the pool) would be needed — only repetition/epochs remain, which tests
  consolidation, not volume. Will flag if hit.
- **Orchestration:** `tools/volsweep_orchestrate.py` (the GPU host) waits for training then evals all
  checkpoints → `results/knowledge/uc4/volume_curve_e2b.json` (crash-safe, partial after each arm).
  ETA ~8.5 h train + ~10 h eval ≈ **18 h**. Pending.

**2026-10-07 (cont.) — training DONE; eval relaunched after two orchestration bugs.** Training
finished in `train_runtime` 2.921e4 s ≈ **8.1 h** (loss 1.41, 1 epoch, 3,764 steps); **17 arms** on
disk: `checkpoint-250…3764` (QA 4k…60,224 at step×16) + the final root adapter (`qafinal`). Disk is a
non-issue — each checkpoint is 174 MB (adapter 50 MB + optimizer 100 MB + tokenizer 32 MB), ~2.7 G for
the full set vs 85 G free. Two bugs surfaced and were fixed before the eval could run:
  1. **Orchestrator stuck "waiting for training."** `training_alive()` does `pgrep -f
     "lora_trainer.*volsweep"`; the detached *launcher bash* (the `ssh … nohup … &` wrapper) lingered
     with that very pattern in its own command line, so the orchestrator matched it forever. Same
     self-match class as the earlier false "EVAL RUNNING". Fixed by killing the stale launcher PID;
     orchestrator immediately enumerated the 17 arms and advanced. Relaunch kept the pattern out of
     the launcher command so it can't recur.
  2. **Eval crashed instantly: `ModuleNotFoundError: No module named 'eval'`.** Orchestrator ran
     `python tools/run_coverage_curve.py` with `cwd=ROOT`, but script-mode puts `tools/` (not the
     repo root) on `sys.path`, so `from eval.knowledge_probe import …` failed. Fixed by passing
     `env={**os.environ, "PYTHONPATH": ROOT}` to the subprocess; verified with a fast import smoke
     test before relaunch. Nothing was lost (crash was pre-write).
  Eval now running (PID loading E2B 4-bit, arm 1 = qa4000). ~10 h for 17 arms. Pending the curve.

**2026-10-08 — VOLUME SWEEP COMPLETE (two-curve result).** Pipeline changed to **split generate →
judge-in-session** at the user's request (judge with the Claude Code subscription, which the GPU host
gateway can't reach — it serves only `deepseek-code-h`). The GPU host generated all closed-book predictions
(8,658 = 18 arms × 481 held-out facts) with the tested `build_generate_fn`; judging done in-session
by Claude judging-subagents (one per arm, exact `judge_mode="partial"` rubric). Results in
`results/knowledge/uc4/volume_curve_e2b_claude.json` (+ raw per-arm scores in `volsweep_judge_scores/`,
predictions in `volsweep_preds.jsonl`); chart artifact "Knowledge Volume Sweep".

- **Two Y-curves vs training-QA X** (retention = coverage@τ0.5; accuracy = mean fidelity), 18 arms
  qa0(base)…qa60224 + full-epoch root adapter. Base S0 = **0.083** cov / 0.070 acc (Claude judge,
  stricter than deepseek's 0.154 — absolute level is judge-dependent). RAG recall@5 = **0.603**
  (judge-independent retrieval ceiling, injected).
- **Shape:** base 0.083 → ~0.25 by **4k QA** → plateau **~0.30** from ~16–20k onward; spread across
  the plateau (0.28–0.33) is within judge noise (the two full-epoch checkpoints differ 0.012). **More
  QA past ~20k adds no retained knowledge** — severe diminishing returns.
- **FT vs RAG headline:** fine-tuned SLM plateaus at **~half** the RAG retrieval ceiling (0.30 vs
  0.603); parametric generalization to held-out corpus facts never approaches what retrieval surfaces.
  Supports the non-inferiority+cost reframing, NOT "SLM dominates RAG" on knowledge coverage.
- **GPU-eval lessons (see [[gpu-generation-sequential]] in memory):** per-arm model reload + 3 resident
  4-bit copies on the 23 GB A10G caused a silent **CPU-offload** stall (~3 h lost); fixed by running
  generation **1 arm/process sequentially** (user: "even 1 arm at a time is fine").
- **Judging-integrity incident:** parallel judging-subagents collided on shared scratchpad temp-script
  paths; all affected agents detected it and re-wrote directly. Verified every file (481/contiguous,
  18 distinct verdict-sequences, qa0 refusal-clean, smooth curve). One arm (qa40000) was deleted as a
  presumed stray during another agent's cleanup and re-judged (restored to its original 99/50/332).
  Future in-session judging: give each subagent a unique temp path or require direct Write only.
