# Parametric Consolidation Roadmap
## Five levers to close the closed-book ↔ with-retrieval gap

**Date:** 2026-09-04
**Status:** Working document — one idea is implemented/validated at a time.
**Author context:** KVForge reframing from "KV-cache latency" to "an SLM progressively replaces RAG."

> **Active pre-registration (2026-10-09):** `docs/levers_bca_experiment_design.md` — tests three
> pure-parametric levers (B: all-linear LoRA incl. MLP; C: relational/EntiGraph-style augmentation;
> A: more exposures/epochs) on the cheap from-base A/B testbed with locked works/doesn't-work bands,
> then opens **Phase 2: FT+RAG complementarity (RAFT, lever 4 here)** as its own study.

---

## 0. The yardstick: what "impact on the KVForge philosophy" means

The thesis is **progressive parametric consolidation** — knowledge moves *out of* the
external retrieval store and *into* the model's weights, so retrieval shrinks over time
to a fallback. Every idea below is scored against one question:

> *How directly does this move knowledge into parameters and shrink retrieval?*

The honest floor (from prior research: Ovadia EMNLP 2024, 2026 drift studies) is that
**full "replace RAG" is contradicted**. The defensible target is *progressive consolidation
with retrieval as a shrinking fallback* — parametric+RAG compounds (EntiGraph 62.60% vs
60.35% alone), it does not replace outright.

---

## 1. Current evidence (the gap we are trying to close)

Three-arm factual-accuracy (FA = 0.5·LLM-judge + 0.5·cosine), held-out set of 89 UC4
questions, judge = `bedrock-claude-sonnet-4-6` via CodeVector (single judge for all rows).

| Arm | Label | redo N=3500 | phase3 avg (N500–6000) |
|-----|-------|------------:|------------------------:|
| **A** | base + RAG (control) | 0.612 | 0.615 |
| **B** | LoRA + RAG | **0.621** | 0.594 |
| **C** | LoRA closed-book | 0.522 | 0.548 |

- **Gap to close:** Arm C is ~**0.09–0.10 FA** below Arm A/B. Closed-book answers are
  "on-topic but wrong" (cosine stays ~0.73, judge collapses to ~0.31).
- **Key diagnosis (corrected 2026-09-04 after reading `pipeline/lora_trainer.py`):** the redo
  recipe is *answer-masked SFT on question → answer pairs with **no context*** — it is
  **already closed-book**. The redo FAQ data (`faqs_N3500.json`) also has **zero** diverse
  rewrites (0% carry a `(variant N)` suffix). So the closed-book gap is *not* caused by an
  open-book objective, nor by "missing rewrites being applied" — the model reproduces
  RAG-generated answers from the question alone at 2B params and only learns shallow surface
  associations, not the underlying facts.

---

## 2. The five levers

### Idea #1 — Diverse-rewrite + chunk-augmented training *(highest impact — START HERE)*

- **What:** the training is *already* closed-book (question → answer, no context). The lever is
  to make that signal **build parametric memory**: (a) generate 10–40 diverse rewrites per chunk
  and mix them with QA so the model learns the *fact*, not the surface question string; (b) train
  on the source chunks themselves so the model internalizes the documents and can answer *novel*
  questions. This is the roadmap's top-ranked direction, and the generator **already exists**
  (`pipeline/corpus_diversifier.py`, `pipeline/entigraph_generator.py`) but was never applied to
  the redo sweep.
- **Why it matters most:** Arm C is the whole thesis, and it is low because of a weak signal
  (surface Q→A memorization at 2B), not a wrong objective. This lever directly raises the
  quality of the parametric memory.
- **Mechanism:** keep P(answer | question) but diversify the *inputs* (rewrites) and add the
  *documents* (chunks) so the weights capture underlying facts, not exact question strings.
- **Cheap test:** N=500 baseline vs N=500 with ~10 rewrites/chunk (same seed), plus a chunk-training
  variant; run all through `three_arm_evaluator`.
- **Go signal:** the diverse-rewrite LoRA's **Arm C rises** vs baseline (target ≥ +0.03 FA),
  *and* **Arm B stays ≥ base+RAG**.
- **Cost:** rewrite generation (local GPU or teacher) + 2 trains + 2 evals ≈ 1–2 hours.
- **Open questions:** rewrite count (10 vs 40); generator (local model vs teacher); whether to add
  source-chunk examples and at what ratio.

### Idea #2 — Distillation from a strong teacher *(high impact)*

- **What:** a large teacher (Claude Opus, or a big open model) generates domain answers; the
  small model is trained to match them. `train_distillation()` already exists.
- **Why:** *efficient* consolidation — one teacher answer carries more factual signal than one
  FAQ pair, so the model internalizes knowledge with fewer examples. Complementary to #1
  (you can distill *and* train closed-book).
- **Cheap test:** reuse `distill_pairs_v1.json` at small N (~200–300), train, eval Arm C.
- **Go signal:** distillation LoRA's Arm C beats the FAQ-trained LoRA at equal N.
- **Cost:** teacher generation (partly done) + 1 train + 1 eval.

### Idea #3 — Scale up capacity (2B → 7–8B) *(necessary, not distinctive)*

- **What:** bigger base model, more memorization capacity.
- **Why lower *philosophical* impact:** it's an *enabler*, not the insight — everyone can make
  the model bigger. Sprint 1 concluded "3B is the bottleneck", so it is likely *required*,
  but only pays off combined with #1/#2.
- **Cheap test:** train on the same N=500 with a larger model (gemma-4-4B or 7B in 4-bit);
  compare its (Arm A − Arm C) gap to the 2B model at identical N.
- **Go signal:** the larger model's closed-book gap is visibly smaller at the same N.
- **Cost:** 1 train on the bigger model (slower; A10G 24 GB caps ~7B 4-bit).

### Idea #4 — KV-cache parametric memory *(low philosophical impact; distinctive branding)*

- **What:** inject document **KV-cache** instead of text (kv_meanpool / kv_fulltoken / CacheBlend / KVLink).
- **Why lower for *this* goal:** it's a different axis. KV injection does **not** move knowledge
  into the model — it is a *cheaper form of retrieval*. It serves the old KV/latency narrative.
  Paper finding: kv_fulltoken is 2.2–2.5× *worse* than text_rag; only kv_meanpool *matches* it.
- **Cheap test:** on ~25 held-out queries, run text_rag vs kv_meanpool vs closed-book; record
  FA + latency.
- **Go signal:** kv_meanpool FA ≈ text_rag FA at lower latency/cost.
- **Cost:** no training, inference-only.

### Idea #5 — Hybrid gating / routing *(high pragmatic, medium philosophical impact)*

- **What:** answer closed-book when confident, retrieve only when unsure (PRS v2 = semantic
  entropy + R-Tuning + conformal calibration), or route per-doc via adapter banks (Poly-PRAG).
- **Why — nuanced:** it does not consolidate anything by itself, but it is the **enforcement +
  measurement** of the philosophy. The gate is what makes retrieval actually *shrink* and makes
  consolidation *observable* (retrieval rate becomes the progress metric). Essential as the
  *control plane*, not the *consolidation engine*.
- **Cheap test:** on the 89 held-out questions, compute a confidence score (semantic entropy or
  the existing PRS confidence token), sort by confidence, simulate "retrieve only the least-confident K%".
- **Go signal:** retrieving on ~30–50% of questions recovers ≈ Arm A quality.
- **Cost:** no training — pure computation on saved answers.

---

## 3. Sequencing recommendation

| Rank | Idea | Role | Effort |
|------|------|------|--------|
| 1 | #1 Closed-book training | the consolidation engine | ~1 hr, cheap |
| 2 | #2 Distillation | efficient consolidation | ~1 hr |
| 3 | #5 Gating/routing | control plane (safe + measurable) | no GPU |
| 4 | #3 Scale-up | capacity enabler | expensive |
| 5 | #4 KV-cache | different axis (latency, not consolidation) | no GPU |

**Plan:** do **#1 first** (cheapest, most on-the-nose — it tells us within an hour whether the
"progressively replace RAG" idea has legs on the current model). Pair it with **#5's gate**
(free, non-GPU) so we also get the "shrinking retrieval rate" metric that *measures* progress.
Scale-up (#3) follows once #1 proves the mechanism.

---

## 4. Work log

- **2026-09-04** — Document created. Next: implement Idea #1 (`--closed-book` training).
- **2026-09-09** — Idea #3 (scale-up) validated: E4B r64/3ep vs E2B r16/1ep on UC4 under a single judge (`deepseek-v4-lite` via CodeVector, since `deepseek-v4-flash` is no longer reachable). E4B wins **+0.035 FA** (0.155 vs 0.120; paired t=1.66) — confirms capacity helps but is modest, exactly as the roadmap predicted. Judge note: `deepseek-v4-flash` was ~2× more lenient than `deepseek-v4-lite`, so the old 0.234 baseline is not comparable to new numbers. Next: Idea #1 (diverse-rewrite + chunk-augmented training).
- **2026-09-10** — **Idea #1 validated (full corpus).** Trained E4B r64 on 121,763 examples (60,209 QA + 61,554 diverse rewrites) for 1 epoch. Arm C parametric FA jumped **0.155 → 0.235** (+0.080 vs the E4B QA-only baseline, paired t=2.51; +0.115 vs E2B, t=4.10) under a single `deepseek-v4-lite` judge — ~3× the +0.03 go-signal. This confirms Idea #1 is the dominant lever (diverse signal, not scale or judge). Learning curve (ckpt 7500 vs 7611 = 0.230 vs 0.235) shows the gain plateaus before full corpus consumption. Resumable checkpoints saved every 1,500 steps. Next levers: more epochs / distillation (#2), or gating/routing (#5) to make the gain operationally observable.
- **2026-09-12** — **Held-out set expanded 89 → 500, and two eval-harness bugs found.** (1) Built a fresh 500-Q UC4 eval set with `deepseekv4p`: 795 generated from evenly-sampled chunks, deduplicated by normalized question *and* answer (exact + Jaccard ≥ 0.80) against both the 89-eval and 60,209-train sets → **zero question-overlap and zero answer-overlap**, 500 distinct chunks spanning the full corpus. Answer-level dedup was essential — the generator reproduced training facts verbatim when seeded with shared chunks. (2) Found `measure_baseline_fkds.py` never passed the adaptive `max_new_tokens` into `_generate_parametric` (defaulted to 256), so open-ended questions rambled and depressed token-F1; fixed to pass the adaptive budget (~30–40 tokens).
- **2026-09-12** — **500-Q re-benchmark (all `deepseek-v4-lite`, parametric, n=500, corrected generation).** E2B = **0.147**, E4B QA-only (r64/3ep) = **0.235**, E4B+diverse checkpoints = **0.278 / 0.275 / 0.288 / 0.280 / 0.302 / 0.295** (steps 9000→15222). Two revisions to the roadmap's priors: **(a)** the scale-up gap is **+0.088 FA (~60% rel., ~7.4σ)**, not +0.035 — the n=89 estimate was depressed by the rambling bug (E2B answers ran ~26 words vs E4B ~17), so capacity (#3) is a *stronger* lever than originally logged. **(b)** Idea #1 still wins: +0.04 to +0.07 over the E4B QA-only baseline at 5.5× the sample size, with the gain plateauing early (peak ckpt-15000 = 0.302). Numbers are not directly comparable to the n=89 figures (set size + generation fix); the *relative* comparisons are the takeaway. Next: decide whether to filter page-number trivia and re-run; then #2 (distillation) or #5 (gating) as the next lever.
- **2026-09-14** — **60k-QA 2×2 ablation (QA volume vs rewrites × E2B/E4B).** Arm A (E4B @ 60,209 QA, no rewrites) = **0.244 ± 0.011** ≈ the 337-QA baseline (0.235) — QA *volume* is inert for E4B (337→60k moved nothing). Arm C (E2B @ 60,209 QA + 61,554 rewrites, r64) = **0.249 ± 0.012**, matching E4B's no-rewrite level — rewrites let the small model catch up to the larger model's baseline. Arm B (E2B @ 60k QA, no rewrites) is C's same-rank/same-QA control and is still training; it isolates the rewrites effect for E2B. **Refined hypothesis:** the Idea #1 gain is the *combination* of QA + rewrites (neither extra QA volume nor rewrites alone), and the **quality of the rewrite generator** (Claude Haiku 4.5) may be a real factor — two follow-up experiments recorded in §5, *not started*.

- **2026-09-14** — **Llama-3.2-3B architecture-transfer experiment DONE — rewrite lever is architecture-agnostic (null REJECTED).** Train `meta-llama/Llama-3.2-3B-Instruct` (r=64, 1 ep) on the *same* 60,209 QA + 61,554 rewrites as the E2B/E4B arms, eval on 500-Q (`deepseek-v4-lite`). **Result: FA 0.254 ± 0.012** (token-F1 0.312, judge 0.196, latency p50 1.47 s) — *above* E2B's 0.249 at the same recipe, and just below E4B's 0.286. This **overturns** both the prior null and the "architecture is the primary determinant" reading from the QA-only `pubmedqa_crossover` data (where Llama-3.2-3B was flat ~0.11–0.12 and never crossed its RAG): with rewrites, a *smaller* (3.2B) *non-Gemma* model matches the larger E2B (5.1B). The lever is the diverse-rewrite data, not base-model architecture or raw capacity. Runner `run_llama_60k.sh`; gated-cache workaround: materialized `~/kvforge/models/Llama-3.2-3B-Instruct/` from HF blobs (snapshot symlinks were missing) and point `llm_model` at the local path + `HF_HUB_OFFLINE=1`.
- **2026-09-14** — **Rank-confound correction to the 2026-09-12 scale-up claim.** The "+0.088 FA scale-up" logged on 2026-09-12 compared E2B-r16 (0.147) against E4B-r64 (0.235) and is **confounded**: at matched r=64 (60k QA) the true E2B→E4B capacity gap is **+0.025 (no rewrites) / +0.037 (with rewrites)**. The ~+0.063 remainder is the r16 adapter under-fitting E2B, not model scale (E2B r16→r64 = 0.147→0.219 = +0.072, minus ~+0.009 inert QA-volume). Capacity is therefore a real but *marginal* lever, as §5.9 originally read. RAG baselines also established on the 500-Q set (E2B 0.175, E4B 0.161): trained parametric strictly dominates retrieval on both accuracy and latency. Paper §5.9/§5.11/§5.12 corrected and rebuilt (Figures 11–12).

---

## 5. Planned experiments (recorded — not started)

**Refined hypothesis (from the 60k ablation, §4).** The E4B gain is not attributable to extra QA volume alone (arm A: 337→60k QA = +0.009, within noise) and is not expected to come from rewrites alone (rewrites store facts but do not teach the Q→A mapping). The gain appears to come from the *combination* of QA pairs + rewrites. A second-order factor is **rewrite-generator quality**: the current rewrites were produced by Claude Haiku 4.5 (a frontier model), whose injected diversity may be higher-quality than what the small models could produce for themselves.

Two experiments test the source/quality of the rewrite generator by having the small models generate their own rewrites:

1. **E2B self-rewrites** — use the *out-of-the-box (untrained) E2B* to rewrite the UC4 chunks, then train E2B on (60,209 QA + those self-generated rewrites). Tests whether a weak generator's diversity suffices, or whether the frontier-model quality (Haiku 4.5) is doing the work.

2. **E4B self-rewrites** — the *untrained E4B* generates the rewrites, then E4B is trained on (60,209 QA + those self-generated rewrites). Isolates whether the rewrite-quality benefit depends on the generator's capacity.

**Status: recorded only — do not start.** Each requires a self-rewrite generation pass (adapt `tools/generate_diversity_data.py` to a local HF model instead of Claude) plus one train + one eval (~12–18 h per arm on the A10G).
