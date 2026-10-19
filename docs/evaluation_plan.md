# KVForge Evaluation Plan

**Objective:** Test not only whether KVForge's parametric (Phase 3) mode matches or beats standard RAG, but **under which specific conditions** it is a stronger choice — and measure that on the axes the system actually claims an advantage on (cost, tail latency, deployment footprint), not only on factual accuracy.

**Scope:** 12 experiments, ordered by expected impact. The plan covers (a) the system's economic premise, which the paper asserts but has never measured; (b) operational and deployment claims (tail latency, CPU-only serving); (c) the accuracy claim, rebuilt on a fair measurement protocol; (d) the boundary conditions under which KVForge *loses*; and (e) the paper's two positive sub-claims (the diverse-rewrite lever, and PRS gating).

Every experiment states a hypothesis, baselines, setup, datasets, metrics, **pre-registered success and falsification criteria**, and threats to validity.

---

## Revision note (this version)

This revision was produced after a matched-format study on the UC1 corpus and a set of diagnostics on that study's outputs. Those results changed three things about this plan, and they are the reason several sections below differ from the previous version.

**1. The accuracy claim is weaker than the paper states, and part of it is a measurement artifact.**

In the matched-format study, the UC1 parametric-vs-RAG gap was **+0.2714** as originally measured. Under a fair protocol — format instruction actually reaching the RAG arm, and gold answers matched in style to the training targets — the same comparison gave **+0.0593 (short)** and **+0.0286 (long)**. The cause was not knowledge: the RAG arm was emitting 30-word answers against a 14-word gold while the parametric arm emitted 13 words. A length-mismatched arm is penalised by token-F1 for verbosity, not for being wrong.

On the external panel (expertqa / techqa / emanual, pre-registered), the primary interaction estimate D was **negative in all four domains** (pooled **−0.042**, and −0.040 on token-F1 alone, −0.045 on the judge alone). D never approached the pre-registered +0.05 threshold, and its sign was unstable under length control (+0.012 at a 20-token budget, −0.023 at 40).

**Consequence for this plan:** the accuracy claim must be restated as a **non-inferiority** claim with a declared margin, not a dominance claim, and it must be measured with length control. See §0 and Exp 5.

**2. The judge used for accuracy scoring is not length-neutral, and failed on a third of calls.**

Measured on the study outputs: the judge's CORRECT rate rose monotonically with recall of the gold — **0.020** at recall < 0.2, **0.146** at 0.2–0.4, **0.362** at 0.4–0.6, **0.571** at 0.6–0.8, **0.846** at ≥ 0.8. Against a 135-word gold, a 30-word answer is therefore judged INCORRECT almost by construction. Separately, **730 of 2152 judge calls (33.9%)** failed with a transport/parse error and were being scored as *incorrect*; the failure rate was **15–33% in short-answer cells and 50–63% in long-answer cells**, i.e. the bias landed hardest on the cells under test.

**Consequence:** §0 mandates a length-neutral or length-controlled primary metric, and forbids scoring a judge failure as an incorrect answer.

**3. Parametric's advantage is information *density*, not coverage.**

In every long-answer cell, the parametric arm had **higher precision and lower recall** than RAG:

| corpus | arm | precision | recall |
|---|---|---|---|
| expertqa | RAG | 0.456 | 0.216 |
| expertqa | parametric | **0.560** | 0.137 |
| techqa | RAG | 0.413 | 0.306 |
| techqa | parametric | **0.657** | 0.255 |
| emanual | RAG | 0.466 | 0.461 |
| emanual | parametric | **0.506** | 0.274 |

At a **common token budget** the parametric arm's recall was equal or higher (20-token budget: 0.615 vs 0.539 on expertqa, 0.694 vs 0.517 on techqa). Parametric is not less knowledgeable; it writes fewer words.

**Consequence:** answer **format** is the wrong moderator to test. The variable that matters is **answer length / coverage budget**. The plan therefore reframes around length control, and treats long-form extraction as a boundary condition rather than the headline.

**What this revision does *not* change:** the diverse-rewrite lever (§5.10–5.13) remains the paper's best-evidenced claim and its experiment (Exp 9) is retained largely as written. The Phase-2 known-negative is still treated honestly and its improvement experiment (Exp 10) is retained. The calibration experiment (Exp 7) is retained and expanded.

### Mapping from the previous version

| Previous | This version | Change |
|---|---|---|
| — | **Exp 1: Amortisation & break-even N\*** | **New.** The paper's economic premise, never measured. |
| — | **Exp 2: Tail latency under concurrency** | **New.** p50 was reported; the tail was not. |
| — | **Exp 3: CPU-only serving** | **New.** "GPU potentially not required" was never tested. |
| Exp 1: Crossover on public benchmarks | Exp 4 (revised) | Corpus size 100 → sweep; HotpotQA demoted from primary to adversarial; models/hardware realigned with the paper; length control added. |
| Exp 2: Judge invariance | Exp 5 (revised) | Gold-length invariance added as a second factor; common-budget scoring added; `Recall@K` misdefinition fixed. |
| — | **Exp 6: Absorbable-regime boundary** | **New.** Defines the corpus-size limit of the claim. |
| Exp 7: Confidence calibration | Exp 7 (expanded) | Gate AUROC and selective-risk curves added; validation-split threshold selection made a protocol rule. |
| — | **Exp 8: Churn & staleness** | **New.** Previously only a threat; now measured as the boundary where KVForge loses. |
| Exp 3: Rewrite lever | Exp 9 (criterion fixed) | Inverted success criterion corrected; token-matched arm and self-rewrite control added. |
| Exp 6: Phase 2 + realignment | Exp 10 (retained) | Per-correction attribution and matched-output-length latency added. |
| Exp 4: External baselines | Exp 11 (revised) | Reprioritised to last; reframed from superiority to non-inferiority + cost/latency; parity rule added; effort estimate raised. |
| Exp 5: Retrieval-hard robustness | Exp 12 (retained, demoted) | Now an explicit boundary condition; multi-hop moved here from Exp 4. |

---

## 0. Measurement protocol (applies to every experiment)

These are mandatory. Without them, no accuracy number in this plan is interpretable. Each was derived from a concrete failure observed in the matched-format study.

### 0.1 Scoring

| Rule | Rationale |
|---|---|
| **Declare one FA formula per experiment, in the results text.** Report its components separately. | The repository has contained at least two incompatible FA definitions (`0.5·f1 + 0.5·judge` and `0.5·judge + 0.5·cosine`). Values from different harnesses are not comparable. |
| **Primary accuracy metric must be length-controlled.** Use either (a) a **common-token-budget F1** (both arms truncated to the same budget, report at ≥2 budgets), or (b) **length-matched golds**, where the gold for each item is generated by the *same generator and instruction* as the training targets. | The 4.5× gold/answer length mismatch on expertqa was the single largest driver of the apparent parametric deficit. |
| **No judge may be the sole primary metric.** Report a judge-free metric alongside every judged metric. | The judge's verdict is a near-monotone function of gold recall (§Revision note 2). |
| **Never score a judge failure as incorrect.** Retry with backoff; on exhaustion record `judge_failed=True`, exclude that item's judge term, and report the failure count per cell. | 730/2152 calls were silently scored 0, biased 2–3× toward the long-answer cells. |
| **Report precision / recall / F1 separately**, not F1 alone. | F1 alone hides the density-vs-coverage trade-off that is the actual finding. |
| **Judge input token budget must exceed the reasoning budget.** | A 256-token cap was consumed by reasoning tokens, producing empty `content` and the failures above. |
| **Report refusal/empty-answer rate and length-compliance rate** as diagnostics. | A short answer that is *correct but truncated by the harness* is not a short answer. |

### 0.2 Statistical

| Rule | Rationale |
|---|---|
| **≥3 seeds** for every trained arm; report mean ± 95% CI. Single-seed results are labelled as such and cannot support a headline claim. | The paper's §6.5 concedes single-seed, no-CI throughout. |
| **Paired at the question level**: identical questions across all arms and cells. | Removes question difficulty as a confound. |
| **Pre-register a small set of PRIMARY tests** (≤3 per experiment) with multiplicity correction; report everything else as exploratory with raw p-values. | 35+ tests at raw p-values risks a false-positive headline. |
| **State the minimum effect of interest (MEI) before running.** Default: **0.05 FA** for superiority, **0.02 FA** for non-inferiority margins. Results inside the MEI are reported as "within noise", not as a win. | Prevents post-hoc rationalisation. |
| **Report achieved n per arm**, never a nominal target. | RAGBench dedup reduced techqa to n=59 and emanual to n=122 (see §0.4). |

### 0.3 Latency and cost

| Rule | Rationale |
|---|---|
| Report **p50 / p95 / p99 / max**, not p50 alone. | KVForge's strongest operational claim is the **tail** (RAG p95 ≈ 30 s vs parametric ≈ 3 s); reporting p50 undersells it. |
| Measure under **concurrent load**, not single-query. Report **throughput-at-SLO**. | Single-query latency does not predict a service. |
| Compute **$/1M queries including amortised training + indexing + KV recompute**; report the **break-even volume N\***. | The paper's amortisation premise has never been measured. |
| Report **GPU-seconds/query, VRAM at load, energy/query**, and the **hardware class required**. | The "no GPU needed" claim is a deployment differentiator and is currently untested. |
| Every cross-system comparison states a **parity rule** (matched hardware, matched base model, matched query set, matched prompt budget). | §6.5 lists matched conditions as the missing bar. |

### 0.4 Data hygiene

| Rule | Rationale |
|---|---|
| **Deduplicate held-out questions** on normalised question text (first occurrence wins) and record the removed count in the manifest. | RAGBench splits contain duplicate question *rows*: emanual validation has 124 rows but only 66 distinct questions, and duplicated rows carry identical evidence. Left in, this double-counts and invalidates paired tests. |
| **Subset corpora by a content-derived, result-blind rule** (e.g. md5 of normalised chunk text), and record the manifest md5. | Removes researcher degrees of freedom in corpus selection. |
| **Verify zero train/held-out overlap at both question and answer level** (exact + near-duplicate via token-set Jaccard). | The paper's §5.11 shows answer-level leakage is real: seeding a generator with training chunks reproduces training facts verbatim. |
| **Report every domain attempted**, including those excluded, with the exclusion rule. | Anti-cherry-picking. |

---

# Experiment 1: Amortisation and Break-Even Volume *(Highest Impact — new)*

### Hypothesis
KVForge's cost advantage is an **amortisation** claim, not a per-query claim. Cost-per-query for KVForge decreases with total query volume while RAG's stays flat, so there exists a break-even volume **N\*** beyond which KVForge is strictly cheaper. N\* should **decrease as query-distribution skew increases**, because repeats are what amortise.

**Rationale:** Paper claim C6 ("never re-encodes; per-query cost becomes a one-time amortised cost") is the entire economic justification for the architecture, and the paper reports only a ~20 min/round pipeline timing (§5.3). No workload, no volume sweep, no break-even point, and no query distribution have been measured. This is the largest gap between what the paper argues and what it demonstrates.

### Baselines
1. **Standard RAG** (top-K, text-in-context), same base model, same retriever, same hardware.
2. **RAG with prefix caching enabled** (vLLM `--enable-prefix-caching`) — the fair modern RAG baseline. Omitting this would overstate KVForge's advantage, since prefix caching captures some of the same reuse.
3. **KVForge Phase 3** (parametric) and, where available, **Phase 2**.

### Setup
- **Corpora:** UC1 (customer support) and UC4 (Bedrock guide) — both already have trained adapters, so this experiment requires **no training**.
- **Query distribution:** synthetic workloads drawn over the corpus with *Zipf(α)* over chunks, α ∈ {0, 0.5, 0.8, 1.0, 1.2}. α=0 is uniform (no repeats). Where a real query log is available, use it and report its fitted α; otherwise document the simulation parameters.
- **Volume:** V ∈ {10³, 10⁴, 10⁵, 10⁶} queries per condition.
- **Hardware:** one A10G (matching the paper) and one CPU-only class (feeds Exp 3).
- **One-time costs to include:** indexing + embedding, KV precomputation (0.20 s/chunk per §5.3), LoRA training, and the post-training KV recompute.

### Metrics
| Metric | Interpretation |
|---|---|
| **GPU-seconds/query** and **$/1M queries**, at each V, including amortised one-time cost | Primary quantity |
| **N\*** — break-even volume | **Pre-registered primary.** Report per α, with CI across seeds |
| **Marginal cost per repeat query** | Should be ≈0 for KVForge, >0 for RAG |
| **Cost-ratio curve** KVForge/RAG vs V | Shows where the crossover sits |
| p50 / p95 latency under the same workloads | Secondary |

### Success criteria
- **N\* exists and is finite** for α ≥ 0.8, and is **monotonically decreasing in α**.
- N\* ≤ 10⁵ queries for at least one realistic α on at least one corpus.
- Marginal cost of a repeat query for KVForge is **< 20%** of RAG's.

### Falsification criteria
- N\* is not finite within 10⁶ queries at α = 1.0 → **amortisation premise fails**, and the paper's framing must change from "cheaper at scale" to "faster per query".
- N\* does not decrease with α → KVForge is not exploiting repetition, contradicting the architecture's stated mechanism.
- A prefix-cached RAG baseline matches KVForge's cost at all α → the advantage is attributable to generic cache reuse, not to consolidation.

### Threats to validity
1. Synthetic workloads may not match production skew. Mitigation: report fitted α from any available real log; present results as a function of α so any deployment can locate itself.
2. Training cost is dominated by one-time setup; amortising over an arbitrary V could flatter either system. Mitigation: report the cost curve, not a single point, and include KV recompute.
3. Prefix-caching implementation variance. Mitigation: use the vLLM default; report its hit rate alongside.

### Estimated cost
**Low.** No training. Reuses existing adapters. 1–2 weeks.

---

# Experiment 2: Tail Latency and Throughput Under Concurrent Load *(High Impact — new)*

### Hypothesis
At matched concurrency, KVForge's **p95/p99 latency and latency variance** are dramatically lower than RAG's, because it performs no variable-length prefill and no retrieval round-trip. The advantage grows with concurrency, since RAG's prefill cost is itself variable.

**Rationale:** The paper reports p50 (1.2–1.6 s vs 3.2–3.5 s) and mentions a p95 of ~30 s only in a figure caption. The tail is the strongest operational differentiator and is currently under-sold and unmeasured.

### Baselines
Same three as Exp 1, plus a **long-context single-shot** configuration where applicable.

### Setup
- **Load generator:** open-loop (Poisson arrivals) and closed-loop (fixed concurrency C ∈ {1, 8, 32, 128}).
- **Serving stack:** identical for both systems (vLLM), identical batching configuration. Report the config.
- **Corpus/models:** UC1 and UC4, existing adapters; A10G.

### Metrics
| Metric | Interpretation |
|---|---|
| **TTFT, TPOT**, p50/p95/p99/max | Primary latency |
| **Throughput-at-SLO** (req/s within a 2 s budget) | Primary operational quantity |
| Latency **coefficient of variation** | Measures predictability, which production cares about |
| VRAM at load, queue depth, request drops | Capacity |
| Retriever service time distribution | Isolates where the tail comes from |

### Success criteria
- KVForge p95 ≤ 0.5× RAG p95 at C ≥ 32, with non-overlapping CIs.
- Throughput-at-SLO ≥ 2× RAG at C = 128.
- Latency CV ≤ 0.5× RAG's.

### Falsification criteria
- p95 advantage disappears at C ≥ 32 (batching hides RAG's prefill) → the operational case weakens substantially.
- KVForge's throughput saturates below RAG's at high C (e.g. a serialising gate or KV lookup becomes the bottleneck) → the architecture's serving premise fails.

### Threats to validity
1. Batching configuration can dominate results. Mitigation: sweep batch size and report throughput-vs-batch for both.
2. Retrieval backend (Qdrant) may be the RAG bottleneck rather than the LLM. Mitigation: report retriever time separately and run a retrieval-only control.
3. Hardware-specific kernels. Mitigation: report hardware explicitly; note the paper's single-A10G regime.

### Estimated cost
**Low–medium.** 1–2 weeks.

---

# Experiment 3: CPU-Only Serving *(High Impact — new)*

### Hypothesis
Phase 3 on a quantised small model serves within a usable latency budget **on CPU**, where the matched RAG arm cannot (prefill of ~3k context tokens is prohibitive without a GPU). This makes KVForge a category-level deployment alternative, not merely a faster GPU system.

**Rationale:** The paper's §5.8 comparison table lists "GPU required: Potentially no" for KVForge's Phase 3 and never tests it. If it holds, it is arguably the strongest single differentiator in the paper — no RAG baseline can match it, because retrieval-and-prefill is inherently GPU-shaped. It is also cheap to test.

### Baselines
1. **Standard RAG on CPU** (same quantised model, same retriever).
2. **KVForge Phase 3 on CPU** (quantised, GGUF/llama.cpp or an equivalent CPU runtime).
3. **Standard RAG on GPU** as a latency reference point.

### Setup
- **Model sizes:** E2B, E4B, Llama-3.2-3B (all already trained in the paper's §5.13 work).
- **Quantisation:** match bit-width across arms (e.g. 4-bit both).
- **Instances:** one CPU-only class (e.g. `c7i`) vs one GPU class (`g5`) for the cost comparison.

### Metrics
| Metric | Interpretation |
|---|---|
| p50 / p95 latency on CPU | Primary feasibility gate |
| Throughput (req/s) at C ∈ {1, 8} | Capacity |
| RSS / peak memory | Fits commodity instance? |
| **$/1M queries on CPU vs GPU** | The deployment argument |
| Energy/query (or instance-hours/10⁶ queries) | Sustainability angle |
| Accuracy on CPU vs GPU (same weights) | Confirms no degradation from the runtime switch |

### Success criteria
- Phase 3 CPU p95 ≤ 5 s at C = 1 on at least one model size.
- CPU $/1M queries < GPU $/1M queries for the same accuracy (accounting for the cheaper instance class).
- Accuracy identical to GPU within CI (same weights, different runtime).

### Falsification criteria
- CPU p95 > 10 s → drop the claim.
- CPU throughput < 1 req/s → not a service.
- Accuracy degrades beyond CI on CPU (runtime/quantisation mismatch) → the claim is not deliverable.

### Threats to validity
1. CPU runtime quality differs from the GPU stack (different kernels, quantisation schemes). Mitigation: report both, and hold bit-width constant.
2. Instance pricing is volatile. Mitigation: report instance-hours as well as $, so the result survives repricing.

### Estimated cost
**Very low.** No training. 3–5 days.

---

# Experiment 4: Crossover Curve and Corpus-Size Sweep *(Revised from original Exp 1)*

### Changes from the previous version, and why
- **Corpus size raised from a fixed 100 chunks to a sweep {500, 2 000, 5 000, 10 000, 25 000}.** A 100-chunk corpus is 20× smaller than the paper's smallest corpus (UC1 = 2 000). At 100 chunks a parametric model memorises the corpus almost trivially and RAG has essentially nothing to retrieve — the crossover would be manufactured by the experimental design rather than demonstrated. This was the most serious flaw in the previous plan.
- **HotpotQA demoted from primary to adversarial.** HotpotQA is **multi-hop**: it requires composing evidence across documents, which is what retrieval is good at and what a rank-64 adapter on a 5–7B model is worst at. Using it as the primary benchmark biases against KVForge and, if KVForge won anyway, would invite the objection that the task rewarded memorisation. It becomes a condition in Exp 12.
- **Models and hardware aligned with the paper** (Gemma-4 E2B/E4B, Llama-3.2-3B, single A10G) unless a change is explicitly justified. The previous plan's Llama-3.2-8B / Gemma-2-9B on A100 breaks comparability with every number in the paper, changes the cost regime, and contradicts §5.13's "capacity is secondary" thesis.
- **Length-controlled scoring** per §0.1 replaces plain FA.
- **Success criterion now requires CIs and the declared MEI**, not a bare point comparison.

### Hypothesis
Parametric Phase 3 reaches **non-inferiority** with a tuned RAG baseline at 100–500 training pairs on **fact-dense, single-hop** public corpora, and the training coverage required grows with corpus size.

### Baselines
1. Standard RAG (untrained), same base model + retriever + hardware.
2. RAG + LoRA on dataset-native golds (same rank/alpha), to separate "fine-tuning" from "generation pipeline".
3. RAG with prefix caching enabled.
4. Long-context single-shot (where the context fits).

### Setup
- **Models:** Gemma-4 E2B, Gemma-4 E4B, Llama-3.2-3B. All rank r=64 α=128 where applicable (rank-matched, per §5.12's finding that r=16 conflated rank with capacity).
- **Hardware:** single A10G (paper regime).
- **Corpora:** two fact-dense single-hop public corpora (e.g. a documentation/technical manual and a structured fact-dense QA corpus) plus UC1 and UC4 as internal references.
- **Corpus sizes:** {500, 2 000, 5 000, 10 000, 25 000} chunks, selected by the content-hash rule (§0.4).
- **Training tiers:** N ∈ {50, 100, 200, 500, 1 000, 2 000, 4 000} pairs.
- **Seeds:** 3.

### Metrics
| Metric | Interpretation |
|---|---|
| **Length-controlled FA** (primary) | Common-budget F1 at 2 budgets + judge, reported separately |
| ΔFA with paired 95% CI | Non-inferiority / superiority decision |
| **Crossover N** (parametric first ≥ RAG) | Primary for the sweep |
| token-F1, judge, cosine (separately) | Metric-sensitivity check |
| p50/p95 latency | Secondary |
| Length-compliance and refusal rate | Diagnostics |
| Precision / recall | Density vs coverage |

### Success criteria
- **Non-inferiority:** lower bound of the 95% CI on ΔFA > **−0.02** at N ≤ 500, on ≥2 corpora.
- Crossover occurs at N ≤ 500 on ≥2 corpora at 2 000 chunks.
- Result reproduces across ≥2 of 3 model families (architecture-agnostic).
- Crossover N increases monotonically with corpus size (supports the absorbable-regime story, quantified further in Exp 6).

### Falsification criteria
- No crossover by N = 2 000 on any corpus → the claim does not hold on public data.
- Crossover exists only at the two smallest corpus sizes → the effect is a small-corpus memorisation artifact, and the paper's regime must be stated as "under ~2 000 chunks".
- ΔFA CI width > ±0.05 → **underpowered**; report as such rather than as a win or loss.
- Parametric fails on one model family while passing on another with the same data → report as architecture-dependence.

### Threats to validity
1. Generated training pairs and held-out golds may share a generator, inflating parametric scores. Mitigation: held-out golds authored independently of the training generator where possible; report gold provenance; include a gold-independent metric.
2. Public corpora differ in retrieval difficulty, so "crossover N" is not comparable across corpora. Mitigation: report the RAG baseline level for each corpus alongside N\*.
3. Long golds from some benchmarks are orders of magnitude longer than the answers any arm produces. Mitigation: length control (§0.1) and explicit gold-length reporting.

### Estimated cost
**High.** ~7 tiers × 5 sizes × 3 models is far too many; run a reduced design (3 sizes × 4 tiers × 2 models × 3 seeds) and document the reduction as a pre-registered choice. 4–6 weeks GPU.

---

# Experiment 5: Judge Invariance **and Gold-Length Invariance** *(Revised from original Exp 2)*

### Changes from the previous version, and why
The previous version tested **judges** (four models + BERTScore + Spearman agreement). That is good and is retained. But it did **not** address the dominant confound, which the diagnostics showed is **gold-vs-answer length mismatch**, not judge identity:

| corpus | median gold | median parametric answer | ratio |
|---|---|---|---|
| expertqa | 135 w | 30 w | **4.5×** |
| techqa | 99 w | 29 w | 3.4× |
| emanual | 58 w | 27 w | 2.1× |
| uc1 (generated gold) | 29 w | 33 w | **0.9×** ✅ |

In the one cell where golds *were* matched, the measured gap fell from **+0.2714** to **+0.059**. A judge-only experiment could therefore report "judges disagree" while leaving the real confound untouched. This experiment now varies **both** factors in a factorial design.

### Hypothesis
The parametric-vs-RAG comparison is sensitive to **(a) judge identity** and **(b) gold length/style**, and the length factor dominates. Once answer lengths are matched or the metric is length-controlled, the gap shrinks substantially, and what remains is a **density** difference, not a knowledge difference.

### Baselines / factors
- **Judge factor:** Claude-Sonnet-class, an open-weights judge (e.g. Llama-70B-class), GPT-4o, plus **token-F1** and **BERTScore** as judge-free references.
- **Gold factor:** three strata —
  - **A:** dataset-native, short extractive golds (≤15 tokens)
  - **B:** long narrative golds (reference-style, 3–4 sentences)
  - **C:** **length-matched golds** generated by the *same* generator + instruction as the training targets *(new)*
- **Length-control factor:** common-token-budget scoring at 20 and 40 tokens *(new)*.

### Setup
- 300–500 held-out items, identical across all arms and factors.
- Same trained checkpoints as Exp 4 (no retraining).

### Metrics
| Metric | Interpretation |
|---|---|
| FA under each (judge × gold stratum) cell | The factorial result |
| **Spearman ρ between judges**, per stratum | Judge agreement |
| **ΔFA(judge) vs ΔFA(stratum)** — effect sizes on each factor | **Primary: which factor dominates** |
| Common-budget F1 at 20 and 40 tokens *(new primary)* | Length-controlled comparison |
| Precision / recall separately | Density vs coverage |
| σ across judges | Judge fragility |
| Judge-failure rate per cell | Protocol health (§0.1) |

### Success criteria
- Spearman ρ ≥ 0.7 across judges, per stratum.
- **The gold-stratum effect size exceeds the judge effect size** — i.e. length/style matters more than who judges.
- The parametric advantage (if any) persists on **stratum C (length-matched)** as well as stratum B.
- Common-budget F1 at 20 tokens shows parametric ≥ RAG recall (the density result reproduces).

### Falsification criteria
- ρ < 0.5 → results are judge-fragile and no accuracy claim can be made.
- Parametric wins on stratum B (generated/reference-style) but loses on stratum A (native) → **the win is a gold-style artifact**, and the plan's headline claim must be withdrawn or restated as a gold-provenance finding.
- The gap is unchanged under length control → length is not the confound, and the earlier finding does not replicate.

### Threats to validity
1. Judge prompts may themselves bias. Mitigation: identical prompts across judges; run with and without "tolerate paraphrasing" phrasing.
2. Small n per cell (300–500 split by 3 strata × 4 judges). Mitigation: report CIs per cell and flag cells with CI width > 0.05.
3. Cost of many judgments. Mitigation: stratify; judge a random 150 per cell if budget-constrained, flagged as underpowered.

### Estimated cost
**Medium** (API judging dominates). 2–3 weeks.

---

# Experiment 6: Corpus-Size Sweep and the Absorbable Regime *(New — defines the conditions)*

### Hypothesis
There is a **corpus size beyond which retrieval wins**, because parametric accuracy saturates while RAG's stays flat, and the training coverage required for crossover grows super-linearly with corpus size. This defines the "absorbable regime" and is the experiment that converts "KVForge wins" into "KVForge wins **below X chunks**".

**Rationale:** The matched-format work showed the parametric margin shrinking where corpora were larger and denser. The paper asserts crossover across four corpora of 2 000–2 920 chunks but never varies corpus size, so the boundary is unknown. Without it, the central claim is unbounded.

### Baselines
Standard RAG; RAG + prefix caching; RAG + LoRA on native golds.

### Setup
- **One corpus family**, subsampled by the content-hash rule (§0.4) to {500, 2 000, 5 000, 10 000, 25 000, 50 000} chunks.
- **Fixed training recipe** across sizes; tiers scaled to a fixed *fraction* of chunks (e.g. 0.25, 0.5, 1.0 QA per chunk).
- 3 seeds; rank-matched models.

### Metrics
| Metric | Interpretation |
|---|---|
| Parametric FA vs corpus size (at fixed coverage fraction) | Saturation curve |
| RAG FA vs corpus size | Does retrieval degrade or hold? |
| **N\* vs corpus size** | **Primary: the boundary** |
| Crossover corpus size (with CI) | The stated limit of the regime |
| Latency/cost vs corpus size (both arms) | Index and prefill costs grow differently |

### Success criteria
- A crossover corpus size is identified with a CI, and parametric FA is shown to saturate (flat or declining slope) while RAG FA does not.
- The boundary is reported as an explicit number with uncertainty, e.g. "parametric advantage holds below ~X thousand chunks (95% CI …)".

### Falsification criteria
- No boundary within 50 000 chunks → the claim generalises further than expected (a positive result; report it).
- Parametric degrades *faster* than RAG with size → the architecture is only viable for very small corpora, and the paper's four-corpus range must be narrowed.

### Threats to validity
1. Subsample content may differ in difficulty across sizes. Mitigation: content-hash subsampling (result-blind) and report a difficulty proxy (e.g. retriever hit@5) per size.
2. Training budget confounded with corpus size. Mitigation: hold the coverage fraction fixed and report total tokens.

### Estimated cost
**High.** ~6 sizes × 3 seeds. 3–5 weeks.

---

# Experiment 7: Confidence Calibration and Routing Threshold Sensitivity *(Retained, expanded)*

### Hypothesis
The PRS composite (0.5·accuracy + 0.3·calibration + 0.2·consistency) and the Phase-3 gate threshold are well-calibrated and stable across corpora, and sweeping the threshold yields a predictable coverage-vs-quality trade-off.

**Rationale:** The paper fixes thresholds (0.75 gate, 0.80 advancement) with no sensitivity analysis, and §6.5 flags this. Phase 3's autonomy rests on the gate being a *valid* confidence signal; if it is not, routing is unsafe.

### Additions to the previous version
- **Gate AUROC** as a primary discriminative metric (does the gate separate correct from incorrect answers at all?), with a null of 0.5.
- **Selective risk / risk-coverage curves**, the standard evaluation for a reject option.
- **Per-corpus threshold stability** (the previous version's "±0.05" criterion is retained but must be reported with CIs).
- Gate thresholds selected on a **separate validation split**, never on the test set (the previous version flagged this as a threat; it is now a protocol rule).

### Baselines
Fixed gate = 0.75; threshold sweep ∈ {0.5 … 0.9}; PRS-as-confidence; RAG fallback.

### Metrics
Coverage; FA(routed); FA(unrouted); overall FA; **AUROC**; **selective risk**; ECE; precision at high gate; per-difficulty-stratum results.

### Success criteria
- **AUROC ≥ 0.70** for the gate as a correctness predictor.
- Monotone coverage/quality trade-off; an interior optimum identified on validation and confirmed on test.
- ECE < 0.12 across thresholds; optimal threshold varies ≤ 0.05 across corpora.
- At gate ≥ 0.85, precision ≥ 0.90.

### Falsification criteria
- **AUROC ≈ 0.5** → the gate is uninformative, Phase-3 routing is unsafe, and the architecture's autonomous-progression claim fails.
- ECE > 0.20 → confidence scores are unusable.
- Optimal threshold varies > 0.15 across corpora → no portable default; per-domain calibration is mandatory.

### Threats to validity
1. Threshold overfitting. Mitigation: select on validation, confirm on test (protocol rule).
2. Confidence may be uniformly low early in training. Mitigation: run after convergence; report ECE across training.
3. Difficulty imbalance masquerading as calibration. Mitigation: report precision/recall per difficulty stratum.

### Estimated cost
**Low–medium.** 1–2 weeks.

---

# Experiment 8: Churn and Staleness — Where KVForge Loses *(New — honest boundary)*

### Hypothesis
KVForge is strictly worse for high-churn corpora. Every corpus update invalidates the adapter *and* every stored KV tensor, so there exists an update rate λ above which RAG dominates on cost and time-to-serving.

**Rationale:** §6.4 concedes that KV recompute adds ~42% overhead per round. The previous plan listed staleness only as a *threat to validity*; an objective evaluation must **measure the boundary where the system loses**. This is also the condition that, combined with Exp 6, gives a two-dimensional statement of the regime.

### Baselines
Standard RAG with incremental re-embedding of changed chunks only.

### Setup
- Inject churn at λ ∈ {0.1%, 1%, 5%, 20%} of chunks per update cycle, over 8 cycles.
- For KVForge: measure retrain + KV recompute (with and without the tier-ordered warm-chunk shortcut described in §6.4).
- For RAG: re-embed changed chunks only; index update.

### Metrics
| Metric | Interpretation |
|---|---|
| **Time-to-serving after an update** | Primary |
| Cost per update cycle; amortised cost/query at each λ | Primary |
| Accuracy drift post-update (both arms) | Does stale parametric knowledge hurt correctness? |
| **Stale-answer rate** — fraction of parametric answers containing facts that changed | **Safety-relevant** |
| Fraction of KV recomputed | Efficiency |

### Success criteria
- Time-to-serving after a 1%-churn update is under a documented budget (e.g. < 15 min) with the warm-chunk shortcut.
- Stale-answer rate stays below a stated threshold (e.g. < 1%) at λ ≤ 1%.

### Falsification criteria
- Time-to-serving exceeds the update cycle at λ = 5% → the system cannot keep up with realistic churn, and the "stable corpus" condition must be stated as a hard requirement.
- **Stale-answer rate is material at low λ** → a correctness risk that would dominate the cost argument.

### Threats to validity
1. Churn patterns differ from production (which is often concentrated in specific sections). Mitigation: report both uniform and clustered churn.
2. Recomputation may overlap with serving. Mitigation: report both serialised and background modes.

### Estimated cost
**Low–medium.** 1–2 weeks.

---

# Experiment 9: Rewrite Lever Isolation and Multi-Architecture Transfer *(Retained, one criterion fixed)*

### Hypothesis
Diverse rewrites, not LoRA rank or QA volume, drive parametric gains, and this is architecture-agnostic.

**Rationale:** §5.12 reports rewrites worth +0.030 (E2B) and +0.042 (E4B) against +0.009 for a 178× increase in QA volume — three to five times larger, and the most robust result in the paper. §5.13 extends it to Llama-3.2-3B. This experiment is retained because it is the paper's best-evidenced claim; the fixes below are about making the criterion match the claim.

### Fixed from the previous version
The previous success criterion read:

> *"No scale confound: FA gain from rewrites (Arm 3 − Arm 2) ≥ 80% of gain from scale (Arm 3@8B − Arm 3@3B)"*

That states rewrites are **at least 80% as powerful as** scale. The paper's claim is that rewrites **dominate** scale. As written, a result satisfying the criterion would **not** support the claim. The criterion is inverted here.

Also: **self-rewrite controls** (each model generating its own rewrites), which §5.12 flags as planned-but-unrun, are now required rather than optional — otherwise generator quality is confounded with diversity.

### Baselines
1. LoRA on basic Q&A (1 per chunk).
2. LoRA on diverse-rewrite Q&A.
3. LoRA on **volume-matched** basic Q&A (same *token* count as the rewrite arm) — new, and necessary to separate diversity from volume.
4. RAG (untrained).
5. Scale control at matched rank.

### Setup
- **Models:** Gemma-4 E2B, Gemma-4 E4B, Llama-3.2-3B — **rank-matched** at r=64.
- **Rewrite generators:** two frontier generators + **self-rewrites** per model.
- **Token matching:** the volume-matched arm is the key control. Without it, "rewrites beat volume" is not established; only "rewrites beat a particular volume arm" is.

### Metrics
FA (length-controlled, §0.1); **ΔFA(rewrites)** and **ΔFA(scale)** with paired CIs; rewrite diversity (distinct-3-gram ratio / embedding dispersion); token count per arm; latency.

### Success criteria
- **ΔFA(rewrites) > ΔFA(scale)** with non-overlapping CIs at matched rank and matched tokens.
- **ΔFA(rewrites vs volume-matched basic QA) > 0.02** with CI excluding 0 — this is the actual "diversity, not volume" claim.
- Effect reproduces across ≥2 of 3 architectures.
- **Self-rewrites retain ≥ 50% of the frontier-generated rewrite gain** — otherwise the effect is generator-quality, not diversity.

### Falsification criteria
- ΔFA(rewrites) ≈ ΔFA(volume-matched QA) → the lever is volume, not diversity.
- ΔFA(scale) > ΔFA(rewrites) with non-overlapping CIs → scale dominates, contradicting §6.1.
- Self-rewrites yield no gain → the lever is generator quality and the paper must say so.

### Threats to validity
1. Generator quality confounded with diversity. Mitigation: self-rewrite control (now required), plus an explicit diversity metric.
2. Rank may interact with model size. Mitigation: report r ∈ {32, 64, 128} on the smallest model.
3. Judge/gold confounds. Mitigation: §0.1 applies.

### Estimated cost
**Medium.** 2–3 weeks.

---

# Experiment 10: Phase 2 KV Injection with Position/Attention Realignment *(Retained)*

### Hypothesis
Phase 2 can become quality-preserving, rather than known-negative, when equipped with the corrections TurboRAG, CacheBlend, or APE use.

**Rationale:** §3.1 and §6.5 honestly present uncorrected KV injection as a known negative, and identify position/attention realignment as the highest-value follow-up. This is the experiment that converts an honest negative into a possible positive.

### Baselines
1. Phase 2 without corrections (known negative).
2. Phase 2 + attention-mask/position-ID realignment (TurboRAG-style).
3. Phase 2 + selective recompute of high-deviation tokens (CacheBlend-style).
4. Phase 1 (quality and latency reference) and Phase 3.

### Additions to the previous version
- **Report per-paper corrections separately**, not bundled — bundling would hide which correction does the work.
- **Latency must be measured at matched output length**, since TTFT gains can be erased by longer generation.
- **`attention_mask` and position-ID correctness is asserted as a precondition**, not assumed.

### Metrics
Length-controlled FA vs Phase 1 (non-inferiority, margin −0.02); TTFT p50/p95; total latency at matched output length; KV recompute overhead; memory bandwidth; peak VRAM.

### Success criteria
- **Non-inferiority:** lower bound of the 95% CI on ΔFA vs Phase 1 > **−0.02**.
- TTFT ≤ 0.6× Phase 1 at matched conditions.
- Each correction's individual contribution is reported.

### Falsification criteria
- ΔFA vs Phase 1 worse than −0.05 → the trade-off is unfavourable and Phase 2 should remain deprecated.
- Corrections add overhead that erases the TTFT gain at matched output length.

### Threats to validity
1. Unequal tuning effort across corrections. Mitigation: use published configurations; state any custom tuning.
2. RoPE vs ALiBi differences make corrections architecture-specific. Mitigation: test on ≥2 architectures.
3. Interaction with Phase 3 LoRA. Mitigation: Phase 3 disabled here; interaction noted as separate work.

### Estimated cost
**High** (engineering-heavy). 3–4 weeks.

---

# Experiment 11: External Baseline Comparison *(Revised — reprioritised and reframed)*

### Changes from the previous version, and why
- **Reprioritised from #4 to last.** It was rated "High Impact", but the paper's own §3.1 concedes Phase 2 loses to TurboRAG/CacheBlend, so a *quality* win against them is unlikely. Running it first would consume the most effort for the most predictable outcome.
- **Reframed from superiority to non-inferiority + cost/latency superiority.** The previous criterion — *"FA(KVForge) ≥ FA(best-other-method) on both datasets"* — is an extremely strong claim that will almost certainly fail on quality. And it is not apples-to-apples: **Self-RAG is an adaptive-retrieval method; it retrieves by design.** Comparing a parametric-only system against retrieval-based systems on quality alone misrepresents both.
- **Parity rule added** (§0.3): matched hardware, base model, query set, prompt budget.
- **Effort estimate raised** from "$1K–2K / 5–6 weeks" to **8–12 weeks and $3–5K**: faithfully reproducing TurboRAG *and* Self-RAG *and* LMCache is three research-grade implementations.

### Hypothesis
At matched hardware, base model, and query set, KVForge Phase 3 is **non-inferior in accuracy** to each external baseline **at that baseline's own designed operating point**, while being superior on p99 latency and on cost at high query repetition.

### Baselines
1. Standard RAG (+ prefix caching).
2. TurboRAG (Phase 2's architectural peer).
3. CacheBlend.
4. A KV-quantisation system (e.g. LMCache).
5. A long-context single-shot configuration (separate tier; not style-matched).
6. Published-weights adaptive-retrieval method (e.g. Self-RAG), evaluated as a *retrieval* system.

### Metrics
Length-controlled FA; p50/p95/p99; $/1M queries (Exp 1 protocol); peak VRAM; throughput-at-SLO; **staleness** (Exp 8 protocol); and — new — the **operating point** of each system (coverage, cache hit rate).

### Success criteria
- **Non-inferiority on accuracy** (margin −0.02) against each baseline at its own operating point.
- **Superiority on p99** and on **$/1M queries at α ≥ 0.8** (Exp 1 protocol).
- Reported per baseline, not pooled, so partial wins are visible.

### Falsification criteria
- Accuracy loss beyond −0.05 against any baseline.
- A baseline dominates KVForge on **both** accuracy and cost at high repetition → the value proposition is not established against the relevant prior art.
- Any reproduction cannot be validated against the original paper's reported numbers (report the reproduction gap explicitly).

### Threats to validity
1. Reproductions may not match the original systems. Mitigation: validate each against its published numbers first; report the reproduction gap.
2. Unequal tuning effort. Mitigation: pre-register tuning budgets; report them.
3. Hardware sensitivity. Mitigation: state hardware; note that vendor-reported speedups are measured on different hardware (e.g. LMCache on 8×H100).

### Estimated cost
**Very high.** 8–12 weeks, $3–5K.

---

# Experiment 12: Retrieval-Hard Conditions and Robustness *(Retained, demoted to a boundary condition)*

### Hypothesis
Because knowledge is in the weights, Phase 3 is more robust than RAG when retrieval is degraded — paraphrased queries, noisy or diluted retrieval, and distractor-heavy contexts.

**Rationale:** Phase 3's independence from retrieval is an architectural claim with no direct test. Retained, but as a **boundary/adversarial condition** rather than a headline, and with multi-hop moved here (from Exp 4).

### Conditions
- **A — Paraphrase:** 3–5 paraphrases per question, degree of semantic drift measured.
- **B — Noise injection:** 30% of retrieved chunks replaced with **high-similarity non-answers** (reranked confusers), not random text — random injection is not a realistic retrieval failure.
- **C — Distractor dilution:** top-K raised to 50 with mostly irrelevant content.
- **D — Multi-hop (moved from Exp 4):** HotpotQA-style questions requiring evidence composition.
- **E — Out-of-domain:** corpus held fixed, questions from a different domain.

### Metrics
ΔFA(clean → degraded) per arm; **robustness ratio** = ΔFA(KVForge) / ΔFA(RAG); retrieval-rank vs FA correlation, **for the parametric arm specifically** (if the parametric arm's FA correlates with retrieval rank, it is not retrieval-independent); gate behaviour under degradation; overconfidence rate.

### Success criteria
- Robustness ratio < 1.0 on ≥2 conditions (parametric degrades less).
- Parametric FA is **uncorrelated** with retrieval rank (the architectural claim, tested directly).
- Gate confidence falls when retrieval degrades → fallback fires.

### Falsification criteria
- Robustness ratio ≈ 1.0 → Phase 3 is no more robust than RAG; the "retrieval-independent" claim is unsupported.
- Parametric FA correlates with retrieval rank → retrieval independence is false.
- Overconfidence under degradation → the gate is unsafe precisely when it matters.

### Threats to validity
1. Paraphrases may be too mild. Mitigation: measure semantic distance; report the distribution.
2. Noise realism. Mitigation: reranked confusers; report similarity distribution of injected chunks.
3. Gate threshold fixed. Mitigation: report sensitivity (links to Exp 7).

### Estimated cost
**Low–medium.** 1–2 weeks.

---

## Summary: Ranking and Resource Allocation

Ordered by expected impact per unit of effort. Note the deliberate inversion of the previous ranking: the two cheapest experiments (1 and 3) are now ranked first, and the most expensive (11) last.

| Rank | Experiment | Why it matters | Effort |
|---|---|---|---|
| **1** | **Exp 1: Amortisation & break-even N\*** | The paper's central economic premise, asserted but never measured. No training needed. | Low |
| **2** | **Exp 3: CPU-only serving** | Potentially the strongest single differentiator; no RAG baseline can match it. Very cheap. | Very low |
| **3** | **Exp 2: Tail latency under concurrency** | Where KVForge genuinely wins; currently under-sold (p50 reported, tail not). | Low–med |
| **4** | **Exp 5: Judge **and** gold-length invariance** | Closes the accuracy claim's largest hole; the length factor is the real confound. | Medium |
| **5** | **Exp 7: Gate calibration** | Phase-3 autonomy is unsafe if the gate's AUROC ≈ 0.5. | Low–med |
| **6** | **Exp 4: Crossover + corpus-size sweep** | Validates the core claim on public data, with the 100-chunk flaw fixed. | High |
| **7** | **Exp 8: Churn/staleness** | Defines where KVForge *loses*; honest and cheap. | Low–med |
| **8** | **Exp 6: Absorbable-regime boundary** | Turns "KVForge wins" into "wins below X chunks". | High |
| **9** | **Exp 9: Rewrite lever** (criterion fixed) | The paper's best-evidenced claim; keep, but fix the inverted test and add token matching. | Medium |
| **10** | **Exp 10: Phase 2 + realignment** | Converts an honest negative into a possible positive. | High |
| **11** | **Exp 12: Retrieval-hard robustness** | Tests the retrieval-independence claim directly. | Low–med |
| **12** | **Exp 11: External baselines** | Required for credibility, but the most expensive and least likely to favour KVForge on quality. | Very high |

### If only two experiments can be run

**Exp 1 + Exp 3.** Both reuse existing adapters, need no training, and together support a claim the plan previously could not make:

> For a **stable corpus up to a few thousand chunks** with a **repetitive query distribution**, KVForge delivers **factual-accuracy non-inferiority** (CI lower bound > −0.02) at **2–3× lower p95 latency** and lower **$/1M queries beyond break-even volume N\***, and can be **served without a GPU** — while being **strictly worse** for high-churn corpora and long-form extractive answers.

That claim is checkable, falsifiable, and — unlike "strictly dominates retrieval", which the diagnostics suggest is partly a gold-length artifact — likely to survive review.

---

## Threats to Validity (Cross-Experiment)

### Systematic
1. **Accuracy claims rest on golds and judges that are not neutral.** Mitigation: Exp 5's factorial design; §0.1 length control; judge-free primaries.
2. **Corpus selection.** Mitigation: content-hash subsetting (§0.4); report every domain attempted; no post-hoc domain dropping.
3. **Single seed.** Mitigation: ≥3 seeds for every trained arm; any single-seed result is labelled and cannot support a headline claim.
4. **Base model / architecture.** Mitigation: ≥2 model families at matched rank (Exp 9); align with the paper's regime unless justified.
5. **Hardware regime shift.** Mitigation: parity rules (§0.3); report hardware with every latency/cost number.
6. **Generator-quality confound in training data.** Mitigation: self-rewrite controls (required in Exp 9).

### Statistical
7. **Multiple comparisons.** Mitigation: pre-register ≤3 primary tests per experiment with correction; report the rest as exploratory at raw p-values; emphasise effect sizes and CIs.
8. **Underpowered cells.** Mitigation: report CI width; label any cell with CI > ±0.05 as underpowered rather than reporting a win or a loss.
9. **Metric shopping.** Mitigation: declare the primary metric per experiment before running; report all others as secondary.

### Operational
10. **Reproducibility.** Mitigation: tag exact model/API/LoRA versions; publish corpus manifests with md5s; publish splits; archive a tagged release (per §6.5, a DOI-tied commit).
11. **External-service drift.** Mitigation: log judge/generator API versions; re-run key judged experiments with a pinned snapshot.
12. **Harness drift.** Mitigation: one declared FA formula per experiment; never compare numbers produced under different formulas (the repository has contained at least two).

---

## Next Steps

1. **Run Exp 1 and Exp 3 first** — cheapest, highest information, no training, reuse existing adapters.
2. **Implement §0.1–0.4 as shared harness code before any judged experiment.** Every accuracy number produced without length control and judge-failure handling is unusable.
3. **Run Exp 5 early on existing records** — it can be executed as a **re-scoring** of already-collected outputs, at near-zero cost, and it determines whether Exp 4's and Exp 6's expensive training runs are worth starting.
4. **Run Exp 7** once Exp 5 establishes a valid metric.
5. **Defer Exp 11** until Exp 1/2/3 have established the value proposition; if the cost/latency case is not established, an expensive external-baseline comparison will not rescue it.
6. **Reporting:** Exp 1, 2, 3, 5, 7 in the main paper (system value); Exp 4, 6, 8, 9, 12 in the main paper or appendix per results; Exp 10, 11 in appendix unless they produce positive results.
7. **Contingency:** if Exp 1 shows N\* is not finite within 10⁶ queries, pivot the paper's framing from "cheaper at scale" to "faster and more predictable per query", and re-scope the contribution accordingly. If Exp 5 confirms the gold-length confound dominates, restate every accuracy claim as gold-conditional.

---

## References

- **KVForge paper (§1–7):** architecture, three-phase design, PRS, crossover results, limitations.
- **Preceding work cited by this plan:** TurboRAG (Geng et al.), CacheBlend (Yao et al.), LMCache, APE, Self-RAG (Asai et al.), PromptCache, PagedAttention/vLLM, RETRO, FiD, DPR, ColBERT.
- **Benchmarks:** HotpotQA (Yang et al.) — now used as an adversarial multi-hop condition, not a primary benchmark; Natural Questions (Kwiatkowski et al.); technical-documentation and fact-dense single-hop corpora (primary).
- **Diagnostic basis for this revision:** the matched-format study and its output diagnostics — UC1 gap +0.2714 → +0.0593/+0.0286 under fair measurement; the expertqa/techqa/emanual gold-length mismatch (4.5× / 3.4× / 2.1× vs 0.9× on UC1); judge CORRECT rate rising 0.020 → 0.846 with gold recall; 730/2152 (33.9%) judge calls failing and being scored as incorrect, biased 15–33% (short) vs 50–63% (long); parametric higher-precision/lower-recall with equal-or-higher recall at a common token budget; RAGBench duplicate question rows (emanual 244 → 122 distinct, techqa 118 → 59).
