# Matched-Format Experiment: Long vs Short Answers on Identical Rewritten Chunks

**Status:** DESIGN — pre-registration draft, not yet run
**Scope:** KVForge parametric consolidation, Gemma-4-E4B (4-bit), LoRA r64/α128
**Owner:** KVForge research
**Design:** two arms — an **internal** run on UC1 (explains an existing effect) and an **external** run on a pre-registered RAGBench domain panel (tests whether the explanation generalises to a dataset chosen without reference to KVForge results). See §3.3 and §11.
**Supersedes in authority:** the README/paper claim that "RAG beats parametric" (corpus-specific, not general)

---

## 1. Motivation — the observation this experiment has to explain

Across the corpora KVForge has evaluated, the parametric-vs-RAG verdict **flips by corpus**, and the flips line up with answer format rather than with corpus topic:

| corpus | answer shape | judged answer length | RAG | parametric | winner |
|---|---|---|---|---|---|
| UC1 Customer Support | free-form prose | multi-sentence | FA 0.1525 | **FA 0.4238** | parametric |
| UC4 Bedrock docs | free-form prose | multi-sentence | F1 0.163 / judge 0.000 | F1 0.198 / judge 0.286 | parametric *(n=7 only)* |
| UC2 PubMedQA | categorical (yes/no/maybe) | 1 word | **0.680** | 0.365 | RAG |
| UC3 SQuAD | extractive span | 1–3 words | **FA 0.6535** | FA 0.2374 | RAG |

The UC1 number is the only one of these measured on a clean held-out set with a single harness; the rest carry known defects (§12). The pattern is suggestive but **not established**, because every row differs in corpus, model era, harness, sample size, and FA formula simultaneously. Answer format has never been isolated.

**This experiment isolates exactly one variable: answer format.**

---

## 2. Objective

> Determine whether the performance gap between parameterically-consolidated answering and retrieval-augmented answering **depends on the format of the target answer** — specifically whether parametric answering outperforms RAG when gold answers are long-form/free-form, and underperforms when gold answers are short/extractive — holding the corpus, chunks, rewrites, questions, model, recipe, and evaluation harness fixed.

### 2.1 Hypotheses

Let `FA(arm, format)` be factual accuracy on the held-out set, where `arm ∈ {par, rag}` and `format ∈ {short, long}`.

**H1 (interaction — the primary hypothesis).** The parametric-minus-RAG gap is larger for long answers than for short answers:

```
D = [FA(par,long) − FA(rag,long)] − [FA(par,short) − FA(rag,short)]
H1: D > 0
```

**H1a (directional sub-claims).** `FA(par,long) > FA(rag,long)` and `FA(par,short) < FA(rag,short)`.
These are *predictions*, not requirements — H1 can hold on the interaction alone.

**H2 (mechanism — secondary).** Long-form training produces answers that are penalised by surface metrics without being factually worse. Tested by comparing the **token-F1 gap** against the **judge gap**: predict that `gap_F1 < gap_judge` for the long format (surface metric over-penalises verbosity).

**H3 (latency — secondary).** The parametric arm's latency advantage is format-invariant: parametric is faster in both formats, because it avoids the retrieval round-trip.

### 2.2 What would falsify the hypothesis

Stated in advance so the result cannot be rationalised after the fact:

- **Reject H1** if the 95% CI on `D` includes zero, **or** if `|D| < 0.05` FA absolute (the pre-registered minimum effect of interest).
- **Strong falsification** if parametric loses in *both* formats with similar margins (`D ≈ 0`), which would say answer format is not the operative variable and the UC1 result needs another explanation.
- **Reject H2** if the F1 gap and judge gap move together for the long format.

A finding of "RAG wins in both formats" is a legitimate and publishable outcome — it would mean consolidation does not beat retrieval on this corpus at this scale regardless of format, and it would strengthen rather than weaken the paper's honest-floor framing.

---

## 3. Experimental design

**2 × 2 factorial, fully crossed, within-corpus and within-question.**

| | short-answer target | long-answer target |
|---|---|---|
| **parametric** (closed-book, LoRA) | `par_short` | `par_long` |
| **RAG** (base model + retrieved context) | `rag_short` | `rag_long` |

### 3.1 The controlled comparison

Everything except answer format is held identical **within each column and across rows**:

- **Identical corpus** in the Qdrant index for all four cells.
- **Identical chunk set** used to generate training data.
- **Identical rewrites** — the *same* `rewrites.json` feeds both adapters. This is the "same rewritten chunks" condition and is what makes the contrast matched: `par_short` and `par_long` see byte-identical replay/rewrite content and differ **only** in the QA pairs' answer format.
- **Identical questions** — for every chunk the same question stems are used; only the answer rendered for that question differs. This makes the comparison **paired at the question level**, eliminating question-difficulty as a confound.
- **Identical recipe** — E4B, r64/α128, dropout 0.05, lr 2e-4, 1 epoch, from base, same seed, same effective batch.
- **Identical harness** — one evaluation script, one FA formula, one judge, one rubric.
- **Identical held-out set** — the same 200 questions are scored against the format-matched gold for each cell.

### 3.2 Why not just use two different datasets

Because that reintroduces exactly the confounds that make the current UC1–UC4 table uninterpretable: different domains, chunk counts, retrieval difficulty, answer distributions, model eras, and metric definitions. A within-corpus matched-format design trades external validity (it is one corpus) for internal validity (format is genuinely the only difference). **A matched-format result on one corpus is worth more than four unmatched results on four corpora.**

### 3.3 Corpus choice: one internal, one external

Two corpora, with deliberately different jobs:

| role | corpus | why this corpus |
|---|---|---|
| **internal (primary)** | **UC1 Customer Support** (`customer-support`, 2,000 chunks) | the only corpus with a statistically solid *clean* parametric win to explain (n=200, paired 179 won / 19 lost, FA 0.4238 vs 0.1525); already fully built (data, held-out set, adapter, baselines); and in the absorbable size regime (279,766 words) |
| **external (validation)** | **RAGBench**, pre-registered 3-domain panel (§11) | never used to select it, so it answers the circularity objection that UC1 and UC4 are precisely the corpora where parametric already won |

The internal run **explains a known effect**. The external run **tests whether that explanation generalises** to a dataset chosen without reference to KVForge results.

> **Reporting rule.** The internal result alone is not sufficient evidence for the hypothesis, because it is the corpus where the effect was found. **A claim is reportable only if the external panel does not contradict it.** If the panel contradicts it, that contradiction is reported with equal prominence — the internal result does not get to outrank it.

---

## 4. Variables

### Independent
1. **Answer format** ∈ {`short`, `long`} — the variable of interest.
2. **Inference mode** ∈ {`parametric`, `rag`} — the arms being compared.

### Dependent
| metric | definition | role |
|---|---|---|
| **FA** | `0.5 × token_f1 + 0.5 × judge` (this harness's formula — see §6.2) | **primary** |
| judge correctness | binary CORRECT/INCORRECT from the LLM judge | primary component |
| token-F1 | `eval.metrics.token_f1(pred, gold)` | secondary, format-sensitive |
| cosine similarity | embedding cosine pred vs gold | secondary |
| latency | wall-clock per query, s | secondary |
| answer length | words in generated answer | diagnostic |
| refusal rate | fraction of empty/non-answers | diagnostic |

### Controlled (held fixed)
corpus · chunk set · rewrite set · question set · base model · quantization · LoRA rank/alpha/dropout/lr/epochs/seed · effective batch size · retrieval `top_k=5` · prompt template · judge model · judge rubric · decoding (`do_sample=False`) · hardware.

### Deliberately *not* controlled
**Total training tokens.** The long-format arm necessarily sees more target tokens per example. This is intrinsic to the variable: format *is* supervision density. We therefore match on **number of training examples** (primary) and additionally report a **token-matched** variant (`long` arm subsampled so total target tokens match `short`) as a robustness check. If H1 holds under both, the effect is not merely a token-count artifact.

---

## 5. Data construction protocol

All artifacts live under `examples/usecase1_customer_support/matched_format/`.

### 5.1 Corpus and splits

1. **Chunk split, disjoint.** `chunk_id % 5 == 4` → eval chunks (400); remainder → train chunks (1,600). Identical split for both formats.
2. **Assert disjointness** before generating anything; record `train ∩ eval = ∅`.
3. Provenance: the generator renumbers chunks 0..N−1, so write index maps (`train_chunk_index_map.json`, `eval_chunk_index_map.json`) to recover original `chunk_id` for every generated item.

### 5.2 Rewrites (shared by both arms)

4. Generate **one rewrite per train chunk** from the 1,600 train chunks, with `--rewrites-per-chunk 1 --style dialogue`. This single `rewrites.json` is used verbatim by both adapters.

### 5.3 Twin QA generation — the core manipulation

5. For each train chunk, generate **the same question stems** twice, once with a short-answer instruction and once with a long-answer instruction:

| | target | instruction (summary) |
|---|---|---|
| SHORT | a phrase, ≤ 5 words, the key fact | *"Answer with only the key fact as a short phrase, at most 5 words."* |
| LONG | 2–4 sentences, self-contained explanation | *"Answer in 2–4 sentences as a support agent would, stating the fact and the steps/caveats around it."* |

6. **Question identity check (required):** the question text must match across formats after normalization. Any question that differs between the two generations is **dropped from both**, so the paired structure is preserved. Record the drop count.
7. **Length separation check (required):** assert the two sets are actually different formats — report the word-length distributions and require `median(long) ≥ 3 × median(short)`. If generation collapses to similar lengths, the manipulation failed and the run is void.
8. Target yield: ~5 QA per chunk per format over 1,600 chunks → ~8,000 QA per format, ~16,000 total.

### 5.4 Leakage and duplicate control (both formats)

Applied identically to each format's training set:

9. **Within-training dedup** — exact normalized question, then near-duplicate at Jaccard ≥ 0.80 on **question and answer**.
10. **Against the held-out set** — every surviving training item checked against the held-out questions at the same thresholds; overlaps removed.
11. **Report the audit** — exact removed, near removed, survivors, distance from generation, and the **max train↔eval Jaccard before and after**. This is not optional: on the UC1 pilot the pre-filter max Jaccard was **1.000** despite a chunk-disjoint split (the corpus is templated, so identical questions arise from different chunks), and **71.6 %** of eval candidates had to be removed.
12. **Residual-leakage disclosure:** report the post-filter max train↔eval Jaccard. On the pilot it was 0.786 — compliant with the 0.80 gate but only just. If it exceeds 0.75, run an additional stricter pass at 0.70 and report both.

### 5.5 Held-out evaluation set

13. Generate eval candidates from the **400 eval chunks** (format-independent questions).
14. Cap to a **verified-distinct** set: report `raw n` and `distinct n` at J ≥ 0.90/0.80/0.70. Do **not** report a nominal n larger than the distinct count supports. Pilot measured `743 → 549` distinct at 0.80 and `50 → 5` at 0.70 on UC1 data, so redundancy is real and large.
15. Produce two gold-answer variants per held-out question: `eval_short.json` and `eval_long.json`, **same questions, same order**, differing only in the `answer` field. Both must have identical `question` arrays — assert this.

### 5.6 Expected artifact sizes (planning)

| artifact | expected |
|---|---|
| train chunks / eval chunks | 1,600 / 400 |
| shared rewrites | ~1,600 |
| `qa_pairs_short.json` | ~8,000 |
| `qa_pairs_long.json` | ~8,000 |
| `eval_short.json` / `eval_long.json` | ~200 each, matched questions |

---

## 6. Training protocol

### 6.1 Commands

Two adapters, identical except for the `--faqs` file:

```bash
UC=examples/usecase1_customer_support
MF=$UC/matched_format

# SHORT-answer adapter
python pipeline/lora_trainer.py \
  --config $UC/config_e4b_matched.json \
  --faqs   $MF/train/qa_pairs_short.json \
  --rewrites $MF/train/rewrites.json \
  --seed 42 --from-base \
  --checkpoint-dir $UC/lora_checkpoints/e4b_mf_short/

# LONG-answer adapter  (same rewrites, same seed, same recipe)
python pipeline/lora_trainer.py \
  --config $UC/config_e4b_matched.json \
  --faqs   $MF/train/qa_pairs_long.json \
  --rewrites $MF/train/rewrites.json \
  --seed 42 --from-base \
  --checkpoint-dir $UC/lora_checkpoints/e4b_mf_long/
```

### 6.2 Recipe

| parameter | value | rationale |
|---|---|---|
| base model | `google/gemma-4-E4B-it` | matches the current arm |
| quantization | 4-bit | A10G 24 GB |
| `lora_rank` / `lora_alpha` | 64 / 128 | the established E4B recipe |
| `lora_target_modules` | `q_proj, k_proj, v_proj` | as elsewhere |
| `lora_dropout` | 0.05 | |
| `lora_epochs` | 1 | matches the SQuAD/UC1 E4B arms |
| `lora_lr` | 2e-4 | |
| seed | 42 (primary), 43/44 (replication) | |

### 6.3 Trainer preconditions (verify in the log)

16. The log **must** print `SFT max_length=<n>` where `n` covers the full rendered sequence. The original trainer sized this from the *answer* alone and on UC1 dropped **5.21 % of answer tokens across 22.4 % of examples**; it has been fixed to size from p99 of the rendered sequence (floor 64). For the LONG arm this is critical — long answers make the old sizing catastrophic.
17. **Assert zero examples with all-masked labels** (`_has_supervision` guard). A fully-masked example contributes no gradient and previously passed the dataset filter silently.
18. Record `train_loss`, step count, runtime, and the rendered-length distribution **separately for each arm**, so a length-driven compute difference is visible.

---

## 7. Evaluation protocol

### 7.1 Arms

Four cells, one harness invocation per cell:

```bash
# parametric cells: --checkpoint points at the matching adapter
# rag cells: same config, --modes text_rag (base model + top-5), format-matched prompt
python measure_baseline_fkds.py \
  --config $UC/config_e4b_matched_eval.json \
  --eval-set $MF/eval_short.json \
  --output-dir results/mf/short_parametric \
  --modes parametric \
  --judge-model deepseek-code-h --judge-provider openai \
  --judge-base-url <CodeVector gateway> --judge-api-key <key> \
  --checkpoint $UC/lora_checkpoints/e4b_mf_short/
```

Cell matrix: `{short,long} × {parametric, rag}`, each on its format-matched held-out set, with the **same 200 questions**.

### 7.2 RAG arm definition (must be pinned, not defaulted)

19. **Base model, no adapter** — set `rag_model: "base"` so the RAG arm is a genuine "no fine-tuning needed" baseline.
20. **Force text-in-context** — set `force_text_rag: true`. Without it `answer_with_retrieval()` may select KV injection, and the `customer-support` collection's KV predates the current model; the mean-pool path only null-checks the tensor and never validates shape, so a cross-model injection would silently corrupt the RAG arm.
21. **Same generation path as parametric** — remove `vllm_url` so both arms use the local HF model. Routing RAG through vLLM and parametric through HF would be a serving-path confound.
22. `top_k = 5`, `do_sample=False`, `max_new_tokens` matched to the format.

### 7.3 Judge rubric — must be format-neutral

23. The judge is asked **factual correctness only**, with an explicit instruction not to reward or penalise length:

> *"Judge whether the candidate answer is factually correct with respect to the reference, and whether it contains the specific fact the question targets. Judge only factual content. Do NOT penalise length, verbosity, or wording differences — a correct short answer and a correct long answer both score 1.0."*

24. Judge mode: binary CORRECT/INCORRECT (deterministic, `temperature=0`). A partial-credit mode may be reported additionally but not substituted.
25. **One judge for all four cells**, fixed for the whole experiment. Never compare FA across judges — prior work in this project saw a ~2× leniency difference between judge models.

### 7.4 FA formula — declare one and use it everywhere

26. **`FA = 0.5 × token_f1 + 0.5 × judge`** (the `measure_baseline_fkds.py` definition).
    ⚠️ The repo currently has **two incompatible FA definitions**: `tools/squad_eval.py` uses `0.5 × judge + 0.5 × cosine`. AGENTS.md documents only the cosine version. **All four cells must use one harness**; state which formula every reported number uses, and do not compare these FA values against the retired SQuAD FA values.
27. Report the judge component and the token-F1 component **separately** so a reader can recompute FA under either definition.

### 7.5 Statistical analysis plan

28. **Paired design.** All cells share the same 200 questions, so analyse pairwise on questions, never as independent aggregates.
29. **Primary test:** paired bootstrap (10,000 resamples) on `D` (§2.1), reporting the 95 % CI. Reject H1 if the CI contains 0.
30. **Judge-correctness test:** McNemar exact test on the 2×2 concordance table for each contrast (`eval/stats.py:mcnemar_exact`).
31. **Replication:** repeat training at seeds 43 and 44 and report mean ± 95 % CI across seeds, so the result is not a single-checkpoint artifact.
32. **Report** for every cell: n, effective distinct n, judge model, FA formula, mean/SEM of each component, latency mean/p50/p95, and answer-length distribution.
33. **Pre-register** this analysis plan before running; deviations get reported as deviations.

---

## 8. Assumptions

Each assumption is paired with the check that would expose it.

| # | assumption | how it is checked / falsified |
|---|---|---|
| A1 | The corpus is absorbable at E4B/r64 — i.e. parametric *can* learn this content | `par_short` must beat a base-model closed-book floor; if both adapters sit at the no-training floor, the corpus exceeds capacity and the experiment is void |
| A2 | The short/long manipulation actually produces two distinct formats | §5.3 step 7 — require `median(long) ≥ 3 × median(short)` |
| A3 | Questions are identical across formats | §5.3 step 6 — normalized-question set equality asserted |
| A4 | The judge rubric is format-neutral | anchor the judge with control items: a correct short answer and a correct long answer to the same question must both score 1.0 |
| A5 | Retrieval quality is adequate and not the bottleneck | report retrieval hit rate / whether the gold chunk appears in top-5; if it is low, RAG is handicapped and the comparison is confounded |
| A6 | token-F1 penalises verbosity rather than correctness | this is H2, tested not assumed |
| A7 | Teacher-generated answers are factually grounded in the chunk | verify a sample against source chunks; on UC1 pilot the generic prompt produced meta-questions ("what does the customer ask about?") — the dialogue style was required |
| A8 | Held-out answers are derivable from the corpus | eval questions come from eval chunks and are deduped against training; residual overlap disclosed (§5.4) |
| A9 | 200 distinct questions give adequate power | report the CI; if too wide to exclude the 0.05 threshold, say so rather than claiming a null |
| A10 | Teacher model change does not invalidate the contrast | both formats use the **same** teacher, so a quality shift affects both arms equally; it limits cross-corpus comparability, not the internal contrast |
| A11 | The 2,000-chunk RAGBench subset is representative of its parent corpus | no claim about the full corpus is made; the subset is the declared population and corpus size is reported with every result |
| A12 | RAGBench questions are answerable from their bundled documents | check the groundedness/adherence labels of the shipped responses; if a domain's responses are largely unsupported, its documents are insufficient and the domain is dropped — by that rule, declared in advance |
| A13 | 600-char chunking is appropriate for RAGBench documents | RAGBench documents run 2.2k–25.4k chars median by domain; 600/60 matches UC1's convention, and a per-domain chunk-size sensitivity check at 300/1200 chars is reported if chunking is shown to matter |

---

## 9. Evaluation standard — what counts as proof

| outcome | interpretation |
|---|---|
| `D > 0`, 95 % CI excludes 0, `|D| ≥ 0.05`, holds at ≥2 seeds, judge-free metrics agree in direction | **H1 supported.** Answer format is a moderator of the parametric-vs-RAG gap. |
| `D > 0` but CI includes 0, or holds at only one seed | **inconclusive — report as such**, do not claim support |
| `D ≈ 0`, both formats show the same sign and similar margin | **H1 rejected.** Format is not the operative variable; the UC1 result needs a different explanation. |
| Parameters lose in both formats | **H1 rejected.** Strengthens the honest-floor framing: consolidation does not beat retrieval here regardless of format. |
| Format effects appear in token-F1 but not judge | **H2 supported** — a metric artefact, not a capability difference. Report prominently; it changes how all prior token-F1 comparisons should be read. |

**Minimum reporting bar for any claim:** n, effective distinct n, judge model, FA formula, seeds, CI, and the leakage audit. A claim missing any of these is not reportable.

---

## 10. Dependencies

### Compute
- 1 × A10G GPU (24 GB) — same box as prior runs.
- E4B 4-bit training peaks ~10.5 GB, so two adapters run sequentially, never concurrently.
- Disk: ~2 GB per adapter (checkpoints + final adapter); budget ~10 GB for artifacts.

### Models and weights
- `google/gemma-4-E4B-it` (HF, 4-bit) — already cached (~15 GB).
- Tokenizer from the same repo (used to verify the max-length fix).

### Data
- **Internal:** UC1 corpus `examples/usecase1_customer_support/data/corpus.jsonl` (2,000 chunks); Qdrant collection `customer-support` (2,000 points) for the RAG arm.
- **External:** `galileo-ai/ragbench`, pre-registered panel `expertqa` / `techqa` / `emanual` (§11.3). Documents arrive inside the dataset rows, so corpus construction is local. Questions come from `validation` + `test`; `train` questions are never used as targets.
- Per-domain Qdrant collections built from the subset, plus `subset_manifest.json` recording the ordered-chunk md5 so the subset is reproducible byte-for-byte.

### Teacher / judge API
- **CodeVector gateway** (`codevector_creds.json`), OpenAI-compatible.
- ⚠️ **Only `deepseek-code-h` is currently granted.** `deepseekv4p` — used for all prior corpora — is retired (HTTP 403). Gemini and OpenAI keys are empty in a scripted shell; the Anthropic key returns HTTP 400 (credit exhausted).
- **This forces generator == judge** (§12, T1).

### Code (with the fixes this experiment depends on)
| component | required state |
|---|---|
| `pipeline/lora_trainer.py` | max-length sized from p99 of the rendered sequence; `_has_supervision` guard present |
| `measure_baseline_fkds.py` | flattens `addon_config` into the top level (else `text_rag` raises `KeyError: embed_model`) |
| `pipeline/kv_inference.py` | honours `force_text_rag` and `rag_model` |
| `tools/gen_diversity_deepseek.py` | supports `--model` and `--style dialogue` |
| `eval/metrics.py`, `eval/stats.py` | `token_f1`, `llm_judge`, `paired_bootstrap`, `mcnemar_exact` |

### Config files
- `config_e4b_matched.json` — E4B, r64/α128, 1 epoch (training).
- `config_e4b_matched_eval.json` — same plus `vllm_url` removed, `force_text_rag: true`, `rag_model: "base"`.

### Human steps
- Review a sample of generated long/short pairs for format separation and groundedness (A2, A7) **before** committing to the training spend.

---

## 11. External validation: RAGBench (pre-registered)

### 11.1 Why an external dataset is required

UC1 and UC4 are the two corpora where parametric already won. Validating the hypothesis on them would be circular: a reviewer can reasonably object that the effect was measured where it was found. The hypothesis therefore needs an **external dataset that was not used to select it**, with both the domain panel and the corpus subsetting rule fixed **before any result is seen**.

### 11.2 The dataset

`galileo-ai/ragbench` (Friel et al., 2024). 12 domain configs, every one bundling `documents` per example, with train/validation/test splits and precomputed judge labels (`trulens_groundedness`, `ragas_faithfulness`, `gpt3_adherence`, `adherence_score`, `relevance_score`, `utilization_score`, `completeness_score`).

Three properties make it suitable for an objective test:

1. **One published protocol across domains** — a multi-domain claim can be made with one harness, one FA formula, one judge.
2. **Real questions**, not teacher-generated ones — avoids the UC1 failure mode where **71.6 %** of generated eval candidates had to be discarded for overlap with training.
3. **Precomputed groundedness labels** — part of the analysis is judge-free, reducing exposure to the T1 self-preference confound.

Measured response lengths by domain: `expertqa` 144 w, `cuad` 98 w, `emanual` 53 w, `delucionqa` 51 w, `finqa` 43 w, `covidqa` 28 w.

### 11.3 Pre-registered domain panel

Three domains, chosen for answer-length spread and domain distance **before seeing any KVForge result**:

| domain | why it is in the panel | measured corpus |
|---|---|---|
| `expertqa` | longest free-form responses in the panel | 5,921 docs / 16.1 M chars |
| `techqa` | technical documentation — closest external analogue to UC4, but a **different underlying dataset** | 2,751 docs / 10.4 M chars |
| `emanual` | procedural manuals — a third answer shape | ~181 docs sampled |

**Exclusions, by stated rule rather than by outcome:** `pubmedqa` (it *is* UC2), `hotpotqa` (evaluated in the paper), `cuad` (it *is* the paper's UC5), `covidqa` (biomedical — too close to UC2).

### 11.4 Pre-registered corpus subsetting rule

RAGBench corpora are **8–13× larger** than the regime in which parametric consolidation was demonstrated:

| corpus | unique docs | chars | est. chunks @600 | UC1 reference |
|---|---|---|---|---|
| ExpertQA | 5,921 | 16.1 M | ~26,800 | 2,000 chunks / 279,766 words |
| TechQA | 2,751 | 10.4 M | ~17,300 | |

USING THEM WHOLE WOULD PLACE THE EXPERIMENT **OUTSIDE THE ABSORBABLE REGIME**, BIASING TOWARD RAG AND POTENTIALLY MASKING THE FORMAT EFFECT ENTIRELY — PRODUCING A NULL RESULT THAT DOES **NOT** TEST THE HYPOTHESIS.

### 11.4 Pre-registered corpus subsetting rule

**Budget: 3,000 chunks per domain** (chosen over 2,000 because of the measured question counts below).

#### A correction to this section, made during implementation

The rule originally drafted here was *"document pool from the RAGBench **train** split only; questions from `validation` + `test`"*. **That combination is unanswerable by construction and has been replaced.** Measured while implementing:

- RAGBench does **not** ship a shared retrieval corpus. Each example carries its own per-example `documents` field (the context its original RAG system retrieved).
- The `validation`/`test` documents are almost **disjoint** from `train`: overlap was **0.9 %** for `expertqa` and **11–13 %** for `techqa`.
- Consequence: keeping the corpus to `train` retained **2 of 203** held-out questions. One earlier variant that sampled the corpus independently of the questions retained **0.0 %** — at both chunk level and whole-document level.

So the pool must be the **union of all splits**: a corpus that does not contain the evidence for the held-out questions cannot be answered by the RAG arm and the comparison would be unfair.

#### The rule (fixed before running)

1. **Document pool = union of `train` + `validation` + `test`, deduplicated** on `md5(normalised text)`. This is necessary, not discretionary (§11.4 correction).
2. **Chunk** each document to **600 chars with 60-char overlap**, matching UC1's indexing convention.
3. **Select whole documents** needed by the held-out questions, taking questions in `md5(question text)` order and documents in `md5(document text)` order, until **(1 − filler_reserve) × budget** is spent, with `filler_reserve = 0.25`.
4. **Top up** with filler documents in `md5(document text)` order until the budget is reached. Filler keeps the corpus from being literally only the evaluation evidence, which would otherwise maximise the parametric arm's advantage.
5. A question is retained **only if every chunk of every document it depends on is inside the subset**. A question whose evidence cannot be resolved into the pool is recorded as `unmapped` and is **never** reported as covered.
6. `validation` + `test` questions are the held-out candidates. **`train` questions are never used as supervised targets.** (The generator produces its own training questions from the subset chunks; those are deduped against the held-out set per §5.4.)
7. Write `subset_manifest.json`: budget, achieved chunk and document counts, `md5` of the ordered chunk list, `filler_reserve`, `unmapped_questions`, and per-split row counts.

Every ordering is content-derived, so the subset is reproducible from the corpus alone and **cannot be tuned by rerunning** (R2). No RNG seed exists to leak a degree of freedom; the manifest records `seed: content-hash`.

#### Measured achievable held-out question counts

**After deduplication** (see below — RAGBench carries duplicate question rows):

| domain | raw rows (val+test) | **distinct** | covered @3,000 |
|---|---|---|---|
| `expertqa` | 406 | 406 | **157** |
| `techqa` | 618 | 307 | **59** |
| `emanual` | 264 | 132 | **122** |

Two consequences to carry with the results:

- The original "~150–200 questions per domain" target is **not reachable**; the achieved `n` is **157 / 59 / 122**. Report the **achieved n per domain**, not the nominal target. `techqa` at n=59 is the weakest arm and its per-domain result should be read as the least-powered.
- `emanual`'s entire pool is only **440 chunks**, so it cannot fill any budget ≥ 440 and its numbers are identical at all three settings. It also has **10 unmappable** questions. `emanual` therefore tests a *much smaller* corpus than the other two and should be read as a small-corpus case, not a peer of `expertqa`/`techqa`.

#### Duplicate question rows (found during implementation)

RAGBench's own splits contain **duplicate question rows**: `emanual` validation has 124 rows but only **66 distinct** questions, and the duplicated rows carry **identical evidence**. Concatenating `validation` + `test` therefore double-counts: measured, `emanual` was 244 rows for 122 distinct questions and `techqa` 118 rows for 59 — every duplicate exactly 2×. The overlap is **within** a split, not between splits (`validation ∩ test` by question text is empty), so it is a property of the dataset, not of the concatenation.

Left in, this would (a) inflate the reported `n` by 2× and (b) **invalidate every paired test**, since the paired bootstrap and exact McNemar both assume independent items. RAGBench's published per-domain sizes should be read with this in mind.

**Rule adopted:** collapse questions to one item per normalised question text (first occurrence wins). This is content-derived and result-blind, so it adds no degree of freedom, and the removed count is recorded per domain in `subset_manifest.json` as `duplicate_question_rows_removed`.

#### Gold-answer construction for the external arm

Gold answers are **not** synthesised from the retrieved evidence. Measured on the first attempt, the held-out questions depend on a median of ~4k chars (`expertqa`) rising to a median ~20k and a maximum ~96k (`techqa`); with the evidence truncated to a generation budget the model legitimately could not find the fact and returned the sentinel for **102/157** `expertqa` and **104/118** `techqa` questions. That is a data fault, not a result.

The long gold therefore comes from the dataset's **own reference `response`** field — a real answer to the same question produced from the full document set — and the short gold is **condensed from that same long answer**, so both formats describe identical content at different lengths. The generator now (a) refuses to fall back to using the question text as its own context, and (b) aborts rather than writing a gold set in which a sentinel such as `UNKNOWN` exceeds 50% of answers. Both guards exist because the original silent fallback turned a resolvable data error into 244/244 plausible-looking `UNKNOWN` answers.

**Anti-cherry-picking commitments:**

- panel and subset rule timestamped before the run;
- **every domain attempted is reported**, not only the favourable one;
- identical harness, FA formula, judge and rubric across all domains;
- a domain that contradicts H1 is reported as a contradiction.

### 11.5 Format construction on the external corpus

Same manipulation as §5.3: for each retained question, generate **both** a short and a long gold answer with the teacher, using **identical question text**, and drop any question whose text differs between the two generations.

⚠️ RAGBench's `response` field is a **model generation, not a gold answer**, so it is **not** used as a reference. It is used only as an independent cross-check via the precomputed groundedness / adherence labels, which are available for every example and require no judge call.

### 11.6 Follow-up (deferred): corpus size as a second factor

If the external validation succeeds, the natural follow-up adds **corpus size** as a second factor: the same domains at ~2,000 chunks versus the full corpus (~17,300 for `techqa`, ~26,800 for `expertqa`).

This separates two claims that the present single-factor design deliberately keeps confounded:

- *format* moderates the parametric-vs-RAG gap (the present hypothesis), and
- *corpus size* determines whether parametric consolidation is viable at all.

**Pre-registered prediction for that follow-up:** the format interaction is largest at the absorbable size and shrinks toward zero as corpus size grows — i.e. **format and size interact**. It is deferred precisely because running both factors now would confound them, and because 17k–26k-chunk corpora are 8–13× the demonstrated absorbable regime, so a failure there would not be evidence about format.

---

## 12. Threats to validity

### T1 — Judge self-preference (generator == judge) — **highest risk**
`deepseek-code-h` generates the training data *and* judges the outputs. Because only the parametric arm trains on that text, the bias is **asymmetric and points in favour of the hypothesis under test.**
**Mitigations:** (a) rank claims by the judge-free metrics (token-F1, cosine); (b) quantify the confound — compute each cell's answers' max-Jaccard to the nearest training answer and report the gap between cells; on the UC1 pilot this was **0.738 parametric vs 0.236 RAG, with 18.5 % of parametric answers verbatim training answers**; (c) if a second model can be granted, re-judge the saved answers (no GPU needed) and report both.

### T2 — Surface metrics are length-biased
token-F1 and cosine penalise verbosity. **Mitigation:** the format-neutral judge is primary for the long format; H2 explicitly tests the divergence; report both components separately (§7.4).

### T3 — Corpus redundancy / templating
The UC1 corpus is template-with-slots ("Customer: question about cancelling order {{Order Number}}"), so "distinct" questions may share an underlying intent. **Mitigation:** report distinct counts at multiple thresholds (§5.5), and state the effective-sample caveat with every claim.

### T4 — Residual near-duplicate leakage
Post-filter max train↔eval Jaccard was 0.786 on the pilot — inside the 0.80 gate but marginal. **Mitigation:** disclose it; run the 0.70 variant as a sensitivity check.

### T5 — Single-seed training noise
A 0.0165 F1 gap between two SQuAD arms was **not** distinguishable from noise at one seed. **Mitigation:** seeds 43/44 replication; report cross-seed CI; do not claim an effect smaller than the seed spread.

### T6 — Two FA formulas in the repo
Comparing these numbers to retired SQuAD FA values would be invalid. **Mitigation:** one harness, one formula per experiment, formula stated for every number.

### T7 — Teacher quality regression
`code-h` produced weaker QA than `v4p` under the generic prompt. **Mitigation:** the dialogue style restores usable quality (verified); both formats share the teacher so the internal contrast is unaffected; cross-corpus comparability is conceded as broken.

### T8 — Truncation of long answers
The long arm is exactly where the old `max_length` sizing would have silently dropped answer tokens. **Mitigation:** §6.3 steps 16–18, verified in the training log before accepting the run.

### T9 — Subset selection could itself become the cherry-pick
Subsetting RAGBench to ~2,000 chunks introduces a new discretionary step, and a subset chosen after seeing results would be as circular as choosing UC1. **Mitigation:** the rule in §11.4 is content-hash-ordered, not random and not result-dependent, so there is no degree of freedom to exploit; the md5 manifest makes it verifiable; and the rule is timestamped in the pre-registration block before the run.

### T10 — Domain heterogeneity within the panel
`expertqa` / `techqa` / `emanual` differ in style, chunk length and answer length, so a single pooled estimate would hide disagreement between domains. **Mitigation:** report per-domain estimates alongside the pooled one, and treat **panel disagreement as a finding**, not as noise to be pooled away.

---

## 13. Risks and mitigations

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| Long/short formats collapse to similar lengths | medium | experiment void | check A2 before training; regenerate with stronger instructions if it fails |
| Questions differ across formats | medium | breaks pairing | assert question-set equality; drop mismatches from both |
| Adapter capacity insufficient | low | void | A1 floor check |
| Retrieval hit rate low | low-medium | confounds RAG | report hit rate (A5) |
| Judge API rate limits / cost | low | delay | batch, retry, and cache judge responses to disk so a re-judge is free |
| GPU contention | low | delay | run adapters sequentially |
| 200 questions underpowered for a 0.05 effect | medium | inconclusive | report CI honestly; optionally extend the held-out set rather than reusing near-duplicates |

---

## 14. Deliverables

### Internal arm (UC1)
1. `examples/usecase1_customer_support/matched_format/` — train/eval data + index maps + `leakage_audit.json`.
2. Two adapters: `lora_checkpoints/e4b_mf_short/`, `lora_checkpoints/e4b_mf_long/`, with training metadata.
3. `results/mf/{short,long}_{parametric,rag}/` — per-question records + summaries.

### External arm (RAGBench)
4. `datasets/ragbench/<domain>/subset_manifest.json` — ordered-chunk **md5**, chunk count, total chars, doc count, question retention rate. This is the proof that the subset is reproducible and was not result-selected.
5. Per-domain Qdrant collections built from the subset, plus the twin QA sets (`qa_pairs_short.json`, `qa_pairs_long.json`) and format-matched held-out sets.
6. Six adapters (`<domain>_mf_short`, `<domain>_mf_long` × 3 domains) with training metadata.
7. `results/mf_external/<domain>/{short,long}_{parametric,rag}/` — per-question records + summaries, **for every domain attempted**.
8. `results/mf_external/panel_summary.json` — per-domain and pooled `D`, CIs, McNemar p-values, and the groundedness/adherence cross-check. **Includes any domain whose result contradicts H1.**

### Shared
9. An analysis script that emits the primary `D` estimate, CIs, McNemar p-values, the T1 confound check, the T9 subset-manifest verification, and the full metric table.
10. A short results section stating: the interaction estimate for both the internal and external arms, whether H1/H2/H3 hold, whether the external panel contradicts the internal result, and the leakage/confound disclosures — written so it can drop into the paper.
11. This document, updated with pre-registration timestamps and any deviations.

---

## 15. Cost and timeline

| phase | estimate | notes |
|---|---|---|
| data generation (twin QA + rewrites, 1,600 chunks) | ~2–3 h | ~3,200 teacher calls at concurrency 12 |
| dedup + audit + held-out build | ~15 min | pure CPU |
| human review of a sample (A2/A7) | ~30 min | **gate before training** |
| train `par_short` | ~1.5–2 h | 1 epoch; short sequences |
| train `par_long` | ~2–3 h | longer sequences → more compute |
| 4 evaluation cells + judging | ~1.5–2 h | 200 questions × 4 arms + judge latency |
| analysis + write-up | ~1 h | |
| **internal total, one seed** | **~9–13 h** | ×~2.5 if seeds 43/44 replication is run |
| external: corpus build + subsetting (3 domains) | ~30 min | CPU only; 600-char chunking + md5 ordering |
| external: twin QA generation (3 domains × 1,600 supervised chunks) | ~3–4 h | ~9,600 teacher calls at concurrency 12 |
| external: train 2 adapters per domain × 3 domains | ~8–14 h | 6 adapters; the dominant external cost |
| external: 4 eval cells per domain × 3 domains | ~3–5 h | 12 cells + judging; groundedness labels need no judge call |
| **external total, one seed** | **~15–24 h** | |
| **grand total, one seed** | **~24–37 h** | internal + external |

---

## 16. Reproducibility

- Record `git` commit, config hashes, and the exact CLI for every run in the training metadata.
- Pin the teacher and judge model IDs; a retired teacher makes a run non-reproducible (as happened with `v4p`) — snapshot all generated data to disk so re-training never depends on a live API.
- Cache all judge responses so a re-score is offline and free.
- Log which artefacts were deleted if any are pruned.

---

## 17. Known constraints carried in

1. **`deepseekv4p` retired** — the generator used for every prior corpus is gone; `code-h` is the only granted model. Cross-corpus comparability with SQuAD/PubMedQA is therefore broken regardless of how this experiment is run.
2. **Generator == judge** (T1) — the dominant validity risk, mitigated by judge-free primaries and a quantified confound check.
3. **Two FA formulas** in the repo (T6).
4. **UC1's original eval set was the training set** (`faqs.json` ≡ `eval_heldout.json`, md5 `f7a284a4…`) — which is why any new UC1 run must rebuild a held-out set from scratch.
5. **The long-format arm is the first workload in this project where the `max_length` sizing bug would be catastrophic**, not marginal.
6. **Prior per-corpus verdicts are not comparable to each other**: three harnesses, two formulas, two model eras, n ∈ {7, 30, 50, 200, 500, 1000}, and at least one contaminated eval set.

---

## Appendix A — definitions

- **Parametric (closed-book):** the model answers from weights alone; no retrieval.
- **RAG:** retrieve top-`k` chunks, answer in-context.
- **Consolidation:** the paper's framing — knowledge moving from retrieval into weights as corpus coverage grows.
- **FA:** factual accuracy, formula declared per experiment (§7.4).
- **Verified-distinct n:** item count after exact + near-duplicate (J ≥ 0.80) dedup on question and answer, within the set and against training.

## Appendix B — pre-registration block

To be filled and timestamped **before** the training run:

```
# ---- internal arm ----
date_preregistered:      ____
commit:                  ____
corpus:                  UC1 customer-support (2,000 chunks)
train/eval chunks:       1,600 / 400 (disjoint)
held-out n (raw/distinct): ____ / ____

# ---- external arm (RAGBench) ----
dataset:                 galileo-ai/ragbench
domain panel:            expertqa, techqa, emanual
panel_exclusions_rule:   pubmedqa(=UC2), hotpotqa(paper), cuad(=UC5), covidqa(biomedical aware UC2)
subset_rule:             pool = union of all splits, dedup; 600-char/60-overlap chunks;
                         WHOLE documents needed by held-out questions in md5(question) order,
                         then filler documents in md5(document) order
subset_budget_chunks:    3000
subset_filler_reserve:   0.25
subset_seed:             content-hash (no RNG)
subset_manifest_md5:     expertqa f1b37d2d68d10db74be5440340ea8577 | techqa f87323c56409a35803ee6b89e35677cb | emanual 4e8d070691bc93a8b728c31ae8c3f6dd
achieved_heldout_n:      expertqa 157 | techqa 59 | emanual 122   (ACHIEVED, after dedup; report these)
questions_dedupe_rule:   collapse to one item per normalised question text (first wins).
                         RAGBench carries duplicate question rows WITHIN a split
                         (emanual validation: 124 rows, 66 distinct); leaving them in
                         would double-count and invalidate every paired test
eval_gold_source:        long gold = dataset reference `response`; short gold = condensed
                         from that SAME long answer. NOT synthesised from truncated
                         evidence (that produced 102/157 and 104/118 UNKNOWN sentinels)
questions_train_used_as_targets: NO
questions_heldout_source: RAGBench validation + test (deduped)
unmapped_questions:      record per domain; never report unmapped as covered
eval_gold_guards:        no fallback to question-as-context; abort if a sentinel
external_reporting:      ALL domains attempted reported, incl. contradictions

# ---- shared ----
primary estimand:        D = [FA(par,long)-FA(rag,long)] - [FA(par,short)-FA(rag,short)]
min effect of interest:  0.05 FA
alpha:                   0.05, paired bootstrap 10k, 95% CI
seeds:                   42 (+43,44 replication)
judge:                   deepseek-code-h, binary, format-neutral rubric
FA formula:              0.5*token_f1 + 0.5*judge
judge-free cross-check:  RAGBench groundedness / adherence labels
falsification:           CI contains 0, or |D| < 0.05, or both formats same sign/margin
reporting gate:          internal result alone is insufficient; external panel must not contradict
followup_prediction:     format x size interaction - format effect largest at ~2k, shrinks toward full corpus
```
