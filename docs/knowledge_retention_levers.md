# Closing the FT→RAG Retention Gap: Grounded Lever Research

**Date:** 2026-10-08
**Author:** Claude (Opus 4.8), under Dr. Hemant Joshi
**Question:** How can a fine-tuned SLM's knowledge retention be increased to get closer to the traditional-RAG retrieval ceiling?

---

## 0. The gap we are trying to close

From the volume sweep (`results/knowledge/uc4/volume_curve_e2b_claude.json`, Claude-judged, 481 held-out facts):

| | coverage@τ0.5 | mean accuracy |
|---|---|---|
| Base SLM (qa0) | 0.083 | 0.070 |
| Fine-tuned SLM (plateau, ≥16k QA) | **~0.30** | ~0.25 |
| Traditional RAG (recall@5) | **0.603** | — |

The fine-tuned SLM tops out at roughly **half** the RAG ceiling, and more training QA past ~20k does not help:

- Plateau slope (≥16k QA): **+0.005 coverage per +10k QA**, t = 1.25 — not distinguishable from zero.
- Point-to-point scatter on the plateau (sd 0.0207) equals the pure sampling noise of a 481-fact proportion (sd 0.0209): the wiggle *is* noise, there is no trend underneath it.
- Extrapolating even the (non-significant) slope to 120k QA predicts +0.02–0.03 — still ~0.33, nowhere near 0.603.

> **The ~0.30 plateau is genuine generalization, not leak-memorization (measured 2026-10-08).** 97/481 (20.2%) of held-out golds are substring-leaked into the 60k QA (`docs/knowledge_measurement_plan.md:348-382`). Splitting coverage by the per-idx leak flag shows the leak *does not* inflate the result — the opposite:
>
> | arm | ALL (481) | leaked (97) | non-leak (384) |
> |---|---|---|---|
> | base qa0 | 0.083 | 0.093 | 0.081 |
> | qafinal | 0.308 | 0.216 | **0.331** |
> | qa60000 | 0.331 | 0.227 | **0.357** |
>
> True-generalization coverage (non-leak) is **0.331**, slightly *higher* than the all-items 0.308, and the FT lift is larger on non-leaked facts (+0.250) than leaked (+0.123). The leaked items are mostly page-number/navigational trivia the closed-book model fails regardless. This resolves the plan's pending "worst case = entire jump is leak-memorization" threat (`knowledge_measurement_plan.md:368-382`) in favour of real generalization. The ~0.30 / ~half-of-RAG framing stands.

## 1. Why it plateaus — the mechanism

Two facts about the training setup, verified in the repo:

1. **Each fact is seen ~once.** The ~10 QA per chunk (6,154 chunks × ~9.8 QA ≈ 60,209) are **10 *distinct* facts, not paraphrases** — the generator prompt requires "each pair must test a DISTINCT fact" (`tools/gen_diversity_deepseek.py:88-95`). So a given fact gets ≈1 QA exposure.
2. **The repeated-exposure signal was excluded.** The ~10 diverse *rewrites* per chunk (the one component that gives multiple phrasings of the same passage content) are **not in the volume sweep** — it ran QA-only, `--replay-ratio 0`, no `--rewrites` (`docs/knowledge_measurement_plan.md:405-406`). The repo's own diversifier docstring (`pipeline/corpus_diversifier.py`) cites "10-40 diverse rewrites per chunk" as the effective regime; the sweep trained at ~1.

**What the ~0.25 lift actually is (measured, §0 blockquote):** the held-out eval facts were **deduped out of the 60k QA** and the sweep trained QA-only with **no rewrites**, so each eval fact's exact QA was **not in training**. Yet the model answers **33%** of the *never-leaked, deduped* facts (vs 8% base) — this is **genuine domain generalization**: training on 60k *distinct* Bedrock facts teaches enough of the domain (and sibling facts from the same chunks) to recover ~a third of unseen facts. The substring leak contributes *little* and if anything drags the average down (leaked items 0.22 < non-leak 0.33). So the lift is real learning that transfers, not memorized leak.

**The ceiling is therefore a *generalization* ceiling, not a coverage or capacity ceiling:** the model extracts ~0.33 of corpus facts into weights from QA-only training, and more *distinct* QA past ~16–20k does not push it higher. To get from 0.33 toward RAG's 0.60 the lever is to put **more of the fact content, in more diverse forms, into training** — i.e. the **rewrites** (which carry the actual passage text, including held-out facts) and paraphrased QA. This is the Allen-Zhu 3.1 / EntiGraph recipe: augmented passage text *plus* QA raises extractable coverage, whereas adding more distinct QA at ~1 view/fact has saturated.

*(Caveat: the "sibling facts per chunk accumulate over the run" intuition assumes shuffled single-epoch training; the EC2 scheduler was not inspected, so treat the step-wise reading as indicative — see caveat 5.)*

## 2. What the literature says (with numbers)

| Paper | Finding relevant here |
|---|---|
| **Allen-Zhu & Li, "Physics of LMs" 3.1** (knowledge storage vs extraction) | A model can store a fact yet score **0% on QA about it** unless the knowledge was **augmented** (paraphrase, sentence-shuffle, translation) during training. Augmentation changes *how* the fact is encoded (linearly, in the entity embedding), which is what makes it extractable. Directly describes our bottleneck. |
| **Allen-Zhu & Li, "Physics of LMs" 3.3** (capacity scaling) | ~2 bits of knowledge per parameter, but a fact must be seen **~100–1000× in varied forms** to be reliably stored; at ~100× it stores much less. We trained at ~1 exposure/fact. |
| **Ovadia et al. 2023** (FT vs RAG knowledge injection) | RAG consistently beats unsupervised FT; FT only works for new facts when exposed to **many paraphrases** of the same fact. Paraphrase augmentation gives a large lift over baseline FT. |
| **EntiGraph / Synthetic CPT (Stanford, 2024)** | 1.3M-token corpus → **455M synthetic tokens (~350×)** of entity-connecting rewrites → closed-book QA **39.5% → 56.2%**. Log-linear scaling, no clear plateau to 455M. **EntiGraph+RAG (62.6%) > base+RAG (60.4%)** — parametric knowledge **compounds with** retrieval; reaches >80% of RAG's standalone gain. |
| **RAFT (2024)** | Fine-tune *open-book* with distractor docs + cite-the-relevant-span; improves domain RAG on PubMed/HotpotQA/Gorilla. A different target: better *use* of retrieval, not pure parametric recall. |
| **Gekhman et al. 2024** (does FT on new knowledge cause hallucination?) | New facts are learned **slowly** via FT, and as they are learned they **linearly increase hallucination**. FT mostly teaches a model to *use* pretrained knowledge. Caution flag for aggressive volume/epoch pushes. |
| **"LoRA learns less and forgets less" (2024)** | LoRA matches full FT on small data but **underperforms when data exceeds its capacity**; higher rank (r≈256) narrows the gap; LoRA memorizes less. Matches our repo sweep (r=64 best, r=128 overfits). |

## 3. Recommended levers, ranked

### ★ LEAD — Diversity / repeated-exposure augmentation per fact
Bring **rewrites back into training** and additionally generate **multiple paraphrased QA of the *same* fact** (varied phrasing, entity-first ordering, sentence-shuffle), targeting the **10–40 views/fact** regime the repo and literature both point to — instead of ~1.

- **Why this one — now measured & significance-tested (§6):** on the **matched 48-token harness** vs the clean pure-QA arm (qafinal, replay 0), adding ~10 passage rewrites/chunk lifts coverage **0.308 → 0.378 (+0.071, paired McNemar p≈0.0004)** — robust, not the earlier 39-vs-48-token artifact. It is also the only lever that puts the held-out facts' *content* into training (§1). **Caveat on the mechanism (new):** a volume-matched A/B (from-base, 450 treated facts) shows distinct rewrites beat a *single repeated* rewrite by only **+0.013 (n.s., χ²=0.52)** — repetition recovers **~91%** of the over-base lift, within the judge's ~0.012 pass-to-pass noise floor. So the lever is **passage-view augmentation** (exposure), *not* specifically diversity: generating many *distinct* rewrites may be largely unnecessary. Rename accordingly; see `results/knowledge/uc4/rwscale_AB_diversity.json`.
- **Expected effect:** the strongest evidence (EntiGraph's +16.7 pts closed-book from pure augmentation) suggests this can move the plateau materially toward RAG, not by 0.01s. It is the highest expected-value, lowest-architectural-risk change.
- **Cost:** synthetic generation + one retrain. The rewrites pipeline (`tools/gen_diversity_deepseek.py`) already exists; the work is scaling views/fact and re-running one arm.
- **First experiment (cheap, maybe already trained):** before commissioning a new 20×-views generation job, check whether the existing **`phase3_E2B_r64_60kqa_rw` arm** (`docs/knowledge_measurement_plan.md:338`) or **Arm C** (`run_ablation_60k.sh:70-72`, `--rewrites`) adapter still exists on EC2. If so, generate closed-book predictions for it (one sequential arm, `-P 1`) and **re-judge under the same Claude rubric** — a direct, same-judge data point against qafinal's 0.308. Only if that is promising (or missing) build the scaled ~20-views/fact arm, then plot coverage vs views/fact (expect log-linear per EntiGraph).

### 2 — EntiGraph-style synthetic continued pretraining (the heavy version of the lead)
Extract entities per chunk, synthesize diverse *connecting* text across entities, and continue-pretrain on it (full-sequence LM, not just QA).
- **Why:** most powerful demonstrated recipe; its gains **compound with RAG**.
- **Cost/caveat:** their 350× ratio on our ~0.9M-token corpus ≈ a **~300M-token** generation job, and they used *full* CPT on an 8B model, not LoRA. Treat as the scale-up if the lead arm proves out.

### 3 — Expand LoRA to the MLP, not just raise rank
Current adapters are **attention-only (q/k/v), embeddings frozen** (`config_e4b_r64_ep3.json`). Factual associations are commonly argued to live largely in the FFN/MLP (Geva et al. 2021, "FFN layers are key-value memories"; Meng et al. 2022, ROME). Add `gate_proj/up_proj/down_proj` (and `o_proj`) as LoRA targets.
- **Why "where" not "how much":** the repo sweep found **r=128 overfits** and epochs plateau ~6 — so raising rank is spent. Widening *which modules* adapt is an untested, cheap capacity lever better matched to knowledge storage.
- **First experiment:** one arm, q/k/v/o + MLP targets at r=64, everything else fixed.

### 4 — RAFT (open-book fine-tuning) — if the real goal is best end-to-end domain answers
Train with retrieved context + distractors so the model learns to exploit retrieval.
- **Why separate:** this doesn't raise *parametric* retention; it makes the SLM a better RAG *reader*. If the product goal is "beat traditional RAG on answers," this is the pragmatic path and pairs naturally with lever 1.

### 5 — Modest repeated epochs on diverse views (not identical repeats)
Allen-Zhu's ~100–1000 exposures implies more passes help — **but** only with *diverse* exposures (lever 1), and bounded: repo epochs overfit by ~12 and Gekhman warns repeated new-knowledge FT raises hallucination. Use ≤~3–6 epochs *over the augmented set*, not more epochs over identical QA.

## 4. Honest caveats (must survive into the paper)

1. **The 0.603 RAG target is recall@5 — a retrieval hit-rate, not a judged end-to-end answer.** The true closed-book-vs-RAG gap on judged answers is **smaller** than 0.30 vs 0.603. Do not state the gap as if both numbers are the same metric.
2. **EntiGraph's headline cuts against "SLM replaces RAG."** Its best configuration is FT **+** RAG (62.6 > 60.4): parametric knowledge **complements** retrieval. This supports the already-planned reframing of the paper from "dominates" to **non-inferiority + cost + complementarity** (the FT≈½×RAG result is consistent with this, not with replacement).
3. **~20% substring leak** inflates current coverage (`docs/knowledge_measurement_plan.md:348-382`); true generalization is lower, so the 0.30 plateau is, if anything, optimistic.
4. **Judge-dependence:** absolute levels shift with the judge (base 0.083 Claude vs 0.154 deepseek). The *shape* and the *FT≈½×RAG ratio* are the robust claims; any augmentation gain should be measured against the same Claude judge to be comparable.
5. **The volume-sweep X-axis is checkpoints along one 1-epoch run**, not independent models — so it measures "progress through one LR schedule," which is a further reason it cannot answer "would more data help." The augmentation arm (lever 1) is what actually tests the real question.

## 5. One-line recommendation

The plateau is an **injection/coverage** ceiling, not a capacity or raw-volume ceiling: QA-only training barely puts the held-out facts into the model at all. Spend effort on **diverse exposures of the facts themselves (rewrites + paraphrased QA, ~20× per fact)**, re-judged on the same 481 held-out facts — not on more distinct QA, more rank, or more epochs over identical data. If that arm lifts the curve, scale it EntiGraph-style and report parametric knowledge as *complementary* to RAG.

## 6. Next runs (ordered)

1. **Leak split — ✅ DONE (2026-10-08).** Reproduced the 97/481 (20.2%) substring leak on EC2 (CPU-only, `CUDA_VISIBLE_DEVICES=""`, exact `overlap_check.frac_ans_in_blob` definition), emitted per-idx flags, idx-aligned perfectly to the Claude judge scores, and split qa0/qa60000/qafinal coverage. **Result (§0):** non-leak generalization coverage = **0.331**, *higher* than all-items 0.308; leak does not inflate the plateau. The ~0.30 / half-of-RAG framing is confirmed as real generalization.
2. **Re-judge the rewrites arm under Claude — ✅ DONE (2026-10-08), GPU-free.** The adapter exists on EC2 (`lora_checkpoints/phase3_E2B_r64_60kqa_rw`, base `google/gemma-4-E2B-it`, r64 α128). Its stored closed-book predictions (`results/eval500_e2b_60kqa_rw/parametric_records.json`) aligned perfectly to the same 481 prompts (481/481, 0 gold mismatches), so they were re-judged in-session under the identical Claude partial rubric — **no GPU, no new training**. Result:

   | arm | ALL (481) | non-leak (384) | leak (97) |
   |---|---|---|---|
   | QA-only qafinal | 0.308 | 0.331 | 0.216 |
   | QA-only qa60000 (best) | 0.331 | 0.357 | 0.227 |
   | **QA + rewrites** | **0.389** | **0.398** | 0.351 |

   **UPDATED 2026-10-09 (matched harness + controlled test):** on the same 48-token harness, adding ~10 rewrites/chunk lifts coverage **0.308 → 0.378 (+0.071, p≈0.0004)** vs the clean pure-QA arm (0.391 non-leak). The honest headline is **"passage views beat more QA"** — the ★ changes format, example-count (~2×) *and* distinctness at once, so it does not by itself prove diversity. A volume-matched A/B that isolates distinctness (from-base, 500 QA + 4,421 rewrites of the eval chunks; A = distinct, B = one rewrite repeated to the same count) gives **A 0.233 vs B 0.220 on 450 treated facts (Δ +0.013, n.s., χ²=0.52)**; B recovers ~91% of A's over-base lift, within the ~0.012 judge pass-to-pass noise floor. **Conclusion: it is passage *exposure*, not *distinctness*, that does ~90% of the work** (bounds any distinctness effect to ≲0.03 in this regime; can't exclude a small one). The A/B arms sit at ~0.23 (below N0), so this tests injection in isolation, not views on top of broad N10 knowledge. Persisted: `results/knowledge/uc4/rwscale_N0_N10_48tok.json`, `rwscale_AB_diversity.json`. (Earlier deepseek/E4B +0.056 and the 39-token +0.068 are superseded by these matched-harness numbers.)

**Both diagnostic runs are done and both point the same way: the ceiling is a generalization ceiling, and diverse exposures lift it. The next real experiment is scaling rewrites to ~20–40 views/fact (EntiGraph-style) to see how far past 0.40 it goes — that one needs GPU training.**

---
*Sources: Allen-Zhu & Li 2404.05405 (3.3), 2309.14316 (3.1); Ovadia et al. 2312.05934; EntiGraph 2409.07431; RAFT 2403.10131; Gekhman et al. 2405.05904; "LoRA Learns Less and Forgets Less". Repo evidence cited inline.*
