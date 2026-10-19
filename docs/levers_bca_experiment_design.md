# Levers B → C → A — pre-registered experiment design (2026-10-09)

Goal: close the parametric↔RAG gap (N10 QA+rewrites coverage@0.5 = **0.378** vs RAG recall@5 = **0.603**)
by finding which of three *unspent* levers actually lets the model represent more corpus knowledge in
weights. Levers chosen (ordered B, C, A per decision):
- **B — placement/capacity:** LoRA currently targets `q_proj,k_proj,v_proj` only (r64 α128). Factual
  knowledge lives in the MLP/FFN; literature says target *coverage* beats rank. Add MLP projections.
- **C — structured augmentation:** rewrites are single-chunk surface paraphrases; A/B showed their
  *distinctness* is ~null. Test EntiGraph-style *relational* (entity-linked, cross-chunk) rewrites.
- **A — exposure:** training is 1 epoch over ~10 views; Physics-3.3 says reliable storage needs ~1000
  exposures. Test more epochs.

This file is the **pre-registration**: metrics, controls, and the works/doesn't-work bands below are
fixed BEFORE running. Results are filled in after (§8). **No band is to be moved post hoc.**

---

## 0. Shared protocol (held constant across B, C, A — only the one variable named per experiment changes)

**Testbed (same as the A/B contrast).** From-base `google/gemma-4-E2B-it`, 4-bit. Training set =
500 QA (`faqs_sample500.json`) + rewrites of the **450 treated eval chunks**. From-base so each adapter
evals cleanly on bare base (no adapter-stacking mismatch). ~40 min train + ~12 min regen per arm,
single A10G, **one arm at a time** (3 resident 4-bit copies caused silent CPU offload before).

**Primary metric.** `coverage@0.5` = (#CORRECT + #PARTIAL)/450 on the 450 treated facts.
**Secondary.** `mean_fidelity`; leak vs non-leak split (per `eval500-heldout-and-leak`).

**Judge.** In-session Claude PARTIAL rubric (`eval/metrics.py`), temp 0, 3 lean subagents, NO
self-validation, golds inline in the preds (question/gold/prediction), idx 0–480.
- **Blind + co-judged (REQUIRED from C onward):** each treatment is judged in the SAME session as its
  control, with control and treatment predictions for the SAME fact **shuffled into the same subagent
  shard**, so each subagent judges both arms' version of every fact and instance-level leniency cancels
  in the pairing. (B1 was judged separately vs prior-session scores; the verdict was instead confirmed via
  same-session seed-43 control + strict-CORRECT-only checks — see §8. Do not repeat that shortcut.)
- **One judge model/session per comparison** — never mix judge models/sessions in one comparison; re-judge
  the control in-session rather than reusing older scores.

**Two noise sources — both must be characterised, not just judge noise:**
- *Judge pass-to-pass:* **0.012 coverage** (qafinal vs qa60224, same weights, 97% agree) — measured.
- *Training seed:* changing modules/data/order changes init + data order. **Step 0 (below) measures it.**

**Statistic.** Paired **McNemar** on correctness, each treatment vs its co-judged Arm A control, NOT vs
base and NOT vs N10. b = control-only-correct, c = treat-only-correct; χ² = (|b−c|−1)²/(b+c);
paired SE ≈ √(b+c)/450 ≈ **0.020** at the observed ~18% discordant rate.

### Step 0 (MANDATORY, run first) — measure the training-seed null  ✅ DONE 2026-10-13
Retrained Arm A with **seed 43**, co-judged vs seed-42. **Result: Δcoverage = +0.0166, McNemar χ²=0.79
(n.s.)** — seeds statistically indistinguishable, but the **total per-arm comparison noise is SE ≈ 0.0164**
(judge ~0.012 ⊕ seed ~0.0115), above the judge-only 0.012 floor. (`results/knowledge/uc4/rwscale_seednull.json`)

**Adopted rule — ADAPTIVE 2-SEED (user decision 2026-10-13):** run each arm single-seed first, then:
- Δ ≤ **+0.012** → **DEAD** (within noise), accept single-seed.
- Δ ≥ **+0.050** (≈3×SE, clear of noise) AND χ²≥3.84 → **WORKS**, accept single-seed.
- **+0.012 < Δ < +0.050** → **AMBIGUOUS** → run a **2nd training seed**, average the two, re-judge; the
  averaged Δ is then read against the DEAD(≤0.012)/WORKS(≥0.040) bands (SE halves to ~0.012 with 2 seeds).

### Pre-registered decision bands (every experiment, treatment vs its co-judged control)
| verdict | rule | action |
|---|---|---|
| **WORKS** | Δcoverage ≥ **+0.040** AND McNemar χ² ≥ 3.84 (p<0.05, \|b−c\|≳19) | promote (§6) |
| **DEAD** | Δcoverage ≤ **+0.012** (≤ judge noise floor) OR Δ negative | reject lever at this operating point |
| **AMBIGUOUS** | +0.012 < Δ < +0.040, or magnitude met but χ²<3.84 | do NOT promote; escalate = **add a 2nd training seed per arm** (481-fact + 2nd-judge-pass only shrink the wrong noise and dilute) |

The +0.040 magnitude gate is stricter than bare significance, so it also absorbs the 3-test family-wise
concern (Bonferroni α/3 ⇒ Δ≈0.043 — essentially the same bar). Report raw p regardless.

**Placebo channel (descriptive, not a hard gate).** The 31 eval facts NOT among the 450 treated chunks are
scored too but never promoted. n=31 ⇒ its paired SE ≈ 0.08, so it fires at random and a better adapter can
genuinely transfer to them — so it is reported for context only. **Void an arm only on a gross move:**
placebo Δ > treated Δ AND placebo Δ > 2×SE (~0.16). Otherwise record and move on.

**Regime caveat (stamped on every result).** From-base injection at ~0.23 (below N0 0.308) isolates the
*mechanism* in isolation; it is NOT a magnitude vs N10. Any WORKS verdict must be re-validated per §6.

---

## 1. Experiment B — LoRA placement/capacity (MLP targeting)

**Control:** Arm A (`rwscale_armA_diverse`, q/k/v, r64) — coverage **0.233**, co-judged fresh each comparison.

**⚠ Why attention-only today, and why MLP is feasible (investigated 2026-10-09).** Gemma-4-E2B is
**multimodal**: a vision tower + audio tower + language backbone. The projection leaf-names
(`q_proj`…`down_proj`) exist in BOTH the towers and the LM. The **tower** projections are
`Gemma4ClippableLinear`, which PEFT cannot inject (verified: naive all-name targeting →
`ValueError: Target module Gemma4ClippableLinear … is not supported`, reproduced in 4-bit and fp16).
The repo's Gemma-4 trainer `tools/train_gemma4.py` solves this with an explicit **unwrap step** (L36–45):
it replaces `Gemma4ClippableLinear` with its inner `nn.Linear` *in the vision/audio towers only*, then
sets a **hardcoded** `target_modules=["q_proj","k_proj","v_proj","o_proj"]`. So attention-only is a
hardcoded list inherited from the KV-cache phase (the paper uses q/k/v because K/V updates drive KV-tensor
staleness) — **not** a technical bar on MLP. (`pipeline/lora_trainer.py` on EC2 solves the same multimodal
problem a different way — a filter to all-`nn.Linear` leaf names + `per_layer_projection` fallback.)
- **Gate B0 (dry-run attach, no training, ~2 min):** run `train_gemma4.py`'s unwrap, add
  `gate_proj,up_proj,down_proj` to `target_modules`, call `print_trainable_parameters`, and assert LoRA
  attached to the LM MLP linears (and not to the towers). Low-risk — the unwrap already makes these standard
  `nn.Linear`. Do not read coverage until B0 passes.

**Primary treatment B1 (placement, param-matched):** data = Arm A exactly; targets = **all-linear**
(attn q/k/v/o + MLP gate/up/down); **r reduced so trainable params are within ±15% of Arm A** (compute both;
expect r≈12–16); **α = 2r** (NOT 128 — holding α=128 at low r inflates α/r≈8–10 and changes the effective
LR). Tests the literature claim "coverage > rank" at fixed capacity. (Arm is labelled **all-linear**, not
"MLP", because it also adds `o_proj`; a win is attributable to broader coverage, not MLP alone.)
**Conditional B2 (capacity), only if B1 is DEAD/AMBIGUOUS:** all-linear at **r64, α128** (more params).
Separates "placement/coverage doesn't help" from "we were param-starved."

**Training-health gate (after B0, before judging).** Final train loss ≤ Arm A final × 1.10, no NaN/diverge.
Fail ⇒ **implementation-blocked** (distinct from DEAD); fix, re-run.

| outcome | meaning |
|---|---|
| B0 fails | targeting blocked by Clippable wrapper → patch trainer to inner `.linear` first; not a science result |
| B1 WORKS | broader module coverage is the lever at fixed capacity. Promote; make all-linear the default. |
| B1 DEAD, B2 WORKS | capacity-bound: needed more params, not broader coverage. Promote all-linear-r64. |
| B1 DEAD, B2 DEAD | this base/corpus is not LoRA-placement/capacity-limited. Lever closed. |

Cost: Gate B0 (~2 min) + B1 (1 arm) + conditional B2 (1 arm).

---

## 2. Experiment C — structured (relational) augmentation

**BACKBONE PIVOT (2026-10-13):** since B1 (all-linear r8 α16) is a clear WORKS, C and A now run on the
**B1 backbone** (all-linear r8 α16) with **B1 as the control** — nobody will deploy the q/k/v config, and
C/A may interact with the MLP adapters. The one tested variable per experiment is unchanged; only the
fixed backbone changes from q/k/v-r64 to all-linear-r8. Control B1 is re-generated and **co-judged** in the
same session as the treatment (shuffled shards).

**Control:** B1 all-linear r8 (paraphrase rewrites of the 450 chunks) — coverage **0.3306**.
**Treatment C1:** **relational** rewrites of the SAME 450 chunks, same 500 QA, **same all-linear r8 config as B1**.
Only the rewrite *content* changes: group the 450 chunks by source document; generate text that states and
*links* facts across chunks within a group (multi-hop / entity-relational), instead of rephrasing one chunk
in isolation.

**Match EXPOSURE, not rewrite count.** A/B says exposure drives ~90% of the lift, and relational texts span
2–3 chunks and run longer — so matching total rewrite count (4,421) would confound structure with exposure.
Instead match, and audit before training:
1. **Per-fact mention count** ≈ Arm A's (~**10×** per fact, not "≥1×").
2. **Total training tokens** within ±10% of Arm A's rewrite tokens.

**Runnability pre-check (do FIRST — may kill the experiment as designed):** count how many of the 450 chunks
share a source document with another treated chunk. If most are **singletons**, there is nothing to link
across and C1 cannot be run as specified → either widen the treated set to include doc-neighbours or drop C.

**Data-validity gate (before training):**
1. **Fact preservation:** majority-vote number validator ≥ 98% accept (`validate_rewrites.py` rule).
2. **No fabricated relations:** the number validator cannot catch invented cross-fact claims → **spot-audit**
   a random 30 relational texts by hand for fabricated links; >1 fabrication ⇒ fix the generation prompt.
3. **Closed corpus:** links stay WITHIN the 450 (within source-doc groups ⊆ 450); no out-of-set facts.
Fail any ⇒ fix generation; do not train.

| outcome | meaning |
|---|---|
| C1 WORKS | structured augmentation is a real lever on single-fact recall — relational > paraphrase at matched exposure. The knowledge-representation result. Promote. |
| C1 DEAD | on 600-char single-fact chunks (few entities each), relational structure ≈ paraphrase. Consistent with "our eval has too little relational structure to exploit"; may differ on entity-rich corpora (EntiGraph's regime). Lever closed for THIS corpus. |
| C1 AMBIGUOUS | underpowered; add a 2nd training seed before believing either way. |

Cost: relational generation (budget-gated; pilot 100 chunks, confirm ~16 rw/min + spend headroom) + 1 arm.

### C PIVOT — pre-registered 2026-10-14 (advisor-reviewed, BEFORE any generation)
Reconstruction of B1's exact rewrite→chunk mapping failed: the targeted rewrites (`rewrites_armA_diverse.json`
= `rewrites_targeted_new.json`, a contiguous 4,421-string tail block of `rewrites_targeted_N20.json`) carry
**no chunk key**, no builder survives in `tools/`, and B1's rewrites were generated by an **earlier** generator
run (possibly the retired `deepseekv4p`/Haiku), whereas C must use today's `deepseek-code-h`. Comparing C1
(new generator) directly to the historical B1 would therefore confound *structure* with *generator version*
and *chunk set*. **Decision: make C fully self-contained** — build BOTH arms fresh with the same generator,
QA, chunk set and exposure, changing only rewrite *structure*. B1-s42 is co-judged alongside as a continuity
anchor (descriptive), but the **inferential test is C1 vs C0**.

- **Chunk set.** All **481 eval-fact chunk_ids** (authoritative, recovered from the prediction JSONLs; §0's
  "450 treated + 31 placebo" split is not recoverable, so BOTH arms treat all 481 → no placebo channel for C;
  it was descriptive-only anyway). Primary metric → `coverage@0.5` over 481.
- **QA (held constant).** `faqs_sample500.json` (the exact B1 QA), reused verbatim in both arms.
- **Config (held constant).** all-linear r8 α16, 1 epoch, lr 2e-4 linear decay (= B1 config `config_e2b_allLinear_r8.json`).
- **C0 (paraphrase control).** B1's exact `_rewrites_prompt` from `gen_diversity_deepseek.py`, applied per-chunk,
  ~10 rewrites/chunk. The ONLY thing C1 changes relative to C0 is the content/structure of the rewrites.
- **C1 (relational treatment).** Same scaffold + the group's neighbouring chunk texts + an instruction to
  STATE and LINK facts across the group's chunks (multi-hop), preserving every fact exactly, **inventing no
  relation not present in the provided texts**.
- **Grouping rule.** Adjacency windows on sorted chunk_id with gap **G=10** (median NN gap=8; spot-checked that
  neighbours within 10 are in the same guide section and share entities → genuine links, not fabrication).
  At G=10: 185 groups — 74 singletons, 111 groups≥2 (41 pairs / 25 triples / 16 quads / …); 85% of chunks in
  groups≥2. **Singletons (74): both arms get identical C0-style paraphrases** (nothing to link) → they dilute
  the effect toward null by construction, hence the secondary analysis below.
- **TOC/positional handling.** Only **4/481** facts are positional ("what topic comes after X"); they stay in
  the primary all-481 analysis but are **excluded from the secondary**, and the C1 prompt forbids inventing
  ordering claims.
- **Exposure match (audit before training).** "Mention of fact X" := X's chunk was in the generation context
  AND the number/term validator finds X's key numbers/terms in the text. Match C1 to C0 on the **per-fact
  mention distribution (~10×)** and **total training tokens within ±10%**.
- **Validity gates (before training, both arms).** (1) number validator ≥98% accept; (2) hand-audit 30 random
  C1 texts for fabricated cross-fact links, >1 ⇒ fix prompt; (3) closed corpus — links stay within the 481.
- **Judging.** Co-judge **C0, C1, B1-s42** blind in ONE session (~1,443 items, 4 shards); inferential =
  **C1 vs C0 per-fact Wilcoxon** (unit=fact), plus McNemar for continuity. **Secondary:** same test on
  grouped content facts only (drop 74 singletons + 4 positional).
- **Bands (§0).** WORKS Δ≥+0.040 & sig; DEAD Δ≤+0.012; else AMBIGUOUS → paired 2nd seed = **TWO** new runs
  (C0-s43 AND C1-s43), co-judged, per-fact Wilcoxon on two-seed-averaged scores.
- **C1 exposure-parity fix — pre-registered 2026-10-14 (advisor-reviewed, after v1 C1 failed the gate).**
  First C1 pass (10 compact relational rewrites/group) FAILED parity: C1 = 0.69M chars vs C0 2.36M (0.29×),
  and measured per-fact coverage (distinctive tokens present in the group's rewrites) was only **0.40** on
  multi-groups, collapsing with group size (size-2 0.66 → size-8 0.21; **7 facts at zero coverage**) — facts
  in big groups got ~2 effective mentions, not 10. Trainer truncates replay "raw chunks" at **512 tokens**
  (`lora_trainer.py` dynamic max_len), so long fact-complete rewrites would lose tail facts. **Fix:**
  (1) split every multi-group into **consecutive non-overlapping windows of ≤3 chunks** (≈≤370 tokens
  fact-complete, safe under 512); (2) regenerate each window as 10 rewrites that **fully restate every fact
  in the window AND link them** (target ≈490×k chars); (3) validator asserts each chunk's distinctive tokens
  appear in **all 10** of its window's rewrites. This matches C0 on per-fact mentions (10), per-fact tokens
  (~4,900), and total tokens (projected C1 ≈2.26M, 0.96× C0, within ±10%). Singletons (74) are KEPT as-is
  (coverage 0.91, and parity holds without them). **Unavoidable mismatch (pre-registered):** multi-fact
  examples cannot match mentions, tokens AND example count simultaneously — C1 will have ~2–2.5× fewer
  training examples than C0, hence fewer optimizer steps at 1 epoch under linear decay. We match mentions +
  tokens (the §2 quantities) and record the step difference; if C1 reads DEAD we re-check it isn't a step
  artifact (e.g. step-matched re-run) before concluding.
- **Spend gate.** Pilot BOTH prompts on the same 100 chunks (whole groups); log usage tokens incl. hidden
  reasoning; extrapolate to ~9k rewrites (≈2× the original single-arm budget) and check vs the spend limit
  BEFORE the full run.
- **Generator = THIS Claude Code session (Opus 4.8) via subagents (user decision 2026-10-14).** The CodeVector
  grant allows ONLY `deepseek-code-h` (a reasoning model; all cheaper models 403), whose full-run cost was
  projected at ~5.5–8.5M completion tokens. User directed: generate C with the in-session model instead →
  **zero external API spend**. Both arms use the SAME generator, so the C1-vs-C0 contrast stays clean (B1 is
  only a descriptive anchor, so the generator differing from B1's is irrelevant to the inferential test).
  Mechanism: shard the 481 chunks / 185 groups; one Opus subagent per shard writes JSON rewrites; aggregate.
- **Pilot validated (2026-10-14).** 15-chunk C0 + 5-group C1 pilot → after fixing a validator regex bug
  (trailing sentence-periods captured as numbers), **both arms no-fabrication = 1.00** (gate ≥0.98),
  distinctness 1.00; hand audit clean (genuine cross-passage links; traps avoided — "12 hours" not tied to the
  deny policy, `2012-10-17`/`2024-01-01` exact, no invented region availability).
- **Unified edge-case policy (BOTH arms, from pilot).** (1) **DROP** page numbers, section numbers and running
  footers — document-navigation artifacts, and **0/481 eval golds are page-based**, so dropping costs no fact
  and cuts TOC fabrication risk. (2) **DROP** truncated sentence fragments at chunk edges (do not complete
  them). (3) Preserve every other fact/number/ID/name verbatim. (4) C1: link only genuinely-related facts;
  invent no relation/ordering. Both arms get the identical policy so only *structure* differs.

---

## 3. Experiment A — exposure (epochs)

**On the B1 backbone (see §2 pivot).**
**Control:** B1 all-linear r8 at **1 epoch** — coverage **0.3306**.
**Treatment:** same B1 data/config, train to **3 epochs**, checkpoint at epoch 1 / 2 / 3.

**LR-schedule confound (must resolve before running).** The trainer sets `learning_rate` but default
`lr_scheduler_type` is linear decay — so under a 3-epoch run the epoch-1/2 checkpoints sit at a *higher* LR
than a finished 1-/2-epoch run and are not comparable to the control. Resolve one of two ways, pre-registered:
- **(preferred)** set `lr_scheduler_type="constant"` after warmup for BOTH the control and the 3-epoch run,
  making all checkpoints comparable; then every epoch is inferential; OR
- keep decay and pre-register that **only epoch-3 vs Arm A is inferential**; epoch-1/2 are descriptive only.
Either way, **drop the "rise-then-fall" verdict** (it depends on non-comparable interior checkpoints under decay).

| outcome (single seed) | meaning |
|---|---|
| epoch-3 Δ ≥ +0.040 (χ²≥3.84) | exposure is the lever our A/B predicted — cheapest win. Promote peak epoch. |
| epoch-3 Δ ≤ +0.012 | already saturated at 1 epoch for this data scale; DEAD. |
| in between | AMBIGUOUS; 2nd seed (paired — see below). |

**RESOLVED: ran decay path.** `save_strategy="no"` ⇒ no interior checkpoints and default linear decay ⇒
epoch-1/2 snapshots not comparable, so A is a **complete 3-epoch job (own decay-to-0)** judged vs the finished
B1 1-epoch run. Only **epoch-3 vs B1 is inferential** (pre-registered option 2); "rise-then-fall" dropped.

**Paired 2nd seed (2026-10-14, advisor-corrected).** A and B1 differ ONLY in epochs (same modules/rank). So
the 2nd seed must be **paired**: Δ_s43 = A-ep3-seed43 **vs B1-seed43** (both trained at seed 43, co-judged
blind together), NOT vs the seed-42 B1 control. Comparing A-s43 to B1-s42 would mix seeds and leave all of
B1's seed noise (~0.0115) in Δ; pairing cancels the shared LoRA-init + data-order component. Averaged
Δ = mean(Δ_s42, Δ_s43). Requires training B1-seed43 (1-epoch, ~52 min) — it did not exist (the earlier
seed-43 run was Arm A attention-only, not all-linear B1).

**Pre-registered 2-seed averaged read (decided before seed-43 data, 2026-10-14):**
| averaged Δ | verdict |
|---|---|
| ≥ **+0.040** AND pooled McNemar χ² ≥ 3.84 | **WORKS** — exposure is a real lever; promote peak epoch (§6). |
| ≤ **+0.012** | **DEAD** — saturated at 1 epoch. |
| **+0.012 < Δ < +0.040, or χ² n.s.** | **WEAK / INCONCLUSIVE** — NOT promoted, **no 3rd seed**, lever closed at this operating point; proceed to C. (The 481-fact + extra-judge-pass only shrink the wrong noise; a 3rd seed is not worth the compute for a sub-0.04 effect.) |

Cost: 2 training runs (A-ep3 ×2 seeds) + 1 B1-seed43 control + 2 co-judge passes.

---

## 4. Sequencing
Step 0 (seed null) → B0 gate → B1 → (B2 if needed) → C runnability+validity → C1 → A. One arm at a time.
Each arm: train → gates → regen preds → co-judge (shuffled, blind) vs fresh Arm A → coverage + leak split +
placebo (descriptive) → McNemar → record verdict against §0 bands in §8.

## 5. (merged into §6)

## 6. Promotion rule (what a WORKS buys)
A WORKS at from-base is NOT yet a result. Promote to ONE confirmation: **re-run the full N10 recipe FROM BASE
with the lever applied** (~17h) — NOT continued-training-on-top-of-N10 (that reintroduces the adapter-stacking
eval mismatch we already found). Re-judge on the full 481. Only a lever still clearing +0.04 at the N10
operating point enters the memo/paper. Levers may be **stacked** (e.g. all-linear + 3 epochs) only after each
clears alone.

---

## 7. Roadmap — Phase 2: FT+RAG complementarity (RAFT-style)

The B/C/A levers push *pure-parametric* recall, bounded well below the RAG ceiling on exact-fact recall —
a structural limit, not a tuning gap. The reliable way *past* it is to combine FT and RAG rather than treat
them as alternatives. Gated on Phase-1 completing (we want the best parametric recipe in hand first), then
run as its own pre-registered study.

**Question.** Does a fine-tuned generator *with retrieval at inference* beat plain RAG and beat the best
pure-parametric arm — i.e. genuine complementarity (parametric priors + retrieved passage)?

**Method.** RAFT — Retrieval-Augmented Fine-Tuning ([Zhang et al. 2024, arXiv:2403.10131](https://arxiv.org/abs/2403.10131),
UC Berkeley). Train on (question, context, answer) where context = oracle chunk **+ distractor chunks**, with
a fraction of examples having the oracle *withheld* (forces parametric fallback), and the answer **cites**
the oracle verbatim.

**Arms — ALL eval with top-5 retrieval at inference, SAME retriever/k; only the generator differs.**
⚠ **0.603 is RAG *recall@5* (retrieval metric, no `mean_fidelity`), NOT a judged answer score** — do not
compare against it. R0 must be a real judged arm.
| arm | generator | retrieval at inference | isolates |
|---|---|---|---|
| R0 (control) | base `gemma-4-E2B-it` | top-5 | **judged** plain-RAG (base + top-5, same 48-tok harness + same judge) |
| R1 | best Phase-1 parametric adapter | top-5 | does FT recall *add* on top of retrieval? |
| R2 | RAFT-tuned adapter | top-5 | the complementarity recipe |
| R3 (ablation) | RAFT-tuned adapter | none (closed-book) | how much RAFT also lifts pure-parametric |

**Discriminating test (pre-register fully when Phase 2 opens).** The informative contrast is **R2 vs R0**
(and R1 vs R0), **split by retrieval hit vs miss**. Complementarity predicts the gain concentrates on the
**misses** (where parametric memory covers what retrieval failed to fetch). "R2 ≥ best-parametric + 0.04" is
met by construction and is NOT a test — drop it.
- **Complementarity confirmed:** R2 ≥ R0 + 0.040 (χ²≥3.84), gain concentrated on retrieval-misses.
- **RAG-dominated:** R2 ≈ R0 (±0.012) — FT adds nothing once the passage is in context.
- **Interference:** R2 < R0 − 0.012 — FT hurts grounded answering (priors override the passage); a real,
  publishable negative.

**Why it matters.** Phase 1 = "how far can weights alone go" (SLM-replaces-RAG, bounded ~⅔ ceiling).
Phase 2 = "best system when you have both" — the product question. R3 vs R1 also says whether the RAFT
objective injects parametric knowledge *better* than QA+rewrites, feeding back into Phase 1.

**Cost note.** Larger than a Phase-1 arm: needs a retrieval-formatted training set (oracle+distractor
contexts from the live Qdrant index — see `uc4-two-corpora`) and inference-time retrieval in the eval
harness. Scope as its own build once Phase 1 picks a winner.

---

## 8. Results (filled in after running — DO NOT edit bands above)

| exp | arm | coverage@0.5 | Δ vs control | χ² | b/c | placebo Δ(31) | verdict |
|---|---|---|---|---|---|---|---|
| 0 | Arm A seed 43 (null) | 0.2516 (vs seed42 0.2349) | **+0.0166** | 0.79 n.s. | 27/35 | — | **noise SE≈0.0164 > 0.012 floor → marginal effects need 2 seeds** (see below) |
| B0 | all-linear attach (4-bit, via model_loader) | — | — | — | — | — | **PASS** (MLP attaches, config-only; `_unwrap_gemma4` unwraps all LM Clippables, deletes towers) |
| B1 (prelim, non-blind) | all-linear r8 α16 vs Arm A, cross-session judge | 0.3306 | +0.0956 | 20.25 | 27/73 | — | ⚠ **inflated** — corrected by two-seed blind below |
| **B (2-seed BLIND)** | all-linear r8 α16 vs Arm A attention, **blind co-judge** both seeds | **0.316** (ArmA 0.250) | **+0.0665** | Wilcoxon z=3.92 **p=0.0001**; sign p=0.0003 | 91/47 (343 tied) | — | ✅ **WORKS (robust)** — s42 +0.081, s43 +0.052, avg +0.067 |
| B2 | all-linear r64 α128 (7.74× params) | *(not needed — B WORKS)* | | | | | |

**B0 param table (2026-10-09):** Arm A q/k/v r64 = 12.48M trainable; all-linear r64 = 96.6M (×7.74),
r16 = 24.2M (×1.94), r12 = 18.1M (×1.45); ≈1.51M/rank ⇒ **r8 α16 = 12.1M (×0.97)** chosen for B1.

**B blind-correction (advisor, 2026-10-14):** the prelim +0.0956 compared B1 and Arm A in *separate* judge
sessions; B1's absolute coverage is judge-session-sensitive (0.29–0.33), so that gap was partly session drift.
Re-run as blind co-judge (B1 + Arm A predictions for each fact shuffled into one shard) at **both** seeds:
Δ = +0.081 (s42), +0.052 (s43), **avg +0.0665**, per-fact Wilcoxon p=0.0001. Still a decisive WORKS and the
dominant lever (~1.7× A), but the honest magnitude is **+0.067, not +0.096**. "Bigger than the rewrites lift"
is dropped (that +0.071 is the 60k-QA regime — not comparable to this from-base regime).
| **C (clean re-run, C1 vs C0, blind co-judge)** | relational (C1) vs paraphrase (C0), cap512 + step-matched + singleton-fixed | C1clean 0.231 (C0 0.256)† | **−0.025 (n.s.)** | McNemar χ²=1.64 p=0.20 | 43/31 | — | ✅ **DEAD — relational adds ~0 (diff-in-diff +0.002). VALID clean run.** |

**C result (2026-10-14) — PROVISIONAL, do not cite the −0.071.** Self-contained C0/C1 built fresh (same
in-session Opus generator, same 500 QA `faqs_sample500`, same all-linear r8 config; rewrite-corpus exposure
matched: C1/C0 chars = 1.03×, per-fact mentions 10, per-fact source coverage 0.97). Blind 3-arm co-judge
(C0/C1/B1-s42, 481 × 3 = 1,443 items, 4 shards, uniform in-session Opus judge, each fact's 3 arms co-located).
Raw **coverage: C0 0.3243, C1 0.2536, B1 0.3306**.
- **Three confounds ALL disadvantage C1 for non-structural reasons (advisor-caught 2026-10-14):**
  1. **Replay truncation at 256 tokens** (`pipeline/lora_trainer.py` chat-SFT `_chunk_max_len=256`). C0 rewrites
     mean 107 tok (0.04% over 256, 0% dropped); **C1 rewrites mean 217 tok — 44.4% exceed 256, 14.3% of C1
     tokens dropped** (the fact-complete tails never reached the model). C0 untouched. B1-style rewrites are
     also <256 tok → **the 17h promotion run is NOT affected.**
  2. **Fewer optimizer steps:** C1 ~183 steps vs C0 332 (C1 has fewer, denser examples at 1 epoch).
  3. **Singleton paraphrase mismatch:** C1's 74 singletons are v1 363-char paraphrases vs C0's 491-char.
- **Δ by window size (per-fact):** single/tail −0.109 · 2-chunk −0.090 · 3-chunk −0.049. The deficit *shrinks*
  as windows grow — the OPPOSITE of truncation/multi-fact interference concentrating in big windows, and
  consistent with the confounds (worst where C1's paraphrases are weakest/shortest). Predictions are short
  (C1 mean 76 ch, 8/481 near any cap) → no decode-blending. B1 anchor 0.3306 is *consistent* with (not proof
  of) prior; the C1-vs-C0 verdict stands on the single-session pairing regardless.
- **Clean re-run (pre-registered, before any DEAD/WORKS call):** raise `_chunk_max_len` to ≥512 (C0 unchanged,
  lets C1's full fact-complete text train), **step-match C1 to C0's 332 steps** (biases toward C1), re-dump +
  re-co-judge. Secondary: multi-chunk facts only (drop singletons + 4 positional). Only then read vs §0 bands.

**C CLEAN RE-RUN RESULT (2026-10-16, `results/knowledge/uc4/rwscale_Cclean_verdict.json`) — DEAD.** All three
original confounds fixed: (1) cap raised 256→**512** (C1clean max 460 tok → 0% truncated; C0 max 258); (2) step
matched via **grad_accum=9 (config-driven, single pass)** → 325 steps vs C0's 332 — *not* extra epochs, so no
lever-A repetition confound; (3) the 74 G=10 **singletons swapped to C0's paraphrases** (a singleton has nothing
to relate), exposure parity C1/C0 chars = 1.068×. C0 reused unchanged; only C1 retrained. Blind co-judge, uniform
in-session Opus, grouped-by-fact, singleton/multi tagged.
- **Primary (full-481): C1clean 0.231 vs C0 0.256, Δ=−0.025, McNemar χ²=1.64 p=0.20 (n.s.)** → ≤+0.012 floor → **DEAD** (single seed sufficient per rule; 95% upper bound ≈+0.02, well below the +0.050 WORKS line).
- **Diff-in-diff isolates structure:** multi-chunk Δ −0.025 − singleton Δ −0.027 = **+0.002 ≈ zero**. Relational
  cross-chunk linking adds ~0 retention; the small arm gap is a *global offset* (grad_accum 9/16, replay 2420/4810,
  higher QA share, new trainer build), NOT structure.
- **Singleton null** (identical text both arms): Δ=−0.027, χ²=0.125, p=0.72 — **ruled out gross breakage** (it
  caught an earlier base-model misload at −0.189); with n=74/8 discordant it cannot *confirm* Δ≈0, but its estimate
  matches the primary.
- **† rubric caveat:** this judge session scores stricter (same C0 pred file 0.324 first session → 0.256 now,
  >~0.03 drift) — do NOT compare 0.231/0.256 to the B/A §8 absolutes. The first run's −0.071 (confounded, older
  rubric) **does not replicate** under the clean design.
- **Trainer edits (EC2-only, `.bak` kept; never scp local→EC2):** `pipeline/lora_trainer.py` line 483
  `_chunk_max_len = 512`; line 500 (FIRST `TrainingArguments` block = QA chat-SFT path only)
  `gradient_accumulation_steps=effective_cfg.get("grad_accum_steps", 16)`. The second block (distillation, ~799)
  is untouched. **The local repo `lora_trainer.py` is a stale 256-line lineage lacking the chat-SFT path** — not
  patched (can't mirror an edit to code that isn't there); EC2 is authoritative. One-time sync EC2→local is owed.

- **Lever scoreboard (from-base):** **B placement +0.067 WORKS (dominant; promoted at full scale +0.123 / +0.106
  clean-facts)** · A epochs +0.040 marginal WORKS · **C relational DEAD (~0, diff-in-diff +0.002).** Placement is
  the lever that matters. **Phase 1 closed.**

**PROMOTION RUN — B at full 60k-from-base scale (2026-10-16, `results/knowledge/uc4/promo_allLinear_verdict.json`).**
Trained `promo_E2B_allLinear_60kqa_rw` (all-linear r8 α16, 7,611 steps, train_loss 1.601) and blind co-judged vs
the incumbent `phase3_E2B_r64_60kqa_rw` (attention-only r64 α128) on the 481 facts. **This is a clean paired
control:** the two adapters' `kvforge_training_metadata.json` differ in *only* rank(8/64)+placement(all-linear/
q-k-v) — **matched trainable params** (~12.1M vs 12.48M) — and are identical on seed(42), data (faqs_full 60,209
+ rewrites 61,554), `from_base`, epochs(1), lr(2e-4), dropout(0.05), replay(0.2), base model. Both adapters loaded
symmetrically (Unwrapped 0 ClippableLinear, 6.71 GB, no missing/unexpected keys); baseline 0.339 matches its ~0.33
history → not format-handicapped.
- **Result: promo 0.4615 vs baseline 0.3389, Δ = +0.1227**, per-fact McNemar χ²=29.25 p<0.0001 (87 promo-only vs
  28 baseline-only), sign z=5.41. **Verdict: PROMOTE — WORKS at full scale**, clears +0.040 by ~3×.
- **Leak split (`promo_allLinear_leak_split.json`):** 75/481 facts (15.6%) are residual-substring-leaked vs 60k QA.
  Gain is NOT a memorization artifact — **clean (non-leaked) facts n=406: Δ = +0.106** (promo 0.456 / base 0.350);
  leaked n=75: Δ=+0.213. Honest "corpus-retention" number = **+0.106 on genuinely unseen facts.**
- **Caveats (do not over-read):** single seed (Δ ≈ 7 seed-SE, clears the adaptive rule alone). Δ=+0.123 is *larger*
  than the pilot B +0.067, but different data regime, different control adapter, and a different (stricter
  wrong-entity) judge rubric in a different session → report **descriptively** ("larger at full scale, consistent
  with MLP capacity mattering more as fact-load grows"), **not** as "placement scales with data" (untested).
  Absolute 0.4615 is under the stricter rubric — do not compare raw to older §8 absolutes without that caveat.
| A (s42) | all-linear r8 @ **ep3** vs B1 ep1 **co-judged** | 0.3493 (B1 0.3243) | **+0.0249** | 1.34 n.s. | 39/51 | — | AMBIGUOUS (single seed) |
| A (s43) | all-linear r8 @ ep3 vs **B1-seed43** (paired, co-judged) | 0.3514 (B1 0.2952) | **+0.0561** | 5.98 * | 43/70 | — | WORKS (single seed) |
| **A (avg)** | mean(Δ_s42, Δ_s43); **per-fact paired test** (fact=unit) | — | **+0.0405** | Wilcoxon z=2.36 **p=0.019**; sign p=0.049 | 87/62 (332 tied) | — | ✅ **WORKS (marginal)** — avg Δ≥0.040 AND per-fact p<0.05 |

**Stats correction (advisor, 2026-10-14):** the pooled McNemar (χ²=7.11) **double-counted** each fact (summed
481×2 discordant cells as if independent). The correct unit is the fact: two-seed-averaged covered score per
fact, paired across arms → **Wilcoxon signed-rank z=2.36, p=0.019** (sign test p=0.049, borderline). Verdict
unchanged (WORKS) but the honest significance is far closer to the line than the pooled χ² implied.

**A two-seed verdict (`results/knowledge/uc4/rwscale_A_twoseed_verdict.json`, 2026-10-14):** clears the
pre-registered WORKS band, but **marginally** (avg +0.0405, line is +0.040) and **asymmetrically across seeds**
(+0.025 n.s. / +0.056 sig). Key texture: **A-ep3 coverage is stable (0.3493 / 0.3514, spread 0.002); the B1
control swung (0.3243 / 0.2952, spread 0.029)** — so Δ_s43's size is driven mostly by a *low seed-43 control*,
not a strong treatment. Honest magnitude ≈ **+0.04**, roughly **½ of the placement lever (B1, +0.096)** and
right at the promotion threshold. 3-epoch train_loss (1.605/1.616) ≪ 1-epoch (2.587/2.583): the extra epochs
mostly deepen train-set fit; held-out recall moves only ~+0.04. Reading per pre-registered rule: **WORKS**, so
exposure is admissible as a (small) stackable lever, but B1 remains the dominant effect.
