# Experiment Plan: Scaling Rewrites — Coverage vs Views-per-Fact

**Date:** 2026-10-08
**Author:** Claude (Opus 4.8), under Dr. Hemant Joshi
**Status:** proposed — needs GPU (sequential, one arm at a time on the A10G)
**Depends on:** `docs/knowledge_retention_levers.md`, `results/knowledge/uc4/volume_curve_e2b_claude.json`, `results/knowledge/uc4/volsweep_rewrites_arm.json`

---

## 1. Objective & question

We have two established points for the AWS-Bedrock corpus (Gemma-E2B, r64, closed-book, Claude-judged, 481 held-out facts, coverage@τ0.5):

| recipe | views/fact (rewrites/chunk) | coverage ALL | coverage non-leak |
|---|---|---|---|
| QA-only (plateau) | 0 | 0.308–0.331 | 0.331–0.357 |
| QA + rewrites | ~10 | 0.389 | 0.398 |
| *RAG recall@5 ceiling* | — | *0.603* | — |

**Question:** how does held-out retention scale as we increase diverse *views per fact* from 0 → ~40, and does it keep climbing toward the RAG ceiling (EntiGraph predicts log-linear growth) or plateau below it? Produce a clean **coverage vs views/fact** curve, with the *diversity-vs-volume* confound controlled.

## 2. Hypotheses

- **H1 (scaling):** coverage rises monotonically and ~log-linearly in views/fact over 0→40, per EntiGraph (2409.07431) and Allen-Zhu 3.1/3.3.
- **H2 (diversity, not volume):** the lift comes from *distinct rephrasings*, not merely more training tokens. A volume-matched control (same example count, non-diverse) will **not** reproduce the lift.
- **H3 (ceiling):** even at 40 views/fact the SLM stays below RAG's 0.603 — parametric knowledge complements, not replaces, retrieval.

## 3. Design

### 3.1 Primary sweep (varies only views/fact)
Fixed corpus (6,154 chunks), fixed ~10 distinct QA/chunk, vary **rewrites/chunk N**:

| arm | N (rewrites/chunk) | status | notes |
|---|---|---|---|
| `rw_N0` | 0 | **have** | = QA-only `qafinal` (re-gen at 48 tok for clean comparison, §5) |
| `rw_N5` | 5 | new | |
| `rw_N10` | 10 | **have** | `phase3_E2B_r64_60kqa_rw` — but re-gen predictions at 48 tok (§5) |
| `rw_N20` | 20 | new | |
| `rw_N40` | 40 | new | |

→ **3 new training runs** (N5, N20, N40); N0 and N10 adapters exist.

### 3.2 Controls (rule out the "just more data" explanation — the advisor's flag)
- **C1 — volume-matched, low-diversity:** train an arm with the **same number of training examples** as `rw_N20` but using **duplicated/near-identical rewrites** (e.g. 2–3 distinct rewrites each repeated to hit the count) instead of 20 diverse ones. If C1 ≈ `rw_N0` and ≪ `rw_N20`, the lift is diversity (H2 confirmed). One run.
- **C2 — distinct-QA volume control (cheap, partly done):** the volume sweep already showed *more distinct QA* is inert past ~20k. Cite it; no new run. It establishes that raw example-count alone does not move coverage.

### 3.3 Fixed factors (hold constant for comparability with the existing curve)
Base `google/gemma-4-E2B-it` · 4-bit · LoRA **r64 α128**, target_modules attention-only (q/k/v, matches the curve) · 1 epoch · effective batch 16 · lr 2e-4 · seed = data_seed = 42. Same 6,154-chunk corpus, same ~10 distinct QA/chunk. Only `--rewrites` content/count varies.

> *Optional second factor (do only after the primary curve):* repeat the N20 arm with MLP targets added (`gate/up/down_proj`, lever 3 in the retention memo) to test the where-not-how-much capacity hypothesis. Keep out of the main sweep to avoid confounding.

## 4. Generation (DeepSeek via CodeVector — CPU/API, no GPU)

Use the existing generator `tools/gen_diversity_deepseek.py` (`--rewrites-per-chunk`, prompt already enforces "preserve ALL factual content exactly; vary structure/vocabulary substantially").

**Nested generation (do once, subsample):** generate **40** diverse rewrites/chunk for all 6,154 chunks a single time, then build each arm's training file from the **first N** rewrites (random but fixed order) so arms are strictly nested (N5 ⊂ N10 ⊂ N20 ⊂ N40). This (a) saves ~½ the API spend vs regenerating per arm and (b) removes generation noise between arms.

- Output: `examples/usecase4_bedrock_userguide/rewrites_N40.json` (then slice to `rewrites_N5/10/20.json`).
- Volume for N40 ≈ 6,154 × 40 ≈ **246k rewrites** (~passage-length). Budget/throughput check against the CodeVector `deepseekv4p` rate before committing; generate in resumable batches by chunk range.
- C1 file: take 2–3 distinct rewrites/chunk, duplicate to the N20 example count.

## 5. Training & evaluation (reuse the validated pipeline exactly)

**Train** (sequential, **one arm per GPU at a time** — see `gpu-generation-sequential` memory; 3 resident 4-bit models caused silent CPU offload):
```
# per arm, on the GPU host venv_gpu, from-base, rewrites appended as continuation LM text:
python -m pipeline.lora_trainer --config <e2b_r64_idea1.json> \
    --faqs faqs_full_idea1.json --rewrites rewrites_N<N>.json \
    --epochs 1 --output lora_checkpoints/phase3_E2B_r64_rw_N<N>
```

**Generate predictions — CRITICAL for comparability:** regenerate closed-book predictions for **every** arm (including N0 and N10) with the **same harness and `max_new_tokens=48`** used by the volume sweep (`build_generate_fn`, sft_format="chat"), via the sequential dump driver (`xargs -P 1`, one ~7 GB model per process). This removes the 39-vs-48-token / harness caveat that currently qualifies the N10 point.

**Judge:** in-session Claude, **partial rubric** (`eval/metrics.py:PARTIAL_SYSTEM_PROMPT`), temperature 0, one judging subagent per arm writing `{idx,verdict,score}`; refusals/empties = INCORRECT. Same judge as the curve → directly comparable.

**Aggregate:** `eval/knowledge_profile.coverage(…,τ=0.5)` + `mean_fidelity`, and **split leak/non-leak** using the already-computed, idx-aligned `results/knowledge/uc4/volsweep_leak_flags.jsonl` (reusable as-is — same 481 ordering). Report ALL / non-leak / leak per arm.

## 6. Outputs & plot

- `results/knowledge/uc4/rewrites_scaling_curve.json` — per-arm {N, coverage_all, coverage_nonleak, mean, counts, n_examples}.
- **Chart:** coverage@τ0.5 (and non-leak) on Y vs **views/fact N** on X (log scale: 0,5,10,20,40), base floor 0.083 + RAG ceiling 0.603 reference lines, C1 control point marked. Add as a second panel to the existing artifact. Expect: log-linear rise (H1); flatten approaching some asymptote < 0.603 (H3).

## 7. Decision criteria

- **Keeps climbing log-linearly through N40** → diversity augmentation is the lever; scale further toward EntiGraph-style synthetic CPT (lever 2) and report the scaling law.
- **Plateaus well below 0.603** → parametric recall saturates; pivot to **complementarity** (FT + RAG) and/or RAFT (lever 4) for the product path.
- **C1 ≈ N20** (volume, not diversity, drives it) → the whole rewrites story is really a data-volume story; re-interpret and stop the sweep. *(Expected not to happen, given QA-volume is inert — C2.)*

> **OUTCOME 2026-10-09 — this criterion is WEAKLY TRIGGERED.** The C1-style control was run early (as a from-base A/B, not continued-from-N10): distinct rewrites (A) vs one rewrite repeated to matched count (B) gave **A 0.233 vs B 0.220 on 450 treated facts, Δ +0.013, n.s. (χ²=0.52)**; B recovered ~91% of A's over-base lift, within the judge's ~0.012 noise floor. So within this regime it is **passage *exposure*, not distinctness**, that drives most of the lift — i.e. closer to a "volume/exposure" story than a "diversity" story. Caveat: the A/B is a from-base injection test at ~0.23 (below N0), low-powered for small effects (bounds distinctness to ≲0.03), so this weakly — not decisively — triggers the re-interpret criterion. See `results/knowledge/uc4/rwscale_AB_diversity.json`.

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Generation cost/time for 246k rewrites | Nested (generate 40 once, subsample); resumable batches; sanity-check rate/cost on a 100-chunk pilot first. |
| Rewrite quality drift / factual drift at scale | Spot-audit a sample per chunk; the generator prompt enforces fact preservation; a bad rewrite set would *lower* coverage — detectable. |
| GPU serialization slow (5–6 sequential runs) | Accept; 1 epoch each is the existing ~45-min/run cost; schedule overnight. Never run >1 resident model. |
| Hallucination rise at high N (Gekhman) | Track mean_fidelity and INCORRECT-rate alongside coverage; flag if INCORRECT climbs with N. |
| Judge drift across arms | Same rubric + temperature 0 + same 481; re-judge N0/N10 in the same batch as new arms. |

## 9. Effort estimate

**CORRECTED 2026-10-09 (measured, not estimated):** the original "~3–4 h total / ~45 min per run"
was wrong by ~20×. On `config_e2b_r64_60kqa` the trainer runs **per_device_batch=1 ×
grad_accum=16 + gradient_checkpointing**, and the rewrites are passage-length. A from-base arm on
60k QA + ~66k rewrites = ~126k examples = **~7,887 optimizer steps at ~8 s/it ≈ 17 h GPU** (measured
on the targeted-N20 run: 876/7887 in 117 min). Implications:
- A single from-base N20 arm ≈ **~17 h**; full N40 ≈ **~25 h**; the full sweep (N5/N20/N40/C1) ≈
  **4–5 GPU-days** sequential. **GPU time, not generation tokens, now bounds this experiment.**
- A from-base arm also carries run-to-run noise (adding rows reshuffles the whole dataset), and a
  power check shows the expected N10→N20 lift (~0.018 if the 0.068 N0→N10 lift is log-linear) is
  about the same size as the paired SE (~0.018 at ~15% discordant facts). So a single 17 h from-base
  targeted arm would likely return an **ambiguous "no lift."**
- **Cheaper, higher-power contrast (preferred):** *continued* training from the N10 adapter — Arm A
  (N10 + ~10 new views/treated-chunk) vs **Arm B = control C1** (N10 + each chunk's first new
  rewrite duplicated to the same count). Identical start weights (no reshuffle noise), ~280 steps ≈
  **~1 h each**, and it directly tests the diversity hypothesis H2 that nothing else does. Include
  the same ~500-QA sample in both (empty `--faqs` breaks the adaptive max_length). Treat A vs B as a
  mechanism test (does diversity help at all), not a magnitude estimate — recency inflates both.
- Original generation estimate also revised: external `deepseekv4p` is revoked; in-session Claude is
  the working generator (see §10 pilot + the targeted-N20 generation, 2026-10-08).

## 10. Checklist
- [x] **Generation pilot (2026-10-08) → see `results/knowledge/uc4/rewrites_pilot_report.json`.** Outcome: the planned external generator is **broken** — `deepseekv4p` grant revoked (403); the only granted model `deepseek-code-h` is a slow reasoning model (~24–89s/call, truncates JSON), making full N40 ~64–128 h and unreliable. In-session Claude generation is **viable** (200/200 rewrites, ~16 rw/min, ~6% first-pass defect rate needing a validate-and-repair loop, moderate diversity) but ~30 M tokens for full N40. **Decision: NO-GO as written.** Preferred path **B** (reduced sweep N∈{0,10,20}: only N20 needs generating, +10/chunk ≈ 61.5 k rewrites in-session) now; path **A** (restore a fast generator grant) unlocks the full N40 curve. Also surfaced: corpus chunks have OCR header residue + 600-char mid-sentence truncation that caps answerable facts — worth a cleaner re-chunk.
- [x] **N0/N10 re-measured at matched 48 tok (2026-10-09)** → `results/knowledge/uc4/rwscale_N0_N10_48tok.json`. N0 (pure-QA qafinal) 0.308 → N10 0.378, **+0.071 paired p≈0.0004**. The ~10-view lift is real and significant; corrected N0 (earlier `phase3_E2B_r64_60kqa` used replay 0.2 and was discarded).
- [x] **A/B distinctness control run (2026-10-09, replaces the from-base-N20 targeted gate — killed as underpowered ~17 h)** → `rwscale_AB_diversity.json`. A(distinct) 0.233 vs B(1-rewrite-repeated) 0.220 on 450 treated facts, **Δ +0.013 n.s.**; B recovers ~91% of A's over-base lift ⇒ passage *exposure*, not distinctness, drives most of the lift (see §7 outcome). Artifact updated (v4).
- [ ] *(optional, needs fast generator + GPU budget — see §9)* full nested N5/N20/N40 sweep for the coverage-vs-views curve; build a proper C1 continued-from-N10 control if the distinctness question is revisited at the N10 operating point.
- [ ] *(deferred)* pivot per §7: parametric views plateau ~⅔ of RAG ceiling → complementarity (FT+RAG) / RAFT for the product path. **Now pre-registered** in `docs/levers_bca_experiment_design.md` §7 (Phase 2), gated behind the B/C/A parametric levers.
- [ ] Record outcome + decision in `knowledge_measurement_plan.md`
