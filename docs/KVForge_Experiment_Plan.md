# KVForge Experiment Plan: Hypotheses, Implementation, and Evaluation

*Companion to `KVForge_Improvement_Research.md`. Each proposed improvement is stated as a falsifiable hypothesis with a pre-registered success criterion, an experiment that can confirm or refute it on the existing UC1–UC4 corpora, and implementation tasks mapped to actual KVForge modules. Experiment numbering continues the paper's E1–E5 series.*

**Hardware assumptions:** single AWS g5.xlarge (A10G, 24GB), Llama-3.2-3B-Instruct, cloud LLM (Gemini 2.5 Flash / Claude) for synthetic generation and judging. LoRA round ≈ 20 min; full KV recompute ≈ 8.5 min per 2,520 chunks.

---

## E0 — Evaluation-harness hardening (prerequisite, not a hypothesis)

Every hypothesis below is decided by the harness, so it must be fixed first. Current weaknesses (from the paper itself): UC4 has n=7 held-out questions; max-samples 50 elsewhere; EM is uninformative (0.000 everywhere); no confidence intervals; single seed.

**Tasks**
1. **Expand held-out sets to n ≥ 200 per corpus** where splits allow (UC2 PubMedQA and UC3 SQuAD have official dev splits; sample 200 stratified by chunk). For UC1 raise the FAQ holdout to ≥ 100. For UC4, generate 150 candidate questions with the teacher pipeline (`pipeline/distillation.py` already builds query pools), then **manually verify ~100** (researcher pass; ~2–3 hours) — these become the frozen `UC4-heldout-v2` set. Held-out questions are quarantined: never enter FAQ generation, replay buffer, or training mixtures (enforce by question-ID blocklist in `pipeline/sleep_faq_generator.py` and `core/replay_buffer.py`).
2. **Metrics**: keep token-F1 and LLM-judge as primary; replace raw EM with normalized/relaxed EM (lowercase, strip articles/punctuation, numeric equivalence) so it becomes informative for short answers; add abstention-aware metrics for gate experiments (coverage, selective risk, AURC).
3. **Statistics**: paired bootstrap (10k resamples) over questions for F1 deltas; McNemar's test for judge-binary deltas; report mean ± 95% CI. Decision rule used throughout this plan: *a hypothesis is confirmed only if the pre-registered delta is met AND the 95% CI excludes zero.*
4. **Seeds**: every trainable experiment runs 3 seeds (42/43/44); report mean ± std across seeds. Inference-only experiments run 1 pass with `do_sample=False`.
5. **Run tracking**: one JSON result file per run under `results/E{n}/{uc}/{condition}_{seed}.json` (schema: config hash, git commit, adapter version, per-question scores). A ~100-line `eval/run_registry.py` indexes them.
6. **Cost/GPU**: ~2 GPU-hours (regenerating baselines at n=200) + ~$15 cloud judging + one manual verification session.

**Baseline re-measurement:** rerun the Table-8 Phase Quality Matrix (text RAG / KV mean-pool / KV full-token / parametric) at n=200 with CIs. All later experiments compare against *these* baselines, not the paper's n=50 numbers.

---## H1 (E6) — Mixed rewrites+QA training beats FAQ-only training

**Hypothesis H1:** Training the LoRA adapter on a mixture of diverse per-chunk document rewrites *and* QA pairs lifts held-out parametric token-F1 and judge scores over FAQ-only training by ≥ 0.10 absolute token-F1 on at least 3 of 4 corpora, at equal training-token budget.

**H1a (dose–response):** Parametric accuracy increases monotonically with variations-per-chunk over {1, 5, 10, 20, 40}, with the knee at or below 20.

**Rationale:** Physics of LMs 3.1 (extraction requires diversity during knowledge learning, mixed-not-sequential), EntiGraph (raw/paraphrase-only fails), Synthetic Mixed Training (mixture restores scaling). KVForge's FAQ-only pipeline is the indicted anti-pattern.

**Design**
- *Independent variable:* training-data recipe. Conditions: (A) FAQ-only (current, control); (B) rewrites-only; (C) mixed rewrites+QA 1:1; (D) mixed 3:1 rewrites:QA. All matched on total training tokens (pad the smaller recipes by extra epochs so tokens seen are equal — record epochs as a covariate).
- *Dose–response arm:* condition C at 1/5/10/20/40 variations per chunk on UC3 (the near-zero corpus, biggest headroom) and UC4 (the production corpus).
- *Dependent:* held-out token-F1, judge, relaxed-EM; PRS as secondary; catastrophic-forgetting control — 100 fixed TriviaQA/commonsense questions measured before/after (flag if drop > 3pp).
- *Held constant:* base model, LoRA rank/lr/schedule, seed set, eval harness.

**Implementation tasks**
1. `pipeline/sleep_faq_generator.py` → add a `--mode diversify` path emitting per-chunk rewrites: entity-centric summary, style transfer (tutorial/reference/FAQ voice), question-embedded restatement, compressed factsheet. Prompt templates in `pipeline/prompts/diversify.py`. Config: `variations_per_chunk`, `rewrite_qa_ratio` in `core/config.py`.
2. `pipeline/lora_trainer.py` → accept a mixed dataset manifest (rewrites as continued-LM text, QA as chat-SFT) and interleave them in one training mixture; keep the Sprint-2.5 confidence-suffix path working for the QA portion.
3. Generation cost control: Gemini 2.5 Flash at ~$0.02/chunk-batch; 40 variations × 2,520 chunks (UC4) ≈ manageable in one sleep-time run; cache generations under `examples/<uc>/diversified/` so conditions reuse them.

**Procedure:** generate once per corpus at max dose (40) → subsample for lower doses (so lower-dose sets are strict subsets — removes generator variance) → train 4 conditions × 3 seeds on UC3+UC4, conditions A and C only on UC1+UC2 → evaluate on E0 held-out sets → paired bootstrap vs condition A.

**Success / refutation:** H1 confirmed if C or D beats A by ≥ 0.10 F1 (CI excluding 0) on ≥ 3 corpora. H1a confirmed if F1(dose) is monotone (Spearman ρ > 0.8 across doses) — the knee location is exploratory. If B ≈ A but C ≫ both, that itself is the Physics-3.1 "mixture matters" result and worth reporting. If all conditions ≈ A, the bottleneck hypothesis is wrong for this corpus scale and the paper's narrative must stay latency-led — that is a publishable negative result.

**Cost:** ~14 LoRA rounds ≈ 5 GPU-hours + ~$60–100 generation + judging. **Priority: first — highest expected impact, no architecture change.**

---

## H2 (E7) — Entity-graph diversification beats flat rewriting

**Hypothesis H2:** At equal synthetic-token budget, EntiGraph-style entity-graph synthesis (entity extraction → relation texts over entity pairs/paths, including *cross-chunk* connections) beats flat per-chunk rewriting (H1's best condition) by ≥ 0.05 token-F1, and disproportionately improves multi-hop/held-out questions touching ≥ 2 chunks.

**Rationale:** EntiGraph's mechanism argument: paraphrase diversity saturates; relation-path synthesis generates combinatorial diversity flat rewrites cannot. KVForge corpora (Bedrock docs, customer support) are entity-dense.

**Design**
- Conditions (equal token budget, same trainer as H1): (C*) H1's winning flat-rewrite mixture (control); (E) entity-graph synthesis mixture; (F) union C*+E at the same total budget (half each).
- New dependent metric: split held-out questions into single-chunk vs multi-chunk (annotate via retrieval trace: which chunks contain the gold answer); report F1 per stratum.
- Corpora: UC4 (entity-dense docs, primary), UC1 (paraphrastic, adversarial case — entity graphs may not help).

**Implementation tasks**
1. New `pipeline/entigraph_generator.py`: (i) per-chunk entity extraction (cloud LLM, JSON schema); (ii) corpus-level entity table with chunk back-references (SQLite alongside the replay buffer); (iii) relation-text generation for sampled entity pairs and 3-entity paths, biased toward pairs co-occurring in *different* chunks; (iv) emits the same dataset-manifest format as H1 so `lora_trainer` is unchanged.
2. Multi-chunk question annotation: extend `prs_evaluator`'s eval loader with a `chunks_touched` field.

**Success / refutation:** H2 confirmed if E or F beats C* by ≥ 0.05 F1 overall (CI excludes 0) OR by ≥ 0.10 on the multi-chunk stratum. Refuted if E ≤ C* everywhere — then flat rewrites suffice at this corpus size and the 350× EntiGraph regime is not worth its cost; report the knee found in H1a as the practical recommendation.

**Cost:** ~6 LoRA rounds ≈ 2 GPU-hours + ~$40 generation. **Depends on H1 completing (uses its winner as control).**

---

## H3 (E8) — Fixed adapter bank + per-document routing absorbs new documents without forgetting or KV invalidation

**Hypothesis H3:** A Poly-PRAG-style fixed bank of k jointly-trained LoRA adapters with per-document routing vectors can absorb a batch of *new* documents by training routing vectors only, reaching ≥ 90% of the F1 that full LoRA retraining achieves on the new documents, while (a) old-document F1 drops < 2pp and (b) zero stored KV tensors are invalidated.

**Rationale:** Poly-PRAG tables (routing-only new-doc F1 41.37 vs 42.46 full) at 300-doc scale; untested at KVForge scale — hence a pilot with an explicit go/no-go.

**Design (pilot on UC4)**
- Split UC4 corpus 90/10 by document: `D_base` (~2,270 chunks) and `D_new` (~250 chunks, held out of all initial training).
- Conditions: (G) current pipeline — monolithic LoRA on D_base, then continued LoRA round on D_base+D_new (measures forgetting + KV invalidation, control); (H) adapter bank k=8 and (I) k=20 trained on D_base, then **routing-vectors-only** training for D_new documents with the bank frozen.
- *Dependent:* F1/judge on three question sets — Q_new (questions over D_new), Q_old (over D_base), Q_general (forgetting control); wall-clock update time; bytes written per new document; count of invalidated KV tensors (should be 0 for H/I by construction — verify via `kv_version` fields).
- 3 seeds for bank training; routing-vector training is cheap enough to run per-seed.

**Implementation tasks**
1. New `core/adapter_bank.py`: k parallel LoRA adapter sets on q/k/v_proj; forward hook combines adapter outputs weighted by a routing vector z ∈ R^k (softmax). Train bank + per-document z jointly on D_base (documents batched by routing group); freeze bank for incremental docs, train z_new only (~0.3MB, minutes on A10G).
2. Routing storage: z vectors go into the vector-store payload next to embeddings (`pipeline/kv_indexer.py` upsert path). At query time, retrieve top-K chunks → aggregate their z (mean) → set active routing → generate. Wire into `pipeline/kv_inference.py` behind config flag `adapter_bank: {enabled, k}`.
3. Trainer entry point `python -m pipeline.adapter_bank_trainer --config cfg.json [--incremental docs/]`.
4. Guardrail: because base weights and bank are frozen during incremental updates, `kv\{}_version` never bumps — assert this in the E8 harness.

**Success / refutation:** H3 confirmed if the better of H/I reaches ≥ 90% of condition G's Q_new F1 with Q_old drop < 2pp. Partial outcome — bank F1 on D_base substantially below monolithic LoRA (> 5pp) — means the bank trades peak accuracy for updateability; report the tradeoff curve and gate adoption on H6's routing (retrieval covers the gap). Refuted if routing-only updates land < 70% of full retraining — then daily freshness must come from retrieval (strengthens the H6 stability-gate design; the paper narrative shifts to "batch consolidation, not per-document").

**Cost:** highest-effort item: ~3–4 days implementation + ~8 GPU-hours. **Run after H1 locks the training recipe (the bank trains on H1's mixture).**

---

## H4 (E9) — A probe+refusal+conformal gate beats cosine-PRS at selective parametric answering

**Hypothesis H4:** A gate built from (a) a linear semantic-entropy/correctness probe on hidden states, (b) R-Tuning-style refusal training, and (c) conformal threshold calibration achieves, at matched coverage, ≥ 30% relative reduction in selective risk (wrong-answer rate among answered questions) versus both the cosine-PRS gate and the Sprint-2.5 confidence-token gate; and its conformal variant holds a stated 10% risk bound on held-out data.

Sub-hypotheses, evaluated independently:
- **H4a (probe):** hidden-state linear probe AUROC ≥ 0.75 for answer correctness, beating confidence-token p(yes) AUROC.
- **H4b (refusal):** adding refusal-training examples (uncertain-set questions labeled "I don't know") raises abstention precision without dropping answered-question F1 by more than 2pp.
- **H4c (conformal):** split-conformal calibration on 100 calibration questions yields empirical risk ≤ the stated bound on the disjoint test set.

**Rationale:** The gate is the safety-critical unknown (PRS-cosine Pearson −0.12–0.43). SEP/R-Tuning/conformal are the three strongest literature candidates; KVForge has unique infrastructure for all three (hidden states already computed; replay buffer knows per-question correctness; conformal needs only a held-out split).

**Design**
- Data: for each corpus, run the Phase-3 model over all training-FAQ questions + E0 held-out; label each generation correct/incorrect by token-F1 ≥ 0.3 ∨ judge-correct (the fKDS blend already in the codebase). Split: probe-train (60%) / calibration (20%) / test (20%), split by *chunk* to prevent leakage.
- Gates compared (all evaluated as risk–coverage curves on the same test split): (J) cosine-PRS accuracy component (control 1); (K) confidence-token p(yes) (control 2, Sprint 2.5); (L) linear probe on layer-ℓ hidden state at last prompt token (sweep ℓ ∈ {50%, 75%, 90% depth}); (M) L + refusal-trained model; (N) M with conformal threshold at α = 0.10.
- Metrics: AUROC (correctness prediction), AURC, selective risk at coverage ∈ {20%, 40%, 60%}, empirical risk vs conformal bound, ECE of each signal.
- No sampling-based semantic entropy in the main arm (10× inference cost); run it once on UC4 only as an oracle upper bound.

**Implementation tasks**
1. `core/gate_probe.py`: extract hidden states during `prs_evaluator` runs (one hook, no extra forward), train logistic probe (sklearn, CPU), save per-corpus probe weights next to LoRA checkpoints.
2. R-Tuning data: `core/replay_buffer.py` already stores per-question outcomes — emit an uncertain-set refusal dataset (`answer → "I don't know…"`), mix at ~10% into the H1 training mixture; tag so H1 comparisons can exclude it.
3. `core/confidence_gate.py`: add `gate_mode: {cosine, token, probe, probe+conformal}`; conformal calibration = pick threshold as the ⌈(n+1)(1−α)⌉-th score on the calibration split (~30 lines).
4. Reuse Figure-15/16-style plots from the paper's tooling for risk–coverage output.

**Success / refutation:** H4 confirmed if N dominates J and K on the risk–coverage curve at all three coverages and the conformal bound holds. If H4a fails (probe AUROC < 0.75), fall back to reporting which signal wins; if *all* signals plateau near cosine-PRS, that is strong evidence that post-hoc gating cannot rescue a weak parametric model — Phase 3 must then be gated on *training-time* evidence (per-chunk fKDS), and the paper should say so.

**Cost:** ~2 GPU-hours (labeling generations) + CPU probe training + 3 refusal-augmented LoRA rounds ≈ 1 GPU-hour. **Runs in parallel with H1 (uses its runs' generations for labels).** Note H4b interacts with H1's mixture — run H4b on H1's frozen winner only.

---

## H5 (E10) — The KV-fidelity ladder closes the injection quality gap

**Hypothesis H5:** Cheap attention-fidelity fixes close most of the KV-injection gap: specifically **H5a**, recomputing only the first k ≤ 32 tokens of each injected chunk (sink fix) recovers ≥ 50% of the text-RAG-vs-full-token-KV F1 gap and cuts attention KL below 0.4; **H5b**, adding APE-style attention temperature/scaling recovers ≥ 80% of the gap, training-free; **H5c**, with both applied, full-token KV injection is statistically indistinguishable from text RAG (F1 CI overlapping) while retaining ≥ 2× prefill speedup.

**Rationale:** LegoLink (sink tokens are the dominant failure; k=2–32 suffices), APE (98% of sequential performance via temperature/scaling), CacheBlend (15% HKVD recompute; already partially implemented in KVForge as Sprint-1 `recompute_ratio`). The paper's own per-layer KL data (Fig. 18, divergence broadly distributed) will show whether the sink hypothesis transfers.

**Design**
- Inference-only conditions on the frozen H1-winner model, Enhanced-Tier full-token KV, all four corpora: (O) text RAG ceiling; (P) naive full-token injection (control, current); (Q) P + sink recompute k ∈ {2, 8, 32}; (R) P + APE temperature/scale (grid: temp ∈ {0.9, 0.8, 0.5}, scale per APE paper); (S) Q(best-k) + R(best); (T) existing `recompute_ratio` CacheBlend path at 15% for comparison.
- Dependent: held-out F1/judge; per-layer attention KL vs true prefill (harness exists — Fig. 18 tooling); TTFT and end-to-end latency; recompute token count.
- Diagnostic first: recompute attention maps on 20 questions and *look* at chunk-boundary sink mass before running the grid — if sink mass is not concentrated at chunk starts, reorder the ladder (APE first).

**Implementation tasks**
1. `pipeline/kv_inference.py`: (i) `sink_recompute_k` — after cache assembly, re-run forward for the first k positions of each injected chunk with full left context and overwrite those KV entries (bounded variant of the existing partial-recompute path); (ii) `attention_temperature`/`attention_scale` applied to injected-chunk keys at assembly time (APE); both behind config flags, composable.
2. RoPE re-basing check: verify current serialization stores position-rotated keys; if so, add rotation-delta correction when a chunk lands at a different offset than at index time (pure tensor op in `core/kv_utils.py`). *(Project folder is named RoPE — this experiment is the namesake.)*
3. Extend the attention-divergence harness to emit per-position KL (not just per-layer) to visualize sink behavior.

**Success / refutation:** thresholds as stated per sub-hypothesis; primary decision at k and temperature chosen on UC2 (worst KV corpus), validated untouched on the other three. If H5c holds, Phase 2 becomes "a strict accuracy win" and the Enhanced Tier justifies itself (Component-6 question answered). If even S leaves a > 0.05 F1 deficit on ≥ 2 corpora, pre-registered conclusion: **deprecate mean-pool injection entirely and demote full-token injection to latency-optional**, making Phase 2 = text-RAG + background consolidation — a simpler, honest architecture for the paper.

**Cost:** inference-only; ~4 GPU-hours for the grid. No cloud cost beyond judging. **Independent of H1–H4; can run any time after E0.**

---

## H6 (E11) — Stability-gated routing beats both pure-parametric and pure-RAG under document churn

**Hypothesis H6:** Under a simulated document-churn workload, routing queries by knowledge stability (documents added/revised < N days → retrieval; stable → parametric) achieves ≥ 95% of pure-RAG's F1 on fresh-document questions while retaining the parametric path's latency advantage on stable-document questions — beating pure-parametric on freshness by ≥ 15pp F1.

**Rationale:** Ovadia + Chronos: parametric loses on fresh facts; EntiGraph: parametric compounds with retrieval on stable facts. This experiment turns the "retrieval as scaffolding" narrative into a measured system result — it is the paper's flagship *systems* experiment for the new abstract.

**Design**
- Churn simulation on UC4: start with D_base (90%); inject D_new in 5 daily batches; additionally *revise* 5% of D_base docs (teacher-LLM edits that change facts — store old/new answer pairs). Question stream: 40% fresh-doc, 40% stable-doc, 20% revised-doc questions.
- Conditions: (U) pure parametric (Phase 3 always); (V) pure text RAG; (W) stability gate with N = 1 consolidation cycle (docs enter parametric after the next training round); (X) stability gate + H4 confidence gate stacked (the full proposed system).
- Consolidation between simulated days: routing-vector updates if H3 confirmed, else full LoRA round (records which regime the result generalizes to).
- Dependent: F1 by question stratum (fresh/stable/revised), staleness errors (model answers with pre-revision fact — the critical failure), mean latency per path, fraction of queries served parametrically over time (the "scaffolding shrinks" curve — Figure candidate for the paper).

**Implementation tasks**
1. `core/access_tracker.py` / vector-store payload: add `doc_added_at`, `doc_revised_at`; `core/confidence_gate.py` consults max-recency of retrieved chunks before allowing parametric routing.
2. `tools/churn_simulator.py`: batch injection + revision driver over the existing `index_and_train.py` orchestrator.
3. Revised-doc probes: on revision, auto-generate (old-fact, new-fact) question pairs → staleness metric.

**Success / refutation:** as stated. Watch the revised-doc stratum: if the stability gate fails there (parametric keeps serving pre-revision facts even after consolidation), that motivates H7 as the correction channel; if H7 is also refuted, revision-heavy corpora must pin to retrieval permanently — a scope condition the paper must state.

**Cost:** ~6 GPU-hours (5 consolidation cycles × 3 conditions) + generation for revisions. **Run last among the training experiments — composes H1/H3/H4 results.**

---

## H7 (E12) — AlphaEdit-style editing works as a correction scalpel on Llama-3.2-3B

**Hypothesis H7:** Null-space-constrained locate-then-edit achieves ≥ 90% edit efficacy and ≥ 80% paraphrase generalization for batches of ≤ 200 single-fact corrections on Llama-3.2-3B, with < 2pp degradation on a 500-question preserved-knowledge probe and no measurable fluency loss.

**Rationale:** AlphaEdit is strong on LLaMA-family at ≤ 3k edits but **fails completely on some small models** (Phi3-3.8B efficacy 0.00). Llama-3.2-3B's architecture is close to validated LLaMA models (unfused projections), so it *should* transfer — but this must be tested before the correction-channel design enters the paper. This is a genuine reproduction-style experiment with real refutation risk.

**Design**
- Edit sets: 50 / 100 / 200 counterfactual-style edits built from UC4 revisions (H6's old→new fact pairs) + CounterFact-formatted controls.
- Metrics (standard editing suite): efficacy (edited fact returned), generalization (paraphrase), specificity/locality (neighborhood facts unchanged), preserved-knowledge probe (500 questions incl. corpus + general), fluency (n-gram entropy), all vs pre-edit model. Compare against the honest baseline: routing-vector retrain (H3) or mini-LoRA on the corrected facts at equal wall-clock.
- Interaction check: apply edits to the model *with* the H1 adapter active vs base — does editing compose with LoRA?

**Implementation tasks**
1. Adapt the open-source AlphaEdit implementation to Llama-3.2-3B (layer stats via 100k-token covariance sample — one-time ~1 GPU-hour); wrap as `pipeline/fact_editor.py --edits edits.jsonl`.
2. Nightly regression probe integration into `prs_evaluator` (the preserved-knowledge probe doubles as a production safety check).

**Success / refutation:** thresholds as stated. If efficacy collapses (Phi3-style), pre-registered conclusion: corrections route through H3 routing-vector retraining instead, and the paper reports the negative transfer result (valuable — extends the reproduction study's architecture-sensitivity finding to Llama-3.2-3B). **Timebox: 2 days; this is the highest-risk, lowest-centrality experiment — do not let it block the paper.**

---

## Sequencing, dependencies, and decision tree

```
E0 (harness) ──┬── H1/E6 (mixture + dose) ──┬── H2/E7 (entity graph)
               │                            ├── H3/E8 (adapter bank pilot)
               │                            └── H4b (refusal arm)
               ├── H4a/c/E9 (probe+conformal — parallel with H1)
               ├── H5/E10 (KV ladder — independent, inference-only)
               └────────────► H6/E11 (churn systems experiment; composes H1+H3+H4)
                              H7/E12 (editing; feeds on H6 revision pairs; timeboxed)
```

- **Week 1:** E0 + launch H1 grid; H5 diagnostic (sink-mass visualization) on idle GPU time.
- **Week 2:** finish H1/H1a; H4a/c probes on H1's generations; H5 grid.
- **Week 3:** H2 + H3 implementation; H4b refusal round.
- **Week 4:** H3 pilot runs; H6 simulator build.
- **Week 5:** H6 full runs; H7 timebox.
- Total ≈ 30 GPU-hours on the A10G + roughly $150–250 cloud generation/judging. Every experiment ends with a go/no-go that feeds the paper: confirmations populate the new results sections; refutations become scope conditions in §8.

**Paper mapping:** H1/H2 → new §7 "Training-signal experiments" (replaces the E4 ablation story); H4 → rewritten §7.9 (gate); H5 → §7.8 Phase Quality Matrix v2; H6 → new flagship systems section + the "scaffolding shrinks" figure; H3/H7 → new "Continual updating" section; all refutations → §8.5 Limitations with pre-registered criteria cited.

**Pre-registration discipline:** this document is the registry. Before each run: freeze the success criterion here (git commit), then run. Deviations get logged in a `Deviations` appendix, not silently edited. That, plus reporting the refuted-claims list from the research report, makes the revised paper's methodology section notably stronger than v1's.
