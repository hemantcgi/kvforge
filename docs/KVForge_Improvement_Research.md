# Deep Research: Making KVForge a Credible "SLM Replaces RAG" System

*Compiled 2026-07-19. Produced by a 106-agent deep-research run: 5 search angles, 24 primary sources fetched, 117 claims extracted, top 25 adversarially verified (3-vote refutation panels): 21 confirmed, 4 refuted. Claims from directions 3–4 (gating, KV fidelity) were extracted from primary sources but not adversarially verified — marked accordingly.*

## Verdict

**The evidence supports an evolution of KVForge, not a rewrite — but the target narrative needs one word changed: "replace" → "progressively replace, with retrieval as the shrinking fallback."** Every 2023–2026 result that gets parametric answering to RAG level still finds parametric + retrieval strictly better than either alone (EntiGraph + RAG 62.60% vs 60.35% for base + RAG), and two independent negative-result lines show naive fine-tuning loses to retrieval on genuinely new facts and under continuous document drift. The defensible claim — and the one KVForge's three-phase design is uniquely positioned to make — is **progressive parametric consolidation**: knowledge migrates from retrieval into parameters as it stabilizes and proves out, per-knowledge-item, with a calibrated gate deciding which path serves each query.

The good news: KVForge's own headline finding (training-signal quality, not model capacity, is the bottleneck) is **independently confirmed by three peer-reviewed research lines**, and the literature says exactly how to fix it.

---

## Direction 1 — Training signal: the root cause of weak parametric accuracy is confirmed, and fixable

**Confidence: HIGH (peer-reviewed, adversarially verified 3-0).**

1. **Knowledge trained without diversity is memorized but not extractable.** Physics of Language Models 3.1 (ICML 2024, arXiv:2309.14316): without augmentation (paraphrases, shuffling, translations) applied *during knowledge learning*, QA extraction accuracy is ~0% *regardless of subsequent instruction fine-tuning*. Extraction ability correlates strongly with training-data diversity. Prescription: rewrite the corpus with auxiliary models and **mix QA data into the knowledge-learning stage — not after it**. KVForge Phase 3 (LoRA on FAQs alone, documents never diversified) is precisely the anti-pattern this paper indicts. (Caveat: the literal 0% is from-scratch pretraining on synthetic biographies; with a pretrained Llama-3.2-3B the effect attenuates.)

2. **EntiGraph (ICLR 2025, arXiv:2409.07431) is the best-evidenced closer of the parametric-vs-RAG gap.** Entity-graph synthetic continued pretraining lifts Llama-3-8B closed-book QuALITY from 39.49% → 56.22% (>80% of RAG's absolute gain), log-linear scaling up to 455M synthetic tokens (~350× per source token). Raw continued pretraining on the source corpus performs *worse than base*; plain paraphrasing scales poorly. Parametric knowledge **compounds** with retrieval (62.60% with RAG vs 60.35% base+RAG).

3. **Synthetic Mixed Training (Mar 2026 preprint, arXiv:2603.23562) is the only result claiming parametric-beats-RAG end-to-end**: a mixture of synthetic documents + synthetic QAs restores log-linear scaling where any single method plateaus below RAG, and beats RAG in 5 of 6 settings (+4.4% relative on QuALITY, Llama-8B). *Medium confidence: unreviewed preprint, no independent replication.*

4. **The tunable lever is variations-per-chunk, effective range 10–40.** Accuracy is monotonically increasing in paraphrase/rewrite count (Ovadia et al., EMNLP 2024, verified up to 10 GPT-4 paraphrases; Abonizio et al. 2025 up to 40), with some evidence of saturation near ~10. One-to-few FAQ pairs per chunk — KVForge's current regime — is far below the effective range. **Refuted in verification (do not cite): "40× rewrites reach RAG parity (~88% vs ~90%)" — killed 1-2.**

**Bounding negative results (HIGH confidence, EMNLP 2024, arXiv:2312.05934):** RAG consistently beats unsupervised fine-tuning for knowledge injection at 7B, on both seen and new knowledge; fine-tuning Llama-2-7B on new facts can *degrade* it (0.353 → 0.219); even 10-paraphrase FT reaches 0.588 vs 0.875 for base+RAG. Also: fine-tuned models memorize training QA but fail rephrased/derivative questions (~37% new-knowledge generalization, ~19% for updates; arXiv:2411.05059), cannot synthesize two learned facts, and *overconfidently answer* out-of-scope variants — fine-tuning worsens the abstention problem.

**KVForge integration sketch:**
- Extend `sleep_faq_generator` into a **sleep-time corpus diversifier**: per chunk, emit (a) 10–40 diverse rewrites — entity-centric summaries, style transfers, Q-embedded restatements; (b) an entity graph across chunks with relation-path texts connecting entity pairs (the EntiGraph trick — this is what generates *diversity that paraphrasing can't*); (c) QA pairs including multi-hop questions spanning chunks.
- Change `lora_trainer` to train on **one joint mixture** of rewrites + QA (mixed training), not FAQ-only.
- Make `variations_per_chunk` a first-class config knob; ablate 1/5/10/20/40 on UC3 (SQuAD, the near-zero corpus) — this is the paper's E-series follow-up experiment.
- Cost note: full EntiGraph's 350× ratio is expensive; the verified-cheap 10–40× regime is the right first step. Open question: where the cost/accuracy knee sits between 40× and 350×.

---

## Direction 2 — Continual updates without forgetting: freeze the base, route to adapters

**Confidence: MEDIUM (preprints, verified 3-0 against their own tables).**

1. **Poly-PRAG (arXiv:2511.17044) solves KVForge failure mode (d) structurally.** Replace the monolithic per-round LoRA with a **fixed bank of ~20 jointly-trained shared LoRA adapters + a learned routing matrix**. Beats P-RAG and DyPRAG on avg F1 across four QA benchmarks (32.68% vs 26.99%/28.80% at 1B; 42.68% vs 41.59%/37.16% at 8B); storage 2–3 orders of magnitude smaller (42MB vs 12GB at 300-doc/1B scale). Critically, the Poly-z variant absorbs a **new document by training only a 0.32MB routing vector** with the bank frozen — comparable F1 (41.37 vs 42.46) on held-out docs. For KVForge: base weights never change → **stored KV tensors are never invalidated** (kills the 42% recompute overhead), forgetting of the bank is structurally avoided, and routing vectors live in the vector DB next to the embeddings. Caveats: new-doc eval was only 10 HotpotQA documents; 2,000–3,000-chunk scale with daily additions untested — pilot first. Open question: how often the bank itself needs retraining (which would re-trigger KV invalidation).

2. **Under continuous knowledge drift, retrieval wins — route by stability.** The 2026 "RAG or Learning?" study (arXiv:2604.05096): the purely retrieval-based, time-aware Chronos system gains +28.8pp over ReAct RAG on complex multi-source queries, and the authors report parametric-update methods have limited robustness under drift. **Refuted in verification (do not cite): the specific LoRA-forgetting numbers from this paper (35.14% vs 61.26%; 4.13%/10.17% on C2/C3) — killed 0-3.** Integration: add a **stability gate** — route queries touching recently-added/revised documents (age < N days, or revision-active) to Phase 1/2 retrieval; only knowledge that has stabilized is promoted to parametric serving.

3. **AlphaEdit (ICLR 2025 oral, arXiv:2410.02355) is a scalpel, not an ingestion path.** Null-space-projected editing preserves prior knowledge with a theoretical guarantee; boosts locate-then-edit methods by 36.7% avg. The 2026 reproduction (arXiv:2606.26783) confirms ~99% efficacy at ~2,000 edits on LLaMA3-8B but: degradation beyond ~5,000 sequential edits (BoolQ collapses 0.79 → 0.08 F1 at 10,000), **complete failure on some small architectures (0.00 efficacy on Phi3-3.8B; 0.43 on Gemma2-2B)** due to fused projections/post-FFN layernorm, and fluency/consistency worse than originally reported. Integration: an `edit` CLI path for single-fact corrections (renames, deprecations) between adapter refreshes, with a nightly regression probe over preserved knowledge — but **validate the mechanism on Llama-3.2-3B first**; transfer to arbitrary 1–4B models cannot be assumed.

---

## Direction 3 — A factual gate that actually works (replacing cosine PRS)

**Extracted from primary sources; NOT adversarially verified. Numbers below are as-reported by the papers.**

The paper's own data (PRS-cosine Pearson −0.12–0.43, ECE 0.066–0.166, non-monotonic) makes this the most safety-critical fix. The literature offers a three-layer replacement:

1. **Semantic Entropy Probes** (arXiv:2406.15927): linear probes on hidden states approximating semantic entropy from a **single forward pass** (vs ~10× compute for sampled semantic entropy). AUROC ~0.7–0.95 with later layers; generalize out-of-distribution better than accuracy probes (+7.7 to +10.5 AUROC across Llama-2-7B / Phi-3 / Mistral-7B); can gate **before generation begins**. Exactly the right cost profile for per-query gating on one A10G. KVForge already extracts hidden states for embeddings — a probe head is nearly free.
2. **R-Tuning refusal training** (NAACL 2024, arXiv:2311.09677): partition training questions into certain/uncertain by the model's own correctness, append refusal expressions to the uncertain set. Refusal rates >87% on unanswerable benchmarks vs 0.96–34.79% for baselines; better calibration than post-hoc estimation; validated at 3B (OpenLLaMA-3B) though gains shrink at that scale. Natural fit: KVForge's replay buffer already knows which chunks/questions the model gets right — that partition is sitting in the SQLite DB. Also composes with the Sprint 2.5 confidence-token work (same "trained self-assessment" family).
3. **Conformal abstention** (arXiv:2405.01563, DeepMind): conformal prediction over self-consistency scores yields an abstention rule with a **theoretically bounded hallucination rate** — turns the Phase 3 gate from a tuned threshold into a stated risk guarantee ("≤5% error at measured coverage"), which is both a production story and a paper-strengthening claim. Cost: multiple sampled generations per query — usable for *phase-transition auditing* (offline, per training round) even if too expensive per-query.
4. **Cautions:** entropy-only gating has model-dependent failure modes — combine entropy with an internal-state correctness probe and evaluate the *full selective-prediction policy at a stated risk level* (arXiv:2603.21172). And semantic entropy detects only *confabulations* (seed-variable errors) — **not systematically trained-in errors** from flawed synthetic FAQs (Nature 2024 / OATML), which is exactly the error class LoRA-on-bad-data produces. The FAQ quality filter (factual-accuracy filtering against teacher answers, already built in Sprint 2) is therefore a required complement, not an optional one.

**Integration sketch:** PRS v2 = (SEP probe AUROC on held-out corpus questions) + (R-Tuned refusal behavior) + (conformal calibration of the phase-transition threshold), replacing the cosine accuracy component the paper already recommends retiring (§7.9). Immediate, cheap, and publishable as a contribution in its own right.

---

## Direction 4 — Fixing Phase 2 KV-injection fidelity (the 0.82–1.14 attention-KL gap)

**Extracted from primary sources; NOT adversarially verified. Numbers below are as-reported by the papers.**

The literature converges on *why* naive non-prefix KV concatenation fails — lost cross-chunk attention, wrong positions, and spurious attention sinks at every chunk boundary — and offers fixes at three cost points:

1. **Cheapest — LegoLink/EPIC static sink-token recompute** (arXiv:2410.15332): the dominant failure is a spurious **attention sink at the first tokens of every cached chunk**; recomputing just the first k ≤ 32 tokens per chunk (even k=2) keeps accuracy drops within 0–7% at up to 8× TTFT improvement — far cheaper than CacheBlend's ~15% dynamic recompute. Validated on 7–9B models. *This is the first thing to try in `kv_inference.py` — it's a ~50-line change.*
2. **Training-free re-alignment — APE** (ICLR 2025, arXiv:2502.05431): shared prefix + attention temperature + scaling factor re-align parallel-encoded caches with the sequential attention distribution; recovers 98% of sequential performance on RAG tasks, 4.5× end-to-end speedup, zero recompute and zero training.
3. **Dynamic recompute — CacheBlend** (EuroSys'25 best paper, arXiv:2405.16444): recompute ~15% high-KV-deviation tokens; F1 within 0.01–0.03 of full prefill, TTFT 2.2–3.3× better than full recompute; recompute hidable behind KV load from storage (~3ms vs ~16ms per layer) — directly relevant to KVForge's vector-DB tensor storage. Also confirms **RoPE positional re-alignment is cheap** (a rotation on K); cross-attention is the real problem.
4. **Highest quality, needs training — KVLink** (arXiv:2502.16002): store KV *without* RoPE baked in, re-apply global RoPE at inference, add ~5 trained link tokens per document. **At exactly Llama-3.2-3B: 64.4% vs CacheBlend's 42.6% on NQ; 69.5% vs 32.7% on HotpotQA**; TTFT −60–90%. Cost: mixed-data fine-tuning (~892K samples on 8×A100) — likely out of single-A10G budget unless a scaled-down recipe works.

**Integration order:** LegoLink sink-fix → APE temperature/scaling → CacheBlend HKVD for Enhanced-Tier chunks → KVLink only if a training budget materializes. Success metric: close the attention-KL gap below ~0.1 and demonstrate KV-injection ≥ text-RAG on the Phase Quality Matrix — which would finally make Phase 2 "a strict accuracy win" as §7.8 demands. Note KVLink's RoPE-free storage changes the KV serialization format (`core/kv_utils.py`) — decide before regenerating tensors at scale.

---

## Ranked roadmap (expected impact on parametric factual accuracy + freshness)

| # | Change | Attacks | Evidence tier | Effort |
|---|--------|---------|---------------|--------|
| 1 | Mixed diverse-rewrites + QA training, 10–40 variations/chunk | token-F1 0.01–0.30 | Peer-reviewed ×3, verified | Medium (data pipeline only) |
| 2 | EntiGraph-style entity-graph corpus diversifier | diversity ceiling above paraphrasing | ICLR 2025, verified | Medium |
| 3 | Fixed adapter bank + per-doc routing vectors (Poly-PRAG) | 42% KV invalidation, forgetting, daily updates | Preprint, verified vs tables | High (arch change; pilot first) |
| 4 | PRS v2: SEP probe + R-Tuning refusal + conformal threshold | gate Pearson −0.12–0.43, ECE | Peer-reviewed, *not verified by panel* | Low–Medium |
| 5 | KV fidelity ladder: sink-fix → APE → CacheBlend → (KVLink) | attention KL 0.82–1.14 | Peer-reviewed + preprints, *not verified by panel* | Low → High |
| 6 | Stability gate: fresh docs → retrieval, stable → parametric | drift losses | Preprint, verified | Low |
| 7 | AlphaEdit correction channel (after 3B validation) | stale single facts | ICLR oral + reproduction, verified | Medium |

## What the abstract can honestly claim (proposed narrative)

> Retrieval-augmented generation treats external memory as permanent infrastructure: every query pays retrieval and re-encoding forever. KVForge treats it as **scaffolding**: a small language model progressively absorbs its document corpus into parameters through diversity-augmented synthetic training, serving each query from weights the moment a calibrated gate certifies the knowledge as learned — and falling back to retrieval precisely where it is not yet, or no longer, true of the parameters. New documents enter through retrieval on day one, are diversified and trained into a routed adapter bank as they stabilize, and individual stale facts are corrected by targeted, null-space-constrained edits — so the system stays current without invalidating its knowledge or its precomputed KV state.

This narrative is (a) supported by the verified literature (parametric knowledge compounds with retrieval; drift favors retrieval for fresh facts; mixed synthetic training approaches/exceeds RAG on stable corpora), (b) unclaimed by any existing system — CAG has no learning loop, LMCache/CacheBlend have no parametric tier, Parametric-RAG has no lifecycle/gating — and (c) falsifiable with the E-series experiments above. Avoid claiming *full* replacement: that claim is currently contradicted by EMNLP 2024 and the 2026 drift study, and reviewers will know it.

## Refuted claims (do not cite in the paper)

1. "40× rewrites reach RAG parity (~88% vs ~90% TiEBe)" — killed 1-2 (arXiv:2508.06178).
2. "Paraphrase-only ≈ chunk-RAG accuracy; QA-style alone stays at baseline" — killed 0-3 (arXiv:2508.06178).
3. Chronos-paper LoRA catastrophic-forgetting specifics (35.14% vs 61.26%) — killed 0-3.
4. Chronos-paper LoRA multi-fact/multi-source specifics (4.13% / 10.17%) — killed 0-3.

## Open questions for the E-series experiments

1. Best selective-QA gate at 3B and its risk-coverage curve (SEP vs R-Tuning vs conformal vs combinations) — unresearched, safety-critical.
2. Can the KV fidelity ladder close the KL gap to <0.1 and make Phase 2 ≥ text RAG, or should Phase 2 be deprecated?
3. Where between 10–40× and 350× synthetic-tokens does the cost/accuracy knee sit for a 2,000–3,000-chunk corpus on one A10G?
4. Does Poly-PRAG routing hold at thousands of docs with daily additions, and how often must the bank retrain?

## Source quality summary

Peer-reviewed & verified: EntiGraph (ICLR'25), Physics of LMs 3.1 (ICML'24), Ovadia et al. (EMNLP'24), AlphaEdit (ICLR'25 oral). Preprints, verified against own tables: Synthetic Mixed Training (2026), Poly-PRAG (2025), Chronos (2026), AlphaEdit reproduction (2026). Peer-reviewed but not panel-verified (direction 3/4): SEP, R-Tuning (NAACL'24), conformal abstention (DeepMind), CacheBlend (EuroSys'25 best paper), APE (ICLR'25), EPIC/LegoLink, KVLink, Nature 2024 semantic entropy. Nearly all quantitative results are at 7–8B or 1–1.5B (KVLink is the exception with exact Llama-3.2-3B numbers); most benchmarks are multiple-choice, easier than KVForge's free-form token-F1 — directions transfer, absolute numbers will not.
