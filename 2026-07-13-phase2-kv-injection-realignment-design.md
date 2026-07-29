# Phase-2 KV-Injection Realignment: Bug Fixes + Knowledge-Differentiation-Gated Eligibility

**Date:** 2026-07-13
**Status:** Implemented (code components 1–5; Component 6 validation pending GPU corpus runs)
**Related:**
- [2026-07-12-prs-gate-rework-design.md](2026-07-12-prs-gate-rework-design.md) — the gate rework this design mirrors (accuracy-gated eligibility instead of a cosine proxy)
- [2026-07-12-sft-training-signal-design.md](2026-07-12-sft-training-signal-design.md) — the SFT/paraphrase-augmentation work whose held-out-honesty discipline this design reuses
- This is item **#1** ("Phase-2 KV-injection realignment") from the original latency-vs-quality decomposition — the last of the three original pending items.

---

## Problem

KV injection (bypassing text encoding for retrieved chunks by injecting pre-computed KV tensors instead) does not consistently beat text-in-context RAG on factual accuracy, despite injecting the KV cache for the *same* retrieved content. Real evaluation data (`docs/scientific_revision_real/scientific_revision_results.json`, pre-dating the #2/#3 fixes) shows the gap is inconsistent and sometimes large:

| Corpus | text_rag judge | kv_meanpool judge | kv_fulltoken judge |
|---|---|---|---|
| UC1 | 0.133 | 0.167 | 0.100 |
| UC2 | 0.360 | 0.020 | 0.060 |
| UC3 | 0.280 | 0.260 | 0.080 |
| UC4 | 0.000 | 0.000 | 0.143 |

Two things stand out: `kv_fulltoken` (full per-token KV, in principle lossless relative to text) does not reliably beat `kv_meanpool` (a single mean-pooled pseudo-token per chunk) — on UC1 and UC3 it's *worse* — which undermines a pure "compression is the problem" theory. Something more systemic is going on.

Direct code inspection found four concrete, well-evidenced issues:

1. **Prompt asymmetry.** `generate_with_kv` and `generate_with_kv_fulltoken` (`pipeline/kv_inference.py`) prompt with a bare `"Based on the context provided, answer: {query}"` — no instructions. `generate_text_in_context` (text-RAG) uses a carefully engineered prompt: *"Using only the context below, answer in 2-4 sentences. Cite page numbers..."*. A `SYSTEM_PROMPT` constant with the right instructions (answer-only-from-context, cite sources, confidence) already exists in the file — and is never referenced anywhere.
2. **Phase-gating gap.** Per the architecture (`CLAUDE.md`), KV injection is supposed to activate "from Phase 2 onward." The code doesn't implement this: `decide_inference_mode` selects `kv_meanpool`/`kv_fulltoken` purely based on `kv_version` freshness, with no phase check at all — it can fire in Phase 1, before the corpus has proven anything.
3. **Dead enhanced-tier promotion.** `promote_chunk_to_enhanced_tier` (`pipeline/kv_background.py`), which would compute and cache full-token KV for frequently-accessed chunks, has zero callers anywhere in the pipeline. In production, nothing ever reaches the full-token path outside of eval's on-the-fly recompute — the "enhanced tier" described in `CLAUDE.md`'s tier system doesn't actually happen.
4. **Dead FAQ-to-chunk tagging.** `tag_faq_with_chunk_ids` (`pipeline/sleep_faq_generator.py`) is defined and has a consumer (`is_faq_stale` reads `source_chunk_ids`), but the real `generate()` loop never calls it — the FAQ dict it appends (`{q_key: ..., a_key: ...}`) has no `source_chunk_ids` field, even though `point_id` (the chunk's vector-store ID) is in scope at that exact point in the loop. Every FAQ in a real corpus today is untraceable back to the chunk that produced it.

Beyond these four bugs, there is no accuracy-based eligibility mechanism for KV injection at all — unlike Phase-2 *parametric* answering, which (per #2's gate rework) now has a hard eligibility gate (`is_eligible_for_parametric`) requiring measured similarity to a known-good query before the confidence gate even runs. KV-injection-of-retrieved-chunks has no equivalent: it fires unconditionally whenever KV tensors are fresh, with zero evidence it doesn't hurt quality for that content.

## Scope boundary (explicit and honest)

This targets **KV-injection-of-retrieved-chunks** (`kv_meanpool` / `kv_fulltoken` modes in `kv_inference.py`) — the Phase-2 mechanism that bypasses text encoding for matched chunks while still retrieving. It does **not** re-open pure-parametric answering (`confidence_gate.py`'s Phase 2/3 weights-only path), which #2 and #3 already reworked; the new Knowledge Differentiation Score (KDS) metric introduced here is deliberately built to *also* apply as an eligibility signal for that path later, but wiring it in there is out of scope for this plan.

The KDS metric is a new, less-precedented mechanism (unlike PRS's factual-accuracy signal, which had years of NLP precedent behind F1/EM/judge). Its eligibility threshold is **not** guessed — it is calibrated empirically in the validation phase (Component 6 below), the same discipline #2 used to data-derive its PRS weight rebalance from a 4-corpus backtest.

## Design

### Component 1 — Baseline bug fixes (groundwork)

Fixed *before* any KDS measurement, so the eligibility gate isn't calibrated against a baseline that's broken for reasons unrelated to KV injection itself:

1. **Phase-gating.** Thread `phase` into `decide_inference_mode` (`pipeline/kv_inference.py`): when `phase < 2`, always return `"text_fallback"` regardless of `kv_version` freshness. `answer_with_mode` passes `ver.get_phase()` through.
2. **Prompt parity.** Replace the bare `f"Based on the context provided, answer: {query}"` in `generate_with_kv` and `generate_with_kv_fulltoken` with a prompt built from the existing (currently unused) `SYSTEM_PROMPT` constant plus the query — appended as fresh tokens after the injected KV prefix (this doesn't require re-encoding chunk text; only the instruction+query span is newly tokenized, preserving the whole point of KV injection).
3. **Wire FAQ-to-chunk tagging.** In `pipeline/sleep_faq_generator.py`'s `generate()` loop, call `tag_faq_with_chunk_ids(faq, [point_id])` when appending each newly-generated FAQ, instead of appending an untagged dict. This is a prerequisite for Component 2 (KDS needs to map each chunk to its probe question) — without it, KDS has no way to find "the FAQ for this chunk."
4. **Refactor `_self_consistency` to expose embeddings, not just the scalar.** Checked directly: `_self_consistency` (`pipeline/prs_evaluator.py`) generates N sampled answers, embeds them, and returns *only* `float(np.mean(sims))` — the embeddings (`embs`) are a local variable, never returned. Component 2's within-chunk term needs those same embeddings to also compute the between-chunk term without re-generating and re-embedding identical samples a second time. Change `_self_consistency` to optionally return `(consistency_score, embs)` (default behavior unchanged for existing callers) so Component 2 can consume the embeddings directly.

### Component 2 — Knowledge Differentiation Score (KDS)

A new per-chunk metric measuring whether the fine-tuned model's parametric knowledge is genuinely differentiated by chunk topic — an empirical proxy for mutual information `I(C; A)` between chunk identity `C` and the model's generated answer `A` when probed about that chunk's topic without being shown its text.

**Why not perplexity.** Perplexity (surprise at the chunk's raw text under the model) was considered and rejected as the primary signal: it measures surface-text predictability, not knowledge. A model can have low perplexity on boilerplate without knowing the specific fact in it, or high perplexity on an unusual phrasing of a fact it knows well.

**Computation** (extends `pipeline/prs_evaluator.py`'s existing per-FAQ evaluation loop):

1. **Topic probes.** For each chunk, use its already-generated FAQ question(s) (tagged via `source_chunk_ids`, per Component 1 item 3) as probes — a chunk may have multiple FAQs (default `n_per_chunk=3`); pool samples from all of a chunk's probes into that chunk's within-chunk statistics rather than picking just one. When `sft_format`/paraphrase augmentation (#3) is enabled, prefer held-out paraphrases over the exact training phrasing — same held-out-honesty discipline as #3's validation, so the metric isn't just rewarding verbatim memorization of question wording.
2. **Sampling.** Generate N=3 parametric answers per probe question at temperature, reusing the existing `_generate_parametric`/`pipe_sample` machinery that `_self_consistency` already uses.
   - **Coverage strategy (rotating sample, not the existing `--sample 50` default).** `prs_evaluator`'s CLI currently evaluates `random.sample(all_faqs, min(50, len(all_faqs)))` per round — fine for factual-accuracy scoring, but far too small a slice for KDS's between-chunk term to be representative on a corpus with hundreds of chunks, and it would starve Component 4's replay-buffer feedback of signal for most of the corpus. KDS's per-round sample instead prioritizes chunks with no `last_kds_round` yet, then fills the remainder with the least-recently-measured chunks (a rotating sweep, similar in spirit to the existing hot/warm/cold/frozen access tiers). Persist `last_kds_round` per chunk alongside the `KDS` score (Storage, below) so each round can pick up where the last one left off. Full-corpus coverage accumulates over several rounds rather than requiring one expensive full sweep; corpus-level `mean(KDS)` is computed only over chunks that have been measured at least once.
3. **Embed.** Embed all sampled answers (existing `TextEmbedding` embedder, already used elsewhere in the file).
4. **Aggregate, per PRS-evaluation round:**
   - **Within-chunk dispersion `W`** — average squared distance of a chunk's N sample embeddings from their own mean. Nearly free once Component 1 item 4 lands: `_self_consistency`'s existing pairwise-cosine computation and this term are built from the same underlying sampled embeddings, so this reuses that generation/embedding work rather than duplicating it (the aggregation math itself — a variance around the mean vs. a mean pairwise cosine similarity — is a small new step on top of the same inputs).
   - **Between-chunk spread `B`** — squared distance of each chunk's mean embedding from the grand mean across *all* chunks evaluated this round.
   - **`KDS(chunk) = B_i / (B_i + W_i)`** — chunk `i`'s contribution to the corpus-wide variance-ratio (the ANOVA / intraclass-correlation structure — a standard normalized-effect-size proxy for MI under a cluster-separation assumption on the embeddings). In `[0, 1]`: 0 means the model's answer for this chunk's topic is indistinguishable from its answers on other topics (no chunk-specific knowledge, confidently generic — exactly the UC4 under-training failure mode from #3); 1 means the chunk elicits a distinct, self-consistent answer.
5. **Storage.** Persist per-chunk `KDS`, `last_kds_round`, and the round it was measured in the chunk's vector-store payload, alongside `kv_version`/`tier` — same pattern as existing per-chunk metadata. Persist corpus-level `mean(KDS)` (over measured chunks only) in `version.json`, trended over rounds like `prs_history`.
6. **Cost.** The within-chunk term is already computed today (no new generation cost). The only new cost is retaining per-chunk embedding centroids to compute the between-chunk term after the full sample is processed, plus no LLM-judge calls are needed for KDS itself — meaningfully *cheaper* per-chunk than the full PRS pass.

### Component 3 — KDS-driven eligibility gating

The actual behavioral change. In `decide_inference_mode` (or a wrapper consulted by `answer_with_mode`): a query's retrieved chunks must **all** clear the calibrated per-chunk KDS threshold for `kv_meanpool`/`kv_fulltoken` to be attempted — reusing the exact conjunctive "all chunks must qualify" pattern the function already uses for `kv_version` freshness (`for chunk in chunks: if not fresh: return "text_fallback"` extends naturally to `if kds < threshold: return "text_fallback"`). Any chunk below threshold forces `text_rag` for that query, corpus-wide phase gating from Component 1 still applies on top.

### Component 4 — Replay-buffer feedback loop

Low-KDS chunks get elevated priority in the next training round's replay buffer (`core/replay_buffer.py`) — a new weighting factor layered onto the existing access-based hot/warm/cold/frozen tier system (CLAUDE.md: hot=top15% by access, warm=next50%, cold=remainder, frozen=never-accessed). This is what makes "the coverage gap should reduce over time" an actual closed loop rather than a number that's merely reported: chunks the model doesn't yet differentiate well get preferentially retrained, directly targeting where the gap is open rather than uniformly re-sampling the whole corpus.

### Component 5 — Enhanced-tier (full-token KV) decision

Resolved empirically, not assumed. `promote_chunk_to_enhanced_tier` is dead code today; the pre-existing eval data shows `kv_fulltoken` doesn't reliably beat `kv_meanpool`. After the Component 1 prompt-parity fix (which could change this, since the old data was measured against the broken bare-prompt baseline), re-measure `kv_fulltoken` vs `kv_meanpool` factual accuracy on all 4 corpora. Wire `promote_chunk_to_enhanced_tier` into an actual trigger (e.g., called when a chunk's access count crosses into "hot" tier) only if the data justifies the extra storage/compute cost; otherwise remove the dead code.

### Component 6 — Validation (GPU, all 4 corpora, before/after — matching #2/#3's rigor)

0. **Fresh, tagged FAQs for all 4 corpora.** Checked directly: none of the 4 example corpora's existing `faqs.json` have `source_chunk_ids` (confirmed on UC1: 200/200 untagged, UC4: 50/50 untagged) — they pre-date Component 1 item 3. Regenerate `faqs.json` from scratch for all 4 corpora with the fixed generator before any of the steps below, rather than merging new tagged FAQs into old untagged ones via `_deduplicate` (which would leave a mix of tagged/untagged entries and silently under-cover the corpus in Component 2). This is a clean prerequisite step, not a backfill/migration of the old files.
1. **Bug-fix-only measurement.** Re-run the mode-comparison eval (`text_rag` vs `kv_meanpool` vs `kv_fulltoken`, forced modes, same harness as the existing `docs/scientific_revision_real` data) after Component 1's fixes alone, before KDS exists — quantify how much of the original gap closes for free from prompt parity and phase-gating.
2. **Per-chunk factual accuracy for KV-injection modes (new measurement, doesn't exist today).** Checked directly: `prs_evaluator`'s main loop scores F1/EM/judge only on `param_ans` (pure parametric) — `rag_ans` is only used for the diagnostic cosine ratio, never graded. The only place `kv_meanpool`/`kv_fulltoken` factual accuracy is measured today is the separate `generate_scientific_revision_results.py` harness, and only as a per-corpus aggregate, not per-chunk. Threshold calibration (step 4) needs per-chunk `kv_meanpool`/`kv_fulltoken` factual accuracy joined to that same chunk's KDS score — extend the scientific-revision-style harness to score each forced-mode answer per FAQ (it already has `question`/`ground_truth`/`prediction` per-question, so this is aggregation-and-join work, not new generation) and tag each result with its source chunk via Component 1 item 3's `source_chunk_ids`.
3. **KDS validity check.** Run 2-3 real training rounds (FAQ generation with #3's paraphrase augmentation → chat-SFT training → KV recompute), tracking per-chunk and corpus-level KDS. Confirm KDS is not spurious: chunks whose KDS rises across rounds should also show rising factual accuracy (F1/judge, from step 2) on their held-out probe — if KDS and factual accuracy diverge, the metric doesn't mean what we think it means and needs revisiting before being trusted as a gate.
4. **Threshold calibration.** Using step 2's per-chunk factual accuracy joined to per-chunk KDS, calibrate the per-chunk KDS eligibility threshold (Component 3) — e.g., find the KDS value above which `kv_meanpool`/`kv_fulltoken` factual accuracy is empirically close to `text_rag`'s, mirroring how #2 data-derived its PRS weights from a 4-corpus backtest rather than guessing a constant.
5. **Enhanced-tier resolution.** From the same data, decide Component 5's keep/wire-in/remove call.
6. Report honestly, including if some corpora end up with few or no chunks ever clearing the eligibility bar — a valid, honest outcome per this whole investigation's ethos (#3 found real generalization limits; #1 may too).

---

## Out of scope (explicit)

- Wiring KDS into pure-parametric (Phase 2/3, weights-only) eligibility — the metric is designed to generalize there, but that integration is a separate future spec.
- Chunk-cluster or topic-hierarchy-aware KDS aggregation — this design computes KDS per individual chunk; hierarchical/cluster-level rollups are not addressed.
- Any change to `prs_evaluator.py`'s existing factual-accuracy scoring (F1/judge/EM) or the #2 gate-rework's PRS weights/thresholds — those are complete and untouched by this work.
- Re-litigating #3's paraphrase-augmentation findings — this design consumes them (held-out probes, `sft_format="chat"`) but does not re-validate them.

## Error handling

No new external dependencies. KDS reuses the existing `TextEmbedding` embedder and the existing `_generate_parametric`/sampling machinery already present in `pipeline/prs_evaluator.py`. When a chunk has no associated FAQ (no topic probe available), it is excluded from KDS aggregation for that round and defaults to KV-injection-ineligible (fails closed, consistent with the existing `_query_similarity_to_known_good` self-regulating floor: no evidence means no shortcut). The same fail-closed default applies to a chunk that *has* a tagged FAQ but hasn't been reached yet by the rotating sample (Component 2's coverage strategy) — no `KDS` recorded is treated identically to a measured low score, never as an implicit pass. The per-chunk KDS threshold defaults to an empirically-calibrated value from Component 6's validation, not a guessed constant; a corpus with insufficient data to calibrate falls back to KV-injection disabled entirely (text_rag-only), matching Phase 1 behavior.
