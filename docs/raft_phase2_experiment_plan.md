# Phase 2 — FT + RAG Complementarity (RAFT): Experiment Plan

**Status:** DESIGN — pre-registration draft, NOT YET RUN. (Gated on Phase 1, now CLOSED: B placement promoted +0.123/+0.106 clean · A epochs marginal +0.040 · C relational DEAD.)
**Owner:** Dr. Hemant Joshi. **Generator for any data:** in-session Opus via subagents (memory `c-gen-in-session`). **Compute:** single A10G EC2, one GPU job at a time.
**Supersedes the sketch in** `levers_bca_experiment_design.md §7` (arms kept; this fills in the pre-registration §7 deferred).
**Tested data layer:** `tools/raft_data.py` + `tests/test_raft_data.py` (§13).

---

## 1. Question & hypotheses

**Product question.** With retrieval at inference, does a *fine-tuned* generator beat plain RAG — genuine **complementarity** (parametric priors + retrieved passage), or does retrieval dominate (FT adds nothing once the passage is in context), or interfere (priors override the passage)?

**Mechanistic hypothesis (RAFT).** Training on `(question, context = oracle + distractors, answer)` with the oracle **withheld** on a fraction of examples (forcing parametric fallback) teaches the generator to (a) use the gold passage when retrieved and (b) answer from weights when retrieval misses. The gain should therefore **concentrate on retrieval-misses**. (RAFT: Zhang et al. 2024, [arXiv:2403.10131](https://arxiv.org/abs/2403.10131), Berkeley Gorilla.)

---

## 2. Arms (all eval with the SAME retriever/k — only the generator differs)

| arm | generator | retrieval at inference | isolates |
|---|---|---|---|
| **R0** (control) | base `gemma-4-E2B-it` | raw top-5 | **judged** plain-RAG (base + top-5) |
| **R1** | best Phase-1 adapter (`promo_E2B_allLinear_60kqa_rw`, all-linear r8 α16, 60k QA) | raw top-5 | does Phase-1 FT *add* on top of retrieval? |
| **R2** | RAFT adapter (this build, **same 60k QA**, RAFT objective) | raw top-5 | the complementarity recipe |
| **R3** (ablation) | RAFT adapter | **none (closed-book)** | how much RAFT also lifts pure-parametric |

Secondary/optional **R2-cot** — RAFT adapter trained+eval'd with chain-of-thought + verbatim citation (`##begin_quote##…##end_quote##`). Descriptive only (format change → confound, §6); rescored with `tools/rescore_matched_format.py`; cannot overturn the primary verdict.

**⚠ `recall@5 = 0.586` is a retrieval metric, NOT a judged score** (`recall_at_k_top5.json`). R0 is a real judged arm; never compare an arm against the recall number.

---

## 3. ⚠ Coverage GATE (resolved before anything else) — why R2 must train on the 60k, oracle-recovered

Complementarity is detected as **gain on retrieval-misses**, which requires R2's *weights* to know the miss facts. Measured (`scratchpad/raft_coverage.py`, verified 2026-10-16):

| pool | linkage | covers 481 gold | covers the 191 **misses** |
|---|---|---|---|
| `faqs_all.json` (3,858) | `source_chunk_ids` ✓ | **113/481 = 23.5%** | **29/191 = 15.2%** ← fatal |
| `faqs_full_idea1.json` (60,209, = R1's data) | **none** (`{question,answer}` only) | covers the corpus R1 learned | (no per-Q chunk id) |

- **`faqs_all` is rejected:** 15% miss-coverage would force a **false DEAD** (R2 can't show miss-gain on facts it never trained). §12 "scale only if ambiguous" would never trigger — the plan would silently misreport.
- **R2 trains on the same 60k QA as R1**, so R1 and R2 differ **only in training objective** (plain QA vs RAFT context) — the clean complementarity isolation and the clean R3-vs-R1.
- The 60k has no chunk linkage, so the oracle is **recovered** (§5.1): retrieve top-k per training question, pick the chunk whose text contains the normalized answer. This is QA-only RAFT data construction (standard when oracles aren't stored).
- **GATE v1 (recovered-oracle proxy) — RETIRED (2026-10-16, no outcome data seen).** The original gate counted whether each gold/miss `chunk_id` was a recovered oracle of ≥1 example (target ≥80% gold / ≥70% miss). Measured: **gold 0.778, miss 0.670 → FAIL**, but the proxy is invalid here: oracle recovery is verbatim answer-matching and the 60k answers are **abstractive** (only **35.3%** recover anywhere in the corpus), so the proxy measures answer-extractability, not whether the fact is *trained*. Using it would force a false FAIL.
- **GATE v2 (direct measurement) — PASS.** The question the gate really asks — *does this recipe teach the miss facts parametrically?* — is answered directly by R1's existing closed-book promotion co-judge, split by the 290/191 table (`scratchpad/r1_hitmiss_split.py`): **R1 closed-book coverage = 0.461 on misses vs 0.462 on hits vs base ~0.08.** R1 knows the miss facts at ~6× base and uniformly across hit/miss, so the miss test has ample headroom and is *not* biased toward DEAD. R2 trains on the same 60k + rewrites (§5.4), so it inherits this coverage. **This replaces a pre-registered blocker — flagged to the user.** The pre-registered remedy (targeted QA for uncovered eval chunks) is **dropped**: it would treat only test items, which R1 never got.

---

## 4. Fixed experimental surface (controls that make the four arms comparable)

| knob | value | source |
|---|---|---|
| Eval set | the **481 facts** = `eval_500.json` minus 19 `\bpage\b` items (`run_knowledge_measurement.py:61-73`; verified 500→481, 0 null `chunk_id`) | code map |
| Collection | **`bedrock-userguide-6154`** (6,154 × ~600-char chunks; ids match `eval_500.chunk_id`) — **NOT** config-default `bedrock-userguide` (2,520 word-chunks) | code map; `recall_at_k_top5.json` |
| Embedder | `BAAI/bge-small-en-v1.5` (384-d) | `bedrock_rag.py:48,51` |
| k | **top-5**, **`score_threshold=None`** (raw top-5 to every arm) | §4-gotcha below |
| Gen cap | **fixed 48** `--max-new-tokens`, identical across arms | logged, not inferred |
| Answer format | short answer, **single shared instruction**, matched across arms (primary) | §5.2, §6 |
| Judge | **blind, same-session** in-session Opus co-judge; unit = `chunk_id`; `coverage@0.5 = (CORRECT+PARTIAL)/481` | memory `cojudge-methodology` |
| Seed | 42 primary; 2nd paired seed only if a contrast is AMBIGUOUS | §8 noise SE≈0.016 |

**Retrieval hit/miss & its gotcha.** The recall table (`recall_at_k.py:62`) computes hits with `score_threshold=None`, but the live answer path (`bedrock_rag.py:55/240`) drops chunks below **0.55** before the prompt — so a recall-hit ≠ gold-was-in-prompt. **Fix (pre-registered):** run all arms with `score_threshold=None` so the raw top-5 is always in the prompt; then "hit" ≡ gold `chunk_id` ∈ top-5 ≡ exactly the `recall_at_k_top5` table. On the 481: **hit = 290, miss = 191** (0 not-in-table). Keeping 0.55 would make the split ill-defined → dropped.

---

## 5. Build steps (grounded; reused vs. new)

**No trainer change.** `build_sft_example` (`lora_trainer.py:204-228`) renders `[user: question] → [assistant: answer]`, answer-masked. **Pack the rendered context into the `question` field** → the whole `instruction+context+question` is the masked prompt and only the answer is trained = RAFT's loss. The RAFT set is a plain `--faqs` JSON; no edit to the freshly-synced trainer, no new EC2 sync.

**5.1 Oracle-recovery + distractors** (`tools/raft_data.py`, pure logic; retrieval injected so it is GPU/Qdrant-free and unit-tested).
- Retrieve top-k (k≈8) per 60k question from `bedrock-userguide-6154` (reuse `recall_at_k.py::score_inventory`; CPU pass, one-time, materialize `{question, retrieved_ids, retrieved_texts}`).
- `recover_oracle(answer, candidates)` → **two passes.** Pass 1: first chunk whose normalized text *contains* the normalized answer (verbatim substring, highest precision). Pass 2 (abstractive answers): among chunks covering ≥80% of the answer's content tokens (stopwords dropped), the **max-overlap** chunk (ties → retrieval order). Pass 2 requires **≥3 content tokens** (`min_content_toks=3`) so a couple of generic words can't bind an arbitrary chunk. Taking the *max*-overlap chunk (not the first over threshold) matters because in the full-corpus path "first" is just the lowest `chunk_id` of generic boilerplate — a defect that produced ~20% false binds on the overlap-only subset (fixed 2026-10-17, TDD). **Search the WHOLE ordered corpus** (retrieved top-k first, then the remaining 6,154), **not** just the retrieved set — otherwise a *miss* fact (gold not retrievable) silently loses its training QA and the miss test is biased toward a false DEAD (the very failure §3 guards). Guard: a purely-numeric or very short answer (e.g. "2024") matches many chunks → treated as unrecoverable.
- **Never drop** an unrecovered question: keep it as a **withheld, distractor-only** example — that is exactly the parametric-fallback signal the miss test needs. Report the recovery rate; it is not a gate.
- Distractors = retrieved chunks that are **not** the oracle (hard negatives, as in the paper); **gold is never a distractor**.
- With prob **P = 0.8** keep oracle in context; with 0.2 **withhold** (fill with an extra distractor). Exactly **5 context chunks** per example; order **shuffled**.
- **P_eff ≈ 0.56 (RESOLVED 2026-10-17 — overlap recovery chosen, user: "improve recovery first, then launch").** Verbatim-only recovery was 35.3% → P_eff ≈ 0.28 (~72% distractor-only). Adding content-token-overlap recovery (max-overlap, ≥3 content tokens) raised recovery to **0.7008** → *effective* oracle-in-context rate = recovery(0.7008) × keep(0.8) ≈ **0.56**. Coverage GATE now **PASS**: gold **0.915** (≥0.80), miss **0.843** (≥0.70); 5,611 distinct trained oracle chunks.
- **Recovery precision (honest, measured — do NOT quote an inferred "~93%").** Hand-judged sample n=50, seed=7, **including verbatim** matches, each tagged by method×source: **45/50 = 90% overall.** By source: **TOPK 35/37 = 95%**, **CORPUS-SCAN 10/13 = 77%** (the weak path). By method: **verbatim 24/24 = 100%**, **overlap 21/26 = 81%**. The 30-match overlap-only re-check after the max-overlap fix: TOPK 20/21, CORPUS-SCAN 6/9 (was 6 wrong pre-fix → 4). Residual failures are abstractive/list/degenerate answers with no exact ≥0.80 source (e.g. "playgrounds", the 14-region list), not a systematic bug; a **question-token tiebreak** would fix most and is noted for any future rebuild. A wrong oracle does **not** threaten the fact signal — the answer is trained regardless; it only lowers true P_eff and mildly inflates coverage.
- `recover_oracle` never uses top-1-retrieved as a fallback (for a miss fact top-1 is by definition not gold → would make the gate pass by construction).

**5.2 Single shared render** (fixes the advisor's format-parity break). The default `DEFAULT_RAG_INSTRUCTION` ("2-4 sentences. Cite page numbers.") is **not used** — base R0 would obey it and overrun the 48-tok cap while RAFT-R2 learns to ignore it (= the very confound §6 removes), and `chunks.json` has no `page`. Define ONE short-answer instruction, e.g. `"Answer the question in as few words as possible using the context."`, in `render_prompt(instruction, context_chunks, question)` in `tools/raft_data.py`. **Both** the training builder **and** the future `dump_rag_preds.py` import this one function, so train and eval prompts are identical except for the retrieved chunks.

**5.3 Assemble** `faqs_raft.json` (standard `--faqs`: `question` = `render_prompt(...)`, `answer` = short gold). Assert leak-dedup vs eval_500 (§7); log P, distractor count, withhold fraction, recovery rate, #examples, coverage-gate numbers.

**5.4 Train R2** (EC2, 1 GPU): mirror the promotion recipe exactly — all-linear r8 α16, from base, seed 42, `--config config_e2b_allLinear_r8.json`, `--faqs faqs_raft.json`, **AND `--rewrites rewrites_full_idea1.json`** (RESTORED — fixes advisor #1). R1 trained on 60k QA **plus** 61,554 chunk rewrites at full LM loss; the RAFT context sits inside the *masked* prompt so R2 takes no loss on chunk text, so without `--rewrites` R2 would know less parametrically than R1 and a weak miss-gain / low R3-vs-R1 would be an artifact. With rewrites restored, R1 and R2 differ **only in training objective** (plain QA vs RAFT-context QA). The synced trainer caps replay at 512 vs R1's 256, but only **0.58%** of rewrites exceed 256 tok and **0%** exceed 512 → negligible, R2 ≈ R1 on replay.

**Live-run readiness (R2 launched 2026-10-17, `faqs_raft.json` md5 `0b5929b0e36b40de0df7ea38f98f9734`):**
- **Chat-SFT answer-masked path CONFIRMED** (`🎓 Chat-SFT instruction fine-tuning: 60208 Q&A (answer-masked) + 61554 replay = 118504 examples`).
- **`SFT max_length = 1207`** (p99 of rendered seqs) — NOT the 256 fallback, so the guard passes. (R1's was 71: R2's packed 5-chunk context is the whole R1→R2 contrast.)
- **Silent drop: 2.68%** — `len(examples) = 118,504` of 121,762 (60,208 faqs + 61,554 replay); **3,258** long-context faqs dropped by `_has_supervision` (answer truncated past 1207). Below the dry-run's 5.5% estimate. **Data-match to R1:** R1 trained 121,763 (60,209 + 61,554, drop 0 — its prompts were question-only at max_length 71); R2's 118,504 differs only by the 1 eval-leak-dedup QA and the 3,258 longest packed prompts, inherent to RAFT packing. (The 3,258-drop's hit-vs-miss skew is not yet measured; the related §9 recovery-source asymmetry check IS done.)
- **ETA ≈ 25 h** on one A10G (7,407 steps × ~12.1 s/it) — a 1-day job, not multi-day; GPU 11.4 GB / actively computing (no CPU offload).
- **OOM pressure:** fragmentation OOM-warnings at 24 GB that the caching allocator recovered from; launched with **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (+ `setsid`/`PYTHONUNBUFFERED=1`/stdin from `/dev/null` to dodge the ssh channel-hang and stdout-buffering traps) to avoid a hard OOM (4-bit + grad-checkpointing + batch=1 already on).

**5.5 RAG-mode dump** (`tools/dump_rag_preds.py`, new — EC2 `rwscale_dump_preds.py` is closed-book + EC2-only). Per 481 fact: retrieve raw top-5 (`threshold=None`), render via the shared `render_prompt` (R0/R1/R2); R3 closed-book (question only); generate at cap 48; dump `{chunk_id, arm, question, prompt, pred, retrieved_ids, hit, answer_in_context, gold}`. **Absolute `--adapter` + grep `🔌 Applying LoRA`** for R1/R2/R3 (DUMP GOTCHA); sanity-check R0 count = 481 and refusal rate.

**Built + validated (2026-10-18, TDD, 8 tests; EC2-staged, ready to auto-fire via `run_phase2_dump.sh` on R2 save):**
- **Top-5 source = `recall_at_k_top5.json.retrieved_ids`** — these *are* the `threshold=None` top-5 from `bedrock-userguide-6154`, so no live Qdrant is needed at dump time and **`hit` matches the 290/191 split by construction** (verified: select_items → 481, 0 gold-chunk mismatches, 0 retrieved_ids missing from `chunks.json`, hit=290).
- **Generate path does NOT truncate** (verified by reading EC2 `prs_evaluator._generate_parametric`): `apply_chat_template` is called with no `max_length`/`truncation`, single user turn, no injected system/"answer from knowledge" text → the full prompt (question last) reaches the model, and chat wrapping matches the trainer's chat-SFT packing.
- **Eval-prompt lengths (base gemma-4 tokenizer, over 481):** max **1257**, p99 1232, median 811. **13/481 (2.7%) exceed the 1207 training `max_length`** → mild length-OOD for R2 on those 13 (note in verdict, not a blocker; R0/R1 are base/plain-QA so no training-length prior).

**5.6 Blind co-judge + analyze** (reuse Phase-1 pipeline): all four arms' 481 preds shuffled into grouped-by-fact blind shards; uniform in-session Opus; per-fact pairwise McNemar/Wilcoxon on `coverage@0.5`.

---

## 6. Answer-format control (why CoT-cite is secondary)

Phase-1's matched-format work (memory `exp5-tier1-length-control`) showed judged scores flip with length/format. RAFT's native output (CoT + citation) is a different format from the short-answer harness all Phase-1 arms used, so a CoT R2 would confound "complementarity" with "format change." **Primary:** all four arms emit short answers via the one shared instruction (§5.2), identical cap 48, identical judge → R2-vs-R0 is format-clean. **Secondary (R2-cot):** faithful CoT-cite, descriptive, length-controlled rescoring; reported separately.

---

## 7. Leakage control

`eval_500` is deduped against the 60k (memory `eval500-heldout-and-leak`) with ~20% residual substring leak. For R2's RAFT set, dedup the source questions vs `eval_500` on **Q-exact ∪ Jaccard≥0.80 ∪ A-exact** (Phase-1 criterion). Report every contrast **split clean vs leaked** — the gain must hold on the clean subset to be real, not a seen-QA artifact (as the promotion showed: clean n=406 Δ=+0.106).

---

## 8. Pre-registered decision bands (DO NOT edit after running)

Unit = fact (`chunk_id`); per-fact McNemar χ² + Wilcoxon on paired coverage. Primary contrast **R2 vs R0**, and the **interaction** (where the gain lands):

**Overall magnitude:**
- `R2 ≥ R0 + 0.040` AND χ² ≥ 3.84 → R2 beats plain RAG (necessary, not sufficient — see interaction).
- `|R2 − R0| ≤ 0.012` → **RAG-DOMINATED (DEAD)**: FT adds nothing once the passage is in context.
- `R2 ≤ R0 − 0.012` → **INTERFERENCE** (publishable negative): priors override the passage.
- otherwise → **AMBIGUOUS** → 2nd paired seed (noise SE≈0.016 > 0.012 floor).

**Interaction (pre-registered as diff-in-diff, like C):** `structure_effect = Δ_miss − Δ_hit`, where `Δ_subset = coverage(R2|subset) − coverage(R0|subset)`; significance by **per-fact paired permutation / bootstrap CI** on the interaction (NOT a point comparison).
- **COMPLEMENTARITY CONFIRMED (WORKS):** overall `R2 ≥ R0 + 0.040` (χ²≥3.84) **AND** `Δ_miss − Δ_hit > 0` with CI excluding 0 — gain concentrates on misses.
- **GROUNDED-READING GAIN (new band, fixes advisor #4):** overall `R2 ≥ R0 + 0.040` **but** the gain lands on **hits** (`Δ_miss − Δ_hit ≤ 0`). RAFT improves grounded reading, not complementarity — a real, distinct, reportable outcome (NOT AMBIGUOUS, NOT a 2nd seed).

**"R2 ≥ best-parametric + 0.04" is met by construction, NOT a test — dropped** (§7 of levers doc).

Secondary (descriptive, same judge session): **R1 vs R0** (does the Phase-1 adapter add on top of retrieval?); **R3 vs R1** (does the RAFT objective inject parametric knowledge better than plain QA — now data-matched since both train on the 60k, so this is a clean feed-back to Phase 1).

---

## 9. Confounds & handling

| confound | handling |
|---|---|
| Miss-coverage too low → false DEAD | §3 coverage GATE (≥80% gold / ≥70% miss) on recovered oracles |
| Collection mismatch (2,520 vs 6,154) | force `bedrock-userguide-6154`; verify gold ids resolve |
| hit ≠ in-prompt (0.55 threshold) | `score_threshold=None` for all arms (§4) |
| Judge-session drift (abs coverage 0.29–0.33) | all 4 arms in ONE blind session; report **within-session Δ only** |
| Answer-format (CoT vs short) | one shared instruction + format parity primary; CoT secondary + rescored (§6) |
| Seen-QA leak | dedup train vs eval_500; clean/leaked split (§7) |
| Answer truncation (packed ≈800-900 tok) | §5.4 guard: verify `len(examples)` and `SFT max_length` in log |
| Dump scores base model | absolute `--adapter` + grep `🔌 Applying LoRA` (DUMP GOTCHA) |
| **R3 closed-book is OOD for R2** | R2 always trains WITH distractor context; R3 (no context) is off-distribution — report R3 as directional, note the OOD |
| **miss-by-`chunk_id` may still have the answer in another retrieved chunk** | dump `answer_in_context` on misses; report the answer-in-context rate so "miss" gain isn't attributed wrongly |
| **Recovery-source asymmetry → Δ_miss bias** (measured 2026-10-17) | Miss facts lean on the weak CORPUS-SCAN path far more than hits: among trained-covered facts, CORPUS-SCAN-only is **21.7% for miss vs 5.4% for hit** (coverage 0.843 vs 0.962; ≥1 TOPK 78.3% vs 94.6%). At CORPUS 77% / TOPK 95% precision, the *effectively-closed-book-via-wrong-oracle* excess on misses is **~3.8pp** (≈5.0% vs 1.2%), which can inflate Δ_miss. **Small vs the ±0.040 bands but NOT ruled out** — report the hit/miss source mix next to the diff-in-diff in the verdict; if Δ_miss−Δ_hit is marginal, flag that ~4pp could be artifactual. A question-token tiebreak in `recover_oracle` would raise CORPUS precision and shrink this. |
| Replay corpus when no `--rewrites` | replay comes from the `ReplayBuffer` DB (`lora_trainer.py:900`); **log which corpus** the sample is from |
| Single seed | DEAD/WORKS readable on one seed; AMBIGUOUS → 2nd |

---

## 10. Execution checklist (one GPU job at a time)

1. [ ] CPU retrieval pass over the 60k → `{question, retrieved_ids, retrieved_texts}` (reuse `score_inventory`).
2. [ ] `tools/raft_data.py` build → `faqs_raft.json`; oracle-recovery rate logged; leak-dedup asserted vs eval_500.
3. [ ] **Coverage GATE** (§3): recovered oracles cover ≥80% of 481 / ≥70% of 191 misses — else generate targeted QA and re-check. BLOCKS step 4.
4. [ ] Train R2 on EC2 (absolute paths; confirm adapter dir; **verify SFT max_length & len(examples) in log**, §5.4).
5. [ ] `dump_rag_preds.py` R0/R1/R2 (top-5, threshold=None) + R3 (closed-book). Grep `🔌 Applying LoRA` for R1/R2/R3; R0 count=481; log refusal rate + `answer_in_context`.
6. [ ] Blind 4-arm co-judge + analyze → McNemar overall + **diff-in-diff hit/miss interaction** + clean/leaked split.
7. [ ] `results/knowledge/uc4/raft_phase2_verdict.json`; fill §8 results; update dashboard + memory. **Advisor-verify before durable.**

---

## 11. Cost / time (rough)

- 60k retrieval pass + oracle-recovery + assembly: CPU, ~1 session, no GPU.
- R2 training: promotion-run order of magnitude, but **long context** (5 chunks + Q ≈ 800–900 tok/example vs ~72 for plain QA) → ~10× tokens/step; confirm it fits A10G (fp16 + grad-checkpointing on) and budget more wall-clock; watch the SFT max_length guard.
- 4 dumps + one blind judge session: ~1 session. **Zero external API spend.**

---

## 12. Open decisions to confirm before build

1. **P (oracle-kept):** fix **0.8** primary, or `{0.6,0.8,1.0}` mini-sweep (+2 runs)? Recommend fix 0.8.
2. **Distractors:** retrieved hard-negatives (recommend) vs random.
3. **Oracle-recovery misses:** **keep as withheld distractor-only** (never drop — dropping removes miss-fact QA and biases the miss test toward a false DEAD); report the recovery rate.
4. **If the coverage GATE fails on misses:** in-session targeted QA generation for uncovered miss-chunks (adds a generation step).

---

## 13. Unit tests (`tests/test_raft_data.py` over `tools/raft_data.py`)

Pure data-layer only (no Qdrant/GPU; retrieval injected). TDD: tests written first. Cases:

1. **Prompt parity** — `render_prompt` gives byte-identical output for the same `(instruction, context, question)` whether called by the builder or the (simulated) dumper. Guards §5.2.
2. **Oracle present/withheld** — a kept example contains the gold chunk exactly once; a withheld example never contains it.
3. **No gold-as-distractor** — gold `chunk_id` never appears in the distractor set; every example has exactly **5** context chunks.
4. **Withhold fraction** — under a fixed seed over N examples, kept-oracle rate ≈ P (±tolerance).
5. **Withheld answer-leak** — withheld examples do not contain the gold *answer string* in any distractor (or the rate is computed and returned).
6. **Oracle-recovery** — pass 1 returns the first (highest-ranked) verbatim answer-containing chunk; `None` when none contains it; normalization handles case/whitespace/punctuation. **Recovers gold even when it is NOT in the retrieved prefix** (full-corpus contract). **Short/numeric answers are guarded** → `None` (→ withheld-only), so "2024" never binds an arbitrary chunk.
   - **Overlap pass 2** — paraphrased answer recovered via content-token overlap; **rejected below the 0.80 threshold**; **requires ≥3 content tokens** (2 generic tokens never bind). **Picks the MAX-overlap candidate, not the first over threshold** (ties → retrieval order) — guards the lowest-`chunk_id`-boilerplate false-bind defect fixed 2026-10-17.
7. **Leak dedup** — Q-exact, Jaccard≥0.80, and A-exact matches against eval_500 are all removed; a clearly-unrelated QA survives.
8. **481 filter** — the `\bpage\b` filter reproduces 500→481 on `eval_500.json`.
9. **Hit/miss** — computed with `threshold=None`; hit ≡ gold ∈ retrieved top-5; the 481 split is 290/191.
10. **Collection constant** — `RAFT_COLLECTION == "bedrock-userguide-6154"` (guards the silent wrong-collection bug).

Run: `python -m pytest tests/test_raft_data.py -v --override-ini="addopts="` (CLAUDE.md). `tests/` has pre-existing dirty-tree edits — report any unrelated red tests by name, don't fix silently.

---

## 14. RESULTS (run 2026-10-18 — bands in §8 NOT edited; recorded post-hoc)

**Blind same-session Opus co-judge**, 481 facts × 5 arms (R0/R1/R2 RAG top-5, R3 RAFT closed-book, R1cb promo closed-book). Grouped-by-fact, deduped within fact (2405→1967 judge items), 10 shards × fresh Opus agents. **Verified:** 0 blindness breaches (no judge touched key/meta/raw preds/other shards), per-shard output↔input fact-id/label provenance OK, q/gold identical across all 5 arms (0 mismatches). Artifacts: `results/knowledge/uc4/raft_phase2_verdict.json`, durable copy `ec2_backup/cojudge_phase2/`. Harness: `tools/cojudge_phase2.py` + `tests/test_cojudge_phase2.py` (9 tests green).

coverage@0.5 (all / hit / miss): **R0** 0.728 / 0.934 / 0.414 · **R1** 0.730 / 0.834 / 0.571 · **R2** 0.809 / 0.962 / 0.576 · R3(cb) 0.462 · R1cb(cb) 0.528.

- **PRIMARY R2 vs R0 → WORKS (COMPLEMENTARITY CONFIRMED, pre-registered §8).** Δ=+0.081 (≥0.040), McNemar χ²=22.2 (b=13/c=52), p≈0. Diff-in-diff Δ_miss−Δ_hit=+0.135, stratified-bootstrap 95% CI [0.062, 0.206], p=0.0003 → gain concentrates on misses. Clean subset (n=406) Δ=+0.071, χ²=14.8, p<1e-4 (not a leak artifact). DiD lower CI 0.062 > §9 ~0.038 recovery-artifact bound.
- **EXPLORATORY R2 vs R1 (POST-HOC, not pre-registered — same 60k, isolates the RAFT objective).** miss Δ=+0.005 (R1 0.571 ≈ R2 0.576): parametric miss-recall comes from the 60k, which plain-QA FT already achieves. Plain-QA FT INTERFERES on hits (R1 0.834 vs R0 0.934, −0.100, χ²=21.2 p≈0); RAFT REMOVES it (R2 0.962, +0.128 vs R1, χ²=33.2 p≈0, 38 restored/1 lost); DiD(R2 vs R1)=−0.122 CI[−0.196,−0.050] p=0.001; Wilcoxon p=5e-9. **Read as: RAFT's distinctive effect is undoing the grounded-reading damage plain FT causes, NOT adding parametric knowledge.** CAVEATS: (a) R1+top-5 is itself length-OOD — R1 trained question-only (max_length 71), never saw retrieved context, so its hit drop mixes "FT degrades reading" with "R1 untrained on context"; (b) not exactly data-matched — R2 dropped 3,258 long packed examples (§5.4), hit/miss skew unmeasured.
- **MECHANISM (direct discriminating test).** Among the 112 misses R0 fails: R3 (closed-book) covers **68.4%** of the 38 facts R2 RESCUES vs only **24.3%** of the 74 R2 also fails (base R3-over-all-misses 47%). R2 rescues precisely the misses the model knows in weights (~3× enrichment) → parametric fallback dominates. (Replaces an earlier weak verbatim-substring argument.)
- **SECONDARY R3 vs R1cb (closed-book).** R3 0.462 < R1cb 0.528, Δ=−0.067, p=0.005 — RAFT's raw closed-book recall slightly worse than plain-QA's (directional; R3 is OOD for R2). Format parity: median pred words R0=3, R2=5 (primary format-clean); R3=14 (verbose closed-book).
- **JUDGE VALIDITY & DRIFT.** Blind, 0 breaches, provenance+q/gold verified. Rubric RECONSTRUCTED (3-way + content-only clause), not verbatim. R1cb is byte-identical text to the Phase-1 promotion arm: scored 0.461 there vs 0.528 here = **+0.067 cross-session drift** (2× the ~0.03 on record). ⇒ Absolute coverage here is NOT comparable to Phase-1; only WITHIN-session deltas (all contrasts above) are valid.
