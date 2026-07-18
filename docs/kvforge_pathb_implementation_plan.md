# KVForge Path B Improvement — Implementation Plan (v3)

## Execution Environment

| Item | Value |
|------|-------|
| Training / inference / latency measurements | AWS EC2 g5.xlarge (A10G, 24 GB VRAM) at 13.217.195.243, path `/home/ubuntu/kvforge` |
| Sync mechanism | `deploy_and_run.sh` → rsync local repo to remote + SSH execution |
| GPU contention | Treat GPU as a job queue; chain ingestion KV-compute jobs after training jobs; do not run training + KV indexing concurrently on the same GPU |
| Local Mac role | Orchestration and code edits only; no GPU measurements |

---

## Assumptions

| # | Assumption |
|---|------------|
| 1 | KVForge can modify the LoRA trainer, KV inference path, and confidence gate. |
| 2 | Budget is not a constraint; API/GPU cost is acceptable for experimentation. |
| 3 | Primary goal: shift traffic from Path A to Path B automatically, at ≥90% of Path A quality. |
| 4 | fKDS is the quality metric; KDS alone is misleading. |
| 5 | UC4 is the production use case; public benchmarks provide literature-comparable validation. |
| 6 | Path A output is a usable teacher signal; judge filter catches most Path A errors. |
| 7 | Existing dashboard, query logger, replay buffer, and routing logic are stable and extensible. |

---

## Idea Disposition (from v1)

| ID | Disposition | Role in v3 |
|----|-------------|-----------|
| E3 | Retained | Path A baseline — defines target and teacher. |
| E1 | Retained | Better LLM judge — filters teacher outputs and scores fKDS. |
| G5 | Retained | Reproducibility — seeds, logged configs, versioned adapters. |
| B1 | Retained | Higher replay ratio — training hygiene in distillation. |
| B2 | Retained | Lower LR / fewer epochs — training hygiene in distillation. |
| B5 | Retained | KL-to-base loss — optional flag-gated in distillation. |
| F1 | Retained | Quality filtering — applied to teacher answers before distillation. |
| D2 | Retained | Strict adapter acceptance — gate before deploying distilled adapters. |
| D3 | Retained | A/B testing — candidate vs deployed adapter. |
| D5 | Retained | Rollback — fast recovery. |
| D1 | Retained | Train-on-drop trigger — starts distillation rounds. |
| D6 | Retained | Dynamic routing — consumes confidence-token signal. |
| G3 | Retained | Dashboard integration — recompute, distillation, calibration metrics. |
| A1 | Dropped | Meta synthetic data kit — replaced by Path A-as-teacher on-policy data. |
| A2 | Dropped | Raw chunk causal LM — optional data mix only; revisit if forgetting persists. |
| C1/C2 | Dropped | Causal LM / masked SFT — subsumed by distillation objective. |
| E7 | Dropped | Post-hoc calibration — realized as trained confidence tokens. |
| v1 rejections | Still rejected | Same reasons as v1. |

---

## Confidence Signal: "yes" / "no" Pseudo-Tokens

- No reserved tokens or tokenizer modification.
- During distillation, append the target suffix:
  ```
  Answer: <text>
  Confidence: <yes/no>
  ```
- Labels are derived from the student's own on-policy sample scored against the teacher answer (token-F1 + judge).
- At inference, read the probability of the single token immediately after `Confidence:`, restricted to the two-token softmax {" yes", " no"}, as the routing confidence score.
- Inference is two-phase:
  1. Generate the answer to EOS (or stop if the model emits `\nConfidence:`).
  2. Force-append `\nConfidence:` and run one forward pass; read the restricted two-token softmax at the next position. Never sample the confidence token.
- Strip the confidence suffix before fKDS, judge, and token-F1 scoring.
- Verify single-token assert on " yes" and " no" (with leading space) in the Llama 3 tokenizer.
- If the base model changes, re-verify the single-token assert.

---

## Base Model and Escalation Path

| Phase | Model |
|-------|-------|
| Primary | `meta-llama/Llama-3.2-3B-Instruct` |
| Escalation trigger | Path B < 90% of Path A fKDS after 3 on-policy distillation rounds |
| Escalation candidate 1 | `meta-llama/Llama-3.1-8B-Instruct` (QLoRA on 24 GB) |
| Escalation candidate 2 | `Qwen/Qwen2.5-7B-Instruct` (cross-family alternative) |

---

## Sprint Plan

### Sprint 0: Foundation, Baselines, and Eval Hardening (1 week)

**Goal:** Reproducible baselines, leakage-proof eval, trustworthy judge.

**Tasks:**
- [ ] Verify EC2 instance reachability and `/home/ubuntu/kvforge` sync.
- [ ] Lock random seeds, log exact configs, version every adapter (already partially done in `pipeline/lora_trainer.py`).
- [ ] Improve LLM judge prompt (already partially done in `eval/metrics.py`); measure judge variance by repeat-scoring the same answers 5×.
- [ ] Build frozen held-out eval set for UC4: never-used-for-training questions + paraphrases + novel questions. Version it.
- [ ] Measure Path A baseline on UC4: (a) text-RAG, (b) current KV injection, (c) per-chunk KV vs text gap.
- [ ] Measure Path B baseline on UC4.
- [ ] Add latency (p50/p95 TTFT) and cost-per-query instrumentation.

**Deliverable:** Baseline report with text-RAG vs KV-injection vs Path B fKDS, latency, cost, and measured KV quality gap.

---

### Sprint 0.5: v1 Cleanup (2 days)

**Goal:** Remove artifacts that no longer fit v3.

**Tasks:**
- [ ] Delete `kvforge_rules.yaml`.
- [ ] Delete `core/rules_engine.py`.
- [ ] Delete `tests/test_rules_engine.py`.
- [ ] Clean `pipeline/lora_trainer.py` metadata notes to be v3-agnostic.
- [ ] Finalize `eval/metrics.py` judge prompt.

**Deliverable:** Clean workspace aligned with v3.

---

### Sprint 1: Public Benchmark Ingestion + Path A KV Recompute (1–2 weeks)

**Goal:** Close the KV-injection quality gap and validate on a literature-comparable dataset.

**Datasets:**
- **LongBench `2WikiMQA`** 200-example subset (as-is, no resampling) — literature comparison.
- **LongBench `MuSiQue`** 200-example subset — literature comparison.
- **Fixed-seed 500-question slice of original `2WikiMultihopQA` dev set** (~12.5k questions) — `recompute_ratio` sweep.

**Why two datasets:**
- LongBench 200-example is the exact benchmark CacheBlend/CacheClip report on.
- 200 examples gives ~±4% CI on token-F1, too thin for ordering five adjacent ratios; 500 examples gives ~±2–3% CI. Sweep = 5 ratios × 500 questions = 2,500 generations (overnight on A10G).
- Optional: run chosen ratio once on a larger slice for a tighter headline.

**Tasks:**
- [ ] Ingest LongBench 2WikiMQA + MuSiQue into KVForge datasource format.
- [ ] Add `recompute_ratio` config field to `DatasourceConfig`.
- [ ] In `kv_inference.py`, implement CacheBlend-style selective recomputation:
  - Load per-chunk KV tensors.
  - Compare cached KV against fresh KV on first recomputed layer for concatenated context.
  - Select high-deviation tokens.
  - Recompute selected tokens through all layers.
- [ ] Handle attention sinks: single leading sink for assembled context.
- [ ] Sweep `recompute_ratio` ∈ {0.05, 0.10, 0.15, 0.20, 0.30} on 500-question slice; plot fKDS vs TTFT.
- [ ] Validate anchors: `recompute_ratio=0` ≡ current KV injection; `recompute_ratio=1.0` ≡ text-RAG.
- [ ] Re-measure Path A baseline with chosen ratio; freeze as **teacher config**.
- [ ] Add recompute-ratio and per-query stats to dashboard.

**Deliverable:** Upgraded Path A with quality-vs-latency curve, chosen operating point, and frozen teacher config.

---

### Sprint 2: Path A-as-Teacher On-Policy Distillation (2 weeks)

**Goal:** Train Path B to match the upgraded Path A.

**Tasks:**
- [ ] Build query pool: logged real queries + paraphrase-expanded FAQs + chunk-conditioned generated questions.
- [ ] Build teacher pipeline: run every pool query through frozen Sprint-1 Path A; store answer + retrieved chunk ids + token logprobs where available.
- [ ] Quality filter: judge-score teacher answers; drop below threshold.
- [ ] Extend `lora_trainer.py` with distillation mode:
  - Off-policy SFT on (query → filtered teacher answer) pairs.
  - On-policy step (GKD-style): sample student answers, score against teacher, train on corrections.
- [ ] Add **confidence-supervision hook** to the trainer (optional extra supervised position, off by default) so Sprint 2.5 only needs a config flip.
- [ ] Training hygiene: replay ratio 0.5–1.0, LR 5e-5, 1 epoch.
- [ ] Optional flag-gated KL-to-base loss (off by default).
- [ ] Log per-source training loss (teacher-SFT, on-policy, replay).

**Deliverable:** Distillation-mode LoRA trainer + first confidence-free distilled adapter for UC4.

---

### Sprint 2.5: Confidence Pseudo-Tokens (1 week)

**Goal:** Add calibrated confidence signal at near-zero marginal cost.

**Prerequisite:** A working Sprint 2 adapter producing sensible on-policy samples.

**Tasks:**
- [ ] Generate confidence labels from Sprint 2 on-policy samples vs teacher answer.
- [ ] Train with target suffix `Answer: <text>\nConfidence: <yes/no>`; read the single-token probability of " yes" / " no".
- [ ] Force-decode the confidence suffix at inference; read the restricted two-token softmax.
- [ ] Strip suffix before all quality metrics.
- [ ] Flip confidence-supervision hook on and retrain at 1 epoch.
- [ ] Build calibration evaluation: reliability diagram + ECE.
- [ ] Rewire `confidence_gate.py` to use confidence-token probability; keep entropy+hedging heuristic behind a config flag for A/B.

**Test:** Sprint 2.5 adapter fKDS ≤ 1% relative delta vs Sprint 2 adapter.

**Deliverable:** Calibrated confidence signal integrated into the gate.

---

### Sprint 3: Gated Acceptance, A/B, Rollback (1 week)

**Goal:** Never deploy a worse adapter.

**Tasks:**
- [ ] Acceptance rule: aggregate held-out fKDS improves AND non-targeted chunks degrade ≤ 5% relative AND delta exceeds 2× judge noise.
- [ ] A/B comparison script on frozen held-out set.
- [ ] Rollback mechanism restoring previous adapter + KV tensor version + version.json.

**Deliverable:** Safe training loop.

---

### Sprint 4: Routing + End-to-End Loop (1 week)

**Goal:** Close the loop.

**Tasks:**
- [ ] Routing rule: Path B when P(" yes") ≥ threshold; else Path A with partial recompute.
- [ ] Log low-confidence queries into the distillation query pool.
- [ ] Train-on-drop trigger: start distillation when held-out fKDS or confidence-weighted quality drops.
- [ ] Run 3 loop rounds on UC4.
- [ ] Dashboard: Path B traffic share, fallback rate, confidence distribution, per-round fKDS.

**Deliverable:** Autonomous loop + UC4 results report.

---

### Sprint 4.5: Heavy Dataset Ingestion (1 week, parallel with Sprint 4)

**Goal:** Prepare full public benchmark corpora.

**Tasks:**
- [ ] Ingest WixQA.
- [ ] Ingest full HotpotQA.
- [ ] Ingest full 2WikiMultihopQA.
- [ ] Ingest TechQA if IBM agreement accepted.
- [ ] Build unified eval harness reporting token-F1 + fKDS.

**GPU queue caveat:** Chain ingestion KV-tensor jobs after training jobs; do not run concurrently.

**Deliverable:** Ingested benchmark corpora + eval harness.

---

### Sprint 5: Public Benchmark Validation + UC Replication (2 weeks)

**Goal:** Prove on published datasets and replicate on UC1/UC3.

**Tasks:**
- [ ] Run full loop on WixQA.
- [ ] Run full loop on 2WikiMultihopQA + HotpotQA.
- [ ] Evaluate partial recompute on LongBench multi-doc QA subsets for CacheBlend comparison.
- [ ] Replicate winning recipe on UC1 and UC3.

**Deliverable:** Literature-comparable benchmark report + multi-UC report.

---

## Success Criteria

| Metric | Target |
|--------|--------|
| KV-injection quality gap | ≥ 80% of injection-vs-text fKDS gap closed at ≤ 50% of text-RAG TTFT. |
| Path B vs Path A fKDS | Path B ≥ 90% of upgraded Path A on held-out eval within 3 rounds. |
| Confidence calibration | ECE lower than entropy+hedging baseline; higher answered-subset fKDS at equal coverage. |
| Path B traffic share | ≥ 50% of queries routed to Path B at pinned threshold. |
| Non-targeted chunk degradation | ≤ 5% relative fKDS drop; significance requires delta > 2× judge noise. |
| Latency/cost | Reported alongside quality in every comparison. |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Partial recompute engineering is deep | Start with `recompute_ratio=1.0` anchor and shrink; use public CacheBlend as algorithm reference. |
| Teacher (Path A) is wrong sometimes | Judge filter on teacher answers; log rejection rate; cap training weight of low-judge examples. |
| Student never reaches 90% of teacher | On-policy rounds target failures; if still < 90% after 3 rounds, escalate to Llama 3.1 8B or Qwen2.5-7B. |
| Confidence tokens miscalibrate under shift | Recalibrate labels each round from fresh student samples; keep heuristic gate as fallback. |
| Judge noise swamps acceptance deltas | Sprint 0 measures noise and sizes eval set; acceptance requires 2× noise. |
| Public benchmark corpora differ from enterprise docs | WixQA/TechQA provide enterprise middle ground; UC4 remains production case. |
| GPU contention between training and ingestion | Chain jobs via GPU queue; never run KV indexing concurrently with training. |

---

## DevTorch Governance Note

The MCP `devtorch_commit` endpoint has been timing out in this session, so none of these decisions have been persisted yet. Before implementation starts, reconnect DevTorch and log this decision batch retroactively.

---

## PR Cadence

At the end of each sprint, raise a pull request summarizing the sprint's changes, measurement results, and next-sprint dependencies. Do not start the next sprint until the PR is reviewed and merged.

