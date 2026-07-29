# KVForge Path B Diversity Validation — Execution Plan

*Focused implementation of the H1/H1a research roadmap from `KVForge_Experiment_Plan.md`, with explicit cross-architecture and scale-up branches and a pre-registered trend-analysis framework.*

**Goal:** Determine whether Path B (parametric answering) can reach ≥90% of Path A (text-RAG) quality on the current Llama 3.2 3B base model, and if not, whether the bottleneck is training-signal diversity, model architecture, or parameter count.

**Hardware:** AWS g5.xlarge (A10G, 24GB). All 3–4B models fit in 4-bit QLoRA; 7–8B models fit with the same recipe.

**Time estimate:** 1–3 days depending on which branches execute.

---

## 1. Sprint Overview

| Sprint | Objective | Output | Go/No-go trigger |
|---|---|---|---|
| **S0** | Generate diverse training corpora | 1×/5×/10× diversity datasets for 3 validation corpora | Quality checks pass |
| **S1** | Diversity ceiling on Llama 3.2 3B | Per-dataset trendlines (1×→5×→10×) | Trendline bends toward or away from 90% |
| **S2** | Cross-architecture A/B at 3–4B | Llama vs Qwen vs Gemma trendlines | Architecture delta > noise floor |
| **S3** | Scale-up confirmation | 7–8B model on 10× data | 90% target reached or parameter ceiling confirmed |

S1 is mandatory. S2 and S3 are conditional on S1's trendline shape.

---

## 2. Sprint 0 — Data Generation & Quality Checks

### 2.1 Goal
Produce strict-subset diversity datasets for three validation corpora so that differences between diversity levels are due to *quantity* of synthetic data, not generator variance.

### 2.2 Datasets

| Dataset | Domain | Chunks | Eval questions | Why included |
|---|---|---|---|---|
| **UC4 Bedrock** | Technical docs | ~2,500 | 30 | Production use case; large corpus |
| **UC2 PubMedQA** | Biomedical | ~1,000 | 50 | Different vocabulary; yes/no + rationale |
| **2WikiMQA** (LongBench) | Multi-hop QA | ~1,986 | 200 | Literature benchmark; multi-hop synthesis |

### 2.3 Diversity levels

Generate at maximum dose (10×), then subsample for 5× and 1×.

| Level | Rewrites/chunk | QA pairs/chunk | Examples per chunk | Total examples (UC4) |
|---|---|---|---|---|
| 1× | 1 | 2 | 3 | ~7,500 |
| 5× | 5 | 5 | 10 | ~25,000 |
| 10× | 10 | 10 | 20 | ~50,000 |

### 2.4 Detailed tasks

#### Task S0.1 — Generate 10× paraphrases
- For each chunk, produce 10 diverse rewrites using the cloud LLM.
- Rewrite styles: entity-centric summary, tutorial voice, reference voice, FAQ voice, compressed factsheet, bullet points, conversational, formal, table-style, and a “question-embedded” restatement.
- Cache all outputs under `examples/<dataset>/diversified/<chunk_id>/rewrites.json`.

#### Task S0.2 — Generate 10× QA pairs per chunk
- For each chunk, generate 10 question-answer pairs covering distinct facts.
- Cache under `examples/<dataset>/diversified/<chunk_id>/qa_pairs.json`.

#### Task S0.3 — Build strict-subset datasets
- 1× = first rewrite + first 2 QA pairs.
- 5× = first 5 rewrites + first 5 QA pairs.
- 10× = all rewrites + all QA pairs.
- This ensures lower-dose sets are subsets of the 10× set, removing generator variance as a confounder.

#### Task S0.4 — Quality gates
Run these checks before any training starts. If any fails, fix the prompts and regenerate the offending corpus.

| Check | Command | Pass criterion |
|---|---|---|
| Rewrite coverage | `tools/validate_diversity_data.py` | ≥90% of rewrites preserve the chunk's factual content (human spot-check on 20 samples) |
| QA-answer containment | `tools/validate_diversity_data.py` | ≥90% of generated answers are grounded in the source chunk text |
| Duplicate detection | `tools/validate_diversity_data.py` | ≤5% near-duplicate QA pairs within a corpus |
| Format compliance | `tools/validate_diversity_data.py` | All outputs parse into the trainer's manifest schema |

### 2.5 Cost
- ~132,000 API calls total.
- With 50 concurrent calls: ~2–3 hours.
- Estimated cost: $10–20 using Gemini 2.5 Flash or GPT-4o-mini.

---

## 3. Sprint 1 — Diversity Ceiling on Llama 3.2 3B

### 3.1 Goal
Measure the diversity dose-response curve on the current base model and determine whether the bottleneck is training signal or model capacity.

### 3.2 Detailed tasks

#### Task S1.1 — Train 1× baseline (3 datasets, 1 seed each)
- Use the existing `pipeline/lora_trainer.py` with the 1× dataset as chat-format SFT.
- Hyperparameters: r=16, lora_alpha=32, lr=5e-5, 1 epoch, batch_size=1, grad_accum=16, max_seq_length=256.
- Record training metadata with `save_training_metadata`.

#### Task S1.2 — Train 5× and 10× adapters (3 datasets, 1 seed each)
- Same hyperparameters as 1×.
- 9 additional training runs total.

#### Task S1.3 — Evaluate Path A baselines
- For each dataset, run `tools/measure_baseline_fkds.py` in `text_rag` mode only to get the Path A denominator.
- This needs the existing KV tensors / text retrieval only.

#### Task S1.4 — Evaluate Path B for each adapter
- Run `tools/measure_baseline_fkds.py` in `parametric` mode for each of the 9 trained adapters.
- No KV tensors needed; this is the cheapest evaluation mode.

#### Task S1.5 — Compute trend analysis and visualization
- Use `tools/analyze_diversity_trends.py` to produce:
  - Per-dataset fKDS vs diversity level line plots.
  - Path B as % of Path A vs diversity level.
  - Gap to 90% target vs diversity level.
  - CSV summary table.
- Commit all figures and CSV to `results/pathb_diversity/figures/` and `results/pathb_diversity/summary.csv`.

### 3.3 Checks & go/no-go

| Check | Pass criterion | Consequence if failed |
|---|---|---|
| Training completes | All 9 adapters finish without OOM | Lower batch size / increase grad_accum and retry |
| Evaluation completes | All 9 adapters evaluated on held-out set | Debug inference path and re-run |
| Trend curve computable | fKDS data exists for 1×, 5×, 10× on all 3 datasets | Cannot make a decision; fix data pipeline |

### 3.4 Decision rules

Based on the trendline shape, not fixed thresholds (because sample noise is ±0.02–0.05 fKDS).

| Trendline shape | Interpretation | Next sprint |
|---|---|---|
| Rising through 10× and approaching 90% | Model has capacity; data diversity is the lever | Continue to 20×/40× (extension of S1) or proceed to S2 for marginal architecture check |
| Rising through 10× but clearly asymptoting below 85% | Data diversity helps, but 3B capacity is the ceiling | Skip S2; go to S3 (scale up) |
| Flat from 5× to 10× (any level) | Model is saturated; parameter count is the bottleneck | Skip S2; go to S3 |
| Divergent across datasets (some rise, some flat) | Approach is domain-dependent; need per-dataset analysis | Expand to 2WikiMQA + HotpotQA + TechQA; do not generalize prematurely |

---

## 4. Sprint 2 — Cross-Architecture A/B at 3–4B (Conditional)

### 4.1 Goal
Execute only if S1 shows the model is near but not at 90% (e.g., trendline rising and sitting at 85–89% at 10×). If S1 shows a clear capacity ceiling or already reaches 90%, skip this sprint.

### 4.2 Candidate models

| Model | Size | VRAM (4-bit) | Rationale |
|---|---|---|---|
| `meta-llama/Llama-3.2-3B-Instruct` | 3.2B | ~2GB | Control from S1 |
| `Qwen/Qwen2.5-3B-Instruct` | 3.1B | ~2GB | Same family as v3 escalation path; no known plasticity issues |
| `google/gemma-3-4b-it` | 4.0B | ~2.5GB | Freshest 4B model; tests whether newer pretraining recipe matters |

### 4.3 Detailed tasks

#### Task S2.1 — Add candidate models to datasource configs
- Create `model_library` overrides in each datasource config so `core/model_loader.py` can detect KV shape and LoRA targets automatically.
- Or use the existing `llm_model` field if the model is already in `core/model_registry.json` (Llama is; Qwen and Gemma may need registry entries).

#### Task S2.2 — Train 1× and 10× on Qwen and Gemma
- Reuse the strict-subset datasets from S0.
- 6 additional training runs: 2 models × 2 diversity levels × 3 datasets = 12 runs? No — we run the full S1 sweep on each model to compare trendlines.
- Actually: 2 new models × 3 datasets × 3 diversity levels = 18 training runs. That's a lot.
- **Optimization:** If S1 already shows the 5× level is in the flat region, only train 1× and 10× on the new models. That reduces to 2 models × 3 datasets × 2 levels = 12 runs. If S1 shows 5× is informative, include it.

#### Task S2.3 — Evaluate Path B for each architecture
- Same as S1.4, but now with model family as an additional grouping variable.

#### Task S2.4 — Cross-architecture trend analysis
- Update `tools/analyze_diversity_trends.py` to overlay model-family trendlines on the same plots.
- Statistical check: report whether the difference between the best 10× adapter and the Llama 10× adapter exceeds the 95% noise floor (~0.05 fKDS) on any dataset.

### 4.4 Checks & go/no-go

| Check | Pass criterion | Consequence |
|---|---|---|
| Model downloads succeed | Qwen and Gemma load on A10G without OOM | Use 4-bit quantization if needed |
| LoRA targets detected | `model_loader.detect_lora_targets()` finds valid modules | Add registry entries with correct target names |
| Cross-model eval complete | fKDS for 1× and 10× on all models and datasets | Re-run failed inferences |

### 4.5 Decision rules

| Outcome | Interpretation | Next sprint |
|---|---|---|
| Best 3–4B model ≥ 90% of Path A | 3–4B architecture is sufficient; pick the winner | Stop, document, implement winner |
| Best 3–4B model 85–89% and still rising | Architecture gives marginal gain, but parameter ceiling likely | Go to S3 (7–8B) to confirm |
| All 3–4B models ≤ 85% | Parameter count is the bottleneck | Go to S3 |

---

## 5. Sprint 3 — Scale-Up Confirmation (Conditional)

### 5.1 Goal
Run the winning diversity recipe on a 7–8B parameter model to confirm whether parameter count was the true bottleneck.

### 5.2 Candidate models

| Model | Size | VRAM (4-bit) | Rationale |
|---|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | ~5GB | Same family as current base; v3 escalation candidate 1 |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ~5GB | Cross-family alternative; v3 escalation candidate 2 |

### 5.3 Detailed tasks

#### Task S3.1 — Select the winning diversity recipe
- Use the 10× dataset from S1 (or S2 if a different model family's 10× data is better). If the same training data works across models, reuse it to isolate model capacity.

#### Task S3.2 — Train 7–8B models on 10× data (3 datasets)
- 2 models × 3 datasets = 6 training runs.
- Use QLoRA with 4-bit quantization. The training time per run is roughly 2× longer than 3B because of larger model and more parameters, but still fits on A10G.

#### Task S3.3 — Evaluate Path B on each scale-up adapter
- Same as S1.4.

#### Task S3.4 — Compare 3B vs 8B trendlines
- Overlay 3B and 8B results on the same plots.
- Compute the absolute fKDS gain from 3B → 8B at 10× diversity.

### 5.4 Checks & go/no-go

| Check | Pass criterion | Consequence |
|---|---|---|
| 8B fits on A10G | Training completes without OOM | Increase grad_accum or reduce max_seq_length |
| 8B evaluation completes | Path B fKDS measured for all 6 adapters | Re-run if needed |
| Gap is measurable | 8B fKDS differs from 3B by more than noise floor | If within noise, more eval data or seeds needed |

### 5.5 Decision rules

| Outcome | Interpretation | Final recommendation |
|---|---|---|
| 8B ≥ 90% of Path A on ≥2 datasets | Parameter count is the bottleneck; use 8B | Switch production to 8B model; regenerate KV tensors |
| 8B < 90% of Path A | Training-signal quality is still the bottleneck, but architecture/scale changes are insufficient | Re-examine data quality (entity graph, judge filtering, etc.) or reconsider the 90% target |
| 8B beats 3B but not enough | Partial confirmation; need more data or a better gate | Run H2/H4 from the main experiment plan before deciding |

---

## 6. Trend Analysis Methodology

### 6.1 Metrics tracked

For each `(dataset, model, diversity_level)` tuple:

| Metric | Computation | Purpose |
|---|---|---|
| `path_a_fkds` | `measure_baseline_fkds.py --modes text_rag` | Denominator (target ceiling) |
| `path_b_fkds` | `measure_baseline_fkds.py --modes parametric` | Numerator (parametric quality) |
| `relative_quality` | `path_b_fkds / path_a_fkds` | Primary coverage metric |
| `gap_to_90` | `0.90 - relative_quality` | Signed distance to target |
| `token_f1` | From eval harness | Secondary quality signal |
| `judge_accuracy` | From eval harness | Factual correctness signal |
| `latency_p50` | From eval harness | Cost/speed signal |

### 6.2 Statistical treatment

- **Bootstrap confidence intervals:** For each condition, run 10,000 paired bootstrap resamples over eval questions to get a 95% CI for `relative_quality`.
- **Decision noise floor:** With 30–200 eval questions, expect ±0.02–0.05 fKDS. Treat deltas smaller than 0.05 as noise.
- **Sample-size warning:** If a CI is wider than 0.10 on the gap-to-90 metric, the dataset is underpowered and the trend is exploratory, not decisive.
- **Multiple seeds:** If time allows, run 3 seeds (42, 43, 44) on the final 10× condition to separate training noise from model noise. The plan assumes 1 seed per condition for speed, but reports std across seeds if available.

### 6.3 Trend shape classification

After running the analysis script, classify each trendline into one of four shapes:

| Shape | Operational definition | Interpretation |
|---|---|---|
| **Rising** | fKDS(10×) > fKDS(5×) > fKDS(1×) by >0.02 each | Model has capacity; more data helps |
| **Plateau early** | fKDS(5×) ≈ fKDS(10×) within 0.02 | Saturation reached at 5× or earlier |
| **Noisy flat** | No consistent ordering | Measurement underpowered or model unresponsive |
| **Divergent** | Different datasets have different shapes | Approach is not generalizable; boundary conditions exist |

---

## 7. Visualization Plan

All plots are generated by `tools/analyze_diversity_trends.py` and saved to `results/pathb_diversity/figures/`.

### 7.1 Required plots

| Plot | X-axis | Y-axis | Lines | Filename |
|---|---|---|---|---|
| Per-dataset fKDS trend | Diversity level | Mean fKDS | One line per dataset | `fkds_by_diversity.png` |
| Path B coverage trend | Diversity level | Path B / Path A | One line per dataset | `coverage_by_diversity.png` |
| Gap to 90% | Diversity level | `0.90 - coverage` | One line per dataset + horizontal 90% line | `gap_to_90.png` |
| Cross-model comparison | Diversity level | Coverage | One line per model family | `coverage_by_model.png` |
| Cross-model fKDS | Diversity level | fKDS | One line per model family | `fkds_by_model.png` |
| Dataset summary table | — | — | Heatmap of coverage | `coverage_heatmap.png` |
| Confidence intervals | Diversity level | Coverage with 95% CI | Error bars per dataset | `coverage_with_ci.png` |

### 7.2 Plot conventions

- Use a shared color map across datasets and models.
- Mark the 90% target as a horizontal dashed line on coverage plots.
- Mark the 0 gap line on gap-to-90 plots.
- Include sample size (n) in each plot title or caption.
- Save both PNG and SVG.

---

## 8. Go / No-Go Decision Tree

```
Start S1
│
├─ S1 trendline rising through 10× and ≥90% on ≥2 datasets
│   └─ STOP: 3B Llama is sufficient. Proceed to productionize S1 recipe.
│
├─ S1 trendline rising but asymptoting 85–89% on ≥2 datasets
│   └─ S2: test Qwen2.5-3B and Gemma 3 4B
│       ├─ Best 3–4B ≥90% → STOP: use that model
│       ├─ Best 3–4B 85–89% → S3: scale to 7–8B
│       └─ Best 3–4B <85% → S3: scale to 7–8B
│
├─ S1 trendline flat or asymptoting <85% on ≥2 datasets
│   └─ SKIP S2 → S3: scale to 7–8B
│       ├─ 8B ≥90% → STOP: use 8B model
│       └─ 8B <90% → STOP: need H2/H4 from main experiment plan (entity graph, better gate) or reconsider target
│
└─ S1 results divergent across datasets
    └─ Expand to more datasets (HotpotQA, MuSiQue, TechQA) before any model decision
```

---

## 9. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| S1 flatlines below 85% | Medium | S3 is pre-planned; no time lost |
| API generation quality is poor | Medium | S0 quality gates; regenerate if spot-check fails |
| 8B does not fit on A10G | Low | 4-bit QLoRA with grad_accum=16; fallback to 7B Qwen |
| A10G occupied by Sprint 0 baseline | High | Coordinate with running tmux session; run experiments in queue |
| Different tokenizer breaks confidence-token format | Low | Re-verify `yes`/`no` are single tokens per model family |
| Gemma 3 4B model ID changes | Low | Verify exact HF Hub ID before S2 |

---

## 10. Appendix A — Example Commands

### Generate diversity data
```bash
python -m tools.generate_diversity_data \
  --config examples/usecase4_bedrock_userguide/config.json \
  --output examples/usecase4_bedrock_userguide/diversified_v1 \
  --variations 10 --qa-pairs 10
```

### Train 10× adapter on Llama 3.2 3B
```bash
python -m pipeline.lora_trainer \
  --config examples/usecase4_bedrock_userguide/config.json \
  --faqs examples/usecase4_bedrock_userguide/diversified_v1/10x_qa.json
```

### Evaluate Path B
```bash
python -m tools.measure_baseline_fkds \
  --config examples/usecase4_bedrock_userguide/config.json \
  --eval-set examples/usecase4_bedrock_userguide/eval_heldout_v1.json \
  --output-dir results/pathb_diversity/uc4_bedrock/llama32_3b_10x \
  --modes parametric \
  --judge-model gpt-4o-mini
```

### Analyze trends
```bash
python -m tools.analyze_diversity_trends \
  --results-dir results/pathb_diversity \
  --output-dir results/pathb_diversity/figures
```

---

## 11. Appendix B — Result Registry Schema

Each experimental run produces a JSON file following this schema:

```json
{
  "experiment_id": "uc4_bedrock_llama32_3b_10x_seed42",
  "dataset": "uc4_bedrock",
  "model_family": "llama3",
  "model_id": "meta-llama/Llama-3.2-3B-Instruct",
  "diversity_level": 10,
  "seed": 42,
  "path_a_fkds": { "mean": 0.45, "sem": 0.02, "ci_lower": 0.41, "ci_upper": 0.49 },
  "path_b_fkds": { "mean": 0.40, "sem": 0.03, "ci_lower": 0.34, "ci_upper": 0.46 },
  "relative_quality": { "mean": 0.89, "sem": 0.04, "ci_lower": 0.81, "ci_upper": 0.97 },
  "gap_to_90": { "mean": 0.01, "sem": 0.04, "ci_lower": -0.07, "ci_upper": 0.09 },
  "token_f1": { "mean": 0.35, "sem": 0.02 },
  "judge_accuracy": { "mean": 0.42, "sem": 0.03 },
  "latency_p50": 0.8,
  "records": []
}
```

The trend-analysis script reads this schema and produces the visualizations in §7.

---

*Pre-registered decision criteria. Deviations will be logged in a `DEVIATIONS.md` file, not silently edited.*
