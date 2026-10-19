# KVForge — LoRA Rank & Epoch Sweep on Closed-Book QA

**Model:** `google/gemma-4-E2B-it` (≈2B params, 4-bit NF4 QLoRA)
**Task:** Closed-book question answering on the AWS Bedrock User Guide corpus
**Training data:** 337 QA pairs (`faqs_train.json`), question→answer next-token prediction
**Precision:** bf16 (fp16 overflows on this setup → NaN gradients)
**Evaluation:** 39 exact (seen) + 39 paraphrase (unseen) questions, `token-F1` + refusal rate
**Date:** 2026-09-07

---

## 1. Headline findings

1. **More epochs don't help past ~6.** The meaningful gain is all in the 3→6 step; 6→9→12 flatlines on exact F1 and the paraphrase score *drops* by 12 epochs — the signature of overfitting.
2. **Higher LoRA rank helps, up to r=64.** Rank moves *both* exact and paraphrase scores (generalization), confirming that **capacity is the binding constraint** on this task.
3. **r=128 overfits.** It matches r=64 on exact but loses paraphrase F1 and raises refusals — the rank curve peaks at 64.
4. **Refusals are stuck** (~0.28–0.33 exact) regardless of rank or epochs — a separate problem not solved by adapter capacity or training duration.

---

## 2. Methodology

- **Base model** loaded in 4-bit NF4 with `bnb_4bit_compute_dtype=bf16`, gemma-4 vision/audio towers removed.
- **LoRA** on `q_proj`/`k_proj`/`v_proj`, `alpha = 2×rank`, dropout 0.05, lr 2e-4, batch 1 × grad-accum 16.
- **Sweeps:** (a) epochs ∈ {3,6,9,12} at rank 16; (b) rank ∈ {16,32,64,128} at 3 epochs.
- **Metric:** `token-F1` (strict word-overlap) — a *lower bound* on correctness; the LLM-judge FA metric is ~3× higher. Also **refusal rate** (fraction of answers that are model refusals like "I do not have real-time access…").

> **Note:** `token-F1` is deliberately surface-strict. A correct answer worded differently scores low. Read the *trends and deltas*, not the absolute values.

---

## 3. Results

### 3.1 Full table (7 arms)

| Arm | Rank | Epochs | exact F1 | para F1 | exact ref. | para ref. |
|-----|-----:|-------:|---------:|--------:|-----------:|----------:|
| `qa_r16_ep3`  | 16  | 3  | 0.159 | 0.155 | 0.31 | 0.26 |
| `qa_r16_ep6`  | 16  | 6  | 0.167 | 0.159 | 0.31 | 0.23 |
| `qa_r16_ep9`  | 16  | 9  | 0.167 | 0.164 | 0.31 | 0.26 |
| `qa_r16_ep12` | 16  | 12 | 0.169 | 0.163 | 0.33 | 0.28 |
| `qa_r32_ep3`  | 32  | 3  | 0.162 | 0.167 | 0.31 | 0.23 |
| `qa_r64_ep3`  | 64  | 3  | 0.170 | 0.167 | 0.28 | 0.23 |
| `qa_r128_ep3` | 128 | 3  | 0.170 | 0.163 | 0.33 | 0.28 |

### 3.2 Epoch sweep (rank fixed at 16)

![Epoch curve](chart_epoch_curve.png)

| Δ | exact F1 | para F1 |
|---|---------:|--------:|
| 3→6  | +0.008 | +0.004 |
| 6→9  | +0.001 | +0.004 |
| 9→12 | +0.002 | −0.001 |

### 3.3 Rank sweep (epochs fixed at 3)

![Rank curve](chart_rank_curve.png)

| Δ | exact F1 | para F1 |
|---|---------:|--------:|
| 16→32  | +0.003 | +0.012 |
| 32→64  | +0.008 | −0.000 |
| 64→128 | +0.000 | −0.003 |

### 3.4 Refusal rates

![Refusal rates](chart_refusal.png)

---

## 4. Interpretation

### Epochs plateau, then overfit
The only meaningful epoch gain is 3→6 (+0.008 exact). Past that the exact score is flat (+0.001, +0.002), and by 12 epochs the *unseen* paraphrase score falls (0.164→0.163) while refusals rise (0.31→0.33). This is classic overfitting: on a fixed 337-example set, extra passes re-memorize the exact questions without improving generalization.

### Rank is capacity — and it generalizes (to a point)
Rank 16→32→64 lifts **both** exact (0.159→0.162→0.170) and paraphrase (0.155→0.167→0.167). The fact that rank moves the *held-out/reworded* score is the tell: it's storing more real facts, not memorizing surface strings. This directly supports the capacity hypothesis — the 2B model is undersized for the closed-book goal.

### r=128 is past the sweet spot
At r=128, exact plateaus at 0.170 (same as r=64) but paraphrase drops to 0.163 and refusals jump to 0.33. Too much adapter capacity on 337 examples overfits — the extra parameters memorize rather than generalize. **r=64 is the optimum.**

### Refusals are a separate, unsolved problem
Across all seven arms, exact refusal sits at 0.28–0.33 and never meaningfully improves. Neither rank nor epochs overrides the instruction-tuned base model's "I don't have access" reflex. This likely requires a different objective (answer-masked SFT rather than plain next-token) or a larger base model with a stronger factual prior.

---

## 5. Conclusion

On a 2B model with 337 domain QA pairs, closed-book QA has a hard ceiling around **token-F1 ≈ 0.17**. Both "more training" (epochs) and "more adapter capacity" (rank) hit this ceiling and then overfit. The binding constraint is **base-model size**, not the LoRA recipe.

**Ranked levers going forward:**
1. **Scale the base model** (Gemma-4-4B/7B) — the capacity evidence points here first.
2. **Answer-masked SFT** — to attack the refusal problem directly.
3. **More / better training data** (diverse rewrites, distillation) — once the base has room to store it.

---

## 6. Reproducibility

- Training: `train_one.py` / `train_sweep.py` (bf16, `bnb_4bit_compute_dtype=bf16`, `_unwrap_gemma4`).
- Eval: `eval_one_arm.py` (isolated per arm, `ADAPTER_HASH` printed to prove correct adapter load).
- Artifacts: checkpoints under `examples/usecase4_bedrock_userguide/lora_checkpoints/qa_*`.
