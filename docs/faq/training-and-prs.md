# Training & PRS

← [Back to FAQ index](../../FAQ.md)

---

### How do I tune the PRS threshold?

PRS (Parametric Readiness Score) gates phase transitions. The threshold defaults to `0.75` — the model must score ≥ 0.75 before Phase 2 activates.

#### Setting the threshold

```json
{
  "prs_threshold": 0.70
}
```

#### Interpreting PRS values

| PRS range | Interpretation | Recommended action |
|-----------|---------------|-------------------|
| < 0.50 | Model has not learned the corpus | Check chunk quality, increase epochs, use better base model |
| 0.50–0.65 | Partial learning | More FAQs for evaluation, higher `lora_rank`, more epochs |
| 0.65–0.75 | Below default threshold | Try lowering threshold to 0.70 or run another training round |
| 0.75–0.85 | Good | Default threshold met; Phase 2 reliable |
| 0.85–0.92 | Excellent | Consider activating Phase 3 |
| > 0.92 | Near-optimal | Phase 3 appropriate; monitor for overfit |

#### Separate thresholds for Phase 2 and Phase 3

Both phase transitions use the same `prs_threshold`. If you want Phase 2 to activate easily but require a higher bar for Phase 3, run the evaluator manually and check the PRS score before manually calling `ver.activate_phase_3()`:

```python
# Check PRS and only activate Phase 3 if score is high enough
prs = run_prs_evaluation(cfg, faqs)
if prs >= 0.88:
    ver.activate_phase_3()
    print(f"Phase 3 activated (PRS={prs:.4f})")
else:
    print(f"PRS={prs:.4f} — staying in Phase 2")
```

---

### My PRS is not improving across training rounds — what do I do?

Work through this checklist in order:

#### 1. Verify FAQ quality first

Auto-generated FAQs can contain hallucinations. Inspect the output:

```bash
python tools/generate_faqs.py \
  --config datasource_my-corpus.json \
  --output faqs_review.json \
  --n 20
cat faqs_review.json | python -m json.tool | head -60
```

Check that question–answer pairs are factually grounded in your corpus. Delete any that are not. Hallucinated FAQs cause `accuracy` to be measured against incorrect ground truths, making PRS look artificially low.

#### 2. Increase the number of FAQs

PRS is averaged across all FAQ pairs. With fewer than 20 FAQs, a single bad answer swings the score significantly. Generate more:

```bash
python tools/generate_faqs.py \
  --config datasource_my-corpus.json \
  --output my_faqs.json \
  --n 100
```

#### 3. Increase training epochs

```json
{ "lora_epochs": 6 }
```

#### 4. Increase LoRA capacity

```json
{
  "lora_rank":  32,
  "lora_alpha": 64
}
```

Higher rank = more trainable parameters = higher capacity, at the cost of ~2× more VRAM for the LoRA adapter and slower training.

#### 5. Check chunk size and overlap

Very short chunks (< 50 words) give the model too little context to answer from weights. Very long chunks (> 1000 words) may confuse training. Optimal range is 150–600 words:

```json
{
  "chunk_size":    400,
  "chunk_overlap": 80
}
```

You will need to re-index if you change chunk sizes (existing chunks in the vector store were built with the old parameters).

#### 6. Use a larger base model

Smaller models have less memorization capacity. If you are using TinyLlama-1.1B on a large corpus, the model may not have the capacity to achieve high PRS regardless of training duration. Try Gemma-4-E2B-it (2B) or Mistral-7B.

#### 7. Check for catastrophic forgetting

If PRS was high in round N but drops in round N+1, the model is forgetting previously learned knowledge. Increase the replay buffer diversity:

```json
{
  "lora_epochs": 3,
  "lora_rank":   16
}
```

Lower learning rates also help: `"lora_lr": 0.0001` (default is `0.0002`).

---

### How do I bring my own FAQs for PRS evaluation?

Any JSON array where each object contains a question field and an answer field works.

#### Standard format

```json
[
  {
    "question": "What is the maximum file upload size?",
    "answer":   "100 MB per file, 1 GB per day"
  },
  {
    "question": "How do I reset my API key?",
    "answer":   "Go to Settings → API Keys → Revoke and regenerate"
  }
]
```

#### Custom field names

If your dataset uses different field names (common for HuggingFace QA datasets):

```json
{
  "faq_question_key": "query",
  "faq_answer_key":   "ground_truth"
}
```

Then your FAQ file can use:

```json
[
  {"query": "What year was the treaty signed?", "ground_truth": "1847"},
  {"query": "Who was the first president?",     "ground_truth": "George Washington"}
]
```

Common HuggingFace QA dataset schemas and the config keys they require:

| Dataset | Question field | Answer field | Config keys |
|---------|---------------|-------------|-------------|
| SQuAD | `question` | `answers.text[0]` | (standard — preprocess to flat) |
| Natural Questions | `question` | `annotations.short_answers` | (preprocess) |
| TriviaQA | `question` | `answer.value` | (preprocess) |
| Custom RAGAs format | `question` | `ground_truth` | `faq_question_key: question, faq_answer_key: ground_truth` |
| Custom Q&A CSV | any | any | set both keys accordingly |

#### Running evaluation

```bash
python prs_evaluator.py \
  --config datasource_my-corpus.json \
  --faqs my_faqs.json
```

Output:

```
PRS Evaluation
  Accuracy:    0.82  (41/50 questions answered correctly)
  Calibration: 0.74  (stated confidence correlates with correctness)
  Consistency: 0.81  (answers agree across 2 independent samples)
  ─────────────────────────────────────────────────────
  PRS Score:   0.79  ✅ Above threshold (0.75) — Phase 2 eligible
```

---

### How do I change the PRS scoring weights?

The default formula weights accuracy most heavily:

```
PRS = 0.5 × accuracy + 0.3 × calibration + 0.2 × consistency
```

Adjust in your config:

```json
{
  "prs_weights": {
    "accuracy":    0.7,
    "calibration": 0.2,
    "consistency": 0.1
  }
}
```

Weights must sum to 1.0.

#### When to change weights

**Emphasize accuracy** — you care most about factual correctness, less about confidence calibration:
```json
{ "prs_weights": { "accuracy": 0.8, "calibration": 0.1, "consistency": 0.1 } }
```

**Emphasize calibration** — your use case requires the model to know what it doesn't know (e.g. medical, legal):
```json
{ "prs_weights": { "accuracy": 0.4, "calibration": 0.5, "consistency": 0.1 } }
```

**Disable consistency** — you are running evaluation quickly and want to skip the second sampling pass (which doubles evaluation time):
```json
{ "prs_weights": { "accuracy": 0.6, "calibration": 0.4, "consistency": 0.0 } }
```

---

← [Back to FAQ index](../../FAQ.md)
