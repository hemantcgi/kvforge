# ModelScout Design Spec

**Date:** 2026-04-22
**Status:** Approved for implementation

---

## Problem

KVForge currently hard-codes `meta-llama/Llama-3.2-3B-Instruct` as the LLM for every use case. Different domains, languages, corpus sizes, and hardware profiles benefit from different base models. Choosing the wrong model means slower phase advancement, lower PRS ceilings, and unnecessary VRAM pressure. There is no automated way to evaluate which open model performs best for a given UC before or after indexing.

---

## Core Idea

ModelScout is an **interactive AI agent** that autonomously experiments across candidate models, adjusts its own experimental parameters based on observed results, and keeps the user informed and in control throughout. It draws directly from Andrej Karpathy's autoresearch pattern: fixed evaluation harness + agent-controlled variables + result log + program instructions.

Unlike autoresearch (which runs unattended overnight), ModelScout is **conversational** — it announces intent, reports results, pauses at decision points, and accepts mid-session redirects from the user.

---

## Two Deployment Modes

### Pre-index Mode
Run before any documents are indexed. ModelScout samples raw documents, generates a small FAQ set, runs a mini-LoRA round per candidate, evaluates PRS, and recommends the best model before the user commits to a full index.

### Post-index Mode
Run against an existing VDB. ModelScout uses the already-indexed chunks (preferring hot-tier) as the corpus, uses the existing FAQ file or generates one from the hot-tier, runs mini-LoRA rounds per candidate against the real replay buffer, and evaluates full three-signal PRS. Useful when a UC is underperforming or when the user wants to explore switching models.

Both modes share the same agent loop, evaluation harness, and results format. Only corpus and FAQ sourcing differs.

---

## Architecture

### Components

```
pipeline/model_scout.py         — Core agent: GPU detection, experiment loop,
                                  parameter adjustment, result logging, I/O adapter interface
core/model_registry.py          — Registry loader, VRAM filter, candidate scorer
core/model_registry.json        — Curated model list (ships with KVForge, user-extensible)
model_scout_program.md          — Agent instructions (operator-editable, analogous to
                                  autoresearch's program.md)
pipeline/model_scout_cli.py     — Interactive CLI adapter (thin wrapper over model_scout.py)
studio/routes.py (extend)       — SSE endpoint streaming ModelScout events to Studio UI
tests/test_model_scout.py
tests/test_model_registry.py
```

### I/O Adapter Pattern

`model_scout.py` communicates through an `IOAdapter` protocol with two implementations:

```
IOAdapter.send(message: str)         → display progress/results to user
IOAdapter.ask(question: str, options: list[str] | None) → str   → get user input
IOAdapter.stream_progress(label: str, pct: float)               → update progress bar
```

- `CLIAdapter` — prints to stdout, reads from stdin with `input()`
- `SSEAdapter` — pushes Server-Sent Events to the Studio browser; user responses come via a companion POST endpoint

This means the entire agent loop is written once in `model_scout.py` and tested without a browser or GPU.

---

## Startup Sequence

```
1. Auto-detect GPU
   torch.cuda.get_device_name(0) + torch.cuda.mem_get_info(0)
   → report: GPU model, total VRAM, free VRAM, CUDA version

2. Hard-filter model registry
   Eliminate any model where:
     (vram_fp16_gb + vram_lora_overhead_gb) > free_vram  (fp16 path)
     AND
     (vram_4bit_gb + vram_lora_overhead_gb) > free_vram  (4-bit path)
   Report eligible models to user.

3. Ask mode
   "Are you running ModelScout before indexing (pre-index) or against
    an existing VDB (post-index)?"

4. Ask budget
   Present four options:
   A) Total wall-clock time — e.g. "run for up to 4 hours"
   B) Total experiment count — e.g. "run at most 15 experiments"
   C) Per-experiment step cap + total count — e.g. "max 1000 steps/experiment, 12 experiments"
   D) Agent decides — agent stops when it is confident in a recommendation
   User selects one; agent confirms parameters.

5. Corpus setup
   Pre-index: ask for document path(s); sample up to initial_corpus_chunks (default 200)
   Post-index: scroll VDB hot-tier chunks; use up to initial_corpus_chunks

6. FAQ setup
   Generate initial_faq_count (default 20) FAQs via sleep_faq_generator
   Report: "Generated 20 FAQs covering 7 topic clusters. Ready to begin."

7. Confirm and start experiment loop
```

---

## Experiment Loop

Each iteration:

```
LOOP:
  1. Agent reasons: which model + parameters to try next
     (based on results so far, parameter adjustment rules, remaining budget)

  2. Announce intent
     "Next: Mistral-7B-Instruct (4-bit), 500 steps, rank=16.
      Reason: highest domain-affinity score among untried models.
      Estimated time: ~8 min. Proceeding in 10s — type 'skip' to skip or
      'stop' to end session."

  3. Run experiment (with live progress updates)
     a. Load model (with quantization if needed)
     b. Run mini-LoRA training (fixed step budget for this experiment)
     c. Evaluate PRS on FAQ set
     d. For post-index mode: also compute vdb_coverage sample + realtime_coverage

  4. Report result
     "Mistral-7B: PRS=0.68, VRAM=19.1GB, 493s.
      PRS curve still rising at step 500 (loss: 1.42→0.89).
      Recommendation: re-run with 1000 steps OR accept and move on.
      [A] Re-run with 1000 steps  [B] Move to next model  [C] Stop session"

  5. Log result to model_scout_results.tsv

  6. Apply parameter adjustment rules (see below)

  7. Check budget; if exhausted → emit final recommendation and exit

  CONTINUE
```

### Parameter Adjustment Rules

Encoded in `model_scout_program.md`. Agent applies these to decide parameter changes before announcing the next experiment:

| Observation | Agent action |
|---|---|
| `prs < 0.55 AND training_loss still falling` | Increase `lora_steps` × 2, retry same model |
| `prs variance across FAQs > 0.15` | Increase `faq_count` × 1.5, regenerate |
| `domain_complexity_score > 0.70` | Increase `corpus_chunks` to ≥ 400 |
| `OOM at fp16` | Retry same model with `quantization=4bit` |
| `OOM at 4bit` | Skip model, note VRAM ceiling in results |
| `prs improvement > 0.10 vs best so far` | Try smaller/larger variant of same model family next |
| `3 consecutive models prs < 0.55` | Pause, ask user: "Results are lower than expected — is the domain language or format different from what I've been told?" |
| `prs plateau (< 0.01 improvement over 2 rounds)` | Stop experimenting with this model, move on |
| `lora_steps > 2000 AND prs < 0.60` | Abandon model family, note in results |

All parameter changes are announced to the user before execution, with a short override window.

### User Interrupts (accepted at any time)

- `"skip"` — skip current model, move to next
- `"stop"` — end session, emit final recommendation from results so far
- `"try <model_name>"` — jump to a specific model next
- `"more steps"` — double current `lora_steps` for the next run
- `"more faqs"` — increase `faq_count` by 10 and regenerate
- `"change budget"` — re-open budget dialog

---

## Model Registry

### core/model_registry.json

```json
{
  "meta-llama/Llama-3.2-3B-Instruct": {
    "vram_fp16_gb": 7,
    "vram_4bit_gb": 2,
    "vram_lora_overhead_gb": 4,
    "languages": ["en"],
    "strengths": ["factual_qa", "instruction_following"],
    "weaknesses": ["multilingual", "long_reasoning"],
    "lora_targets": ["q_proj", "k_proj", "v_proj"],
    "min_corpus_chunks": 50,
    "license": "commercial_ok",
    "family": "llama3"
  },
  "meta-llama/Llama-3.1-8B-Instruct": {
    "vram_fp16_gb": 17,
    "vram_4bit_gb": 5,
    "vram_lora_overhead_gb": 6,
    "languages": ["en"],
    "strengths": ["factual_qa", "reasoning", "long_context"],
    "weaknesses": ["multilingual"],
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "min_corpus_chunks": 100,
    "license": "commercial_ok",
    "family": "llama3"
  },
  "mistralai/Mistral-7B-Instruct-v0.3": {
    "vram_fp16_gb": 15,
    "vram_4bit_gb": 5,
    "vram_lora_overhead_gb": 6,
    "languages": ["en", "fr", "de", "es", "it"],
    "strengths": ["factual_qa", "european_languages", "long_context"],
    "weaknesses": ["cjk_languages", "code"],
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "min_corpus_chunks": 100,
    "license": "commercial_ok",
    "family": "mistral"
  },
  "Qwen/Qwen2.5-7B-Instruct": {
    "vram_fp16_gb": 16,
    "vram_4bit_gb": 5,
    "vram_lora_overhead_gb": 6,
    "languages": ["en", "zh", "ja", "ko", "multilingual"],
    "strengths": ["multilingual", "reasoning", "technical_docs", "code"],
    "weaknesses": [],
    "lora_targets": ["q_proj", "k_proj", "v_proj"],
    "min_corpus_chunks": 100,
    "license": "commercial_ok",
    "family": "qwen2"
  },
  "Qwen/Qwen2.5-1.5B-Instruct": {
    "vram_fp16_gb": 4,
    "vram_4bit_gb": 1.5,
    "vram_lora_overhead_gb": 3,
    "languages": ["en", "zh", "multilingual"],
    "strengths": ["low_vram", "fast_inference", "multilingual"],
    "weaknesses": ["complex_reasoning", "long_context"],
    "lora_targets": ["q_proj", "k_proj", "v_proj"],
    "min_corpus_chunks": 30,
    "license": "commercial_ok",
    "family": "qwen2"
  },
  "microsoft/Phi-3-mini-4k-instruct": {
    "vram_fp16_gb": 8,
    "vram_4bit_gb": 3,
    "vram_lora_overhead_gb": 4,
    "languages": ["en"],
    "strengths": ["low_vram", "factual_qa", "efficiency"],
    "weaknesses": ["multilingual", "very_long_context"],
    "lora_targets": ["qkv_proj", "o_proj"],
    "min_corpus_chunks": 50,
    "license": "commercial_ok",
    "family": "phi3"
  },
  "google/gemma-2-9b-it": {
    "vram_fp16_gb": 19,
    "vram_4bit_gb": 6,
    "vram_lora_overhead_gb": 7,
    "languages": ["en"],
    "strengths": ["instruction_following", "safety", "factual_qa"],
    "weaknesses": ["multilingual", "code"],
    "lora_targets": ["q_proj", "k_proj", "v_proj"],
    "min_corpus_chunks": 100,
    "license": "commercial_ok",
    "family": "gemma2"
  },
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
    "vram_fp16_gb": 17,
    "vram_4bit_gb": 5,
    "vram_lora_overhead_gb": 6,
    "languages": ["en", "zh"],
    "strengths": ["reasoning", "analytical_qa", "technical_docs"],
    "weaknesses": ["conversational_qa", "very_short_answers"],
    "lora_targets": ["q_proj", "k_proj", "v_proj"],
    "min_corpus_chunks": 100,
    "license": "commercial_ok",
    "family": "deepseek"
  }
}
```

### Candidate Scoring (post VRAM filter)

Each eligible model gets a score 0–1 from four signals:

```
language_score   = 1.0 if corpus_language in model.languages else 0.0
domain_score     = affinity(model.strengths, detected_task_type)  # 0, 0.5, or 1.0
corpus_score     = 1.0 if chunk_count >= model.min_corpus_chunks else chunk_count / model.min_corpus_chunks
vram_score       = 1.0 - (model_vram_used / free_vram)  # prefer models that leave headroom

final_score = 0.35*language_score + 0.30*domain_score + 0.20*corpus_score + 0.15*vram_score
```

`detected_task_type` is inferred from corpus characteristics: entity density → `technical_docs`; short Q&A format → `factual_qa`; code blocks detected → `code`; etc.

The agent is shown the scored shortlist and reasons about experiment order, but is not bound to follow it exactly.

---

## Results Format

`model_scout_results.tsv` (per UC, created fresh each ModelScout session):

```
commit	model	quantization	lora_steps	lora_rank	corpus_chunks	faq_count	prs	vram_gb	wall_seconds	status	agent_reasoning
a1b2c3d	Mistral-7B-Instruct-v0.3	4bit	500	16	200	20	0.68	19.1	493	keep	PRS rising, re-run with 1000 steps planned
b2c3d4e	Mistral-7B-Instruct-v0.3	4bit	1000	16	200	20	0.74	19.3	981	keep	Improvement confirmed, best so far
c3d4e5f	Qwen2.5-7B-Instruct	4bit	1000	16	300	20	0.71	20.1	1024	discard	Lower than Mistral despite more chunks; domain is English-only
d4e5f6g	Llama-3.2-3B-Instruct	fp16	1000	16	200	20	0.63	9.8	743	discard	Smaller model, expected lower ceiling
```

Final recommendation is appended as a comment row at the bottom:
```
# RECOMMENDATION: mistralai/Mistral-7B-Instruct-v0.3 (4bit, rank=16)
# PRS=0.74 | VRAM=19.3GB | Reasoning: Best PRS with stable training curve.
#   Domain is English technical docs — Mistral's strength. Qwen2 offers no
#   advantage here despite multilingual capability.
```

---

## model_scout_program.md

The operator-editable instruction file given to the LLM agent. Structure:

```markdown
# ModelScout Program

## Your Role
You are ModelScout, an interactive AI agent that helps KVForge users identify
the best open LLM for their domain. You run experiments, observe results,
adjust parameters, and keep the user informed.

## What You Control
- Which model to try next (from the eligible filtered list)
- lora_steps: 100–2000 (default 500)
- lora_rank: 8, 16, 32 (default 16)
- corpus_chunks: 50–2000 (default 200)
- faq_count: 10–100 (default 20)
- quantization: fp16 | 4bit | 8bit (default: use 4bit if fp16 > 80% free VRAM)

## What You Cannot Change
- The evaluation metric (PRS) — this is fixed
- The model registry — you select from it, not modify it
- The user's budget choice — respect it strictly

## Experiment Order Heuristic
Start with the highest-scoring model from the registry scorer.
After each result, reason: should I try a variant of this model family,
or move to a new family? Prefer breadth early, depth later.

## Parameter Adjustment Rules
[see Parameter Adjustment Rules table in spec]

## Stopping Criteria
- Budget exhausted → emit recommendation from best result
- User types 'stop' → emit recommendation immediately
- You are confident (best model beats second-best by > 0.08 PRS AND
  you've tried ≥ 3 models) → offer to stop, let user decide

## Communication Style
- Always announce what you're about to do and why, before doing it
- Give the user a short override window (mention "type 'skip' to skip")
- After each result, give a one-sentence interpretation
- Ask questions when results are surprising
- Be concise — users are watching a live session
```

---

## Configuration

New per-UC fields in `DatasourceConfig`:

```json
{
  "model_registry_path": "core/model_registry.json",
  "model_scout_program": "model_scout_program.md",
  "model_scout_results": "model_scout_results.tsv",
  "scout_initial_corpus_chunks": 200,
  "scout_initial_faq_count": 20,
  "scout_initial_lora_steps": 500,
  "scout_initial_lora_rank": 16,
  "scout_max_lora_steps": 2000,
  "scout_max_corpus_chunks": 2000,
  "scout_max_faq_count": 100
}
```

---

## Data Flow

### Pre-index mode
```
Raw documents
  → sample scout_initial_corpus_chunks chunks
  → sleep_faq_generator → scout_initial_faq_count FAQs
  → [experiment loop]
       load candidate model
       → mini-LoRA (scout_initial_lora_steps, adjustable)
       → PRS eval on FAQs (faq_coverage only — no VDB yet)
       → log to model_scout_results.tsv
  → final recommendation
  → user proceeds to full index with recommended model
```

### Post-index mode
```
Existing VDB (hot-tier chunks preferred)
  → scroll up to scout_initial_corpus_chunks hot chunks
  → existing FAQ file OR generate from hot-tier chunks
  → [experiment loop]
       load candidate model
       → mini-LoRA (uses real replay buffer, hot-tier weighted)
       → three-signal PRS eval (faq + vdb_coverage sample + realtime)
       → log to model_scout_results.tsv
  → final recommendation
  → user optionally re-indexes with recommended model
```

---

## Studio Integration

New SSE endpoint: `GET /api/uc/{config_name}/model-scout/stream`

Events emitted by `SSEAdapter`:
```
event: scout_message    data: {"text": "Detected NVIDIA A10G, 21GB free..."}
event: scout_question   data: {"text": "Select budget mode:", "options": ["A","B","C","D"]}
event: scout_progress   data: {"label": "Training Mistral-7B step 234/500", "pct": 0.47}
event: scout_result     data: {"model": "Mistral-7B", "prs": 0.68, "status": "keep"}
event: scout_done       data: {"recommendation": "Mistral-7B-Instruct-v0.3", "prs": 0.74}
```

User responses sent to: `POST /api/uc/{config_name}/model-scout/respond`
```json
{"response": "B"}
```

---

## What Does Not Change

- LoRA training mechanism (`lora_trainer.py`) — ModelScout calls it with a step cap
- PRS evaluation (`prs_evaluator.py`) — called directly, results read back
- FAQ generation (`sleep_faq_generator.py`) — called with configurable count
- VDB backends — ModelScout uses whichever backend the UC is configured for
- KV tensor infrastructure — not used during ModelScout (no KV indexing in scout mode)

---

## Success Criteria

1. ModelScout correctly filters out models that exceed available VRAM before running a single experiment
2. Agent adjusts `lora_steps` upward when PRS curve is still falling at step limit
3. Agent increases `faq_count` when FAQ-level PRS variance exceeds 0.15
4. User can interrupt with "skip", "stop", "try <model>" at any point
5. `model_scout_results.tsv` is written correctly after each experiment with all fields
6. Pre-index and post-index modes produce valid PRS scores and a final recommendation
7. CLI and Studio SSE interfaces both work with the same agent core
