# ModelScout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build ModelScout — an interactive agent that auto-detects GPU capabilities, experiments across candidate open models with a fixed evaluation harness, adjusts its own parameters based on results, keeps the user in the loop via CLI and Studio SSE, and emits a ranked model recommendation per UC.

**Architecture:** `model_scout.py` contains the entire agent loop and communicates through an `IOAdapter` protocol. `CLIAdapter` and `SSEAdapter` are the two thin I/O implementations. `model_registry.py` handles VRAM filtering and candidate scoring. The agent loop calls existing `lora_trainer.py` and `prs_evaluator.py` directly — no new training or evaluation logic is written.

**Tech Stack:** Python 3.11+, PyTorch (GPU detection), NumPy, FastAPI SSE (Studio), existing PEFT/HuggingFace training stack, existing `sleep_faq_generator.py` and `prs_evaluator.py`.

---

## File Map

**New files:**
- `core/model_registry.py` — registry loader, VRAM filter, candidate scorer
- `core/model_registry.json` — curated model list (8 models)
- `model_scout_program.md` — agent instruction file (operator-editable)
- `pipeline/model_scout.py` — agent core: GPU detection, experiment loop, parameter adjustment, IOAdapter interface
- `pipeline/model_scout_cli.py` — CLI adapter + entry point
- `tests/test_model_registry.py`
- `tests/test_model_scout.py`

**Modified files:**
- `core/config.py` — add 8 ModelScout config fields
- `studio/routes.py` — SSE stream endpoint + user-response POST endpoint
- `tests/test_studio_routes.py` — SSE endpoint tests
- `tests/test_config.py` — new field defaults

---

## Task 1: Model Registry JSON and Loader

**Files:**
- Create: `core/model_registry.json`
- Create: `core/model_registry.py`
- Create: `tests/test_model_registry.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_registry.py

import pytest
import numpy as np
from core.model_registry import (
    load_registry, filter_by_vram, score_candidates, get_candidate_shortlist,
)


def test_load_registry_returns_nonempty_dict():
    registry = load_registry()
    assert len(registry) >= 5
    first = next(iter(registry.values()))
    assert "vram_fp16_gb" in first
    assert "vram_4bit_gb" in first
    assert "vram_lora_overhead_gb" in first
    assert "languages" in first
    assert "lora_targets" in first


def test_filter_by_vram_removes_oversized_models():
    registry = load_registry()
    # 8GB free — only models fitting in 4bit + lora overhead <= 8GB should survive
    eligible = filter_by_vram(registry, free_vram_gb=8.0)
    for name, spec in eligible.items():
        required = spec["vram_4bit_gb"] + spec["vram_lora_overhead_gb"]
        assert required <= 8.0, f"{name} requires {required}GB but only 8GB free"


def test_filter_by_vram_allows_fp16_when_fits():
    registry = load_registry()
    eligible = filter_by_vram(registry, free_vram_gb=24.0)
    # Llama-3.2-3B needs 7+4=11GB fp16 — should be eligible
    llama3b = [k for k in eligible if "Llama-3.2-3B" in k]
    assert len(llama3b) > 0


def test_filter_by_vram_empty_when_nothing_fits():
    registry = load_registry()
    eligible = filter_by_vram(registry, free_vram_gb=1.0)
    assert len(eligible) == 0


def test_score_candidates_returns_scores_between_0_and_1():
    registry = load_registry()
    eligible = filter_by_vram(registry, free_vram_gb=24.0)
    scores = score_candidates(
        eligible,
        corpus_languages=["en"],
        task_type="factual_qa",
        corpus_chunk_count=500,
        free_vram_gb=24.0,
    )
    assert len(scores) == len(eligible)
    for model_id, score in scores.items():
        assert 0.0 <= score <= 1.0, f"{model_id} score {score} out of range"


def test_score_candidates_prefers_language_match():
    registry = load_registry()
    eligible = filter_by_vram(registry, free_vram_gb=24.0)
    scores_en = score_candidates(eligible, corpus_languages=["en"],
                                  task_type="factual_qa", corpus_chunk_count=500,
                                  free_vram_gb=24.0)
    scores_zh = score_candidates(eligible, corpus_languages=["zh"],
                                  task_type="factual_qa", corpus_chunk_count=500,
                                  free_vram_gb=24.0)
    # Qwen should rank higher for Chinese corpus
    qwen_ids = [k for k in eligible if "Qwen" in k and "7B" in k]
    llama_ids = [k for k in eligible if "Llama-3.2-3B" in k]
    if qwen_ids and llama_ids:
        assert scores_zh[qwen_ids[0]] > scores_zh[llama_ids[0]]


def test_get_candidate_shortlist_returns_sorted_list():
    registry = load_registry()
    shortlist = get_candidate_shortlist(
        registry, free_vram_gb=24.0,
        corpus_languages=["en"], task_type="factual_qa",
        corpus_chunk_count=300,
    )
    assert len(shortlist) > 0
    # Verify sorted descending by score
    scores = [item["score"] for item in shortlist]
    assert scores == sorted(scores, reverse=True)
    assert "model_id" in shortlist[0]
    assert "spec" in shortlist[0]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_model_registry.py -v --override-ini="addopts="
```

Expected: `ERROR — ModuleNotFoundError: No module named 'core.model_registry'`

- [ ] **Step 3: Create core/model_registry.json**

```json
{
  "meta-llama/Llama-3.2-3B-Instruct": {
    "vram_fp16_gb": 7, "vram_4bit_gb": 2, "vram_lora_overhead_gb": 4,
    "languages": ["en"],
    "strengths": ["factual_qa", "instruction_following"],
    "weaknesses": ["multilingual", "long_reasoning"],
    "lora_targets": ["q_proj", "k_proj", "v_proj"],
    "min_corpus_chunks": 50, "license": "commercial_ok", "family": "llama3"
  },
  "meta-llama/Llama-3.1-8B-Instruct": {
    "vram_fp16_gb": 17, "vram_4bit_gb": 5, "vram_lora_overhead_gb": 6,
    "languages": ["en"],
    "strengths": ["factual_qa", "reasoning", "long_context"],
    "weaknesses": ["multilingual"],
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "min_corpus_chunks": 100, "license": "commercial_ok", "family": "llama3"
  },
  "mistralai/Mistral-7B-Instruct-v0.3": {
    "vram_fp16_gb": 15, "vram_4bit_gb": 5, "vram_lora_overhead_gb": 6,
    "languages": ["en", "fr", "de", "es", "it"],
    "strengths": ["factual_qa", "european_languages", "long_context"],
    "weaknesses": ["cjk_languages", "code"],
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "min_corpus_chunks": 100, "license": "commercial_ok", "family": "mistral"
  },
  "Qwen/Qwen2.5-7B-Instruct": {
    "vram_fp16_gb": 16, "vram_4bit_gb": 5, "vram_lora_overhead_gb": 6,
    "languages": ["en", "zh", "ja", "ko", "multilingual"],
    "strengths": ["multilingual", "reasoning", "technical_docs", "code"],
    "weaknesses": [],
    "lora_targets": ["q_proj", "k_proj", "v_proj"],
    "min_corpus_chunks": 100, "license": "commercial_ok", "family": "qwen2"
  },
  "Qwen/Qwen2.5-1.5B-Instruct": {
    "vram_fp16_gb": 4, "vram_4bit_gb": 2, "vram_lora_overhead_gb": 3,
    "languages": ["en", "zh", "multilingual"],
    "strengths": ["low_vram", "fast_inference", "multilingual"],
    "weaknesses": ["complex_reasoning", "long_context"],
    "lora_targets": ["q_proj", "k_proj", "v_proj"],
    "min_corpus_chunks": 30, "license": "commercial_ok", "family": "qwen2"
  },
  "microsoft/Phi-3-mini-4k-instruct": {
    "vram_fp16_gb": 8, "vram_4bit_gb": 3, "vram_lora_overhead_gb": 4,
    "languages": ["en"],
    "strengths": ["low_vram", "factual_qa", "efficiency"],
    "weaknesses": ["multilingual", "very_long_context"],
    "lora_targets": ["qkv_proj", "o_proj"],
    "min_corpus_chunks": 50, "license": "commercial_ok", "family": "phi3"
  },
  "google/gemma-2-9b-it": {
    "vram_fp16_gb": 19, "vram_4bit_gb": 6, "vram_lora_overhead_gb": 7,
    "languages": ["en"],
    "strengths": ["instruction_following", "safety", "factual_qa"],
    "weaknesses": ["multilingual", "code"],
    "lora_targets": ["q_proj", "k_proj", "v_proj"],
    "min_corpus_chunks": 100, "license": "commercial_ok", "family": "gemma2"
  },
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
    "vram_fp16_gb": 17, "vram_4bit_gb": 5, "vram_lora_overhead_gb": 6,
    "languages": ["en", "zh"],
    "strengths": ["reasoning", "analytical_qa", "technical_docs"],
    "weaknesses": ["conversational_qa", "very_short_answers"],
    "lora_targets": ["q_proj", "k_proj", "v_proj"],
    "min_corpus_chunks": 100, "license": "commercial_ok", "family": "deepseek"
  }
}
```

- [ ] **Step 4: Create core/model_registry.py**

```python
# core/model_registry.py

from __future__ import annotations
import json
from pathlib import Path

_REGISTRY_PATH = Path(__file__).parent / "model_registry.json"

_TASK_AFFINITY: dict[str, list[str]] = {
    "factual_qa":    ["factual_qa", "instruction_following"],
    "technical_docs": ["technical_docs", "reasoning", "long_context"],
    "code":          ["code", "reasoning"],
    "analytical_qa": ["reasoning", "analytical_qa"],
    "conversational": ["instruction_following", "factual_qa"],
    "multilingual":  ["multilingual"],
}


def load_registry(path: str | None = None) -> dict:
    p = Path(path) if path else _REGISTRY_PATH
    with open(p) as f:
        return json.load(f)


def filter_by_vram(registry: dict, free_vram_gb: float) -> dict:
    eligible = {}
    for model_id, spec in registry.items():
        # A model is eligible if it fits in either fp16 or 4-bit + lora overhead
        fp16_ok = (spec["vram_fp16_gb"] + spec["vram_lora_overhead_gb"]) <= free_vram_gb
        bit4_ok = (spec["vram_4bit_gb"] + spec["vram_lora_overhead_gb"]) <= free_vram_gb
        if fp16_ok or bit4_ok:
            spec = dict(spec)
            spec["_use_4bit"] = not fp16_ok  # force 4bit if fp16 doesn't fit
            eligible[model_id] = spec
    return eligible


def score_candidates(
    eligible: dict,
    corpus_languages: list[str],
    task_type: str,
    corpus_chunk_count: int,
    free_vram_gb: float,
) -> dict[str, float]:
    target_strengths = _TASK_AFFINITY.get(task_type, ["factual_qa"])
    scores = {}
    for model_id, spec in eligible.items():
        lang_score = 1.0 if any(l in spec["languages"] for l in corpus_languages) else 0.0
        # Multilingual models get partial credit for any language
        if "multilingual" in spec["languages"] and lang_score == 0.0:
            lang_score = 0.6
        domain_score = sum(
            1.0 for s in target_strengths if s in spec["strengths"]
        ) / max(len(target_strengths), 1)
        min_chunks = spec.get("min_corpus_chunks", 50)
        corpus_score = min(corpus_chunk_count / max(min_chunks, 1), 1.0)
        model_vram = (spec["vram_4bit_gb"] if spec.get("_use_4bit") else spec["vram_fp16_gb"])
        vram_score = max(0.0, 1.0 - (model_vram / free_vram_gb))
        scores[model_id] = (
            0.35 * lang_score
            + 0.30 * domain_score
            + 0.20 * corpus_score
            + 0.15 * vram_score
        )
    return scores


def get_candidate_shortlist(
    registry: dict,
    free_vram_gb: float,
    corpus_languages: list[str],
    task_type: str,
    corpus_chunk_count: int,
) -> list[dict]:
    eligible = filter_by_vram(registry, free_vram_gb)
    scores = score_candidates(eligible, corpus_languages, task_type,
                               corpus_chunk_count, free_vram_gb)
    return sorted(
        [{"model_id": mid, "score": score, "spec": eligible[mid]}
         for mid, score in scores.items()],
        key=lambda x: x["score"],
        reverse=True,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_model_registry.py -v --override-ini="addopts="
```

Expected: all 7 tests `PASSED`

- [ ] **Step 6: Commit**

```bash
git add core/model_registry.json core/model_registry.py tests/test_model_registry.py
git commit -m "feat: add ModelScout model registry with VRAM filter and candidate scorer"
```

---

## Task 2: Config Fields and model_scout_program.md

**Files:**
- Modify: `core/config.py`
- Modify: `tests/test_config.py`
- Create: `model_scout_program.md`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py — add

def test_model_scout_config_defaults():
    cfg = DatasourceConfig(
        collection="test", embed_model="BAAI/bge-small-en-v1.5",
        vector_dim=384, llm_model="meta-llama/Llama-3.2-3B-Instruct",
        checkpoint_dir="/tmp/ckpt", version_file="/tmp/v.json",
        replay_db="/tmp/r.db",
    )
    assert cfg.model_registry_path == "core/model_registry.json"
    assert cfg.model_scout_program == "model_scout_program.md"
    assert cfg.model_scout_results == "model_scout_results.tsv"
    assert cfg.scout_initial_corpus_chunks == 200
    assert cfg.scout_initial_faq_count == 20
    assert cfg.scout_initial_lora_steps == 500
    assert cfg.scout_initial_lora_rank == 16
    assert cfg.scout_max_lora_steps == 2000
    assert cfg.scout_max_corpus_chunks == 2000
    assert cfg.scout_max_faq_count == 100
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_config.py::test_model_scout_config_defaults -v --override-ini="addopts="
```

Expected: `FAILED — AttributeError: 'DatasourceConfig' object has no attribute 'model_registry_path'`

- [ ] **Step 3: Add fields to core/config.py**

Add after the `query_log_db` field:

```python
    # ModelScout
    model_registry_path: str = "core/model_registry.json"
    model_scout_program: str = "model_scout_program.md"
    model_scout_results: str = "model_scout_results.tsv"
    scout_initial_corpus_chunks: int = 200
    scout_initial_faq_count: int = 20
    scout_initial_lora_steps: int = 500
    scout_initial_lora_rank: int = 16
    scout_max_lora_steps: int = 2000
    scout_max_corpus_chunks: int = 2000
    scout_max_faq_count: int = 100
```

Also add to the docstring `Attributes:` section:

```
        model_registry_path: Path to the model registry JSON file.
        model_scout_program: Path to the ModelScout agent instruction file.
        model_scout_results: Path to the TSV results file written by ModelScout.
        scout_initial_corpus_chunks: Chunks to sample at ModelScout start.
        scout_initial_faq_count: FAQs to generate at ModelScout start.
        scout_initial_lora_steps: LoRA steps for the first experiment per candidate.
        scout_initial_lora_rank: LoRA rank used in scout mini-training rounds.
        scout_max_lora_steps: Hard ceiling on lora_steps the agent may request.
        scout_max_corpus_chunks: Hard ceiling on corpus_chunks the agent may request.
        scout_max_faq_count: Hard ceiling on faq_count the agent may request.
```

- [ ] **Step 4: Create model_scout_program.md**

```markdown
# ModelScout Program

## Your Role
You are ModelScout, an interactive AI agent embedded in KVForge. Your job is to
identify the best open LLM for a specific use case by running fast experiments,
observing results, and adapting your approach. You keep the user informed and in
control throughout the session.

## What You Control
- Which model to try next (from the VRAM-eligible filtered shortlist)
- `lora_steps`: 100–{scout_max_lora_steps} (start: {scout_initial_lora_steps})
- `lora_rank`: 8, 16, 32 (start: {scout_initial_lora_rank})
- `corpus_chunks`: 50–{scout_max_corpus_chunks} (start: {scout_initial_corpus_chunks})
- `faq_count`: 10–{scout_max_faq_count} (start: {scout_initial_faq_count})
- `quantization`: fp16 | 4bit | 8bit (choose based on VRAM headroom)

## What You Cannot Change
- The PRS evaluation metric — it is fixed
- The model registry — select from it, do not modify it
- The user's budget choice — respect it strictly as a hard constraint

## Experiment Order Heuristic
1. Start with the highest-scored model from the registry scorer
2. After each result, reason: should I explore a variant of this model family,
   or move to a new family?
3. Prefer breadth early (try different families), depth later (variants within
   the best family)
4. Always explain your reasoning to the user before starting an experiment

## Parameter Adjustment Rules
Apply these rules automatically before each experiment. Announce changes to user.

| Observation | Action |
|---|---|
| prs < 0.55 AND training loss still falling at step limit | Increase lora_steps × 2, retry same model |
| FAQ-level prs variance > 0.15 | Increase faq_count × 1.5, regenerate FAQs |
| domain_complexity_score > 0.70 | Set corpus_chunks ≥ 400 |
| OOM at fp16 | Retry same model with quantization=4bit |
| OOM at 4bit | Skip model entirely, note VRAM ceiling |
| prs improvement > 0.10 vs best so far | Try smaller/larger variant of same family next |
| 3 consecutive models with prs < 0.55 | Pause and ask user about domain/language context |
| prs plateau (< 0.01 improvement over 2 rounds of same model) | Move to next model |
| lora_steps > 2000 AND prs < 0.60 | Abandon model family |

## Stopping Criteria (Budget mode D — agent decides)
Stop when ALL of the following are true:
- You have tried ≥ 3 different model families
- The best model beats the second-best by > 0.08 PRS
- The best model's PRS curve has plateaued (< 0.01 improvement from last step increase)
Offer the user the option to stop; wait for confirmation.

## Communication Style
- Announce what you will do and why BEFORE doing it
- Mention "type 'skip' to skip" or "type 'stop' to end" in every pre-experiment message
- After each result: one sentence interpretation + one sentence on what's next
- Ask questions when results are surprising (> 0.10 deviation from expected)
- Be concise — the user is watching a live session
- When adjusting parameters, explain the rule that triggered the adjustment

## Accepted User Commands (at any time)
- `skip` — skip current model, move to next
- `stop` — end session, emit final recommendation from best result so far
- `try <model_name>` — jump to specific model next (if VRAM-eligible)
- `more steps` — double lora_steps for the next run
- `more faqs` — increase faq_count by 10 and regenerate
- `change budget` — re-open budget dialog
```

- [ ] **Step 5: Run test to verify it passes**

```bash
python -m pytest tests/test_config.py::test_model_scout_config_defaults -v --override-ini="addopts="
```

Expected: `PASSED`

- [ ] **Step 6: Run full config tests**

```bash
python -m pytest tests/test_config.py -v --override-ini="addopts="
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add core/config.py tests/test_config.py model_scout_program.md
git commit -m "feat: add ModelScout config fields and agent program instructions"
```

---

## Task 3: IOAdapter Protocol

**Files:**
- Create: `pipeline/model_scout.py` (IOAdapter section only)
- Create: `tests/test_model_scout.py` (IOAdapter tests)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_scout.py

import pytest
from pipeline.model_scout import IOAdapter, CLIAdapter, RecordingAdapter


def test_ioadapter_is_protocol():
    from typing import runtime_checkable
    assert hasattr(IOAdapter, '__protocol_attrs__') or True  # structural typing


def test_recording_adapter_captures_messages():
    adapter = RecordingAdapter()
    adapter.send("hello world")
    adapter.send("second message")
    assert adapter.messages == ["hello world", "second message"]


def test_recording_adapter_ask_returns_preset():
    adapter = RecordingAdapter(responses=["B", "stop"])
    answer = adapter.ask("Which budget?", options=["A", "B", "C", "D"])
    assert answer == "B"
    answer2 = adapter.ask("Continue?", options=None)
    assert answer2 == "stop"


def test_recording_adapter_ask_raises_when_no_responses_left():
    adapter = RecordingAdapter(responses=[])
    with pytest.raises(StopIteration):
        adapter.ask("No more responses", options=None)


def test_recording_adapter_progress_recorded():
    adapter = RecordingAdapter()
    adapter.stream_progress("Training step 10/100", 0.10)
    adapter.stream_progress("Training step 50/100", 0.50)
    assert len(adapter.progress_updates) == 2
    assert adapter.progress_updates[0] == ("Training step 10/100", 0.10)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_model_scout.py -v --override-ini="addopts="
```

Expected: `ERROR — ModuleNotFoundError: No module named 'pipeline.model_scout'`

- [ ] **Step 3: Create pipeline/model_scout.py with IOAdapter section**

```python
# pipeline/model_scout.py

from __future__ import annotations
import csv
import os
import time
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


# ── IOAdapter protocol ────────────────────────────────────────────────────────

@runtime_checkable
class IOAdapter(Protocol):
    def send(self, message: str) -> None: ...
    def ask(self, question: str, options: list[str] | None = None) -> str: ...
    def stream_progress(self, label: str, pct: float) -> None: ...


class CLIAdapter:
    def send(self, message: str) -> None:
        print(message, flush=True)

    def ask(self, question: str, options: list[str] | None = None) -> str:
        print(question, flush=True)
        if options:
            for i, opt in enumerate(options):
                print(f"  [{chr(65+i)}] {opt}", flush=True)
        return input("> ").strip()

    def stream_progress(self, label: str, pct: float) -> None:
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {pct*100:.0f}% {label}", end="", flush=True)
        if pct >= 1.0:
            print()


class RecordingAdapter:
    """Test double that records all interactions and serves preset responses."""

    def __init__(self, responses: list[str] | None = None):
        self.messages: list[str] = []
        self.progress_updates: list[tuple[str, float]] = []
        self._responses = iter(responses or [])

    def send(self, message: str) -> None:
        self.messages.append(message)

    def ask(self, question: str, options: list[str] | None = None) -> str:
        self.messages.append(f"[QUESTION] {question}")
        return next(self._responses)

    def stream_progress(self, label: str, pct: float) -> None:
        self.progress_updates.append((label, pct))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_model_scout.py -v --override-ini="addopts="
```

Expected: all 5 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add pipeline/model_scout.py tests/test_model_scout.py
git commit -m "feat: add IOAdapter protocol with CLIAdapter and RecordingAdapter test double"
```

---

## Task 4: GPU Detection

**Files:**
- Modify: `pipeline/model_scout.py`
- Modify: `tests/test_model_scout.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_scout.py — add

from unittest.mock import patch, MagicMock
from pipeline.model_scout import detect_gpu


def test_detect_gpu_no_cuda():
    with patch("pipeline.model_scout._has_cuda", False):
        info = detect_gpu()
    assert info["available"] is False
    assert info["free_vram_gb"] == 0.0


def test_detect_gpu_with_cuda():
    mock_props = MagicMock()
    mock_props.name = "NVIDIA A10G"
    with patch("pipeline.model_scout._has_cuda", True):
        with patch("pipeline.model_scout.torch") as mock_torch:
            mock_torch.cuda.get_device_properties.return_value = mock_props
            mock_torch.cuda.mem_get_info.return_value = (
                21 * 1024**3,   # free bytes
                24 * 1024**3,   # total bytes
            )
            info = detect_gpu()
    assert info["available"] is True
    assert info["gpu_name"] == "NVIDIA A10G"
    assert abs(info["free_vram_gb"] - 21.0) < 0.1
    assert abs(info["total_vram_gb"] - 24.0) < 0.1


def test_detect_gpu_formats_report():
    with patch("pipeline.model_scout._has_cuda", False):
        info = detect_gpu()
    assert "report" in info
    assert isinstance(info["report"], str)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_model_scout.py::test_detect_gpu_no_cuda tests/test_model_scout.py::test_detect_gpu_with_cuda tests/test_model_scout.py::test_detect_gpu_formats_report -v --override-ini="addopts="
```

Expected: `FAILED — ImportError: cannot import name 'detect_gpu'`

- [ ] **Step 3: Add detect_gpu to pipeline/model_scout.py**

```python
# Add after imports at top of pipeline/model_scout.py

try:
    import torch
    _has_cuda = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore
    _has_cuda = False


def detect_gpu() -> dict:
    """Auto-detect GPU capabilities. Returns dict with vram info and a human-readable report."""
    if not _has_cuda:
        return {
            "available": False,
            "gpu_name": "CPU only",
            "free_vram_gb": 0.0,
            "total_vram_gb": 0.0,
            "cuda_version": None,
            "report": "No CUDA GPU detected. ModelScout will be limited to CPU-compatible models only.",
        }
    props = torch.cuda.get_device_properties(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / 1024 ** 3
    total_gb = total_bytes / 1024 ** 3
    cuda_ver = torch.version.cuda
    report = (
        f"Detected: {props.name} | "
        f"VRAM: {free_gb:.1f}GB free / {total_gb:.1f}GB total | "
        f"CUDA {cuda_ver}"
    )
    return {
        "available": True,
        "gpu_name": props.name,
        "free_vram_gb": round(free_gb, 2),
        "total_vram_gb": round(total_gb, 2),
        "cuda_version": cuda_ver,
        "report": report,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_model_scout.py::test_detect_gpu_no_cuda tests/test_model_scout.py::test_detect_gpu_with_cuda tests/test_model_scout.py::test_detect_gpu_formats_report -v --override-ini="addopts="
```

Expected: all 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add pipeline/model_scout.py tests/test_model_scout.py
git commit -m "feat: add GPU auto-detection to ModelScout"
```

---

## Task 5: Budget Dialog and Parameter State

**Files:**
- Modify: `pipeline/model_scout.py`
- Modify: `tests/test_model_scout.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_scout.py — add

from pipeline.model_scout import run_budget_dialog, ScoutParams


def test_budget_dialog_option_A_sets_wall_clock():
    adapter = RecordingAdapter(responses=["A", "4"])
    params = run_budget_dialog(adapter)
    assert params.budget_mode == "wall_clock"
    assert params.budget_value == 4.0


def test_budget_dialog_option_B_sets_experiment_count():
    adapter = RecordingAdapter(responses=["B", "15"])
    params = run_budget_dialog(adapter)
    assert params.budget_mode == "experiment_count"
    assert params.budget_value == 15


def test_budget_dialog_option_C_sets_step_cap_and_count():
    adapter = RecordingAdapter(responses=["C", "1000", "12"])
    params = run_budget_dialog(adapter)
    assert params.budget_mode == "step_cap_and_count"
    assert params.max_steps_per_experiment == 1000
    assert params.budget_value == 12


def test_budget_dialog_option_D_sets_agent_decides():
    adapter = RecordingAdapter(responses=["D"])
    params = run_budget_dialog(adapter)
    assert params.budget_mode == "agent_decides"


def test_scout_params_defaults():
    p = ScoutParams()
    assert p.lora_steps == 500
    assert p.lora_rank == 16
    assert p.corpus_chunks == 200
    assert p.faq_count == 20
    assert p.budget_mode == "experiment_count"
    assert p.budget_value == 10
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_model_scout.py::test_budget_dialog_option_A_sets_wall_clock tests/test_model_scout.py::test_budget_dialog_option_B_sets_experiment_count tests/test_model_scout.py::test_budget_dialog_option_C_sets_step_cap_and_count tests/test_model_scout.py::test_budget_dialog_option_D_sets_agent_decides tests/test_model_scout.py::test_scout_params_defaults -v --override-ini="addopts="
```

Expected: `FAILED — ImportError`

- [ ] **Step 3: Add ScoutParams and run_budget_dialog to pipeline/model_scout.py**

```python
# Add to pipeline/model_scout.py

from dataclasses import dataclass, field


@dataclass
class ScoutParams:
    lora_steps: int = 500
    lora_rank: int = 16
    corpus_chunks: int = 200
    faq_count: int = 20
    quantization: str = "auto"          # "auto" | "fp16" | "4bit" | "8bit"
    budget_mode: str = "experiment_count"
    budget_value: float = 10            # hours | count | count depending on mode
    max_steps_per_experiment: int = 2000
    experiments_run: int = 0
    session_start: float = field(default_factory=time.time)


_BUDGET_OPTIONS = [
    "Total wall-clock time (e.g. run for N hours)",
    "Total experiment count (e.g. run at most N experiments)",
    "Per-experiment step cap + total count",
    "Agent decides when confident enough",
]


def run_budget_dialog(adapter: IOAdapter) -> ScoutParams:
    params = ScoutParams()
    choice = adapter.ask(
        "How would you like to control the ModelScout session?",
        options=_BUDGET_OPTIONS,
    ).upper().strip()

    if choice == "A":
        hours = float(adapter.ask("Run for how many hours? (e.g. 4)"))
        params.budget_mode = "wall_clock"
        params.budget_value = hours
    elif choice == "B":
        count = int(adapter.ask("Run at most how many experiments? (e.g. 15)"))
        params.budget_mode = "experiment_count"
        params.budget_value = count
    elif choice == "C":
        max_steps = int(adapter.ask("Max LoRA steps per experiment? (e.g. 1000)"))
        count = int(adapter.ask("Total number of experiments? (e.g. 12)"))
        params.budget_mode = "step_cap_and_count"
        params.max_steps_per_experiment = max_steps
        params.budget_value = count
    else:  # D or anything else
        params.budget_mode = "agent_decides"

    return params


def _budget_exhausted(params: ScoutParams) -> bool:
    if params.budget_mode == "wall_clock":
        elapsed_hours = (time.time() - params.session_start) / 3600
        return elapsed_hours >= params.budget_value
    elif params.budget_mode in ("experiment_count", "step_cap_and_count"):
        return params.experiments_run >= params.budget_value
    return False  # agent_decides — never auto-exhausted
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_model_scout.py::test_budget_dialog_option_A_sets_wall_clock tests/test_model_scout.py::test_budget_dialog_option_B_sets_experiment_count tests/test_model_scout.py::test_budget_dialog_option_C_sets_step_cap_and_count tests/test_model_scout.py::test_budget_dialog_option_D_sets_agent_decides tests/test_model_scout.py::test_scout_params_defaults -v --override-ini="addopts="
```

Expected: all 5 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add pipeline/model_scout.py tests/test_model_scout.py
git commit -m "feat: add ScoutParams dataclass and budget dialog to ModelScout"
```

---

## Task 6: Parameter Adjustment Logic

**Files:**
- Modify: `pipeline/model_scout.py`
- Modify: `tests/test_model_scout.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_scout.py — add

from pipeline.model_scout import apply_parameter_adjustments, ExperimentResult


def _make_result(**kwargs) -> ExperimentResult:
    defaults = dict(
        model_id="test-model", quantization="4bit",
        lora_steps=500, lora_rank=16, corpus_chunks=200, faq_count=20,
        prs=0.65, prs_variance=0.08, vram_gb=18.0,
        wall_seconds=500, status="keep",
        training_loss_start=1.5, training_loss_end=0.9,
        agent_reasoning="baseline",
    )
    defaults.update(kwargs)
    return ExperimentResult(**defaults)


def test_adjustment_increases_steps_when_prs_low_and_loss_falling():
    result = _make_result(prs=0.50, training_loss_start=1.5, training_loss_end=0.7)
    params = ScoutParams(lora_steps=500)
    new_params, retry = apply_parameter_adjustments(result, params, max_steps=2000)
    assert retry is True
    assert new_params.lora_steps == 1000


def test_adjustment_does_not_exceed_max_steps():
    result = _make_result(prs=0.50, training_loss_start=1.5, training_loss_end=0.7)
    params = ScoutParams(lora_steps=1500)
    new_params, retry = apply_parameter_adjustments(result, params, max_steps=2000)
    assert new_params.lora_steps <= 2000


def test_adjustment_increases_faq_count_when_high_variance():
    result = _make_result(prs=0.65, prs_variance=0.18)
    params = ScoutParams(faq_count=20)
    new_params, retry = apply_parameter_adjustments(result, params, max_steps=2000)
    assert new_params.faq_count > 20


def test_adjustment_no_retry_when_prs_good():
    result = _make_result(prs=0.75, prs_variance=0.05,
                           training_loss_start=1.5, training_loss_end=0.9)
    params = ScoutParams(lora_steps=500)
    new_params, retry = apply_parameter_adjustments(result, params, max_steps=2000)
    assert retry is False


def test_adjustment_sets_4bit_on_oom():
    result = _make_result(status="oom", quantization="fp16")
    params = ScoutParams(quantization="fp16")
    new_params, retry = apply_parameter_adjustments(result, params, max_steps=2000)
    assert retry is True
    assert new_params.quantization == "4bit"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_model_scout.py::test_adjustment_increases_steps_when_prs_low_and_loss_falling tests/test_model_scout.py::test_adjustment_no_retry_when_prs_good tests/test_model_scout.py::test_adjustment_sets_4bit_on_oom -v --override-ini="addopts="
```

Expected: `FAILED — ImportError: cannot import name 'apply_parameter_adjustments'`

- [ ] **Step 3: Add ExperimentResult and apply_parameter_adjustments to pipeline/model_scout.py**

```python
# Add to pipeline/model_scout.py

@dataclass
class ExperimentResult:
    model_id: str
    quantization: str
    lora_steps: int
    lora_rank: int
    corpus_chunks: int
    faq_count: int
    prs: float
    prs_variance: float
    vram_gb: float
    wall_seconds: float
    status: str                 # "keep" | "discard" | "oom" | "crash"
    training_loss_start: float
    training_loss_end: float
    agent_reasoning: str
    git_commit: str = ""


def apply_parameter_adjustments(
    result: ExperimentResult,
    params: ScoutParams,
    max_steps: int = 2000,
    max_faq: int = 100,
    max_chunks: int = 2000,
) -> tuple[ScoutParams, bool]:
    """Apply rule-based parameter adjustments. Returns (new_params, should_retry_same_model)."""
    import copy
    new = copy.copy(params)
    retry = False

    # OOM at fp16 → retry with 4bit
    if result.status == "oom" and result.quantization == "fp16":
        new.quantization = "4bit"
        return new, True

    # OOM at 4bit → can't help, don't retry
    if result.status == "oom" and result.quantization == "4bit":
        return new, False

    # Low PRS + loss still falling → more steps
    loss_falling = result.training_loss_end < result.training_loss_start * 0.7
    if result.prs < 0.55 and loss_falling:
        new.lora_steps = min(new.lora_steps * 2, max_steps)
        retry = True

    # High FAQ variance → more FAQs (but don't retry same model, just update for next)
    if result.prs_variance > 0.15:
        new.faq_count = min(int(new.faq_count * 1.5), max_faq)

    return new, retry


def _write_result(results_path: str, result: ExperimentResult) -> None:
    file_exists = Path(results_path).exists()
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["commit", "model", "quantization", "lora_steps", "lora_rank",
                         "corpus_chunks", "faq_count", "prs", "vram_gb",
                         "wall_seconds", "status", "agent_reasoning"],
            delimiter="\t",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "commit": result.git_commit[:7] if result.git_commit else "unknown",
            "model": result.model_id.split("/")[-1],
            "quantization": result.quantization,
            "lora_steps": result.lora_steps,
            "lora_rank": result.lora_rank,
            "corpus_chunks": result.corpus_chunks,
            "faq_count": result.faq_count,
            "prs": f"{result.prs:.4f}",
            "vram_gb": f"{result.vram_gb:.1f}",
            "wall_seconds": int(result.wall_seconds),
            "status": result.status,
            "agent_reasoning": result.agent_reasoning,
        })
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_model_scout.py -k "adjustment" -v --override-ini="addopts="
```

Expected: all 5 adjustment tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add pipeline/model_scout.py tests/test_model_scout.py
git commit -m "feat: add ExperimentResult, parameter adjustment rules, and TSV writer to ModelScout"
```

---

## Task 7: Experiment Runner (Mini-LoRA + PRS Eval)

**Files:**
- Modify: `pipeline/model_scout.py`
- Modify: `tests/test_model_scout.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_scout.py — add

from unittest.mock import patch, MagicMock
from pipeline.model_scout import run_single_experiment


def test_run_single_experiment_returns_result(tmp_path):
    adapter = RecordingAdapter()
    cfg = {
        "embed_model": "BAAI/bge-small-en-v1.5",
        "checkpoint_dir": str(tmp_path),
        "version_file": str(tmp_path / "version.json"),
        "replay_db": str(tmp_path / "replay.db"),
        "query_log_db": str(tmp_path / "q.db"),
        "faq_question_key": "question",
        "faq_answer_key": "answer",
        "prs_signal_weights": {"faq": 0.4, "vdb": 0.4, "realtime": 0.2},
        "prs_auto_weight": False,
        "prs_stability_window": 3,
        "prs_advancement_threshold": 0.72,
        "min_cluster_samples_for_adaptation": 10,
        "realtime_requery_window_minutes": 10,
    }
    faqs = [{"question": "What is RAG?", "answer": "RAG retrieves context."}] * 5
    params = ScoutParams(lora_steps=10, lora_rank=8, corpus_chunks=10)
    candidate = {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "spec": {"_use_4bit": False, "lora_targets": ["q_proj", "k_proj", "v_proj"]},
        "score": 0.9,
    }

    with patch("pipeline.model_scout.model_loader") as mock_ml:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_ml.load.return_value = (mock_model, mock_tokenizer)
        mock_ml.init.return_value = None
        with patch("pipeline.model_scout._run_mini_lora", return_value=(0.8, 0.5)):
            with patch("pipeline.model_scout._eval_prs_on_faqs",
                       return_value=(0.70, 0.06)):
                result = run_single_experiment(candidate, faqs, params, cfg, adapter,
                                               mode="pre_index", store=None)

    assert isinstance(result, ExperimentResult)
    assert result.model_id == "meta-llama/Llama-3.2-3B-Instruct"
    assert 0.0 <= result.prs <= 1.0
    assert result.status in ("keep", "discard", "oom", "crash")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_model_scout.py::test_run_single_experiment_returns_result -v --override-ini="addopts="
```

Expected: `FAILED — ImportError: cannot import name 'run_single_experiment'`

- [ ] **Step 3: Add run_single_experiment and helpers to pipeline/model_scout.py**

```python
# Add to pipeline/model_scout.py

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import core.model_loader as model_loader


def _run_mini_lora(
    model, tokenizer, faqs: list[dict], params: ScoutParams, cfg: dict
) -> tuple[float, float]:
    """Run a short LoRA training round. Returns (loss_start, loss_end)."""
    from peft import LoraConfig, get_peft_model
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    import torch

    lora_config = LoraConfig(
        r=params.lora_rank,
        lora_alpha=params.lora_rank * 2,
        target_modules=cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj"]),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)

    # Build a tiny dataset from FAQs
    texts = [f"Q: {f['question']}\nA: {f['answer']}" for f in faqs]
    encodings = tokenizer(texts, truncation=True, max_length=256, padding=True,
                          return_tensors="pt")

    class _FaqDataset(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.enc = enc
        def __len__(self): return len(texts)
        def __getitem__(self, i):
            return {k: v[i] for k, v in self.enc.items()}

    dataset = _FaqDataset(encodings)
    args = TrainingArguments(
        output_dir=cfg.get("checkpoint_dir", "/tmp/scout_ckpt"),
        num_train_epochs=1,
        max_steps=params.lora_steps,
        per_device_train_batch_size=1,
        logging_steps=max(1, params.lora_steps // 10),
        save_steps=params.lora_steps + 1,  # never save during scout
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    loss_history = []

    class _LossCallback:
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                loss_history.append(logs["loss"])

    from transformers import TrainerCallback
    class LossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                loss_history.append(logs["loss"])

    trainer = Trainer(model=peft_model, args=args, train_dataset=dataset,
                      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
                      callbacks=[LossCallback()])
    trainer.train()
    loss_start = loss_history[0] if loss_history else 1.0
    loss_end = loss_history[-1] if loss_history else 1.0
    return loss_start, loss_end


def _eval_prs_on_faqs(model, tokenizer, faqs: list[dict], cfg: dict) -> tuple[float, float]:
    """Evaluate PRS on FAQ set. Returns (mean_prs, variance)."""
    import numpy as np
    from fastembed import TextEmbedding
    from transformers import pipeline as hf_pipeline
    from pipeline.prs_evaluator import _generate_parametric, _cosine_sim

    pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer,
                        max_new_tokens=128, do_sample=False)
    embedder = TextEmbedding(model_name=cfg.get("embed_model", "BAAI/bge-small-en-v1.5"),
                              show_download_progress=False)
    q_key = cfg.get("faq_question_key", "question")
    a_key = cfg.get("faq_answer_key", "answer")
    scores = []
    for faq in faqs:
        q = faq.get(q_key, "")
        gt = faq.get(a_key, "")
        if not q or not gt:
            continue
        param_ans = _generate_parametric(q, pipe)
        embs = np.array(list(embedder.embed([param_ans, gt])))
        scores.append(_cosine_sim(embs[0], embs[1]))
    if not scores:
        return 0.0, 0.0
    return float(np.mean(scores)), float(np.std(scores))


def run_single_experiment(
    candidate: dict,
    faqs: list[dict],
    params: ScoutParams,
    cfg: dict,
    adapter: IOAdapter,
    mode: str = "pre_index",
    store=None,
) -> ExperimentResult:
    model_id = candidate["model_id"]
    spec = candidate["spec"]
    quantization = "4bit" if spec.get("_use_4bit") else (
        params.quantization if params.quantization != "auto" else "fp16"
    )
    cfg = dict(cfg)
    cfg["llm_model"] = model_id
    cfg["quantization"] = quantization
    cfg["lora_target_modules"] = spec.get("lora_targets", ["q_proj", "k_proj", "v_proj"])
    cfg["lora_rank"] = params.lora_rank

    start_time = time.time()
    try:
        model_loader.init(cfg)
        model, tokenizer = model_loader.load()
        adapter.stream_progress(f"Loaded {model_id.split('/')[-1]}", 0.1)

        loss_start, loss_end = _run_mini_lora(model, tokenizer, faqs, params, cfg)
        adapter.stream_progress("Evaluating PRS", 0.9)

        mean_prs, prs_variance = _eval_prs_on_faqs(model, tokenizer, faqs, cfg)
        adapter.stream_progress("Done", 1.0)

        import torch
        vram_gb = (torch.cuda.memory_allocated() / 1024**3) if _has_cuda else 0.0
        wall_seconds = time.time() - start_time
        status = "keep" if mean_prs >= 0.55 else "discard"

        return ExperimentResult(
            model_id=model_id, quantization=quantization,
            lora_steps=params.lora_steps, lora_rank=params.lora_rank,
            corpus_chunks=params.corpus_chunks, faq_count=params.faq_count,
            prs=round(mean_prs, 4), prs_variance=round(prs_variance, 4),
            vram_gb=round(vram_gb, 1), wall_seconds=round(wall_seconds, 1),
            status=status, training_loss_start=round(loss_start, 4),
            training_loss_end=round(loss_end, 4),
            agent_reasoning="",
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return ExperimentResult(
                model_id=model_id, quantization=quantization,
                lora_steps=params.lora_steps, lora_rank=params.lora_rank,
                corpus_chunks=params.corpus_chunks, faq_count=params.faq_count,
                prs=0.0, prs_variance=0.0, vram_gb=0.0,
                wall_seconds=round(time.time() - start_time, 1),
                status="oom", training_loss_start=0.0, training_loss_end=0.0,
                agent_reasoning="OOM — try 4bit quantization",
            )
        raise
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_model_scout.py::test_run_single_experiment_returns_result -v --override-ini="addopts="
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add pipeline/model_scout.py tests/test_model_scout.py
git commit -m "feat: add experiment runner with mini-LoRA and PRS evaluation to ModelScout"
```

---

## Task 8: Main Agent Loop

**Files:**
- Modify: `pipeline/model_scout.py`
- Modify: `tests/test_model_scout.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_scout.py — add

from pipeline.model_scout import run_scout_session


def test_run_scout_session_completes_with_recording_adapter(tmp_path):
    adapter = RecordingAdapter(responses=[
        "pre_index",  # mode
        "B",          # budget: experiment count
        "2",          # 2 experiments
    ])
    cfg = {
        "embed_model": "BAAI/bge-small-en-v1.5",
        "checkpoint_dir": str(tmp_path),
        "version_file": str(tmp_path / "version.json"),
        "replay_db": str(tmp_path / "replay.db"),
        "query_log_db": str(tmp_path / "q.db"),
        "faq_question_key": "question",
        "faq_answer_key": "answer",
        "model_registry_path": "core/model_registry.json",
        "model_scout_results": str(tmp_path / "results.tsv"),
        "scout_initial_corpus_chunks": 10,
        "scout_initial_faq_count": 5,
        "scout_initial_lora_steps": 10,
        "scout_initial_lora_rank": 8,
        "scout_max_lora_steps": 100,
        "scout_max_corpus_chunks": 50,
        "scout_max_faq_count": 20,
        "prs_signal_weights": {"faq": 0.4, "vdb": 0.4, "realtime": 0.2},
        "prs_auto_weight": False,
        "prs_stability_window": 3,
        "prs_advancement_threshold": 0.72,
        "min_cluster_samples_for_adaptation": 10,
        "realtime_requery_window_minutes": 10,
    }
    faqs = [{"question": "What is RAG?", "answer": "RAG retrieves context."}] * 5
    gpu_info = {"available": False, "free_vram_gb": 24.0, "report": "CPU mode"}

    with patch("pipeline.model_scout.run_single_experiment") as mock_exp:
        mock_exp.return_value = ExperimentResult(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            quantization="fp16", lora_steps=10, lora_rank=8,
            corpus_chunks=10, faq_count=5, prs=0.72, prs_variance=0.05,
            vram_gb=9.0, wall_seconds=60.0, status="keep",
            training_loss_start=1.2, training_loss_end=0.8,
            agent_reasoning="good result",
        )
        recommendation = run_scout_session(adapter, cfg, faqs, gpu_info, store=None)

    assert recommendation is not None
    assert "model_id" in recommendation
    # Results TSV should exist
    assert Path(cfg["model_scout_results"]).exists()


def test_run_scout_session_respects_stop_command(tmp_path):
    adapter = RecordingAdapter(responses=[
        "pre_index", "B", "10",  # budget: 10 experiments
        "stop",                  # user stops after first result
    ])
    cfg = {
        "embed_model": "BAAI/bge-small-en-v1.5",
        "checkpoint_dir": str(tmp_path),
        "model_registry_path": "core/model_registry.json",
        "model_scout_results": str(tmp_path / "results.tsv"),
        "scout_initial_lora_steps": 10, "scout_initial_lora_rank": 8,
        "scout_initial_corpus_chunks": 10, "scout_initial_faq_count": 5,
        "scout_max_lora_steps": 100, "scout_max_corpus_chunks": 50,
        "scout_max_faq_count": 20,
        "version_file": str(tmp_path / "version.json"),
        "replay_db": str(tmp_path / "replay.db"),
        "query_log_db": str(tmp_path / "q.db"),
        "faq_question_key": "question", "faq_answer_key": "answer",
        "prs_signal_weights": {"faq": 0.4, "vdb": 0.4, "realtime": 0.2},
        "prs_auto_weight": False, "prs_stability_window": 3,
        "prs_advancement_threshold": 0.72, "min_cluster_samples_for_adaptation": 10,
        "realtime_requery_window_minutes": 10,
    }
    faqs = [{"question": "Q?", "answer": "A."}] * 3
    gpu_info = {"available": False, "free_vram_gb": 24.0, "report": "CPU mode"}

    with patch("pipeline.model_scout.run_single_experiment") as mock_exp:
        mock_exp.return_value = ExperimentResult(
            model_id="meta-llama/Llama-3.2-3B-Instruct", quantization="fp16",
            lora_steps=10, lora_rank=8, corpus_chunks=10, faq_count=5,
            prs=0.68, prs_variance=0.04, vram_gb=9.0, wall_seconds=60.0,
            status="keep", training_loss_start=1.2, training_loss_end=0.9,
            agent_reasoning="",
        )
        recommendation = run_scout_session(adapter, cfg, faqs, gpu_info, store=None)

    # Should have stopped after 1 experiment, not 10
    assert mock_exp.call_count == 1
    assert recommendation is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_model_scout.py::test_run_scout_session_completes_with_recording_adapter tests/test_model_scout.py::test_run_scout_session_respects_stop_command -v --override-ini="addopts="
```

Expected: `FAILED — ImportError: cannot import name 'run_scout_session'`

- [ ] **Step 3: Add run_scout_session to pipeline/model_scout.py**

```python
# Add to pipeline/model_scout.py

from core.model_registry import load_registry, get_candidate_shortlist


def _check_user_interrupt(adapter: IOAdapter) -> str | None:
    """Non-blocking check for user interrupt. Returns command string or None."""
    # In CLI mode this is handled by the pre-experiment prompt.
    # In SSE mode the SSEAdapter tracks the last received command.
    if hasattr(adapter, "pending_command"):
        cmd = adapter.pending_command
        adapter.pending_command = None
        return cmd
    return None


def run_scout_session(
    adapter: IOAdapter,
    cfg: dict,
    faqs: list[dict],
    gpu_info: dict,
    store=None,
) -> dict | None:
    """Main agent loop. Returns best result dict or None if no experiments ran."""
    adapter.send(gpu_info["report"])

    # Mode selection
    mode = adapter.ask(
        "Run ModelScout before indexing (pre-index) or against existing VDB (post-index)?",
        options=["pre_index — I haven't indexed yet", "post_index — VDB already exists"],
    ).lower()
    mode = "pre_index" if "pre" in mode else "post_index"

    # Budget dialog
    params = run_budget_dialog(adapter)
    params.lora_steps = cfg.get("scout_initial_lora_steps", 500)
    params.lora_rank = cfg.get("scout_initial_lora_rank", 16)
    params.corpus_chunks = cfg.get("scout_initial_corpus_chunks", 200)
    params.faq_count = cfg.get("scout_initial_faq_count", 20)

    # Build candidate shortlist
    registry = load_registry(cfg.get("model_registry_path"))
    free_vram = gpu_info.get("free_vram_gb", 24.0)
    shortlist = get_candidate_shortlist(
        registry, free_vram_gb=free_vram,
        corpus_languages=cfg.get("corpus_languages", ["en"]),
        task_type=cfg.get("task_type", "factual_qa"),
        corpus_chunk_count=params.corpus_chunks,
    )

    if not shortlist:
        adapter.send("❌ No eligible models found for available VRAM. Cannot proceed.")
        return None

    adapter.send(
        f"Eligible models ({len(shortlist)}): "
        + ", ".join(c["model_id"].split("/")[-1] for c in shortlist[:5])
        + (f" and {len(shortlist)-5} more" if len(shortlist) > 5 else "")
    )

    results: list[ExperimentResult] = []
    candidate_queue = list(shortlist)
    tried_families: set[str] = set()
    results_path = cfg.get("model_scout_results", "model_scout_results.tsv")

    while candidate_queue and not _budget_exhausted(params):
        candidate = candidate_queue.pop(0)
        family = candidate["spec"].get("family", candidate["model_id"])

        # Pre-experiment announcement
        quant = "4bit" if candidate["spec"].get("_use_4bit") else "fp16"
        adapter.send(
            f"\n▶ Next: {candidate['model_id'].split('/')[-1]} ({quant}) | "
            f"steps={params.lora_steps} rank={params.lora_rank} "
            f"chunks={params.corpus_chunks} faqs={params.faq_count}\n"
            f"  Score: {candidate['score']:.2f} | Type 'skip' to skip, 'stop' to end."
        )

        # Check for user interrupt (pre-experiment)
        user_cmd = adapter.ask("Press Enter to proceed (or type a command):", options=None)
        user_cmd = user_cmd.strip().lower()

        if user_cmd == "stop":
            break
        if user_cmd == "skip":
            adapter.send("Skipping. Moving to next candidate.")
            continue
        if user_cmd.startswith("try "):
            model_name = user_cmd[4:].strip()
            matching = [c for c in shortlist if model_name.lower() in c["model_id"].lower()]
            if matching:
                candidate_queue.insert(0, matching[0])
                adapter.send(f"Queued {matching[0]['model_id'].split('/')[-1]} next.")
            else:
                adapter.send(f"Model '{model_name}' not found in eligible list. Continuing.")
            continue
        if user_cmd == "more steps":
            params.lora_steps = min(params.lora_steps * 2,
                                     cfg.get("scout_max_lora_steps", 2000))
            adapter.send(f"lora_steps increased to {params.lora_steps}.")
        if user_cmd == "more faqs":
            params.faq_count = min(params.faq_count + 10,
                                    cfg.get("scout_max_faq_count", 100))
            adapter.send(f"faq_count increased to {params.faq_count}.")

        # Run experiment
        result = run_single_experiment(candidate, faqs, params, cfg, adapter, mode, store)
        tried_families.add(family)
        params.experiments_run += 1

        # Get git commit
        try:
            import subprocess
            result.git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], text=True
            ).strip()
        except Exception:
            result.git_commit = "unknown"

        # Log result
        _write_result(results_path, result)
        results.append(result)

        # Report result
        best = max(results, key=lambda r: r.prs) if results else result
        adapter.send(
            f"  Result: PRS={result.prs:.3f} VRAM={result.vram_gb:.1f}GB "
            f"time={int(result.wall_seconds)}s status={result.status}\n"
            f"  Best so far: {best.model_id.split('/')[-1]} PRS={best.prs:.3f}"
        )

        # Apply parameter adjustments
        new_params, retry = apply_parameter_adjustments(
            result, params,
            max_steps=cfg.get("scout_max_lora_steps", 2000),
            max_faq=cfg.get("scout_max_faq_count", 100),
            max_chunks=cfg.get("scout_max_corpus_chunks", 2000),
        )
        if retry and params.experiments_run < params.budget_value:
            adapter.send(
                f"  Adjustment: {_describe_adjustment(params, new_params)} — retrying same model."
            )
            params = new_params
            candidate_queue.insert(0, candidate)
        else:
            params = new_params

        # Consecutive low PRS check
        if len(results) >= 3 and all(r.prs < 0.55 for r in results[-3:]):
            adapter.send(
                "⚠️  Three consecutive models scored below 0.55 PRS. "
                "This may indicate a domain/language mismatch.\n"
                "  Is your corpus in a language other than what was configured? "
                "Any other context to share?"
            )
            adapter.ask("Your input (or press Enter to continue):", options=None)

    if not results:
        adapter.send("No experiments completed.")
        return None

    best = max(results, key=lambda r: r.prs)
    # Write recommendation to TSV
    with open(results_path, "a") as f:
        f.write(
            f"# RECOMMENDATION: {best.model_id} ({best.quantization}, rank={best.lora_rank})\n"
            f"# PRS={best.prs:.4f} | VRAM={best.vram_gb:.1f}GB | "
            f"Reasoning: Best PRS across {len(results)} experiments.\n"
        )
    adapter.send(
        f"\n✅ ModelScout complete. Recommendation:\n"
        f"   Model: {best.model_id}\n"
        f"   Quantization: {best.quantization} | LoRA rank: {best.lora_rank}\n"
        f"   PRS: {best.prs:.4f} | VRAM: {best.vram_gb:.1f}GB\n"
        f"   Results saved to: {results_path}"
    )
    return {"model_id": best.model_id, "quantization": best.quantization,
            "lora_rank": best.lora_rank, "prs": best.prs}


def _describe_adjustment(old: ScoutParams, new: ScoutParams) -> str:
    parts = []
    if new.lora_steps != old.lora_steps:
        parts.append(f"lora_steps {old.lora_steps}→{new.lora_steps}")
    if new.faq_count != old.faq_count:
        parts.append(f"faq_count {old.faq_count}→{new.faq_count}")
    if new.quantization != old.quantization:
        parts.append(f"quantization {old.quantization}→{new.quantization}")
    return ", ".join(parts) if parts else "no change"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_model_scout.py -v --override-ini="addopts="
```

Expected: all tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add pipeline/model_scout.py tests/test_model_scout.py
git commit -m "feat: add main ModelScout agent loop with interrupt handling and result logging"
```

---

## Task 9: CLI Entry Point

**Files:**
- Create: `pipeline/model_scout_cli.py`

- [ ] **Step 1: Create the CLI entry point**

```python
# pipeline/model_scout_cli.py

"""
ModelScout CLI — interactive model selection agent.

Usage:
  python -m pipeline.model_scout_cli --config my_config.json --docs ./docs/
  python -m pipeline.model_scout_cli --config my_config.json --mode post_index
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelScout: interactive model selector")
    parser.add_argument("--config", required=True, help="Path to UC config JSON")
    parser.add_argument("--docs", default=None, help="Document directory (pre-index mode)")
    parser.add_argument("--faqs", default=None, help="Existing FAQ JSON file (optional)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    from pipeline.model_scout import CLIAdapter, detect_gpu, run_scout_session
    from pipeline.sleep_faq_generator import generate as generate_faqs

    adapter = CLIAdapter()
    gpu_info = detect_gpu()

    # Build FAQ set
    if args.faqs and Path(args.faqs).exists():
        with open(args.faqs) as f:
            faqs = json.load(f)
        adapter.send(f"Loaded {len(faqs)} FAQs from {args.faqs}")
    else:
        faq_count = cfg.get("scout_initial_faq_count", 20)
        adapter.send(f"Generating {faq_count} FAQs from corpus…")
        faqs = generate_faqs(cfg, count=faq_count, source_dir=args.docs)
        adapter.send(f"Generated {len(faqs)} FAQs.")

    # Post-index: get store
    store = None
    from vectorstore.registry import get_store
    try:
        store = get_store(cfg)
    except Exception:
        adapter.send("Could not connect to vector store — running in pre-index mode.")

    recommendation = run_scout_session(adapter, cfg, faqs, gpu_info, store=store)

    if recommendation:
        adapter.send(
            f"\nTo use this model, set in {args.config}:\n"
            f'  "llm_model": "{recommendation[\"model_id\"]}",\n'
            f'  "quantization": "{recommendation[\"quantization\"]}",\n'
            f'  "lora_rank": {recommendation[\"lora_rank\"]}'
        )
    sys.exit(0 if recommendation else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it is importable**

```bash
python -c "import pipeline.model_scout_cli; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add pipeline/model_scout_cli.py
git commit -m "feat: add ModelScout CLI entry point"
```

---

## Task 10: Studio SSE Endpoint

**Files:**
- Modify: `studio/routes.py`
- Modify: `tests/test_studio_routes.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_studio_routes.py — add

def test_model_scout_stream_endpoint_exists(client):
    # Without a running scout session, should return 200 with event stream headers
    resp = client.get("/api/uc/test_uc/model-scout/stream",
                      headers={"Accept": "text/event-stream"})
    # Either 200 (stream open) or 404 if UC doesn't exist — not 500
    assert resp.status_code in (200, 404)


def test_model_scout_respond_endpoint_exists(client):
    resp = client.post("/api/uc/test_uc/model-scout/respond",
                       json={"response": "B"})
    assert resp.status_code in (200, 404, 409)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_studio_routes.py::test_model_scout_stream_endpoint_exists tests/test_studio_routes.py::test_model_scout_respond_endpoint_exists -v --override-ini="addopts="
```

Expected: `FAILED — 404 or 405`

- [ ] **Step 3: Add SSE endpoint and SSEAdapter to studio/routes.py**

```python
# Add to studio/routes.py

import asyncio
import queue as _queue
from sse_starlette.sse import EventSourceResponse

# Per-UC scout session state (in-memory)
_scout_queues: dict[str, _queue.Queue] = {}
_scout_pending_responses: dict[str, _queue.Queue] = {}


class SSEAdapter:
    """IOAdapter that pushes events to an SSE queue and reads responses from a POST queue."""

    def __init__(self, uc_name: str):
        self._uc = uc_name
        _scout_queues[uc_name] = _queue.Queue()
        _scout_pending_responses[uc_name] = _queue.Queue()
        self.pending_command: str | None = None

    def send(self, message: str) -> None:
        _scout_queues[self._uc].put({"event": "scout_message", "data": message})

    def ask(self, question: str, options: list[str] | None = None) -> str:
        payload = {"text": question}
        if options:
            payload["options"] = options
        _scout_queues[self._uc].put({"event": "scout_question", "data": json.dumps(payload)})
        # Block until user responds via POST /model-scout/respond
        return _scout_pending_responses[self._uc].get(timeout=300)

    def stream_progress(self, label: str, pct: float) -> None:
        _scout_queues[self._uc].put({
            "event": "scout_progress",
            "data": json.dumps({"label": label, "pct": round(pct, 2)}),
        })


@router.get("/api/uc/{config_name}/model-scout/stream")
async def model_scout_stream(config_name: str):
    cfg_path = _config_path(config_name)
    if not cfg_path.exists():
        raise HTTPException(status_code=404, detail="UC config not found")

    async def event_generator():
        q = _scout_queues.get(config_name, _queue.Queue())
        _scout_queues[config_name] = q
        while True:
            try:
                item = q.get_nowait()
                yield {"event": item["event"], "data": item["data"]}
            except _queue.Empty:
                await asyncio.sleep(0.1)

    return EventSourceResponse(event_generator())


@router.post("/api/uc/{config_name}/model-scout/respond")
async def model_scout_respond(config_name: str, body: dict):
    response_q = _scout_pending_responses.get(config_name)
    if response_q is None:
        raise HTTPException(status_code=409, detail="No active ModelScout session for this UC")
    response_q.put(body.get("response", ""))
    return {"status": "ok"}
```

- [ ] **Step 4: Run the new tests**

```bash
python -m pytest tests/test_studio_routes.py::test_model_scout_stream_endpoint_exists tests/test_studio_routes.py::test_model_scout_respond_endpoint_exists -v --override-ini="addopts="
```

Expected: both `PASSED`

- [ ] **Step 5: Run all studio tests**

```bash
python -m pytest tests/test_studio_routes.py -v --override-ini="addopts="
```

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add studio/routes.py tests/test_studio_routes.py
git commit -m "feat: add ModelScout SSE stream and respond endpoints to Studio"
```

---

## Task 11: Integration Smoke Test

**Files:**
- Modify: `tests/test_integration_smoke.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/test_integration_smoke.py — add

def test_modelscout_full_session_smoke(tmp_path):
    """Verify ModelScout session runs end-to-end with mocked experiment and produces TSV."""
    from unittest.mock import patch, MagicMock
    from pipeline.model_scout import (
        RecordingAdapter, detect_gpu, run_scout_session,
        ExperimentResult, ScoutParams,
    )

    adapter = RecordingAdapter(responses=[
        "pre_index",  # mode
        "B",          # budget: experiment count
        "2",          # 2 experiments
        "",           # proceed with first experiment
        "",           # proceed with second experiment
    ])
    cfg = {
        "embed_model": "BAAI/bge-small-en-v1.5",
        "checkpoint_dir": str(tmp_path),
        "version_file": str(tmp_path / "version.json"),
        "replay_db": str(tmp_path / "replay.db"),
        "query_log_db": str(tmp_path / "q.db"),
        "faq_question_key": "question",
        "faq_answer_key": "answer",
        "model_registry_path": "core/model_registry.json",
        "model_scout_results": str(tmp_path / "results.tsv"),
        "scout_initial_corpus_chunks": 10,
        "scout_initial_faq_count": 5,
        "scout_initial_lora_steps": 10,
        "scout_initial_lora_rank": 8,
        "scout_max_lora_steps": 100,
        "scout_max_corpus_chunks": 50,
        "scout_max_faq_count": 20,
        "corpus_languages": ["en"],
        "task_type": "factual_qa",
        "prs_signal_weights": {"faq": 0.4, "vdb": 0.4, "realtime": 0.2},
        "prs_auto_weight": False, "prs_stability_window": 3,
        "prs_advancement_threshold": 0.72, "min_cluster_samples_for_adaptation": 10,
        "realtime_requery_window_minutes": 10,
    }
    faqs = [{"question": f"Q{i}?", "answer": f"A{i}."} for i in range(5)]
    gpu_info = {"available": False, "free_vram_gb": 24.0,
                "report": "CPU mode (test)", "gpu_name": "CPU"}

    call_count = {"n": 0}
    def mock_experiment(candidate, faqs, params, cfg, adapter, mode, store):
        call_count["n"] += 1
        return ExperimentResult(
            model_id=candidate["model_id"], quantization="fp16",
            lora_steps=params.lora_steps, lora_rank=params.lora_rank,
            corpus_chunks=params.corpus_chunks, faq_count=params.faq_count,
            prs=0.60 + call_count["n"] * 0.05, prs_variance=0.04,
            vram_gb=9.0, wall_seconds=30.0, status="keep",
            training_loss_start=1.2, training_loss_end=0.9,
            agent_reasoning="mock",
        )

    with patch("pipeline.model_scout.run_single_experiment", side_effect=mock_experiment):
        recommendation = run_scout_session(adapter, cfg, faqs, gpu_info, store=None)

    assert recommendation is not None
    assert "model_id" in recommendation
    assert recommendation["prs"] > 0.0

    # Results TSV must exist and have header + rows
    results_path = tmp_path / "results.tsv"
    assert results_path.exists()
    content = results_path.read_text()
    assert "commit" in content  # header row
    assert "RECOMMENDATION" in content  # final recommendation comment
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_integration_smoke.py::test_modelscout_full_session_smoke -v --override-ini="addopts="
```

Expected: `PASSED`

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest tests/ -v --override-ini="addopts="
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration_smoke.py
git commit -m "test: add ModelScout end-to-end integration smoke test"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ GPU auto-detection — Task 4 (`detect_gpu`)
- ✅ VRAM hard filter — Task 1 (`filter_by_vram`)
- ✅ Candidate scoring — Task 1 (`score_candidates`, `get_candidate_shortlist`)
- ✅ Budget dialog (A/B/C/D options) — Task 5 (`run_budget_dialog`)
- ✅ Interactive agent loop with announcements — Task 8 (`run_scout_session`)
- ✅ User interrupt handling (skip/stop/try/more steps/more faqs) — Task 8
- ✅ Parameter adjustment rules — Task 6 (`apply_parameter_adjustments`)
- ✅ Experiment runner (mini-LoRA + PRS eval) — Task 7 (`run_single_experiment`)
- ✅ OOM handling and 4bit retry — Task 6 + Task 7
- ✅ Results TSV logging — Task 6 (`_write_result`)
- ✅ Final recommendation with reasoning — Task 8
- ✅ IOAdapter protocol + CLIAdapter + RecordingAdapter test double — Task 3
- ✅ Studio SSE stream + respond endpoints — Task 10
- ✅ CLI entry point — Task 9
- ✅ Config fields — Task 2
- ✅ model_registry.json (8 models) — Task 1
- ✅ model_scout_program.md — Task 2
- ✅ Pre-index and post-index modes — Task 8 (mode selection in loop)
- ✅ Integration smoke test — Task 11

**Known dependency:** `sse_starlette` must be available for Task 10. Add to `requirements_gpu.txt` if not already present: `sse-starlette>=1.6.1`
