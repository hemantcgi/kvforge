# tests/test_model_registry.py

import pytest
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
    scores_zh = score_candidates(
        eligible,
        corpus_languages=["zh"],
        task_type="factual_qa",
        corpus_chunk_count=500,
        free_vram_gb=24.0,
    )
    # Qwen should rank higher than Llama-3.2-3B for a Chinese corpus
    qwen_ids = [k for k in eligible if "Qwen2.5-7B" in k]
    llama_ids = [k for k in eligible if "Llama-3.2-3B" in k]
    if qwen_ids and llama_ids:
        assert scores_zh[qwen_ids[0]] > scores_zh[llama_ids[0]]


def test_get_candidate_shortlist_returns_sorted_list():
    registry = load_registry()
    shortlist = get_candidate_shortlist(
        registry,
        free_vram_gb=24.0,
        corpus_languages=["en"],
        task_type="factual_qa",
        corpus_chunk_count=300,
    )
    assert len(shortlist) > 0
    # Verify sorted descending by score
    scores = [item["score"] for item in shortlist]
    assert scores == sorted(scores, reverse=True)
    assert "model_id" in shortlist[0]
    assert "spec" in shortlist[0]
