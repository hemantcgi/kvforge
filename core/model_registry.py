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
    """Load the model registry JSON. Returns a dict keyed by model_id."""
    p = Path(path) if path else _REGISTRY_PATH
    with open(p) as f:
        return json.load(f)


def filter_by_vram(registry: dict, free_vram_gb: float) -> dict:
    """Return only models that fit in available VRAM (fp16 or 4-bit + LoRA overhead)."""
    eligible = {}
    for model_id, spec in registry.items():
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
    """Score each eligible model 0–1 based on language match, domain affinity,
    corpus size fit, and VRAM headroom."""
    target_strengths = _TASK_AFFINITY.get(task_type, ["factual_qa"])
    scores = {}
    for model_id, spec in eligible.items():
        # Language match
        lang_score = 1.0 if any(lang in spec["languages"] for lang in corpus_languages) else 0.0
        if "multilingual" in spec["languages"] and lang_score == 0.0:
            lang_score = 0.6  # partial credit for multilingual models

        # Domain affinity
        domain_score = sum(
            1.0 for s in target_strengths if s in spec["strengths"]
        ) / max(len(target_strengths), 1)

        # Corpus size fit
        min_chunks = spec.get("min_corpus_chunks", 50)
        corpus_score = min(corpus_chunk_count / max(min_chunks, 1), 1.0)

        # VRAM headroom (prefer models that leave free VRAM)
        model_vram = (
            spec["vram_4bit_gb"] if spec.get("_use_4bit") else spec["vram_fp16_gb"]
        )
        vram_score = max(0.0, 1.0 - (model_vram / free_vram_gb))

        scores[model_id] = (
            0.40 * lang_score
            + 0.30 * domain_score
            + 0.20 * corpus_score
            + 0.10 * vram_score
        )
    return scores


def get_candidate_shortlist(
    registry: dict,
    free_vram_gb: float,
    corpus_languages: list[str],
    task_type: str,
    corpus_chunk_count: int,
) -> list[dict]:
    """Filter by VRAM, score, and return sorted shortlist dicts with model_id, score, spec."""
    eligible = filter_by_vram(registry, free_vram_gb)
    scores = score_candidates(
        eligible, corpus_languages, task_type, corpus_chunk_count, free_vram_gb
    )
    return sorted(
        [
            {"model_id": mid, "score": score, "spec": eligible[mid]}
            for mid, score in scores.items()
        ],
        key=lambda x: x["score"],
        reverse=True,
    )
