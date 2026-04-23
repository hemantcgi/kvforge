"""Brownfield per-chunk confidence scoring for KVForge dynamic PRS.

In brownfield deployments (existing VDB without cluster assignments), this
module scores each chunk by asking the model to answer a question derived from
its text and comparing the answer to the chunk via cosine similarity.

Chunks whose ``confidence_lora_version`` equals the current LoRA version are
considered fresh and skipped.  All others are re-scored after each LoRA round.

Public API
----------
* ``score_chunk(chunk_text, model, tokenizer, embed_model)`` → float in [0, 1].
* ``get_eligible_chunks(points, current_lora_version)`` → filtered list.
* ``brownfield_coverage_stats(points, confidence_floor)`` → stats dict.
* ``run_brownfield_scoring(store, model, tokenizer, ...)`` → stats dict.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    from fastembed import TextEmbedding
except ImportError:  # pragma: no cover
    TextEmbedding = None  # type: ignore[assignment,misc]

try:
    from transformers import pipeline as hf_pipeline
except ImportError:  # pragma: no cover
    hf_pipeline = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Internal helpers (defined locally so tests can monkeypatch via
# pipeline.chunk_confidence._generate_parametric / _cosine_sim)
# ---------------------------------------------------------------------------


def _generate_parametric(question: str, pipe) -> str:
    """Generate a parametric answer using a HuggingFace text-generation pipeline."""
    out = pipe(question)
    return out[0]["generated_text"][len(question):].strip()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a_norm, b_norm))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_chunk(
    chunk_text: str,
    model,
    tokenizer,
    embed_model: str,
    accuracy_threshold: float = 0.0,
) -> float:
    """Estimate how well the LLM can answer a question derived from *chunk_text*.

    The procedure is:
    1. Derive a question by prompting "Summarize the key information in: <chunk[:300]>".
    2. Ask the model to generate a parametric answer (no retrieved context).
    3. Embed both the generated answer and the original chunk text.
    4. Return their cosine similarity as the confidence score.

    Args:
        chunk_text: The raw text of the chunk to score.
        model: HuggingFace model object.
        tokenizer: HuggingFace tokenizer.
        embed_model: Model name for the fastembed ``TextEmbedding`` embedder.
        accuracy_threshold: Unused in the score computation; reserved for
            downstream filtering (kept for API compatibility).

    Returns:
        Cosine similarity in [0, 1] (higher = model knows the content).
    """
    question = f"Summarize the key information in: {chunk_text[:300]}"
    pipe = hf_pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=128, do_sample=False,
    )
    param_ans = _generate_parametric(question, pipe)
    embedder = TextEmbedding(model_name=embed_model, show_download_progress=False)
    embs = np.array(list(embedder.embed([param_ans, chunk_text])))
    return _cosine_sim(embs[0], embs[1])


def get_eligible_chunks(points: list[Any], current_lora_version: int) -> list[Any]:
    """Filter *points* to those whose confidence score is stale or missing.

    A chunk is eligible for re-scoring when its ``confidence_lora_version``
    payload field is ``None`` or less than *current_lora_version*.

    Args:
        points: List of VDB point objects with a ``.payload`` dict.
        current_lora_version: The LoRA round number to compare against.

    Returns:
        Subset of *points* that need re-scoring.
    """
    return [
        p for p in points
        if p.payload.get("confidence_lora_version") is None
        or p.payload["confidence_lora_version"] < current_lora_version
    ]


def brownfield_coverage_stats(points: list[Any], confidence_floor: float) -> dict:
    """Summarise how many chunks exceed the confidence floor.

    Args:
        points: All VDB point objects (typically retrieved via ``store.scroll``).
        confidence_floor: Minimum ``model_confidence`` to count as mastered.

    Returns:
        Dict with keys:

        * ``'coverage_pct'`` — fraction of chunks above floor (float in [0, 1]).
        * ``'total_chunks'`` — total number of points.
        * ``'mastered_chunks'`` — count of chunks above floor.
    """
    if not points:
        return {"coverage_pct": 0.0, "total_chunks": 0, "mastered_chunks": 0}
    mastered = sum(
        1 for p in points
        if p.payload.get("model_confidence") is not None
        and p.payload["model_confidence"] >= confidence_floor
    )
    return {
        "coverage_pct": mastered / len(points),
        "total_chunks": len(points),
        "mastered_chunks": mastered,
    }


def run_brownfield_scoring(
    store,
    model,
    tokenizer,
    embed_model: str,
    current_lora_version: int,
    cfg: dict,
) -> dict:
    """Score all stale chunks; update ``model_confidence`` in the VDB.

    This is intended to run as a background process after each LoRA round.
    Fresh chunks (``confidence_lora_version == current_lora_version``) are
    skipped to avoid redundant computation.

    Args:
        store: Vector store object with ``scroll(limit)`` and
            ``update_payload(id, payload)`` methods.
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        embed_model: Embedder model name.
        current_lora_version: LoRA round number (used for staleness check).
        cfg: Config dict; uses ``'brownfield_confidence_floor'`` (default 0.80).

    Returns:
        Dict from :func:`brownfield_coverage_stats` reflecting the full VDB state.
    """
    all_points = store.scroll(limit=10_000)
    eligible = get_eligible_chunks(all_points, current_lora_version)
    for point in eligible:
        text = point.payload.get("text", "")
        if not text:
            continue
        confidence = score_chunk(text, model, tokenizer, embed_model)
        store.update_payload(point.id, {
            "model_confidence": round(float(confidence), 4),
            "confidence_lora_version": current_lora_version,
        })
    return brownfield_coverage_stats(
        all_points, cfg.get("brownfield_confidence_floor", 0.80)
    )
