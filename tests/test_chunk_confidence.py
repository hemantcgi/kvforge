"""Tests for pipeline/chunk_confidence.py — brownfield per-chunk confidence scoring."""

from unittest.mock import MagicMock, patch
import pytest
from pipeline.chunk_confidence import (
    score_chunk,
    get_eligible_chunks,
    brownfield_coverage_stats,
)


def test_score_chunk_high_similarity():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = iter([[0.1, 0.2], [0.3, 0.4]])
    with patch("pipeline.chunk_confidence._generate_parametric", return_value="good answer"):
        with patch("pipeline.chunk_confidence._cosine_sim", return_value=0.88):
            with patch("pipeline.chunk_confidence.hf_pipeline", return_value=MagicMock()):
                with patch("pipeline.chunk_confidence.TextEmbedding", return_value=mock_embedder):
                    score = score_chunk(
                        "RAG retrieves context from a database.",
                        mock_model,
                        mock_tokenizer,
                        embed_model="BAAI/bge-small-en-v1.5",
                    )
    assert 0.0 <= score <= 1.0


def test_get_eligible_chunks_filters_stale(tmp_path):
    points = [
        MagicMock(id="a", payload={"confidence_lora_version": 1}),
        MagicMock(id="b", payload={"confidence_lora_version": None}),
        MagicMock(id="c", payload={"confidence_lora_version": 2}),
    ]
    eligible = get_eligible_chunks(points, current_lora_version=2)
    ids = [p.id for p in eligible]
    assert "b" in ids   # None → stale → eligible
    assert "a" in ids   # version 1 < 2 → stale → eligible
    assert "c" not in ids  # version 2 == current → fresh → skip


def test_brownfield_coverage_stats_all_above_floor():
    points = [
        MagicMock(payload={"model_confidence": 0.9}),
        MagicMock(payload={"model_confidence": 0.85}),
    ]
    stats = brownfield_coverage_stats(points, confidence_floor=0.80)
    assert stats["coverage_pct"] == 1.0
    assert stats["total_chunks"] == 2


def test_brownfield_coverage_stats_partial():
    points = [
        MagicMock(payload={"model_confidence": 0.9}),
        MagicMock(payload={"model_confidence": 0.5}),
        MagicMock(payload={"model_confidence": None}),
    ]
    stats = brownfield_coverage_stats(points, confidence_floor=0.80)
    assert abs(stats["coverage_pct"] - 1 / 3) < 1e-6


def test_brownfield_coverage_stats_empty():
    stats = brownfield_coverage_stats([], confidence_floor=0.80)
    assert stats["coverage_pct"] == 0.0
    assert stats["total_chunks"] == 0
