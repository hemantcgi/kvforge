import pytest
import numpy as np
from unittest.mock import MagicMock


def _chunk(cid, text, embedding, hit_count=0, kv_token_path=None, status="active"):
    return {
        "id": cid,
        "vector": embedding.tolist(),
        "payload": {
            "text": text,
            "kv_cache": "AAAA",
            "kv_token_path": kv_token_path,
            "status": status,
            "hit_count": hit_count,
        }
    }


def test_coverage_sweep_builds_map():
    from pipeline.corpus_curation import run_coverage_sweep
    faqs = ["What is RAG?", "How does KV cache work?"]
    chunks = [
        _chunk("c1", "RAG is retrieval...", np.array([1, 0, 0], dtype=np.float32)),
        _chunk("c2", "KV cache stores...", np.array([0, 1, 0], dtype=np.float32)),
        _chunk("c3", "Embeddings encode...", np.array([0, 0, 1], dtype=np.float32)),
    ]
    mock_embedder = MagicMock()
    mock_embedder.embed.side_effect = [
        np.array([[1, 0, 0]], dtype=np.float32),
        np.array([[0, 1, 0]], dtype=np.float32),
    ]
    result = run_coverage_sweep(faqs, chunks, mock_embedder, top_k=1)
    assert "c1" in result[0]
    assert "c2" in result[1]


def test_curation_identifies_enhanced_candidates():
    from pipeline.corpus_curation import identify_tier_actions
    from addons.corpus_intelligence.config import CorpusIntelligenceConfig
    cfg = CorpusIntelligenceConfig(enhanced_tier_threshold=0.6, archive_candidate_threshold=0.3)
    cis_scores    = {"high": 0.85, "mid": 0.50, "low_unique": 0.10, "low_dup": 0.15}
    unique_scores = {"high": 0.9, "mid": 0.5, "low_unique": 0.8, "low_dup": 0.05}

    actions = identify_tier_actions(cis_scores, unique_scores, cfg)
    assert "high" in actions["promote_to_enhanced"]
    assert "mid" not in actions["promote_to_enhanced"]
    assert "mid" not in actions["archive_candidates"]
    assert "low_dup" in actions["archive_candidates"]
    assert "low_unique" not in actions["archive_candidates"]


def test_curation_does_not_double_promote():
    from pipeline.corpus_curation import identify_tier_actions
    from addons.corpus_intelligence.config import CorpusIntelligenceConfig
    cfg = CorpusIntelligenceConfig(enhanced_tier_threshold=0.5)
    cis_scores    = {"already": 0.9}
    unique_scores = {"already": 0.9}

    actions = identify_tier_actions(
        cis_scores, unique_scores, cfg,
        already_enhanced={"already"},
    )
    assert "already" not in actions["promote_to_enhanced"]
