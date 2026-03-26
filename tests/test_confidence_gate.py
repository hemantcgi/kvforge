# tests/test_confidence_gate.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.confidence_gate import compute_hedging_score, decide_gate


def test_hedging_score_high_for_uncertain_text():
    text = "I think it might be approximately 50, or maybe 100, I'm not sure."
    score = compute_hedging_score(text)
    assert score > 0.3


def test_hedging_score_zero_for_confident_text():
    text = "Amazon Bedrock is a fully managed service for foundation models."
    score = compute_hedging_score(text)
    assert score == 0.0


def test_gate_low_entropy_passes():
    result = decide_gate(
        token_entropy=0.15,
        hedging_score=0.0,
        query_similarity=0.92,
        threshold=0.75,
    )
    assert result == "direct"


def test_gate_high_entropy_retrieves():
    result = decide_gate(
        token_entropy=0.82,
        hedging_score=0.5,
        query_similarity=0.30,
        threshold=0.75,
    )
    assert result == "retrieve"


def test_answer_falls_back_to_retrieval_when_phase_lt_3():
    """answer() must delegate to kv_inference when phase < 3 (no gate)."""
    from unittest.mock import patch, MagicMock

    cfg = {"embed_model": "BAAI/bge-small-en-v1.5", "gate_threshold": 0.75}
    # Patch ver inside confidence_gate and patch the function at source
    with patch("core.confidence_gate.ver") as mock_ver, \
         patch("pipeline.kv_inference.answer_with_retrieval", return_value="fallback answer") as mock_fn:
        mock_ver.get_phase.return_value = 2  # below Phase 3
        from core.confidence_gate import answer
        result = answer("what is bedrock?", cfg)
        mock_fn.assert_called_once()
