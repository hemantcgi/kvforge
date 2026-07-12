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


def test_answer_falls_back_to_retrieval_when_phase_lt_2():
    """answer() must delegate to kv_inference when phase < 2 (no gate, no eligibility check)."""
    from unittest.mock import patch, MagicMock

    cfg = {"embed_model": "BAAI/bge-small-en-v1.5", "gate_threshold": 0.75}
    # Patch ver inside confidence_gate and patch the function at source
    with patch("core.confidence_gate.ver") as mock_ver, \
         patch("pipeline.kv_inference.answer_with_retrieval", return_value="fallback answer") as mock_fn:
        mock_ver.get_phase.return_value = 1  # below Phase 2
        from core.confidence_gate import answer
        result = answer("what is bedrock?", cfg)
        mock_fn.assert_called_once()


def test_is_eligible_for_parametric_hard_gate():
    from core.confidence_gate import is_eligible_for_parametric
    assert is_eligible_for_parametric(0.90, 0.85) is True
    assert is_eligible_for_parametric(0.85, 0.85) is True
    assert is_eligible_for_parametric(0.84, 0.85) is False
    assert is_eligible_for_parametric(0.0, 0.85) is False


def test_similarity_returns_zero_for_empty_known_good(tmp_path):
    import json
    from pathlib import Path
    import core.version as ver
    from core import confidence_gate
    vfile = tmp_path / "version.json"
    vfile.write_text(json.dumps({
        "current_lora_version": 0, "checkpoint_path": None, "phase": 2,
        "prs_history": [], "known_good_queries": [], "clusters": {},
    }))
    ver.VERSION_FILE = vfile
    sim = confidence_gate._query_similarity_to_known_good(
        "any query", {"embed_model": "BAAI/bge-small-en-v1.5"})
    assert sim == 0.0


def test_phase2_ineligible_query_never_loads_model(tmp_path):
    """End-to-end, no-GPU safety test for the Phase-2 hard eligibility gate.

    Phase 2 with an empty known-good set means every query has similarity
    0.0 and is therefore INELIGIBLE for parametric answering
    (``is_eligible_for_parametric(0.0, threshold)`` is always False). This
    test proves ``confidence_gate.answer`` short-circuits straight to
    retrieval in that case and never touches ``model_loader.load`` — i.e.
    an ineligible query does zero parametric/model work.
    """
    import json
    from unittest.mock import patch
    import core.version as ver

    vfile = tmp_path / "version.json"
    vfile.write_text(json.dumps({
        "current_lora_version": 0, "checkpoint_path": None, "phase": 2,
        "prs_history": [], "known_good_queries": [], "clusters": {},
    }))
    ver.VERSION_FILE = vfile

    cfg = {"embed_model": "BAAI/bge-small-en-v1.5", "gate_threshold": 0.75}
    sentinel = "RETRIEVAL_SENTINEL_ANSWER"

    with patch("pipeline.kv_inference.answer_with_retrieval",
               return_value=sentinel) as mock_retrieval, \
         patch("core.model_loader.load") as mock_model_load:
        from core.confidence_gate import answer
        result = answer("any query", cfg)

    assert result == sentinel
    mock_retrieval.assert_called_once()
    mock_model_load.assert_not_called()
