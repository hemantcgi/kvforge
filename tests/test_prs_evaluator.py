"""Tests for flexible FAQ schema support in prs_evaluator."""
import pytest


def test_standard_schema():
    from pipeline.prs_evaluator import _extract_qa
    faq = {"question": "What is X?", "answer": "X is Y."}
    q, a = _extract_qa(faq)
    assert q == "What is X?"
    assert a == "X is Y."


def test_custom_schema_q_a():
    from pipeline.prs_evaluator import _extract_qa
    faq = {"q": "What is X?", "a": "X is Y."}
    q, a = _extract_qa(faq, q_key="q", a_key="a")
    assert q == "What is X?"
    assert a == "X is Y."


def test_custom_schema_query_ground_truth():
    from pipeline.prs_evaluator import _extract_qa
    faq = {"query": "What is X?", "ground_truth": "X is Y."}
    q, a = _extract_qa(faq, q_key="query", a_key="ground_truth")
    assert q == "What is X?"


def test_missing_key_raises_clear_error():
    from pipeline.prs_evaluator import _extract_qa
    faq = {"q": "What?", "a": "This."}
    with pytest.raises(KeyError, match="FAQ missing key 'question'"):
        _extract_qa(faq, q_key="question", a_key="answer")


def test_prs_weights_fully_accuracy():
    from pipeline.prs_evaluator import _compute_prs
    prs = _compute_prs(
        accuracy_ratios=[1.0, 1.0],
        calibrations=[0.0, 0.0],
        consistencies=[0.0, 0.0],
        weights={"accuracy": 1.0, "calibration": 0.0, "consistency": 0.0}
    )
    assert abs(prs - 1.0) < 0.001


def test_prs_uses_default_weights_when_none():
    from pipeline.prs_evaluator import _compute_prs
    prs = _compute_prs([0.8], [0.9], [0.7], weights=None)
    expected = 0.5 * 0.8 + 0.3 * 0.9 + 0.2 * 0.7
    assert abs(prs - expected) < 0.001


def test_get_cluster_stats_receives_string_not_dict():
    """Regression: prs_evaluator must pass db_path str, not cfg dict."""
    from pipeline import query_logger
    import tempfile, os

    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "q.db")
        query_logger.init_db(db_path)
        # Passing a dict (the old bug) raises AttributeError — dicts have no .connect
        import pytest
        cfg_dict = {"query_log_db": db_path}
        with pytest.raises((TypeError, AttributeError)):
            query_logger.get_cluster_stats(cfg_dict, "0")
        # Passing the resolved string works
        result = query_logger.get_cluster_stats(db_path, "0")
        assert result == {"realtime_coverage": 0.0, "query_count": 0}
