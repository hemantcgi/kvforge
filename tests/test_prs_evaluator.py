"""Tests for flexible FAQ schema support in prs_evaluator."""
import os
import tempfile

import numpy as np
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
    expected = 0.7 * 0.8 + 0.15 * 0.9 + 0.15 * 0.7
    assert abs(prs - expected) < 0.001


def test_factual_accuracy_formula():
    from pipeline.prs_evaluator import _factual_accuracy
    assert abs(_factual_accuracy(f1=0.8, judge_correct=True) - 0.9) < 0.001
    assert abs(_factual_accuracy(f1=0.4, judge_correct=False) - 0.2) < 0.001
    assert abs(_factual_accuracy(f1=0.0, judge_correct=False) - 0.0) < 0.001
    assert abs(_factual_accuracy(f1=1.0, judge_correct=True) - 1.0) < 0.001


def test_fixed_calibration_formula():
    from pipeline.prs_evaluator import _fixed_calibration
    # Perfect calibration: confidence exactly matches factual accuracy.
    assert abs(_fixed_calibration(self_conf=0.5, factual_acc=0.5) - 1.0) < 0.001
    # Overconfident: confidence 0.9, factual accuracy 0.1 -> calibration 0.2.
    assert abs(_fixed_calibration(self_conf=0.9, factual_acc=0.1) - 0.2) < 0.001


def test_coverage_ratio():
    from pipeline.prs_evaluator import _coverage_ratio
    assert abs(_coverage_ratio([0.9, 0.3, 0.6, 0.2], threshold=0.5) - 0.5) < 0.001
    assert _coverage_ratio([], threshold=0.5) == 0.0
    assert _coverage_ratio([0.1, 0.2], threshold=0.5) == 0.0
    assert _coverage_ratio([0.9, 0.9], threshold=0.5) == 1.0


def test_get_cluster_stats_receives_string_not_dict():
    """Regression: prs_evaluator must pass db_path str, not cfg dict."""
    from pipeline import query_logger

    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "q.db")
        query_logger.init_db(db_path)
        # sqlite3.connect rejects a dict — raises TypeError
        cfg_dict = {"query_log_db": db_path}
        with pytest.raises((TypeError, AttributeError)):
            query_logger.get_cluster_stats(cfg_dict, "0")
        # Passing the resolved string works
        result = query_logger.get_cluster_stats(db_path, "0")
        assert result == {"realtime_coverage": 0.0, "query_count": 0}


class _FakeTok:
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        # deterministic fake: wrap the user content in a marker scaffold
        content = messages[0]["content"]
        return f"<<USER>>{content}<<ASSISTANT>>"


def test_format_query_bare_is_identity():
    from pipeline.prs_evaluator import _format_query
    assert _format_query("What is X?", _FakeTok(), "bare") == "What is X?"


def test_format_query_chat_wraps():
    from pipeline.prs_evaluator import _format_query
    out = _format_query("What is X?", _FakeTok(), "chat")
    assert out != "What is X?"
    assert "What is X?" in out
    assert out.startswith("<<USER>>")


def test_format_query_chat_strips_variant_suffix():
    from pipeline.prs_evaluator import _format_query
    out = _format_query("How do I reset? (variant 3)", _FakeTok(), "chat")
    assert "(variant 3)" not in out       # suffix stripped before templating
    assert "How do I reset?" in out
    # bare mode does NOT strip (keeps bare train/eval consistent)
    assert _format_query("How do I reset? (variant 3)", _FakeTok(), "bare") == "How do I reset? (variant 3)"


def test_generate_parametric_suppresses_bos_in_chat_mode():
    """Regression: chat-template strings already contain a literal
    <|begin_of_text|> BOS token. The HF text-generation pipeline's default
    add_special_tokens=True would prepend a SECOND BOS at eval time, while
    chat-SFT training (apply_chat_template(..., tokenize=True)) produces only
    a single BOS. That train/eval mismatch biases the A/B experiment. Chat
    mode must pass add_special_tokens=False; bare mode must keep the default
    (True/unset) since the bare string has no BOS and needs the pipeline to
    add one, matching bare training.
    """
    from pipeline.prs_evaluator import _generate_parametric

    calls = {}

    def fake_pipe(prompt, **kwargs):
        calls.clear()
        calls.update(kwargs)
        calls["prompt"] = prompt
        return [{"generated_text": prompt + "ANSWER"}]

    # Chat mode -> add_special_tokens must be explicitly False.
    result = _generate_parametric("q? (variant 1)", fake_pipe, _FakeTok(), "chat")
    assert calls.get("add_special_tokens") is False
    assert result == "ANSWER"

    # Bare mode -> default pipeline behavior (True or unset), unchanged.
    result = _generate_parametric("q?", fake_pipe, None, "bare")
    assert calls.get("add_special_tokens") in (True, None)
    assert result == "ANSWER"


def test_self_consistency_suppresses_bos_in_chat_mode():
    """Same BOS-duplication fix applies to the self-consistency sampler,
    which also feeds the formatted (chat-template) prompt into a pipeline
    call and must not let the pipeline add a second BOS.
    """
    from pipeline.prs_evaluator import _self_consistency

    calls = []

    def fake_pipe_sample(prompt, **kwargs):
        calls.append(kwargs)
        return [{"generated_text": prompt + "ANSWER"}]

    class _FakeEmbedder:
        def embed(self, texts):
            # deterministic fake embeddings, one dim per call so cosine sim is defined
            return [np.array([1.0, float(i)]) for i, _ in enumerate(texts)]

    _self_consistency("q?", fake_pipe_sample, _FakeEmbedder(), _FakeTok(), "chat", n=2)
    assert all(kw.get("add_special_tokens") is False for kw in calls)

    calls.clear()
    _self_consistency("q?", fake_pipe_sample, _FakeEmbedder(), None, "bare", n=2)
    assert all(kw.get("add_special_tokens") in (True, None) for kw in calls)
