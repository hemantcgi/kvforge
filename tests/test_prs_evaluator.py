"""Tests for flexible FAQ schema support in prs_evaluator."""
import pytest


def test_standard_schema():
    from prs_evaluator import _extract_qa
    faq = {"question": "What is X?", "answer": "X is Y."}
    q, a = _extract_qa(faq)
    assert q == "What is X?"
    assert a == "X is Y."


def test_custom_schema_q_a():
    from prs_evaluator import _extract_qa
    faq = {"q": "What is X?", "a": "X is Y."}
    q, a = _extract_qa(faq, q_key="q", a_key="a")
    assert q == "What is X?"
    assert a == "X is Y."


def test_custom_schema_query_ground_truth():
    from prs_evaluator import _extract_qa
    faq = {"query": "What is X?", "ground_truth": "X is Y."}
    q, a = _extract_qa(faq, q_key="query", a_key="ground_truth")
    assert q == "What is X?"


def test_missing_key_raises_clear_error():
    from prs_evaluator import _extract_qa
    faq = {"q": "What?", "a": "This."}
    with pytest.raises(KeyError, match="FAQ missing key 'question'"):
        _extract_qa(faq, q_key="question", a_key="answer")
