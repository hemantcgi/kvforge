import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import metrics as eval_metrics


def test_exact_match_normalization():
    """Exact match should ignore articles, case, and punctuation."""
    assert eval_metrics.exact_match("The answer is 42.", "answer is 42") == 1


def test_token_f1_perfect_match():
    assert eval_metrics.token_f1("hello world", "hello world") == 1.0


def test_token_f1_no_overlap():
    assert eval_metrics.token_f1("foo bar", "baz qux") == 0.0


def test_heuristic_judge_correct():
    """High token-F1 answer should be marked correct by heuristic judge."""
    result = eval_metrics._heuristic_judge(
        "What color is the sky?", "The sky is blue.", "blue"
    )
    assert result["factually_correct"] is True
    assert "token-F1" in result["rationale"]


def test_heuristic_judge_incorrect():
    """Low token-F1 answer should be marked incorrect."""
    result = eval_metrics._heuristic_judge(
        "What color is the sky?", "The sky is green.", "blue"
    )
    assert result["factually_correct"] is False


def test_judge_prompt_contains_key_facts_requirement():
    """The judge prompt must require key facts and not just style."""
    # We can't call the API in a unit test, but we can inspect the prompt text.
    import inspect

    source = inspect.getsource(eval_metrics.llm_judge)
    assert "key facts" in source
    assert "CORRECT:" in source
    assert "INCORRECT:" in source


def test_bootstrap_ci_empty():
    """Bootstrap CI on empty input returns zeros."""
    point, lo, hi = eval_metrics.bootstrap_ci([])
    assert point == 0.0 and lo == 0.0 and hi == 0.0


def test_expected_calibration_error_empty():
    """ECE on empty input returns zero-filled structure."""
    ece = eval_metrics.expected_calibration_error([], [])
    assert ece["ece"] == 0.0
    assert ece["per_bin_accuracy"] == []
    assert ece["per_bin_confidence"] == []

