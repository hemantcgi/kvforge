# tests/test_ab_evaluator.py
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_rouge_l_exact_match():
    from pipeline.ab_evaluator import _rouge_l
    assert _rouge_l("the cat sat", "the cat sat") == 1.0


def test_rouge_l_partial():
    from pipeline.ab_evaluator import _rouge_l
    score = _rouge_l("the cat sat on the mat", "the cat sat")
    assert 0.5 < score < 1.0


def test_rouge_l_empty():
    from pipeline.ab_evaluator import _rouge_l
    assert _rouge_l("", "reference") == 0.0
    assert _rouge_l("hypothesis", "") == 0.0


def test_rouge_l_no_overlap():
    from pipeline.ab_evaluator import _rouge_l
    assert _rouge_l("foo bar baz", "one two three") == 0.0


def test_cosine_identical():
    from pipeline.ab_evaluator import _cosine
    v = np.array([1.0, 0.0, 0.0])
    assert abs(_cosine(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal():
    from pipeline.ab_evaluator import _cosine
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(_cosine(a, b)) < 1e-6
