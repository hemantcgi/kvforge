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


def test_run_eval_output_schema(tmp_path):
    """run_eval returns list of dicts with required keys."""
    from unittest.mock import patch, MagicMock
    import json
    from pipeline.ab_evaluator import run_eval

    # Minimal config
    cfg = {
        "embed_model": "BAAI/bge-small-en-v1.5",
        "faq_question_key": "question",
        "faq_answer_key": "answer",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))

    faqs = [{"question": "What is X?", "answer": "X is a thing."}]
    faqs_path = tmp_path / "faqs.json"
    faqs_path.write_text(json.dumps(faqs))

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "answer_a": "X is a thing.",
        "answer_b": "X is something.",
        "mode_a": "parametric",
        "latency_a_ms": 100,
        "latency_b_ms": 500,
        "generation_a_ms": 100,
        "generation_b_ms": 450,
    }

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        results = run_eval(
            config_path=str(config_path),
            faqs_path=str(faqs_path),
            dashboard_url="http://localhost:9999",
            gemini_api_key="fake-key",
            max_samples=1,
        )

    assert len(results) == 1
    r = results[0]
    required = {"question", "ground_truth", "answer_a", "answer_b", "mode_a",
                "latency_a_ms", "latency_b_ms", "generation_a_ms", "generation_b_ms",
                "sem_sim_a", "sem_sim_b", "rouge_l_a", "rouge_l_b"}
    assert required.issubset(r.keys())
    assert 0.0 <= r["sem_sim_a"] <= 1.0
    assert 0.0 <= r["rouge_l_a"] <= 1.0


def test_run_eval_missing_faqs(tmp_path):
    """run_eval raises FileNotFoundError with helpful message when faqs.json absent."""
    import json
    from pipeline.ab_evaluator import run_eval

    cfg = {"embed_model": "BAAI/bge-small-en-v1.5",
           "faq_question_key": "question", "faq_answer_key": "answer"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    # faqs.json NOT created

    try:
        run_eval(str(config_path), str(tmp_path / "faqs.json"),
                 "http://localhost:9999", "", 1)
        assert False, "Should have raised"
    except FileNotFoundError as e:
        assert "run the pipeline first" in str(e)


def test_generate_html_contains_data():
    """generate_html produces valid HTML with const AB_DATA embedded."""
    from pipeline.ab_evaluator import generate_html
    results = [{"question": "Q", "ground_truth": "A", "answer_a": "A", "answer_b": "B",
                "mode_a": "parametric", "latency_a_ms": 100, "latency_b_ms": 500,
                "generation_a_ms": 100, "generation_b_ms": 450,
                "sem_sim_a": 0.9, "sem_sim_b": 0.8, "rouge_l_a": 0.5, "rouge_l_b": 0.4}]
    html = generate_html(results, title="Test Eval")
    assert "const AB_DATA" in html
    assert "Test Eval" in html
    assert "<!DOCTYPE html>" in html
