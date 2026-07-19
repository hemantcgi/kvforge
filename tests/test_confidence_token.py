"""Tests for pipeline/confidence_token.py — confidence pseudo-token helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.confidence_token import (
    CONFIDENCE_PREFIX,
    YES_TOKEN,
    NO_TOKEN,
    append_confidence_suffix,
    strip_confidence_suffix,
    factual_accuracy_to_label,
    generate_confidence_label,
    verify_confidence_tokens,
    extract_confidence_probability,
    generate_with_confidence_suffix,
    ConfidenceTokenError,
)


# ---------------------------------------------------------------------------
# Suffix formatting / stripping
# ---------------------------------------------------------------------------


def test_append_confidence_suffix_yes():
    out = append_confidence_suffix("Paris is the capital of France.", True)
    assert out.endswith("\nConfidence: yes")


def test_append_confidence_suffix_no():
    out = append_confidence_suffix("Paris is the capital of France.", False)
    assert out.endswith("\nConfidence: no")


def test_strip_confidence_suffix_full():
    text = "Paris is the capital of France.\nConfidence: yes"
    assert strip_confidence_suffix(text) == "Paris is the capital of France."


def test_strip_confidence_suffix_no():
    text = "Paris is the capital of France.\nConfidence: no"
    assert strip_confidence_suffix(text) == "Paris is the capital of France."


def test_strip_confidence_suffix_truncated():
    text = "Paris is the capital of France.\nConfidence:"
    assert strip_confidence_suffix(text) == "Paris is the capital of France."


def test_strip_confidence_suffix_missing():
    text = "Paris is the capital of France."
    assert strip_confidence_suffix(text) == text


def test_strip_confidence_suffix_empty():
    assert strip_confidence_suffix("") == ""


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------


def test_factual_accuracy_to_label():
    assert factual_accuracy_to_label(0.6, threshold=0.5) is True
    assert factual_accuracy_to_label(0.49, threshold=0.5) is False
    assert factual_accuracy_to_label(0.5, threshold=0.5) is True


def test_generate_confidence_label_uses_heuristic(monkeypatch):
    """When no judge client is supplied, the heuristic labeler is used."""
    from eval import metrics

    def fake_token_f1(a, b):
        return 0.9

    def fake_llm_judge(*args, **kwargs):
        return {"factually_correct": True}

    monkeypatch.setattr(metrics, "token_f1", fake_token_f1)
    monkeypatch.setattr(metrics, "llm_judge", fake_llm_judge)

    label = generate_confidence_label("Q", "A", "A")
    assert label is True


def test_generate_confidence_label_no_false(monkeypatch):
    from eval import metrics

    def fake_token_f1(a, b):
        return 0.1

    def fake_llm_judge(*args, **kwargs):
        return {"factually_correct": False}

    monkeypatch.setattr(metrics, "token_f1", fake_token_f1)
    monkeypatch.setattr(metrics, "llm_judge", fake_llm_judge)

    label = generate_confidence_label("Q", "A", "B")
    assert label is False


# ---------------------------------------------------------------------------
# Tokenizer verification
# ---------------------------------------------------------------------------


def test_verify_confidence_tokens_single_token():
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(side_effect=[
        [100],  # " yes"
        [200],  # " no"
    ])
    yes_id, no_id = verify_confidence_tokens(tokenizer)
    assert yes_id == 100
    assert no_id == 200


def test_verify_confidence_tokens_multi_token_raises():
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(side_effect=[
        [100, 101],  # " yes" -> multi-token
        [200],       # " no" -> single
    ])
    with pytest.raises(ConfidenceTokenError):
        verify_confidence_tokens(tokenizer)


# ---------------------------------------------------------------------------
# Inference-time probability extraction
# ---------------------------------------------------------------------------


def _make_mock_tokenizer(yes_id=100, no_id=200):
    tokenizer = MagicMock()

    def encode(text, add_special_tokens=False):
        if text == YES_TOKEN:
            return [yes_id]
        if text == NO_TOKEN:
            return [no_id]
        raise ValueError(f"unexpected tokenize: {text!r}")

    tokenizer.encode = encode
    return tokenizer


def _make_mock_model(logits: np.ndarray) -> MagicMock:
    """Return a MagicMock whose forward pass yields the given 1-D logits."""
    model = MagicMock()
    model.device = "cpu"
    # Real model outputs are [batch, seq_len, vocab]; we give [1, 1, vocab].
    model.return_value.logits = torch.from_numpy(logits).float().unsqueeze(0).unsqueeze(0)
    return model


def test_extract_confidence_probability_yes():
    tokenizer = _make_mock_tokenizer(yes_id=10, no_id=20)
    logits = np.zeros(100, dtype=np.float32)
    logits[10] = 10.0
    logits[20] = 1.0
    model = _make_mock_model(logits)

    p_yes = extract_confidence_probability("The answer is 42.", model, tokenizer)
    assert 0.9 < p_yes < 1.0


def test_extract_confidence_probability_no():
    tokenizer = _make_mock_tokenizer(yes_id=10, no_id=20)
    logits = np.zeros(100, dtype=np.float32)
    logits[10] = 1.0
    logits[20] = 10.0
    model = _make_mock_model(logits)

    p_yes = extract_confidence_probability("The answer is 42.", model, tokenizer)
    assert 0.0 < p_yes < 0.1


def test_extract_confidence_probability_strips_existing_suffix():
    tokenizer = _make_mock_tokenizer(yes_id=10, no_id=20)
    logits = np.zeros(100, dtype=np.float32)
    logits[10] = 10.0
    logits[20] = 1.0
    model = _make_mock_model(logits)

    p_yes = extract_confidence_probability(
        "The answer is 42.\nConfidence: no", model, tokenizer
    )
    assert 0.9 < p_yes < 1.0


def test_extract_confidence_probability_fallback_to_half():
    tokenizer = _make_mock_tokenizer(yes_id=10, no_id=20)
    logits = np.zeros(100, dtype=np.float32)
    model = _make_mock_model(logits)

    p_yes = extract_confidence_probability("The answer is 42.", model, tokenizer)
    assert p_yes == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------


def test_generate_with_confidence_suffix_returns_probability():
    tokenizer = _make_mock_tokenizer(yes_id=10, no_id=20)
    tokenizer.decode = MagicMock(return_value="The answer is 42.")
    tokenizer.pad_token_id = 0

    logits = np.zeros(100, dtype=np.float32)
    logits[10] = 10.0
    logits[20] = 1.0
    model = _make_mock_model(logits)

    inputs = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }
    # shape[1] is used to slice generated tokens.
    inputs["input_ids"].shape = (1, 5)

    answer, p_yes = generate_with_confidence_suffix(
        model, tokenizer, inputs, max_new_tokens=10, do_sample=False
    )
    assert answer == "The answer is 42."
    assert 0.9 < p_yes < 1.0
    model.generate.assert_called_once()
    model.assert_called_once()
