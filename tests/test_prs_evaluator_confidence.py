"""Tests for Sprint 2.5 confidence-token wiring in prs_evaluator.py."""

import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.prs_evaluator import _generate_parametric, _extract_confidence


def _mock_pipe():
    pipe = MagicMock()
    pipe.tokenizer = MagicMock()
    pipe.model = MagicMock()
    return pipe


def test_generate_parametric_strips_confidence_suffix():
    pipe = _mock_pipe()
    query = "What is the answer?"
    pipe.return_value = [{
        "generated_text": f"{query}The answer is 42.\nConfidence: yes",
    }]

    result = _generate_parametric(
        query, pipe, tokenizer=None, sft_format="bare"
    )
    assert result == "The answer is 42."


def test_extract_confidence_legacy_integer_path():
    pipe = _mock_pipe()
    answer = "The answer is 42."
    suffix = (
        "\n\nOn a scale of 0 to 100, how confident are you in your answer above? "
        "Reply with a single integer only."
    )
    pipe.return_value = [{
        "generated_text": f"{answer}{suffix} 87",
    }]
    model = MagicMock()
    tokenizer = MagicMock()

    conf = _extract_confidence(
        answer, pipe, model, tokenizer, use_confidence_token=False
    )
    assert conf == 0.87


def test_extract_confidence_token_path():
    """When use_confidence_token=True, the restricted softmax path is used."""
    pipe = _mock_pipe()
    model = MagicMock()
    tokenizer = MagicMock()

    expected_p = 0.91
    with patch(
        "pipeline.confidence_token.extract_confidence_probability",
        return_value=expected_p,
    ) as mock_extract:
        conf = _extract_confidence(
            "The answer is 42.", pipe, model, tokenizer, use_confidence_token=True
        )

    assert conf == expected_p
    mock_extract.assert_called_once_with("The answer is 42.", model, tokenizer)
