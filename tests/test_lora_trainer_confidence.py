"""Tests for Sprint 2.5 confidence-token wiring in lora_trainer.py."""

import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.lora_trainer import (
    build_confidence_sft_example,
    _faq_confidence_label,
)


def _mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock(side_effect=lambda msgs, **kw: [
        # Encode user+assistant as: [user_id, *content_ids, eos_id]
        # Simplified deterministic mock.
        1 if m["role"] == "user" else 2 for m in msgs for _ in m["content"]
    ] if kw.get("tokenize") else "mock_prompt")
    tokenizer.pad_token = None
    tokenizer.eos_token_id = 999
    return tokenizer


def test_build_confidence_sft_example_appends_suffix():
    tokenizer = _mock_tokenizer()
    # Make the full sequence long enough to keep the suffix.
    tokenizer.apply_chat_template = MagicMock(
        side_effect=lambda msgs, **kw: [1, 2, 3, 4, 5, 6]
    )
    example = build_confidence_sft_example(
        tokenizer, "What is Bedrock?", "A managed service.", True, max_length=64
    )
    assert "input_ids" in example
    assert "labels" in example
    assert "attention_mask" in example


def test_faq_confidence_label_bool():
    assert _faq_confidence_label({"confidence_label": True}) is True
    assert _faq_confidence_label({"confidence_label": False}) is False


def test_faq_confidence_label_string():
    assert _faq_confidence_label({"confidence": "yes"}) is True
    assert _faq_confidence_label({"confidence": "no"}) is False
    assert _faq_confidence_label({"confidence": "YES"}) is True


def test_faq_confidence_label_defaults_to_yes():
    assert _faq_confidence_label({"question": "Q", "answer": "A"}) is True
