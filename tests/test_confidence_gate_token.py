"""Tests for Sprint 2.5 confidence-token wiring in confidence_gate.py."""

import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_parametric_or_retrieve_uses_confidence_token_when_enabled(tmp_path):
    """When use_confidence_token=True, the gate routes by P(yes)."""
    import json
    import core.version as ver

    vfile = tmp_path / "version.json"
    vfile.write_text(json.dumps({
        "current_lora_version": 1,
        "checkpoint_path": "dummy_ckpt",
        "phase": 2,
    }))
    ver.VERSION_FILE = vfile

    cfg = {"embed_model": "BAAI/bge-small-en-v1.5"}
    effective_cfg = {
        "gate_threshold": 0.75,
        "use_confidence_token": True,
        "sft_format": "chat",
    }

    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock(return_value="user prompt")
    tokenizer.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }
    tokenizer.return_value["input_ids"].shape = (1, 3)

    with patch("core.confidence_gate.model_loader.load",
               return_value=(model, tokenizer)) as mock_load, \
         patch("core.confidence_gate._generate_answer_with_confidence",
               return_value=("parametric answer", 0.85)) as mock_gen, \
         patch("core.confidence_gate._log_would_have_retrieved") as mock_log, \
         patch("core.confidence_gate._log_query_with_confidence") as mock_qlog:
        from core.confidence_gate import _parametric_or_retrieve
        ans, conf = _parametric_or_retrieve("Q", cfg, effective_cfg, similarity=0.9)

    assert ans == "parametric answer"
    assert conf == 0.85
    mock_load.assert_called_once()
    mock_gen.assert_called_once()
    mock_log.assert_called_once()


def test_parametric_or_retrieve_falls_back_to_retrieval_when_confidence_low(tmp_path):
    """When P(yes) is below threshold, route to retrieval."""
    import json
    import core.version as ver

    vfile = tmp_path / "version.json"
    vfile.write_text(json.dumps({
        "current_lora_version": 1,
        "checkpoint_path": "dummy_ckpt",
        "phase": 2,
    }))
    ver.VERSION_FILE = vfile

    cfg = {"embed_model": "BAAI/bge-small-en-v1.5"}
    effective_cfg = {
        "gate_threshold": 0.75,
        "use_confidence_token": True,
        "sft_format": "chat",
    }

    model = MagicMock()
    tokenizer = MagicMock()

    with patch("core.confidence_gate.model_loader.load",
               return_value=(model, tokenizer)), \
         patch("core.confidence_gate._generate_answer_with_confidence",
               return_value=("parametric answer", 0.30)) as mock_gen, \
         patch("core.confidence_gate._log_would_have_retrieved") as mock_log, \
         patch("core.confidence_gate._log_query_with_confidence") as mock_qlog, \
         patch("pipeline.kv_inference.answer_with_retrieval",
               return_value="retrieved answer") as mock_retrieve:
        from core.confidence_gate import _parametric_or_retrieve
        ans, conf = _parametric_or_retrieve("Q", cfg, effective_cfg, similarity=0.9)

    assert ans == "retrieved answer"
    assert conf == 0.30
    mock_gen.assert_called_once()
    mock_log.assert_not_called()
    mock_retrieve.assert_called_once()


def test_answer_with_confidence_token_uses_chat_format():
    """_generate_answer_with_confidence builds chat-format inputs and suppresses BOS."""
    import torch
    from core.confidence_gate import _generate_answer_with_confidence

    model = MagicMock()
    model.device = "cpu"
    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock(return_value="chat prompt")
    inputs_mock = MagicMock()
    inputs_mock.to = MagicMock(return_value=inputs_mock)
    tokenizer.return_value = inputs_mock

    with patch("pipeline.confidence_token.generate_with_confidence_suffix",
               return_value=("ans", 0.8)) as mock_gen:
        answer, p_yes = _generate_answer_with_confidence(
            "Q", model, tokenizer, sft_format="chat"
        )

    assert answer == "ans"
    assert p_yes == 0.8
    tokenizer.apply_chat_template.assert_called_once()
    tokenizer.assert_called_once_with(
        "chat prompt", return_tensors="pt", add_special_tokens=False
    )
    mock_gen.assert_called_once()
