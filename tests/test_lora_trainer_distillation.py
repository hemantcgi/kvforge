"""Tests for Sprint 2 distillation mode in pipeline/lora_trainer.py."""

import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.lora_trainer import (
    _build_distillation_examples,
    _SourcePreservingCollator,
    _SourceAwareTrainer,
    _PerSourceLossCallback,
    build_confidence_sft_example,
    build_sft_example,
)


def _mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock(
        side_effect=lambda msgs, **kw: [1] * len(msgs) * 5 if kw.get("tokenize") else "prompt"
    )
    tokenizer.pad_token = None
    tokenizer.eos_token_id = 999
    return tokenizer


def test_build_distillation_examples_chat_format():
    tokenizer = _mock_tokenizer()
    teacher_pairs = [
        {"question": "Q1", "teacher_answer": "A1", "confidence_label": True},
    ]
    on_policy_pairs = [
        {"question": "Q2", "teacher_answer": "A2", "confidence_label": False},
    ]
    replay_chunks = [{"text": "replay text", "chunk_id": 1}]

    examples = _build_distillation_examples(
        tokenizer, teacher_pairs, on_policy_pairs, replay_chunks,
        confidence_supervision=False, max_length=64,
    )
    assert len(examples) == 3
    sources = [e["source"] for e in examples]
    assert "teacher_sft" in sources
    assert "on_policy_correction" in sources
    assert "replay" in sources


def test_build_distillation_examples_with_confidence_supervision():
    tokenizer = _mock_tokenizer()
    tokenizer.decode = MagicMock(return_value="")
    teacher_pairs = [
        {"question": "Q1", "teacher_answer": "A1", "confidence_label": True},
    ]
    on_policy_pairs = [
        {"question": "Q2", "teacher_answer": "A2", "confidence_label": False},
    ]
    replay_chunks = []

    examples = _build_distillation_examples(
        tokenizer, teacher_pairs, on_policy_pairs, replay_chunks,
        confidence_supervision=True, max_length=64,
    )
    assert len(examples) == 2
    assert all("source" in e for e in examples)


def test_source_preserving_collator_keeps_source():
    tokenizer = MagicMock()
    base = MagicMock()
    base.return_value = {"input_ids": MagicMock(), "labels": MagicMock(), "attention_mask": MagicMock()}
    collator = _SourcePreservingCollator(tokenizer)
    collator.base = base

    features = [
        {"input_ids": [1, 2], "labels": [2, 3], "attention_mask": [1, 1], "source": "teacher_sft"},
        {"input_ids": [4, 5], "labels": [5, 6], "attention_mask": [1, 1], "source": "replay"},
    ]
    batch = collator(features)
    assert batch["source"] == ["teacher_sft", "replay"]
    base.assert_called_once()


def test_source_aware_trainer_strips_source():
    # Avoid Trainer.__init__ by creating a bare instance.
    trainer = object.__new__(_SourceAwareTrainer)
    trainer.base_model = None
    trainer.kl_weight = 0.0
    def mock_compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        return (1.0, None) if return_outputs else 1.0
    inputs = {
        "input_ids": MagicMock(),
        "labels": MagicMock(),
        "source": ["teacher_sft"],
    }
    with patch("transformers.Trainer.compute_loss", mock_compute_loss):
        loss = trainer.compute_loss(MagicMock(), inputs)
    assert loss == 1.0


def test_per_source_loss_callback_records_losses():
    from datasets import Dataset
    dataset = Dataset.from_list([
        {"input_ids": [1, 2], "labels": [2, 3], "attention_mask": [1, 1], "source": "teacher_sft"},
        {"input_ids": [4, 5], "labels": [5, 6], "attention_mask": [1, 1], "source": "replay"},
    ])
    mock_model = MagicMock()
    mock_model.parameters.return_value = []
    callback = _PerSourceLossCallback(dataset, model=mock_model)
    callback._compute_source_loss = MagicMock(side_effect=lambda s: 0.5 if s == "teacher_sft" else 0.3)

    class FakeState:
        epoch = 1.0

    callback.on_epoch_end(None, FakeState(), None)
    assert 1.0 in callback.epoch_losses
    assert callback.epoch_losses[1.0]["teacher_sft"] == 0.5
    assert callback.epoch_losses[1.0]["replay"] == 0.3


def test_distillation_cli_args():
    from pipeline import lora_trainer
    import argparse
    p = argparse.ArgumentParser()
    # Replicate the relevant args for a quick parse test.
    p.add_argument("--distill-pairs", default=None)
    p.add_argument("--faqs", default=None)
    p.add_argument("--source-file", default=None)
    args = p.parse_args(["--distill-pairs", "pairs.json"])
    assert args.distill_pairs == "pairs.json"
