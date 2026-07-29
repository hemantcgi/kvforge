"""Tests for KL-to-base loss in pipeline/lora_trainer.py distillation mode."""

import sys
from pathlib import Path

import pytest
import torch
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.lora_trainer import _SourceAwareTrainer


def test_source_aware_trainer_adds_kl_loss():
    # Bare instance to avoid Trainer.__init__.
    trainer = object.__new__(_SourceAwareTrainer)

    # Base model returns logits.
    base_model = MagicMock()
    base_logits = torch.randn(2, 5, 10, requires_grad=False)
    base_model.return_value.logits = base_logits
    base_model.return_value.loss = torch.tensor(0.0)

    # Student model returns logits.
    student_model = MagicMock()
    student_logits = torch.randn(2, 5, 10, requires_grad=True)
    student_model.return_value.logits = student_logits
    student_model.return_value.loss = torch.tensor(1.0)

    trainer.base_model = base_model
    trainer.kl_weight = 0.5

    inputs = {
        "input_ids": torch.randint(0, 10, (2, 5)),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3, -100, -100], [1, 2, 3, 4, 5]]),
    }

    def mock_super_compute(self, model, inputs, return_outputs=False, **kwargs):
        return torch.tensor(1.0)

    with patch("transformers.Trainer.compute_loss", mock_super_compute):
        loss = trainer.compute_loss(student_model, inputs)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 1.0  # base loss + positive KL
    base_model.assert_called_once()
    student_model.assert_called_once()


def test_source_aware_trainer_no_kl_when_weight_zero():
    trainer = object.__new__(_SourceAwareTrainer)
    trainer.base_model = None
    trainer.kl_weight = 0.0

    def mock_super_compute(self, model, inputs, return_outputs=False, **kwargs):
        return torch.tensor(1.0)

    inputs = {
        "input_ids": torch.randint(0, 10, (2, 5)),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3, -100, -100], [1, 2, 3, 4, 5]]),
    }
    with patch("transformers.Trainer.compute_loss", mock_super_compute):
        loss = trainer.compute_loss(MagicMock(), inputs)

    assert loss == 1.0


def test_kl_to_base_weight_in_config():
    from core.config import KVForgeConfig
    cfg = KVForgeConfig(
        use_case_name="UC",
        collection="col",
        version_file="v.json",
        kl_to_base_weight=0.1,
    )
    assert cfg.kl_to_base_weight == 0.1
    assert cfg.get_merged_config()["kl_to_base_weight"] == 0.1
