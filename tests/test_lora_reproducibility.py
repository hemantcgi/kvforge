import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure KVForge is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.lora_trainer import save_training_metadata, set_seed


def test_set_seed_reproduces_random():
    """Setting the same seed twice produces the same random sequence."""
    set_seed(42)
    a = [random.random() for _ in range(10)]
    set_seed(42)
    b = [random.random() for _ in range(10)]
    assert a == b


def test_set_seed_changes_numpy():
    """Setting different seeds produces different numpy sequences."""
    set_seed(123)
    a = np.random.rand(10).tolist()
    set_seed(456)
    b = np.random.rand(10).tolist()
    assert a != b


def test_save_training_metadata_writes_file():
    """Metadata is written and contains the expected fields."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {"use_case_name": "test", "collection": "test"}
        save_training_metadata(
            output_dir=tmp,
            cfg=cfg,
            seed=123,
            command=["python", "lora_trainer.py", "--seed", "123"],
            notes={"mode": "qa", "examples": 42},
        )
        meta_path = Path(tmp) / "kvforge_training_metadata.json"
        assert meta_path.exists()
        import json

        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["seed"] == 123
        assert meta["config"] == cfg
        assert meta["notes"]["examples"] == 42
        assert meta["timestamp"].endswith("Z")


def test_save_training_metadata_creates_dir():
    """Metadata save still works if the output dir must be created."""
    with tempfile.TemporaryDirectory() as tmp:
        subdir = Path(tmp) / "v99"
        save_training_metadata(
            output_dir=str(subdir),
            cfg={},
            seed=0,
            command=[],
        )
        assert (subdir / "kvforge_training_metadata.json").exists()


def test_set_seed_deterministic_cudnn():
    """Deterministic mode sets the expected cuDNN flags and restores them after."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    prev_det = torch.backends.cudnn.deterministic
    prev_bench = torch.backends.cudnn.benchmark
    try:
        set_seed(7, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.backends.cudnn.deterministic = prev_det
        torch.backends.cudnn.benchmark = prev_bench


def test_set_seed_non_deterministic_cudnn():
    """Non-deterministic mode preserves existing cuDNN flags."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    prev_det = torch.backends.cudnn.deterministic
    prev_bench = torch.backends.cudnn.benchmark
    try:
        # Start from a known non-deterministic state.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        set_seed(7, deterministic=False)
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.backends.cudnn.deterministic = prev_det
        torch.backends.cudnn.benchmark = prev_bench

