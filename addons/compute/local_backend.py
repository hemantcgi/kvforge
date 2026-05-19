"""LocalComputeBackend — runs KV tensor computation in-process (CPU or local GPU)."""
from __future__ import annotations

import numpy as np

import core.model_loader as model_loader
from core.compute import compute_kv_for_chunk


class LocalComputeBackend:
    """ComputeBackend that runs compute_kv_for_chunk in the current process.

    Loads the model once at construction time and keeps it in memory.

    Args:
        cfg: KVForge addon config dict. Uses the inference.llm_model and
            training.checkpoint_path fields to load the correct model+LoRA.
    """

    def __init__(self, cfg: dict) -> None:
        lora_ckpt = None
        # Load LoRA checkpoint path from version file if available
        try:
            import core.version as ver
            lora_ckpt = ver.load().get("checkpoint_path")
        except Exception:
            pass
        self._model, self._tokenizer = model_loader.load(lora_ckpt)
        self._cfg = cfg

    def compute_kv_batch(
        self,
        texts: list[str],
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> list[np.ndarray]:
        return [
            compute_kv_for_chunk(
                text, self._model, self._tokenizer,
                num_layers, num_kv_heads, head_dim,
            )
            for text in texts
        ]

    def health(self) -> dict:
        import torch
        return {
            "backend": "local",
            "device": str(next(self._model.parameters()).device),
            "model": type(self._model).__name__,
        }
