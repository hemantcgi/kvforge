"""ComputeBackend protocol — structural interface for KV tensor computation.

Any object implementing compute_kv_batch() and health() satisfies this protocol
regardless of where the computation runs (local GPU, remote worker, mock in tests).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ComputeBackend(Protocol):
    def compute_kv_batch(
        self,
        texts: list[str],
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> list[np.ndarray]:
        """Compute mean-pooled KV tensors for a batch of texts.

        Args:
            texts: Plain-text chunk contents.
            num_layers: Number of transformer layers (e.g. 28 for Llama-3.2-3B).
            num_kv_heads: Number of KV attention heads.
            head_dim: Head dimensionality.

        Returns:
            List of float16 arrays, each shaped [num_layers, 2, num_kv_heads, head_dim].
        """
        ...

    def health(self) -> dict:
        """Return health/status info (backend type, device, model loaded, etc.)."""
        ...
