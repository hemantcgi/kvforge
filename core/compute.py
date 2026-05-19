import numpy as np
import torch

import core.kv_utils as kv_utils


def compute_kv_for_chunk(
    text: str,
    model,
    tokenizer,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> np.ndarray:
    """Run a single text chunk through the LLM and return its mean-pooled KV array.

    Tokenises *text* (truncated to 512 tokens), runs a forward pass with
    ``use_cache=True``, then calls ``kv_utils.mean_pool_kv`` to compress the
    per-token KV tensors into a fixed-size float16 array.

    Args:
        text: Plain-text content of the chunk.
        model: Loaded HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        num_layers: Expected number of transformer layers (used for shape assertion).
        num_kv_heads: Expected number of KV attention heads.
        head_dim: Expected head dimensionality.

    Returns:
        Float16 numpy array of shape ``[num_layers, 2, num_kv_heads, head_dim]``.

    Raises:
        AssertionError: If the produced KV shape does not match the expected
            dimensions.
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    arr = kv_utils.mean_pool_kv(outputs.past_key_values)
    expected = (num_layers, 2, num_kv_heads, head_dim)
    assert arr.shape == expected, (
        f"KV shape mismatch: expected {expected}, got {arr.shape}. "
        "Check kv_num_layers/kv_num_heads/kv_head_dim in config."
    )
    return arr
