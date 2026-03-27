"""Thin client for a vLLM inference server (OpenAI-compatible REST API).

vLLM achieves 6-10× higher decode throughput than HuggingFace transformers
for batch_size=1 inference by using CUDA graphs, continuous batching, and
PagedAttention.  This module exposes a ``generate()`` helper so the rest of
the codebase treats vLLM as a drop-in replacement for local model.generate().

Typical deployment (one server per use case / GPU):

    CUDA_VISIBLE_DEVICES=3 vllm serve meta-llama/Llama-3.2-3B-Instruct \\
        --enable-lora \\
        --lora-modules uc4=examples/usecase4_bedrock_userguide/lora_checkpoints/v3/ \\
        --max-lora-rank 16 \\
        --port 8090 \\
        --gpu-memory-utilization 0.85 \\
        --max-model-len 4096

Config fields used (add to the relevant config.json):

    "vllm_url"   : "http://localhost:8090"   <- port where vLLM is listening
    "vllm_model" : "uc4"                     <- LoRA module name, or model ID
                                                for base-model-only serving
"""

from __future__ import annotations

import httpx


def is_healthy(url: str, timeout: float = 5.0) -> bool:
    """Return True if the vLLM server at *url* is up and responding."""
    try:
        resp = httpx.get(f"{url}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def generate(
    prompt: str,
    *,
    url: str,
    model: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    stop: list[str] | None = None,
    timeout: int = 120,
) -> str:
    """Generate text via vLLM's OpenAI-compatible ``/v1/completions`` endpoint.

    Args:
        prompt: The full prompt string (pre-formatted, including any chat
            template already applied by the caller).
        url: Base URL of the vLLM server, e.g. ``"http://localhost:8090"``.
        model: Model/LoRA identifier to pass as the ``model`` field.  When
            ``--enable-lora`` is used, this is the LoRA module name supplied
            at server startup.
        max_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (0.0 = greedy, deterministic).
        stop: Optional list of stop strings.
        timeout: HTTP request timeout in seconds.

    Returns:
        Generated text with leading/trailing whitespace stripped.

    Raises:
        httpx.HTTPStatusError: If the server returns a non-2xx response.
        httpx.TimeoutException: If the server does not respond within *timeout*.
    """
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop:
        payload["stop"] = stop

    resp = httpx.post(
        f"{url}/v1/completions",
        json=payload,
        headers={"Authorization": "Bearer EMPTY"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"].strip()
