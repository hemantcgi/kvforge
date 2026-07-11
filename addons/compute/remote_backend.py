"""RemoteComputeBackend — dispatches KV computation to a remote worker via HTTP.

Optimised for low latency:
- Batched requests reduce HTTP round-trips
- httpx Client with connection keepalive avoids reconnect overhead
- gzip compression halves payload size for base64-encoded tensors
- Callers may pipeline: submit next batch while writing previous results
"""
from __future__ import annotations

import numpy as np

import core.kv_utils as kv_utils


class RemoteComputeBackend:
    """ComputeBackend that POSTs batches to a KVForge compute worker.

    Args:
        worker_url: Base URL of the worker (e.g. "http://54.198.243.26:8091").
        api_key: Optional bearer token sent as X-API-Key header.
        timeout: Request timeout in seconds. Use a large value for big batches.
        compress: If True, send gzip-compressed request bodies.
    """

    def __init__(
        self,
        worker_url: str,
        api_key: str = "",
        timeout: float = 300.0,
        compress: bool = True,
    ) -> None:
        import httpx
        self._url = worker_url.rstrip("/")
        self._timeout = timeout
        self._compress = compress
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        if compress:
            headers["Accept-Encoding"] = "gzip"
        self._client = httpx.Client(
            headers=headers,
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    def compute_kv_batch(
        self,
        texts: list[str],
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> list[np.ndarray]:
        """POST texts to the worker; deserialize returned base64 tensors."""
        if not texts:
            return []
        resp = self._client.post(
            f"{self._url}/compute_kv",
            json={
                "texts": texts,
                "num_layers": num_layers,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        shape = (num_layers, 2, num_kv_heads, head_dim)
        return [kv_utils.deserialize_kv(t, shape) for t in data["tensors"]]

    def health(self) -> dict:
        resp = self._client.get(f"{self._url}/health", timeout=10.0)
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._client.close()
