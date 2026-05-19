"""Compute addon — pluggable KV tensor computation backend.

Supports local (in-process) and remote (HTTP worker) backends.
Configure via addon_config.compute in the use-case config.
"""
from __future__ import annotations

from pydantic import BaseModel

from addons.registry import AddonManifest, AddonRegistry


class ComputeConfig(BaseModel):
    backend: str = "local"          # "local" or "remote"
    worker_url: str = ""            # required when backend == "remote"
    api_key: str = ""               # X-API-Key sent to remote worker
    batch_size: int = 16            # chunks per HTTP request to remote worker
    timeout_seconds: float = 300.0  # remote worker request timeout
    compress: bool = True           # gzip request bodies to remote worker


def get_backend(cfg: dict):
    """Return the appropriate ComputeBackend for this use-case config.

    Falls back to LocalComputeBackend if the compute addon is not configured
    or if backend is "local".

    Args:
        cfg: Full use-case config dict (addon_config lives under cfg["addon_config"]).

    Returns:
        An object satisfying the ComputeBackend Protocol.
    """
    compute_cfg = cfg.get("addon_config", {}).get("compute", {})
    backend = compute_cfg.get("backend", "local")

    if backend == "remote":
        from addons.compute.remote_backend import RemoteComputeBackend
        return RemoteComputeBackend(
            worker_url=compute_cfg["worker_url"],
            api_key=compute_cfg.get("api_key", ""),
            timeout=compute_cfg.get("timeout_seconds", 300.0),
            compress=compute_cfg.get("compress", True),
        )

    from addons.compute.local_backend import LocalComputeBackend
    return LocalComputeBackend(cfg)


AddonRegistry.register(
    AddonManifest(
        name="compute",
        display_name="Compute Backend",
        description="Pluggable KV tensor computation — local GPU or remote worker.",
        config_schema=ComputeConfig,
    )
)
