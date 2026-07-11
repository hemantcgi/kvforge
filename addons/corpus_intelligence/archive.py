"""Archive backend protocol and implementations.

Backends store chunk text outside the vector store.
The vector store retains only the embedding + a pointer string.
"""
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ArchiveBackend(Protocol):
    def write(self, chunk_id: str, text: str) -> None: ...
    def read(self, chunk_id: str) -> str | None: ...
    def delete(self, chunk_id: str) -> None: ...
    def get_pointer(self, chunk_id: str) -> str: ...


class LocalArchiveBackend:
    """Stores archived chunk text as plain .txt files on local filesystem."""

    def __init__(self, archive_dir: str):
        self._dir = Path(archive_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, chunk_id: str) -> Path:
        safe_id = chunk_id.replace("/", "_")
        return self._dir / f"{safe_id}.txt"

    def write(self, chunk_id: str, text: str) -> None:
        self._path(chunk_id).write_text(text, encoding="utf-8")

    def read(self, chunk_id: str) -> str | None:
        p = self._path(chunk_id)
        return p.read_text(encoding="utf-8") if p.exists() else None

    def delete(self, chunk_id: str) -> None:
        p = self._path(chunk_id)
        if p.exists():
            p.unlink()

    def get_pointer(self, chunk_id: str) -> str:
        return str(self._path(chunk_id))


def get_backend(cfg) -> ArchiveBackend:
    """Factory — returns the configured archive backend."""
    backend_name = getattr(cfg, "archive_backend", "local")
    if backend_name == "local":
        return LocalArchiveBackend(cfg.archive_dir)
    raise ValueError(f"Unknown archive backend: {backend_name!r}. "
                     "Supported: 'local'.")
