import json
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class CredentialStore(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> None: ...
    def delete(self, key: str) -> None: ...


class LocalFileCredentialStore:
    """Store credentials as a JSON file on the local filesystem.

    Not encrypted — suitable for development and self-hosted deployments
    where the file is protected by OS-level permissions.
    """

    def __init__(self, path: str = "~/.kvforge/credentials.json"):
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text())

    def _save(self, data: dict) -> None:
        import tempfile, os
        tmp = self._path.parent / (self._path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self._path)

    def get(self, key: str) -> str | None:
        return self._load().get(key)

    def set(self, key: str, value: str) -> None:
        data = self._load()
        data[key] = value
        self._save(data)

    def delete(self, key: str) -> None:
        data = self._load()
        data.pop(key, None)
        self._save(data)
