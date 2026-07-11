from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable


@dataclass
class SourceFile:
    """Connector-agnostic representation of a remote file."""
    id: str
    name: str
    path: str
    size: int
    modified_at: datetime
    mime_type: str | None = None
    extra: dict = field(default_factory=dict)


@runtime_checkable
class SourceConnector(Protocol):
    """Protocol all source connectors must implement."""

    def list_files(self) -> list[SourceFile]: ...
    def download(self, file: SourceFile) -> bytes: ...
    def get_modified_at(self, file: SourceFile) -> datetime: ...
    def supports_delta(self) -> bool: ...
    def get_delta(self, token: str | None) -> tuple[list[SourceFile], str]: ...
