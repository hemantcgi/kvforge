"""vectorstore/base.py — VectorStore Protocol + shared dataclasses."""
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Point:
    id: int | str
    vector: list[float]
    payload: dict = field(default_factory=dict)


@dataclass
class ScoredPoint:
    id: int | str
    score: float
    payload: dict = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    def create_collection(self, name: str, dim: int) -> None: ...
    def collection_exists(self, name: str) -> bool: ...
    def delete_collection(self, name: str) -> None: ...
    def upsert(self, collection: str, points: list[Point]) -> None: ...
    def query(self, collection: str, vector: list[float], top_k: int,
               score_threshold: float | None = None) -> list[ScoredPoint]: ...
    def scroll(self, collection: str, limit: int = 100,
                with_payload: bool = True, with_vectors: bool = False,
                offset: Any = None, scroll_filter: Any = None) -> tuple[list, Any]: ...
    def set_payload(self, collection: str, point_id: int | str, payload: dict) -> None: ...
    def count(self, collection: str) -> int: ...
