"""ingestion/base.py — DocumentLoader Protocol."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class DocumentLoader(Protocol):
    def load(self, source: str) -> list[dict]:
        """Load documents from source.

        Returns:
            List of dicts: [{"text": str, "metadata": {"source": str, ...}}, ...]
        """
        ...
