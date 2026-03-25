"""Structural protocol for document-loading backends.

All concrete loaders (PDF, Markdown, JSONL, HTML, Directory) must satisfy the
``DocumentLoader`` protocol defined here.  This allows higher-level ingestion
code to remain agnostic of the underlying file format.
"""
from typing import Protocol, runtime_checkable


@runtime_checkable
class DocumentLoader(Protocol):
    """Protocol for objects that load documents from a source path.

    Implementations must return a list of document dicts, each containing
    at minimum a ``"text"`` key and a ``"metadata"`` dict.  The protocol is
    ``runtime_checkable`` so ``isinstance`` checks work at runtime.
    """

    def load(self, source: str) -> list[dict]:
        """Load documents from *source* and return them as structured dicts.

        Args:
            source: File path or directory path to load from.

        Returns:
            A list of document dicts, each with the shape::

                {
                    "text": str,
                    "metadata": {
                        "source": str,
                        "chunk_id": int,
                        ...  # loader-specific keys such as "page" or "section"
                    }
                }
        """
        ...
