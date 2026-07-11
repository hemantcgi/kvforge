"""Directory loader that recursively ingests all supported document types.

Dispatches to the appropriate per-format loader based on file extension.
Supported extensions: ``.pdf``, ``.md``, ``.markdown``, ``.jsonl``,
``.html``, ``.htm``, ``.docx``, ``.pptx``, ``.xlsx``, ``.zip``.
"""
from pathlib import Path


EXTENSION_MAP = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
    ".jsonl": "jsonl",
    ".html": "html",
    ".htm": "html",
    ".docx": "docx",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".zip": "zip",
}


class DirectoryLoader:
    """Walk a directory tree and load all supported document types.

    Dispatches each file to the appropriate ``DocumentLoader`` implementation
    based on its extension.  Files with unsupported extensions are silently
    skipped.

    Args:
        recursive: If ``True`` (default), descend into sub-directories using a
            ``**/*`` glob.  If ``False``, only files directly inside *source*
            are processed.
        **loader_kwargs: Additional keyword arguments forwarded to each
            per-format loader (e.g. ``chunk_size``, ``chunk_overlap``).
    """

    def __init__(self, recursive: bool = True, **loader_kwargs):
        self.recursive = recursive
        self.loader_kwargs = loader_kwargs

    def load(self, source: str) -> list[dict]:
        """Load all supported documents from *source* directory.

        Args:
            source: Path to the root directory to scan.

        Returns:
            Concatenated list of document dicts from all supported files,
            ordered by file path (sorted alphabetically).
        """
        from ingestion.registry import get_loader
        path = Path(source)
        pattern = "**/*" if self.recursive else "*"
        docs = []
        for file_path in sorted(path.glob(pattern)):
            if not file_path.is_file():
                continue
            loader_name = EXTENSION_MAP.get(file_path.suffix.lower())
            if not loader_name:
                continue
            loader = get_loader({"loader": loader_name, **self.loader_kwargs})
            docs.extend(loader.load(str(file_path)))
        return docs
