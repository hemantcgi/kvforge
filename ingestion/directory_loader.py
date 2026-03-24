"""ingestion/directory_loader.py — Recursively load all supported docs in a dir."""
from pathlib import Path


EXTENSION_MAP = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
    ".jsonl": "jsonl",
    ".html": "html",
    ".htm": "html",
}


class DirectoryLoader:
    """Walk a directory and load all supported document types."""

    def __init__(self, recursive: bool = True, **loader_kwargs):
        self.recursive = recursive
        self.loader_kwargs = loader_kwargs

    def load(self, source: str) -> list[dict]:
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
