"""ingestion/markdown_loader.py — Load Markdown files, split by heading."""
import re
from pathlib import Path


class MarkdownLoader:
    """Split a Markdown file into sections at each heading (# / ## / ###)."""

    def __init__(self, min_chunk_words: int = 10):
        self.min_chunk_words = min_chunk_words

    def load(self, source: str) -> list[dict]:
        path = Path(source)
        text = path.read_text(encoding="utf-8")
        sections = re.split(r"(?m)^#{1,3}\s+", text)
        docs = []
        for i, section in enumerate(sections):
            clean = section.strip()
            if not clean or len(clean.split()) < self.min_chunk_words:
                continue
            docs.append({
                "text": clean,
                "metadata": {"source": path.name, "section": i, "chunk_id": i},
            })
        return docs
