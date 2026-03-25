"""Markdown document loader that splits files into sections by heading level.

Uses a simple regex split on ``#``, ``##``, and ``###`` headings to produce
one chunk per section.  No external dependencies required.
"""
import re
from pathlib import Path


class MarkdownLoader:
    """Load a Markdown file and split it into per-heading sections.

    The file is split on level-1 through level-3 headings (``#``, ``##``,
    ``###``).  The heading text itself is not included in the chunk — only
    the body content that follows it.

    Args:
        min_chunk_words: Sections with fewer words than this threshold are
            discarded.
    """

    def __init__(self, min_chunk_words: int = 10):
        self.min_chunk_words = min_chunk_words

    def load(self, source: str) -> list[dict]:
        """Read a Markdown file and return each section as a document dict.

        Args:
            source: Path to the Markdown file.

        Returns:
            List of dicts with the shape::

                {
                    "text": str,
                    "metadata": {
                        "source": str,    # filename (not full path)
                        "section": int,   # 0-indexed section number
                        "chunk_id": int   # same as section index
                    }
                }
        """
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
