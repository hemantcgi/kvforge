"""HTML document loader that strips tags and splits content into sections.

Uses BeautifulSoup4 to parse HTML.  Install with ``pip install beautifulsoup4``.
Sections are delimited by block-level heading tags (``h1``, ``h2``, ``h3``).
"""
from pathlib import Path


class HTMLLoader:
    """Load an HTML file, strip markup, and split the text into heading-delimited sections.

    Block-level elements (``p``, ``h1``–``h4``, ``li``, ``td``) are extracted
    in document order.  A new section is started whenever an ``h1``, ``h2``,
    or ``h3`` tag is encountered.

    Args:
        min_chunk_words: Sections with fewer words than this threshold are
            discarded.

    Raises:
        ImportError: If ``beautifulsoup4`` is not installed.
    """

    def __init__(self, min_chunk_words: int = 10):
        self.min_chunk_words = min_chunk_words

    def load(self, source: str) -> list[dict]:
        """Parse an HTML file and return each section as a document dict.

        Args:
            source: Path to the HTML file.

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
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("HTMLLoader requires beautifulsoup4: pip install beautifulsoup4")
        path = Path(source)
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        sections = []
        current = []
        for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "li", "td"]):
            text = tag.get_text(separator=" ", strip=True)
            if not text:
                continue
            if tag.name in ("h1", "h2", "h3") and current:
                sections.append(" ".join(current))
                current = [text]
            else:
                current.append(text)
        if current:
            sections.append(" ".join(current))

        docs = []
        for i, section in enumerate(sections):
            if len(section.split()) < self.min_chunk_words:
                continue
            docs.append({
                "text": section,
                "metadata": {"source": path.name, "section": i, "chunk_id": i},
            })
        return docs
