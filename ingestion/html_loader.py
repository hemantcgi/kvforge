"""ingestion/html_loader.py — Load HTML files, strip tags, split by section."""
from pathlib import Path


class HTMLLoader:
    """Strip HTML tags and return text content as chunks."""

    def __init__(self, min_chunk_words: int = 10):
        self.min_chunk_words = min_chunk_words

    def load(self, source: str) -> list[dict]:
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
