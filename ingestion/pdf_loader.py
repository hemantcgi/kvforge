"""ingestion/pdf_loader.py — Load and chunk PDF files."""
from pathlib import Path
from pypdf import PdfReader


class PDFLoader:
    """Load a PDF and split it into overlapping word-based chunks."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 60,
                 min_chunk_words: int = 30):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_words = min_chunk_words

    def load(self, source: str) -> list[dict]:
        """Read a PDF file and return chunks as document dicts.

        Each dict: {"text": str, "metadata": {"page": int, "source": str, "chunk_id": int}}
        """
        path = Path(source)
        reader = PdfReader(str(path))
        docs = []
        step = max(self.chunk_size - self.chunk_overlap, 1)
        chunk_id = 0
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            words = text.split()
            for start in range(0, len(words), step):
                chunk_words = words[start: start + self.chunk_size]
                if len(chunk_words) < self.min_chunk_words:
                    continue
                docs.append({
                    "text": " ".join(chunk_words),
                    "metadata": {
                        "page": page_num,
                        "source": path.name,
                        "chunk_id": chunk_id,
                    },
                })
                chunk_id += 1
        return docs
