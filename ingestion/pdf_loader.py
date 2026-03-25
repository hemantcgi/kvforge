"""PDF document loader that splits pages into overlapping word-based chunks.

Depends on ``pypdf`` for text extraction.  Install with ``pip install pypdf``.
"""
from pathlib import Path
from pypdf import PdfReader


class PDFLoader:
    """Load a PDF file and split its text into overlapping word-count chunks.

    Each page is extracted as plain text, tokenised by whitespace, and then
    sliced into windows of ``chunk_size`` words with ``chunk_overlap`` words
    of overlap between consecutive windows.  Short fragments are dropped.

    Args:
        chunk_size: Target size of each chunk in words.
        chunk_overlap: Number of words shared between consecutive chunks.
        min_chunk_words: Chunks shorter than this threshold are discarded.
    """

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 60,
                 min_chunk_words: int = 30):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_words = min_chunk_words

    def load(self, source: str) -> list[dict]:
        """Read a PDF file and return its text as a list of chunk dicts.

        Args:
            source: Path to the PDF file.

        Returns:
            List of dicts with the shape::

                {
                    "text": str,
                    "metadata": {
                        "page": int,       # 1-indexed page number
                        "source": str,     # filename (not full path)
                        "chunk_id": int    # global 0-indexed chunk counter
                    }
                }
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
