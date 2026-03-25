"""JSONL document loader that converts each line into a document dict.

Each non-empty line of the file must be a valid JSON object.  One line
becomes one document; no additional chunking is performed.
"""
import json
from pathlib import Path


class JSONLLoader:
    """Load a JSONL file where each line is a JSON object representing one document.

    A configurable key is used to extract the document text; all other keys
    in the object are preserved as metadata.

    Args:
        text_key: JSON field name that holds the document text (default
            ``'text'``).
        min_chunk_words: Lines whose text field contains fewer words than this
            threshold are skipped.
    """

    def __init__(self, text_key: str = "text", min_chunk_words: int = 5):
        self.text_key = text_key
        self.min_chunk_words = min_chunk_words

    def load(self, source: str) -> list[dict]:
        """Read a JSONL file and return each qualifying line as a document dict.

        Args:
            source: Path to the ``.jsonl`` file.

        Returns:
            List of dicts with the shape::

                {
                    "text": str,
                    "metadata": {
                        "source": str,     # filename (not full path)
                        "chunk_id": int,   # 0-indexed line number (blank lines excluded)
                        ...                # all other JSON keys from the line
                    }
                }
        """
        path = Path(source)
        docs = []
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get(self.text_key, "")
                if not text or len(text.split()) < self.min_chunk_words:
                    continue
                metadata = {k: v for k, v in obj.items() if k != self.text_key}
                metadata.update({"source": path.name, "chunk_id": line_num - 1})
                docs.append({"text": text, "metadata": metadata})
        return docs
