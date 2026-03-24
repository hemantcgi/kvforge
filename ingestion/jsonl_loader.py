"""ingestion/jsonl_loader.py — Load JSONL files (one JSON object per line)."""
import json
from pathlib import Path


class JSONLLoader:
    """Each line of the JSONL file becomes one document."""

    def __init__(self, text_key: str = "text", min_chunk_words: int = 5):
        self.text_key = text_key
        self.min_chunk_words = min_chunk_words

    def load(self, source: str) -> list[dict]:
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
