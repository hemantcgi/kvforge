"""HuggingFace datasets ingestion loader."""
from __future__ import annotations
import os
import re

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None  # patched in tests; real usage requires `pip install datasets`

# Matches wikitext section headers like "= = Heading = =" or "= Title ="
_WIKITEXT_HEADER_RE = re.compile(r"^\s*=+\s+.*\s+=+\s*$")


class HuggingFaceLoader:
    def __init__(
        self,
        dataset_id: str,
        config_name: str | None = None,
        split: str = "train",
        text_column: str = "text",
        max_rows: int = 0,
        hf_token: str | None = None,
        trust_remote_code: bool = False,
        min_chunk_words: int = 30,
        target_chunk_words: int = 250,
    ):
        self._dataset_id = dataset_id
        self._config_name = config_name
        self._split = split
        self._text_column = text_column
        self._max_rows = max_rows
        self._hf_token = (
            hf_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        self._trust_remote_code = trust_remote_code
        self._min_chunk_words = min_chunk_words
        self._target_chunk_words = target_chunk_words

    def load(self, source: str = "") -> list[dict]:
        kwargs: dict = {
            "split": self._split,
            "trust_remote_code": self._trust_remote_code,
        }
        if self._config_name:
            kwargs["name"] = self._config_name
        if self._hf_token:
            kwargs["token"] = self._hf_token

        ds = load_dataset(self._dataset_id, **kwargs)

        if self._max_rows and self._max_rows > 0:
            ds = ds.select(range(min(self._max_rows, len(ds))))

        # Collect all text rows
        raw_texts = []
        for row in ds:
            text = row.get(self._text_column, "")
            if not isinstance(text, str):
                text = str(text)
            raw_texts.append(text)

        # Aggregate into paragraph-level chunks
        chunks = self._aggregate(raw_texts)
        return chunks

    def _aggregate(self, rows: list[str]) -> list[dict]:
        """Aggregate line-by-line rows into target-size text chunks.

        - Skips wikitext-style section headers (= = Heading = =)
        - Groups non-empty lines into chunks until target_chunk_words is reached
        - Discards chunks below min_chunk_words
        """
        chunks = []
        buffer: list[str] = []
        buffer_words = 0
        chunk_index = 0

        def flush():
            nonlocal chunk_index
            text = " ".join(buffer).strip()
            # Collapse multiple spaces
            text = re.sub(r" {2,}", " ", text)
            if len(text.split()) >= self._min_chunk_words:
                chunks.append({
                    "text": text,
                    "metadata": {
                        "source": self._dataset_id,
                        "chunk_id": chunk_index,
                        "page": 0,
                    },
                })
                chunk_index += 1
            buffer.clear()

        for row in rows:
            line = row.strip()

            # Skip empty lines — treat as soft paragraph boundary
            if not line:
                if buffer_words >= self._min_chunk_words:
                    flush()
                    buffer_words = 0
                continue

            # Skip wikitext section headers (they're not content)
            if _WIKITEXT_HEADER_RE.match(line):
                if buffer_words >= self._min_chunk_words:
                    flush()
                    buffer_words = 0
                continue

            words_in_line = len(line.split())

            # If adding this line would exceed target, flush first
            if buffer_words + words_in_line > self._target_chunk_words and buffer_words >= self._min_chunk_words:
                flush()
                buffer_words = 0

            buffer.append(line)
            buffer_words += words_in_line

        # Flush remaining
        if buffer_words >= self._min_chunk_words:
            flush()

        return chunks
