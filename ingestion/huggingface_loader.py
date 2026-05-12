"""HuggingFace datasets ingestion loader."""
from __future__ import annotations
import os

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None  # patched in tests; real usage requires `pip install datasets`


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
        min_chunk_words: int = 5,
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

        docs = []
        for i, row in enumerate(ds):
            text = row.get(self._text_column, "")
            if not isinstance(text, str):
                text = str(text)
            if len(text.split()) < self._min_chunk_words:
                continue
            metadata = {k: v for k, v in row.items() if k != self._text_column}
            metadata.update({"source": self._dataset_id, "chunk_id": i})
            docs.append({"text": text, "metadata": metadata})
        return docs
