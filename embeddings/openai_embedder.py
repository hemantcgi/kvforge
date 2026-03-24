"""embeddings/openai_embedder.py — Wraps OpenAI Embeddings API."""
import os


class OpenAIEmbedder:
    def __init__(self, model_name: str = "text-embedding-3-small", dim: int = 1536,
                 api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAIEmbedder requires: pip install openai")
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._model_name = model_name
        self._dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(input=texts, model=self._model_name)
        return [item.embedding for item in resp.data]

    @property
    def dim(self) -> int:
        return self._dim
