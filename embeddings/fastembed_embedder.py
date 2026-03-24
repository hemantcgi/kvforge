"""embeddings/fastembed_embedder.py — Wraps fastembed.TextEmbedding."""
from fastembed import TextEmbedding


class FastEmbedEmbedder:
    def __init__(self, model_name: str, dim: int, show_progress: bool = False):
        self._model_name = model_name
        self._dim = dim
        self._embedder = TextEmbedding(model_name=model_name,
                                        show_download_progress=show_progress)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._embedder.embed(texts)]

    @property
    def dim(self) -> int:
        return self._dim
