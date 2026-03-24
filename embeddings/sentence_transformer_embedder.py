"""embeddings/sentence_transformer_embedder.py — Wraps sentence-transformers."""


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, dim: int | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "SentenceTransformerEmbedder requires: pip install sentence-transformers"
            )
        self._model = SentenceTransformer(model_name)
        self._dim = dim or self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    @property
    def dim(self) -> int:
        return self._dim
