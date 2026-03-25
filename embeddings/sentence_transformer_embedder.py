"""Sentence-Transformers-backed embedder.

Wraps ``sentence_transformers.SentenceTransformer`` to satisfy the
``Embedder`` protocol.  Install with ``pip install sentence-transformers``.
"""


class SentenceTransformerEmbedder:
    """Embedder that delegates to a ``sentence_transformers.SentenceTransformer`` model.

    Supports any model available on HuggingFace Hub or locally.  Inference
    runs on GPU if available, otherwise CPU.

    Args:
        model_name: HuggingFace model identifier or local path (e.g.
            ``'sentence-transformers/all-MiniLM-L6-v2'``).
        dim: Expected output dimensionality.  If ``None``, it is read from
            the model configuration via
            ``get_sentence_embedding_dimension()``.

    Raises:
        ImportError: If ``sentence-transformers`` is not installed.
    """

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
        """Embed *texts* using the loaded SentenceTransformer model.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors as Python float lists, one per input
            string.
        """
        return self._model.encode(texts, show_progress_bar=False).tolist()

    @property
    def dim(self) -> int:
        """Dimensionality of the output vectors."""
        return self._dim
