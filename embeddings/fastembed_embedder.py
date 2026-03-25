"""FastEmbed-backed embedder for efficient CPU inference.

Wraps ``fastembed.TextEmbedding`` to satisfy the ``Embedder`` protocol.
Install with ``pip install fastembed``.  Models are automatically downloaded
to a local cache on first use.
"""
from fastembed import TextEmbedding


class FastEmbedEmbedder:
    """Embedder that delegates to ``fastembed.TextEmbedding``.

    FastEmbed runs entirely in-process on CPU using ONNX Runtime, making it
    a good default for environments without a GPU.

    Args:
        model_name: HuggingFace-style model identifier understood by
            fastembed (e.g. ``'BAAI/bge-small-en-v1.5'``).
        dim: Expected output dimensionality.  Must match the chosen model.
        show_progress: If ``True``, display a progress bar during model
            download.
    """

    def __init__(self, model_name: str, dim: int, show_progress: bool = False):
        self._model_name = model_name
        self._dim = dim
        self._embedder = TextEmbedding(model_name=model_name,
                                        show_download_progress=show_progress)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* and return a list of float vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input string.
        """
        return [v.tolist() for v in self._embedder.embed(texts)]

    @property
    def dim(self) -> int:
        """Dimensionality of the output vectors."""
        return self._dim
