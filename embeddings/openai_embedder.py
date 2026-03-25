"""OpenAI Embeddings API-backed embedder.

Wraps the ``openai`` Python client to satisfy the ``Embedder`` protocol.
Install with ``pip install openai``.  An API key must be provided either via
the ``api_key`` constructor argument or the ``OPENAI_API_KEY`` environment
variable.
"""
import os


class OpenAIEmbedder:
    """Embedder that delegates to the OpenAI Embeddings API.

    Each call to ``encode`` makes a single batched API request.  Network
    latency and token costs apply.

    Args:
        model_name: OpenAI embedding model name (e.g.
            ``'text-embedding-3-small'`` or ``'text-embedding-ada-002'``).
        dim: Expected output dimensionality.  Must match the chosen model
            (``text-embedding-3-small`` produces 1536-dimensional vectors).
        api_key: OpenAI API key.  If ``None``, the value of the
            ``OPENAI_API_KEY`` environment variable is used.

    Raises:
        ImportError: If the ``openai`` package is not installed.
        KeyError: If *api_key* is ``None`` and ``OPENAI_API_KEY`` is not set.
    """

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
        """Embed *texts* via a single batched call to the OpenAI Embeddings API.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input string.
        """
        resp = self._client.embeddings.create(input=texts, model=self._model_name)
        return [item.embedding for item in resp.data]

    @property
    def dim(self) -> int:
        """Dimensionality of the output vectors."""
        return self._dim
