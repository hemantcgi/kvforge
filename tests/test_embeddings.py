"""Tests for embedding dimension validation."""
import pytest
from unittest.mock import MagicMock


def test_validation_passes_when_dims_match():
    from bedrock_rag import validate_embed_dim
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = iter([[0.1] * 1024])

    class FakeCfg:
        embed_model = "some-model"
        vector_dim = 1024

    validate_embed_dim(mock_embedder, FakeCfg())  # should not raise


def test_validation_fails_when_dims_mismatch():
    from bedrock_rag import validate_embed_dim
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = iter([[0.1] * 384])

    class FakeCfg:
        embed_model = "some-model"
        vector_dim = 1024

    with pytest.raises(ValueError, match="produces 384-dim"):
        validate_embed_dim(mock_embedder, FakeCfg())


def test_embedder_protocol_has_required_interface():
    from embeddings.fastembed_embedder import FastEmbedEmbedder
    assert hasattr(FastEmbedEmbedder, "encode")
    assert hasattr(FastEmbedEmbedder, "dim")


def test_fastembed_embedder_encode_returns_correct_shape():
    from embeddings.fastembed_embedder import FastEmbedEmbedder
    from unittest.mock import patch, MagicMock
    with patch("embeddings.fastembed_embedder.TextEmbedding") as mock_cls:
        import numpy as np
        mock_instance = MagicMock()
        mock_instance.embed.return_value = iter([np.array([0.1] * 384), np.array([0.2] * 384)])
        mock_cls.return_value = mock_instance
        embedder = FastEmbedEmbedder(model_name="BAAI/bge-small-en-v1.5", dim=384)
        result = embedder.encode(["text one", "text two"])
    assert len(result) == 2
    assert len(result[0]) == 384


def test_registry_returns_fastembed_by_default():
    from embeddings.registry import get_embedder
    from embeddings.fastembed_embedder import FastEmbedEmbedder
    from unittest.mock import patch
    with patch("embeddings.fastembed_embedder.TextEmbedding"):
        cfg = {"embed_model": "BAAI/bge-small-en-v1.5", "vector_dim": 384}
        embedder = get_embedder(cfg)
    assert isinstance(embedder, FastEmbedEmbedder)
