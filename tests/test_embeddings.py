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
