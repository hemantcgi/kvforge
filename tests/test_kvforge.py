"""Tests for kvforge.py CLI commands."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_cmd_init_creates_config(tmp_path, monkeypatch):
    """init creates a valid datasource JSON and checkpoint dir."""
    monkeypatch.chdir(tmp_path)
    import argparse
    from kvforge import cmd_init

    args = argparse.Namespace(
        name="test-corpus",
        loader="pdf",
        embed_model="BAAI/bge-small-en-v1.5",
        vector_dim=384,
        llm_model="meta-llama/Llama-3.2-3B-Instruct",
        force=False,
    )
    cmd_init(args)

    cfg_path = tmp_path / "datasource_test-corpus.json"
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text())
    assert cfg["collection"] == "test-corpus"
    assert cfg["addon_config"]["indexing"]["vector_dim"] == 384
    assert cfg["addon_config"]["indexing"]["loader"] == "pdf"
    assert (tmp_path / "test-corpus").is_dir()


def test_cmd_init_refuses_overwrite_without_force(tmp_path, monkeypatch):
    """init refuses to overwrite existing config without --force."""
    monkeypatch.chdir(tmp_path)
    import argparse, sys
    from kvforge import cmd_init

    args = argparse.Namespace(
        name="myds", loader="pdf", embed_model="BAAI/bge-small-en-v1.5",
        vector_dim=384, llm_model="meta-llama/Llama-3.2-3B-Instruct", force=False,
    )
    cmd_init(args)  # first call creates it
    with pytest.raises(SystemExit):
        cmd_init(args)  # second call without force should exit


def test_cmd_init_force_overwrites(tmp_path, monkeypatch):
    """init with --force overwrites existing config."""
    monkeypatch.chdir(tmp_path)
    import argparse
    from kvforge import cmd_init

    args = argparse.Namespace(
        name="myds", loader="markdown", embed_model="BAAI/bge-small-en-v1.5",
        vector_dim=384, llm_model="meta-llama/Llama-3.2-3B-Instruct", force=True,
    )
    cmd_init(args)
    cmd_init(args)  # should not raise
    cfg = json.loads((tmp_path / "datasource_myds.json").read_text())
    assert cfg["addon_config"]["indexing"]["loader"] == "markdown"


def test_cmd_search_calls_store_query(tmp_path, monkeypatch):
    """search embeds the query and calls store.query."""
    monkeypatch.chdir(tmp_path)
    cfg = {
        "collection": "test-col",
        "vector_store": "qdrant",
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "embed_model": "BAAI/bge-small-en-v1.5",
        "embedder_backend": "fastembed",
        "vector_dim": 384,
        "top_k": 3,
    }
    cfg_path = tmp_path / "test.json"
    cfg_path.write_text(json.dumps(cfg))

    import argparse
    from kvforge import cmd_search
    from vectorstore.base import ScoredPoint

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = [[0.1] * 384]
    mock_store = MagicMock()
    mock_store.query.return_value = [
        ScoredPoint(id=1, score=0.9, payload={"text": "hello world"})
    ]

    with patch("embeddings.registry.get_embedder", return_value=mock_embedder), \
         patch("vectorstore.registry.get_store", return_value=mock_store):
        args = argparse.Namespace(config=str(cfg_path), query="test query")
        cmd_search(args)

    mock_store.query.assert_called_once_with("test-col", [0.1] * 384, top_k=3)


def test_cmd_index_creates_collection_and_upserts(tmp_path, monkeypatch):
    """index loads, embeds, and upserts all chunks."""
    monkeypatch.chdir(tmp_path)
    cfg = {
        "collection": "idx-col",
        "vector_store": "qdrant",
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "embed_model": "BAAI/bge-small-en-v1.5",
        "embedder_backend": "fastembed",
        "vector_dim": 384,
        "upsert_batch": 10,
        "loader": "markdown",
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    import argparse
    from kvforge import cmd_index

    mock_loader = MagicMock()
    mock_loader.load.return_value = [
        {"text": "chunk one", "metadata": {"source": "f.md", "section": 0, "chunk_id": 0}},
        {"text": "chunk two", "metadata": {"source": "f.md", "section": 1, "chunk_id": 1}},
    ]
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = [[0.1] * 384, [0.2] * 384]
    mock_embedder.dim = 384
    mock_store = MagicMock()
    mock_store.collection_exists.return_value = False

    with patch("ingestion.registry.get_loader", return_value=mock_loader), \
         patch("embeddings.registry.get_embedder", return_value=mock_embedder), \
         patch("vectorstore.registry.get_store", return_value=mock_store):
        args = argparse.Namespace(config=str(cfg_path), source="/fake/path")
        cmd_index(args)

    mock_store.create_collection.assert_called_once_with("idx-col", 384)
    assert mock_store.upsert.called
