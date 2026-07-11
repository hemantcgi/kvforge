# tests/test_kv_indexer_cli.py
"""Test CLI arg parsing and config-flattening in kv_indexer.main()."""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest


_NESTED_CFG = {
    "use_case_name": "test",
    "collection": "col",
    "version_file": "/tmp/ver.json",
    "addons": ["indexing", "inference"],
    "addon_config": {
        "indexing": {
            "loader": "jsonl",
            "jsonl_text_key": "text",
            "chunk_size": 300,
            "chunk_overlap": 30,
            "embed_batch": 8,
            "upsert_batch": 16,
            "embed_model": "BAAI/bge-small-en-v1.5",
            "embedder_backend": "fastembed",
            "vector_dim": 384,
            "vector_store": "qdrant",
            "qdrant_host": "localhost",
            "qdrant_port": 6333,
            "model_library": {
                "meta-llama/Llama-3.2-3B-Instruct": {
                    "kv_num_layers": 2, "kv_num_heads": 4, "kv_head_dim": 64
                }
            },
        },
        "inference": {
            "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
            "quantization": "4bit",
        },
    },
}


def test_nested_config_is_flattened(tmp_path, monkeypatch):
    """main() must flatten addon_config before calling cmd_index."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(_NESTED_CFG))
    corpus = tmp_path / "data" / "corpus.jsonl"
    corpus.parent.mkdir()
    corpus.write_text('{"text": "enough words here to pass the threshold filter"}\n')

    captured = {}

    def fake_cmd_index(cfg):
        captured["cfg"] = cfg

    monkeypatch.setattr("pipeline.kv_indexer.cmd_index", fake_cmd_index)
    monkeypatch.setattr("pipeline.kv_indexer.ver.init", lambda cfg: None)
    monkeypatch.setattr("pipeline.kv_indexer.model_loader.init", lambda cfg: None)

    sys.argv = ["kv_indexer", "--config", str(cfg_file), "index"]
    # kv_indexer resolves corpus.jsonl relative to the config file's directory
    from pipeline import kv_indexer
    kv_indexer.main()

    assert "chunk_size" in captured["cfg"], "flat key chunk_size must be present after merge"
    assert captured["cfg"]["loader"] == "jsonl"


def test_pdf_file_arg_optional_no_error(tmp_path, monkeypatch):
    """Passing 'index' with no positional arg must not raise argparse error."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(_NESTED_CFG))

    monkeypatch.setattr("pipeline.kv_indexer.cmd_index", lambda cfg: None)
    monkeypatch.setattr("pipeline.kv_indexer.ver.init", lambda cfg: None)
    monkeypatch.setattr("pipeline.kv_indexer.model_loader.init", lambda cfg: None)

    sys.argv = ["kv_indexer", "--config", str(cfg_file), "index"]
    from pipeline import kv_indexer
    kv_indexer.main()  # must not raise SystemExit


def test_pipeline_runner_index_cmd_no_pdf_arg():
    """pipeline_runner._build_cmd for 'index' must not include a bare pdf_file positional."""
    from studio.pipeline_runner import _build_cmd
    cmd = _build_cmd("usecase1_customer_support", "index")
    # Last arg should NOT be a bare file path that doesn't start with '--'
    positional_extra = [a for a in cmd[5:] if not a.startswith("-") and a != "index"]
    assert positional_extra == [], f"Unexpected positional args after 'index': {positional_extra}"
