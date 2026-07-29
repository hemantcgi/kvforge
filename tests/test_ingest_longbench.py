"""Tests for tools.ingest_longbench."""

import json
import os
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import ingest_longbench


@pytest.fixture
def fake_zip(tmp_path):
    """Create a minimal LongBench-style zip with one JSONL task."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    jsonl = data_dir / "2wikimqa.jsonl"
    sample = {
        "_id": "abc123",
        "input": "Where was Ozalj?",
        "context": (
            "Passage 1:\nFirst passage text.\n"
            "Passage 2:\nSecond passage text."
        ),
        "answers": ["Ozalj"],
        "dataset": "2wikimqa",
        "length": 1234,
    }
    with open(jsonl, "w") as f:
        f.write(json.dumps(sample) + "\n")

    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(jsonl, "data/2wikimqa.jsonl")
    return zip_path


def test_split_passages_multiple():
    ctx = "Passage 1:\nAlpha\nPassage 2:\nBeta\nPassage 3:\nGamma"
    assert ingest_longbench.split_passages(ctx) == ["Alpha", "Beta", "Gamma"]


def test_split_passages_no_header():
    assert ingest_longbench.split_passages("Just text") == ["Just text"]


def test_build_kvforge_dataset(fake_zip, tmp_path):
    extract_dir = tmp_path / "extracted"
    output_dir = tmp_path / "out"
    jsonl_path = ingest_longbench.extract_jsonl(fake_zip, "2wikimqa", extract_dir)
    ingest_longbench.build_kvforge_dataset("2wikimqa", jsonl_path, output_dir)

    assert (output_dir / "data" / "2wikimqa.chunks.json").exists()
    assert (output_dir / "eval_2wikimqa.json").exists()
    assert (output_dir / "config.json").exists()

    with open(output_dir / "data" / "2wikimqa.chunks.json") as f:
        chunks = json.load(f)
    assert chunks["chunks"] == ["First passage text.", "Second passage text."]

    with open(output_dir / "eval_2wikimqa.json") as f:
        eval_data = json.load(f)
    assert eval_data["n_items"] == 1
    assert eval_data["items"][0]["question"] == "Where was Ozalj?"
    assert eval_data["items"][0]["answer"] == "Ozalj"
    assert eval_data["items"][0]["all_answers"] == ["Ozalj"]


def test_build_kvforge_dataset_config_fields(fake_zip, tmp_path):
    extract_dir = tmp_path / "extracted"
    output_dir = tmp_path / "out"
    jsonl_path = ingest_longbench.extract_jsonl(fake_zip, "2wikimqa", extract_dir)
    ingest_longbench.build_kvforge_dataset(
        "2wikimqa", jsonl_path, output_dir,
        llm_model="meta-llama/Llama-3.2-3B-Instruct"
    )
    with open(output_dir / "config.json") as f:
        cfg = json.load(f)
    assert cfg["collection"] == "longbench-2wikimqa"
    assert cfg["addon_config"]["inference"]["llm_model"] == "meta-llama/Llama-3.2-3B-Instruct"


@patch("tools.ingest_longbench.hf_hub_download")
def test_download_longbench(mock_hf_hub_download, tmp_path):
    mock_hf_hub_download.return_value = str(tmp_path / "data.zip")
    zip_path = ingest_longbench.download_longbench(tmp_path)
    assert str(zip_path) == str(tmp_path / "data.zip")
    mock_hf_hub_download.assert_called_once()
