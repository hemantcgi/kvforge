# tests/test_curation_manager.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch
import studio.curation_manager as cm


@pytest.fixture
def uc_dir(tmp_path):
    d = tmp_path / "examples" / "test-uc"
    d.mkdir(parents=True)
    return tmp_path


def test_append_creates_file(uc_dir):
    with patch.object(cm, "ROOT", uc_dir):
        result = cm.append("test-uc", "What is RAG?", "RAG is ...", "model_b")
    assert (uc_dir / "examples" / "test-uc" / "faqs_curated.json").exists()
    assert result["count"] == 1


def test_append_increments_count(uc_dir):
    with patch.object(cm, "ROOT", uc_dir):
        cm.append("test-uc", "Q1", "A1")
        result = cm.append("test-uc", "Q2", "A2")
    assert result["count"] == 2


def test_append_stores_fields(uc_dir):
    with patch.object(cm, "ROOT", uc_dir):
        cm.append("test-uc", "Q?", "A.", "model_b")
        records = json.loads((uc_dir / "examples" / "test-uc" / "faqs_curated.json").read_text())
    assert records[0]["question"] == "Q?"
    assert records[0]["answer"] == "A."
    assert records[0]["source_model"] == "model_b"
    assert "curated_at" in records[0]


def test_get_status_empty(uc_dir):
    with patch.object(cm, "ROOT", uc_dir), \
         patch("studio.settings_manager.SETTINGS_FILE", uc_dir / "settings.json"):
        status = cm.get_status("test-uc")
    assert status["count"] == 0
    assert status["threshold"] == 50
    assert status["at_threshold"] is False


def test_get_status_at_threshold(uc_dir):
    with patch.object(cm, "ROOT", uc_dir), \
         patch("studio.settings_manager.SETTINGS_FILE", uc_dir / "settings.json"):
        for i in range(50):
            cm.append("test-uc", f"Q{i}", f"A{i}")
        status = cm.get_status("test-uc")
    assert status["at_threshold"] is True
    assert status["pct"] == 100.0


def test_get_samples_returns_last_n(uc_dir):
    with patch.object(cm, "ROOT", uc_dir):
        for i in range(10):
            cm.append("test-uc", f"Q{i}", f"A{i}")
        samples = cm.get_samples("test-uc", n=3)
    assert len(samples) == 3
    assert samples[-1]["question"] == "Q9"


def test_write_is_atomic(uc_dir):
    with patch.object(cm, "ROOT", uc_dir):
        cm.append("test-uc", "Q", "A")
        assert not (uc_dir / "examples" / "test-uc" / "faqs_curated.tmp").exists()


def test_path_traversal_rejected(uc_dir):
    with patch.object(cm, "ROOT", uc_dir):
        with pytest.raises(ValueError, match="escapes examples directory"):
            cm.get_status("../../etc")
