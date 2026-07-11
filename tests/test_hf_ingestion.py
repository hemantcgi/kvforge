# tests/test_hf_ingestion.py
import pytest
from unittest.mock import patch


def test_get_loader_returns_huggingface_loader():
    from ingestion.registry import get_loader
    from ingestion.huggingface_loader import HuggingFaceLoader
    loader = get_loader({"loader": "huggingface", "dataset_id": "qiaojin/PubMedQA"})
    assert isinstance(loader, HuggingFaceLoader)


def test_huggingface_loader_load_filters_short():
    from ingestion.huggingface_loader import HuggingFaceLoader
    fake_rows = [
        {"text": "Short.", "extra": "a"},
        {"text": "This is a long enough medical abstract about clinical outcomes in patients.", "extra": "b"},
    ]
    with patch("ingestion.huggingface_loader.load_dataset") as mock_ld:
        mock_ld.return_value = fake_rows
        loader = HuggingFaceLoader(dataset_id="qiaojin/PubMedQA", config_name="pqa_labeled")
        docs = loader.load()
    assert len(docs) == 1
    assert "This is a long enough" in docs[0]["text"]
    assert docs[0]["metadata"]["extra"] == "b"


def test_huggingface_loader_load_respects_max_rows():
    from ingestion.huggingface_loader import HuggingFaceLoader

    class _FakeDS(list):
        def select(self, indices):
            return [self[i] for i in indices]

    fake_ds = _FakeDS([
        {"text": f"Long enough text for row number {i} with sufficient word count.", "idx": i}
        for i in range(10)
    ])
    with patch("ingestion.huggingface_loader.load_dataset") as mock_ld:
        mock_ld.return_value = fake_ds
        loader = HuggingFaceLoader(dataset_id="test/ds", max_rows=3)
        docs = loader.load()
    assert len(docs) == 3


def test_get_loader_unknown_raises():
    from ingestion.registry import get_loader
    with pytest.raises(ValueError, match="Unknown loader"):
        get_loader({"loader": "unknown_xyz_loader"})
