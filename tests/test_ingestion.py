"""Tests for document loader abstraction."""
import pytest
from unittest.mock import patch, MagicMock


def test_document_loader_protocol_is_satisfied_by_pdf_loader():
    from ingestion.pdf_loader import PDFLoader
    from ingestion.base import DocumentLoader
    assert hasattr(PDFLoader, "load")


def test_pdf_loader_returns_list_of_dicts_with_text_and_metadata(tmp_path):
    from ingestion.pdf_loader import PDFLoader
    with patch("ingestion.pdf_loader.PdfReader") as mock_reader:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Hello world from page one with many words to fill the chunk requirement properly."
        mock_reader.return_value.pages = [mock_page]
        loader = PDFLoader(chunk_size=5, chunk_overlap=1, min_chunk_words=3)
        docs = loader.load(str(tmp_path / "fake.pdf"))
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert "text" in docs[0]
    assert "metadata" in docs[0]
    assert "page" in docs[0]["metadata"]


def test_pdf_loader_skips_short_chunks(tmp_path):
    from ingestion.pdf_loader import PDFLoader
    with patch("ingestion.pdf_loader.PdfReader") as mock_reader:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Short."
        mock_reader.return_value.pages = [mock_page]
        loader = PDFLoader(chunk_size=600, chunk_overlap=60)
        docs = loader.load("fake.pdf")
    assert docs == []
