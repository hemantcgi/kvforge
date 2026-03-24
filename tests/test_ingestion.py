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


def test_markdown_loader_splits_by_heading(tmp_path):
    from ingestion.markdown_loader import MarkdownLoader
    md = tmp_path / "doc.md"
    md.write_text(
        "# Section One\n\nThis is section one content with many words to fill the chunk "
        "so it passes the minimum word count requirement for this test case.\n\n"
        "# Section Two\n\nThis is section two with different content and also enough "
        "words to pass the minimum word count check in the loader implementation."
    )
    loader = MarkdownLoader()
    docs = loader.load(str(md))
    assert len(docs) >= 1
    assert all("text" in d and "metadata" in d for d in docs)


def test_jsonl_loader_reads_one_doc_per_line(tmp_path):
    from ingestion.jsonl_loader import JSONLLoader
    jl = tmp_path / "data.jsonl"
    jl.write_text(
        '{"text": "First document with enough words to be a real chunk for testing purposes here."}\n'
        '{"text": "Second document also has enough words to pass the minimum check in the loader."}\n'
    )
    loader = JSONLLoader(text_key="text")
    docs = loader.load(str(jl))
    assert len(docs) == 2
    assert docs[0]["text"].startswith("First")


def test_jsonl_loader_custom_text_key(tmp_path):
    from ingestion.jsonl_loader import JSONLLoader
    jl = tmp_path / "data.jsonl"
    jl.write_text('{"content": "Document content here with enough words for minimum."}\n')
    loader = JSONLLoader(text_key="content")
    docs = loader.load(str(jl))
    assert len(docs) == 1


def test_html_loader_strips_tags(tmp_path):
    from ingestion.html_loader import HTMLLoader
    html_file = tmp_path / "page.html"
    html_file.write_text(
        "<html><body><h1>Title Section</h1><p>Body text content with enough words "
        "to pass the minimum word count for this html loader test case here.</p></body></html>"
    )
    loader = HTMLLoader()
    docs = loader.load(str(html_file))
    assert len(docs) >= 1
    assert "<" not in docs[0]["text"]


def test_directory_loader_dispatches_by_extension(tmp_path):
    from ingestion.directory_loader import DirectoryLoader
    (tmp_path / "a.md").write_text(
        "# Doc A\n\nContent for document A with many words to pass the minimum count here."
    )
    (tmp_path / "b.jsonl").write_text(
        '{"text": "Document B content with sufficient words to pass minimum count."}\n'
    )
    loader = DirectoryLoader()
    docs = loader.load(str(tmp_path))
    assert len(docs) >= 2


def test_registry_returns_correct_loader():
    from ingestion.registry import get_loader
    from ingestion.pdf_loader import PDFLoader
    from ingestion.markdown_loader import MarkdownLoader
    from ingestion.jsonl_loader import JSONLLoader
    assert isinstance(get_loader({"loader": "pdf"}), PDFLoader)
    assert isinstance(get_loader({"loader": "markdown"}), MarkdownLoader)
    assert isinstance(get_loader({"loader": "jsonl"}), JSONLLoader)
    assert isinstance(get_loader({}), PDFLoader)  # default is pdf


def test_bedrock_rag_cmd_index_uses_loader(tmp_path):
    """bedrock_rag.cmd_index wires through the DocumentLoader registry."""
    from unittest.mock import patch, MagicMock
    import sys

    # Create a minimal fake PDF
    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_bytes(b"fake")

    with patch("ingestion.pdf_loader.PdfReader") as mock_reader, \
         patch("bedrock_rag.TextEmbedding") as mock_emb_cls, \
         patch("bedrock_rag.QdrantClient") as mock_qdrant:

        mock_page = MagicMock()
        mock_page.extract_text.return_value = " ".join(["word"] * 50)
        mock_reader.return_value.pages = [mock_page]

        import numpy as np
        mock_emb_instance = MagicMock()
        mock_emb_instance.embed.return_value = iter([np.array([0.1] * 384)])
        mock_emb_cls.return_value = mock_emb_instance

        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_qdrant.return_value = mock_client

        from bedrock_rag import Config, cmd_index
        cfg = Config(embed_model="BAAI/bge-small-en-v1.5", vector_dim=384,
                     collection="test-col", loader="pdf")
        cmd_index(fake_pdf, cfg)

        # Confirm Qdrant upsert was called (means loader → embed → index worked)
        assert mock_client.upsert.called
