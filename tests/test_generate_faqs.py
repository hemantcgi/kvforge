"""Tests for automatic FAQ generation."""
import pytest


def test_parse_qa_standard_format():
    from tools.generate_faqs import _parse_qa
    text = "Q: What is Qdrant?\nA: Qdrant is a vector database."
    result = _parse_qa(text)
    assert result is not None
    q, a = result
    assert q == "What is Qdrant?"
    assert a == "Qdrant is a vector database."


def test_parse_qa_question_answer_format():
    from tools.generate_faqs import _parse_qa
    text = "Question: How does KV injection work?\nAnswer: It injects cached tensors."
    result = _parse_qa(text)
    assert result is not None
    q, a = result
    assert q == "How does KV injection work?"
    assert a == "It injects cached tensors."


def test_parse_qa_returns_none_when_unrecognized():
    from tools.generate_faqs import _parse_qa
    result = _parse_qa("This is just some random text without any QA structure at all.")
    assert result is None


def test_sample_chunks_returns_n_items():
    from tools.generate_faqs import _sample_chunks
    from unittest.mock import MagicMock
    mock_store = MagicMock()
    mock_pts = [MagicMock(payload={"text": f"Chunk {i} text content"}) for i in range(20)]
    mock_store.scroll.return_value = (mock_pts, None)
    chunks = _sample_chunks(mock_store, "my-collection", n=5)
    assert len(chunks) == 5
