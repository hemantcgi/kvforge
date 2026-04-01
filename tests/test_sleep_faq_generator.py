"""Tests for sleep-time FAQ generation module."""
import json
import pytest
from unittest.mock import MagicMock, patch


def test_parse_sleep_blocks_single():
    from pipeline.sleep_faq_generator import _parse_sleep_blocks
    text = "Q: What is KV injection?\nA: It injects cached tensors.\nINFERENCE: Pre-loading tensors reduces latency.\n---"
    results = _parse_sleep_blocks(text)
    assert len(results) == 1
    assert results[0]["question"] == "What is KV injection?"
    assert results[0]["answer"] == "It injects cached tensors."
    assert results[0]["inference"] == "Pre-loading tensors reduces latency."


def test_parse_sleep_blocks_multiple():
    from pipeline.sleep_faq_generator import _parse_sleep_blocks
    text = (
        "Q: What is Qdrant?\nA: A vector database.\nINFERENCE: Qdrant stores embeddings.\n---\n"
        "Q: What is RAG?\nA: Retrieval-augmented generation.\nINFERENCE: RAG improves factual grounding.\n---"
    )
    results = _parse_sleep_blocks(text)
    assert len(results) == 2
    assert results[1]["question"] == "What is RAG?"


def test_parse_sleep_blocks_empty():
    from pipeline.sleep_faq_generator import _parse_sleep_blocks
    assert _parse_sleep_blocks("no valid blocks here") == []


def test_parse_sleep_blocks_missing_inference_still_parses():
    from pipeline.sleep_faq_generator import _parse_sleep_blocks
    text = "Q: What is KV cache?\nA: A cached key-value tensor.\n---"
    results = _parse_sleep_blocks(text)
    assert len(results) == 1
    assert results[0]["inference"] == ""


def test_build_sleep_prompt_contains_chunk():
    from pipeline.sleep_faq_generator import _build_sleep_prompt
    prompt = _build_sleep_prompt("This is a test passage.", n_per_chunk=3)
    assert "This is a test passage." in prompt
    assert "3" in prompt
    assert "INFERENCE" in prompt


def test_deduplicate_faqs():
    from pipeline.sleep_faq_generator import _deduplicate
    existing = [{"question": "What is X?", "answer": "X is Y."}]
    new = [{"question": "What is X?", "answer": "X is Y."}, {"question": "What is Z?", "answer": "Z is W."}]
    result = _deduplicate(existing, new)
    assert len(result) == 2
    assert result[1]["question"] == "What is Z?"


def test_call_gemini_returns_text():
    from pipeline.sleep_faq_generator import _call_provider
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Q: Hi\nA: Hello\nINFERENCE: Greeting.\n---"}]}}]
    }
    with patch("pipeline.sleep_faq_generator.httpx.post", return_value=mock_resp):
        text = _call_provider("gemini", "gemini-2.5-flash", "fake-key", "test prompt")
    assert "Q: Hi" in text


def test_call_claude_returns_text():
    from pipeline.sleep_faq_generator import _call_provider
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "content": [{"type": "text", "text": "Q: Hi\nA: Hello\nINFERENCE: Greeting.\n---"}]
    }
    with patch("pipeline.sleep_faq_generator.httpx.post", return_value=mock_resp):
        text = _call_provider("claude", "claude-sonnet-4-6", "fake-key", "test prompt")
    assert "Q: Hi" in text


def test_call_openai_returns_text():
    from pipeline.sleep_faq_generator import _call_provider
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Q: Hi\nA: Hello\nINFERENCE: Greeting.\n---"}}]
    }
    with patch("pipeline.sleep_faq_generator.httpx.post", return_value=mock_resp):
        text = _call_provider("openai", "gpt-4.1", "fake-key", "test prompt")
    assert "Q: Hi" in text
