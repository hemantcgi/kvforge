# tests/test_api_connectors.py
import json
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from connectors.base import SourceFile


# ── Wikipedia ─────────────────────────────────────────────────────────────────

def _wiki_summary_payload(title: str) -> dict:
    return {
        "title": title,
        "extract": f"This is the extract for {title}.",
        "timestamp": "2024-01-15T12:00:00Z",
    }


@patch("connectors.wikipedia_connector.httpx.get")
def test_wikipedia_list_files(mock_get):
    from connectors.wikipedia_connector import WikipediaConnector
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: _wiki_summary_payload("Python_(programming_language)"),
    )
    conn = WikipediaConnector(topics="Python_(programming_language)")
    files = conn.list_files()
    assert len(files) == 1
    assert files[0].id == "Python_(programming_language)"
    assert files[0].name == "Python_(programming_language).txt"
    assert files[0].size > 0


@patch("connectors.wikipedia_connector.httpx.get")
def test_wikipedia_download(mock_get):
    from connectors.wikipedia_connector import WikipediaConnector
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: _wiki_summary_payload("Python_(programming_language)"),
    )
    conn = WikipediaConnector(topics="Python_(programming_language)")
    sf = SourceFile(
        id="Python_(programming_language)",
        name="Python_(programming_language).txt",
        path="Python_(programming_language)",
        size=100,
        modified_at=datetime.now(timezone.utc),
    )
    content = conn.download(sf)
    assert b"Python_(programming_language)" in content


@patch("connectors.wikipedia_connector.httpx.get")
def test_wikipedia_supports_delta(mock_get):
    from connectors.wikipedia_connector import WikipediaConnector
    conn = WikipediaConnector(topics="Python_(programming_language)")
    assert conn.supports_delta() is False


# ── FDA ───────────────────────────────────────────────────────────────────────

def _fda_response_payload():
    return {
        "results": [
            {
                "id": "abc123",
                "openfda": {"brand_name": ["TYLENOL"]},
                "effective_time": "20240101",
                "description": ["Extra Strength Tylenol is an analgesic."],
            }
        ]
    }


@patch("connectors.fda_connector.httpx.get")
def test_fda_list_files(mock_get):
    from connectors.fda_connector import FDAConnector
    mock_get.return_value = MagicMock(status_code=200, json=lambda: _fda_response_payload())
    conn = FDAConnector(drug_name="TYLENOL")
    files = conn.list_files()
    assert len(files) == 1
    assert files[0].id == "abc123"
    assert "TYLENOL" in files[0].name


@patch("connectors.fda_connector.httpx.get")
def test_fda_download(mock_get):
    from connectors.fda_connector import FDAConnector
    mock_get.return_value = MagicMock(status_code=200, json=lambda: _fda_response_payload())
    conn = FDAConnector(drug_name="TYLENOL")
    sf = conn.list_files()[0]
    content = conn.download(sf)
    assert b"TYLENOL" in content
