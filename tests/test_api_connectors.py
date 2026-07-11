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


# ── EDGAR ─────────────────────────────────────────────────────────────────────

def _edgar_response_payload():
    return {
        "hits": {
            "hits": [
                {
                    "_id": "edgar/data/320193/000032019324000008/0000320193-24-000008-index.htm",
                    "_source": {
                        "file_date": "2024-02-02",
                        "entity_name": "Apple Inc.",
                        "form_type": "10-K",
                        "file_num": "0001234",
                    },
                }
            ]
        }
    }


@patch("connectors.edgar_connector.httpx.get")
def test_edgar_list_files(mock_get):
    from connectors.edgar_connector import EDGARConnector
    mock_get.return_value = MagicMock(status_code=200, json=lambda: _edgar_response_payload())
    conn = EDGARConnector(ticker="AAPL", form_type="10-K")
    files = conn.list_files()
    assert len(files) == 1
    assert "Apple" in files[0].name or "AAPL" in files[0].name or "10-K" in files[0].name


@patch("connectors.edgar_connector.httpx.get")
def test_edgar_download_returns_bytes(mock_get):
    from connectors.edgar_connector import EDGARConnector
    mock_resp = MagicMock(status_code=200, text="<html>10-K filing</html>")
    mock_resp.content = b"<html>10-K filing</html>"
    mock_get.return_value = mock_resp
    conn = EDGARConnector(ticker="AAPL", form_type="10-K")
    sf = SourceFile(
        id="https://www.sec.gov/Archives/edgar/data/320193/000032019324000008/aapl-20231230.htm",
        name="AAPL_10-K.html",
        path="",
        size=100,
        modified_at=datetime.now(timezone.utc),
    )
    content = conn.download(sf)
    assert isinstance(content, bytes)
    assert len(content) > 0


# ── Sports / ESPN ─────────────────────────────────────────────────────────────

def _espn_response_payload():
    return {
        "articles": [
            {
                "id": "39001234",
                "headline": "Chiefs win Super Bowl",
                "published": "2024-02-12T03:30:00Z",
                "description": "The Kansas City Chiefs won Super Bowl LVIII.",
                "links": {"api": {"news": {"href": "https://site.api.espn.com/article/39001234"}}},
            }
        ]
    }


@patch("connectors.sports_connector.httpx.get")
def test_sports_list_files(mock_get):
    from connectors.sports_connector import SportsConnector
    mock_get.return_value = MagicMock(status_code=200, json=lambda: _espn_response_payload())
    conn = SportsConnector(sport="football", league="nfl")
    files = conn.list_files()
    assert len(files) == 1
    assert files[0].id == "39001234"
    assert "Chiefs" in files[0].name or "39001234" in files[0].name


@patch("connectors.sports_connector.httpx.get")
def test_sports_download(mock_get):
    from connectors.sports_connector import SportsConnector
    article_payload = {"id": "39001234", "description": "The Kansas City Chiefs won.", "headline": "Chiefs win"}
    mock_get.return_value = MagicMock(status_code=200, json=lambda: article_payload)
    conn = SportsConnector(sport="football", league="nfl")
    sf = SourceFile(
        id="39001234",
        name="Chiefs_win.txt",
        path="https://site.api.espn.com/article/39001234",
        size=100,
        modified_at=datetime.now(timezone.utc),
        extra={"api_url": "https://site.api.espn.com/article/39001234"},
    )
    content = conn.download(sf)
    assert b"Chiefs" in content
