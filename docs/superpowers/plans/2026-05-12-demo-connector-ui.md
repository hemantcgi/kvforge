# Demo Connector & UI Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire up the 4 live-API demo use cases (Wikipedia, FDA, EDGAR, ESPN) with real connector backends, replace the primitive `prompt()` UI in connectors.html with a proper modal, add an "API Source" card to the wizard, add domain-specific model presets, and add Docker setup guidance for Milvus/Weaviate.

**Architecture:** Four new connector classes implement the existing `SourceConnector` Protocol (structural typing, no base class). `connectors/routes.py` is extended to recognise the new types. Two templates (`connectors.html`, `wizard.html`) are updated in-place; no new templates are introduced. All backend changes are covered by unit tests before the frontend is touched.

**Tech Stack:** Python stdlib + `httpx` (already a project dependency), Wikipedia REST API (no key required), openFDA API (no key required), SEC EDGAR full-text search API (no key required), ESPN hidden JSON API (no key required), Jinja2-free inline HTML templates (existing pattern).

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `connectors/wikipedia_connector.py` | Fetch Wikipedia article text by topic list or category |
| Create | `connectors/fda_connector.py` | Fetch openFDA drug-label records by drug name filter |
| Create | `connectors/edgar_connector.py` | Fetch SEC EDGAR filings by ticker / filing type |
| Create | `connectors/sports_connector.py` | Fetch ESPN sports news & schedules by sport / teams |
| Modify | `connectors/routes.py` | Expand `valid_types`; add 4 branches to `_run_test()` |
| Modify | `templates/studio/connectors.html` | Replace `showAddForm()` prompt() with a modal dialog |
| Modify | `templates/studio/wizard.html` | Add API source card (step 1), domain model presets (step 3), Docker tips (step 2 VDB) |
| Create | `tests/test_api_connectors.py` | Unit tests for the 4 new connectors (offline/mocked) |

---

## Task 1: Wikipedia connector

**Files:**
- Create: `connectors/wikipedia_connector.py`
- Test: `tests/test_api_connectors.py`

The connector fetches article summaries via the Wikipedia REST API (`https://en.wikipedia.org/api/rest_v1/page/summary/{title}`). Credentials field: `topics` (comma-separated article titles or a single category slug). `list_files()` returns one `SourceFile` per article; `download()` fetches the full article HTML → plain text.

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_connectors.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_api_connectors.py::test_wikipedia_list_files -v --override-ini="addopts="
```

Expected: `ModuleNotFoundError: No module named 'connectors.wikipedia_connector'`

- [ ] **Step 3: Implement the connector**

Create `connectors/wikipedia_connector.py`:

```python
"""Wikipedia REST API connector — fetches article text by topic list."""
from __future__ import annotations
from datetime import datetime, timezone

import httpx

from connectors.base import SourceConnector, SourceFile

_BASE = "https://en.wikipedia.org/api/rest_v1/page/summary"


class WikipediaConnector:
    """Fetch Wikipedia article summaries and full extracts by topic."""

    def __init__(self, topics: str):
        # topics: comma-separated article titles, e.g. "Python_(programming_language),Rust_(programming_language)"
        self._topics = [t.strip() for t in topics.split(",") if t.strip()]
        self._cache: dict[str, dict] = {}

    def _fetch_summary(self, title: str) -> dict:
        if title not in self._cache:
            r = httpx.get(f"{_BASE}/{title}", timeout=10)
            r.raise_for_status()
            self._cache[title] = r.json()
        return self._cache[title]

    def list_files(self) -> list[SourceFile]:
        files = []
        for title in self._topics:
            try:
                data = self._fetch_summary(title)
                extract = data.get("extract", "")
                ts_raw = data.get("timestamp", "")
                try:
                    modified_at = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    modified_at = datetime.now(timezone.utc)
                files.append(SourceFile(
                    id=title,
                    name=f"{title}.txt",
                    path=title,
                    size=len(extract.encode()),
                    modified_at=modified_at,
                    mime_type="text/plain",
                ))
            except Exception:
                continue
        return files

    def download(self, file: SourceFile) -> bytes:
        data = self._fetch_summary(file.id)
        extract = data.get("extract", "")
        return f"# {file.id}\n\n{extract}".encode()

    def get_modified_at(self, file: SourceFile) -> datetime:
        return file.modified_at

    def supports_delta(self) -> bool:
        return False

    def get_delta(self, token: str | None) -> tuple[list[SourceFile], str]:
        return self.list_files(), ""
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_api_connectors.py::test_wikipedia_list_files tests/test_api_connectors.py::test_wikipedia_download tests/test_api_connectors.py::test_wikipedia_supports_delta -v --override-ini="addopts="
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add connectors/wikipedia_connector.py tests/test_api_connectors.py
git commit -m "feat: add Wikipedia REST API connector"
```

---

## Task 2: openFDA connector

**Files:**
- Modify: `connectors/fda_connector.py` (create)
- Modify: `tests/test_api_connectors.py` (append tests)

The connector calls `https://api.fda.gov/drug/label.json?search=openfda.brand_name:{drug_name}&limit=100`. Each label record becomes one `SourceFile`; `download()` returns the full JSON record as UTF-8 bytes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api_connectors.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_api_connectors.py::test_fda_list_files -v --override-ini="addopts="
```

Expected: `ModuleNotFoundError: No module named 'connectors.fda_connector'`

- [ ] **Step 3: Implement the connector**

Create `connectors/fda_connector.py`:

```python
"""openFDA drug-label connector — no API key required."""
from __future__ import annotations
import json
from datetime import datetime, timezone

import httpx

from connectors.base import SourceFile

_BASE = "https://api.fda.gov/drug/label.json"


class FDAConnector:
    """Fetch FDA drug-label records by brand/generic name."""

    def __init__(self, drug_name: str, limit: int = 100):
        self._drug_name = drug_name.strip()
        self._limit = min(limit, 100)  # FDA caps at 100
        self._records: list[dict] | None = None

    def _fetch(self) -> list[dict]:
        if self._records is None:
            params = {
                "search": f'openfda.brand_name:"{self._drug_name}"',
                "limit": self._limit,
            }
            r = httpx.get(_BASE, params=params, timeout=15)
            if r.status_code == 404:
                self._records = []
            else:
                r.raise_for_status()
                self._records = r.json().get("results", [])
        return self._records

    def list_files(self) -> list[SourceFile]:
        files = []
        for rec in self._fetch():
            rid = rec.get("id", rec.get("set_id", "unknown"))
            brand = rec.get("openfda", {}).get("brand_name", [self._drug_name])[0]
            eff_time = rec.get("effective_time", "")
            try:
                ts = datetime.strptime(eff_time, "%Y%m%d").replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                ts = datetime.now(timezone.utc)
            content = json.dumps(rec).encode()
            files.append(SourceFile(
                id=rid,
                name=f"{brand}_{rid[:8]}.json",
                path=rid,
                size=len(content),
                modified_at=ts,
                mime_type="application/json",
            ))
        return files

    def download(self, file: SourceFile) -> bytes:
        for rec in self._fetch():
            rid = rec.get("id", rec.get("set_id", ""))
            if rid == file.id:
                return json.dumps(rec, indent=2).encode()
        return b"{}"

    def get_modified_at(self, file: SourceFile) -> datetime:
        return file.modified_at

    def supports_delta(self) -> bool:
        return False

    def get_delta(self, token: str | None) -> tuple[list[SourceFile], str]:
        return self.list_files(), ""
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_api_connectors.py::test_fda_list_files tests/test_api_connectors.py::test_fda_download -v --override-ini="addopts="
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add connectors/fda_connector.py tests/test_api_connectors.py
git commit -m "feat: add openFDA drug-label connector"
```

---

## Task 3: SEC EDGAR connector

**Files:**
- Create: `connectors/edgar_connector.py`
- Modify: `tests/test_api_connectors.py` (append tests)

Uses the EDGAR full-text search API: `https://efts.sec.gov/LATEST/search-index?q={query}&dateRange=custom&startdt={start}&forms={form_type}`. Each filing hit becomes a `SourceFile`; `download()` fetches the raw filing document from the EDGAR viewer URL.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api_connectors.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_api_connectors.py::test_edgar_list_files -v --override-ini="addopts="
```

Expected: `ModuleNotFoundError: No module named 'connectors.edgar_connector'`

- [ ] **Step 3: Implement the connector**

Create `connectors/edgar_connector.py`:

```python
"""SEC EDGAR full-text search connector."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta

import httpx

from connectors.base import SourceFile

_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
_HEADERS = {"User-Agent": "KVForge research@kvforge.ai"}


class EDGARConnector:
    """Fetch SEC EDGAR filings by ticker symbol and form type."""

    def __init__(self, ticker: str, form_type: str = "10-K", lookback_days: int = 365):
        self._ticker = ticker.strip().upper()
        self._form_type = form_type.strip()
        self._lookback_days = lookback_days
        self._hits: list[dict] | None = None

    def _search(self) -> list[dict]:
        if self._hits is None:
            start = (datetime.now(timezone.utc) - timedelta(days=self._lookback_days)).strftime("%Y-%m-%d")
            params = {
                "q": f'"{self._ticker}"',
                "dateRange": "custom",
                "startdt": start,
                "forms": self._form_type,
                "hits.hits.total.value": 1,
                "hits.hits._source.period_of_report": 1,
            }
            r = httpx.get(_SEARCH_URL, params=params, headers=_HEADERS, timeout=15)
            r.raise_for_status()
            self._hits = r.json().get("hits", {}).get("hits", [])
        return self._hits

    def list_files(self) -> list[SourceFile]:
        files = []
        for hit in self._search():
            src = hit.get("_source", {})
            hit_id = hit.get("_id", "")
            file_date_raw = src.get("file_date", "")
            try:
                modified_at = datetime.fromisoformat(file_date_raw).replace(tzinfo=timezone.utc)
            except (ValueError, AttributeError):
                modified_at = datetime.now(timezone.utc)
            entity = src.get("entity_name", self._ticker)
            form = src.get("form_type", self._form_type)
            name = f"{entity}_{form}_{file_date_raw}.html".replace(" ", "_").replace("/", "-")
            doc_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&filenum={src.get('file_num','')}&type={form}&dateb=&owner=include&count=1"
            files.append(SourceFile(
                id=hit_id,
                name=name,
                path=doc_url,
                size=0,
                modified_at=modified_at,
                mime_type="text/html",
                extra={"_source": src},
            ))
        return files

    def download(self, file: SourceFile) -> bytes:
        # hit_id looks like "edgar/data/320193/000032019324000008/0000320193-24-000008-index.htm"
        url = f"https://www.sec.gov/Archives/{file.id}"
        r = httpx.get(url, headers=_HEADERS, timeout=30, follow_redirects=True)
        r.raise_for_status()
        return r.content

    def get_modified_at(self, file: SourceFile) -> datetime:
        return file.modified_at

    def supports_delta(self) -> bool:
        return False

    def get_delta(self, token: str | None) -> tuple[list[SourceFile], str]:
        return self.list_files(), ""
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_api_connectors.py::test_edgar_list_files tests/test_api_connectors.py::test_edgar_download_returns_bytes -v --override-ini="addopts="
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add connectors/edgar_connector.py tests/test_api_connectors.py
git commit -m "feat: add SEC EDGAR full-text search connector"
```

---

## Task 4: ESPN Sports connector

**Files:**
- Create: `connectors/sports_connector.py`
- Modify: `tests/test_api_connectors.py` (append tests)

ESPN's hidden JSON API: `https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/news`. Each news article becomes one `SourceFile`. Credentials: `sport` (e.g. `football`), `league` (e.g. `nfl`), optional `team_filter`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api_connectors.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_api_connectors.py::test_sports_list_files -v --override-ini="addopts="
```

Expected: `ModuleNotFoundError: No module named 'connectors.sports_connector'`

- [ ] **Step 3: Implement the connector**

Create `connectors/sports_connector.py`:

```python
"""ESPN sports news connector — no API key required."""
from __future__ import annotations
import re
from datetime import datetime, timezone

import httpx

from connectors.base import SourceFile

_BASE = "https://site.api.espn.com/apis/site/v2/sports"


class SportsConnector:
    """Fetch ESPN news articles by sport and league."""

    def __init__(self, sport: str, league: str, team_filter: str = "", limit: int = 50):
        self._sport = sport.strip().lower()
        self._league = league.strip().lower()
        self._team_filter = team_filter.strip().lower()
        self._limit = limit
        self._articles: list[dict] | None = None

    def _fetch(self) -> list[dict]:
        if self._articles is None:
            url = f"{_BASE}/{self._sport}/{self._league}/news"
            r = httpx.get(url, params={"limit": self._limit}, timeout=15)
            r.raise_for_status()
            articles = r.json().get("articles", [])
            if self._team_filter:
                articles = [
                    a for a in articles
                    if self._team_filter in a.get("headline", "").lower()
                    or self._team_filter in a.get("description", "").lower()
                ]
            self._articles = articles
        return self._articles

    def list_files(self) -> list[SourceFile]:
        files = []
        for art in self._fetch():
            aid = str(art.get("id", ""))
            headline = art.get("headline", f"article_{aid}")
            slug = re.sub(r"[^A-Za-z0-9_-]", "_", headline)[:60]
            pub_raw = art.get("published", "")
            try:
                modified_at = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                modified_at = datetime.now(timezone.utc)
            api_url = (
                art.get("links", {}).get("api", {}).get("news", {}).get("href", "")
                or f"{_BASE}/{self._sport}/{self._league}/news/{aid}"
            )
            description = art.get("description", "")
            files.append(SourceFile(
                id=aid,
                name=f"{slug}.txt",
                path=api_url,
                size=len(description.encode()),
                modified_at=modified_at,
                mime_type="text/plain",
                extra={"api_url": api_url, "headline": headline},
            ))
        return files

    def download(self, file: SourceFile) -> bytes:
        api_url = file.extra.get("api_url", file.path)
        if api_url:
            try:
                r = httpx.get(api_url, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    headline = data.get("headline", file.extra.get("headline", ""))
                    description = data.get("description", "")
                    story = data.get("story", "")
                    return f"# {headline}\n\n{description}\n\n{story}".encode()
            except Exception:
                pass
        return file.extra.get("headline", "").encode()

    def get_modified_at(self, file: SourceFile) -> datetime:
        return file.modified_at

    def supports_delta(self) -> bool:
        return False

    def get_delta(self, token: str | None) -> tuple[list[SourceFile], str]:
        return self.list_files(), ""
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_api_connectors.py::test_sports_list_files tests/test_api_connectors.py::test_sports_download -v --override-ini="addopts="
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add connectors/sports_connector.py tests/test_api_connectors.py
git commit -m "feat: add ESPN sports news connector"
```

---

## Task 5: Expand connectors/routes.py for new types

**Files:**
- Modify: `connectors/routes.py` lines 40-42, 101-149

Two changes: (a) add the 4 new types to `valid_types`; (b) add 4 branches to `_run_test()` that do a lightweight connectivity check for each API.

- [ ] **Step 1: Write the failing test**

Create `tests/test_connector_routes.py`:

```python
# tests/test_connector_routes.py
import asyncio
import pytest


@pytest.mark.parametrize("conn_type", ["wikipedia", "fda", "edgar", "espn"])
def test_valid_types_includes_new_connectors(conn_type):
    """valid_types in routes.py must include all new API connector types."""
    import importlib, inspect
    mod = importlib.import_module("connectors.routes")
    src = inspect.getsource(mod.create_connector)
    # valid_types tuple must contain the new type
    assert conn_type in src, f"'{conn_type}' not found in create_connector source"


@pytest.mark.asyncio
@pytest.mark.parametrize("conn_type", ["wikipedia", "fda", "edgar", "espn"])
async def test_run_test_returns_ok_for_new_types(conn_type, monkeypatch):
    """_run_test must return {ok: True} for the new connector types using mocked httpx."""
    import httpx
    from unittest.mock import patch, MagicMock
    from connectors.routes import _run_test

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"extract": "test"} if conn_type == "wikipedia" else \
                                   {"results": [{"id": "x"}]} if conn_type == "fda" else \
                                   {"hits": {"hits": []}} if conn_type == "edgar" else \
                                   {"articles": []}

    with patch("httpx.get", return_value=mock_resp):
        result = await _run_test(conn_type, {
            "topics": "Python_(programming_language)",
            "drug_name": "TYLENOL",
            "ticker": "AAPL",
            "form_type": "10-K",
            "sport": "football",
            "league": "nfl",
        })
    assert result.get("ok") is True, f"Expected ok=True for {conn_type}, got {result}"
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_connector_routes.py -v --override-ini="addopts="
```

Expected: 8 FAILED (valid_types doesn't contain the new types)

- [ ] **Step 3: Update routes.py**

In `connectors/routes.py`, make these two edits:

**Edit 1** — line 40, change `valid_types`:
```python
    valid_types = ("gdrive", "s3", "sharepoint", "wikipedia", "fda", "edgar", "espn")
```

**Edit 2** — replace `_run_test()` entirely (lines 101-149) with:
```python
async def _run_test(connector_type: str, creds: dict) -> dict:
    if connector_type == "gdrive":
        try:
            from google.oauth2.service_account import Credentials
            from googleapiclient.discovery import build
        except ImportError:
            return {"ok": False, "error": "google-auth is not installed on this server"}
        import json
        info = json.loads(creds.get("service_account_json", "{}"))
        sa_creds = Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/drive.readonly"])
        svc = build("drive", "v3", credentials=sa_creds, cache_discovery=False)
        files = svc.files().list(pageSize=1).execute()
        return {"ok": True, "detail": f"Connected — {len(files.get('files', []))} files visible"}

    elif connector_type == "s3":
        try:
            import boto3
        except ImportError:
            return {"ok": False, "error": "boto3 is not installed on this server"}
        s3 = boto3.client(
            "s3",
            aws_access_key_id=creds.get("access_key_id"),
            aws_secret_access_key=creds.get("secret_access_key"),
            region_name=creds.get("region", "us-east-1"),
        )
        s3.head_bucket(Bucket=creds.get("bucket", ""))
        return {"ok": True, "detail": "S3 bucket reachable"}

    elif connector_type == "sharepoint":
        try:
            import msal
            import httpx
        except ImportError:
            return {"ok": False, "error": "msal or httpx is not installed on this server"}
        msal_app = msal.ConfidentialClientApplication(
            creds["client_id"],
            authority=f"https://login.microsoftonline.com/{creds['tenant_id']}",
            client_credential=creds["client_secret"],
        )
        result = msal_app.acquire_token_for_client(
            scopes=["https://graph.microsoft.com/.default"])
        if "error" in result:
            return {"ok": False, "error": result.get("error_description", result["error"])}
        async with httpx.AsyncClient() as hc:
            r = await hc.get(
                f"https://graph.microsoft.com/v1.0/sites/{creds.get('site_url', '')}",
                headers={"Authorization": f"Bearer {result['access_token']}"},
            )
        return {"ok": r.status_code == 200, "detail": f"HTTP {r.status_code}"}

    elif connector_type == "wikipedia":
        import httpx
        topic = creds.get("topics", "Python_(programming_language)").split(",")[0].strip()
        r = httpx.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}",
            timeout=8,
        )
        if r.status_code == 200:
            return {"ok": True, "detail": f"Wikipedia reachable — '{topic}' found"}
        return {"ok": False, "error": f"Wikipedia returned HTTP {r.status_code}"}

    elif connector_type == "fda":
        import httpx
        drug = creds.get("drug_name", "aspirin")
        r = httpx.get(
            "https://api.fda.gov/drug/label.json",
            params={"search": f'openfda.brand_name:"{drug}"', "limit": 1},
            timeout=10,
        )
        if r.status_code in (200, 404):
            count = len(r.json().get("results", [])) if r.status_code == 200 else 0
            return {"ok": True, "detail": f"openFDA reachable — {count} labels for '{drug}'"}
        return {"ok": False, "error": f"openFDA returned HTTP {r.status_code}"}

    elif connector_type == "edgar":
        import httpx
        ticker = creds.get("ticker", "AAPL")
        r = httpx.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={"q": f'"{ticker}"', "forms": "10-K", "dateRange": "custom",
                    "startdt": "2020-01-01"},
            headers={"User-Agent": "KVForge research@kvforge.ai"},
            timeout=10,
        )
        if r.status_code == 200:
            count = len(r.json().get("hits", {}).get("hits", []))
            return {"ok": True, "detail": f"EDGAR reachable — {count} filings for '{ticker}'"}
        return {"ok": False, "error": f"EDGAR returned HTTP {r.status_code}"}

    elif connector_type == "espn":
        import httpx
        sport = creds.get("sport", "football")
        league = creds.get("league", "nfl")
        r = httpx.get(
            f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/news",
            params={"limit": 1},
            timeout=8,
        )
        if r.status_code == 200:
            count = len(r.json().get("articles", []))
            return {"ok": True, "detail": f"ESPN reachable — {count} articles for {sport}/{league}"}
        return {"ok": False, "error": f"ESPN returned HTTP {r.status_code}"}

    return {"ok": False, "error": f"unknown connector type: {connector_type}"}
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_connector_routes.py -v --override-ini="addopts="
```

Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
git add connectors/routes.py tests/test_connector_routes.py
git commit -m "feat: add wikipedia/fda/edgar/espn to connector valid_types and _run_test"
```

---

## Task 6: Replace prompt() dialogs with Add Connector modal

**Files:**
- Modify: `templates/studio/connectors.html`

Replace the `showAddForm()` function (lines 128-142) and the ICONS object (line 54) with a proper modal dialog. The modal has: (1) a connector-type selector row with icon pills, (2) a dynamic credential form area that swaps content based on selected type, (3) a Name field, (4) a Test Connection button, (5) Save/Cancel actions.

**Screen design:**

```
┌─────────────────────────────────────────────────┐
│  Add Connector                              [×]  │
├─────────────────────────────────────────────────┤
│  Type                                           │
│  [GDrive] [S3] [SharePoint] [Wikipedia]         │
│  [FDA]    [EDGAR] [ESPN]                        │
├─────────────────────────────────────────────────┤
│  Name  ___________________________________      │
│                                                 │
│  ─── Credentials ───────────────────────────── │
│  (fields change based on selected type)         │
│                                                 │
│  Wikipedia:  Topics (comma-sep article titles)  │
│  FDA:        Drug Name (brand/generic)          │
│  EDGAR:      Ticker  ___  Form Type  10-K ▼     │
│  ESPN:       Sport  ___  League  ___            │
│  GDrive:     Service Account JSON (textarea)    │
│  S3:         Access Key  ___  Secret Key  ___   │
│              Bucket  ___  Region  us-east-1     │
│  SharePoint: Tenant ID  ___  Client ID  ___     │
│              Client Secret  ___                 │
│              Site URL  ___                      │
│                                                 │
│  [▶ Test Connection]  status appears here       │
├─────────────────────────────────────────────────┤
│              [Cancel]       [Save Connector]    │
└─────────────────────────────────────────────────┘
```

- [ ] **Step 1: No unit test needed for HTML — write a smoke test instead**

The walkthrough tests in `tests/ui/test_walkthroughs.py` cover modal interactions. Instead, add a quick API smoke test confirming the server returns the connectors page HTML containing the modal div:

```python
# In tests/test_connector_modal_smoke.py
import httpx
import pytest

def test_connectors_page_contains_modal(app_server, db_path):
    """Connectors HTML page must include the add-connector-modal element."""
    import jwt as pyjwt
    from datetime import datetime, timedelta, timezone
    base_url, secret = app_server
    tok = pyjwt.encode({"sub": "t1", "role": "admin",
                        "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
                       secret, algorithm="HS256")
    r = httpx.get(f"{base_url}/studio/connectors",
                  cookies={"kvforge_session": tok},
                  follow_redirects=True, timeout=10)
    assert r.status_code == 200
    assert "add-connector-modal" in r.text
```

This test will FAIL until the modal is added (next step).

- [ ] **Step 2: Update connectors.html — add modal HTML and replace showAddForm()**

In `templates/studio/connectors.html`, make these changes:

**Change 1** — update the ICONS object (line 54) to include all 7 types:
```javascript
const ICONS = {
  gdrive:     {abbr:'GD', bg:'#0F9D58'},
  s3:         {abbr:'S3', bg:'#FF9900'},
  sharepoint: {abbr:'SP', bg:'#0078D4'},
  wikipedia:  {abbr:'WP', bg:'#636466'},
  fda:        {abbr:'FDA',bg:'#1a53a8'},
  edgar:      {abbr:'SEC',bg:'#c0392b'},
  espn:       {abbr:'ESPN',bg:'#d00'},
};
```

**Change 2** — add CSS (insert before the `</style>` tag at the top of the file):
```css
/* Add-Connector Modal */
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:100;display:none;align-items:center;justify-content:center;}
.modal-overlay.open{display:flex;}
.modal-box{background:#1e1e1e;border:1px solid #2d2d2d;border-radius:10px;width:440px;max-width:95vw;overflow:hidden;}
.modal-hd{display:flex;justify-content:space-between;align-items:center;padding:14px 18px;border-bottom:1px solid #2d2d2d;}
.modal-hd h3{color:#d4d4d4;font-size:14px;font-weight:700;}
.modal-close{background:none;border:none;color:#555;font-size:16px;cursor:pointer;line-height:1;}
.modal-close:hover{color:#9cdcfe;}
.modal-body{padding:16px 18px;}
.modal-footer{padding:12px 18px;border-top:1px solid #2d2d2d;display:flex;justify-content:flex-end;gap:8px;}
.type-pills{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px;}
.type-pill{padding:4px 10px;border-radius:5px;border:1px solid #2d2d2d;background:#181818;color:#808080;font-size:11px;font-weight:700;cursor:pointer;transition:all .15s;}
.type-pill:hover{border-color:#4ec9b088;}
.type-pill.sel{border-color:#4ec9b0;background:#1a2e28;color:#4ec9b0;}
.m-field{margin-bottom:10px;}
.m-label{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#6b7280;margin-bottom:3px;}
.m-input{width:100%;background:#181818;border:1px solid #2d2d2d;border-radius:4px;padding:6px 9px;color:#d4d4d4;font-size:12px;font-family:monospace;}
.m-input:focus{outline:none;border-color:#4ec9b0;}
.m-textarea{width:100%;background:#181818;border:1px solid #2d2d2d;border-radius:4px;padding:6px 9px;color:#d4d4d4;font-size:11px;font-family:monospace;min-height:80px;resize:vertical;}
.m-textarea:focus{outline:none;border-color:#4ec9b0;}
.cred-section{display:none;}
.cred-section.visible{display:block;}
.cred-divider{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4ec9b0;margin:10px 0 8px;padding-bottom:4px;border-bottom:1px solid #2d2d2d;}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
.test-bar{display:flex;align-items:center;gap:8px;margin-top:8px;}
.btn-test{background:#264f7822;border:1px solid #264f78;color:#9cdcfe;padding:5px 12px;border-radius:4px;font-size:11px;cursor:pointer;}
.btn-test:hover{background:#264f7844;}
.test-status{font-size:11px;min-height:16px;}
.test-ok{color:#4ec9b0;}.test-err{color:#f87171;}.test-pending{color:#9cdcfe;}
.btn-cancel{background:none;border:1px solid #2d2d2d;color:#808080;padding:6px 14px;border-radius:5px;font-size:12px;cursor:pointer;}
.btn-cancel:hover{border-color:#555;color:#d4d4d4;}
.btn-save{background:linear-gradient(135deg,#264f78,#1e5c4f);border:none;color:#d4d4d4;padding:6px 16px;border-radius:5px;font-size:12px;font-weight:700;cursor:pointer;}
.btn-save:hover{opacity:.9;}
```

**Change 3** — add modal HTML (insert before the `<script>` tag):
```html
<div class="modal-overlay" id="add-connector-modal">
  <div class="modal-box">
    <div class="modal-hd">
      <h3>Add Connector</h3>
      <button class="modal-close" onclick="closeModal()">✕</button>
    </div>
    <div class="modal-body">
      <div class="m-field">
        <div class="m-label">Source Type</div>
        <div class="type-pills">
          <div class="type-pill" data-type="gdrive" onclick="selectType(this)">Google Drive</div>
          <div class="type-pill" data-type="s3" onclick="selectType(this)">Amazon S3</div>
          <div class="type-pill" data-type="sharepoint" onclick="selectType(this)">SharePoint</div>
          <div class="type-pill" data-type="wikipedia" onclick="selectType(this)">Wikipedia</div>
          <div class="type-pill" data-type="fda" onclick="selectType(this)">openFDA</div>
          <div class="type-pill" data-type="edgar" onclick="selectType(this)">SEC EDGAR</div>
          <div class="type-pill" data-type="espn" onclick="selectType(this)">ESPN Sports</div>
        </div>
      </div>
      <div class="m-field">
        <div class="m-label">Connector Name</div>
        <input class="m-input" id="m-name" type="text" placeholder="e.g. My Wikipedia Source" autocomplete="off"/>
      </div>
      <!-- Credential forms per type -->
      <div id="creds-gdrive" class="cred-section">
        <div class="cred-divider">Google Drive Credentials</div>
        <div class="m-field"><div class="m-label">Service Account JSON</div>
          <textarea class="m-textarea" id="m-gdrive-sa" placeholder='{"type":"service_account",...}'></textarea></div>
        <div class="m-field"><div class="m-label">Root Folder ID (optional)</div>
          <input class="m-input" id="m-gdrive-folder" type="text" placeholder="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs"/></div>
      </div>
      <div id="creds-s3" class="cred-section">
        <div class="cred-divider">AWS S3 Credentials</div>
        <div class="row2">
          <div class="m-field"><div class="m-label">Access Key ID</div>
            <input class="m-input" id="m-s3-key" type="text" placeholder="AKIA…"/></div>
          <div class="m-field"><div class="m-label">Secret Access Key</div>
            <input class="m-input" id="m-s3-secret" type="password" placeholder="••••••••"/></div>
        </div>
        <div class="row2">
          <div class="m-field"><div class="m-label">Bucket</div>
            <input class="m-input" id="m-s3-bucket" type="text" placeholder="my-corpus-bucket"/></div>
          <div class="m-field"><div class="m-label">Region</div>
            <input class="m-input" id="m-s3-region" type="text" placeholder="us-east-1"/></div>
        </div>
      </div>
      <div id="creds-sharepoint" class="cred-section">
        <div class="cred-divider">Microsoft SharePoint Credentials</div>
        <div class="m-field"><div class="m-label">Tenant ID</div>
          <input class="m-input" id="m-sp-tenant" type="text" placeholder="00000000-0000-0000-0000-000000000000"/></div>
        <div class="row2">
          <div class="m-field"><div class="m-label">Client ID</div>
            <input class="m-input" id="m-sp-client" type="text"/></div>
          <div class="m-field"><div class="m-label">Client Secret</div>
            <input class="m-input" id="m-sp-secret" type="password" placeholder="••••••••"/></div>
        </div>
        <div class="m-field"><div class="m-label">Site URL or ID</div>
          <input class="m-input" id="m-sp-site" type="text" placeholder="contoso.sharepoint.com:/sites/knowledge"/></div>
      </div>
      <div id="creds-wikipedia" class="cred-section">
        <div class="cred-divider">Wikipedia Configuration</div>
        <div class="m-field"><div class="m-label">Topics (comma-separated article titles)</div>
          <input class="m-input" id="m-wiki-topics" type="text" placeholder="Python_(programming_language),Rust_(programming_language)"/></div>
      </div>
      <div id="creds-fda" class="cred-section">
        <div class="cred-divider">openFDA Configuration</div>
        <div class="m-field"><div class="m-label">Drug Name (brand or generic)</div>
          <input class="m-input" id="m-fda-drug" type="text" placeholder="e.g. aspirin, TYLENOL"/></div>
      </div>
      <div id="creds-edgar" class="cred-section">
        <div class="cred-divider">SEC EDGAR Configuration</div>
        <div class="row2">
          <div class="m-field"><div class="m-label">Ticker Symbol</div>
            <input class="m-input" id="m-edgar-ticker" type="text" placeholder="AAPL"/></div>
          <div class="m-field"><div class="m-label">Form Type</div>
            <input class="m-input" id="m-edgar-form" type="text" placeholder="10-K" value="10-K"/></div>
        </div>
      </div>
      <div id="creds-espn" class="cred-section">
        <div class="cred-divider">ESPN Sports Configuration</div>
        <div class="row2">
          <div class="m-field"><div class="m-label">Sport</div>
            <input class="m-input" id="m-espn-sport" type="text" placeholder="football"/></div>
          <div class="m-field"><div class="m-label">League</div>
            <input class="m-input" id="m-espn-league" type="text" placeholder="nfl"/></div>
        </div>
        <div class="m-field"><div class="m-label">Team Filter (optional)</div>
          <input class="m-input" id="m-espn-team" type="text" placeholder="chiefs"/></div>
      </div>
      <!-- Test connection -->
      <div class="test-bar">
        <button class="btn-test" onclick="testConn()">▶ Test Connection</button>
        <span class="test-status" id="m-test-status"></span>
      </div>
    </div>
    <div class="modal-footer">
      <button class="btn-cancel" onclick="closeModal()">Cancel</button>
      <button class="btn-save" onclick="saveConn()">Save Connector</button>
    </div>
  </div>
</div>
```

**Change 4** — replace `showAddForm()` and add modal JS (inside `<script>`):

Replace `function showAddForm() { ... }` with:

```javascript
let _selectedType = null;

function showAddForm() {
  _selectedType = null;
  document.querySelectorAll('.type-pill').forEach(p=>p.classList.remove('sel'));
  document.querySelectorAll('.cred-section').forEach(s=>s.classList.remove('visible'));
  document.getElementById('m-name').value = '';
  document.getElementById('m-test-status').textContent = '';
  document.getElementById('add-connector-modal').classList.add('open');
}

function closeModal() {
  document.getElementById('add-connector-modal').classList.remove('open');
}

function selectType(pill) {
  _selectedType = pill.dataset.type;
  document.querySelectorAll('.type-pill').forEach(p=>p.classList.remove('sel'));
  pill.classList.add('sel');
  document.querySelectorAll('.cred-section').forEach(s=>s.classList.remove('visible'));
  const sec = document.getElementById('creds-'+_selectedType);
  if (sec) sec.classList.add('visible');
}

function _buildCredentials() {
  const t = _selectedType;
  if (t === 'gdrive') {
    try { return JSON.parse(document.getElementById('m-gdrive-sa').value.trim()); }
    catch(e) { alert('Service account JSON is not valid JSON'); return null; }
  }
  if (t === 's3') return {
    access_key_id: document.getElementById('m-s3-key').value.trim(),
    secret_access_key: document.getElementById('m-s3-secret').value.trim(),
    bucket: document.getElementById('m-s3-bucket').value.trim(),
    region: document.getElementById('m-s3-region').value.trim() || 'us-east-1',
  };
  if (t === 'sharepoint') return {
    tenant_id: document.getElementById('m-sp-tenant').value.trim(),
    client_id: document.getElementById('m-sp-client').value.trim(),
    client_secret: document.getElementById('m-sp-secret').value.trim(),
    site_url: document.getElementById('m-sp-site').value.trim(),
  };
  if (t === 'wikipedia') return { topics: document.getElementById('m-wiki-topics').value.trim() };
  if (t === 'fda') return { drug_name: document.getElementById('m-fda-drug').value.trim() };
  if (t === 'edgar') return {
    ticker: document.getElementById('m-edgar-ticker').value.trim().toUpperCase(),
    form_type: document.getElementById('m-edgar-form').value.trim() || '10-K',
  };
  if (t === 'espn') return {
    sport: document.getElementById('m-espn-sport').value.trim().toLowerCase(),
    league: document.getElementById('m-espn-league').value.trim().toLowerCase(),
    team_filter: document.getElementById('m-espn-team').value.trim().toLowerCase(),
  };
  return {};
}

async function testConn() {
  if (!_selectedType) { alert('Select a connector type first'); return; }
  const creds = _buildCredentials();
  if (!creds) return;
  const status = document.getElementById('m-test-status');
  status.className = 'test-status test-pending';
  status.textContent = 'Testing…';
  // Create a temporary connector to test against
  const tempName = '__kvforge_test_' + Date.now();
  let tempId = null;
  try {
    const cr = await fetch('/studio/api/connectors', {
      method:'POST', credentials:'include',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({type: _selectedType, name: tempName, credentials: creds}),
    });
    if (!cr.ok) { status.className='test-status test-err'; status.textContent='Could not create temp connector'; return; }
    tempId = (await cr.json()).id;
    const tr = await fetch(`/studio/api/connectors/${tempId}/test`, {method:'POST', credentials:'include'});
    const result = await tr.json();
    status.className = result.ok ? 'test-status test-ok' : 'test-status test-err';
    status.textContent = result.ok ? ('✓ ' + (result.detail||'Connected')) : ('✗ ' + (result.error||'Failed'));
  } finally {
    if (tempId) {
      fetch(`/studio/api/connectors/${tempId}`, {method:'DELETE', credentials:'include'});
    }
  }
}

async function saveConn() {
  if (!_selectedType) { alert('Select a connector type first'); return; }
  const name = document.getElementById('m-name').value.trim();
  if (!name) { alert('Enter a connector name'); return; }
  const creds = _buildCredentials();
  if (!creds) return;
  const r = await fetch('/studio/api/connectors', {
    method:'POST', credentials:'include',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({type: _selectedType, name, credentials: creds}),
  });
  if (!r.ok) { alert('Failed to save connector: ' + r.status); return; }
  closeModal();
  load();
}

// Close on overlay click
document.getElementById('add-connector-modal').addEventListener('click', function(e){
  if (e.target === this) closeModal();
});
```

- [ ] **Step 3: Verify the smoke test now passes**

```
python -m pytest tests/test_connector_modal_smoke.py -v --override-ini="addopts="
```

Expected: 1 PASSED (`add-connector-modal` found in page HTML)

- [ ] **Step 4: Commit**

```bash
git add templates/studio/connectors.html tests/test_connector_modal_smoke.py
git commit -m "feat: replace prompt() with proper Add Connector modal (7 types)"
```

---

## Task 7: Wizard Step 1 — Add API Source card

**Files:**
- Modify: `templates/studio/wizard.html`

Add a fifth source-type card "Live API Source" to wizard Step 1. When selected, show a sub-panel with a type selector (same 4 API types) and per-type config fields. The selected type + config gets encoded into the wizard's payload at launch.

**Screen design (Step 1 with API card selected):**

```
┌── Step 1: Data Source ──────────────────────────┐
│  ○ Upload Files      local PDFs, txt, md        │
│  ○ Vector DB URL     existing Qdrant/Chroma etc  │
│  ○ HuggingFace       dataset by ID              │
│  ○ Existing Corpus   reuse indexed data         │
│  ● Live API Source   Wikipedia · FDA · EDGAR...  │
│                                                  │
│  ┌── API Source Config ──────────────────────┐  │
│  │  Source  [Wikipedia▼] [openFDA] [EDGAR] [ESPN]│
│  │                                            │  │
│  │  Topics  ________________________________  │  │
│  │  (comma-separated Wikipedia article titles)│  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

- [ ] **Step 1: Locate Step 1 src-cards in wizard.html and identify insertion point**

The Step 1 panel contains `.src-cards` with 4 `.src-card` divs. After the last card (for "existing"), add the new API card. Then add a conditional sub-panel div after the `.src-cards` div.

- [ ] **Step 2: Add the API card to wizard.html Step 1**

Find the Step 1 panel section. The 4 current src-cards end with the "existing" card. Add after the closing `</div>` of the last card but still inside `.src-cards`:

```html
<div class="src-card unsel" data-src="api" onclick="selSrc(this)">
  <span class="src-icon">🌐</span>
  <div>
    <div class="src-title">Live API Source</div>
    <div class="src-sub">Wikipedia · openFDA · SEC EDGAR · ESPN</div>
  </div>
</div>
```

After the closing `</div>` of `.src-cards`, add the API config sub-panel:

```html
<!-- API Source sub-panel (shown when src=api is selected) -->
<div id="api-src-panel" style="display:none;margin-top:8px;">
  <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4ec9b0;margin-bottom:6px;padding-bottom:3px;border-bottom:1px solid #2d2d2d;">API Source Config</div>
  <div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:8px;" id="api-type-pills">
    <div class="src-card unsel" style="padding:4px 9px;flex-direction:row;gap:6px;" data-api="wikipedia" onclick="selApiType(this)">
      <span style="font-size:12px;">📖</span>
      <div><div class="src-title">Wikipedia</div></div>
    </div>
    <div class="src-card unsel" style="padding:4px 9px;flex-direction:row;gap:6px;" data-api="fda" onclick="selApiType(this)">
      <span style="font-size:12px;">💊</span>
      <div><div class="src-title">openFDA</div></div>
    </div>
    <div class="src-card unsel" style="padding:4px 9px;flex-direction:row;gap:6px;" data-api="edgar" onclick="selApiType(this)">
      <span style="font-size:12px;">📋</span>
      <div><div class="src-title">SEC EDGAR</div></div>
    </div>
    <div class="src-card unsel" style="padding:4px 9px;flex-direction:row;gap:6px;" data-api="espn" onclick="selApiType(this)">
      <span style="font-size:12px;">🏈</span>
      <div><div class="src-title">ESPN Sports</div></div>
    </div>
  </div>
  <!-- Per-type config fields -->
  <div id="api-cfg-wikipedia" class="api-cfg-panel" style="display:none;">
    <div class="field"><div class="flabel">Topics (comma-sep article titles)</div>
      <input class="finput" id="api-wiki-topics" type="text" placeholder="Python_(programming_language),NumPy"/></div>
  </div>
  <div id="api-cfg-fda" class="api-cfg-panel" style="display:none;">
    <div class="field"><div class="flabel">Drug Name (brand or generic)</div>
      <input class="finput" id="api-fda-drug" type="text" placeholder="aspirin"/></div>
  </div>
  <div id="api-cfg-edgar" class="api-cfg-panel" style="display:none;">
    <div class="row2">
      <div class="field"><div class="flabel">Ticker</div>
        <input class="finput" id="api-edgar-ticker" type="text" placeholder="AAPL"/></div>
      <div class="field"><div class="flabel">Form Type</div>
        <input class="finput" id="api-edgar-form" type="text" placeholder="10-K" value="10-K"/></div>
    </div>
  </div>
  <div id="api-cfg-espn" class="api-cfg-panel" style="display:none;">
    <div class="row2">
      <div class="field"><div class="flabel">Sport</div>
        <input class="finput" id="api-espn-sport" type="text" placeholder="football"/></div>
      <div class="field"><div class="flabel">League</div>
        <input class="finput" id="api-espn-league" type="text" placeholder="nfl"/></div>
    </div>
  </div>
</div>
```

- [ ] **Step 3: Update JS — selSrc() and selApiType()**

In wizard.html's `<script>` block, update `selSrc()` to show/hide the API panel:

Find the `selSrc` function and add at the end of it:
```javascript
  document.getElementById('api-src-panel').style.display =
    card.dataset.src === 'api' ? 'block' : 'none';
```

Add a new function `selApiType()`:
```javascript
function selApiType(pill) {
  document.querySelectorAll('#api-type-pills [data-api]').forEach(p => {
    p.classList.remove('sel'); p.classList.add('unsel');
  });
  pill.classList.add('sel'); pill.classList.remove('unsel');
  document.querySelectorAll('.api-cfg-panel').forEach(p => p.style.display='none');
  const panel = document.getElementById('api-cfg-'+pill.dataset.api);
  if (panel) panel.style.display = 'block';
}
```

Update the `buildPayload()` or review-step render to include API connector config when src=api:
```javascript
// inside buildPayload() or wherever src config is collected:
if (wizState.src === 'api') {
  const apiType = document.querySelector('#api-type-pills .sel')?.dataset.api;
  wizState.api_type = apiType;
  if (apiType === 'wikipedia') wizState.api_topics = document.getElementById('api-wiki-topics').value;
  if (apiType === 'fda') wizState.api_drug = document.getElementById('api-fda-drug').value;
  if (apiType === 'edgar') {
    wizState.api_ticker = document.getElementById('api-edgar-ticker').value;
    wizState.api_form = document.getElementById('api-edgar-form').value;
  }
  if (apiType === 'espn') {
    wizState.api_sport = document.getElementById('api-espn-sport').value;
    wizState.api_league = document.getElementById('api-espn-league').value;
  }
}
```

- [ ] **Step 4: Commit**

```bash
git add templates/studio/wizard.html
git commit -m "feat: add Live API Source card to wizard Step 1 (wikipedia/fda/edgar/espn)"
```

---

## Task 8: Wizard Step 3 — Domain model presets

**Files:**
- Modify: `templates/studio/wizard.html`

Add 4 domain-specific preset model cards to Step 3 (the Model selection step). Keep the existing 4 general-purpose cards; add a "Domain Presets" sub-section below with BioMistral-7B (FDA), Mistral-7B-Instruct-v0.3 (EDGAR/SEC), Phi-3-mini-4k (Sports/News), and Llama-3.1-8B (Wikipedia/general).

**Screen design (Step 3 lower section):**

```
─── Domain Presets ───────────────────────────────
○  BioMistral-7B        Medical/Pharma RAG    14 GB
○  Mistral-7B-Instruct  Finance/Legal RAG     14 GB
○  Phi-3-mini-4k        Sports/News (fast)     8 GB
○  Llama-3.1-8B-Instr   General (best)        16 GB
```

- [ ] **Step 1: Locate model-list div in wizard.html and identify insertion point**

Step 3 contains a `.model-list` div with 4 `.model-card` divs. After the last closing `</div>` of `.model-list`, insert a section header and 4 new cards.

- [ ] **Step 2: Insert domain preset section**

After the `.model-list` closing `</div>`, add:

```html
<!-- Domain Presets -->
<div style="font-size:8px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#6b7280;margin:10px 0 6px;padding-bottom:3px;border-bottom:1px solid #2d2d2d;">Domain Presets</div>
<div class="model-list">
  <div class="model-card unsel" data-model="BioMistral/BioMistral-7B-DARE" onclick="selModel(this)">
    <span class="mc-icon">💊</span>
    <div>
      <div class="mc-name">BioMistral-7B</div>
      <div class="mc-meta">Medical / Pharma RAG · Recommended for FDA</div>
    </div>
    <div class="mc-vram"><span class="vram-pill vram-warn">~14 GB</span></div>
  </div>
  <div class="model-card unsel" data-model="mistralai/Mistral-7B-Instruct-v0.3" onclick="selModel(this)">
    <span class="mc-icon">📋</span>
    <div>
      <div class="mc-name">Mistral-7B-Instruct v0.3</div>
      <div class="mc-meta">Finance / Legal RAG · Recommended for EDGAR</div>
    </div>
    <div class="mc-vram"><span class="vram-pill vram-warn">~14 GB</span></div>
  </div>
  <div class="model-card unsel" data-model="microsoft/Phi-3-mini-4k-instruct" onclick="selModel(this)">
    <span class="mc-icon">🏈</span>
    <div>
      <div class="mc-name">Phi-3-mini-4k-instruct</div>
      <div class="mc-meta">Sports / News · Fast inference</div>
    </div>
    <div class="mc-vram"><span class="vram-pill vram-ok">~8 GB</span></div>
  </div>
  <div class="model-card unsel" data-model="meta-llama/Llama-3.1-8B-Instruct" onclick="selModel(this)">
    <span class="mc-icon">📖</span>
    <div>
      <div class="mc-name">Llama-3.1-8B-Instruct</div>
      <div class="mc-meta">General purpose · Best for Wikipedia</div>
    </div>
    <div class="mc-vram"><span class="vram-pill vram-warn">~16 GB</span></div>
  </div>
</div>
```

- [ ] **Step 3: Commit**

```bash
git add templates/studio/wizard.html
git commit -m "feat: add domain model presets to wizard Step 3 (BioMistral, Mistral, Phi-3, Llama-3.1)"
```

---

## Task 9: VDB step — Milvus and Weaviate Docker setup guidance

**Files:**
- Modify: `templates/studio/wizard.html`

In wizard Step 2 (VDB selection), the Milvus and Weaviate tiles are already present but have no guidance. When either tile is selected, show an info callout with a one-liner Docker command the user can copy-paste.

**Screen design (Milvus selected):**

```
┌── Docker Setup Required ───────────────────────┐
│  🐳  Milvus requires a running Docker container  │
│                                                  │
│  docker run -d --name milvus-standalone \        │
│    -p 19530:19530 -p 9091:9091 \                │
│    milvusdb/milvus:v2.4.0 standalone            │
│                                                  │
│  [📋 Copy Command]                               │
│                                                  │
│  Connection URL  localhost:19530                 │
└──────────────────────────────────────────────────┘
```

- [ ] **Step 1: Add Docker callout HTML and JS to wizard.html**

In the VDB sub-panels section (the `.vdb-fields` divs), find the Milvus and Weaviate sub-panels and add a Docker info callout at the top of each:

In the Milvus vdb-fields div, prepend:
```html
<div class="docker-callout" style="background:#0e1218;border:1px solid #1e3a5f;border-radius:6px;padding:10px 12px;margin-bottom:8px;">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">
    <span style="font-size:14px;">🐳</span>
    <span style="font-size:10px;font-weight:700;color:#9cdcfe;">Docker Setup Required</span>
  </div>
  <div style="background:#0a0e14;border:1px solid #1e2a3a;border-radius:4px;padding:7px 10px;font-family:monospace;font-size:9px;color:#4ec9b0;margin-bottom:6px;white-space:pre-wrap;">docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:v2.4.0 standalone</div>
  <button onclick="copyCmd(this,'docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.4.0 standalone')"
    style="background:#264f7822;border:1px solid #264f78;color:#9cdcfe;padding:3px 9px;border-radius:3px;font-size:9px;cursor:pointer;">📋 Copy Command</button>
</div>
```

In the Weaviate vdb-fields div, prepend:
```html
<div class="docker-callout" style="background:#0e1218;border:1px solid #1e3a5f;border-radius:6px;padding:10px 12px;margin-bottom:8px;">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">
    <span style="font-size:14px;">🐳</span>
    <span style="font-size:10px;font-weight:700;color:#9cdcfe;">Docker Setup Required</span>
  </div>
  <div style="background:#0a0e14;border:1px solid #1e2a3a;border-radius:4px;padding:7px 10px;font-family:monospace;font-size:9px;color:#4ec9b0;margin-bottom:6px;white-space:pre-wrap;">docker run -d --name weaviate \
  -p 8080:8080 -p 50051:50051 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  cr.weaviate.io/semitechnologies/weaviate:1.24.1</div>
  <button onclick="copyCmd(this,'docker run -d --name weaviate -p 8080:8080 -p 50051:50051 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true cr.weaviate.io/semitechnologies/weaviate:1.24.1')"
    style="background:#264f7822;border:1px solid #264f78;color:#9cdcfe;padding:3px 9px;border-radius:3px;font-size:9px;cursor:pointer;">📋 Copy Command</button>
</div>
```

Add `copyCmd()` JS function in the `<script>` block:
```javascript
function copyCmd(btn, cmd) {
  navigator.clipboard.writeText(cmd).then(() => {
    const orig = btn.textContent;
    btn.textContent = '✓ Copied!';
    setTimeout(() => btn.textContent = orig, 1500);
  });
}
```

- [ ] **Step 2: Commit**

```bash
git add templates/studio/wizard.html
git commit -m "feat: add Docker setup callouts for Milvus and Weaviate in wizard VDB step"
```

---

## Task 10: Run full test suite and validate

- [ ] **Step 1: Run all tests**

```
python -m pytest tests/ -v --override-ini="addopts=" -x
```

Expected: All tests pass (existing 21 + new connector tests).

- [ ] **Step 2: Start server and do a quick manual smoke check**

```
python kvforge_portal.py --port 8080
```

Verify:
- `/studio/connectors` — "Add Connector" button opens the 7-type modal
- Selecting "Wikipedia" shows the Topics field
- Selecting "ESPN" shows Sport/League fields
- Test Connection button creates a temp connector, tests it, deletes it
- `/studio/wizard` — Step 1 has a 5th card "Live API Source"
- Selecting API source shows the 4 API type pills and config fields
- Step 3 has a "Domain Presets" section with 4 new model cards
- Milvus/Weaviate VDB tiles show the Docker callout on selection

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "chore: final integration pass — all demo connector UI items complete"
```

---

## Self-Review

**Spec coverage check:**

| TODO Flag | Tasks | Status |
|-----------|-------|--------|
| Missing API connector types (Wikipedia, FDA, EDGAR, ESPN) | Tasks 1-4, 5 | ✅ backend + routes |
| No API data source in wizard Step 1 | Task 7 | ✅ src-card + sub-panel |
| Domain model presets missing | Task 8 | ✅ 4 preset cards |
| Milvus/Weaviate Docker setup guidance | Task 9 | ✅ callout + copy button |
| Connector modal (quality improvement) | Task 6 | ✅ proper modal UX |

**Placeholder scan:** No TBDs. All code blocks are complete. Route handler code shows full implementations.

**Type consistency:** `SourceFile`, `SourceConnector` — same types used throughout all 4 connectors (Tasks 1-4) matching the Protocol defined in `connectors/base.py`. `creds` dict structure matches between `_buildCredentials()` JS (Task 6) and `_run_test()` branches (Task 5).
