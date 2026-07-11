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
