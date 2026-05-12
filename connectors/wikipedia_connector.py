"""Wikipedia REST API connector — fetches article text by topic list."""
from __future__ import annotations
from datetime import datetime, timezone

import httpx

from connectors.base import SourceFile

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
