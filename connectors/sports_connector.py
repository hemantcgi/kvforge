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
