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
