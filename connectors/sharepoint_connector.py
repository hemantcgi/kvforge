"""SharePoint / OneDrive connector via Microsoft Graph API."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path

import msal
import requests

from connectors.base import SourceFile


_GRAPH_BASE = "https://graph.microsoft.com/v1.0"


class SharePointConnector:
    """Connect to a SharePoint document library via Microsoft Graph API."""

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        site_id: str,
        drive_id: str,
        local_mirror_path: str = "",
    ):
        self.site_id = site_id
        self.drive_id = drive_id
        self.local_mirror_path = local_mirror_path

        if not local_mirror_path:
            authority = f"https://login.microsoftonline.com/{tenant_id}"
            self._app = msal.ConfidentialClientApplication(
                client_id,
                authority=authority,
                client_credential=client_secret,
            )

    def _token(self) -> str:
        scopes = ["https://graph.microsoft.com/.default"]
        result = self._app.acquire_token_silent(scopes, account=None)
        if not result:
            result = self._app.acquire_token_for_client(scopes=scopes)
        return result["access_token"]

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token()}"}

    def _drive_url(self) -> str:
        return f"{_GRAPH_BASE}/sites/{self.site_id}/drives/{self.drive_id}"

    def _parse_item(self, item: dict) -> SourceFile | None:
        if "file" not in item:
            return None
        modified_raw = item.get("lastModifiedDateTime", "")
        try:
            modified_at = datetime.fromisoformat(modified_raw.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            modified_at = datetime.now(timezone.utc)
        return SourceFile(
            id=item["id"],
            name=item["name"],
            path=item.get("parentReference", {}).get("path", "") + "/" + item["name"],
            size=item.get("size", 0),
            modified_at=modified_at,
            mime_type=item.get("file", {}).get("mimeType"),
        )

    def list_files(self) -> list[SourceFile]:
        if self.local_mirror_path:
            return self._list_local()
        url = f"{self._drive_url()}/root/children?$top=999"
        files: list[SourceFile] = []
        while url:
            resp = requests.get(url, headers=self._headers(), timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("value", []):
                sf = self._parse_item(item)
                if sf:
                    files.append(sf)
            url = data.get("@odata.nextLink")
        return files

    def download(self, file: SourceFile) -> bytes:
        if self.local_mirror_path:
            return Path(file.path).read_bytes()
        url = f"{self._drive_url()}/items/{file.id}/content"
        resp = requests.get(url, headers=self._headers(), timeout=30)
        resp.raise_for_status()
        return resp.content

    def get_modified_at(self, file: SourceFile) -> datetime:
        return file.modified_at

    def supports_delta(self) -> bool:
        return not bool(self.local_mirror_path)

    def get_delta(self, token: str | None) -> tuple[list[SourceFile], str]:
        if self.local_mirror_path:
            return self._list_local(), ""
        url = token or f"{self._drive_url()}/root/delta"
        files: list[SourceFile] = []
        data = {}
        while url:
            resp = requests.get(url, headers=self._headers(), timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("value", []):
                sf = self._parse_item(item)
                if sf:
                    files.append(sf)
            url = data.get("@odata.nextLink")
        new_token = data.get("@odata.deltaLink", token or "")
        return files, new_token

    def _list_local(self) -> list[SourceFile]:
        root = Path(self.local_mirror_path)
        files: list[SourceFile] = []
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            stat = p.stat()
            files.append(SourceFile(
                id=str(p.relative_to(root)),
                name=p.name,
                path=str(p),
                size=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            ))
        return files
