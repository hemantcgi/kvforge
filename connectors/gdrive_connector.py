"""Google Drive connector via Google API Python Client."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build

from connectors.base import SourceConnector, SourceFile


_EXPORT_MAP = {
    "application/vnd.google-apps.document":
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.google-apps.spreadsheet":
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.google-apps.presentation":
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

_EXT_MAP = {
    "application/vnd.google-apps.document": ".docx",
    "application/vnd.google-apps.spreadsheet": ".xlsx",
    "application/vnd.google-apps.presentation": ".pptx",
}

_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class GDriveConnector:
    """List and download files from a Google Drive folder."""

    def __init__(
        self,
        service_account_file: str,
        folder_id: str,
        local_mirror_path: str = "",
    ):
        self.folder_id = folder_id
        self.local_mirror_path = local_mirror_path
        if not local_mirror_path:
            creds = service_account.Credentials.from_service_account_file(
                service_account_file, scopes=_SCOPES
            )
            self._service = build("drive", "v3", credentials=creds)

    def list_files(self) -> list[SourceFile]:
        if self.local_mirror_path:
            return self._list_local()
        query = f"'{self.folder_id}' in parents and trashed=false"
        fields = "files(id,name,size,modifiedTime,mimeType),nextPageToken"
        files: list[SourceFile] = []
        page_token = None
        while True:
            kwargs = {
                "q": query,
                "fields": fields,
                "pageSize": 1000,
                "supportsAllDrives": True,
                "includeItemsFromAllDrives": True,
            }
            if page_token:
                kwargs["pageToken"] = page_token
            result = self._service.files().list(**kwargs).execute()
            for item in result.get("files", []):
                sf = self._parse_item(item)
                if sf:
                    files.append(sf)
            page_token = result.get("nextPageToken")
            if not page_token:
                break
        return files

    def download(self, file: SourceFile) -> bytes:
        if self.local_mirror_path:
            return Path(file.path).read_bytes()
        mime = file.mime_type or ""
        export_mime = _EXPORT_MAP.get(mime)
        if export_mime:
            return self._service.files().export_media(
                fileId=file.id, mimeType=export_mime
            ).execute()
        return self._service.files().get_media(fileId=file.id).execute()

    def get_modified_at(self, file: SourceFile) -> datetime:
        return file.modified_at

    def supports_delta(self) -> bool:
        return False

    def get_delta(self, token: str | None) -> tuple[list[SourceFile], str]:
        return self.list_files(), ""

    def _parse_item(self, item: dict) -> SourceFile | None:
        mime = item.get("mimeType", "")
        if mime == "application/vnd.google-apps.folder":
            return None
        name = item["name"]
        ext_override = _EXT_MAP.get(mime)
        if ext_override and not name.lower().endswith(ext_override):
            name = Path(name).stem + ext_override
        size_str = item.get("size", "0")
        try:
            size = int(size_str)
        except (ValueError, TypeError):
            size = 0
        modified_raw = item.get("modifiedTime", "")
        try:
            modified_at = datetime.fromisoformat(modified_raw.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            modified_at = datetime.now(timezone.utc)
        return SourceFile(
            id=item["id"],
            name=name,
            path=name,
            size=size,
            modified_at=modified_at,
            mime_type=mime,
        )

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
