import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
sys.path.insert(0, str(Path(__file__).parent.parent))


def _mock_drive_service(files: list[dict]):
    service = MagicMock()
    files_resource = MagicMock()
    list_call = MagicMock()
    list_call.execute.return_value = {"files": files, "nextPageToken": None}
    files_resource.list.return_value = list_call
    export_call = MagicMock()
    export_call.execute.return_value = b"exported content"
    get_media_call = MagicMock()
    get_media_call.execute.return_value = b"binary content"
    files_resource.export_media.return_value = export_call
    files_resource.get_media.return_value = get_media_call
    service.files.return_value = files_resource
    return service


def test_gdrive_list_files():
    from connectors.gdrive_connector import GDriveConnector
    mock_files = [{
        "id": "file1",
        "name": "budget.xlsx",
        "size": "1024",
        "modifiedTime": "2026-03-01T10:00:00.000Z",
        "mimeType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }]
    with patch("connectors.gdrive_connector.build", return_value=_mock_drive_service(mock_files)), \
         patch("connectors.gdrive_connector.service_account.Credentials.from_service_account_file",
               return_value=MagicMock()):
        conn = GDriveConnector(service_account_file="fake.json", folder_id="folder1")
        files = conn.list_files()
    assert len(files) == 1
    assert files[0].name == "budget.xlsx"


def test_gdrive_download_binary():
    from connectors.gdrive_connector import GDriveConnector
    from connectors.base import SourceFile
    with patch("connectors.gdrive_connector.build", return_value=_mock_drive_service([])), \
         patch("connectors.gdrive_connector.service_account.Credentials.from_service_account_file",
               return_value=MagicMock()):
        conn = GDriveConnector(service_account_file="fake.json", folder_id="folder1")
        sf = SourceFile(id="f1", name="doc.xlsx", path="/doc.xlsx",
                        size=10, modified_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        data = conn.download(sf)
    assert data == b"binary content"


def test_gdrive_download_workspace_doc():
    """Google Workspace mime type should use export_media, not get_media."""
    from connectors.gdrive_connector import GDriveConnector
    from connectors.base import SourceFile
    with patch("connectors.gdrive_connector.build", return_value=_mock_drive_service([])), \
         patch("connectors.gdrive_connector.service_account.Credentials.from_service_account_file",
               return_value=MagicMock()):
        conn = GDriveConnector(service_account_file="fake.json", folder_id="folder1")
        sf = SourceFile(id="doc1", name="report.docx", path="/report.docx",
                        size=10, modified_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                        mime_type="application/vnd.google-apps.document")
        data = conn.download(sf)
    assert data == b"exported content"


def test_gdrive_local_mirror_fallback(tmp_path):
    from connectors.gdrive_connector import GDriveConnector
    (tmp_path / "notes.docx").write_bytes(b"notes")
    conn = GDriveConnector(service_account_file="fake.json",
                            folder_id="folder1",
                            local_mirror_path=str(tmp_path))
    files = conn.list_files()
    assert any(f.name == "notes.docx" for f in files)
