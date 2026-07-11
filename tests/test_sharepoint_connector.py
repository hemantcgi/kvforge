import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
sys.path.insert(0, str(Path(__file__).parent.parent))


def _mock_msal_app(token: str = "test_token"):
    app = MagicMock()
    app.acquire_token_silent.return_value = None
    app.acquire_token_for_client.return_value = {"access_token": token}
    return app


def _mock_graph_response(items: list[dict]):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"value": items}
    return resp


def test_sharepoint_list_files(tmp_path):
    from connectors.sharepoint_connector import SharePointConnector
    modified = "2026-03-01T12:00:00Z"
    mock_items = [{
        "id": "item1",
        "name": "policy.docx",
        "size": 2048,
        "lastModifiedDateTime": modified,
        "file": {"mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
        "parentReference": {"path": "/drives/root:/docs"},
    }]
    with patch("connectors.sharepoint_connector.msal.ConfidentialClientApplication",
               return_value=_mock_msal_app()), \
         patch("connectors.sharepoint_connector.requests.get",
               return_value=_mock_graph_response(mock_items)):
        conn = SharePointConnector(
            tenant_id="tenant123", client_id="client123", client_secret="secret",
            site_id="site123", drive_id="drive123",
        )
        files = conn.list_files()
    assert len(files) == 1
    assert files[0].name == "policy.docx"
    assert files[0].id == "item1"
    assert files[0].size == 2048


def test_sharepoint_download(tmp_path):
    from connectors.sharepoint_connector import SharePointConnector
    download_resp = MagicMock()
    download_resp.status_code = 200
    download_resp.content = b"docx bytes"
    with patch("connectors.sharepoint_connector.msal.ConfidentialClientApplication",
               return_value=_mock_msal_app()), \
         patch("connectors.sharepoint_connector.requests.get",
               return_value=download_resp):
        conn = SharePointConnector(
            tenant_id="t", client_id="c", client_secret="s",
            site_id="site1", drive_id="drive1",
        )
        from connectors.base import SourceFile
        sf = SourceFile(id="item1", name="f.docx", path="/f.docx",
                        size=10, modified_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
        data = conn.download(sf)
    assert data == b"docx bytes"


def test_sharepoint_local_mirror_fallback(tmp_path):
    from connectors.sharepoint_connector import SharePointConnector
    (tmp_path / "report.docx").write_bytes(b"local content")
    conn = SharePointConnector(
        tenant_id="t", client_id="c", client_secret="s",
        site_id="site1", drive_id="drive1",
        local_mirror_path=str(tmp_path),
    )
    files = conn.list_files()
    assert any(f.name == "report.docx" for f in files)


def test_sharepoint_supports_delta_true():
    from connectors.sharepoint_connector import SharePointConnector
    with patch("connectors.sharepoint_connector.msal.ConfidentialClientApplication",
               return_value=_mock_msal_app()):
        conn = SharePointConnector(
            tenant_id="t", client_id="c", client_secret="s",
            site_id="s1", drive_id="d1",
        )
    assert conn.supports_delta() is True
