import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock
from datetime import datetime, timezone


def _mock_s3_client(objects: list[dict]):
    """objects: list of {Key, Size, LastModified}"""
    client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": objects, "IsTruncated": False}]
    client.get_paginator.return_value = paginator
    client.get_object.return_value = {"Body": MagicMock(read=lambda: b"file content")}
    return client


def test_source_file_dataclass_fields():
    from connectors.base import SourceFile
    sf = SourceFile(
        id="abc123",
        name="report.docx",
        path="/docs/report.docx",
        size=1024,
        modified_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert sf.id == "abc123"
    assert sf.mime_type is None
    assert sf.extra == {}


def test_source_connector_protocol_structural():
    from connectors.base import SourceConnector
    assert hasattr(SourceConnector, "list_files")


def test_s3_list_files(tmp_path):
    from connectors.s3_connector import S3Connector
    modified = datetime(2026, 3, 1, tzinfo=timezone.utc)
    mock_client = _mock_s3_client([
        {"Key": "docs/report.docx", "Size": 512, "LastModified": modified},
    ])
    with patch("connectors.s3_connector.boto3.client", return_value=mock_client):
        conn = S3Connector(bucket="my-bucket", prefix="docs/", region="us-east-1",
                           access_key_id="AK", secret_access_key="SK")
        files = conn.list_files()
    assert len(files) == 1
    assert files[0].name == "report.docx"
    assert files[0].id == "docs/report.docx"
    assert files[0].size == 512


def test_s3_download(tmp_path):
    from connectors.s3_connector import S3Connector
    from connectors.base import SourceFile
    modified = datetime(2026, 3, 1, tzinfo=timezone.utc)
    mock_client = _mock_s3_client([])
    mock_client.get_object.return_value = {"Body": MagicMock(read=lambda: b"hello bytes")}
    with patch("connectors.s3_connector.boto3.client", return_value=mock_client):
        conn = S3Connector(bucket="b", prefix="", region="us-east-1",
                           access_key_id="AK", secret_access_key="SK")
        sf = SourceFile(id="key.docx", name="key.docx", path="key.docx",
                        size=11, modified_at=modified)
        data = conn.download(sf)
    assert data == b"hello bytes"


def test_s3_local_mirror_path(tmp_path):
    from connectors.s3_connector import S3Connector
    (tmp_path / "report.docx").write_bytes(b"local content")
    conn = S3Connector(bucket="b", prefix="", region="us-east-1",
                       access_key_id="", secret_access_key="",
                       local_mirror_path=str(tmp_path))
    files = conn.list_files()
    assert any(f.name == "report.docx" for f in files)


def test_s3_local_mirror_download(tmp_path):
    from connectors.s3_connector import S3Connector
    from connectors.base import SourceFile
    (tmp_path / "data.docx").write_bytes(b"mirror bytes")
    conn = S3Connector(bucket="b", prefix="", region="us-east-1",
                       access_key_id="", secret_access_key="",
                       local_mirror_path=str(tmp_path))
    sf = SourceFile(id="data.docx", name="data.docx", path=str(tmp_path / "data.docx"),
                    size=12, modified_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    data = conn.download(sf)
    assert data == b"mirror bytes"


def test_s3_supports_delta_false():
    from connectors.s3_connector import S3Connector
    with patch("connectors.s3_connector.boto3.client", return_value=MagicMock()):
        conn = S3Connector(bucket="b", prefix="", region="us-east-1",
                           access_key_id="k", secret_access_key="s")
    assert conn.supports_delta() is False


def test_s3_implements_source_connector_protocol():
    from connectors.s3_connector import S3Connector
    from connectors.base import SourceConnector
    with patch("connectors.s3_connector.boto3.client", return_value=MagicMock()):
        conn = S3Connector(bucket="b", prefix="", region="us-east-1",
                           access_key_id="k", secret_access_key="s")
    assert isinstance(conn, SourceConnector)


def test_s3_prefix_stripped_from_file_name(tmp_path):
    from connectors.s3_connector import S3Connector
    modified = datetime(2026, 3, 1, tzinfo=timezone.utc)
    mock_client = _mock_s3_client([
        {"Key": "docs/sub/file.docx", "Size": 100, "LastModified": modified},
    ])
    with patch("connectors.s3_connector.boto3.client", return_value=mock_client):
        conn = S3Connector(bucket="b", prefix="docs/", region="us-east-1",
                           access_key_id="AK", secret_access_key="SK")
        files = conn.list_files()
    assert files[0].name == "file.docx"  # just the filename, not the full key
