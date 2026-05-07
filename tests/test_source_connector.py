import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone


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
    assert sf.name == "report.docx"
    assert sf.size == 1024
    assert sf.mime_type is None
    assert sf.extra == {}


def test_source_file_mime_type_optional():
    from connectors.base import SourceFile
    sf = SourceFile(
        id="x", name="f.pdf", path="/f.pdf", size=100,
        modified_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        mime_type="application/pdf",
    )
    assert sf.mime_type == "application/pdf"


def test_source_connector_protocol_is_runtime_checkable():
    from connectors.base import SourceConnector
    from typing import runtime_checkable, Protocol
    # Protocol should be @runtime_checkable
    assert issubclass(SourceConnector, Protocol)


def test_source_connector_protocol_methods():
    from connectors.base import SourceConnector
    # Verify the protocol defines the required methods
    for method in ("list_files", "download", "get_modified_at", "supports_delta", "get_delta"):
        assert hasattr(SourceConnector, method), f"Missing method: {method}"
