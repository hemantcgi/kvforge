import pytest
from pathlib import Path


def test_local_backend_write_read(tmp_path):
    from addons.corpus_intelligence.archive import LocalArchiveBackend
    backend = LocalArchiveBackend(archive_dir=str(tmp_path))
    backend.write("chunk_abc", "This is the archived chunk text.")
    result = backend.read("chunk_abc")
    assert result == "This is the archived chunk text."


def test_local_backend_delete(tmp_path):
    from addons.corpus_intelligence.archive import LocalArchiveBackend
    backend = LocalArchiveBackend(archive_dir=str(tmp_path))
    backend.write("chunk_del", "some content")
    backend.delete("chunk_del")
    assert backend.read("chunk_del") is None


def test_local_backend_missing_returns_none(tmp_path):
    from addons.corpus_intelligence.archive import LocalArchiveBackend
    backend = LocalArchiveBackend(archive_dir=str(tmp_path))
    assert backend.read("nonexistent_chunk") is None


def test_local_backend_pointer(tmp_path):
    from addons.corpus_intelligence.archive import LocalArchiveBackend
    backend = LocalArchiveBackend(archive_dir=str(tmp_path))
    backend.write("chunk_ptr", "content")
    pointer = backend.get_pointer("chunk_ptr")
    assert pointer.startswith(str(tmp_path))
    assert pointer.endswith(".txt")


def test_backend_protocol_compliance():
    from addons.corpus_intelligence.archive import LocalArchiveBackend, ArchiveBackend
    for method in ("write", "read", "delete", "get_pointer"):
        assert hasattr(LocalArchiveBackend, method), f"Missing method: {method}"
