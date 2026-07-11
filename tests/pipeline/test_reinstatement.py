import pytest
from pipeline.reinstatement_tracker import (
    check_reinstatement_candidates,
    build_reinstatement_recommendation,
)


def test_detects_chunks_over_threshold():
    archived_chunks = [
        {"id": "c1", "payload": {"archive_retrieval_count": 6, "status": "archived", "text": "t1"}},
        {"id": "c2", "payload": {"archive_retrieval_count": 3, "status": "archived", "text": "t2"}},
        {"id": "c3", "payload": {"archive_retrieval_count": 10, "status": "archived", "text": "t3"}},
    ]
    candidates = check_reinstatement_candidates(archived_chunks, threshold=5)
    candidate_ids = [c["id"] for c in candidates]
    assert "c1" in candidate_ids
    assert "c3" in candidate_ids
    assert "c2" not in candidate_ids


def test_recommendation_format():
    chunk = {"id": "cx", "payload": {"archive_retrieval_count": 8, "text": "Some text..."}}
    rec = build_reinstatement_recommendation(chunk, threshold=5)
    assert rec["chunk_id"] == "cx"
    assert "8" in rec["reason"]
    assert "5" in rec["reason"]
    assert rec["action"] == "reinstate"
