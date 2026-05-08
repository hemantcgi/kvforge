import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_faq_tagged_with_source_chunk_ids():
    """Generated FAQs must carry source_chunk_ids so stale detection works."""
    from pipeline.sleep_faq_generator import tag_faq_with_chunk_ids
    faq = {"question": "What is X?", "answer": "X is Y."}
    chunk_ids = ["chunk_001", "chunk_002"]
    tagged = tag_faq_with_chunk_ids(faq, chunk_ids)
    assert tagged["source_chunk_ids"] == chunk_ids


def test_faq_temporal_grounding_in_prompt():
    """FAQ generation prompt must include effective_from date."""
    from pipeline.sleep_faq_generator import build_faq_prompt
    chunk = {
        "text": "The refund period is 30 days.",
        "metadata": {"effective_from": "2026-03-01T00:00:00+00:00", "source_file": "policy.docx"},
    }
    prompt = build_faq_prompt(chunk)
    assert "2026-03-01" in prompt
    assert "The refund period is 30 days." in prompt


def test_faq_staleness_detection():
    """FAQs whose source_chunk_ids are all superseded should be marked stale."""
    from pipeline.sleep_faq_generator import is_faq_stale
    faq = {"question": "Q", "answer": "A", "source_chunk_ids": ["chunk_001"]}
    superseded_ids = {"chunk_001", "chunk_002"}
    assert is_faq_stale(faq, superseded_ids) is True


def test_faq_not_stale_if_chunks_active():
    from pipeline.sleep_faq_generator import is_faq_stale
    faq = {"question": "Q", "answer": "A", "source_chunk_ids": ["chunk_active"]}
    superseded_ids = {"chunk_old"}
    assert is_faq_stale(faq, superseded_ids) is False
