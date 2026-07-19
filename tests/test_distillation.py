"""Tests for pipeline/distillation.py — Sprint 2 query pool and quality filter."""

import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.distillation import (
    normalize_question,
    deduplicate_pool,
    quality_filter,
    load_real_queries,
    expand_faq_questions,
    generate_chunk_questions,
    build_query_pool,
)


def test_normalize_question():
    assert normalize_question("What is Bedrock?") == "what is bedrock"
    assert normalize_question("  WHAT is Bedrock?! ") == "what is bedrock"


def test_deduplicate_pool_keeps_first():
    entries = [
        {"question": "What is Bedrock?", "source": "a"},
        {"question": "what is bedrock", "source": "b"},
        {"question": "How does Bedrock work?", "source": "c"},
    ]
    result = deduplicate_pool(entries)
    assert len(result) == 2
    assert result[0]["source"] == "a"
    assert result[1]["source"] == "c"


def test_quality_filter_keeps_high_accuracy(monkeypatch):
    from eval import metrics

    def fake_token_f1(a, b):
        return 0.9

    def fake_llm_judge(*args, **kwargs):
        return {"factually_correct": True}

    monkeypatch.setattr(metrics, "token_f1", fake_token_f1)
    monkeypatch.setattr(metrics, "llm_judge", fake_llm_judge)

    pairs = [
        {"question": "Q1", "teacher_answer": "A", "expected_answer": "A"},
        {"question": "Q2", "teacher_answer": "B", "expected_answer": "B"},
    ]
    filtered, stats = quality_filter(pairs, threshold=0.7)
    assert len(filtered) == 2
    assert stats["kept"] == 2
    assert stats["drop_rate"] == 0.0


def test_quality_filter_drops_low_accuracy(monkeypatch):
    from eval import metrics

    def fake_token_f1(a, b):
        return 0.1

    def fake_llm_judge(*args, **kwargs):
        return {"factually_correct": False}

    monkeypatch.setattr(metrics, "token_f1", fake_token_f1)
    monkeypatch.setattr(metrics, "llm_judge", fake_llm_judge)

    pairs = [
        {"question": "Q1", "teacher_answer": "A", "expected_answer": "A"},
    ]
    filtered, stats = quality_filter(pairs, threshold=0.7)
    assert len(filtered) == 0
    assert stats["kept"] == 0
    assert stats["dropped"] == 1


def test_load_real_queries(tmp_path):
    from pipeline import query_logger
    db_path = tmp_path / "q.db"
    query_logger.init_db(str(db_path))
    query_logger.log_query(
        db_path=str(db_path),
        query_text="What is Bedrock?",
        answer_text="A managed service.",
        routed_to="retrieval",
        cluster_id="0",
        chunk_id="1",
    )

    rows = load_real_queries(str(db_path))
    assert len(rows) == 1
    assert rows[0]["question"] == "What is Bedrock?"
    assert rows[0]["source"] == "real_query"


def test_expand_faq_questions():
    faqs = [{"question": "What is Bedrock?", "answer": "A service."}]
    with patch("pipeline.distillation._generate_text",
               return_value="- What is Amazon Bedrock?\n- Tell me about Bedrock.\n- Explain Bedrock."):
        entries = expand_faq_questions(faqs, MagicMock(), MagicMock(), n=3)
    assert len(entries) == 3
    assert entries[0]["source"] == "faq_paraphrase"
    assert entries[0]["expected_answer"] == "A service."


def test_generate_chunk_questions():
    chunks = [{"chunk_id": 1, "text": "Bedrock provides managed foundation model access."}]
    with patch("pipeline.distillation._generate_text",
               return_value="What feature does Bedrock provide?\nAnswer"):
        entries = generate_chunk_questions(chunks, MagicMock(), MagicMock())
    assert len(entries) == 1
    assert entries[0]["source"] == "chunk_question"
    assert entries[0]["chunk_id"] == 1


def test_build_query_pool_combines_sources():
    faqs = [
        {"question": "What is Bedrock?", "answer": "A service."},
    ]
    chunks = [{"chunk_id": 1, "text": "Bedrock is a service."}]

    # Return distinct paraphrases and chunk questions so deduplication keeps both sources.
    with patch("pipeline.distillation._generate_text",
               side_effect=["Para1\nPara2", "ChunkQ1\nChunkQ2"]):
        pool = build_query_pool(
            faqs=faqs,
            model=MagicMock(),
            tokenizer=MagicMock(),
            query_log_db=None,
            chunks=chunks,
            faq_paraphrases=2,
        )
    assert len(pool) >= 2
    sources = {e["source"] for e in pool}
    assert "faq_paraphrase" in sources
    assert "chunk_question" in sources
