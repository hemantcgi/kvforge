import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_check_rewrite_coverage_pass():
    from tools.validate_diversity_data import check_rewrite_coverage
    rewrites = [
        {"chunk_id": "c0", "text": "Amazon Bedrock is a managed ML service."},
        {"chunk_id": "c1", "text": "Claude is an LLM by Anthropic."},
    ]
    source = [
        {"chunk_id": "c0", "text": "Amazon Bedrock is a managed service for ML."},
        {"chunk_id": "c1", "text": "Claude is an LLM developed by Anthropic."},
    ]
    result = check_rewrite_coverage(rewrites, source)
    assert result["pass"] is True
    assert result["coverage_rate"] >= 0.9


def test_check_rewrite_coverage_fail_low_overlap():
    from tools.validate_diversity_data import check_rewrite_coverage
    rewrites = [{"chunk_id": "c0", "text": "The weather is nice today."}]
    source = [{"chunk_id": "c0", "text": "Amazon Bedrock is a managed service for ML."}]
    result = check_rewrite_coverage(rewrites, source)
    assert result["pass"] is False
    assert result["coverage_rate"] < 0.5


def test_check_qa_grounding_pass():
    from tools.validate_diversity_data import check_qa_grounding
    qa_pairs = [
        {"chunk_id": "c0", "question": "What is Bedrock?", "answer": "A managed ML service."},
    ]
    source = [{"chunk_id": "c0", "text": "Amazon Bedrock is a managed service for ML."}]
    result = check_qa_grounding(qa_pairs, source)
    assert result["pass"] is True
    assert result["grounding_rate"] >= 0.9


def test_check_entity_graph_quality_cross_chunk():
    from tools.validate_diversity_data import check_entity_graph_quality
    relations = [
        {"chunk_id": "c0", "text": "A relates to B.", "entity_pair": ("A", "B")},
        {"chunk_id": "c1", "text": "B relates to C.", "entity_pair": ("B", "C")},
    ]
    entities = {
        "A": {"chunk_ids": ["c0", "c1"]},
        "B": {"chunk_ids": ["c0", "c1"]},
        "C": {"chunk_ids": ["c1"]},
    }
    result = check_entity_graph_quality(relations, entities)
    assert result["entity_coverage"] >= 0.6
