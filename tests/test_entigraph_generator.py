import json
import pytest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


_MOCK_ENTITIES = [
    {"name": "Amazon Bedrock", "type": "service", "description": "Managed ML platform", "facts": ["Launched in 2023"]},
    {"name": "Claude", "type": "model", "description": "LLM by Anthropic", "facts": ["Available on Bedrock"]},
]

_MOCK_RELATION = "Amazon Bedrock provides access to Claude, an LLM developed by Anthropic for enterprise use."


def test_extract_entities_parses_json():
    from pipeline.entigraph_generator import extract_entities
    mock_response = json.dumps({"entities": _MOCK_ENTITIES})
    with patch("pipeline.entigraph_generator._call_llm", return_value=mock_response):
        result = extract_entities("some text", "gemini", "gemini-2.5-flash", "fake")
    assert len(result) == 2
    assert result[0]["name"] == "Amazon Bedrock"
    assert "facts" in result[0]


def test_extract_entities_handles_malformed_json():
    from pipeline.entigraph_generator import extract_entities
    with patch("pipeline.entigraph_generator._call_llm", return_value="not json"):
        result = extract_entities("some text", "gemini", "gemini-2.5-flash", "fake")
    assert result == []


def test_build_entity_table_finds_cross_chunk_pairs():
    from pipeline.entigraph_generator import build_entity_table
    # EntityA appears in both chunks, EntityB only in chunk_1, EntityC only in chunk_0.
    # A&B share chunk_1, A&C share chunk_0 → both are cross-chunk pairs.
    entities_by_chunk = {
        "chunk_0": [{"name": "EntityA", "type": "concept", "description": "Desc A", "facts": []},
                     {"name": "EntityC", "type": "concept", "description": "Desc C", "facts": []}],
        "chunk_1": [{"name": "EntityA", "type": "concept", "description": "Desc A", "facts": []},
                     {"name": "EntityB", "type": "concept", "description": "Desc B", "facts": []}],
    }
    table = build_entity_table(entities_by_chunk)
    assert "EntityA" in table["entities"]
    assert set(table["entities"]["EntityA"]["chunk_ids"]) == {"chunk_0", "chunk_1"}
    assert len(table["cross_chunk_pairs"]) >= 2


def test_build_entity_table_no_cross_chunk():
    from pipeline.entigraph_generator import build_entity_table
    entities_by_chunk = {
        "chunk_0": [{"name": "A", "type": "x", "description": "d", "facts": []}],
        "chunk_1": [{"name": "B", "type": "x", "description": "d", "facts": []}],
    }
    table = build_entity_table(entities_by_chunk)
    assert len(table["cross_chunk_pairs"]) == 0


def test_generate_relation_texts_with_mock():
    from pipeline.entigraph_generator import generate_relation_texts
    entity_table = {
        "entities": {
            "A": {"name": "EntityA", "type": "concept", "description": "Desc A", "chunk_ids": ["c0", "c1"]},
            "B": {"name": "EntityB", "type": "concept", "description": "Desc B", "chunk_ids": ["c1"]},
        },
        "cross_chunk_pairs": [("A", "B", "c0", "c1")],
    }
    chunks = {"c0": "Text about A.", "c1": "Text about A and B."}
    with patch("pipeline.entigraph_generator._call_llm", return_value=_MOCK_RELATION):
        result = generate_relation_texts(entity_table, chunks, n_per_chunk=1,
                                          provider="gemini", model="gemini-2.5-flash", api_key="fake")
    assert len(result) >= 1
    assert "text" in result[0]
    assert "chunk_id" in result[0]


def test_generate_full_pipeline_writes_manifest(tmp_path):
    from pipeline.entigraph_generator import generate
    chunks = [
        {"chunk_id": "c0", "text": "Amazon Bedrock is a managed service."},
        {"chunk_id": "c1", "text": "Claude is available on Bedrock."},
    ]
    # Entity extraction must differ per chunk to yield cross-chunk pairs:
    # c0 → Bedrock only; c1 → Bedrock + Claude. Then Bedrock appears in both
    # chunks while Claude appears only in c1 → cross-chunk pair (Bedrock, Claude).
    def _mock_llm_fn(prompt, *args, **kwargs):
        if "Amazon Bedrock is a managed" in prompt:
            return json.dumps({"entities": [{"name": "Bedrock", "type": "service",
                                              "description": "Cloud ML", "facts": ["Managed"]}]})
        if "Claude is available" in prompt:
            return json.dumps({"entities": [
                {"name": "Claude", "type": "model", "description": "LLM", "facts": ["By Anthropic"]},
                {"name": "Bedrock", "type": "service", "description": "Cloud ML", "facts": ["Managed"]},
            ]})
        return _MOCK_RELATION

    with patch("pipeline.entigraph_generator._call_llm", side_effect=_mock_llm_fn):
        output = generate(chunks, tmp_path / "entigraph_manifest.json",
                          n_per_chunk=1, api_key="fake")
    assert output.exists()
    data = json.loads(output.read_text())
    assert "training_examples" in data
    assert len(data["training_examples"]) >= 1
