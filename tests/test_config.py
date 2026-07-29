"""Tests for KVForgeConfig (replaces old DatasourceConfig tests)."""
import json
import pytest


def test_config_loads_with_required_fields():
    from core.config import KVForgeConfig
    cfg = KVForgeConfig(
        use_case_name="Test",
        collection="test-col",
        version_file="v.json",
    )
    assert cfg.addons == []
    assert cfg.addon_config == {}


def test_config_missing_collection_raises():
    from core.config import KVForgeConfig
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        KVForgeConfig(use_case_name="Test", version_file="v.json")


def test_load_config_from_file(tmp_path):
    from core.config import load_config
    data = {
        "use_case_name": "UC",
        "collection": "col",
        "version_file": "v.json",
        "addons": ["indexing"],
        "addon_config": {
            "indexing": {"embed_model": "BAAI/bge-small-en-v1.5", "vector_dim": 384}
        },
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(data))
    cfg = load_config(str(p))
    assert cfg.use_case_name == "UC"
    assert cfg.has_addon("indexing")


def test_get_merged_config_pipeline_compat():
    from core.config import KVForgeConfig
    cfg = KVForgeConfig(
        use_case_name="UC",
        collection="col",
        version_file="v.json",
        addons=["indexing", "inference"],
        addon_config={
            "indexing": {"embed_model": "BAAI/bge-small-en-v1.5", "vector_dim": 384,
                          "vector_store": "qdrant"},
            "inference": {"llm_model": "meta-llama/Llama-3.2-3B-Instruct", "top_k": 5},
        },
    )
    merged = cfg.get_merged_config("indexing", "inference")
    # These are the exact keys that kv_indexer.py and kv_inference.py use
    assert merged["collection"] == "col"
    assert merged["embed_model"] == "BAAI/bge-small-en-v1.5"
    assert merged["llm_model"] == "meta-llama/Llama-3.2-3B-Instruct"
    assert merged["top_k"] == 5
    assert merged["vector_store"] == "qdrant"


def test_get_merged_config_includes_thresholds():
    from core.config import KVForgeConfig
    cfg = KVForgeConfig(
        use_case_name="UC",
        collection="col",
        version_file="v.json",
        kds_threshold=0.45,
        fkds_threshold=0.55,
    )
    merged = cfg.get_merged_config()
    assert merged["kds_threshold"] == 0.45
    assert merged["fkds_threshold"] == 0.55


def test_load_config_strips_comment_keys(tmp_path):
    from core.config import load_config
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({
        "_comment": "ignored",
        "use_case_name": "UC",
        "collection": "col",
        "version_file": "v.json",
    }))
    cfg = load_config(str(p))
    assert cfg.collection == "col"


def test_confidence_flags_in_config_and_merge():
    from core.config import KVForgeConfig
    cfg = KVForgeConfig(
        use_case_name="UC",
        collection="col",
        version_file="v.json",
        confidence_supervision=True,
        use_confidence_token=True,
        confidence_label_threshold=0.6,
    )
    assert cfg.confidence_supervision is True
    assert cfg.use_confidence_token is True
    assert cfg.confidence_label_threshold == 0.6
    merged = cfg.get_merged_config()
    assert merged["confidence_supervision"] is True
    assert merged["use_confidence_token"] is True
    assert merged["confidence_label_threshold"] == 0.6
