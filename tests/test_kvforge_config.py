import json
import pytest


def test_kvforge_config_minimal_valid():
    from core.config import KVForgeConfig
    cfg = KVForgeConfig(
        use_case_name="Test UC",
        collection="test-col",
        version_file="v.json",
    )
    assert cfg.use_case_name == "Test UC"
    assert cfg.collection == "test-col"
    assert cfg.addons == []
    assert cfg.addon_config == {}


def test_kvforge_config_with_addons():
    from core.config import KVForgeConfig
    cfg = KVForgeConfig(
        use_case_name="My RAG",
        collection="my-col",
        version_file="v.json",
        addons=["indexing", "inference"],
        addon_config={
            "indexing": {"embed_model": "BAAI/bge-small-en-v1.5", "vector_dim": 384,
                          "vector_store": "qdrant"},
            "inference": {"llm_model": "meta-llama/Llama-3.2-3B-Instruct", "top_k": 5},
        },
    )
    assert "indexing" in cfg.addons
    assert cfg.addon_config["indexing"]["embed_model"] == "BAAI/bge-small-en-v1.5"


def test_has_addon():
    from core.config import KVForgeConfig
    cfg = KVForgeConfig(
        use_case_name="UC",
        collection="col",
        version_file="v.json",
        addons=["indexing", "training"],
    )
    assert cfg.has_addon("indexing") is True
    assert cfg.has_addon("mcp") is False


def test_get_merged_config_includes_core_fields():
    from core.config import KVForgeConfig
    cfg = KVForgeConfig(
        use_case_name="UC",
        collection="my-col",
        version_file="v.json",
        addons=["indexing"],
        addon_config={"indexing": {"embed_model": "BAAI/bge-small-en-v1.5",
                                    "vector_dim": 384}},
    )
    merged = cfg.get_merged_config("indexing")
    assert merged["collection"] == "my-col"
    assert merged["version_file"] == "v.json"
    assert merged["embed_model"] == "BAAI/bge-small-en-v1.5"


def test_get_merged_config_multiple_addons():
    from core.config import KVForgeConfig
    cfg = KVForgeConfig(
        use_case_name="UC",
        collection="col",
        version_file="v.json",
        addons=["inference", "training"],
        addon_config={
            "inference": {"llm_model": "llama", "top_k": 5},
            "training": {"lora_rank": 16, "checkpoint_dir": "ckpts/",
                          "replay_db": "r.db"},
        },
    )
    merged = cfg.get_merged_config("inference", "training")
    assert merged["llm_model"] == "llama"
    assert merged["lora_rank"] == 16
    assert merged["collection"] == "col"


def test_get_merged_config_missing_addon_returns_empty():
    from core.config import KVForgeConfig
    cfg = KVForgeConfig(
        use_case_name="UC",
        collection="col",
        version_file="v.json",
    )
    merged = cfg.get_merged_config("indexing")
    assert merged["collection"] == "col"
    # No indexing addon_config — no extra keys beyond core
    assert "embed_model" not in merged


def test_load_config_from_json(tmp_path):
    from core.config import load_config
    data = {
        "use_case_name": "JSON UC",
        "collection": "json-col",
        "version_file": "v.json",
        "addons": ["inference"],
        "addon_config": {
            "inference": {"llm_model": "meta-llama/Llama-3.2-3B-Instruct"}
        },
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(data))
    cfg = load_config(str(p))
    assert cfg.use_case_name == "JSON UC"
    assert cfg.has_addon("inference")


def test_load_config_strips_comment_keys(tmp_path):
    from core.config import load_config
    data = {
        "_comment": "This is a comment",
        "use_case_name": "UC",
        "collection": "col",
        "version_file": "v.json",
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(data))
    cfg = load_config(str(p))
    assert cfg.use_case_name == "UC"


def test_kvforge_config_missing_collection_raises():
    from core.config import KVForgeConfig
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        KVForgeConfig(use_case_name="UC", version_file="v.json")
