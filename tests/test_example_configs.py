"""Verify all example configs load as valid KVForgeConfig in the new format."""
import pytest


@pytest.mark.parametrize("path", [
    "examples/usecase1_customer_support/config.json",
    "examples/usecase2_pubmedqa/config.json",
    "examples/usecase3_squad/config.json",
    "examples/usecase4_bedrock_userguide/config.json",
])
def test_example_config_loads(path):
    from core.config import load_config
    cfg = load_config(path)
    assert cfg.use_case_name, f"{path}: use_case_name is empty"
    assert cfg.collection, f"{path}: collection is empty"
    assert cfg.version_file, f"{path}: version_file is empty"
    assert len(cfg.addons) >= 2, f"{path}: expected at least indexing + inference addons"
    assert "indexing" in cfg.addons
    assert "inference" in cfg.addons


@pytest.mark.parametrize("path", [
    "examples/usecase1_customer_support/config.json",
    "examples/usecase2_pubmedqa/config.json",
    "examples/usecase3_squad/config.json",
    "examples/usecase4_bedrock_userguide/config.json",
])
def test_example_config_addon_schemas_valid(path):
    from core.config import load_config
    from addons.registry import AddonRegistry
    AddonRegistry.load_builtins()

    cfg = load_config(path)
    for addon_name in cfg.addons:
        typed = cfg.get_validated_addon_config(addon_name)
        assert typed is not None, f"{path}: addon '{addon_name}' failed validation"


@pytest.mark.parametrize("path", [
    "examples/usecase1_customer_support/config.json",
    "examples/usecase2_pubmedqa/config.json",
    "examples/usecase3_squad/config.json",
    "examples/usecase4_bedrock_userguide/config.json",
])
def test_example_config_deps_satisfied(path):
    from core.config import load_config
    from addons.registry import AddonRegistry
    AddonRegistry.load_builtins()

    cfg = load_config(path)
    cfg.validate_addon_deps()  # must not raise


@pytest.mark.parametrize("path,expected_store", [
    ("examples/usecase1_customer_support/config.json", "qdrant"),
    ("examples/usecase2_pubmedqa/config.json", "chroma"),
    ("examples/usecase3_squad/config.json", "qdrant"),
    ("examples/usecase4_bedrock_userguide/config.json", "qdrant"),
])
def test_example_config_vector_store(path, expected_store):
    from core.config import load_config
    cfg = load_config(path)
    merged = cfg.get_merged_config("indexing")
    assert merged.get("vector_store") == expected_store


@pytest.mark.parametrize("path", [
    "examples/usecase1_customer_support/config.json",
    "examples/usecase2_pubmedqa/config.json",
    "examples/usecase3_squad/config.json",
    "examples/usecase4_bedrock_userguide/config.json",
])
def test_get_merged_config_contains_pipeline_keys(path):
    """Verify get_merged_config() produces the flat dict pipeline code expects."""
    from core.config import load_config
    cfg = load_config(path)
    merged = cfg.get_merged_config("indexing", "inference", "training")
    # Keys that kv_indexer.py, kv_inference.py, lora_trainer.py use via cfg.get()
    for key in ["collection", "version_file", "embed_model", "vector_dim",
                "vector_store", "llm_model", "top_k", "gate_threshold",
                "lora_rank", "checkpoint_dir", "replay_db"]:
        assert key in merged, f"{path}: merged config missing '{key}'"
