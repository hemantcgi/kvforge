"""Tests for phase-1 indexing and inference addon config defaults."""
import pytest


def test_indexing_config_phase1_defaults():
    from addons.indexing.config import IndexingConfig
    cfg = IndexingConfig(embed_model="BAAI/bge-small-en-v1.5", vector_dim=384)
    assert cfg.vector_store == "qdrant"
    assert cfg.embedder_backend == "fastembed"
    assert cfg.chunk_size == 600
    assert cfg.chunk_overlap == 60


def test_inference_config_phase1_defaults():
    from addons.inference.config import InferenceConfig
    cfg = InferenceConfig(llm_model="meta-llama/Llama-3.2-3B-Instruct")
    assert cfg.gate_threshold == 0.75
    assert cfg.top_k == 5
    assert cfg.max_new_tokens == 256


def test_merged_config_has_all_phase1_keys():
    from core.config import KVForgeConfig
    cfg = KVForgeConfig(
        use_case_name="Phase1 Test",
        collection="test",
        version_file="v.json",
        addons=["indexing", "inference"],
        addon_config={
            "indexing": {
                "embed_model": "BAAI/bge-small-en-v1.5",
                "vector_dim": 384,
                "vector_store": "qdrant",
                "qdrant_host": "localhost",
                "qdrant_port": 6333,
            },
            "inference": {
                "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
                "top_k": 5,
                "gate_threshold": 0.75,
            },
        },
    )
    merged = cfg.get_merged_config("indexing", "inference")
    for key in ["collection", "embed_model", "vector_dim", "vector_store",
                "qdrant_host", "qdrant_port", "llm_model", "top_k", "gate_threshold"]:
        assert key in merged, f"Phase-1 merged config missing '{key}'"
