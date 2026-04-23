"""Tests for Pydantic DatasourceConfig."""
import pytest


def test_config_loads_with_required_fields_and_defaults():
    from core.config import DatasourceConfig
    cfg = DatasourceConfig(
        collection="test-col",
        embed_model="BAAI/bge-small-en-v1.5",
        vector_dim=384,
        llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        checkpoint_dir="checkpoints/",
        version_file="v.json",
        replay_db="r.db"
    )
    assert cfg.loader == "pdf"
    assert cfg.vector_store == "qdrant"
    assert cfg.gate_threshold == 0.75
    assert cfg.prs_threshold == 0.75
    assert cfg.faq_question_key == "question"
    assert cfg.embedder_backend == "fastembed"


def test_config_rejects_unknown_loader():
    from core.config import DatasourceConfig
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        DatasourceConfig(
            collection="x", embed_model="x", vector_dim=1,
            llm_model="x", checkpoint_dir="x", version_file="x",
            replay_db="x", loader="excel"
        )


def test_load_config_from_json_file(tmp_path):
    from core.config import load_config
    import json
    cfg_data = {
        "collection": "my-docs",
        "embed_model": "BAAI/bge-small-en-v1.5",
        "vector_dim": 384,
        "llm_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "checkpoint_dir": "ckpt/",
        "version_file": "v.json",
        "replay_db": "r.db"
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg_data))
    cfg = load_config(str(p))
    assert cfg.collection == "my-docs"
    assert isinstance(cfg.lora_target_modules, list)


def test_all_new_fields_have_correct_defaults():
    from core.config import DatasourceConfig
    cfg = DatasourceConfig(
        collection="t", embed_model="m", vector_dim=384,
        llm_model="m", checkpoint_dir="/t", version_file="/t/v.json", replay_db="/t/r.db"
    )
    # Dynamic PRS
    assert cfg.deployment_mode == "auto"
    assert cfg.prs_advancement_threshold == 0.72
    assert cfg.prs_signal_weights == {"faq": 0.4, "vdb": 0.4, "realtime": 0.2}
    assert cfg.query_log_db == "query_log.db"
    # Flywheel
    assert cfg.analytics_db == ""
    assert cfg.cost_per_1k_tokens == 5.0
    # VDB Expansion
    assert cfg.vector_store == "qdrant"
    assert cfg.pinecone_api_key == ""
    assert cfg.milvus_uri == "http://localhost:19530"
    # ModelScout
    assert cfg.scout_initial_lora_rank == 8
    # Multimodal
    assert cfg.image_collection_suffix == "_images"
    assert cfg.image_kv_inference is False


def test_config_model_dump_has_keys_used_by_existing_code():
    from core.config import DatasourceConfig
    cfg = DatasourceConfig(
        collection="x", embed_model="x", vector_dim=1,
        llm_model="x", checkpoint_dir="x", version_file="x",
        replay_db="x"
    )
    d = cfg.model_dump()
    for key in ["collection", "embed_model", "vector_dim", "llm_model",
                 "lora_rank", "gate_threshold", "prs_weights"]:
        assert key in d, f"Missing key in model_dump(): {key}"
