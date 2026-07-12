"""Validation tests for all built-in addon config schemas."""
import pytest
from pydantic import ValidationError


# ── IndexingConfig ────────────────────────────────────────────────────────

def test_indexing_config_defaults():
    from addons.indexing.config import IndexingConfig
    cfg = IndexingConfig(embed_model="BAAI/bge-small-en-v1.5", vector_dim=384)
    assert cfg.loader == "pdf"
    assert cfg.chunk_size == 600
    assert cfg.chunk_overlap == 60
    assert cfg.embed_batch == 64
    assert cfg.upsert_batch == 128
    assert cfg.vector_store == "qdrant"
    assert cfg.embedder_backend == "fastembed"
    assert cfg.qdrant_host == "localhost"
    assert cfg.qdrant_port == 6333


def test_indexing_config_weaviate():
    from addons.indexing.config import IndexingConfig
    cfg = IndexingConfig(
        embed_model="BAAI/bge-small-en-v1.5", vector_dim=384,
        vector_store="weaviate", weaviate_url="http://localhost:8080"
    )
    assert cfg.vector_store == "weaviate"
    assert cfg.weaviate_url == "http://localhost:8080"


def test_indexing_config_invalid_loader():
    from addons.indexing.config import IndexingConfig
    with pytest.raises(ValidationError):
        IndexingConfig(embed_model="x", vector_dim=384, loader="csv")


# ── InferenceConfig ───────────────────────────────────────────────────────

def test_inference_config_defaults():
    from addons.inference.config import InferenceConfig
    cfg = InferenceConfig(llm_model="meta-llama/Llama-3.2-3B-Instruct")
    assert cfg.top_k == 5
    assert cfg.quantization == "4bit"
    assert cfg.gate_threshold == 0.75
    assert cfg.parametric_eligibility_threshold == 0.85
    assert cfg.max_new_tokens == 256
    assert cfg.query_log_db == "query_log.db"


def test_inference_config_required_llm_model():
    from addons.inference.config import InferenceConfig
    with pytest.raises(ValidationError):
        InferenceConfig()


# ── TrainingConfig ────────────────────────────────────────────────────────

def test_training_config_defaults():
    from addons.training.config import TrainingConfig
    cfg = TrainingConfig(
        checkpoint_dir="ckpts/", replay_db="r.db"
    )
    assert cfg.lora_rank == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05
    assert cfg.lora_epochs == 3
    assert cfg.lora_lr == 0.0002
    assert cfg.prs_threshold == 0.50
    assert cfg.phase2_advance_threshold == 0.30
    assert cfg.phase3_advance_threshold == 0.55
    assert cfg.prs_weights == {"accuracy": 0.7, "calibration": 0.15, "consistency": 0.15}
    assert cfg.prs_advancement_threshold == 0.72
    assert cfg.prs_regression_threshold == 0.25
    assert cfg.faq_question_key == "question"
    assert cfg.faq_answer_key == "answer"


def test_training_config_required_fields():
    from addons.training.config import TrainingConfig
    with pytest.raises(ValidationError):
        TrainingConfig()  # checkpoint_dir and replay_db are required


# ── BackgroundConfig ──────────────────────────────────────────────────────

def test_background_config_defaults():
    from addons.background.config import BackgroundConfig
    cfg = BackgroundConfig()
    assert cfg.flush_seconds == 300
    assert cfg.flush_queries == 50


# ── SyncConfig ────────────────────────────────────────────────────────────

def test_sync_config_defaults():
    from addons.sync.config import SyncConfig
    cfg = SyncConfig()
    assert cfg.interval_minutes == 60
    assert cfg.hitl_mode == "auto"
    assert cfg.pii_detection_enabled is True
    assert cfg.sync_regression_mode == "pct"
    assert cfg.sync_regression_pct_threshold == 0.10


def test_sync_config_invalid_hitl_mode():
    from addons.sync.config import SyncConfig
    with pytest.raises(ValidationError):
        SyncConfig(hitl_mode="invalid")


# ── MonitoringConfig ──────────────────────────────────────────────────────

def test_monitoring_config_defaults():
    from addons.monitoring.config import MonitoringConfig
    cfg = MonitoringConfig()
    assert cfg.port == 8082
    assert cfg.analytics_db == ""


def test_monitoring_config_custom_port():
    from addons.monitoring.config import MonitoringConfig
    cfg = MonitoringConfig(port=8099)
    assert cfg.port == 8099


# ── MCPConfig ─────────────────────────────────────────────────────────────

def test_mcp_config_defaults():
    from addons.mcp.config import MCPConfig
    cfg = MCPConfig()
    assert cfg.port == 8765
    assert "query" in cfg.enabled_tools
    assert "status" in cfg.enabled_tools


# ── ModelScoutConfig ──────────────────────────────────────────────────────

def test_model_scout_config_defaults():
    from addons.model_scout.config import ModelScoutConfig
    cfg = ModelScoutConfig()
    assert cfg.initial_corpus_chunks == 200
    assert cfg.initial_faq_count == 20
    assert cfg.initial_lora_steps == 500
    assert cfg.max_lora_steps == 2000


# ── MultimodalConfig ──────────────────────────────────────────────────────

def test_multimodal_config_defaults():
    from addons.multimodal.config import MultimodalConfig
    cfg = MultimodalConfig()
    assert cfg.image_collection_suffix == "_images"
    assert cfg.multimodal_model == "llava-hf/llava-1.5-7b-hf"
    assert cfg.clip_model == "openai/clip-vit-base-patch32"
    assert cfg.image_kv_inference is False


# ── AnalyticsConfig ───────────────────────────────────────────────────────

def test_analytics_config_defaults():
    from addons.analytics.config import AnalyticsConfig
    cfg = AnalyticsConfig()
    assert cfg.cost_per_1k_tokens == 5.0
    assert cfg.tokens_per_ms_baseline == 0.8
    assert cfg.analytics_db == ""
