import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _cfg(tmp_path, overrides: dict = {}) -> "DatasourceConfig":
    from core.config import load_config
    base = {
        "collection": "test_col",
        "embed_model": "BAAI/bge-small-en-v1.5",
        "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
        "vector_store": "qdrant",
        "vector_dim": 384,
        "checkpoint_dir": "/tmp/checkpoints",
        "version_file": "/tmp/version.json",
        "replay_db": "/tmp/replay.db",
    }
    base.update(overrides)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(base))
    return load_config(str(p))


def test_tenant_id_defaults_to_default(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.tenant_id == "default"


def test_sync_interval_minutes_default(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.sync_interval_minutes == 60


def test_sync_interval_configurable(tmp_path):
    cfg = _cfg(tmp_path, {"sync_interval_minutes": 15})
    assert cfg.sync_interval_minutes == 15


def test_hitl_mode_default_auto(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.hitl_mode == "auto"


def test_hitl_mode_accepts_blocking(tmp_path):
    cfg = _cfg(tmp_path, {"hitl_mode": "blocking"})
    assert cfg.hitl_mode == "blocking"


def test_hitl_mode_accepts_non_blocking(tmp_path):
    cfg = _cfg(tmp_path, {"hitl_mode": "non-blocking"})
    assert cfg.hitl_mode == "non-blocking"


def test_pii_detection_enabled_by_default(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.pii_detection_enabled is True


def test_allowed_pii_categories_default_empty(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.allowed_pii_categories == []


def test_local_mirror_path_default_empty(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.local_mirror_path == ""


def test_pii_rejection_threshold_default(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.pii_rejection_threshold == 3


def test_hitl_sensitivity_default_normal(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.hitl_sensitivity == "normal"


def test_hitl_mode_rejects_invalid(tmp_path):
    import pytest
    try:
        from pydantic import ValidationError
    except ImportError:
        from pydantic.v1 import ValidationError
    with pytest.raises((ValidationError, ValueError)):
        _cfg(tmp_path, {"hitl_mode": "garbage"})
