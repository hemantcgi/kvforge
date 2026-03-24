"""Tests for KV shape auto-discovery and LoRA target detection."""
import pytest
from unittest.mock import MagicMock


def _mock_model_config(num_hidden_layers=28, num_key_value_heads=8,
                        hidden_size=4096, num_attention_heads=32,
                        head_dim=None):
    cfg = MagicMock()
    cfg.num_hidden_layers = num_hidden_layers
    cfg.num_key_value_heads = num_key_value_heads
    cfg.hidden_size = hidden_size
    cfg.num_attention_heads = num_attention_heads
    if head_dim is not None:
        cfg.head_dim = head_dim
    else:
        del cfg.head_dim
    return cfg


def test_kv_shape_auto_discovery_with_head_dim():
    from model_loader import _kv_shape_from_hf_config
    hf_cfg = _mock_model_config(num_hidden_layers=28, num_key_value_heads=8, head_dim=128)
    layers, heads, dim = _kv_shape_from_hf_config(hf_cfg)
    assert layers == 28
    assert heads == 8
    assert dim == 128


def test_kv_shape_auto_discovery_without_head_dim():
    from model_loader import _kv_shape_from_hf_config
    hf_cfg = _mock_model_config(
        num_hidden_layers=32, num_key_value_heads=8,
        hidden_size=4096, num_attention_heads=32, head_dim=None
    )
    layers, heads, dim = _kv_shape_from_hf_config(hf_cfg)
    assert layers == 32
    assert heads == 8
    assert dim == 128  # 4096 // 32


def test_get_kv_shape_uses_registry_when_available():
    from model_loader import get_kv_shape
    cfg = {
        "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
        "model_library": {
            "meta-llama/Llama-3.2-3B-Instruct": {
                "kv_num_layers": 28, "kv_num_heads": 8, "kv_head_dim": 128
            }
        }
    }
    layers, heads, dim = get_kv_shape(cfg)
    assert (layers, heads, dim) == (28, 8, 128)


def test_get_kv_shape_falls_back_to_auto_when_not_in_registry():
    from model_loader import get_kv_shape
    import model_loader
    mock_model = MagicMock()
    mock_model.config.num_hidden_layers = 24
    mock_model.config.num_key_value_heads = 4
    mock_model.config.head_dim = 64
    original = model_loader._model
    model_loader._model = mock_model
    try:
        cfg = {"llm_model": "some/new-model", "model_library": {}}
        layers, heads, dim = get_kv_shape(cfg)
        assert (layers, heads, dim) == (24, 4, 64)
    finally:
        model_loader._model = original


def test_detect_lora_targets_finds_standard_projections():
    from model_loader import detect_lora_targets
    mock_model = MagicMock()
    mock_model.named_modules.return_value = [
        ("model.layers.0.self_attn.q_proj", MagicMock()),
        ("model.layers.0.self_attn.k_proj", MagicMock()),
        ("model.layers.0.self_attn.v_proj", MagicMock()),
        ("model.layers.0.mlp.gate_proj",   MagicMock()),
    ]
    targets = detect_lora_targets(mock_model, ["q_proj", "k_proj", "v_proj"])
    assert set(targets) == {"q_proj", "k_proj", "v_proj"}


def test_detect_lora_targets_warns_when_none_match():
    from model_loader import detect_lora_targets
    mock_model = MagicMock()
    mock_model.named_modules.return_value = [
        ("model.layers.0.self_attn.query_key_value", MagicMock()),
    ]
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        targets = detect_lora_targets(mock_model, ["q_proj", "k_proj", "v_proj"])
        assert len(w) >= 1
        warning_messages = " ".join(str(warning.message) for warning in w)
        assert "lora_target_modules" in warning_messages or "query_key_value" in warning_messages or "not found" in warning_messages.lower()
