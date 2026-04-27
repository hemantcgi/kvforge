# tests/test_settings_manager.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch
import studio.settings_manager as sm


def test_get_all_returns_defaults_when_no_file(tmp_path):
    with patch.object(sm, "SETTINGS_FILE", tmp_path / "settings.json"):
        result = sm.get_all()
    assert result["curation_threshold"] == 50
    assert result["anthropic_api_key"] == ""
    assert result["default_cloud_provider"] == "anthropic"


def test_get_masked_masks_long_key(tmp_path):
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"anthropic_api_key": "sk-ant-api03-abcdefgh"}))
    with patch.object(sm, "SETTINGS_FILE", f):
        result = sm.get_masked()
    assert result["anthropic_api_key"] == "••••efgh"


def test_get_masked_leaves_empty_key_empty(tmp_path):
    with patch.object(sm, "SETTINGS_FILE", tmp_path / "settings.json"):
        assert sm.get_masked()["anthropic_api_key"] == ""


def test_save_writes_and_merges(tmp_path):
    f = tmp_path / "settings.json"
    with patch.object(sm, "SETTINGS_FILE", f):
        sm.save({"curation_threshold": 25})
        result = sm.get_all()
    assert result["curation_threshold"] == 25
    assert result["default_cloud_provider"] == "anthropic"  # default preserved


def test_save_rejects_invalid_anthropic_key(tmp_path):
    with patch.object(sm, "SETTINGS_FILE", tmp_path / "settings.json"):
        with pytest.raises(ValueError, match="anthropic_api_key"):
            sm.save({"anthropic_api_key": "not-a-valid-key"})


def test_save_accepts_valid_anthropic_key(tmp_path):
    f = tmp_path / "settings.json"
    with patch.object(sm, "SETTINGS_FILE", f):
        sm.save({"anthropic_api_key": "sk-ant-api03-xyz"})
        assert sm.get_all()["anthropic_api_key"] == "sk-ant-api03-xyz"


def test_get_setting_returns_single_value(tmp_path):
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"curation_threshold": 99}))
    with patch.object(sm, "SETTINGS_FILE", f):
        assert sm.get_setting("curation_threshold") == 99


def test_save_is_atomic(tmp_path):
    f = tmp_path / "settings.json"
    with patch.object(sm, "SETTINGS_FILE", f):
        sm.save({"curation_threshold": 10})
        assert not (tmp_path / "settings.tmp").exists()
        assert f.exists()
