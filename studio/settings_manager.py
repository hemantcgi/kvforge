# studio/settings_manager.py
import json
import os
from pathlib import Path

SETTINGS_FILE = Path.home() / ".kvforge" / "settings.json"

DEFAULTS: dict = {
    "anthropic_api_key": "",
    "openai_api_key": "",
    "gemini_api_key": "",
    "huggingface_token": "",
    "curation_threshold": 50,
    "default_cloud_provider": "anthropic",
    "default_cloud_model": "claude-haiku-4-5-20251001",
}

_SECRET_KEYS = {"anthropic_api_key", "openai_api_key", "gemini_api_key", "huggingface_token"}

_KEY_VALIDATORS: dict = {
    "anthropic_api_key": lambda v: v.startswith("sk-ant-"),
    "openai_api_key": lambda v: v.startswith("sk-"),
    "gemini_api_key": lambda v: v.startswith("AIza"),
}


def _load() -> dict:
    if not SETTINGS_FILE.exists():
        return dict(DEFAULTS)
    try:
        with open(SETTINGS_FILE) as f:
            data = json.load(f)
        return {**DEFAULTS, **data}
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULTS)


def get_all() -> dict:
    return _load()


def get_masked() -> dict:
    data = _load()
    out = {}
    for k, v in data.items():
        if k in _SECRET_KEYS and isinstance(v, str) and v:
            out[k] = "••••" + v[-4:] if len(v) >= 4 else "••••"
        else:
            out[k] = v
    return out


def get_setting(key: str):
    return _load().get(key, DEFAULTS.get(key))


def save(updates: dict) -> None:
    for key, value in updates.items():
        if key in _KEY_VALIDATORS and isinstance(value, str) and value:
            if not _KEY_VALIDATORS[key](value):
                raise ValueError(f"Invalid format for {key}")
    current = _load()
    current.update(updates)
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(current, f, indent=2)
    os.replace(tmp, SETTINGS_FILE)
