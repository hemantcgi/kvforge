"""
version.py — Atomic read/write of version.json.

Schema:
{
  "current_lora_version": 0,
  "checkpoint_path": null,
  "phase": 1,
  "prs_history": [],
  "known_good_queries": []
}
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

VERSION_FILE = Path(__file__).parent / "version.json"

DEFAULTS: dict[str, Any] = {
    "current_lora_version": 0,
    "checkpoint_path": None,
    "phase": 1,
    "prs_history": [],
    "known_good_queries": [],
}


def init(cfg: dict) -> None:
    """Set the version file path from config. Call once at startup before any other call."""
    global VERSION_FILE
    VERSION_FILE = Path(cfg.get("version_file", "version.json"))


def load() -> dict:
    if not VERSION_FILE.exists():
        return dict(DEFAULTS)
    with open(VERSION_FILE) as f:
        data = json.load(f)
    # back-fill any keys added in later versions
    for k, v in DEFAULTS.items():
        data.setdefault(k, v)
    return data


def save(data: dict) -> None:
    """Write atomically via temp file + rename."""
    tmp = VERSION_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, VERSION_FILE)


def get_lora_version() -> int:
    return load()["current_lora_version"]


def get_phase() -> int:
    return load()["phase"]


def increment_lora_version(checkpoint_path: str) -> int:
    data = load()
    data["current_lora_version"] += 1
    data["checkpoint_path"] = checkpoint_path
    save(data)
    return data["current_lora_version"]


def activate_phase_2() -> None:
    """Call from index_and_train.py after first successful LoRA round + SP3 deployed."""
    data = load()
    if data["phase"] < 2:
        data["phase"] = 2
        save(data)
        print("✅ Phase 2 activated — KV injection enabled")


def append_prs(round_num: int, prs: float) -> None:
    data = load()
    data["prs_history"].append({"round": round_num, "prs": round(prs, 4)})
    # check phase transition: PRS >= 0.80 for 2 consecutive rounds → phase 3
    history = data["prs_history"]
    if (len(history) >= 2
            and all(r["prs"] >= 0.80 for r in history[-2:])
            and data["phase"] < 3):
        data["phase"] = 3
        print("✅ Phase 3 activated — confidence gate now live")
    save(data)
