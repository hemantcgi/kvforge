"""Atomic read/write helpers for the KVForge version state file.

The version file (``version.json`` by default) tracks:

* ``current_lora_version`` — monotonically incrementing LoRA round counter.
* ``checkpoint_path`` — path to the most recently saved LoRA adapter directory.
* ``phase`` — active inference phase (1 = retrieval only, 2 = KV injection,
  3 = confidence gate).
* ``prs_history`` — list of ``{"round": int, "prs": float}`` records.
* ``known_good_queries`` — pre-computed embeddings of queries that the model
  answered accurately (used by the confidence gate).

Writes are atomic via a temp-file rename so that the file is never partially
written.  All public functions re-read the file on every call to remain correct
when multiple processes or threads modify the state.
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
    """Override the module-level version file path from config.

    Must be called once at startup before any other function in this module.

    Args:
        cfg: Datasource configuration dictionary.  Uses
            ``cfg['version_file']`` (default ``'version.json'``).
    """
    global VERSION_FILE
    VERSION_FILE = Path(cfg.get("version_file", "version.json"))


def load() -> dict:
    """Read the version file and return its contents as a dict.

    If the file does not exist the ``DEFAULTS`` dict is returned.  Any keys
    that exist in ``DEFAULTS`` but are absent from the file are back-filled,
    providing forwards compatibility when new keys are added.

    Returns:
        A dict with keys ``current_lora_version``, ``checkpoint_path``,
        ``phase``, ``prs_history``, and ``known_good_queries``.
    """
    if not VERSION_FILE.exists():
        return dict(DEFAULTS)
    with open(VERSION_FILE) as f:
        data = json.load(f)
    # back-fill any keys added in later versions
    for k, v in DEFAULTS.items():
        data.setdefault(k, v)
    return data


def save(data: dict) -> None:
    """Write *data* to the version file atomically via a temp-file rename.

    Args:
        data: Dict to serialise as JSON.
    """
    tmp = VERSION_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, VERSION_FILE)


def get_lora_version() -> int:
    """Return the current LoRA version counter from the version file.

    Returns:
        Integer version number (0 means no LoRA has been trained yet).
    """
    return load()["current_lora_version"]


def get_phase() -> int:
    """Return the current inference phase (1, 2, or 3) from the version file.

    Returns:
        Active phase integer.
    """
    return load()["phase"]


def increment_lora_version(checkpoint_path: str) -> int:
    """Increment the LoRA version counter and record the new checkpoint path.

    Args:
        checkpoint_path: Directory path where the new LoRA adapter was saved.

    Returns:
        The new (incremented) version number.
    """
    data = load()
    data["current_lora_version"] += 1
    data["checkpoint_path"] = checkpoint_path
    save(data)
    return data["current_lora_version"]


def activate_phase_2() -> None:
    """Unconditionally advance the phase to 2 (KV injection enabled).

    Intended for manual or script-driven activation after the first successful
    LoRA round.  Has no effect if the phase is already 2 or higher.
    """
    data = load()
    if data["phase"] < 2:
        data["phase"] = 2
        save(data)
        print("✅ Phase 2 activated — KV injection enabled")


def append_prs(round_num: int, prs: float) -> None:
    """Record a PRS score and automatically advance the phase if thresholds are met.

    Phase transitions:

    * Phase 2: triggered when ``prs >= 0.75`` for the first time.
    * Phase 3: triggered when ``prs >= 0.75`` for two consecutive rounds.

    Args:
        round_num: LoRA training round number (used for record-keeping only).
        prs: Parametric Readiness Score in [0, 1].
    """
    data = load()
    data["prs_history"].append({"round": round_num, "prs": round(prs, 4)})
    history = data["prs_history"]
    # Phase 2: PRS >= 0.75 for at least one round
    if prs >= 0.75 and data["phase"] < 2:
        data["phase"] = 2
        print("✅ Phase 2 activated — KV injection enabled")
    # Phase 3: PRS >= 0.75 for 2 consecutive rounds
    if (len(history) >= 2
            and all(r["prs"] >= 0.75 for r in history[-2:])
            and data["phase"] < 3):
        data["phase"] = 3
        print("✅ Phase 3 activated — confidence gate now live")
    save(data)
