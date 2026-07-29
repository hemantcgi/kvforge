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
    "kds_history": [],
    "known_good_queries": [],
    "clusters": {},
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


def _enforce_threshold_invariant(
    phase2_advance: float, phase3_advance: float, regression_threshold: float
) -> tuple[float, float, float]:
    """Clamp thresholds so ``phase3_advance >= phase2_advance >= regression_threshold``.

    A config that violates this (e.g. advance below regression) would let a single PRS value
    both advance and regress, producing phase flapping. Rather than flap, we raise the
    offending lower bound to its neighbor and warn. Returns the (possibly adjusted) triple.
    """
    adj_p2 = max(phase2_advance, regression_threshold)
    adj_p3 = max(phase3_advance, adj_p2)
    if (adj_p2, adj_p3) != (phase2_advance, phase3_advance):
        print(
            f"⚠️  PRS threshold invariant violated (phase3={phase3_advance}, "
            f"phase2={phase2_advance}, regression={regression_threshold}); clamped to "
            f"phase3={adj_p3}, phase2={adj_p2} to prevent phase flapping."
        )
    return adj_p2, adj_p3, regression_threshold


def decide_phase_transition(
    history: list,
    current_phase: int,
    *,
    phase2_advance: float,
    phase3_advance: float,
    regression_threshold: float,
    stability_window: int,
) -> int:
    """Pure phase-transition policy. Returns the new phase (may equal ``current_phase``).

    Advance: 1->2 when the latest PRS >= ``phase2_advance``; 2->3 when the last two rounds are
    both >= ``phase3_advance``. Regress one phase when the last ``stability_window`` rounds are
    all < ``regression_threshold``. Never advances and regresses in the same call; Phase 1 is
    the floor. Assumes the invariant ``phase3_advance >= phase2_advance >= regression_threshold``
    holds (call :func:`_enforce_threshold_invariant` first).
    """
    if not history:
        return current_phase
    phase = current_phase
    latest = history[-1]["prs"]

    if latest >= phase2_advance and phase < 2:
        phase = 2
    if (len(history) >= 2
            and all(r["prs"] >= phase3_advance for r in history[-2:])
            and phase < 3):
        phase = 3

    if phase == current_phase:  # only consider regression if we did not advance this call
        window = history[-stability_window:]
        if len(window) >= stability_window and all(
            r["prs"] < regression_threshold for r in window
        ):
            if phase == 3:
                phase = 2
            elif phase == 2:
                phase = 1
    return phase


def record_prs(round_num: int, prs: float) -> list:
    """Append a PRS score to history WITHOUT changing the phase.

    The record/decide split lets phase-transition policy be a pure, testable function
    (:func:`decide_phase_transition`) separate from I/O.
    """
    data = load()
    data["prs_history"].append({"round": round_num, "prs": round(prs, 4)})
    save(data)
    return data["prs_history"]


def append_prs(
    round_num: int,
    prs: float,
    *,
    regression_threshold: float = 0.25,
    stability_window: int = 3,
    phase2_advance_threshold: float = 0.30,
    phase3_advance_threshold: float = 0.55,
) -> None:
    """Record a PRS score and apply the phase-transition policy.

    Orchestrates I/O around the pure :func:`decide_phase_transition` policy. Thresholds are
    provisional/data-derived — see docs/superpowers/specs/2026-07-12-prs-gate-rework-design.md.

    Args:
        round_num: LoRA training round number (record-keeping only).
        prs: Parametric Readiness Score in [0, 1].
        regression_threshold: consecutive rounds below this trigger a downgrade. Default 0.25.
        stability_window: consecutive rounds required for a downgrade. Default 3.
        phase2_advance_threshold: PRS floor to advance 1->2 (KV injection + selective
            parametric). Default 0.30.
        phase3_advance_threshold: PRS floor (two consecutive rounds) to advance 2->3
            (corpus-wide parametric trust). Default 0.55.
    """
    p2, p3, reg = _enforce_threshold_invariant(
        phase2_advance_threshold, phase3_advance_threshold, regression_threshold
    )
    data = load()
    data["prs_history"].append({"round": round_num, "prs": round(prs, 4)})
    history = data["prs_history"]
    before = data["phase"]
    new_phase = decide_phase_transition(
        history, before,
        phase2_advance=p2, phase3_advance=p3,
        regression_threshold=reg, stability_window=stability_window,
    )
    if new_phase != before:
        data["phase"] = new_phase
        if new_phase > before:
            label = {
                2: "KV injection + selective parametric enabled",
                3: "corpus-wide confidence gate now live",
            }.get(new_phase, "")
            print(f"✅ Phase {new_phase} activated — {label}")
        else:
            print(
                f"⚠️  Phase regression: {before} → {new_phase} "
                f"(PRS below {reg} for {stability_window} consecutive rounds)"
            )
    save(data)


def append_kds(round_num: int, mean_kds: float, measured_chunks: int) -> None:
    """Append a corpus-level KDS record to version.json.

    Args:
        round_num: LoRA training round number (record-keeping only).
        mean_kds: Mean KDS score over chunks measured in this round, in [0, 1].
        measured_chunks: Number of chunks that contributed a KDS value this round.
    """
    import time
    data = load()
    data["kds_history"].append({
        "round": round_num,
        "mean_kds": round(mean_kds, 4),
        "measured_chunks": measured_chunks,
        "timestamp": int(time.time()),
    })
    save(data)


def append_fkds(round_num: int, mean_fkds: float, measured_chunks: int) -> None:
    """Append a corpus-level factual KDS (fKDS) record to version.json.

    Args:
        round_num: LoRA training round number (record-keeping only).
        mean_fkds: Mean fKDS score over chunks measured in this round, in [0, 1].
        measured_chunks: Number of chunks that contributed an fKDS value this round.
    """
    import time
    data = load()
    data.setdefault("fkds_history", []).append({
        "round": round_num,
        "mean_fkds": round(mean_fkds, 4),
        "measured_chunks": measured_chunks,
        "timestamp": int(time.time()),
    })
    save(data)


def get_cluster_state(cluster_id: str) -> dict:
    """Return per-cluster PRS state dict, or empty dict if cluster not yet tracked.

    Args:
        cluster_id: Cluster identifier string.

    Returns:
        Dict with cluster-specific PRS state, or ``{}`` if not found.
    """
    return load().get("clusters", {}).get(str(cluster_id), {})


def save_cluster_state(cluster_id: str, state: dict) -> None:
    """Atomically update a single cluster's state in version.json.

    Args:
        cluster_id: Cluster identifier string.
        state: Dict with cluster-specific PRS state.
    """
    data = load()
    data.setdefault("clusters", {})[str(cluster_id)] = state
    save(data)


def get_global_phase() -> int:
    """Return minimum phase across all clusters (conservative).

    Falls back to the top-level ``'phase'`` key when no clusters exist.

    Returns:
        Minimum phase integer (1, 2, or 3).
    """
    data = load()
    clusters = data.get("clusters", {})
    if not clusters:
        return data.get("phase", 1)
    return min(c.get("phase", 1) for c in clusters.values())
