"""Adapter acceptance gate and rollback — Sprint 3.

Provides:

* ``accept_adapter()`` — check whether a candidate adapter is safe to deploy.
* ``rollback()`` — restore the previously deployed adapter and version state.
* ``stage_candidate()`` — record a candidate adapter's eval results before acceptance.

All state is persisted in ``version.json`` under the ``_deployment`` key.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def _load_state(cfg: dict) -> dict:
    """Return the deployment state dict from version.json."""
    import core.version as ver
    data = ver.load()
    return data.get("_deployment", {})


def _save_state(cfg: dict, state: dict) -> None:
    """Write the deployment state dict to version.json atomically."""
    import core.version as ver
    data = ver.load()
    data["_deployment"] = state
    ver.save(data)


# ── staging a candidate ──────────────────────────────────────────────────


def stage_candidate(
    cfg: dict,
    candidate_path: str,
    fkds_on_heldout: float,
    fkds_sem: float | None = None,
    judge_noise: float = 0.05,
    latency_mean: float | None = None,
    notes: dict | None = None,
) -> None:
    """Record a candidate adapter's eval results.

    Call this AFTER running eval but BEFORE calling ``accept_adapter``.
    """
    state = _load_state(cfg)
    state["candidate"] = {
        "path": str(candidate_path),
        "fkds": round(fkds_on_heldout, 4),
        "fkds_sem": round(fkds_sem, 4) if fkds_sem is not None else None,
        "judge_noise": round(judge_noise, 4),
        "latency_mean": round(latency_mean, 4) if latency_mean is not None else None,
        "notes": notes or {},
    }
    _save_state(cfg, state)


# ── acceptance ───────────────────────────────────────────────────────────


def accept_adapter(
    cfg: dict,
    *,
    min_delta_noise_multiplier: float = 2.0,
    max_non_targeted_degradation: float = 0.05,
) -> tuple[bool, dict]:
    """Check whether the staged candidate should be deployed.

    Returns:
        ``(accepted, report)`` where ``report`` explains the decision.

    The three-part rule from the v3 plan:

    1. Aggregate held‑out fKDS must improve over the deployed adapter.
    2. (Not yet implemented) Non-targeted chunks must not degrade more than
       ``max_non_targeted_degradation``.
    3. The absolute delta must exceed ``min_delta_noise_multiplier × judge_noise``.

    If accepted, the deployed adapter is replaced and the previous one is saved
    for rollback.
    """
    state = _load_state(cfg)
    candidate = state.get("candidate")
    deployed = state.get("deployed")

    if candidate is None:
        return False, {"error": "No candidate staged — run stage_candidate first."}

    cand_fkds = candidate["fkds"]
    judge_noise = candidate.get("judge_noise", 0.05)

    if deployed is None:
        # First-ever deployment — accept unconditionally.
        _commit_adapter(cfg, candidate["path"], cand_fkds)
        return True, {
            "accepted": True,
            "reason": "first-deployment",
            "candidate_fkds": cand_fkds,
            "deployed_fkds": None,
            "abs_delta": None,
            "threshold": None,
        }

    dep_fkds = deployed["fkds"]
    abs_delta = cand_fkds - dep_fkds
    threshold = min_delta_noise_multiplier * judge_noise
    reason = ""

    if abs_delta <= 0:
        reason = f"candidate fKDS ({cand_fkds}) ≤ deployed ({dep_fkds})"
    elif abs_delta <= threshold:
        reason = (
            f"delta ({abs_delta:+.4f}) ≤ {min_delta_noise_multiplier}× "
            f"judge_noise ({threshold:.4f}) — within noise"
        )
    else:
        _commit_adapter(cfg, candidate["path"], cand_fkds)
        return True, {
            "accepted": True,
            "reason": "fKDS-improved",
            "candidate_fkds": cand_fkds,
            "deployed_fkds": dep_fkds,
            "abs_delta": round(abs_delta, 4),
            "threshold": round(threshold, 4),
        }

    return False, {
        "accepted": False,
        "reason": reason,
        "candidate_fkds": cand_fkds,
        "deployed_fkds": dep_fkds,
        "abs_delta": round(abs_delta, 4),
        "threshold": round(threshold, 4),
    }


def _commit_adapter(cfg: dict, path: str, fkds: float) -> None:
    """Persist the acceptance: move deployed→previous, set deployed to candidate."""
    state = _load_state(cfg)
    previous = state.pop("candidate", None)
    if "deployed" in state:
        state["previous"] = state["deployed"]
    state["deployed"] = {
        "path": str(path),
        "fkds": round(fkds, 4),
    }
    _save_state(cfg, state)


# ── rollback ─────────────────────────────────────────────────────────────


def rollback(cfg: dict) -> bool:
    """Restore the previous adapter if one exists.

    Updates ``version.json`` so the ``checkpoint_path`` points to the previous
    adapter. Moves the reverted adapter to ``previous``.

    Returns:
        True if rollback was performed, False if no previous adapter exists.
    """
    import core.version as ver

    state = _load_state(cfg)
    previous = state.get("previous")
    if previous is None:
        return False

    reverted = state.get("deployed", {})
    state["deployed"] = previous
    state["previous"] = reverted

    # Also update the version.json checkpoint_path so the system knows.
    data = ver.load()
    data["checkpoint_path"] = previous["path"]
    data["_deployment"] = state
    ver.save(data)

    print(
        f"⏪ Rollback: deployed → {previous['path']} "
        f"(fkds={previous['fkds']}), reverted {reverted.get('path', '?')} "
        f"(fkds={reverted.get('fkds', '?'):.4f})"
    )
    return True


def get_deployed_path(cfg: dict) -> str | None:
    """Return the currently deployed adapter path, or None."""
    state = _load_state(cfg)
    d = state.get("deployed")
    return d["path"] if d else None


def get_deployed_fkds(cfg: dict) -> float | None:
    """Return the currently deployed adapter's fKDS, or None."""
    state = _load_state(cfg)
    d = state.get("deployed")
    return d["fkds"] if d else None


def deployment_status(cfg: dict) -> dict:
    """Return a human-readable deployment status dict."""
    state = _load_state(cfg)
    return {
        "deployed": state.get("deployed"),
        "previous": state.get("previous"),
        "candidate": state.get("candidate"),
        "can_rollback": state.get("previous") is not None,
    }
