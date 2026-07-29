"""Tests for Sprint 3 adapter acceptance and rollback."""

import json
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import adapter_acceptance as aa


@pytest.fixture
def fake_version_file(tmp_path):
    """Create a temporary version.json and monkeypatch version.VERSION_FILE."""
    vf = tmp_path / "version.json"
    vf.write_text(json.dumps({
        "current_lora_version": 1,
        "checkpoint_path": "checkpoints/v1/",
        "phase": 1,
        "prs_history": [],
    }))
    with patch("core.version.VERSION_FILE", vf):
        yield vf


def test_first_deployment_accepted(fake_version_file):
    cfg = {"version_file": str(fake_version_file)}
    state = aa._load_state(cfg)
    assert state == {}

    aa.stage_candidate(cfg, "checkpoints/v1/", fkds_on_heldout=0.06)
    accepted, report = aa.accept_adapter(cfg)
    assert accepted
    assert report["reason"] == "first-deployment"
    assert aa.get_deployed_path(cfg) == "checkpoints/v1/"


def test_better_candidate_accepted(fake_version_file):
    cfg = {"version_file": str(fake_version_file)}

    # First deployment.
    aa.stage_candidate(cfg, "checkpoints/v1/", fkds_on_heldout=0.06, judge_noise=0.005)
    aa.accept_adapter(cfg)

    # Better candidate — delta 0.02 > 2×0.005=0.01 threshold.
    aa.stage_candidate(cfg, "checkpoints/v2/", fkds_on_heldout=0.08, judge_noise=0.005)
    accepted, report = aa.accept_adapter(cfg)
    assert accepted
    assert report["reason"] == "fKDS-improved"
    assert report["abs_delta"] == pytest.approx(0.02)
    assert aa.get_deployed_path(cfg) == "checkpoints/v2/"


def test_worse_candidate_rejected(fake_version_file):
    cfg = {"version_file": str(fake_version_file)}

    aa.stage_candidate(cfg, "checkpoints/v1/", fkds_on_heldout=0.08)
    aa.accept_adapter(cfg)

    aa.stage_candidate(cfg, "checkpoints/v2/", fkds_on_heldout=0.07)
    accepted, report = aa.accept_adapter(cfg)
    assert not accepted
    assert "candidate fKDS" in report["reason"]
    # Deployed should still be v1.
    assert aa.get_deployed_path(cfg) == "checkpoints/v1/"


def test_within_noise_rejected(fake_version_file):
    cfg = {"version_file": str(fake_version_file)}

    aa.stage_candidate(cfg, "checkpoints/v1/", fkds_on_heldout=0.0600, judge_noise=0.05)
    aa.accept_adapter(cfg)

    # Delta is only 0.0011, below 2×0.05=0.10 threshold.
    aa.stage_candidate(cfg, "checkpoints/v2/", fkds_on_heldout=0.0611, judge_noise=0.05)
    accepted, report = aa.accept_adapter(cfg)
    assert not accepted
    assert "within noise" in report["reason"]


def test_no_candidate_errors(fake_version_file):
    cfg = {"version_file": str(fake_version_file)}
    accepted, report = aa.accept_adapter(cfg)
    assert not accepted
    assert "error" in report


def test_rollback_restores_previous(fake_version_file):
    cfg = {"version_file": str(fake_version_file)}

    # Deploy v1.
    aa.stage_candidate(cfg, "checkpoints/v1/", fkds_on_heldout=0.06, judge_noise=0.005)
    aa.accept_adapter(cfg)

    # Deploy v2 — delta 0.02 > 2×0.005=0.01 threshold.
    aa.stage_candidate(cfg, "checkpoints/v2/", fkds_on_heldout=0.08, judge_noise=0.005)
    aa.accept_adapter(cfg)

    # Rollback — toggles deployed and previous.
    ok = aa.rollback(cfg)
    assert ok
    assert aa.get_deployed_path(cfg) == "checkpoints/v1/"

    # Rollback again — toggles back.
    ok = aa.rollback(cfg)
    assert ok
    assert aa.get_deployed_path(cfg) == "checkpoints/v2/"

    # Always toggleable while previous exists.
    status = aa.deployment_status(cfg)
    assert status["can_rollback"] is True


def test_rollback_no_previous(fake_version_file):
    cfg = {"version_file": str(fake_version_file)}
    ok = aa.rollback(cfg)
    assert not ok


def test_deployment_status(fake_version_file):
    cfg = {"version_file": str(fake_version_file)}

    status = aa.deployment_status(cfg)
    assert status["can_rollback"] is False

    aa.stage_candidate(cfg, "checkpoints/v1/", fkds_on_heldout=0.06, judge_noise=0.005)
    aa.accept_adapter(cfg)
    status = aa.deployment_status(cfg)
    assert status["deployed"] is not None
    assert status["can_rollback"] is False

    aa.stage_candidate(cfg, "checkpoints/v2/", fkds_on_heldout=0.08, judge_noise=0.005)
    aa.accept_adapter(cfg)
    status = aa.deployment_status(cfg)
    assert status["can_rollback"] is True
    assert status["previous"]["path"] == "checkpoints/v1/"


def test_accept_with_custom_noise_multiplier(fake_version_file):
    cfg = {"version_file": str(fake_version_file)}

    aa.stage_candidate(cfg, "checkpoints/v1/", fkds_on_heldout=0.06, judge_noise=0.01)
    aa.accept_adapter(cfg)

    # Delta 0.002, threshold 2×0.01=0.02 — within noise, reject.
    aa.stage_candidate(cfg, "checkpoints/v2/", fkds_on_heldout=0.062, judge_noise=0.01)
    accepted, _ = aa.accept_adapter(cfg)
    assert not accepted

    # With multiplier=1.0, threshold=0.01, delta=0.002 → reject.
    accepted, _ = aa.accept_adapter(cfg, min_delta_noise_multiplier=1.0)
    assert not accepted

    # Delta 0.03 with noise 0.01, threshold=0.02 → accept.
    aa.stage_candidate(cfg, "checkpoints/v3/", fkds_on_heldout=0.09, judge_noise=0.01)
    accepted, report = aa.accept_adapter(cfg)
    assert accepted
