import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_ver_file(tmp_path, phase, prs_history):
    """Write a version.json and point core.version at it."""
    import core.version as ver
    vfile = tmp_path / "version.json"
    vfile.write_text(json.dumps({
        "current_lora_version": len(prs_history),
        "checkpoint_path": None,
        "phase": phase,
        "prs_history": [{"round": i+1, "prs": p} for i, p in enumerate(prs_history)],
        "known_good_queries": [],
        "clusters": {},
    }))
    ver.VERSION_FILE = vfile
    return ver


def test_phase3_downgrades_to_2_after_stability_window_bad_rounds(tmp_path):
    ver = _make_ver_file(tmp_path, phase=3, prs_history=[0.80, 0.55, 0.54])
    ver.append_prs(4, 0.53, regression_threshold=0.60, stability_window=3)
    data = ver.load()
    assert data["phase"] == 2


def test_phase2_downgrades_to_1_after_stability_window_bad_rounds(tmp_path):
    ver = _make_ver_file(tmp_path, phase=2, prs_history=[0.80, 0.55, 0.54])
    ver.append_prs(4, 0.53, regression_threshold=0.60, stability_window=3, advance_threshold=0.75)
    data = ver.load()
    assert data["phase"] == 1


def test_no_downgrade_if_window_not_full(tmp_path):
    """Only 2 bad rounds with window=3 — should NOT downgrade."""
    ver = _make_ver_file(tmp_path, phase=3, prs_history=[0.80, 0.55])
    ver.append_prs(3, 0.54, regression_threshold=0.60, stability_window=3)
    data = ver.load()
    assert data["phase"] == 3


def test_no_downgrade_in_coast_zone(tmp_path):
    """PRS between regression_threshold (0.60) and advance_threshold (0.75) — stay put."""
    ver = _make_ver_file(tmp_path, phase=3, prs_history=[0.80, 0.65, 0.66])
    ver.append_prs(4, 0.67, regression_threshold=0.60, stability_window=3)
    data = ver.load()
    assert data["phase"] == 3


def test_no_double_downgrade_per_call(tmp_path):
    """Phase 3 with 3 bad rounds drops to 2, not 1, in a single call."""
    ver = _make_ver_file(tmp_path, phase=3, prs_history=[0.80, 0.55, 0.54])
    ver.append_prs(4, 0.53, regression_threshold=0.60, stability_window=3)
    data = ver.load()
    assert data["phase"] == 2  # not 1


def test_advance_still_works_with_new_params(tmp_path):
    """Existing advance logic must still fire with the new keyword args present."""
    ver = _make_ver_file(tmp_path, phase=2, prs_history=[0.80])
    ver.append_prs(2, 0.80, regression_threshold=0.60, stability_window=3)
    data = ver.load()
    assert data["phase"] == 3


def test_existing_callers_unaffected_no_new_args(tmp_path):
    """append_prs called with just (round_num, prs) must not crash."""
    ver = _make_ver_file(tmp_path, phase=1, prs_history=[])
    ver.append_prs(1, 0.30)  # no extra kwargs — must not raise; 0.30 stays below any threshold
    data = ver.load()
    assert data["phase"] == 1


def test_phase1_does_not_regress_below_1(tmp_path):
    """Phase 1 is the floor — bad rounds must not push it to 0."""
    ver = _make_ver_file(tmp_path, phase=1, prs_history=[0.40, 0.41])
    ver.append_prs(3, 0.42, regression_threshold=0.60, stability_window=3)
    data = ver.load()
    assert data["phase"] == 1


def test_advance_threshold_is_configurable(tmp_path):
    """A custom advance_threshold must gate advancement, not the old hardcoded 0.75."""
    ver = _make_ver_file(tmp_path, phase=1, prs_history=[])
    # 0.55 would NOT advance under the old hardcoded 0.75, but must advance
    # when advance_threshold is explicitly lowered to 0.50.
    ver.append_prs(1, 0.55, advance_threshold=0.50)
    data = ver.load()
    assert data["phase"] == 2


def test_advance_threshold_default_is_050(tmp_path):
    """With no advance_threshold passed, the new default (0.50) applies."""
    ver = _make_ver_file(tmp_path, phase=1, prs_history=[])
    ver.append_prs(1, 0.55)
    data = ver.load()
    assert data["phase"] == 2


def test_no_regress_when_advance_fires_same_call(tmp_path):
    """When advance fires, regression must not also fire in the same call.

    With a pathological config (regression_threshold > advance threshold),
    a PRS of 0.76 would advance phase 1→2 and then the regression window
    check could immediately revert it. Guard ensures this never happens.
    """
    ver = _make_ver_file(tmp_path, phase=1, prs_history=[0.80, 0.80])
    # regression_threshold=0.90 means 0.76 is below it, but advance fires first
    ver.append_prs(3, 0.76, regression_threshold=0.90, stability_window=3)
    data = ver.load()
    # advance should have fired (prs >= 0.75 and two consecutive rounds >= 0.75)
    assert data["phase"] == 3  # not regressed back
