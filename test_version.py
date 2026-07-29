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


def _hist(*prs_vals):
    return [{"round": i + 1, "prs": p} for i, p in enumerate(prs_vals)]


def test_decide_advances_1_to_2_at_phase2_threshold():
    from core.version import decide_phase_transition
    new = decide_phase_transition(_hist(0.35), 1, phase2_advance=0.30,
                                  phase3_advance=0.55, regression_threshold=0.25,
                                  stability_window=3)
    assert new == 2


def test_decide_does_not_advance_1_to_2_below_phase2_threshold():
    from core.version import decide_phase_transition
    new = decide_phase_transition(_hist(0.28), 1, phase2_advance=0.30,
                                  phase3_advance=0.55, regression_threshold=0.25,
                                  stability_window=3)
    assert new == 1


def test_decide_advances_2_to_3_only_on_two_consecutive_high_rounds():
    from core.version import decide_phase_transition
    one = decide_phase_transition(_hist(0.40, 0.60), 2, phase2_advance=0.30,
                                  phase3_advance=0.55, regression_threshold=0.25,
                                  stability_window=3)
    assert one == 2  # only the last round clears phase3_advance
    two = decide_phase_transition(_hist(0.60, 0.60), 2, phase2_advance=0.30,
                                  phase3_advance=0.55, regression_threshold=0.25,
                                  stability_window=3)
    assert two == 3


def test_decide_regresses_after_stability_window_below_regression():
    from core.version import decide_phase_transition
    new = decide_phase_transition(_hist(0.60, 0.20, 0.20, 0.20), 3,
                                  phase2_advance=0.30, phase3_advance=0.55,
                                  regression_threshold=0.25, stability_window=3)
    assert new == 2


def test_decide_never_double_regresses():
    from core.version import decide_phase_transition
    new = decide_phase_transition(_hist(0.60, 0.20, 0.20, 0.20), 3,
                                  phase2_advance=0.30, phase3_advance=0.55,
                                  regression_threshold=0.25, stability_window=3)
    assert new == 2  # 3 -> 2, not 3 -> 1


def test_decide_phase1_is_floor():
    from core.version import decide_phase_transition
    new = decide_phase_transition(_hist(0.10, 0.10, 0.10), 1,
                                  phase2_advance=0.30, phase3_advance=0.55,
                                  regression_threshold=0.25, stability_window=3)
    assert new == 1


def test_invariant_clamps_flapping_config():
    from core.version import _enforce_threshold_invariant
    # Old flapping pair: advance 0.50 < regression 0.60.
    p2, p3, reg = _enforce_threshold_invariant(0.50, 0.50, 0.60)
    assert p3 >= p2 >= reg


def test_no_flapping_under_clamped_thresholds():
    from core.version import decide_phase_transition, _enforce_threshold_invariant
    # A steady mediocre PRS must NOT oscillate. Use the clamped thresholds.
    p2, p3, reg = _enforce_threshold_invariant(0.50, 0.50, 0.60)
    phase = 2
    seq = []
    for _ in range(6):
        phase = decide_phase_transition(_hist(0.55, 0.55, 0.55), phase,
                                        phase2_advance=p2, phase3_advance=p3,
                                        regression_threshold=reg, stability_window=3)
        seq.append(phase)
    assert len(set(seq)) == 1  # settled, no 2,3,2,3 oscillation


def test_append_advances_phase2_at_default(tmp_path):
    ver = _make_ver_file(tmp_path, phase=1, prs_history=[])
    ver.append_prs(1, 0.35)  # >= default phase2_advance 0.30
    assert ver.load()["phase"] == 2


def test_append_does_not_advance_phase2_below_default(tmp_path):
    ver = _make_ver_file(tmp_path, phase=1, prs_history=[])
    ver.append_prs(1, 0.20)
    assert ver.load()["phase"] == 1


def test_append_regression_3_to_2(tmp_path):
    ver = _make_ver_file(tmp_path, phase=3, prs_history=[0.60, 0.20, 0.20])
    ver.append_prs(4, 0.20, regression_threshold=0.25, stability_window=3,
                   phase2_advance_threshold=0.30, phase3_advance_threshold=0.55)
    assert ver.load()["phase"] == 2


def test_append_no_new_kwargs_uses_defaults(tmp_path):
    ver = _make_ver_file(tmp_path, phase=1, prs_history=[])
    ver.append_prs(1, 0.10)  # below all thresholds -> stays phase 1
    assert ver.load()["phase"] == 1


def test_append_kds_writes_history(tmp_path):
    ver = _make_ver_file(tmp_path, phase=1, prs_history=[])
    ver.append_kds(2, 0.75, 5)
    data = ver.load()
    assert len(data["kds_history"]) == 1
    entry = data["kds_history"][0]
    assert entry["round"] == 2
    assert entry["mean_kds"] == 0.75
    assert entry["measured_chunks"] == 5
    assert "timestamp" in entry
