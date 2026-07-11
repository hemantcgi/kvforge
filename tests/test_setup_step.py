# tests/test_setup_step.py
import pytest


def test_setup_in_valid_steps():
    from studio.api import VALID_STEPS
    assert "setup" in VALID_STEPS


def test_setup_cmd_points_to_setup_py(tmp_path):
    """_build_cmd for 'setup' must invoke examples/{uc_id}/setup.py directly."""
    import sys
    from studio.pipeline_runner import _build_cmd
    cmd = _build_cmd("usecase1_customer_support", "setup")
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("setup.py")
    assert "usecase1_customer_support" in cmd[1]


def test_setup_not_in_gpu_required_steps():
    from studio.pipeline_runner import GPU_REQUIRED_STEPS
    assert "setup" not in GPU_REQUIRED_STEPS


def test_run_step_rejects_unknown_step():
    """Existing guard still rejects unknown steps."""
    from studio.pipeline_runner import STEP_MODULES
    assert "bogus_step_xyz" not in STEP_MODULES
