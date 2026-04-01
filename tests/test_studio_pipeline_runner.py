"""Tests for studio pipeline runner step registration and command building."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_sleep_faq_step_registered():
    from studio.pipeline_runner import STEP_MODULES
    assert "sleep-faq" in STEP_MODULES
    assert STEP_MODULES["sleep-faq"] == "pipeline.sleep_faq_generator"


def test_sleep_faq_is_not_gpu_required():
    from studio.pipeline_runner import GPU_REQUIRED_STEPS
    assert "sleep-faq" not in GPU_REQUIRED_STEPS


def test_sleep_faq_cmd_includes_output_and_count(tmp_path):
    import json
    from unittest.mock import patch
    from studio.pipeline_runner import _build_cmd
    # Create fake uc_config.json with sleep_faq_count
    uc_dir = tmp_path / "examples" / "test_uc"
    uc_dir.mkdir(parents=True)
    uc_cfg = {"llm": {"sleep_faq_count": 75}}
    (uc_dir / "uc_config.json").write_text(json.dumps(uc_cfg))
    (uc_dir / "config.json").write_text("{}")
    with patch("studio.pipeline_runner.ROOT", tmp_path):
        cmd = _build_cmd("test_uc", "sleep-faq")
    cmd_str = " ".join(cmd)
    assert "--output" in cmd_str
    assert "faqs.json" in cmd_str
    assert "--count" in cmd_str
    assert "75" in cmd_str
