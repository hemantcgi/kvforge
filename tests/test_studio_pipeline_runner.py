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


def test_ab_eval_cmd_includes_dashboard_url(tmp_path):
    """ab-eval step passes --dashboard-url derived from config.json dashboard_port."""
    import json
    from unittest.mock import patch
    from studio.pipeline_runner import _build_cmd
    uc_dir = tmp_path / "examples" / "test_uc"
    uc_dir.mkdir(parents=True)
    (uc_dir / "config.json").write_text(json.dumps({"dashboard_port": 8099}))
    (uc_dir / "uc_config.json").write_text("{}")
    with patch("studio.pipeline_runner.ROOT", tmp_path):
        cmd = _build_cmd("test_uc", "ab-eval")
    assert "--dashboard-url" in cmd
    idx = cmd.index("--dashboard-url")
    assert cmd[idx + 1] == "http://localhost:8099"


def test_ab_eval_cmd_uses_fallback_port_when_no_config(tmp_path):
    """ab-eval defaults to port 8081 when config.json is absent."""
    from unittest.mock import patch
    from studio.pipeline_runner import _build_cmd
    uc_dir = tmp_path / "examples" / "test_uc2"
    uc_dir.mkdir(parents=True)
    with patch("studio.pipeline_runner.ROOT", tmp_path):
        cmd = _build_cmd("test_uc2", "ab-eval")
    assert "--dashboard-url" in cmd
    idx = cmd.index("--dashboard-url")
    assert cmd[idx + 1] == "http://localhost:8081"


def test_prs_eval_cmd_reads_sample_from_uc_config(tmp_path):
    """prs-eval step passes --sample from prs_eval_sample in uc_config.json."""
    import json
    from unittest.mock import patch
    from studio.pipeline_runner import _build_cmd
    uc_dir = tmp_path / "examples" / "test_uc3"
    uc_dir.mkdir(parents=True)
    (uc_dir / "uc_config.json").write_text(json.dumps({"prs_eval_sample": 10}))
    with patch("studio.pipeline_runner.ROOT", tmp_path):
        cmd = _build_cmd("test_uc3", "prs-eval")
    assert "--sample" in cmd
    idx = cmd.index("--sample")
    assert cmd[idx + 1] == "10"


def test_prs_eval_cmd_defaults_to_20_when_no_config(tmp_path):
    """prs-eval defaults --sample to 20 when uc_config.json has no prs_eval_sample."""
    import json
    from unittest.mock import patch
    from studio.pipeline_runner import _build_cmd
    uc_dir = tmp_path / "examples" / "test_uc4"
    uc_dir.mkdir(parents=True)
    (uc_dir / "uc_config.json").write_text(json.dumps({}))
    with patch("studio.pipeline_runner.ROOT", tmp_path):
        cmd = _build_cmd("test_uc4", "prs-eval")
    assert "--sample" in cmd
    idx = cmd.index("--sample")
    assert cmd[idx + 1] == "20"
