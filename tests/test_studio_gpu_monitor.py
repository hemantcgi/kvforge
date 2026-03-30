# tests/test_studio_gpu_monitor.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch
from studio.gpu_monitor import parse_nvidia_smi, find_vllm_processes, get_gpu_status


_SAMPLE_NVIDIA_SMI = """0, NVIDIA A10G, 24564, 2100
1, NVIDIA A10G, 24564, 20500
2, NVIDIA A10G, 24564, 20500
3, NVIDIA A10G, 24564, 1900"""

_SAMPLE_PS = """ubuntu 1956739  0.0  0.5 /home/ubuntu/qdrant/venv/bin/python3 -m vllm.entrypoints.openai.api_server --lora-modules uc3=examples/usecase3_squad/lora_checkpoints/v1/ --port 8093
ubuntu 1771247  0.0  0.5 /home/ubuntu/qdrant/venv/bin/python3 -m vllm.entrypoints.openai.api_server --lora-modules uc1=examples/usecase1_customer_support/lora_checkpoints/v8_dpo/final --port 8091"""


def test_parse_nvidia_smi_returns_four_gpus():
    gpus = parse_nvidia_smi(_SAMPLE_NVIDIA_SMI)
    assert len(gpus) == 4


def test_parse_nvidia_smi_free_gpu():
    import pytest
    gpus = parse_nvidia_smi(_SAMPLE_NVIDIA_SMI)
    assert gpus[0]["free_gb"] == pytest.approx((24564 - 2100) / 1024, abs=0.1)
    assert gpus[0]["status"] == "free"


def test_parse_nvidia_smi_busy_gpu():
    import pytest
    gpus = parse_nvidia_smi(_SAMPLE_NVIDIA_SMI)
    assert gpus[1]["status"] == "busy"
    assert gpus[1]["used_gb"] == pytest.approx(20500 / 1024, abs=0.1)


def test_find_vllm_processes_parses_uc_id():
    procs = find_vllm_processes(_SAMPLE_PS)
    assert len(procs) == 2
    pids = {p["pid"] for p in procs}
    assert 1956739 in pids


def test_find_vllm_processes_extracts_port():
    procs = find_vllm_processes(_SAMPLE_PS)
    ports = {p["port"] for p in procs}
    assert 8093 in ports
    assert 8091 in ports


def test_get_gpu_status_marks_busy_gpus():
    with patch("studio.gpu_monitor._run_nvidia_smi", return_value=_SAMPLE_NVIDIA_SMI), \
         patch("studio.gpu_monitor._run_ps", return_value=_SAMPLE_PS):
        status = get_gpu_status()
    assert status["has_free_gpu"] is True
    busy = [g for g in status["gpus"] if g["status"] == "busy"]
    assert len(busy) == 2


def test_get_gpu_status_no_free_gpu():
    all_busy = "0, NVIDIA A10G, 24564, 21000\n1, NVIDIA A10G, 24564, 21000\n2, NVIDIA A10G, 24564, 21000\n3, NVIDIA A10G, 24564, 21000"
    with patch("studio.gpu_monitor._run_nvidia_smi", return_value=all_busy), \
         patch("studio.gpu_monitor._run_ps", return_value=""):
        status = get_gpu_status()
    assert status["has_free_gpu"] is False


def test_get_gpu_status_nvidia_smi_unavailable():
    with patch("studio.gpu_monitor._run_nvidia_smi", side_effect=FileNotFoundError):
        status = get_gpu_status()
    assert status["error"] == "nvidia-smi not found"
    assert status["gpus"] == []
