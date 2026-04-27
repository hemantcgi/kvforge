# tests/test_gpu_realtime.py
from studio.gpu_monitor import parse_gpu_realtime

STATS = "0, NVIDIA A10G, 1229, 22528, 8, 36, 72.00\n1, NVIDIA A10G, 19865, 22528, 94, 71, 195.50"
UUIDS = "0, GPU-aaaa-1111\n1, GPU-bbbb-2222"
PROCS = "GPU-bbbb-2222, 28431, python vllm.entrypoints.openai.api_server, 19397"


def test_gpu_count():
    r = parse_gpu_realtime(STATS, UUIDS, PROCS)
    assert len(r["gpus"]) == 2


def test_memory_gb_conversion():
    r = parse_gpu_realtime(STATS, UUIDS, PROCS)
    assert r["gpus"][0]["used_gb"] == 1.2
    assert r["gpus"][0]["total_gb"] == 22.0


def test_util_temp_power():
    r = parse_gpu_realtime(STATS, UUIDS, PROCS)
    assert r["gpus"][1]["util_pct"] == 94
    assert r["gpus"][1]["temp_c"] == 71
    assert r["gpus"][1]["power_w"] == 195


def test_process_assigned_to_correct_gpu():
    r = parse_gpu_realtime(STATS, UUIDS, PROCS)
    assert r["gpus"][0]["processes"] == []
    assert len(r["gpus"][1]["processes"]) == 1
    assert r["gpus"][1]["processes"][0]["pid"] == 28431
    assert r["gpus"][1]["processes"][0]["mem_mib"] == 19397


def test_has_free_gpu_true():
    assert parse_gpu_realtime(STATS, UUIDS, PROCS)["has_free_gpu"] is True


def test_has_free_gpu_false():
    busy = "0, NVIDIA A10G, 20000, 22528, 91, 70, 190.00\n1, NVIDIA A10G, 19865, 22528, 94, 71, 195.00"
    assert parse_gpu_realtime(busy, UUIDS, "")["has_free_gpu"] is False


def test_empty_procs():
    r = parse_gpu_realtime(STATS, UUIDS, "")
    assert r["gpus"][0]["processes"] == []
    assert r["gpus"][1]["processes"] == []


def test_malformed_proc_line_skipped():
    r = parse_gpu_realtime(STATS, UUIDS, "bad-line-no-commas")
    assert r["gpus"][0]["processes"] == []
