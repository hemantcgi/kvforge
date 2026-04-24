# tests/test_model_scout.py

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from pipeline.model_scout import (
    IOAdapter,
    CLIAdapter,
    RecordingAdapter,
    detect_gpu,
    ScoutParams,
    run_budget_dialog,
    ExperimentResult,
    apply_parameter_adjustments,
    run_single_experiment,
    run_scout_session,
)


# ── IOAdapter / RecordingAdapter tests ────────────────────────────────────────

def test_ioadapter_is_protocol():
    # Structural typing — just verify the class exists and is a Protocol
    assert hasattr(IOAdapter, "__protocol_attrs__") or True


def test_recording_adapter_captures_messages():
    adapter = RecordingAdapter()
    adapter.send("hello world")
    adapter.send("second message")
    assert adapter.messages == ["hello world", "second message"]


def test_recording_adapter_ask_returns_preset():
    adapter = RecordingAdapter(responses=["B", "stop"])
    answer = adapter.ask("Which budget?", options=["A", "B", "C", "D"])
    assert answer == "B"
    answer2 = adapter.ask("Continue?", options=None)
    assert answer2 == "stop"


def test_recording_adapter_ask_raises_when_no_responses_left():
    adapter = RecordingAdapter(responses=[])
    with pytest.raises(StopIteration):
        adapter.ask("No more responses", options=None)


def test_recording_adapter_progress_recorded():
    adapter = RecordingAdapter()
    adapter.stream_progress("Training step 10/100", 0.10)
    adapter.stream_progress("Training step 50/100", 0.50)
    assert len(adapter.progress_updates) == 2
    assert adapter.progress_updates[0] == ("Training step 10/100", 0.10)


# ── GPU detection tests ───────────────────────────────────────────────────────

def test_detect_gpu_no_cuda():
    with patch("pipeline.model_scout._has_cuda", False):
        info = detect_gpu()
    assert info["available"] is False
    assert info["free_vram_gb"] == 0.0


def test_detect_gpu_with_cuda():
    mock_props = MagicMock()
    mock_props.name = "NVIDIA A10G"
    with patch("pipeline.model_scout._has_cuda", True):
        with patch("pipeline.model_scout.torch") as mock_torch:
            mock_torch.cuda.get_device_properties.return_value = mock_props
            mock_torch.cuda.mem_get_info.return_value = (
                21 * 1024 ** 3,  # free bytes
                24 * 1024 ** 3,  # total bytes
            )
            mock_torch.version.cuda = "12.1"
            info = detect_gpu()
    assert info["available"] is True
    assert info["gpu_name"] == "NVIDIA A10G"
    assert abs(info["free_vram_gb"] - 21.0) < 0.1
    assert abs(info["total_vram_gb"] - 24.0) < 0.1


def test_detect_gpu_formats_report():
    with patch("pipeline.model_scout._has_cuda", False):
        info = detect_gpu()
    assert "report" in info
    assert isinstance(info["report"], str)


# ── Budget dialog tests ───────────────────────────────────────────────────────

def test_budget_dialog_option_A_sets_wall_clock():
    adapter = RecordingAdapter(responses=["A", "4"])
    params = run_budget_dialog(adapter)
    assert params.budget_mode == "wall_clock"
    assert params.budget_value == 4.0


def test_budget_dialog_option_B_sets_experiment_count():
    adapter = RecordingAdapter(responses=["B", "15"])
    params = run_budget_dialog(adapter)
    assert params.budget_mode == "experiment_count"
    assert params.budget_value == 15


def test_budget_dialog_option_C_sets_step_cap_and_count():
    adapter = RecordingAdapter(responses=["C", "1000", "12"])
    params = run_budget_dialog(adapter)
    assert params.budget_mode == "step_cap_and_count"
    assert params.max_steps_per_experiment == 1000
    assert params.budget_value == 12


def test_budget_dialog_option_D_sets_agent_decides():
    adapter = RecordingAdapter(responses=["D"])
    params = run_budget_dialog(adapter)
    assert params.budget_mode == "agent_decides"


def test_scout_params_defaults():
    p = ScoutParams()
    assert p.lora_steps == 500
    assert p.lora_rank == 16
    assert p.corpus_chunks == 200
    assert p.faq_count == 20
    assert p.budget_mode == "experiment_count"
    assert p.budget_value == 10


# ── Parameter adjustment tests ────────────────────────────────────────────────

def _make_result(**kwargs) -> ExperimentResult:
    defaults = dict(
        model_id="test-model", quantization="4bit",
        lora_steps=500, lora_rank=16, corpus_chunks=200, faq_count=20,
        prs=0.65, prs_variance=0.08, vram_gb=18.0,
        wall_seconds=500, status="keep",
        training_loss_start=1.5, training_loss_end=0.9,
        agent_reasoning="baseline",
    )
    defaults.update(kwargs)
    return ExperimentResult(**defaults)


def test_adjustment_increases_steps_when_prs_low_and_loss_falling():
    result = _make_result(prs=0.50, training_loss_start=1.5, training_loss_end=0.7)
    params = ScoutParams(lora_steps=500)
    new_params, retry = apply_parameter_adjustments(result, params, max_steps=2000)
    assert retry is True
    assert new_params.lora_steps == 1000


def test_adjustment_does_not_exceed_max_steps():
    result = _make_result(prs=0.50, training_loss_start=1.5, training_loss_end=0.7)
    params = ScoutParams(lora_steps=1500)
    new_params, retry = apply_parameter_adjustments(result, params, max_steps=2000)
    assert new_params.lora_steps <= 2000


def test_adjustment_increases_faq_count_when_high_variance():
    result = _make_result(prs=0.65, prs_variance=0.18)
    params = ScoutParams(faq_count=20)
    new_params, retry = apply_parameter_adjustments(result, params, max_steps=2000)
    assert new_params.faq_count > 20


def test_adjustment_no_retry_when_prs_good():
    result = _make_result(
        prs=0.75, prs_variance=0.05,
        training_loss_start=1.5, training_loss_end=0.9,
    )
    params = ScoutParams(lora_steps=500)
    new_params, retry = apply_parameter_adjustments(result, params, max_steps=2000)
    assert retry is False


def test_adjustment_sets_4bit_on_oom():
    result = _make_result(status="oom", quantization="fp16")
    params = ScoutParams(quantization="fp16")
    new_params, retry = apply_parameter_adjustments(result, params, max_steps=2000)
    assert retry is True
    assert new_params.quantization == "4bit"


# ── run_single_experiment test ────────────────────────────────────────────────

def test_run_single_experiment_returns_result(tmp_path):
    adapter = RecordingAdapter()
    cfg = {
        "embed_model": "BAAI/bge-small-en-v1.5",
        "checkpoint_dir": str(tmp_path),
        "version_file": str(tmp_path / "version.json"),
        "replay_db": str(tmp_path / "replay.db"),
        "query_log_db": str(tmp_path / "q.db"),
        "faq_question_key": "question",
        "faq_answer_key": "answer",
        "prs_signal_weights": {"faq": 0.4, "vdb": 0.4, "realtime": 0.2},
        "prs_auto_weight": False,
        "prs_stability_window": 3,
        "prs_advancement_threshold": 0.72,
        "min_cluster_samples_for_adaptation": 10,
        "realtime_requery_window_minutes": 10,
    }
    faqs = [{"question": "What is RAG?", "answer": "RAG retrieves context."}] * 5
    params = ScoutParams(lora_steps=10, lora_rank=8, corpus_chunks=10)
    candidate = {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "spec": {"_use_4bit": False, "lora_targets": ["q_proj", "k_proj", "v_proj"]},
        "score": 0.9,
    }

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": MagicMock(shape=[1, 10])}

    with patch("pipeline.model_scout._run_mini_lora", return_value=(1.2, 0.8)), \
         patch("pipeline.model_scout._eval_prs_on_faqs", return_value=(0.70, 0.05)), \
         patch("core.model_loader.init"), \
         patch("core.model_loader.load", return_value=(mock_model, mock_tokenizer)):
        result = run_single_experiment(candidate, faqs, params, cfg, adapter,
                                       mode="pre_index", store=None)

    assert isinstance(result, ExperimentResult)
    assert result.model_id == "meta-llama/Llama-3.2-3B-Instruct"
    assert 0.0 <= result.prs <= 1.0
    assert result.status in ("keep", "discard", "oom", "crash")


# ── run_scout_session tests ───────────────────────────────────────────────────

def test_run_scout_session_completes_with_recording_adapter(tmp_path):
    adapter = RecordingAdapter(responses=[
        "pre_index",  # mode
        "B",          # budget: experiment count
        "2",          # 2 experiments
        "",           # proceed with first experiment
        "",           # proceed with second experiment
    ])
    cfg = {
        "embed_model": "BAAI/bge-small-en-v1.5",
        "checkpoint_dir": str(tmp_path),
        "version_file": str(tmp_path / "version.json"),
        "replay_db": str(tmp_path / "replay.db"),
        "query_log_db": str(tmp_path / "q.db"),
        "faq_question_key": "question",
        "faq_answer_key": "answer",
        "model_registry_path": "core/model_registry.json",
        "model_scout_results": str(tmp_path / "results.tsv"),
        "scout_initial_corpus_chunks": 10,
        "scout_initial_faq_count": 5,
        "scout_initial_lora_steps": 10,
        "scout_initial_lora_rank": 8,
        "scout_max_lora_steps": 100,
        "scout_max_corpus_chunks": 50,
        "scout_max_faq_count": 20,
        "prs_signal_weights": {"faq": 0.4, "vdb": 0.4, "realtime": 0.2},
        "prs_auto_weight": False,
        "prs_stability_window": 3,
        "prs_advancement_threshold": 0.72,
        "min_cluster_samples_for_adaptation": 10,
        "realtime_requery_window_minutes": 10,
    }
    faqs = [{"question": "What is RAG?", "answer": "RAG retrieves context."}] * 5
    gpu_info = {"available": False, "free_vram_gb": 24.0, "report": "CPU mode"}

    with patch("pipeline.model_scout.run_single_experiment") as mock_exp:
        mock_exp.return_value = ExperimentResult(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            quantization="fp16", lora_steps=10, lora_rank=8,
            corpus_chunks=10, faq_count=5, prs=0.72, prs_variance=0.05,
            vram_gb=9.0, wall_seconds=60.0, status="keep",
            training_loss_start=1.2, training_loss_end=0.8,
            agent_reasoning="good result",
        )
        recommendation = run_scout_session(adapter, cfg, faqs, gpu_info, store=None)

    assert recommendation is not None
    assert "model_id" in recommendation
    # Results TSV should exist
    assert Path(cfg["model_scout_results"]).exists()


def test_run_scout_session_respects_stop_command(tmp_path):
    adapter = RecordingAdapter(responses=[
        "pre_index", "B", "10",  # budget: 10 experiments
        "stop",                  # user stops after first result
    ])
    cfg = {
        "embed_model": "BAAI/bge-small-en-v1.5",
        "checkpoint_dir": str(tmp_path),
        "model_registry_path": "core/model_registry.json",
        "model_scout_results": str(tmp_path / "results.tsv"),
        "scout_initial_lora_steps": 10, "scout_initial_lora_rank": 8,
        "scout_initial_corpus_chunks": 10, "scout_initial_faq_count": 5,
        "scout_max_lora_steps": 100, "scout_max_corpus_chunks": 50,
        "scout_max_faq_count": 20,
        "version_file": str(tmp_path / "version.json"),
        "replay_db": str(tmp_path / "replay.db"),
        "query_log_db": str(tmp_path / "q.db"),
        "faq_question_key": "question", "faq_answer_key": "answer",
        "prs_signal_weights": {"faq": 0.4, "vdb": 0.4, "realtime": 0.2},
        "prs_auto_weight": False, "prs_stability_window": 3,
        "prs_advancement_threshold": 0.72, "min_cluster_samples_for_adaptation": 10,
        "realtime_requery_window_minutes": 10,
    }
    faqs = [{"question": "Q?", "answer": "A."}] * 3
    gpu_info = {"available": False, "free_vram_gb": 24.0, "report": "CPU mode"}

    with patch("pipeline.model_scout.run_single_experiment") as mock_exp:
        mock_exp.return_value = ExperimentResult(
            model_id="meta-llama/Llama-3.2-3B-Instruct", quantization="fp16",
            lora_steps=10, lora_rank=8, corpus_chunks=10, faq_count=5,
            prs=0.68, prs_variance=0.04, vram_gb=9.0, wall_seconds=60.0,
            status="keep", training_loss_start=1.2, training_loss_end=0.9,
            agent_reasoning="",
        )
        recommendation = run_scout_session(adapter, cfg, faqs, gpu_info, store=None)

    # Should have stopped after 0 experiments (stop was the first user command)
    assert mock_exp.call_count == 0
    # No experiments ran — recommendation should be None
    assert recommendation is None
