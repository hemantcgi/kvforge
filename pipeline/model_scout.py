# pipeline/model_scout.py
"""ModelScout — interactive agent that experiments across candidate open models
to identify the best fit for a KVForge use case.

Public API:
    IOAdapter, CLIAdapter, RecordingAdapter  — I/O protocol + implementations
    detect_gpu()                              — GPU capability detection
    ScoutParams                               — mutable experiment parameters
    run_budget_dialog(adapter) -> ScoutParams — interactive budget selection
    ExperimentResult                          — per-experiment result dataclass
    apply_parameter_adjustments(...)          — rule-based param tuning
    run_single_experiment(...)                — load model, mini-LoRA, PRS eval
    run_scout_session(...)                    — main agent loop
"""

from __future__ import annotations

import copy
import csv
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

# ── Optional torch import ─────────────────────────────────────────────────────

try:
    import torch
    _has_cuda = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore
    _has_cuda = False


# ── IOAdapter protocol ────────────────────────────────────────────────────────

@runtime_checkable
class IOAdapter(Protocol):
    """Protocol for all ModelScout I/O.  Implement send, ask, stream_progress."""

    def send(self, message: str) -> None:
        """Display a message to the user."""
        ...

    def ask(self, question: str, options: list[str] | None = None) -> str:
        """Ask the user a question, optionally with a labelled options list.
        Returns the user's response as a string."""
        ...

    def stream_progress(self, label: str, pct: float) -> None:
        """Report incremental progress (pct in [0.0, 1.0])."""
        ...


class CLIAdapter:
    """IOAdapter that reads from stdin and writes to stdout."""

    def send(self, message: str) -> None:
        print(message, flush=True)

    def ask(self, question: str, options: list[str] | None = None) -> str:
        print(question, flush=True)
        if options:
            for i, opt in enumerate(options):
                print(f"  [{chr(65 + i)}] {opt}", flush=True)
        return input("> ").strip()

    def stream_progress(self, label: str, pct: float) -> None:
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {pct * 100:.0f}% {label}", end="", flush=True)
        if pct >= 1.0:
            print()


class RecordingAdapter:
    """Test double — records all interactions and serves preset responses."""

    def __init__(self, responses: list[str] | None = None):
        self.messages: list[str] = []
        self.progress_updates: list[tuple[str, float]] = []
        self._responses = iter(responses or [])

    def send(self, message: str) -> None:
        self.messages.append(message)

    def ask(self, question: str, options: list[str] | None = None) -> str:
        self.messages.append(f"[QUESTION] {question}")
        return next(self._responses)  # raises StopIteration when exhausted

    def stream_progress(self, label: str, pct: float) -> None:
        self.progress_updates.append((label, pct))


# ── GPU detection ─────────────────────────────────────────────────────────────

def detect_gpu() -> dict:
    """Auto-detect GPU capabilities.

    Returns a dict with keys:
        available, gpu_name, free_vram_gb, total_vram_gb, cuda_version, report
    """
    if not _has_cuda:
        return {
            "available": False,
            "gpu_name": "CPU only",
            "free_vram_gb": 0.0,
            "total_vram_gb": 0.0,
            "cuda_version": None,
            "report": (
                "No CUDA GPU detected. ModelScout will be limited to "
                "CPU-compatible models only."
            ),
        }
    props = torch.cuda.get_device_properties(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / 1024 ** 3
    total_gb = total_bytes / 1024 ** 3
    cuda_ver = torch.version.cuda
    report = (
        f"Detected: {props.name} | "
        f"VRAM: {free_gb:.1f}GB free / {total_gb:.1f}GB total | "
        f"CUDA {cuda_ver}"
    )
    return {
        "available": True,
        "gpu_name": props.name,
        "free_vram_gb": round(free_gb, 2),
        "total_vram_gb": round(total_gb, 2),
        "cuda_version": cuda_ver,
        "report": report,
    }


# ── ScoutParams dataclass ─────────────────────────────────────────────────────

@dataclass
class ScoutParams:
    """Mutable experiment parameters, adjusted by the agent loop."""

    lora_steps: int = 500
    lora_rank: int = 16
    corpus_chunks: int = 200
    faq_count: int = 20
    quantization: str = "auto"          # "auto" | "fp16" | "4bit" | "8bit"
    budget_mode: str = "experiment_count"
    budget_value: float = 10            # hours | count depending on mode
    max_steps_per_experiment: int = 2000
    experiments_run: int = 0
    session_start: float = field(default_factory=time.time)


# ── Budget dialog ─────────────────────────────────────────────────────────────

_BUDGET_OPTIONS = [
    "Total wall-clock time (e.g. run for N hours)",
    "Total experiment count (e.g. run at most N experiments)",
    "Per-experiment step cap + total count",
    "Agent decides when confident enough",
]


def run_budget_dialog(adapter: IOAdapter) -> ScoutParams:
    """Ask the user to choose a budget mode.  Returns an initialised ScoutParams."""
    params = ScoutParams()
    choice = adapter.ask(
        "How would you like to control the ModelScout session?",
        options=_BUDGET_OPTIONS,
    ).upper().strip()

    if choice == "A":
        hours = float(adapter.ask("Run for how many hours? (e.g. 4)"))
        params.budget_mode = "wall_clock"
        params.budget_value = hours
    elif choice == "B":
        count = int(adapter.ask("Run at most how many experiments? (e.g. 15)"))
        params.budget_mode = "experiment_count"
        params.budget_value = count
    elif choice == "C":
        max_steps = int(adapter.ask("Max LoRA steps per experiment? (e.g. 1000)"))
        count = int(adapter.ask("Total number of experiments? (e.g. 12)"))
        params.budget_mode = "step_cap_and_count"
        params.max_steps_per_experiment = max_steps
        params.budget_value = count
    else:  # D or anything else
        params.budget_mode = "agent_decides"

    return params


def _budget_exhausted(params: ScoutParams) -> bool:
    """Return True when the user's chosen budget has been consumed."""
    if params.budget_mode == "wall_clock":
        elapsed_hours = (time.time() - params.session_start) / 3600
        return elapsed_hours >= params.budget_value
    elif params.budget_mode in ("experiment_count", "step_cap_and_count"):
        return params.experiments_run >= params.budget_value
    return False  # agent_decides — never auto-exhausted


# ── ExperimentResult dataclass ────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    model_id: str
    quantization: str
    lora_steps: int
    lora_rank: int
    corpus_chunks: int
    faq_count: int
    prs: float
    prs_variance: float
    vram_gb: float
    wall_seconds: float
    status: str                  # "keep" | "discard" | "oom" | "crash"
    training_loss_start: float
    training_loss_end: float
    agent_reasoning: str
    git_commit: str = ""


# ── Parameter adjustment rules ────────────────────────────────────────────────

def apply_parameter_adjustments(
    result: ExperimentResult,
    params: ScoutParams,
    max_steps: int = 2000,
    max_faq: int = 100,
    max_chunks: int = 2000,
) -> tuple[ScoutParams, bool]:
    """Apply rule-based parameter adjustments derived from the last experiment result.

    Returns:
        (new_params, should_retry_same_model)
    """
    new = copy.copy(params)
    retry = False

    # OOM at fp16 → retry same model with 4bit
    if result.status == "oom" and result.quantization == "fp16":
        new.quantization = "4bit"
        return new, True

    # OOM at 4bit → can't reduce further, skip
    if result.status == "oom" and result.quantization == "4bit":
        return new, False

    # Low PRS + loss still falling → more steps
    loss_falling = result.training_loss_end < result.training_loss_start * 0.7
    if result.prs < 0.55 and loss_falling:
        new.lora_steps = min(new.lora_steps * 2, max_steps)
        retry = True

    # High FAQ variance → more FAQs (applied for next run, not a retry trigger alone)
    if result.prs_variance > 0.15:
        new.faq_count = min(int(new.faq_count * 1.5), max_faq)

    return new, retry


# ── TSV result writer ─────────────────────────────────────────────────────────

def _write_result(results_path: str, result: ExperimentResult) -> None:
    file_exists = Path(results_path).exists()
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "commit", "model", "quantization", "lora_steps", "lora_rank",
                "corpus_chunks", "faq_count", "prs", "vram_gb",
                "wall_seconds", "status", "agent_reasoning",
            ],
            delimiter="\t",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "commit": result.git_commit[:7] if result.git_commit else "unknown",
            "model": result.model_id.split("/")[-1],
            "quantization": result.quantization,
            "lora_steps": result.lora_steps,
            "lora_rank": result.lora_rank,
            "corpus_chunks": result.corpus_chunks,
            "faq_count": result.faq_count,
            "prs": f"{result.prs:.4f}",
            "vram_gb": f"{result.vram_gb:.1f}",
            "wall_seconds": int(result.wall_seconds),
            "status": result.status,
            "agent_reasoning": result.agent_reasoning,
        })


# ── Mini-LoRA and PRS helpers ─────────────────────────────────────────────────

def _run_mini_lora(
    model, tokenizer, faqs: list[dict], params: ScoutParams, cfg: dict
) -> tuple[float, float]:
    """Run a short LoRA training round.  Returns (loss_start, loss_end)."""
    from peft import LoraConfig, get_peft_model
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback
    import torch as _torch

    lora_config = LoraConfig(
        r=params.lora_rank,
        lora_alpha=params.lora_rank * 2,
        target_modules=cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj"]),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)

    texts = [f"Q: {f['question']}\nA: {f['answer']}" for f in faqs]
    encodings = tokenizer(
        texts, truncation=True, max_length=256, padding=True, return_tensors="pt"
    )

    class _FaqDataset(_torch.utils.data.Dataset):
        def __len__(self):
            return len(texts)

        def __getitem__(self, i):
            return {k: v[i] for k, v in encodings.items()}

    loss_history: list[float] = []

    class LossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                loss_history.append(logs["loss"])

    trainer = Trainer(
        model=peft_model,
        args=TrainingArguments(
            output_dir=cfg.get("checkpoint_dir", "/tmp/scout_ckpt"),
            num_train_epochs=1,
            max_steps=params.lora_steps,
            per_device_train_batch_size=1,
            logging_steps=max(1, params.lora_steps // 10),
            save_steps=params.lora_steps + 1,
            fp16=_torch.cuda.is_available(),
            report_to="none",
        ),
        train_dataset=_FaqDataset(),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[LossCallback()],
    )
    trainer.train()
    loss_start = loss_history[0] if loss_history else 1.0
    loss_end = loss_history[-1] if loss_history else 1.0
    return loss_start, loss_end


def _eval_prs_on_faqs(
    model, tokenizer, faqs: list[dict], cfg: dict
) -> tuple[float, float]:
    """Lightweight PRS evaluation using FAQ cosine similarity.

    Returns (mean_prs, std_prs).
    """
    import numpy as np
    from fastembed import TextEmbedding
    from transformers import pipeline as hf_pipeline

    pipe = hf_pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=128, do_sample=False,
    )
    embedder = TextEmbedding(
        model_name=cfg.get("embed_model", "BAAI/bge-small-en-v1.5"),
        show_download_progress=False,
    )
    q_key = cfg.get("faq_question_key", "question")
    a_key = cfg.get("faq_answer_key", "answer")
    scores: list[float] = []

    for faq in faqs:
        q = faq.get(q_key, "")
        gt = faq.get(a_key, "")
        if not q or not gt:
            continue
        output = pipe(q)
        param_ans = output[0]["generated_text"] if output else ""
        embs = np.array(list(embedder.embed([param_ans, gt])))
        if len(embs) == 2:
            cos = float(
                np.dot(embs[0], embs[1])
                / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-9)
            )
            scores.append(cos)

    if not scores:
        return 0.0, 0.0
    return float(np.mean(scores)), float(np.std(scores))


# ── Single experiment ─────────────────────────────────────────────────────────

def run_single_experiment(
    candidate: dict,
    faqs: list[dict],
    params: ScoutParams,
    cfg: dict,
    adapter: IOAdapter,
    mode: str = "pre_index",
    store=None,
) -> ExperimentResult:
    """Load the candidate model, run mini-LoRA, evaluate PRS, return result."""
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    import core.model_loader as model_loader  # type: ignore

    model_id = candidate["model_id"]
    spec = candidate["spec"]
    if spec.get("_use_4bit"):
        quantization = "4bit"
    elif params.quantization != "auto":
        quantization = params.quantization
    else:
        quantization = "fp16"

    run_cfg = dict(cfg)
    run_cfg["llm_model"] = model_id
    run_cfg["quantization"] = quantization
    run_cfg["lora_target_modules"] = spec.get("lora_targets", ["q_proj", "k_proj", "v_proj"])
    run_cfg["lora_rank"] = params.lora_rank

    start_time = time.time()
    try:
        model_loader.init(run_cfg)
        model, tokenizer = model_loader.load()
        adapter.stream_progress(f"Loaded {model_id.split('/')[-1]}", 0.1)

        loss_start, loss_end = _run_mini_lora(model, tokenizer, faqs, params, run_cfg)
        adapter.stream_progress("Evaluating PRS", 0.9)

        mean_prs, prs_variance = _eval_prs_on_faqs(model, tokenizer, faqs, run_cfg)
        adapter.stream_progress("Done", 1.0)

        vram_gb = 0.0
        if _has_cuda:
            vram_gb = torch.cuda.memory_allocated() / 1024 ** 3

        wall_seconds = time.time() - start_time
        status = "keep" if mean_prs >= 0.55 else "discard"

        return ExperimentResult(
            model_id=model_id,
            quantization=quantization,
            lora_steps=params.lora_steps,
            lora_rank=params.lora_rank,
            corpus_chunks=params.corpus_chunks,
            faq_count=params.faq_count,
            prs=round(mean_prs, 4),
            prs_variance=round(prs_variance, 4),
            vram_gb=round(vram_gb, 1),
            wall_seconds=round(wall_seconds, 1),
            status=status,
            training_loss_start=round(loss_start, 4),
            training_loss_end=round(loss_end, 4),
            agent_reasoning="",
        )

    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            return ExperimentResult(
                model_id=model_id, quantization=quantization,
                lora_steps=params.lora_steps, lora_rank=params.lora_rank,
                corpus_chunks=params.corpus_chunks, faq_count=params.faq_count,
                prs=0.0, prs_variance=0.0, vram_gb=0.0,
                wall_seconds=round(time.time() - start_time, 1),
                status="oom",
                training_loss_start=0.0, training_loss_end=0.0,
                agent_reasoning="OOM — try 4bit quantization",
            )
        raise


# ── Main agent loop ───────────────────────────────────────────────────────────

def _check_user_interrupt(adapter: IOAdapter) -> str | None:
    """Non-blocking command check.  SSEAdapter stores the last received command."""
    if hasattr(adapter, "pending_command"):
        cmd = adapter.pending_command
        adapter.pending_command = None  # type: ignore[attr-defined]
        return cmd
    return None


def _describe_adjustment(old: ScoutParams, new: ScoutParams) -> str:
    parts = []
    if new.lora_steps != old.lora_steps:
        parts.append(f"lora_steps {old.lora_steps}->{new.lora_steps}")
    if new.faq_count != old.faq_count:
        parts.append(f"faq_count {old.faq_count}->{new.faq_count}")
    if new.quantization != old.quantization:
        parts.append(f"quantization {old.quantization}->{new.quantization}")
    return ", ".join(parts) if parts else "no change"


def run_scout_session(
    adapter: IOAdapter,
    cfg: dict,
    faqs: list[dict],
    gpu_info: dict,
    store=None,
) -> dict | None:
    """Main ModelScout agent loop.

    Args:
        adapter: An IOAdapter implementation (CLIAdapter, SSEAdapter, RecordingAdapter).
        cfg:     UC config dict (same schema as DatasourceConfig.model_dump()).
        faqs:    Pre-built FAQ list [{question, answer}, ...].
        gpu_info: Output of detect_gpu().
        store:   Optional VectorStore instance (post-index mode only).

    Returns:
        A dict with the best recommendation {model_id, quantization, lora_rank, prs},
        or None if no experiments completed.
    """
    from core.model_registry import load_registry, get_candidate_shortlist  # lazy import

    adapter.send(gpu_info["report"])

    # Mode selection
    raw_mode = adapter.ask(
        "Run ModelScout before indexing (pre-index) or against existing VDB (post-index)?",
        options=[
            "pre_index — I haven't indexed yet",
            "post_index — VDB already exists",
        ],
    ).lower()
    mode = "pre_index" if "pre" in raw_mode else "post_index"

    # Budget dialog
    params = run_budget_dialog(adapter)
    params.lora_steps = cfg.get("scout_initial_lora_steps", 500)
    params.lora_rank = cfg.get("scout_initial_lora_rank", 16)
    params.corpus_chunks = cfg.get("scout_initial_corpus_chunks", 200)
    params.faq_count = cfg.get("scout_initial_faq_count", 20)

    # Build candidate shortlist
    registry = load_registry(cfg.get("model_registry_path") or None)
    free_vram = gpu_info.get("free_vram_gb", 24.0)
    shortlist = get_candidate_shortlist(
        registry,
        free_vram_gb=free_vram,
        corpus_languages=cfg.get("corpus_languages", ["en"]),
        task_type=cfg.get("task_type", "factual_qa"),
        corpus_chunk_count=params.corpus_chunks,
    )

    if not shortlist:
        adapter.send(
            "No eligible models found for available VRAM. Cannot proceed."
        )
        return None

    names_preview = ", ".join(c["model_id"].split("/")[-1] for c in shortlist[:5])
    extra = f" and {len(shortlist) - 5} more" if len(shortlist) > 5 else ""
    adapter.send(f"Eligible models ({len(shortlist)}): {names_preview}{extra}")

    results: list[ExperimentResult] = []
    candidate_queue = list(shortlist)
    results_path = cfg.get("model_scout_results", "model_scout_results.tsv")

    while candidate_queue and not _budget_exhausted(params):
        candidate = candidate_queue.pop(0)
        family = candidate["spec"].get("family", candidate["model_id"])
        quant = "4bit" if candidate["spec"].get("_use_4bit") else "fp16"

        # Pre-experiment announcement
        adapter.send(
            f"\nNext: {candidate['model_id'].split('/')[-1]} ({quant}) | "
            f"steps={params.lora_steps} rank={params.lora_rank} "
            f"chunks={params.corpus_chunks} faqs={params.faq_count}\n"
            f"  Score: {candidate['score']:.2f} | "
            f"Type 'skip' to skip, 'stop' to end."
        )

        # Ask user to proceed (or intercept a command)
        user_cmd = adapter.ask(
            "Press Enter to proceed (or type a command):", options=None
        ).strip().lower()

        if user_cmd == "stop":
            break
        if user_cmd == "skip":
            adapter.send("Skipping. Moving to next candidate.")
            continue
        if user_cmd.startswith("try "):
            model_name = user_cmd[4:].strip()
            matching = [
                c for c in shortlist
                if model_name.lower() in c["model_id"].lower()
            ]
            if matching:
                candidate_queue.insert(0, matching[0])
                adapter.send(f"Queued {matching[0]['model_id'].split('/')[-1]} next.")
            else:
                adapter.send(
                    f"Model '{model_name}' not found in eligible list. Continuing."
                )
            continue
        if user_cmd == "more steps":
            params.lora_steps = min(
                params.lora_steps * 2,
                cfg.get("scout_max_lora_steps", 2000),
            )
            adapter.send(f"lora_steps increased to {params.lora_steps}.")
        if user_cmd == "more faqs":
            params.faq_count = min(
                params.faq_count + 10,
                cfg.get("scout_max_faq_count", 100),
            )
            adapter.send(f"faq_count increased to {params.faq_count}.")

        # Run the experiment
        result = run_single_experiment(
            candidate, faqs, params, cfg, adapter, mode, store
        )
        params.experiments_run += 1

        # Capture git commit for provenance
        try:
            result.git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], text=True
            ).strip()
        except Exception:
            result.git_commit = "unknown"

        # Persist and accumulate
        _write_result(results_path, result)
        results.append(result)

        # Report result
        best = max(results, key=lambda r: r.prs)
        adapter.send(
            f"  Result: PRS={result.prs:.3f} VRAM={result.vram_gb:.1f}GB "
            f"time={int(result.wall_seconds)}s status={result.status}\n"
            f"  Best so far: {best.model_id.split('/')[-1]} PRS={best.prs:.3f}"
        )

        # Apply parameter adjustments
        new_params, retry = apply_parameter_adjustments(
            result, params,
            max_steps=cfg.get("scout_max_lora_steps", 2000),
            max_faq=cfg.get("scout_max_faq_count", 100),
            max_chunks=cfg.get("scout_max_corpus_chunks", 2000),
        )
        if retry and params.experiments_run < params.budget_value:
            adapter.send(
                f"  Adjustment: {_describe_adjustment(params, new_params)}"
                f" — retrying same model."
            )
            params = new_params
            candidate_queue.insert(0, candidate)
        else:
            params = new_params

        # Three consecutive low-PRS warning
        if len(results) >= 3 and all(r.prs < 0.55 for r in results[-3:]):
            adapter.send(
                "Three consecutive models scored below 0.55 PRS. "
                "This may indicate a domain/language mismatch.\n"
                "  Is your corpus in a language other than what was configured? "
                "Any other context to share?"
            )
            adapter.ask("Your input (or press Enter to continue):", options=None)

    if not results:
        adapter.send("No experiments completed.")
        return None

    best = max(results, key=lambda r: r.prs)

    # Append recommendation comment to TSV
    with open(results_path, "a") as f:
        f.write(
            f"# RECOMMENDATION: {best.model_id} "
            f"({best.quantization}, rank={best.lora_rank})\n"
            f"# PRS={best.prs:.4f} | VRAM={best.vram_gb:.1f}GB | "
            f"Reasoning: Best PRS across {len(results)} experiments.\n"
        )

    adapter.send(
        f"\nModelScout complete. Recommendation:\n"
        f"   Model: {best.model_id}\n"
        f"   Quantization: {best.quantization} | LoRA rank: {best.lora_rank}\n"
        f"   PRS: {best.prs:.4f} | VRAM: {best.vram_gb:.1f}GB\n"
        f"   Results saved to: {results_path}"
    )

    return {
        "model_id": best.model_id,
        "quantization": best.quantization,
        "lora_rank": best.lora_rank,
        "prs": best.prs,
    }
