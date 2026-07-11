"""Auto-configuration for LoRA training hyperparameters.

Derives recommendations from corpus signals already available at training time:
  - n_faqs         : number of FAQ pairs in faqs.json
  - cluster_count  : topic diversity from clusters.json (more = higher rank needed)
  - vram_gb        : detected GPU VRAM (constrains batch size)
  - prs_history    : past PRS scores (drive rank escalation on plateau)
  - lora_version   : current round (first vs. refinement run)

Each recommendation is returned as {"value": <v>, "reason": "<why>"} so the
Studio UI can display the reasoning alongside editable fields.
"""

import json
import math
from pathlib import Path


def _cluster_count(cfg: dict) -> int:
    training = cfg.get("addon_config", {}).get("training", cfg)
    checkpoint_dir = training.get("checkpoint_dir", "")
    if checkpoint_dir:
        cluster_file = Path(checkpoint_dir).parent / "clusters.json"
        if cluster_file.exists():
            try:
                data = json.loads(cluster_file.read_text())
                count = len(data.get("centroids", []))
                if count > 0:
                    return count
            except Exception:
                pass
    return 10  # safe default


def _vram_gb() -> float:
    try:
        from studio.gpu_monitor import get_free_gpus
        gpus = get_free_gpus()
        if gpus:
            return gpus[0].get("memory_total_mb", 24576) / 1024
    except Exception:
        pass
    return 24.0  # A10G default


def recommend(cfg: dict, n_faqs: int, prs_history: list, lora_version: int) -> dict:
    """Return LoRA hyperparameter recommendations with per-field reasoning.

    Returns a dict where each key maps to {"value": <v>, "reason": "<why>"}.
    A ``_meta`` key carries the raw signals used so the UI can show them.
    """
    vram = _vram_gb()
    clusters = _cluster_count(cfg)
    indexing = cfg.get("addon_config", {}).get("indexing", cfg)
    seq_len = min(int(indexing.get("chunk_size", 512)), 1024)

    # ── Batch size — largest power-of-2 that fits in free VRAM ────────────────
    model_vram = 4.0   # Llama-3B 4-bit
    per_sample = (seq_len / 512) * 0.4
    free = max(2.0, vram - model_vram)
    raw_batch = int(free / per_sample)
    batch = max(1, 2 ** int(math.log2(max(1, raw_batch))))
    batch = min(batch, 32)
    batch_reason = (
        f"{vram:.0f} GB total − {model_vram} GB model = {free:.0f} GB free; "
        f"{per_sample:.2f} GB/sample at seq_len {seq_len} → max {raw_batch}, "
        f"rounded to nearest power-of-2"
    )

    # ── Epochs — target ~2–3k gradient steps regardless of dataset size ────────
    steps_per_epoch = max(1, math.ceil(n_faqs / batch))
    target_steps = max(500, min(8000, n_faqs * 2))
    epochs = max(5, min(30, math.ceil(target_steps / steps_per_epoch)))
    epochs_reason = (
        f"{n_faqs} FAQs ÷ batch {batch} = {steps_per_epoch} steps/epoch; "
        f"targeting ~{target_steps} total steps → {epochs} epochs"
    )

    # ── LoRA rank — driven by topic diversity + PRS plateau detection ──────────
    if clusters < 5:
        base_rank, rank_base_why = 4, f"{clusters} topic clusters (low diversity)"
    elif clusters < 15:
        base_rank, rank_base_why = 8, f"{clusters} topic clusters (moderate diversity)"
    elif clusters < 30:
        base_rank, rank_base_why = 16, f"{clusters} topic clusters (high diversity)"
    else:
        base_rank, rank_base_why = 32, f"{clusters} topic clusters (very high diversity)"

    rank = base_rank
    rank_prs_note = ""
    if lora_version > 1 and prs_history:
        last_prs_entry = prs_history[-1]
        last_prs = (last_prs_entry.get("prs", 0)
                    if isinstance(last_prs_entry, dict) else float(last_prs_entry))
        if last_prs < 0.50:
            rank = min(64, base_rank * 4)
            rank_prs_note = f"; PRS {last_prs:.3f} < 0.50 → quadrupled rank to {rank}"
        elif last_prs < 0.70:
            rank = min(32, base_rank * 2)
            rank_prs_note = f"; PRS {last_prs:.3f} plateau → doubled rank to {rank}"
        else:
            rank_prs_note = f"; PRS {last_prs:.3f} on track → holding rank"

    rank_reason = rank_base_why + rank_prs_note

    # ── Alpha = 2×rank keeps effective LR scale constant ──────────────────────
    alpha = rank * 2
    alpha_reason = f"Always 2×rank ({rank}) — effective scale = alpha/rank = 2.0; no need to tune separately"

    # ── Learning rate — scale down as rank grows ───────────────────────────────
    lr = round(2e-4 / math.sqrt(rank / 8), 6)
    lr_reason = (
        f"Base 2e-4 ÷ √(rank/8) = √{rank/8:.2f} → {lr}; "
        f"higher rank needs lower LR to stay stable"
    )

    return {
        "lora_epochs":      {"value": epochs,      "reason": epochs_reason},
        "lora_rank":        {"value": rank,         "reason": rank_reason},
        "lora_alpha":       {"value": alpha,        "reason": alpha_reason},
        "lora_lr":          {"value": lr,           "reason": lr_reason},
        "train_batch_size": {"value": batch,        "reason": batch_reason},
        "_meta": {
            "n_faqs":        n_faqs,
            "cluster_count": clusters,
            "vram_gb":       round(vram, 1),
            "seq_len":       seq_len,
            "lora_version":  lora_version,
        },
    }
