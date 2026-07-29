"""Train-on-drop trigger for KVForge Sprint 4.

Monitors held-out eval fKDS across rounds and automatically triggers
distillation when quality degrades.

Usage::

    python3 -m pipeline.train_on_drop \
        --config examples/usecase4_bedrock_userguide/config_distill.json \
        --eval-set examples/usecase4_bedrock_userguide/eval_heldout_v1.json \
        --history-file examples/usecase4_bedrock_userguide/fkds_history.json \
        --min-delta -0.02 \
        --consecutive-rounds 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def load_history(path: str) -> list[dict]:
    """Load fKDS history, returns empty list if file missing or corrupt."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, list) else data.get("history", [])
    except (json.JSONDecodeError, KeyError):
        return []


def save_history(path: str, history: list[dict]) -> None:
    Path(path).write_text(json.dumps({"history": history}, indent=2, ensure_ascii=False))


def record_eval(
    history_path: str,
    adapter_path: str,
    fkds: float,
    fkds_sem: float,
    round_num: int,
    notes: dict | None = None,
) -> None:
    """Record an eval result into the fKDS history."""
    history = load_history(history_path)
    entry = {
        "round": round_num,
        "adapter": adapter_path,
        "fkds": round(fkds, 4),
        "fkds_sem": round(fkds_sem, 4),
        "timestamp": time.time(),
        "notes": notes or {},
    }
    history.append(entry)
    save_history(history_path, history)


def should_retrain(
    history_path: str,
    min_delta: float = -0.02,
    consecutive_rounds: int = 2,
) -> tuple[bool, dict]:
    """Check whether the fKDS trend triggers retraining.

    Triggers when the last ``consecutive_rounds`` rounds show fKDS values
    that are all at least ``min_delta`` below the best-ever fKDS. Negative
    ``min_delta`` means a drop of that magnitude.

    Returns:
        ``(should_retrain, report)`` where report explains the decision.
    """
    history = load_history(history_path)
    if len(history) < consecutive_rounds + 1:
        return False, {
            "triggered": False,
            "reason": f"need at least {consecutive_rounds + 1} rounds, have {len(history)}",
            "history_len": len(history),
        }

    fkds_vals = [h["fkds"] for h in history]
    best_fkds = max(fkds_vals)
    recent = fkds_vals[-consecutive_rounds:]
    dips = [v for v in recent if v - best_fkds <= min_delta]

    if len(dips) >= consecutive_rounds:
        return True, {
            "triggered": True,
            "reason": (
                f"last {consecutive_rounds} rounds ({', '.join(f'{v:.4f}' for v in recent)}) "
                f"all ≤ {best_fkds:.4f} + {min_delta:.4f} (best was {best_fkds:.4f})"
            ),
            "best_fkds": round(best_fkds, 4),
            "recent": [round(v, 4) for v in recent],
            "history_len": len(history),
        }

    return False, {
        "triggered": False,
        "reason": (
            f"last {consecutive_rounds} rounds not all degraded below "
            f"best={best_fkds:.4f} + delta={min_delta:.4f}"
        ),
        "best_fkds": round(best_fkds, 4),
        "recent": [round(v, 4) for v in recent],
        "history_len": len(history),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train-on-drop monitor")
    p.add_argument("--config", required=True, help="Student datasource config")
    p.add_argument("--eval-set", required=True, help="Held-out eval set JSON")
    p.add_argument("--history-file", required=True, help="JSON file for fKDS history")
    p.add_argument("--min-delta", type=float, default=-0.02,
                   help="Minimum fKDS drop to trigger retrain (negative = drop)")
    p.add_argument("--consecutive-rounds", type=int, default=2,
                   help="Consecutive degraded rounds to trigger")
    p.add_argument("--record", action="store_true",
                   help="Record current eval result instead of checking")
    p.add_argument("--fkds", type=float, help="fKDS value to record")
    p.add_argument("--fkds-sem", type=float, default=0.0)
    p.add_argument("--adapter-path", help="Adapter path for record")
    p.add_argument("--round", type=int, default=1)
    args = p.parse_args()

    if args.record:
        if args.fkds is None or args.adapter_path is None:
            p.error("--record requires --fkds and --adapter-path")
        record_eval(
            args.history_file, args.adapter_path, args.fkds,
            args.fkds_sem, args.round,
        )
        print(f"Recorded round {args.round}: fKDS={args.fkds:.4f}", flush=True)
        return

    triggered, report = should_retrain(
        args.history_file, args.min_delta, args.consecutive_rounds,
    )
    print(json.dumps(report, indent=2))
    if triggered:
        print("\n⚠️  TRAIN-ON-DROP TRIGGERED — queue distillation round.", flush=True)
        sys.exit(1)
    else:
        print("✅ No retrain needed.", flush=True)


if __name__ == "__main__":
    main()
