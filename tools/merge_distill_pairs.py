"""Merge teacher-pairs and on-policy-pairs JSON into a single distill-pairs file.

Usage:

    python3 tools/merge_distill_pairs.py \
        --teacher-pairs teacher_pairs.json \
        --on-policy on_policy_samples.json \
        --output distill_pairs.json

Produces a single JSON with ``teacher_pairs`` and ``on_policy_pairs`` keys
that can be passed directly to ``lora_trainer.py --distill-pairs``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge(teacher_path: str, on_policy_path: str) -> dict:
    teacher = json.loads(Path(teacher_path).read_text())
    on_policy = json.loads(Path(on_policy_path).read_text())
    return {
        "teacher_pairs": teacher.get("teacher_pairs", []),
        "on_policy_pairs": on_policy.get("on_policy_pairs", []),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Merge teacher + on-policy pairs for distillation")
    p.add_argument("--teacher-pairs", required=True, help="Output of run_teacher_pipeline.py")
    p.add_argument("--on-policy", required=True, help="Output of generate_on_policy_samples.py")
    p.add_argument("--output", required=True, help="Merged distill-pairs JSON")
    args = p.parse_args()

    result = merge(args.teacher_pairs, args.on_policy)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))

    n_teacher = len(result["teacher_pairs"])
    n_on_policy = len(result["on_policy_pairs"])
    print(f"✓ Wrote {args.output}")
    print(f"  teacher_pairs: {n_teacher}")
    print(f"  on_policy_pairs: {n_on_policy}")
    print(f"  total: {n_teacher + n_on_policy}")


if __name__ == "__main__":
    main()
