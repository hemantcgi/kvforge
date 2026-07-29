"""Generate on-policy distillation samples with confidence labels.

Usage:

    python3 tools/generate_on_policy_samples.py \
        --config student_datasource.json \
        --teacher-config teacher_datasource.json \
        --query-pool query_pool.json \
        --output on_policy_samples.json \
        --judge-model gpt-4o-mini

For each query in the pool:

1. Generate a student answer with the current (Sprint 2) student model.
2. Generate a teacher answer with the frozen Path A teacher model.
3. Score the student answer against the teacher answer with token-F1 + LLM judge.
4. Assign a confidence label (yes/no) for Sprint 2.5 confidence supervision.

The output JSON contains ``on_policy_pairs`` that can be fed directly to the
``--distill-pairs`` argument of ``pipeline/lora_trainer.py``.

GPU usage: two forward passes per query (student + teacher). Run on EC2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.model_loader as model_loader
import core.version as ver
from pipeline.distillation import generate_on_policy_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate on-policy distillation samples")
    parser.add_argument("--config", required=True, help="Student datasource config JSON")
    parser.add_argument("--teacher-config", required=True, help="Teacher datasource config JSON")
    parser.add_argument("--query-pool", required=True, help="Query pool JSON")
    parser.add_argument("--output", required=True, help="Output JSON")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sft-format", default="chat", choices=["chat", "bare"])
    args = parser.parse_args()

    with open(args.config) as f:
        student_cfg = json.load(f)
    with open(args.teacher_config) as f:
        teacher_cfg = json.load(f)
    with open(args.query_pool) as f:
        query_pool = json.load(f)

    if args.max_samples:
        query_pool = query_pool[:args.max_samples]

    # Load student model.
    model_loader.init(student_cfg)
    ver.init(student_cfg)
    student_model, student_tokenizer = model_loader.load(ver.load().get("checkpoint_path"))

    # Load teacher model.
    model_loader.init(teacher_cfg)
    ver.init(teacher_cfg)
    teacher_model, teacher_tokenizer = model_loader.load(ver.load().get("checkpoint_path"))

    training_cfg = student_cfg.get("addon_config", {}).get("training", {})
    sft_format = args.sft_format or training_cfg.get("sft_format", "chat")

    samples = generate_on_policy_samples(
        query_pool,
        student_model,
        student_tokenizer,
        teacher_model,
        teacher_tokenizer,
        student_cfg,
        sft_format=sft_format,
        judge_model=args.judge_model,
    )

    result = {"on_policy_pairs": samples}
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"✓ Wrote {args.output}")
    print(f"  On-policy samples: {len(samples)}")


if __name__ == "__main__":
    main()
