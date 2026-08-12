"""Capacity gate: assess whether a model+dataset can beat Path A before Phase 2.

This tool is a thin CLI wrapper around ``pipeline.capacity_gate``, which reuses
the production text-RAG and parametric generation paths instead of duplicating
generation logic.

After the first LoRA training round, this script:
1. Evaluates Path A (text RAG) factual accuracy on a held-out set
2. Evaluates Path B (parametric) factual accuracy on the same held-out set
3. If parametric > text RAG: recommends proceeding to Phase 2/3
4. If parametric < text RAG: recommends staying on Phase 1 / trying a larger model

Usage:
    python3 tools/capacity_gate.py \\
        --config examples/usecase4_bedrock_userguide/config.json \\
        --checkpoint lora_checkpoints/uc4_1x_v1 \\
        --eval-set examples/usecase4_bedrock_userguide/eval_heldout_v1.json \\
        --output capacity_gate_result.json
"""
from __future__ import annotations

import argparse
import json
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.capacity_gate import _make_judge_client, run_capacity_gate


def main():
    p = argparse.ArgumentParser(description="Capacity gate: assess Path B readiness")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True, help="LoRA checkpoint path")
    p.add_argument("--eval-set", required=True, help="Held-out eval questions JSON")
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=30)
    p.add_argument("--judge-model", default="claude-sonnet-4-5-20250929")
    args = p.parse_args()

    # Load config
    raw_cfg = json.loads(Path(args.config).read_text())
    addon_config = raw_cfg.get("addon_config", {})
    cfg = {**raw_cfg}
    for section in ("indexing", "inference", "training", "background", "sync", "monitoring"):
        cfg.update(addon_config.get(section, {}))

    # Load questions
    eval_data = json.loads(Path(args.eval_set).read_text())
    questions = eval_data["items"] if "items" in eval_data else eval_data
    if args.max_samples:
        questions = questions[:args.max_samples]
    print(f"Loaded {len(questions)} eval questions")

    # Load model
    import core.model_loader as model_loader
    import core.version as ver
    from transformers import pipeline as hf_pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model_loader.init(cfg)
    ver.init(cfg)
    model, tokenizer = model_loader.load(args.checkpoint)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params/1e6:.1f}M")

    # Pipeline for parametric generation (matches PRS evaluator setup)
    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        return_full_text=False,
    )

    judge_client = _make_judge_client()
    result = run_capacity_gate(
        questions, cfg, pipe, tokenizer,
        judge_client=judge_client,
        judge_model=args.judge_model,
    )
    result["model"] = cfg.get("llm_model", "unknown")
    result["model_params_m"] = round(params / 1e6, 1)
    result["n_questions"] = len(questions)

    print(f"\n{'='*60}")
    print("Capacity Gate Result")
    print(f"{'='*60}")
    print(f"  Model: {result['model']} ({result['model_params_m']}M params)")
    print(f"  Path A (text RAG):   factual_accuracy = {result['path_a']['factual_accuracy_mean']:.4f}")
    print(f"  Path B (parametric): factual_accuracy = {result['path_b']['factual_accuracy_mean']:.4f}")
    print(f"  Beats Path A? {'YES' if result['beats_path_a'] else 'NO'} (margin={result['margin']:+.4f})")
    print(f"  Recommendation: {result['recommendation']}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\n✓ Wrote {args.output}")


if __name__ == "__main__":
    main()
