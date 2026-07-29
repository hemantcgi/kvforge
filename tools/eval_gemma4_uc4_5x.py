"""Evaluate Gemma4 UC4 LoRA checkpoint on held-out eval set.

Loads the Gemma4 model + LoRA adapter, generates answers for 89 held-out
questions, computes token-F1 and LLM-judge factual correctness, then outputs
fKDS metrics.

Usage:
    python3 tools/eval_gemma4_uc4_5x.py --checkpoint lora_checkpoints/gemma4_uc4_1x_v1 --output results/pathb_diversity/uc4/gemma_path_b_1x
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
from torch import nn
from transformers import AutoProcessor, AutoModelForMultimodalLM
from peft import PeftModel


def _unwrap_clippable_linears(model):
    n = 0
    for name, mod in list(model.model.named_modules()):
        if ("vision_tower" in name or "audio_tower" in name) and "Gemma4ClippableLinear" in type(mod).__name__:
            parent_path = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]
            parent = model.model.get_submodule(parent_path)
            setattr(parent, attr_name, mod.linear)
            n += 1
    print(f"Unwrapped {n} ClippableLinear modules in vision/audio towers")
    return model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import metrics as eval_metrics


def estimate_judge_cost(question: str, answer: str, ground_truth: str, model: str) -> float:
    prompt_len = len(question) + len(answer) + len(ground_truth) + 200
    output_len = 50
    prompt_tokens = prompt_len // 4
    output_tokens = output_len // 4
    rates = {
        "claude-sonnet-4-5-20250929": (3.00 / 1e6, 15.0 / 1e6),
    }
    in_rate, out_rate = rates.get(model, (3.00 / 1e6, 15.0 / 1e6))
    return prompt_tokens * in_rate + output_tokens * out_rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="lora_checkpoints/gemma4_uc4_5x_v1")
    p.add_argument("--output", default="results/pathb_diversity/uc4/gemma_path_b_5x")
    p.add_argument("--eval-set", default="examples/usecase4_bedrock_userguide/eval_heldout_v1.json")
    args = p.parse_args()

    base = Path("/home/ubuntu/kvforge")
    checkpoint_dir = base / args.checkpoint
    eval_path = base / args.eval_set
    output_dir = base / args.output
    judge_model = "claude-sonnet-4-5-20250929"

    # Load eval data
    with open(eval_path) as f:
        eval_data = json.load(f)
    items = eval_data["items"]
    print(f"Loaded {len(items)} eval items")

    # Load Gemma model (same as train_gemma4.py)
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")
    print("Loading model (BF16)...")
    model = AutoModelForMultimodalLM.from_pretrained(
        "google/gemma-4-E2B-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM after load: {vram:.2f}GB")

    # Unwrap ClippableLinear (same as training) so PEFT can attach LoRA to nn.Linear
    model = _unwrap_clippable_linears(model)

    # Load LoRA adapter
    print(f"Loading LoRA adapter from {checkpoint_dir}...")
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model.eval()
    model.config.use_cache = False

    # Initialize judge client
    print("Initializing judge client...")
    try:
        from anthropic import Anthropic
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("anthropic_api_key")
        if api_key:
            judge_client = Anthropic(api_key=api_key)
            print("  Using Anthropic judge client")
        else:
            judge_client = None
            print("  WARNING: No Anthropic API key found, using heuristic judge fallback")
    except ImportError:
        judge_client = None
        print("  WARNING: anthropic package not installed, using heuristic judge fallback")

    records = []
    total_cost = 0.0

    for idx, item in enumerate(items):
        q = item["question"]
        gt = item["answer"]

        print(f"\n[{idx+1}/{len(items)}] Q: {q[:80]}...", flush=True)

        # Generate answer using Gemma's chat template
        t0 = time.perf_counter()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": q}]},
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        # Decode, strip the prompt
        input_len = inputs["input_ids"].shape[1]
        answer_ids = outputs[0][input_len:]
        ans = processor.decode(answer_ids, skip_special_tokens=True).strip()
        latency = time.perf_counter() - t0
        print(f"   Answer: {ans[:120]}...")
        print(f"   Generated in {latency:.2f}s", flush=True)

        # Evaluate
        t1 = time.perf_counter()
        f1 = eval_metrics.token_f1(ans, gt)
        judge = eval_metrics.llm_judge(q, ans, gt, client=judge_client, model=judge_model)
        judge_latency = time.perf_counter() - t1
        fkds = 0.5 * f1 + 0.5 * float(judge["factually_correct"])
        print(f"   Judge: {int(judge['factually_correct'])} | F1: {f1:.3f} | fKDS: {fkds:.3f} ({judge_latency:.2f}s)", flush=True)

        cost = estimate_judge_cost(q, ans, gt, judge_model)
        total_cost += cost

        records.append({
            "question": q,
            "ground_truth": gt,
            "answer": ans,
            "mode": "parametric",
            "fkds": round(fkds, 4),
            "f1": round(f1, 4),
            "judge_correct": int(judge["factually_correct"]),
            "judge_rationale": judge.get("rationale", ""),
            "latency_sec": round(latency, 4),
            "cost_usd": round(cost, 6),
        })

    # Aggregate results
    fkds_vals = [r["fkds"] for r in records]
    latencies = [r["latency_sec"] for r in records]
    arr = np.array(fkds_vals)
    sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0

    summary = {
        "config": {
            "eval_set": str(eval_path),
            "checkpoint": str(checkpoint_dir),
        },
        "modes": {
            "parametric": {
                "fkds": {
                    "mean": round(float(np.mean(arr)), 4),
                    "sem": round(sem, 4),
                    "n": len(arr),
                },
                "latency": {
                    "mean": round(float(np.mean(latencies)), 4),
                    "p50": round(float(np.percentile(latencies, 50)), 4),
                    "p95": round(float(np.percentile(latencies, 95)), 4),
                    "p99": round(float(np.percentile(latencies, 99)), 4),
                    "n": len(latencies),
                },
                "cost_usd_total": round(total_cost, 6),
            },
        },
    }

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "parametric_records.json", "w") as f:
        json.dump(records, f, indent=2)
    with open(output_dir / "experiment_metadata.json", "w") as f:
        json.dump({
            "experiment_id": "uc4_gemma_path_b_5x",
            "dataset": "uc4",
            "model_family": "gemma4",
            "model_id": "google/gemma-4-E2B-it",
            "diversity_level": 5,
            "seed": 42,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results written to {output_dir}")
    print(f"Mean fKDS: {summary['modes']['parametric']['fkds']['mean']:.4f} ± {summary['modes']['parametric']['fkds']['sem']:.4f}")
    print(f"Mean latency: {summary['modes']['parametric']['latency']['mean']:.4f}s")
    print(f"Total judge cost: ${total_cost:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
