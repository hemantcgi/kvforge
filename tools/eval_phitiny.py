"""Evaluate Phi-tiny-MoE LoRA checkpoint on held-out eval set.

Usage:
    python3 tools/eval_phitiny.py \\
        --checkpoint lora_checkpoints/phitiny_uc4_1x_v1 \\
        --output results/pathb_diversity/uc4/phitiny_path_b_1x \\
        --eval-set examples/usecase4_bedrock_userguide/eval_heldout_v1.json
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import metrics as eval_metrics


def estimate_judge_cost(question: str, answer: str, ground_truth: str, model: str) -> float:
    prompt_len = len(question) + len(answer) + len(ground_truth) + 200
    output_len = 50
    prompt_tokens = prompt_len // 4
    output_tokens = output_len // 4
    rates = {"claude-sonnet-4-5-20250929": (3.00 / 1e6, 15.0 / 1e6)}
    in_rate, out_rate = rates.get(model, (3.00 / 1e6, 15.0 / 1e6))
    return prompt_tokens * in_rate + output_tokens * out_rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--eval-set", required=True)
    p.add_argument("--model", default="microsoft/Phi-tiny-MoE-instruct")
    args = p.parse_args()

    base = Path("/home/ubuntu/kvforge")
    checkpoint_dir = base / args.checkpoint
    eval_path = base / args.eval_set
    output_dir = base / args.output
    judge_model = "claude-sonnet-4-5-20250929"
    model_id = args.model

    with open(eval_path) as f:
        eval_data = json.load(f)
    items = eval_data["items"] if "items" in eval_data else eval_data
    print(f"Loaded {len(items)} eval items")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (BF16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM after load: {vram:.2f}GB")

    print(f"Loading LoRA from {checkpoint_dir}...")
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model.eval()
    model.config.use_cache = False

    print("Initializing judge client...")
    try:
        from anthropic import Anthropic
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("anthropic_api_key")
        judge_client = Anthropic(api_key=api_key) if api_key else None
        if not api_key:
            print("  WARNING: No Anthropic API key, using heuristic judge")
    except ImportError:
        judge_client = None
        print("  WARNING: anthropic not installed, using heuristic judge")

    records = []
    total_cost = 0.0

    for idx, item in enumerate(items):
        q = item["question"]
        gt = item["answer"]
        print(f"\n[{idx+1}/{len(items)}] Q: {q[:80]}...", flush=True)

        t0 = time.perf_counter()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": q},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False,
            )
        input_len = inputs["input_ids"].shape[1]
        answer_ids = outputs[0][input_len:]
        ans = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        latency = time.perf_counter() - t0
        print(f"   Answer: {ans[:120]}...")
        print(f"   Generated in {latency:.2f}s", flush=True)

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

    fkds_vals = [r["fkds"] for r in records]
    latencies = [r["latency_sec"] for r in records]
    arr = np.array(fkds_vals)
    sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0

    summary = {
        "config": {"eval_set": str(eval_path), "checkpoint": str(checkpoint_dir), "model": model_id},
        "modes": {
            "parametric": {
                "fkds": {"mean": round(float(np.mean(arr)), 4), "sem": round(sem, 4), "n": len(arr)},
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

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "parametric_records.json").write_text(json.dumps(records, indent=2))
    (output_dir / "experiment_metadata.json").write_text(json.dumps({
        "experiment_id": output_dir.name,
        "model_family": "phitiny",
        "model_id": model_id,
        "diversity_level": 5 if "5x" in str(checkpoint_dir) else 1,
        "seed": 42,
    }, indent=2))

    print(f"\n{'='*60}")
    print(f"Results written to {output_dir}")
    print(f"Mean fKDS: {summary['modes']['parametric']['fkds']['mean']:.4f} ± {summary['modes']['parametric']['fkds']['sem']:.4f}")
    print(f"Mean latency: {summary['modes']['parametric']['latency']['mean']:.4f}s")
    print(f"Total judge cost: ${total_cost:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
