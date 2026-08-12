"""Measure Path A, Path B, and KV-injection baselines on a frozen held-out eval set.

Produces:

* per-question JSON with answers, fKDS, latency, and cost estimates
* aggregate report with mean fKDS, latency percentiles, and total cost

Modes:

* ``text_rag``   — Path A text-in-context fallback
* ``kv_meanpool`` — Path A KV-injection (per-chunk mean-pooled KV)
* ``parametric``  — Path B (fine-tuned model answering from weights)

Usage::

    python3 -m tools.measure_baseline_fkds \
        --config examples/usecase4_bedrock_userguide/config.json \
        --eval-set examples/usecase4_bedrock_userguide/eval_heldout_v1.json \
        --output-dir examples/usecase4_bedrock_userguide/baseline_results \
        --modes text_rag kv_meanpool parametric \
        --judge-model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


def estimate_judge_cost(
    question: str,
    answer: str,
    ground_truth: str,
    model: str,
) -> float:
    """Rough API cost per judge call in USD.

    Prices are hard-coded list prices (2024-2025). The estimate is a lower
    bound because it only covers the judge, not the generation model.
    """
    # Approximate token count: 1 token ≈ 4 characters.
    prompt_len = len(question) + len(answer) + len(ground_truth) + 200
    output_len = 50
    prompt_tokens = prompt_len // 4
    output_tokens = output_len // 4

    rates = {
        "gpt-4o-mini": (0.15 / 1e6, 0.60 / 1e6),
        "gpt-4o": (2.50 / 1e6, 10.0 / 1e6),
        "claude-3-haiku": (0.25 / 1e6, 1.25 / 1e6),
        "claude-3-sonnet": (3.00 / 1e6, 15.0 / 1e6),
        "gemini-1.5-flash": (0.075 / 1e6, 0.30 / 1e6),
        "gemini-2.5-flash": (0.075 / 1e6, 0.30 / 1e6),
    }
    in_rate, out_rate = rates.get(model, (0.15 / 1e6, 0.60 / 1e6))
    return prompt_tokens * in_rate + output_tokens * out_rate


def estimate_embedding_cost(text: str) -> float:
    """Return 0.0 because KVForge uses local fastembed by default.

    Kept as a hook in case an OpenAI/COHERE paid embedder is used.
    """
    return 0.0


def summarize_latency(latencies: list[float]) -> dict[str, float]:
    """Return p50/p95/p99 and mean latencies in seconds."""
    arr = np.array(latencies)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
        "n": len(arr),
    }


def summarize_factual_accuracy(scores: list[float]) -> dict[str, float]:
    """Return mean and SEM of raw factual-accuracy scores.

    These scores are 0.5 * token-F1 + 0.5 * LLM-judge correctness, not the
    corpus-level fKDS blend that also incorporates consistency KDS. The name is
    intentionally ``factual_accuracy`` to avoid conflating raw factual scores
    with the fKDS metric used by the KV-injection gate.
    """
    arr = np.array(scores)
    sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "mean": round(float(np.mean(arr)), 4),
        "sem": round(sem, 4),
        "n": len(arr),
    }


def run_mode(
    mode: str,
    eval_items: list[dict],
    cfg: dict,
    judge_model: str,
    judge_client: Any | None,
) -> dict[str, Any]:
    """Run one inference mode and return per-question + aggregate results."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import core.model_loader as model_loader
    import core.version as ver
    from eval import metrics as eval_metrics
    from pipeline.kv_inference import answer_with_mode
    from pipeline.prs_evaluator import _generate_parametric

    ver.init(cfg)
    model_loader.init(cfg)

    # Model load for parametric/Path B.
    if mode == "parametric":
        lora_ckpt = ver.load().get("checkpoint_path")
        model, tokenizer = model_loader.load(lora_ckpt)
        from transformers import pipeline as hf_pipeline

        pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer,
                           max_new_tokens=256, do_sample=False, return_full_text=False)
    else:
        model, tokenizer = None, None

    records = []
    print(f"   Starting loop over {len(eval_items)} items for mode={mode}", flush=True)
    for idx, item in enumerate(eval_items):
        q = item["question"]
        gt = item["answer"]

        print(f"   [{idx+1}/{len(eval_items)}] {mode}: {q[:80]}...", flush=True)
        t0 = time.perf_counter()
        if mode == "parametric":
            ans = _generate_parametric(q, pipe, tokenizer, sft_format="chat")
        else:
            ans, used_mode = answer_with_mode(q, cfg, force_mode=mode)
            if not ans:
                ans = ""
        latency = time.perf_counter() - t0
        print(f"       generated in {latency:.2f}s", flush=True)

        t1 = time.perf_counter()
        f1 = eval_metrics.token_f1(ans, gt)
        judge = eval_metrics.llm_judge(q, ans, gt, client=judge_client, model=judge_model)
        judge_latency = time.perf_counter() - t1
        factual_accuracy = 0.5 * f1 + 0.5 * float(judge["factually_correct"])
        print(f"       judge={int(judge['factually_correct'])} f1={f1:.3f} factual_accuracy={factual_accuracy:.3f} ({judge_latency:.2f}s)", flush=True)

        cost = estimate_judge_cost(q, ans, gt, judge_model) + estimate_embedding_cost(ans)

        records.append({
            "question": q,
            "ground_truth": gt,
            "answer": ans,
            "mode": mode,
            "factual_accuracy": round(factual_accuracy, 4),
            "f1": round(f1, 4),
            "judge_correct": int(judge["factually_correct"]),
            "judge_rationale": judge.get("rationale", ""),
            "latency_sec": round(latency, 4),
            "cost_usd": round(cost, 6),
        })

    fa_list = [r["factual_accuracy"] for r in records]
    latency_list = [r["latency_sec"] for r in records]
    cost_total = sum(r["cost_usd"] for r in records)

    return {
        "mode": mode,
        "factual_accuracy": summarize_factual_accuracy(fa_list),
        "latency": summarize_latency(latency_list),
        "cost_usd_total": round(cost_total, 4),
        "records": records,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--eval-set", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--modes", nargs="+", default=["text_rag", "kv_meanpool", "parametric"])
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--judge-provider", default="openai",
                   help="openai, anthropic, or gemini")
    p.add_argument("--judge-api-key", default="")
    p.add_argument("--recompute-ratio", type=float, default=None,
                   help="Override recompute_ratio in config for partial KV recompute sweeps.")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    if args.recompute_ratio is not None:
        cfg["recompute_ratio"] = args.recompute_ratio
    with open(args.eval_set) as f:
        eval_data = json.load(f)
    eval_items = eval_data.get("items", eval_data)

    # Lazy client setup only if an API key is provided.
    judge_client = None
    api_key = args.judge_api_key or os.environ.get(
        {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}.get(
            args.judge_provider, "OPENAI_API_KEY"
        ),
        "",
    )
    if api_key:
        if args.judge_provider == "openai":
            import openai
            judge_client = openai.OpenAI(api_key=api_key)
        elif args.judge_provider == "anthropic":
            import anthropic
            judge_client = anthropic.Anthropic(api_key=api_key)
        elif args.judge_provider == "gemini":
            # Gemini client uses a different API shape; llm_judge handles it via messages fallback.
            judge_client = None
        else:
            raise ValueError(f"Unknown judge provider: {args.judge_provider}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "config": cfg,
        "eval_set": args.eval_set,
        "judge_model": args.judge_model,
        "modes": {},
    }

    for mode in args.modes:
        print(f"\n🔬 Running mode: {mode}")
        result = run_mode(mode, eval_items, cfg, args.judge_model, judge_client)
        report["modes"][mode] = {
            "factual_accuracy": result["factual_accuracy"],
            "latency": result["latency"],
            "cost_usd_total": result["cost_usd_total"],
        }
        with open(out_dir / f"{mode}_records.json", "w") as f:
            json.dump(result["records"], f, indent=2)
        print(f"   factual_accuracy={result['factual_accuracy']['mean']:.3f} ± {result['factual_accuracy']['sem']:.3f}")
        print(f"   p50 latency={result['latency']['p50']:.3f}s")
        print(f"   total cost=${result['cost_usd_total']:.4f}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Summary written to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

