"""A/B evaluation: Sprint 2 (v1) vs Sprint 2.5 (v2_conf) adapters on held-out eval.

Usage (on EC2):

    python3 tools/eval_ab_adapters.py \
        --config examples/usecase4_bedrock_userguide/config_distill.json \
        --eval-set examples/usecase4_bedrock_userguide/eval_heldout_v1.json \
        --adapters v1=examples/usecase4_bedrock_userguide/lora_checkpoints/v1/ \
                   v2_conf=examples/usecase4_bedrock_userguide/lora_checkpoints/v2_conf/ \
        --output-dir examples/usecase4_bedrock_userguide/ab_results \
        --judge-model claude-fable-5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.model_loader as model_loader
from eval import metrics as eval_metrics
from pipeline.prs_evaluator import _generate_parametric
from pipeline.confidence_token import (
    strip_confidence_suffix,
    extract_confidence_probability,
)


def evaluate_adapter(
    cfg: dict,
    adapter_path: str,
    eval_items: list[dict],
    judge_model: str,
    judge_client: Any | None,
    sft_format: str = "chat",
    extract_confidence: bool = False,
) -> dict:
    model_loader.init(cfg)
    model, tokenizer = model_loader.load(adapter_path)

    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=256, do_sample=False, return_full_text=False,
    )

    records = []
    confidence_scores = []
    for idx, item in enumerate(eval_items):
        q = item["question"]
        gt = item["answer"]
        print(f"  [{idx+1}/{len(eval_items)}] {q[:80]}...", flush=True)

        t0 = time.perf_counter()
        raw_ans = _generate_parametric(q, pipe, tokenizer, sft_format=sft_format)
        latency = time.perf_counter() - t0

        ans = strip_confidence_suffix(raw_ans) if extract_confidence else raw_ans

        conf = None
        if extract_confidence:
            conf = extract_confidence_probability(
                q, raw_ans, pipe, tokenizer, sft_format=sft_format
            )
            if conf is not None and not np.isnan(conf):
                confidence_scores.append(conf)

        t1 = time.perf_counter()
        f1 = eval_metrics.token_f1(ans, gt)
        judge = eval_metrics.llm_judge(q, ans, gt, client=judge_client, model=judge_model)
        judge_lat = time.perf_counter() - t1
        fkds = 0.5 * f1 + 0.5 * float(judge["factually_correct"])

        records.append({
            "question": q,
            "ground_truth": gt,
            "answer": ans,
            "raw_answer": raw_ans,
            "fkds": round(fkds, 4),
            "f1": round(f1, 4),
            "judge_correct": int(judge["factually_correct"]),
            "judge_rationale": judge.get("rationale", ""),
            "latency_sec": round(latency, 4),
            "judge_latency_sec": round(judge_lat, 4),
            "confidence_p_yes": round(conf, 4) if conf is not None else None,
        })

    fkds_vals = [r["fkds"] for r in records]
    lat_vals = [r["latency_sec"] for r in records]

    return {
        "adapter": adapter_path.rstrip("/").split("/")[-1],
        "n": len(records),
        "fkds_mean": round(float(np.mean(fkds_vals)), 4),
        "fkds_sem": round(float(np.std(fkds_vals, ddof=1) / np.sqrt(len(fkds_vals))), 4) if len(fkds_vals) > 1 else 0.0,
        "latency_mean": round(float(np.mean(lat_vals)), 4),
        "latency_p50": round(float(np.percentile(lat_vals, 50)), 4),
        "latency_p95": round(float(np.percentile(lat_vals, 95)), 4),
        "confidence_mean_p_yes": round(float(np.mean(confidence_scores)), 4) if confidence_scores else None,
        "confidence_n": len(confidence_scores) if confidence_scores else 0,
        "records": records,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="A/B evaluate distillation adapters")
    p.add_argument("--config", required=True)
    p.add_argument("--eval-set", required=True)
    p.add_argument("--adapters", nargs="+", required=True,
                   help="label=path pairs, e.g. v1=checkpoints/v1/")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--judge-provider", default="openai")
    p.add_argument("--judge-api-key", default="")
    p.add_argument("--extract-confidence", action="store_true",
                   help="Extract confidence token from v2_conf adapter")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--accept-candidate", type=str, default=None,
                   help="Label of adapter to stage as deployment candidate")
    p.add_argument("--accept-deploy", action="store_true",
                   help="Accept the staged candidate for deployment if it passes the gate")
    p.add_argument("--judge-noise", type=float, default=0.05,
                   help="Estimated judge noise for acceptance threshold")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    with open(args.eval_set) as f:
        eval_data = json.load(f)
    items = eval_data.get("items", eval_data)
    if args.max_samples:
        items = items[:args.max_samples]

    import os
    api_key = args.judge_api_key or os.environ.get(
        {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(
            args.judge_provider, "OPENAI_API_KEY"
        ), ""
    )
    judge_client = None
    if api_key and args.judge_provider == "anthropic":
        import anthropic
        judge_client = anthropic.Anthropic(api_key=api_key)
    elif api_key:
        import openai
        judge_client = openai.OpenAI(api_key=api_key)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    for spec in args.adapters:
        label, adapter_path = spec.split("=", 1)
        extract = args.extract_confidence and "conf" in label
        print(f"\n=== Evaluating {label} ({adapter_path}) ===", flush=True)
        r = evaluate_adapter(
            cfg, adapter_path, items, args.judge_model, judge_client,
            extract_confidence=extract,
        )
        results[label] = r

        out_path = Path(args.output_dir) / f"{label}_results.json"
        out_path.write_text(json.dumps(r, indent=2, ensure_ascii=False))
        print(f"  Wrote {out_path}")
        print(f"  fKDS: {r['fkds_mean']:.4f} ± {r['fkds_sem']:.4f}  "
              f"latency: {r['latency_mean']:.2f}s  n={r['n']}", flush=True)
        if r["confidence_mean_p_yes"] is not None:
            print(f"  Confidence P(yes): {r['confidence_mean_p_yes']:.4f} (n={r['confidence_n']})", flush=True)

    # Comparison
    labels = sorted(results.keys())
    if len(labels) >= 2:
        a, b = labels[0], labels[1]
        delta = results[b]["fkds_mean"] - results[a]["fkds_mean"]
        rel_delta = delta / (results[a]["fkds_mean"] + 1e-10) * 100
        comp = {
            "adapter_a": a,
            "adapter_b": b,
            "fkds_a": results[a]["fkds_mean"],
            "fkds_b": results[b]["fkds_mean"],
            "abs_delta": round(delta, 4),
            "rel_delta_pct": round(rel_delta, 2),
            "within_1pct": abs(rel_delta) <= 1.0,
        }
        cmp_path = Path(args.output_dir) / "comparison.json"
        cmp_path.write_text(json.dumps(comp, indent=2, ensure_ascii=False))
        print(f"\n=== Comparison ===", flush=True)
        print(f"  {a}: fKDS={results[a]['fkds_mean']:.4f}  {b}: fKDS={results[b]['fkds_mean']:.4f}", flush=True)
        print(f"  Delta: {delta:+.4f}  ({rel_delta:+.2f}%)  within_1%: {comp['within_1pct']}", flush=True)

    # ── Sprint 3: acceptance gate ──────────────────────────────────────────
    if args.accept_candidate:
        cand = results.get(args.accept_candidate)
        if cand is None:
            print(f"  ⚠  Candidate {args.accept_candidate} not in results.", flush=True)
        else:
            from core import adapter_acceptance
            adapter_acceptance.stage_candidate(
                cfg,
                candidate_path=args.adapters[
                    [s.split("=")[0] for s in args.adapters].index(args.accept_candidate)
                ].split("=", 1)[1],
                fkds_on_heldout=cand["fkds_mean"],
                fkds_sem=cand["fkds_sem"],
                judge_noise=args.judge_noise,
                latency_mean=cand.get("latency_mean"),
            )
            print(f"  ✓ Staged {args.accept_candidate} as deployment candidate.", flush=True)

            if args.accept_deploy:
                accepted, report = adapter_acceptance.accept_adapter(cfg)
                status = "ACCEPTED" if accepted else "REJECTED"
                print(f"  Gate: {status} — {report['reason']}", flush=True)
                if accepted:
                    print(f"  Deployed adapter → {report.get('candidate_fkds', '?')}"
                          f" (was {report.get('deployed_fkds', '?')})", flush=True)


if __name__ == "__main__":
    main()
