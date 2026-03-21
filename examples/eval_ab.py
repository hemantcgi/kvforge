#!/usr/bin/env python3
"""
eval_ab.py — A/B evaluation: SmartQdrant (Model A) vs Gemini (Model B).

Runs every question in a ground-truth Q&A JSON file through the SmartQdrant
dashboard, computes per-record and aggregate metrics, and prints a comparison
report to the console.

Metrics
-------
  semantic_sim   cosine similarity between answer embedding and ground-truth
                 embedding (fastembed, CPU, no GPU required)
  token_f1       token-level F1 between answer tokens and ground-truth tokens
  latency_ms     wall-clock time reported by the dashboard

Optional (requires: pip install ragas datasets langchain-google-genai)
  ragas_similarity     embedding-based similarity (AnswerSimilarity)
  ragas_correctness    LLM-judged factual correctness (AnswerCorrectness)
                       — needs --gemini-key or GEMINI_API_KEY env var

Usage
-----
  # Basic (calls dashboard at localhost:8080)
  python3 eval_ab.py --faq bedrock_50_faqs.json

  # Point at EC2 dashboard
  python3 eval_ab.py --faq bedrock_50_faqs.json \\
      --dashboard http://100.48.17.48:8080

  # Save results + enable RAGAS with Gemini
  python3 eval_ab.py --faq bedrock_50_faqs.json \\
      --dashboard http://100.48.17.48:8080 \\
      --out results.json \\
      --gemini-key YOUR_KEY

  # Verbose (show full answers per record)
  python3 eval_ab.py --faq bedrock_50_faqs.json --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import numpy as np

# ── Optional RAGAS ────────────────────────────────────────────────────────────
try:
    from datasets import Dataset as HFDataset
    from ragas import evaluate as ragas_evaluate
    try:  # ragas >= 0.2 moved metrics to .collections
        from ragas.metrics.collections import answer_correctness, answer_similarity
    except ImportError:
        from ragas.metrics import answer_correctness, answer_similarity  # type: ignore[no-redef]
    _HAS_RAGAS = True
except ImportError:
    _HAS_RAGAS = False

# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="A/B evaluation: SmartQdrant vs Gemini on a ground-truth FAQ dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--faq",      required=True,
                   help="Path to FAQ JSON: [{question, answer}, …]")
    p.add_argument("--dashboard", default="http://localhost:8080",
                   help="SmartQdrant dashboard base URL")
    p.add_argument("--out",      default=None,
                   help="Save per-record results to this JSON file")
    p.add_argument("--delay",    type=float, default=1.5,
                   help="Seconds to wait between requests (avoids overloading the LLM)")
    p.add_argument("--limit",    type=int,   default=None,
                   help="Evaluate only the first N records (useful for quick smoke-tests)")

    # Model A params
    p.add_argument("--a-top-k",           type=int,   default=3)
    p.add_argument("--a-max-new-tokens",  type=int,   default=128)
    p.add_argument("--a-temperature",     type=float, default=0.7)
    p.add_argument("--a-top-p",           type=float, default=0.9)
    p.add_argument("--a-repetition-penalty", type=float, default=1.2)

    # Model B params
    p.add_argument("--b-top-k",               type=int,   default=5)
    p.add_argument("--b-max-output-tokens",   type=int,   default=1024)
    p.add_argument("--b-temperature",         type=float, default=1.0)

    # Embedder for semantic similarity scoring (separate from the RAG embedder)
    p.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5",
                   help="fastembed model used for semantic similarity scoring")

    # RAGAS
    p.add_argument("--ragas",      action="store_true",
                   help="Enable RAGAS evaluation (requires ragas + datasets packages)")
    p.add_argument("--gemini-key", default=os.getenv("GEMINI_API_KEY"),
                   help="Gemini API key for RAGAS answer_correctness (LLM judge)")

    p.add_argument("--verbose", action="store_true",
                   help="Print full answer text for each record")
    return p.parse_args()


# ── Metrics ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def token_f1(pred: str, ref: str) -> tuple[float, float, float]:
    """Return (precision, recall, F1) at token level."""
    pred_tok = Counter(_tokenize(pred))
    ref_tok  = Counter(_tokenize(ref))
    common   = sum((pred_tok & ref_tok).values())
    if common == 0:
        return 0.0, 0.0, 0.0
    prec   = common / sum(pred_tok.values())
    recall = common / sum(ref_tok.values())
    f1     = 2 * prec * recall / (prec + recall)
    return round(prec, 4), round(recall, 4), round(f1, 4)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ── Dashboard client ──────────────────────────────────────────────────────────

def _query(question: str, url: str, params: dict, timeout: int = 180) -> dict:
    resp = httpx.post(
        f"{url}/api/query",
        json={"query": question, **params},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


# ── Pretty printing ───────────────────────────────────────────────────────────

W = 84  # table width

def _hr(char: str = "─") -> None:
    print(char * W)

def _header(title: str) -> None:
    _hr("═")
    print(f"  {title}")
    _hr("═")

def _row(label: str, a_val: str, b_val: str, winner: str = "") -> None:
    print(f"  {label:<34} {a_val:>14} {b_val:>14}   {winner}")

def _stat_row(label: str, a_vals: list[float], b_vals: list[float]) -> None:
    ma, mb = float(np.mean(a_vals)), float(np.mean(b_vals))
    winner = "A ✓" if ma > mb else ("B ✓" if mb > ma else "tie")
    _row(label, f"{ma:.4f}", f"{mb:.4f}", winner)

def _lat_row(label: str, a_ms: float, b_ms: float) -> None:
    winner = "A ✓" if a_ms < b_ms else ("B ✓" if b_ms < a_ms else "tie")
    _row(label, f"{a_ms:,.0f} ms", f"{b_ms:,.0f} ms", winner)


# ── RAGAS helper ──────────────────────────────────────────────────────────────

def _run_ragas(records: list[dict], gemini_key: str | None) -> dict[str, dict]:
    """Run RAGAS answer_similarity for both A and B; answer_correctness if key given."""
    if not _HAS_RAGAS:
        print("\n[RAGAS] skipped — install with: pip install ragas datasets langchain-google-genai")
        return {}

    llm = None
    if gemini_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from ragas.llms import LangchainLLMWrapper
            llm = LangchainLLMWrapper(
                ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_key)
            )
        except ImportError:
            print("[RAGAS] langchain-google-genai not installed — skipping answer_correctness")

    metrics = [answer_similarity]
    if llm:
        answer_correctness.llm = llm
        metrics.append(answer_correctness)
        answer_similarity.llm = llm

    results = {}
    for model_key, answer_key in [("A", "answer_a"), ("B", "answer_b")]:
        print(f"\n[RAGAS] evaluating Model {model_key}…")
        ds = HFDataset.from_dict({
            "question":     [r["question"]     for r in records],
            "answer":       [r[answer_key]     for r in records],
            "ground_truth": [r["ground_truth"] for r in records],
        })
        try:
            result = ragas_evaluate(ds, metrics=metrics)
            results[model_key] = result.to_pandas().mean().to_dict()
        except Exception as e:
            print(f"[RAGAS] Model {model_key} failed: {e}")
            results[model_key] = {}
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # Load FAQ
    faqs: list[dict] = json.loads(Path(args.faq).read_text())
    if args.limit:
        faqs = faqs[: args.limit]
    n = len(faqs)

    print(f"\nLoaded {n} Q&A pairs from: {args.faq}")
    print(f"Dashboard: {args.dashboard}")

    # Load fastembed model for semantic similarity
    print(f"Loading similarity embedder ({args.embed_model})…")
    from fastembed import TextEmbedding
    embedder = TextEmbedding(model_name=args.embed_model, show_download_progress=False)

    # Query params forwarded to /api/query
    api_params = {
        "a_top_k":               args.a_top_k,
        "a_max_new_tokens":      args.a_max_new_tokens,
        "a_temperature":         args.a_temperature,
        "a_top_p":               args.a_top_p,
        "a_repetition_penalty":  args.a_repetition_penalty,
        "b_top_k":               args.b_top_k,
        "b_max_output_tokens":   args.b_max_output_tokens,
        "b_temperature":         args.b_temperature,
    }

    records: list[dict] = []
    errors = 0

    # ── Per-record loop ───────────────────────────────────────────────────────
    _hr("═")
    print(f"  {'Q#':<4}  {'Sim-A':>7}  {'Sim-B':>7}  {'F1-A':>7}  {'F1-B':>7}  "
          f"{'Lat-A':>9}  {'Lat-B':>9}  {'Win-Sim':>7}  {'Win-F1':>7}")
    _hr()

    for i, item in enumerate(faqs, 1):
        question     = item["question"]
        ground_truth = item["answer"]

        # Call dashboard
        try:
            resp = _query(question, args.dashboard, api_params)
        except Exception as e:
            print(f"  Q{i:02d}  ERROR: {e}")
            errors += 1
            continue

        answer_a   = resp.get("answer_a", "")
        answer_b   = resp.get("answer_b", "")
        latency_a  = resp.get("latency_a_ms", 0)
        latency_b  = resp.get("latency_b_ms", 0)

        # Semantic similarity (embed all 3 texts in one call for efficiency)
        embs      = list(embedder.embed([answer_a, answer_b, ground_truth]))
        sim_a     = cosine_sim(embs[0], embs[2])
        sim_b     = cosine_sim(embs[1], embs[2])

        # Token F1
        _, _, f1_a = token_f1(answer_a, ground_truth)
        _, _, f1_b = token_f1(answer_b, ground_truth)

        win_sim = "A" if sim_a > sim_b else ("B" if sim_b > sim_a else "tie")
        win_f1  = "A" if f1_a  > f1_b  else ("B" if f1_b  > f1_a  else "tie")

        # Print one-line summary
        print(f"  Q{i:02d}   {sim_a:7.4f}  {sim_b:7.4f}  {f1_a:7.4f}  {f1_b:7.4f}  "
              f"{latency_a:>7}ms  {latency_b:>7}ms  {win_sim:>7}  {win_f1:>7}")

        if args.verbose:
            q_disp  = question[:90]
            gt_disp = ground_truth[:120]
            a_disp  = answer_a[:120]
            b_disp  = answer_b[:120]
            print(f"       Q : {q_disp}")
            print(f"       GT: {gt_disp}…")
            print(f"       A : {a_disp}…")
            print(f"       B : {b_disp}…")
            print()

        records.append({
            "q_num":        i,
            "question":     question,
            "ground_truth": ground_truth,
            "answer_a":     answer_a,
            "answer_b":     answer_b,
            "sim_a":        round(sim_a, 4),
            "sim_b":        round(sim_b, 4),
            "f1_a":         round(f1_a,  4),
            "f1_b":         round(f1_b,  4),
            "latency_a_ms": latency_a,
            "latency_b_ms": latency_b,
            "winner_sim":   win_sim,
            "winner_f1":    win_f1,
        })

        if args.delay > 0 and i < n:
            time.sleep(args.delay)

    if not records:
        print("\nNo results — check dashboard URL and that it is running.")
        sys.exit(1)

    # ── Optional RAGAS ────────────────────────────────────────────────────────
    ragas_scores: dict = {}
    if args.ragas:
        ragas_scores = _run_ragas(records, args.gemini_key)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    sim_a_all = [r["sim_a"] for r in records]
    sim_b_all = [r["sim_b"] for r in records]
    f1_a_all  = [r["f1_a"]  for r in records]
    f1_b_all  = [r["f1_b"]  for r in records]
    lat_a_all = [r["latency_a_ms"] for r in records]
    lat_b_all = [r["latency_b_ms"] for r in records]

    win_sim_a = sum(1 for r in records if r["winner_sim"] == "A")
    win_sim_b = sum(1 for r in records if r["winner_sim"] == "B")
    tie_sim   = sum(1 for r in records if r["winner_sim"] == "tie")
    win_f1_a  = sum(1 for r in records if r["winner_f1"]  == "A")
    win_f1_b  = sum(1 for r in records if r["winner_f1"]  == "B")
    tie_f1    = sum(1 for r in records if r["winner_f1"]  == "tie")

    k = len(records)
    _header(f"AGGREGATE SUMMARY  ({k}/{n} records evaluated, {errors} errors)")
    _row("Metric", "Model A (local LLM)", "Model B (Gemini)", "Winner")
    _hr()

    _stat_row("Semantic Similarity  mean",   sim_a_all, sim_b_all)
    _stat_row("Semantic Similarity  median",
              [float(np.median(sim_a_all))], [float(np.median(sim_b_all))])
    _stat_row("Semantic Similarity  min",
              [float(np.min(sim_a_all))], [float(np.min(sim_b_all))])
    _stat_row("Semantic Similarity  max",
              [float(np.max(sim_a_all))], [float(np.max(sim_b_all))])
    _hr()

    _stat_row("Token F1             mean",   f1_a_all, f1_b_all)
    _stat_row("Token F1             median",
              [float(np.median(f1_a_all))], [float(np.median(f1_b_all))])
    _hr()

    _row("Win rate  Semantic Similarity",
         f"{win_sim_a}/{k}", f"{win_sim_b}/{k}",
         "A ✓" if win_sim_a > win_sim_b else ("B ✓" if win_sim_b > win_sim_a else "tie"))
    _row("Win rate  Token F1",
         f"{win_f1_a}/{k}", f"{win_f1_b}/{k}",
         "A ✓" if win_f1_a > win_f1_b else ("B ✓" if win_f1_b > win_f1_a else "tie"))
    _row("Ties  (Sim / F1)",
         f"{tie_sim}", f"{tie_f1}", "")
    _hr()

    _lat_row("Latency  mean",
             float(np.mean(lat_a_all)), float(np.mean(lat_b_all)))
    _lat_row("Latency  p50",
             float(np.percentile(lat_a_all, 50)), float(np.percentile(lat_b_all, 50)))
    _lat_row("Latency  p90",
             float(np.percentile(lat_a_all, 90)), float(np.percentile(lat_b_all, 90)))
    _lat_row("Latency  p99",
             float(np.percentile(lat_a_all, 99)), float(np.percentile(lat_b_all, 99)))
    _hr()

    if ragas_scores:
        print("  RAGAS scores (mean over all records):")
        for model_key, scores in ragas_scores.items():
            for metric, val in scores.items():
                if isinstance(val, float):
                    _row(f"  {metric} (Model {model_key})", f"{val:.4f}", "", "")
        _hr()

    _hr("═")

    # ── Save results ──────────────────────────────────────────────────────────
    if args.out:
        output = {
            "config": {
                "dashboard":    args.dashboard,
                "faq_file":     args.faq,
                "n_evaluated":  k,
                "n_errors":     errors,
                "embed_model":  args.embed_model,
                "api_params":   api_params,
            },
            "summary": {
                "mean_sim_a":      round(float(np.mean(sim_a_all)), 4),
                "mean_sim_b":      round(float(np.mean(sim_b_all)), 4),
                "median_sim_a":    round(float(np.median(sim_a_all)), 4),
                "median_sim_b":    round(float(np.median(sim_b_all)), 4),
                "mean_f1_a":       round(float(np.mean(f1_a_all)), 4),
                "mean_f1_b":       round(float(np.mean(f1_b_all)), 4),
                "win_sim_a":       win_sim_a,
                "win_sim_b":       win_sim_b,
                "tie_sim":         tie_sim,
                "win_f1_a":        win_f1_a,
                "win_f1_b":        win_f1_b,
                "tie_f1":          tie_f1,
                "mean_latency_a_ms":  round(float(np.mean(lat_a_all))),
                "mean_latency_b_ms":  round(float(np.mean(lat_b_all))),
                "p50_latency_a_ms":   round(float(np.percentile(lat_a_all, 50))),
                "p50_latency_b_ms":   round(float(np.percentile(lat_b_all, 50))),
                "p90_latency_a_ms":   round(float(np.percentile(lat_a_all, 90))),
                "p90_latency_b_ms":   round(float(np.percentile(lat_b_all, 90))),
                "ragas": ragas_scores,
            },
            "records": records,
        }
        Path(args.out).write_text(json.dumps(output, indent=2))
        print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
