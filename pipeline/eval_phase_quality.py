"""E1 — Phase quality matrix: text RAG vs KV mean-pool vs KV full-token vs parametric.

Usage (real GPU run, requires a configured use-case and vLLM/dashboard):

    python -m pipeline.eval_phase_quality \
        --config examples/usecase4_bedrock_userguide/config.json \
        --mode all \
        --output examples/usecase4_bedrock_userguide/eval_phase_quality.json \
        --max-samples 200

Usage (dry-run / deterministic simulation, no GPU):

    python -m pipeline.eval_phase_quality \
        --config examples/usecase4_bedrock_userguide/config.json \
        --mode all --dry-run \
        --output examples/usecase4_bedrock_userguide/eval_phase_quality.json

The script reads the evaluation split from ``eval/splits.py`` (or falls back to
the FAQ file for dry-runs) and produces per-mode exact-match, token-F1, and
LLM-judge factuality scores.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import metrics
from eval import splits


# Heuristic quality priors for the deterministic dry-run simulation.
# These encode the empirically expected contingency: mean-pool is faster but
# slightly lossy; full-token recovers most of the text-RAG quality; parametric
# is the lowest because it answers from memory without retrieval.
_DRY_RUN_PARAMS = {
    "text_rag":   {"em": 0.45, "f1": 0.72, "judge": 0.75, "judge_noise": 0.08},
    "kv_meanpool": {"em": 0.32, "f1": 0.58, "judge": 0.62, "judge_noise": 0.10},
    "kv_fulltoken": {"em": 0.42, "f1": 0.70, "judge": 0.73, "judge_noise": 0.08},
    "parametric": {"em": 0.24, "f1": 0.50, "judge": 0.55, "judge_noise": 0.12},
}


def _load_eval_questions(cfg: dict, max_samples: int | None = None, auto_download: bool = True) -> list[dict]:
    """Load a held-out evaluation set for the use-case."""
    uc = cfg.get("use_case", "")
    base = Path(cfg.get("version_file", "config.json")).parent

    # SQuAD
    if "squad" in uc.lower() or "usecase3" in str(base).lower():
        train_path = base / "data" / "train-v2.0.json"
        dev_path = base / "data" / "dev-v2.0.json"
        faq_train = base / "faqs_train.json"
        if not faq_train.exists():
            faq_train = base / "faqs.json"
        split = splits.load_squad_split(
            train_path=train_path,
            dev_path=dev_path,
            faqs_train_path=faq_train if faq_train.exists() else None,
            sample_dev=max_samples,
            auto_download=auto_download,
        )
        return split.get("dev", [])

    # PubMedQA
    if "pubmed" in uc.lower() or "usecase2" in str(base).lower():
        train_path = base / "data" / "train_set.json"
        test_path = base / "data" / "test_set.json"
        faq_train = base / "faqs_train.json"
        if not faq_train.exists():
            faq_train = base / "faqs.json"
        split = splits.load_pubmedqa_split(
            train_path=train_path,
            test_path=test_path,
            faqs_train_path=faq_train if faq_train.exists() else None,
            sample_test=max_samples,
            auto_download=auto_download,
        )
        return split.get("test", [])

    # Bitext / customer support
    if "customer" in uc.lower() or "usecase1" in str(base).lower():
        faq_path = base / "faqs.json"
        if not faq_path.exists():
            return []
        split = splits.load_bitext_split(faq_path, test_fraction=0.15, auto_download=auto_download)
        test = split["test"]
        if max_samples:
            test = test[:max_samples]
        return test

    # Bedrock / technical docs
    faq_train = base / "faqs_train.json"
    if not faq_train.exists():
        faq_train = base / "faqs.json"
    hand = splits.load_bedrock_hand_verified(
        path=base / "hand_verified_test.json",
        faqs_train_path=faq_train if faq_train.exists() else None,
    )
    if hand.get("test"):
        test = hand["test"]
        if max_samples:
            test = test[:max_samples]
        return test

    # Fallback: hold out 15% of the FAQ file
    faq_path = base / "faqs.json"
    if faq_path.exists():
        split = splits.load_bitext_split(faq_path, test_fraction=0.15, auto_download=auto_download)
        test = split["test"]
        if max_samples:
            test = test[:max_samples]
        return test

    return []


def _degrade_answer(answer: str, severity: float, rng: random.Random) -> str:
    """Create a realistic synthetic answer by degrading the ground truth."""
    words = answer.split()
    if len(words) <= 3:
        return answer
    out = []
    for w in words:
        if rng.random() < severity:
            continue
        if rng.random() < severity * 0.5:
            out.append("...")
        else:
            out.append(w)
    if len(out) < 2:
        out = words[: max(2, int(len(words) * (1 - severity)))]
    return " ".join(out)


def _dry_run_answer(question: str, ground_truth: str, mode: str, rng: random.Random) -> str:
    """Generate a deterministic synthetic answer for dry-run mode."""
    params = _DRY_RUN_PARAMS[mode]
    # With probability equal to the expected EM rate, return the exact answer.
    em_rate = params["em"]
    if rng.random() < em_rate:
        return ground_truth
    # Otherwise degrade the ground truth; severity is tuned so that token-F1
    # lands near the mode prior.
    severity = 1.0 - params["f1"] * 0.85
    return _degrade_answer(ground_truth, severity, rng)


def _run_mode_real(
    mode: str,
    questions: list[dict],
    cfg: dict,
    client: Any | None,
    judge_model: str = "gpt-4o-mini",
) -> list[dict]:
    """Real GPU-backed evaluation for a single mode."""
    from pipeline.kv_inference import answer_with_mode
    from pipeline.prs_evaluator import _generate_parametric
    from core import model_loader

    model, tokenizer = model_loader.load(cfg.get("checkpoint_path"))
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer,
                       max_new_tokens=256, do_sample=False)

    results = []
    for item in questions:
        q = item["question"]
        gt = item["answer"]
        if mode == "parametric":
            pred = _generate_parametric(q, pipe)
            used_mode = "parametric"
        else:
            pred, used_mode = answer_with_mode(q, cfg, force_mode=mode)
        em = metrics.exact_match(pred, gt)
        f1 = metrics.token_f1(pred, gt)
        judge = metrics.llm_judge(q, pred, gt, context=item.get("context"), client=client, model=judge_model)
        results.append({
            "question": q,
            "ground_truth": gt,
            "prediction": pred,
            "mode": used_mode,
            "em": em,
            "token_f1": f1,
            "judge_correct": int(judge["factually_correct"]),
            "judge_rationale": judge["rationale"],
        })
    return results


def _run_mode_dry(
    mode: str,
    questions: list[dict],
    seed: int,
) -> list[dict]:
    """Deterministic simulation for a single mode (no GPU/API)."""
    rng = random.Random(seed)
    params = _DRY_RUN_PARAMS[mode]
    results = []
    for item in questions:
        q = item["question"]
        gt = item["answer"]
        pred = _dry_run_answer(q, gt, mode, rng)
        em = metrics.exact_match(pred, gt)
        f1 = metrics.token_f1(pred, gt)
        # Deterministic judge based on the token-F1 relative to the mode prior.
        judge_prob = params["judge"] + rng.gauss(0, params["judge_noise"])
        judge_prob = max(0.0, min(1.0, judge_prob))
        # Make the judge correlate with the actual F1 score.
        judge_prob = 0.6 * judge_prob + 0.4 * f1
        judge_correct = int(rng.random() < judge_prob)
        results.append({
            "question": q,
            "ground_truth": gt,
            "prediction": pred,
            "mode": mode,
            "em": em,
            "token_f1": f1,
            "judge_correct": judge_correct,
            "judge_rationale": (
                f"dry-run heuristic: f1={f1:.2f}, judge_prob={judge_prob:.2f}"
            ),
        })
    return results


def _summarize_mode(rows: list[dict]) -> dict:
    """Aggregate per-mode results with mean, SEM, and bootstrap CI."""
    ems = [r["em"] for r in rows]
    f1s = [r["token_f1"] for r in rows]
    judges = [r["judge_correct"] for r in rows]
    return {
        "n": len(rows),
        "em": metrics.summarize_binary_metric(ems),
        "token_f1": metrics.summarize_binary_metric(f1s),
        "judge": metrics.summarize_binary_metric(judges),
    }


def _get_judge_client(provider: str | None, api_key: str | None, model: str):
    """Return an external judge client for the requested provider."""
    provider = (provider or os.environ.get("JUDGE_PROVIDER") or "openai").lower()
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if provider == "openai":
        import openai
        return openai.OpenAI(api_key=api_key) if api_key else None
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=api_key) if api_key else None
    if provider == "google" or provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="E1: Phase quality matrix")
    parser.add_argument("--config", required=True, help="Path to use-case config.json")
    parser.add_argument("--mode", default="all",
                        help="Comma-separated modes: text_rag,kv_meanpool,kv_fulltoken,parametric,all")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap number of evaluation questions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Deterministic simulation without GPU/API (default: real inference)")
    parser.add_argument("--dry-run-seed", type=int, default=42,
                        help="Seed for deterministic dry-run")
    parser.add_argument("--judge-provider", default="openai",
                        help="Judge provider: openai/anthropic/gemini")
    parser.add_argument("--judge-api-key", default="",
                        help="API key for the judge (or set env var)")
    parser.add_argument("--judge-model", default="gpt-4o-mini",
                        help="Judge model for real LLM-judge eval")
    parser.add_argument("--auto-download", action="store_true", default=True,
                        help="Download datasets from HuggingFace if missing")
    parser.add_argument("--checkpoint", default=None,
                        help="Override LoRA checkpoint path (default: use version.json)")
    args = parser.parse_args()

    raw_cfg = json.loads(Path(args.config).read_text())
    # Flatten the nested addon_config into the top-level dict.
    addon_config = raw_cfg.get("addon_config", {})
    cfg = {**raw_cfg}
    for section in ("indexing", "inference", "training", "background", "sync", "monitoring"):
        cfg.update(addon_config.get(section, {}))

    # Load LoRA checkpoint from version.json.
    import core.version as _ver
    import core.model_loader as _model_loader
    _ver.init(cfg)
    _model_loader.init(cfg)
    version = _ver.load()
    cfg["checkpoint_path"] = args.checkpoint or version.get("checkpoint_path")
    cfg["phase"] = version.get("phase", 1)
    cfg["current_lora_version"] = version.get("current_lora_version", 0)

    questions = _load_eval_questions(cfg, max_samples=args.max_samples, auto_download=args.auto_download)
    if not questions:
        print("⚠️  No evaluation questions found; falling back to FAQ file.")
        base = Path(args.config).parent
        faq_path = base / "faqs.json"
        if faq_path.exists():
            questions = [
                {"question": f.get("question", ""), "answer": f.get("answer", "")}
                for f in json.loads(faq_path.read_text())
            ]
            if args.max_samples:
                questions = questions[:args.max_samples]

    if not questions:
        raise ValueError("Could not load any evaluation questions.")

    modes = [m.strip() for m in args.mode.split(",")]
    if "all" in modes:
        modes = ["text_rag", "kv_meanpool", "kv_fulltoken", "parametric"]

    client = None
    if not args.dry_run:
        client = _get_judge_client(args.judge_provider, args.judge_api_key, args.judge_model)
        if client is None:
            print("⚠️  No external judge client available; using heuristic fallback.")
            print("    Set --judge-api-key or OPENAI_API_KEY/ANTHROPIC_API_KEY/GEMINI_API_KEY.")

    output = {"config": str(args.config), "dry_run": args.dry_run, "modes": {}}
    for mode in modes:
        print(f"▶ Evaluating mode={mode} on {len(questions)} questions …")
        if args.dry_run:
            rows = _run_mode_dry(mode, questions, seed=args.dry_run_seed + hash(mode) % 1000)
        else:
            rows = _run_mode_real(mode, questions, cfg, client, judge_model=args.judge_model)
        output["modes"][mode] = {
            "per_question": rows,
            "summary": _summarize_mode(rows),
        }
        summary = output["modes"][mode]["summary"]
        print(f"   EM={summary['em']['mean']:.3f}  "
              f"F1={summary['token_f1']['mean']:.3f}  "
              f"Judge={summary['judge']['mean']:.3f}")

    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"✓ Wrote {args.output}")


if __name__ == "__main__":
    main()
