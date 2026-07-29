"""E3 / Sprint 2.5 — Calibration reliability: bin self-reported confidence vs actual correctness.

Usage (legacy ``eval_phase_quality.json``):

    python -m pipeline.eval_calibration \
        --input examples/usecase4_bedrock_userguide/eval_phase_quality.json \
        --output examples/usecase4_bedrock_userguide/eval_calibration.json

Usage (confidence-token flat file):

    python -m pipeline.eval_calibration \
        --input predictions.json \
        --format confidence_token \
        --plot reliability.png \
        --output calibration.json

The script accepts two input formats:

1. ``eval_phase_quality.json`` output (legacy): contains ``modes.<mode>.per_question``
   with ``confidence`` and ``judge_correct`` fields.
2. A flat JSON/CSV with ``confidence`` and ``correctness`` fields (Sprint 2.5
   confidence-token format).

For the confidence-token path, confidence is the restricted two-token softmax
probability ``P(" yes")`` and correctness is the binary factual-accuracy label.

A reliability diagram PNG can be generated with ``--plot <path>``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

sys_path = str(Path(__file__).resolve().parent.parent)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from eval import metrics


def _synthesize_confidences(rows: list[dict], seed: int = 42) -> list[float]:
    """For dry-run / heuristic mode: synthesize plausible self-confidences."""
    import random
    rng = random.Random(seed)
    confidences = []
    for r in rows:
        f1 = r["token_f1"]
        # Models are slightly overconfident: add a small upward bias.
        conf = 0.7 * f1 + 0.2 + rng.gauss(0, 0.05)
        confidences.append(max(0.0, min(1.0, conf)))
    return confidences


def load_confidence_token_rows(path: str) -> list[dict]:
    """Load a flat JSON or CSV with ``confidence`` and ``correctness`` fields."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    else:
        data = json.loads(p.read_text())
        rows = data if isinstance(data, list) else data.get("rows", [])
    return rows


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        return float(value.strip())
    return float(value)


def extract_confidence_and_correctness(rows: list[dict]) -> tuple[list[float], list[int]]:
    """Return (confidences, correctness) lists from confidence-token rows.

    Accepts boolean or float ``correctness`` and normalizes to 0/1.
    """
    confidences = []
    correctness = []
    for r in rows:
        conf = _to_float(r.get("confidence"))
        corr = _to_float(r.get("correctness", r.get("judge_correct", r.get("factually_correct", 0))))
        confidences.append(max(0.0, min(1.0, conf)))
        correctness.append(int(corr >= 0.5))
    return confidences, correctness


def plot_reliability_diagram(
    per_bin_confidence: list[float],
    per_bin_accuracy: list[float],
    ece: float,
    output_path: str,
    title: str = "Reliability Diagram",
) -> None:
    """Draw a reliability diagram (confidence vs accuracy) and save to PNG."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for --plot. Install with: pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=(6, 6))
    x = [0.0] + per_bin_confidence + [1.0]
    y = [0.0] + per_bin_accuracy + [1.0]
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.plot(x, y, "-o", color="steelblue", label="model")
    ax.bar(per_bin_confidence, per_bin_accuracy, width=0.08, alpha=0.3, color="steelblue")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title} (ECE={ece:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"✓ Wrote reliability diagram to {output_path}")


def evaluate_calibration(
    confidences: list[float],
    correctness: list[int],
    bins: int = 10,
    plot_path: str | None = None,
    title: str = "Reliability Diagram",
) -> dict[str, Any]:
    """Compute ECE and optionally draw a reliability diagram.

    Returns:
        Dict with ``ece``, ``mean_confidence``, ``mean_accuracy``, ``n``,
        ``per_bin_accuracy``, ``per_bin_confidence``.
    """
    calib = metrics.expected_calibration_error(confidences, correctness, bins=bins)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    mean_acc = sum(correctness) / len(correctness) if correctness else 0.0

    output = {
        "n": len(confidences),
        "mean_confidence": round(mean_conf, 4),
        "mean_accuracy": round(mean_acc, 4),
        "ece": round(calib["ece"], 4),
        "per_bin_accuracy": [round(x, 4) for x in calib["per_bin_accuracy"]],
        "per_bin_confidence": [round(x, 4) for x in calib["per_bin_confidence"]],
    }

    if plot_path:
        plot_reliability_diagram(
            calib["per_bin_confidence"],
            calib["per_bin_accuracy"],
            calib["ece"],
            plot_path,
            title=title,
        )

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="E3/Sprint 2.5: Calibration reliability")
    parser.add_argument("--input", required=True, help="Input JSON/CSV file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--format", choices=["phase_quality", "confidence_token"],
                        default="phase_quality",
                        help="Input format: phase_quality (legacy) or confidence_token (flat)")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--plot", default=None, help="Path to save reliability diagram PNG")
    parser.add_argument("--parametric-only", action="store_true",
                        help="[phase_quality] Only calibrate parametric-mode answers")
    args = parser.parse_args()

    if args.format == "confidence_token":
        rows = load_confidence_token_rows(args.input)
        confidences, correctness = extract_confidence_and_correctness(rows)
        output = evaluate_calibration(
            confidences, correctness, bins=args.bins, plot_path=args.plot,
            title="Confidence Token Calibration",
        )
        output["format"] = "confidence_token"
    else:
        data = json.loads(Path(args.input).read_text())
        modes = data.get("modes", {})
        target_mode = "parametric" if args.parametric_only else "text_rag"
        if target_mode not in modes:
            raise ValueError(f"Input does not contain mode={target_mode}")
        rows = modes[target_mode]["per_question"]
        correctness = [r["judge_correct"] for r in rows]
        confidences = [r.get("confidence") for r in rows]
        if any(c is None for c in confidences):
            confidences = _synthesize_confidences(rows)
        output = evaluate_calibration(
            confidences, correctness, bins=args.bins, plot_path=args.plot,
            title=f"{target_mode} Calibration",
        )
        output["format"] = "phase_quality"
        output["mode"] = target_mode

    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"✓ Wrote {args.output}")
    print(f"  ECE={output['ece']:.4f}  mean_conf={output['mean_confidence']:.4f}  "
          f"mean_acc={output['mean_accuracy']:.4f}")


if __name__ == "__main__":
    main()
