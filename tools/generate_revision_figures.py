"""Generate publication figures for the KVForge scientific revision.

Reads the real experimental results from ``docs/scientific_revision_real/`` and
writes:

* ``docs/figures/figNEW_1_prs_cosine_vs_factual.png`` — PRS cosine ratio vs.
  factual correctness (judge + token-F1), with per-corpus correlation.
* ``docs/figures/figNEW_2_calibration_reliability.png`` — reliability diagrams
  for parametric confidence calibration.
* ``docs/figures/figNEW_3_attention_divergence.png`` — per-layer KL divergence
  between true prefill and KV-injected attention distributions.

Usage::

    python tools/generate_revision_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "docs" / "scientific_revision_real"
FIGURES_DIR = ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


USE_CASES = {
    "usecase1_customer_support": "UC1 Customer Support",
    "usecase2_pubmedqa": "UC2 PubMedQA",
    "usecase3_squad": "UC3 SQuAD",
    "usecase4_bedrock_userguide": "UC4 Bedrock",
}


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _compute_cosine_ratios(parametric_rows: list[dict], rag_rows: list[dict]) -> list[float]:
    """Compute the legacy cosine accuracy ratio from E1 per-question rows."""
    from fastembed import TextEmbedding
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", show_download_progress=False)
    ratios = []
    for p_row, r_row in zip(parametric_rows, rag_rows):
        p = p_row["prediction"]
        r = r_row["prediction"]
        g = p_row["ground_truth"]
        vecs = np.array(list(embedder.embed([p, r, g])))
        a = vecs[0] / (np.linalg.norm(vecs[0]) + 1e-9)
        b = vecs[2] / (np.linalg.norm(vecs[2]) + 1e-9)
        c = vecs[1] / (np.linalg.norm(vecs[1]) + 1e-9)
        param_sim = float(np.dot(a, b))
        rag_sim = float(np.dot(c, b))
        ratios.append(min(param_sim / (rag_sim + 1e-9), 1.0))
    return ratios


def generate_figNEW_1():
    """PRS cosine ratio vs. factual correctness (judge)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (tag, label) in zip(axes, USE_CASES.items()):
        e1 = _load(RESULTS_DIR / tag / "eval_phase_quality.json")
        if not e1 or "parametric" not in e1.get("modes", {}):
            ax.set_visible(False)
            continue
        param_rows = e1["modes"]["parametric"]["per_question"]
        rag_rows = e1["modes"]["text_rag"]["per_question"]
        cos = _compute_cosine_ratios(param_rows, rag_rows)
        judges = [r["judge_correct"] for r in param_rows]
        f1s = [r["token_f1"] for r in param_rows]
        factual = [0.5 * f1 + 0.5 * jc for f1, jc in zip(f1s, judges)]

        if len(set(cos)) > 1 and len(set(factual)) > 1:
            pr = pearsonr(cos, factual)[0]
            sr = spearmanr(cos, factual)[0]
        else:
            pr = sr = np.nan

        ax.scatter(cos, factual, alpha=0.6, s=50)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8)
        ax.axvline(0.85, color="red", linestyle="--", linewidth=0.8)
        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(label)
        ax.set_xlabel("Cosine accuracy ratio")
        ax.set_ylabel("Factual correctness (0.5·F1 + 0.5·judge)")
        ax.text(
            0.05, 0.95,
            f"Pearson r={pr:.2f}\nSpearman ρ={sr:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    fig.suptitle("Figure NEW-1: PRS cosine ratio is a weak proxy for factual correctness", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = FIGURES_DIR / "figNEW_1_prs_cosine_vs_factual.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Wrote {out}")


def generate_figNEW_2():
    """Reliability diagrams for calibration."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (tag, label) in zip(axes, USE_CASES.items()):
        e3 = _load(RESULTS_DIR / tag / "eval_calibration.json")
        if not e3:
            ax.set_visible(False)
            continue
        bin_edges = np.linspace(0, 1, 11)
        conf = np.asarray(e3.get("per_bin_confidence", []))
        acc = np.asarray(e3.get("per_bin_accuracy", []))
        ece = e3.get("ece", 0.0)

        ax.bar(bin_edges[:-1], acc, width=0.1, align="edge", alpha=0.6, label="Accuracy", edgecolor="black")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{label}  (ECE={ece:.3f})")
        ax.set_xlabel("Mean confidence")
        ax.set_ylabel("Accuracy")
        ax.legend(loc="upper left")

    fig.suptitle("Figure NEW-2: Parametric confidence calibration is unreliable", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = FIGURES_DIR / "figNEW_2_calibration_reliability.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Wrote {out}")


def generate_figNEW_3():
    """Per-layer attention divergence (KL and cosine)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for tag, label in USE_CASES.items():
        e5 = _load(RESULTS_DIR / tag / "eval_attention_divergence.json")
        if not e5 or "per_layer" not in e5:
            continue
        layers = [r["layer"] for r in e5["per_layer"]]
        kl_full = [r["kl_fulltoken_vs_prefill"] for r in e5["per_layer"]]
        kl_mean = [r["kl_meanpool_vs_prefill"] for r in e5["per_layer"]]
        cos_full = [r["cosine_fulltoken_vs_prefill"] for r in e5["per_layer"]]
        cos_mean = [r["cosine_meanpool_vs_prefill"] for r in e5["per_layer"]]

        ax1.plot(layers, kl_full, marker="o", label=f"{label} full-token", alpha=0.7)
        ax1.plot(layers, kl_mean, marker="s", linestyle="--", label=f"{label} mean-pool", alpha=0.7)

        ax2.plot(layers, cos_full, marker="o", label=f"{label} full-token", alpha=0.7)
        ax2.plot(layers, cos_mean, marker="s", linestyle="--", label=f"{label} mean-pool", alpha=0.7)

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("KL divergence")
    ax1.set_title("KL divergence vs true prefill")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Cosine distance")
    ax2.set_title("Cosine distance vs true prefill")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Figure NEW-3: Mean-pool KV diverges more from true prefill than full-token KV", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = FIGURES_DIR / "figNEW_3_attention_divergence.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Wrote {out}")


def generate_figNEW_4():
    """E4 ablation: PRS and parametric token-F1 across training conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ablation_ucs = {
        "usecase2_pubmedqa": "UC2 PubMedQA",
        "usecase3_squad": "UC3 SQuAD",
    }
    conditions = ["tier_weighted_cloud", "uniform_cloud", "uniform_heuristic"]
    labels = ["tier-weighted\ncloud", "uniform\ncloud", "uniform\nheuristic"]

    prs_data = {tag: [] for tag in ablation_ucs}
    f1_data = {tag: [] for tag in ablation_ucs}
    any_data = False

    for tag in ablation_ucs:
        abl = _load(RESULTS_DIR / tag / "eval_ablations.json")
        if not abl:
            continue
        for cond in conditions:
            entry = abl.get(cond, {})
            prs_data[tag].append(entry.get("prs", 0.0) or 0.0)
            e1 = entry.get("e1_summary", {})
            param = e1.get("parametric", {})
            f1_data[tag].append(param.get("token_f1", 0.0))
        any_data = True

    if not any_data:
        axes[0].set_visible(False)
        axes[1].set_visible(False)
        plt.close(fig)
        return

    x = np.arange(len(labels))
    width = 0.35

    for ax, data, title, ylabel in [
        (axes[0], prs_data, "PRS", "PRS"),
        (axes[1], f1_data, "Parametric token-F1", "token-F1"),
    ]:
        for offset, (tag, label) in enumerate(ablation_ucs.items()):
            if data[tag]:
                ax.bar(x + (offset - 0.5) * width, data[tag], width, label=label)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Figure NEW-4: E4 ablation — training-signal quality (pilot/full results)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = FIGURES_DIR / "figNEW_4_ablation.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Wrote {out}")


def main():
    generate_figNEW_1()
    generate_figNEW_2()
    generate_figNEW_3()
    generate_figNEW_4()


if __name__ == "__main__":
    main()
