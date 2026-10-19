#!/usr/bin/env python3
"""Generate accuracy + latency comparison charts (UC4, 500-Q held-out set).

E2B baseline (r16, 337 QA) highlighted against every other run.
Data: all runs on the identical 500-question eval, judge deepseek-v4-lite.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, FA, FA_sem, lat_p50, lat_p95, group)
RUNS = [
    ("E4B r64 · 60k QA + rewrites", 0.286, 0.013, 1.64, 3.78, "E4B"),
    ("Llama-3.2-3B · 60k QA + rewrites", 0.254, 0.012, 1.47, 2.42, "Llama"),
    ("E2B r64 · 60k QA + rewrites", 0.249, 0.012, 1.31, 3.11, "E2B"),
    ("E4B r64 · 60k QA",           0.244, 0.011, 1.55, 3.59, "E4B"),
    ("E4B r64 · 337 QA",           0.235, 0.009, 2.99, 5.03, "E4B"),
    ("E2B r64 · 60k QA",           0.219, 0.011, 1.23, 2.72, "E2B"),
    ("E2B RAG (untrained)",        0.175, 0.011, 3.19, 29.87, "RAG"),
    ("E4B RAG (untrained)",        0.161, 0.010, 3.48, 29.73, "RAG"),
    ("E2B baseline · r16 · 337 QA", 0.147, 0.007, 3.92, 3.99, "baseline"),
]

COLORS = {"E2B": "#1f77b4", "E4B": "#2ca02c", "Llama": "#9467bd",
          "RAG": "#9e9e9e", "baseline": "#d62728"}

def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.6)
    ax.set_axisbelow(True)

# ---- Chart 1: Accuracy ----
acc = sorted(RUNS, key=lambda r: -r[1])
labels = [r[0] for r in acc]
fa = [r[1] for r in acc]
err = [r[2] for r in acc]
colors = [COLORS[r[5]] for r in acc]

fig, ax = plt.subplots(figsize=(9, 5.2))
bars = ax.barh(labels, fa, xerr=err, color=colors, capsize=3, error_kw={"ecolor": "#333", "linewidth": 0.8})
for b, v in zip(bars, fa):
    ax.text(v + 0.005, b.get_y() + b.get_height()/2, f"{v:.3f}", va="center", fontsize=8.5)
ax.set_xlabel("Factual accuracy (FA = 0.5·token-F1 + 0.5·LLM-judge)")
ax.set_title("UC4 · 500-Q held-out set — parametric vs RAG accuracy (judge: deepseek-v4-lite)")
ax.set_xlim(0, 0.33)
style_ax(ax)
# legend
from matplotlib.patches import Patch
legend = [Patch(color=COLORS["baseline"], label="E2B baseline (r16, 337 QA)"),
          Patch(color=COLORS["E2B"], label="E2B trained"),
          Patch(color=COLORS["E4B"], label="E4B trained"),
          Patch(color=COLORS["Llama"], label="Llama-3.2-3B trained"),
          Patch(color=COLORS["RAG"], label="RAG (untrained + retrieval)")]
ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig("docs/figures/fig_e2b_comparison_accuracy.png", dpi=150)
plt.close(fig)

# ---- Chart 2: Latency ----
lat = sorted(RUNS, key=lambda r: r[3])  # sort by p50 ascending (fastest first)
labels = [r[0] for r in lat]
p50 = [r[3] for r in lat]
p95 = [r[4] for r in lat]
colors = [COLORS[r[5]] for r in lat]

fig, ax = plt.subplots(figsize=(9, 5.2))
bars = ax.barh(labels, p50, color=colors)
for b, v in zip(bars, p50):
    ax.text(v + 0.05, b.get_y() + b.get_height()/2, f"{v:.2f}s", va="center", fontsize=8.5)
# p95 whisker markers
ax.scatter(p95, range(len(labels)), marker="|", s=60, color="#555", linewidth=1.2,
           label="p95")
ax.set_xlabel("Response latency, p50 (seconds) · | = p95")
ax.set_title("UC4 · 500-Q held-out set — parametric vs RAG latency")
ax.set_xlim(0, max(p95) * 1.08)
style_ax(ax)
from matplotlib.patches import Patch
legend = [Patch(color=COLORS["baseline"], label="E2B baseline (r16, 337 QA)"),
          Patch(color=COLORS["E2B"], label="E2B trained"),
          Patch(color=COLORS["E4B"], label="E4B trained"),
          Patch(color=COLORS["Llama"], label="Llama-3.2-3B trained"),
          Patch(color=COLORS["RAG"], label="RAG (untrained + retrieval)")]
ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig("docs/figures/fig_e2b_comparison_latency.png", dpi=150)
plt.close(fig)

print("saved docs/figures/fig_e2b_comparison_accuracy.png")
print("saved docs/figures/fig_e2b_comparison_latency.png")
