#!/usr/bin/env python3
"""
Generate Supplementary Figure 1:

CV comparison schemes, compares predictive performance across evaluation schemes and model variants.

Panel outputs:
- A: Effect of CV strictness (HALO-Base vs HALO-S-CV1 vs HALO-S-CV2).
- B: Model variants under CV1 (HALO-S-CV1 vs HALO-CV1).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================
# Paths
# ==========================
from halo.paths import FIGURE_PIPELINE

OUT_DIR = FIGURE_PIPELINE / "supp_fig1_cv_comparison" / "supp_fig1_panels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# Data
# ==========================

DPI = 600

metric_labels = ["Accuracy", "F1-macro", "ROC-AUC"]

# Panel A: CV strictness
labels_A = ["HALO-Base", "HALO-S-CV1", "HALO-S-CV2"]
acc_A = np.array([0.778, 0.734, 0.536])
f1_A  = np.array([0.778, 0.733, 0.529])
auc_A = np.array([0.863, 0.816, 0.577])
values_A = np.vstack([acc_A, f1_A, auc_A])

# Panel B: model variants under CV1
labels_B = ["HALO-S-CV1", "HALO-CV1"]
acc_B = np.array([0.734, 0.745])
f1_B  = np.array([0.733, 0.745])
auc_B = np.array([0.816, 0.822])
values_B = np.vstack([acc_B, f1_B, auc_B])

# Colors
main_blue    = "#1f77b4"
pastel_teal  = "#8DC5C1"
pastel_red   = "#E8A5A5"
pastel_green = "#A7D7A0"

colors_A = [main_blue, pastel_teal, pastel_red]
colors_B = [pastel_teal, pastel_green]

plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})

# ==========================
# Helper
# ==========================
def plot_grouped_bars(ax, values, model_labels, colors):
    n_metrics, n_models = values.shape
    x = np.arange(n_metrics)

    bar_width = 0.8 / n_models
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_width

    for i in range(n_models):
        vals = values[:, i]
        xpos = x + offsets[i]

        ax.bar(
            xpos, vals,
            width=bar_width,
            color=colors[i],
            edgecolor="black",
            linewidth=0.4,
            label=model_labels[i],
        )

        for px, py in zip(xpos, vals):
            ax.text(px, py + 0.01, f"{py:.2f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

# ==========================
# Figure
# ==========================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4))

# Panel A
plot_grouped_bars(axA, values_A, labels_A, colors_A)
axA.set_title(r"$\mathbf{A}$.   CV strictness vs performance",
              fontsize=11, loc="center", pad=8)
axA.legend(
    fontsize=9,
    title="Evaluation scheme",
    title_fontsize=9,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),
    ncol=len(labels_A),
    frameon=True,
)

# Panel B
plot_grouped_bars(axB, values_B, labels_B, colors_B)
axB.set_title(r"$\mathbf{B}$.   Model variants under CV1",
              fontsize=11, loc="center", pad=8)
axB.legend(
    fontsize=9,
    title="Model variant (CV1)",
    title_fontsize=9,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),
    ncol=len(labels_B),
    frameon=True,
)

# ==========================
# Force identical y-limits and ticks on both panels
# ==========================
ymin, ymax = 0.45, 0.95
yticks = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

for ax in (axA, axB):
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.1f}" for t in yticks])

# leave extra bottom space for legends
fig.tight_layout(rect=[0, 0.12, 1, 1])

OUT_DIR = OUT_DIR / "supp_fig1_panelA_B"
fig.savefig(OUT_DIR.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
