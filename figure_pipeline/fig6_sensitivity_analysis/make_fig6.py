"""
Generate Figure 6:

Sensitivity analysis for Bliss neutrality cutoff 

Panel outputs:
- A: Line/scatter plot of metrics vs cutoff
- B: Bar plot with error bars
- C: Violin/box distribution of p_synergy
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from halo.paths import RESULTS, FIGURES

BASE_DIR = RESULTS / "sensitivity_analysis" 
PLOT_DIR = FIGURES / "main"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
FIG6_PATH_PNG = PLOT_DIR / "fig6.png"

DPI = 600

# colors
main_blue     = "#1f77b4"
pastel_teal   = "#8DC5C1"
pastel_red    = "#E8A5A5"
pastel_green  = "#A7D7A0"
pastel_blue   = "#7BAFD4"
pastel_yellow = "#F4E3A3"
pastel_peach  = "#F7C9A9"
warm_gray     = "#C7C7C7"


# ==========================
# Load data
# ==========================
def cutoff_to_str(c):
    return f"{c:.2f}".replace(".", "p")

cutoff_values = [0.2, 0.1, 0.05, 0.02, 0.0]

def load_data(SCHEME="cv1"):
    metrics_dict = {}
    preds_dict = {}

    for c in cutoff_values:
        c_str = cutoff_to_str(c)

        metrics_path = BASE_DIR / f"metrics_test_{SCHEME.lower()}_{c_str}.csv"
        preds_path   = BASE_DIR / f"test_predictions_{SCHEME.lower()}_{c_str}.csv"

        metrics_dict[c] = pd.read_csv(metrics_path).iloc[0].to_dict()
        preds_dict[c]   = pd.read_csv(preds_path)

    return metrics_dict, preds_dict

metrics_dict, preds_dict = load_data("cv1")


# ==========================
# Plotting
# ==========================
fig, axes = plt.subplots(3, 1, figsize=(8, 15), dpi=DPI)
axA, axB, axC = axes

# Increase spacing *between* the three panels
plt.subplots_adjust(hspace=0.45)


# ==========================
# Panel A: Line/scatter plot
# ==========================

cutoffs = cutoff_values

auc_means = [metrics_dict[c]["roc_auc_test"] for c in cutoffs]
auc_stds  = [metrics_dict[c]["roc_auc_test_std"] for c in cutoffs]

f1_means = [metrics_dict[c]["f1_macro_test"] for c in cutoffs]
f1_stds  = [metrics_dict[c]["f1_macro_test_std"] for c in cutoffs]

acc_means = [metrics_dict[c]["accuracy_test"] for c in cutoffs]
acc_stds  = [metrics_dict[c]["accuracy_test_std"] for c in cutoffs]

axA.errorbar(cutoffs, auc_means, yerr=auc_stds, color=main_blue,
             marker="o", capsize=4, label="AUC")
axA.errorbar(cutoffs, f1_means, yerr=f1_stds, color=pastel_red,
             marker="o", capsize=4, label="F1-score")
axA.errorbar(cutoffs, acc_means, yerr=acc_stds, color=pastel_green,
             marker="o", capsize=4, label="Accuracy")

axA.set_title(r"$\mathbf{A.}$" + "Model Performance vs. Bliss Neutrality Cutoff",
              fontsize=14, pad=14)
axA.set_xlabel("Bliss Neutrality Cutoff")
axA.set_ylabel("Performance Score")
axA.set_xticks(cutoffs)
axA.grid(alpha=0.3)
axA.legend(frameon=False)


# ==========================
# Panel B: Bar plot (mean ± std)
# ==========================

width = 0.25
x = np.arange(len(cutoffs))

axB.bar(x - width, auc_means, width, yerr=auc_stds, color=main_blue,
         alpha=0.8, label="AUC", capsize=4)
axB.bar(x, f1_means, width, yerr=f1_stds, color=pastel_red,
         alpha=0.8, label="F1-score", capsize=4)
axB.bar(x + width, acc_means, width, yerr=acc_stds, color=pastel_green,
         alpha=0.8, label="Accuracy", capsize=4)

axB.set_title(r"$\mathbf{B.}$" +
    "Metric Comparison Across Cutoffs",
    fontsize=14,
    pad=14
)
axB.set_ylabel("Mean Performance")
axB.set_xticks(x)
axB.set_xticklabels(cutoffs)
axB.legend(frameon=False)
axB.grid(axis="y", alpha=0.3)


# ==========================
# Panel C: Violin + box plot
# ==========================

data_for_violin = [preds_dict[c]["p_synergy"] for c in cutoffs]
violin_colors = [pastel_blue, pastel_peach, pastel_yellow, pastel_teal, warm_gray]

parts = axC.violinplot(
    data_for_violin,
    positions=range(len(cutoffs)),
    showmeans=True,
    showextrema=False
)

for body, col in zip(parts["bodies"], violin_colors):
    body.set_facecolor(col)
    body.set_edgecolor("black")
    body.set_alpha(0.8)

axC.boxplot(
    data_for_violin,
    positions=range(len(cutoffs)),
    widths=0.1,
    patch_artist=True,
    boxprops=dict(facecolor="white", alpha=0.7),
    medianprops=dict(color="black")
)

axC.set_title(r"$\mathbf{C.}$" +
    "p_synergy Distribution Across Cutoffs",
    fontsize=14,
    pad=14
)
axC.set_ylabel("Predicted p(Synergy)")
axC.set_xlabel("Bliss Neutrality Cutoff")   # ← Added
axC.set_xticks(range(len(cutoffs)))
axC.set_xticklabels(cutoffs)
axC.grid(alpha=0.3)


# ==========================
# Save
# ==========================
plt.savefig(FIG6_PATH_PNG, dpi=DPI)
plt.close()

print(f"Saved Figure 6 → {FIG6_PATH_PNG}")










