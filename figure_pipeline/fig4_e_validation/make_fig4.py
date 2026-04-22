#!/usr/bin/env python3
"""
Generate Figure 4:

External validation sanity check on Chandrasekaran et al. dataset (EXP06d).
using exp06d LightGBM model (CC-only, elementwise features).

Panel outputs:
- Fig4_panelB.png
- Fig4_panelC.png
- Fig4_panelD.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score, average_precision_score

# ==========================
# Paths
# ==========================
from halo.paths import FIGURE_PIPELINE, MODEL_RESULTS

BASE_DIR = MODEL_RESULTS / "e_validation" / "external_eval_chandrasekaran"
OUT_DIR = FIGURE_PIPELINE / "fig4_e_validation" / "fig4_panels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

roc_path = BASE_DIR / "external_roc_curve_chandrasekaran.csv"
pr_path = BASE_DIR / "external_pr_curve_chandrasekaran.csv"
cm_path = BASE_DIR / "external_confusion_matrix_chandrasekaran.csv"
pred_path = BASE_DIR / "external_predictions_chandrasekaran.csv"

# ==========================
# Palette
# ==========================

main_blue   = "#1f77b4"

pastel_blue   = "#7BAFD4"
pastel_teal   = "#8DC5C1"
pastel_green  = "#A7D7A0"
pastel_yellow = "#F4E3A3"
pastel_peach  = "#F7C9A9"
pastel_red    = "#E8A5A5"
warm_gray     = "#C7C7C7"

CONF_CMAP = LinearSegmentedColormap.from_list(
    "conf_cmap", ["#ffffff", main_blue]
)


def load_data():
    roc_df = pd.read_csv(roc_path).copy()
    pr_df = pd.read_csv(pr_path).copy()
    cm_df = pd.read_csv(cm_path, index_col=0).copy()
    pred_df = pd.read_csv(pred_path).copy()

    # allow both naming variants for alpha
    alpha_col = None
    for candidate in ["Experimental Interaction Score", "alpha", "Alpha", "Score"]:
        if candidate in pred_df.columns:
            alpha_col = candidate
            break
    if alpha_col is None:
        raise ValueError("Could not find alpha column in external_predictions CSV.")

    return roc_df, pr_df, cm_df, pred_df, alpha_col


def make_fig4():
    roc_df, pr_df, cm_df, pred_df, alpha_col = load_data()

    # ===== Basic metrics =====
    y_true = (pred_df["y_true_label"] == "synergy").astype(int).values
    p_synergy = pred_df["p_synergy"].values

    auc = roc_auc_score(y_true, p_synergy)
    ap = average_precision_score(y_true, p_synergy)

    # Confusion matrix from CSV (rows: true antag / syn, cols: pred antag / syn)
    cm = cm_df.values.astype(int)
    tn_csv, fp_csv, fn_csv, tp_csv = cm.ravel()

    # Precompute ROC + PR arrays
    fpr = roc_df["fpr"].values
    tpr = roc_df["tpr"].values
    rec = pr_df["recall"].values
    prec = pr_df["precision"].values
    pos_rate = y_true.mean()

    # Prepare stuff for scatter
    alpha_vals = pred_df[alpha_col].values
    labels_true = pred_df["y_true_label"].values
    mask_syn = labels_true == "synergy"
    mask_ant = labels_true == "antagonism"

    # prediction threshold in probability
    pred_syn = p_synergy >= 0.5
    true_syn = mask_syn

    # Confusion groups based on panel itself
    tn_mask = (~true_syn) & (~pred_syn)
    fp_mask = (~true_syn) & pred_syn
    fn_mask = true_syn & (~pred_syn)
    tp_mask = true_syn & pred_syn

    tn = int(tn_mask.sum())
    fp = int(fp_mask.sum())
    fn = int(fn_mask.sum())
    tp = int(tp_mask.sum())

    # sanity check – should match cm_df
    assert tn == tn_csv and fp == fp_csv and fn == fn_csv and tp == tp_csv, \
        "Confusion numbers from points do not match CSV confusion matrix!"

    # x-position: median α of true-synergy (left) vs true-antagonism (right)
    x_syn = np.median(alpha_vals[true_syn])
    x_ant = np.median(alpha_vals[~true_syn])

    # ==========================
    # Main figure: 3 panels (B, C, D)
    # ==========================
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2])

    ax_roc     = fig.add_subplot(gs[0, 0])   # Panel B
    ax_pr      = fig.add_subplot(gs[0, 1])   # Panel C
    ax_scatter = fig.add_subplot(gs[1, :])   # Panel D (bottom, full width)

    # --------------------------
    # Panel B: ROC curve
    # --------------------------
    ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color=main_blue, lw=2)
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color=warm_gray, lw=1)

    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title(r"$\mathbf{B.}$" + " ROC on external set", loc="center", pad=14)
    ax_roc.legend(frameon=False)
    ax_roc.grid(alpha=0.2, color=warm_gray)

    # --------------------------
    # Panel C: Precision–Recall
    # --------------------------
    ax_pr.plot(rec, prec, color=main_blue, lw=2, label=f"AP = {ap:.2f}")
    ax_pr.hlines(pos_rate, 0, 1, linestyle="--", color=warm_gray, lw=1,
                 label=f"Pos. prevalence = {pos_rate:.2f}")

    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    ax_pr.set_xlabel("Recall (synergy)")
    ax_pr.set_ylabel("Precision (synergy)")
    ax_pr.set_title(r"$\mathbf{C.}$" + " Precision–recall curve", loc="center", pad=14)
    ax_pr.legend(frameon=False)
    ax_pr.grid(alpha=0.2, color=warm_gray)

    # --------------------------
    # Panel D: α vs P(synergy) *as* confusion matrix
    # --------------------------
    ax_scatter.scatter(
        alpha_vals[mask_ant], p_synergy[mask_ant],
        s=30, alpha=0.8, edgecolor="none", color=pastel_red, label="True antagonism"
    )
    ax_scatter.scatter(
        alpha_vals[mask_syn], p_synergy[mask_syn],
        s=30, alpha=0.8, edgecolor="none", color=pastel_green, label="True synergy"
    )

    # thresholds
    ax_scatter.axvline(-0.5, linestyle="--", color=warm_gray, lw=1)
    ax_scatter.axvline(1.0,  linestyle="--", color=warm_gray, lw=1)
    ax_scatter.axhline(0.5,  linestyle="--", color=warm_gray, lw=1)

    ax_scatter.set_xlabel("Experimental interaction score (Loewe α)")
    ax_scatter.set_ylabel("Predicted P(synergy)")
    ax_scatter.set_ylim(-0.02, 1.02)
    ax_scatter.set_title(
        r"$\mathbf{D.}$" + r" $\alpha$-score vs predicted synergy probability",
        loc="center",
        pad=14,
    )

    # Legend outside the plot
    ax_scatter.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5)
    )

    # aligned label rows in y
    ymin, ymax = ax_scatter.get_ylim()
    y_low  = ymin + 0.20 * (ymax - ymin)   # predicted antagonism
    y_high = ymin + 0.80 * (ymax - ymin)   # predicted synergy

    def put_label(ax, x, y, text):
        ax.text(
            x, y, text,
            ha="center", va="center",
            fontsize=9, weight="bold",
            bbox=dict(boxstyle="round", facecolor="white",
                      edgecolor=warm_gray, alpha=0.85),
            zorder=5,
        )

    # row: predicted (low/high), column: true (syn/ant)
    put_label(ax_scatter, x_syn, y_high, f"TP = {tp}")   # true syn, predicted syn
    put_label(ax_scatter, x_ant, y_high, f"FP = {fp}")   # true antag, predicted syn
    put_label(ax_scatter, x_syn, y_low,  f"FN = {fn}")   # true syn, predicted antag
    put_label(ax_scatter, x_ant, y_low,  f"TN = {tn}")   # true antag, predicted antag

    ax_scatter.grid(alpha=0.2, color=warm_gray)

    # ----- layout + save combined figure -----
    fig.tight_layout()
    fig.subplots_adjust(
        hspace=0.6, wspace=0.4,
        left=0.10, right=0.80,  
        top=0.94, bottom=0.09
    )

    # out_png = OUT_DIR / "Fig4.png"
    # fig.savefig(out_png, dpi=600)
    # print("Saved combined Fig.4 to:")
    # print("  ", out_png)

    # =======================================================
    # Save PANELS B, C, D independently
    # =======================================================

    # ----- Panel B: ROC -----
    figB, axB = plt.subplots(figsize=(4.5, 4))
    axB.plot(fpr, tpr, color=main_blue, lw=2, label=f"AUC = {auc:.2f}")
    axB.plot([0, 1], [0, 1], linestyle="--", color=warm_gray, lw=1)
    axB.set_xlim(0, 1)
    axB.set_ylim(0, 1)
    axB.set_xlabel("False positive rate")
    axB.set_ylabel("True positive rate")
    axB.set_title(r"$\mathbf{B.}$" + " ROC on external set")
    axB.legend(frameon=False)
    axB.grid(alpha=0.2, color=warm_gray)
    figB.tight_layout()
    out_png_B = OUT_DIR / "fig4_panelB_ROC.png"
    figB.savefig(out_png_B, dpi=600)
    plt.close(figB)

    # ----- Panel C: PR curve -----
    figC, axC = plt.subplots(figsize=(4.5, 4))
    axC.plot(rec, prec, color=main_blue, lw=2, label=f"AP = {ap:.2f}")
    axC.hlines(pos_rate, 0, 1, linestyle="--", color=warm_gray, lw=1,
               label=f"Pos. prevalence = {pos_rate:.2f}")
    axC.set_xlim(0, 1)
    axC.set_ylim(0, 1)
    axC.set_xlabel("Recall (synergy)")
    axC.set_ylabel("Precision (synergy)")
    axC.set_title(r"$\mathbf{C.}$" + " Precision–recall curve")
    axC.legend(frameon=False)
    axC.grid(alpha=0.2, color=warm_gray)
    figC.tight_layout()
    out_png_C = OUT_DIR / "fig4_panelC_PR.png"
    figC.savefig(out_png_C, dpi=600)
    plt.close(figC)

    # ----- Panel D: scatter + confusion labels -----
    figD, axD = plt.subplots(figsize=(6, 4))

    axD.scatter(alpha_vals[mask_ant], p_synergy[mask_ant],
                s=30, alpha=0.8, edgecolor="none", color=pastel_red,
                label="True antagonism")
    axD.scatter(alpha_vals[mask_syn], p_synergy[mask_syn],
                s=30, alpha=0.8, edgecolor="none", color=pastel_green,
                label="True synergy")

    axD.axvline(-0.5, linestyle="--", color=warm_gray, lw=1)
    axD.axvline(1.0,  linestyle="--", color=warm_gray, lw=1)
    axD.axhline(0.5,  linestyle="--", color=warm_gray, lw=1)

    axD.set_xlabel("Experimental interaction score (Loewe α)")
    axD.set_ylabel("Predicted P(synergy)")
    axD.set_ylim(-0.02, 1.02)
    axD.set_title(
        r"$\mathbf{D.}$" + r" $\alpha$-score vs predicted synergy probability"
    )

    # legend inside or outside – your call; here I keep it outside like main fig
    axD.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

    put_label(axD, x_syn, y_high, f"TP = {tp}")
    put_label(axD, x_ant, y_high, f"FP = {fp}")
    put_label(axD, x_syn, y_low,  f"FN = {fn}")
    put_label(axD, x_ant, y_low,  f"TN = {tn}")

    axD.grid(alpha=0.2, color=warm_gray)
    figD.tight_layout()
    out_png_D = OUT_DIR / "fig4_panelD_scatter_CM.png"
    figD.savefig(out_png_D, dpi=600)
    plt.close(figD)

    print("Saved independent panels:")
    print("  ", out_png_B)
    print("  ", out_png_C)
    print("  ", out_png_D)

    # finally close main fig
    plt.close(fig)


if __name__ == "__main__":
    make_fig4()
