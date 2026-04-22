#!/usr/bin/env python3
"""
Generate Figure 2:

Output panels:
- A: Metric summary (train vs test, aggregated)
- B: Confusion matrix (test set, average fold)
- C: ROC curve (test set, aggregated, synergy = positive)
- D: Precision–recall curve (test set, aggregated, synergy = positive)
- E: Per-fold test performance (SynChecker-CV1 outer folds)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# ==========================
# Paths
# ==========================

from halo.paths import MODEL_RESULTS, FIGURES 

PLOT_DIR = FIGURES / "main"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
FIG2_PATH_PNG = PLOT_DIR / "fig2.png"

RESULT_DIR = MODEL_RESULTS / "exp06d_lgbm_bin_nosspace_elementwise_reduced_nestedcv_bliss005"
SCHEME_SUFFIX = "cv1"
METRICS_FOLDS_PATH = RESULT_DIR / f"metrics_per_fold_{SCHEME_SUFFIX}.csv"

# ==========================
# Load data
# ==========================

def load_data():
    """Load metrics + prediction CSVs (aggregated) and per-fold metrics."""
    metrics_train_path = RESULT_DIR / f"metrics_train_{SCHEME_SUFFIX}.csv"
    metrics_test_path = RESULT_DIR / f"metrics_test_{SCHEME_SUFFIX}.csv"
    train_pred_path = RESULT_DIR / f"train_predictions_{SCHEME_SUFFIX}.csv"
    test_pred_path = RESULT_DIR / f"test_predictions_{SCHEME_SUFFIX}.csv"

    metrics_train = pd.read_csv(metrics_train_path).iloc[0].to_dict()
    metrics_test = pd.read_csv(metrics_test_path).iloc[0].to_dict()
    df_train = pd.read_csv(train_pred_path).copy()
    df_test = pd.read_csv(test_pred_path).copy()

    # per-fold test metrics: one row per outer fold
    metrics_folds = pd.read_csv(METRICS_FOLDS_PATH).copy()

    return metrics_train, metrics_test, df_train, df_test, metrics_folds


# ==========================
# Colors & style
# ==========================

COLOR_TRAIN = "#1f77b4"      # dark blue
COLOR_TEST = "#8DC5C1"       # pastel teal
COLOR_BASELINE = "#7BAFD4"   # pastel blue

COLOR_FOLD_ACC = "#7BAFD4"   # Pastel Blue
COLOR_FOLD_F1W = "#8DC5C1"   # Pastel Teal
COLOR_FOLD_AUC = "#A7D7A0"   # Pastel Green

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


# ==========================
# Panel A
# ==========================

def plot_panel_A(ax, metrics_train, metrics_test):
    metric_keys = [
        ("accuracy", "accuracy_train", "accuracy_test"),
        ("F1 macro", "f1_macro_train", "f1_macro_test"),
        ("F1 weighted", "f1_weighted_train", "f1_weighted_test"),
        ("ROC-AUC", "roc_auc_train", "roc_auc_test"),
    ]

    labels = [m[0] for m in metric_keys]
    train_vals = [metrics_train[m[1]] for m in metric_keys]
    test_vals = [metrics_test[m[2]] for m in metric_keys]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, train_vals, width, label="Train", color=COLOR_TRAIN)
    ax.bar(x + width/2, test_vals, width, label="Test", color=COLOR_TEST)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel("Score")

    ax.set_title(r"$\mathbf{A.}$" + "  Model performance")

    ax.legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")

    for i, v in enumerate(train_vals):
        ax.text(x[i] - width/2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(test_vals):
        ax.text(x[i] + width/2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)


# ==========================
# Panel B
# ==========================

CONF_CMAP = LinearSegmentedColormap.from_list(
    "conf_cmap", ["#ffffff", COLOR_TRAIN]
)

def plot_panel_B(ax, df_test):
    """
    Panel B: average confusion matrix over outer CV test folds.

    Steps:
    - split df_test by 'fold' (each is an outer-fold *test* set)
    - compute confusion matrix for each fold
    - sum them
    - divide by number of folds to get an "average test fold" confusion matrix
    """
    labels = ["antagonism", "synergy"]

    if "fold" not in df_test.columns:
        raise ValueError("df_test must contain a 'fold' column.")

    fold_ids = sorted(df_test["fold"].unique())
    n_folds = len(fold_ids)

    # fold-level class balance diagnostic
    fold_balance = (
        df_test.groupby("fold")["y_true_label"]
        .value_counts(normalize=False)
        .unstack(fill_value=0)
    )
    print("Per-fold test label counts:\n", fold_balance)

    # sum of per-fold test confusion matrices
    cm_sum = np.zeros((len(labels), len(labels)), dtype=float)

    for f in fold_ids:
        df_f = df_test[df_test["fold"] == f]

        cm_f = confusion_matrix(
            df_f["y_true_label"],
            df_f["y_pred_label"],
            labels=labels,
        ).astype(float)

        cm_sum += cm_f

    # average per fold (so total ≈ size of one test set)
    cm_avg = cm_sum / n_folds

    # to have an integer-looking counts, round:
    cm_avg_rounded = np.rint(cm_avg).astype(int)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_avg_rounded,
                                  display_labels=labels)
    disp.plot(cmap=CONF_CMAP, ax=ax, values_format="d", colorbar=False)

    ax.set_title(r"$\mathbf{B.}$" + "  Confusion matrix (avg over test folds)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

# ==========================
# Panel C
# ==========================

def plot_panel_C(ax, df_test):

    y_true = (df_test["y_true_label"] == "synergy").astype(int).values
    y_score = df_test["p_synergy"].values

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc_value = auc(fpr, tpr)

    ax.plot(fpr, tpr, linewidth=2,
            label=f"ROC (AUC = {roc_auc_value:.2f})",
            color=COLOR_TEST)

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1,
            color=COLOR_BASELINE)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(r"$\mathbf{C.}$" + "  ROC curve (held-out test)")
    ax.legend(loc="lower right", frameon=False)


# ==========================
# Panel D (PR curve)
# ==========================

def plot_panel_D(ax, df_test):

    y_true = (df_test["y_true_label"] == "synergy").astype(int).values
    y_score = df_test["p_synergy"].values

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    ax.step(recall, precision, where="post",
            label=f"PR (AP = {ap:.2f})", color=COLOR_TEST)

    baseline = y_true.mean()
    ax.hlines(baseline, 0, 1,
              linestyles="--",
              linewidth=0.8,
              color=COLOR_BASELINE,
              label=f"Baseline (pos frac = {baseline:.2f})")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(r"$\mathbf{D.}$" + "  Precision–recall curve (held-out test)")
    ax.legend(loc="upper right", frameon=False)


# ==========================
# Panel E (per-fold metrics)
# ==========================
def plot_panel_E(ax, metrics_folds: pd.DataFrame):
    """
    Per-fold test performance for exp06d (SynChecker-CV1 outer folds).

    Expects columns (from exp06d training script):
        - 'fold'
        - 'roc_auc_test'
        - 'accuracy_test'
        - 'f1_weighted_test'
    """
    # sort by fold just in case
    metrics_folds = metrics_folds.sort_values("fold").reset_index(drop=True)

    folds = metrics_folds["fold"].values
    acc = metrics_folds["accuracy_test"].values
    f1w = metrics_folds["f1_weighted_test"].values
    auc_vals = metrics_folds["roc_auc_test"].values

    x = np.arange(len(folds))
    width = 0.25

    ax.bar(x - width, acc, width, label="Accuracy", color=COLOR_FOLD_ACC)
    ax.bar(x,        f1w, width, label="F1 weighted", color=COLOR_FOLD_F1W)
    ax.bar(x + width, auc_vals, width, label="ROC-AUC", color=COLOR_FOLD_AUC)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {int(f)}" for f in folds])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")

    ax.set_title(r"$\mathbf{E.}$" + "  Per-fold held-out performance")
    ax.legend(frameon=False, ncol=3, loc="upper center")

    # tiny annotations above bars
    for i, v in enumerate(acc):
        ax.text(x[i] - width, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7)
    for i, v in enumerate(f1w):
        ax.text(x[i], v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7)
    for i, v in enumerate(auc_vals):
        ax.text(x[i] + width, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7)


# ==========================
# Assemble figure
# ==========================

def main():
    metrics_train, metrics_test, df_train, df_test, metrics_folds = load_data()

    # 3 rows x 2 columns grid; E spans the bottom row
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[1.0, 1.2],
        height_ratios=[1.0, 1.0, 0.9],
    )

    axA = fig.add_subplot(gs[0, 0])
    axC = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[2, :])

    plot_panel_A(axA, metrics_train, metrics_test)
    plot_panel_B(axB, df_test)
    plot_panel_C(axC, df_test)
    plot_panel_D(axD, df_test)
    plot_panel_E(axE, metrics_folds)

    fig.tight_layout()

    fig.savefig(FIG2_PATH_PNG, dpi=600)
    plt.close(fig)

    print("Saved Fig 2 PNG to:", FIG2_PATH_PNG)


if __name__ == "__main__":
    main()
