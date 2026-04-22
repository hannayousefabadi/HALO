#!/usr/bin/env python3
"""
Script: exp06b_cc_vs_ss_analysis.py (HALO-S-CV1 feature-group analysis)

Post-hoc feature-group importance analysis for exp06b.

Purpose
- Load the aggregated LightGBM feature importances produced by exp06b
  (feature_importances_cv1.csv; importance_gain_mean ± std).
- Classify each reduced elementwise feature as originating from:
    - CC : Chemical Checker base dimensions
    - SS : strain-space (S-space) base dimensions
  using feature_metadata_cc_s_full.csv to infer the CC dimensionality boundary.
- Quantify the contribution of CC vs SS to total (normalized) gain.
- Count how many CC/SS features appear among the top-k ranked features.

Inputs
- MODEL_RESULTS/exp06b_lgbm_bin_sspace_elementwise_reduced_nestedcv_bliss005/
    - feature_importances_cv1.csv
- FEATURES/
    - feature_metadata_cc_s_full.csv

Grouping logic
- Elementwise feature names are expected to follow: {cos|euc}_elem_{idx}.
- Parse the trailing integer idx.
- Infer n_cc_dims from metadata as: max(CC dimension) + 1.
- Assign group:
    - CC if idx < n_cc_dims
    - SS if idx >= n_cc_dims
  (Any non-matching feature name is labeled "Unknown".)

Outputs (written into the exp06b result directory)
- cc_vs_ss_importance_summary.csv : per-group counts + total normalized gain + fraction of total gain
- cc_vs_ss_topk_counts.csv        : CC vs SS counts within top k features (k = 20, 50, 100)

**Data integrity note:**
All preprocessing (NA handling, dtype enforcement, column validation, etc.) was completed upstream.
This script assumes clean, validated input data and an existing exp06b importance file.
"""

import pandas as pd

from pathlib import Path
from halo.paths import FEATURES, MODEL_RESULTS

RESULT_DIR = MODEL_RESULTS / "exp06b_lgbm_bin_sspace_elementwise_reduced_nestedcv_bliss005"
FI_PATH = RESULT_DIR / "feature_importances_cv1.csv"

FEATURE_META_PATH = FEATURES / "feature_metadata_cc_s_full.csv"

OUT_SUMMARY_PATH = RESULT_DIR / "cc_vs_ss_importance_summary.csv"
OUT_TOPK_PATH = RESULT_DIR / "cc_vs_ss_topk_counts.csv"

# ==========================
# 1) Load data
# ==========================

def load_importances(fi_path: Path) -> pd.DataFrame:
    if not fi_path.exists():
        raise FileNotFoundError(f"Feature importance file not found: {fi_path}")

    df = pd.read_csv(fi_path)

    # Enforce: exp06b always outputs importance_gain_mean
    if "importance_gain_mean" not in df.columns:
        raise ValueError(
            "Expected column 'importance_gain_mean' in feature importance file. "
            f"Found: {list(df.columns)}"
        )

    gain_col = "importance_gain_mean"

    # Create normalized gain
    if "importance_gain_norm" not in df.columns:
        total_gain = df[gain_col].sum()
        if total_gain <= 0:
            raise ValueError("Total importance_gain_mean is non-positive; cannot normalize.")
        df["importance_gain_norm"] = df[gain_col] / total_gain

    # Sort by importance_gain_mean
    df = df.sort_values(gain_col, ascending=False).reset_index(drop=True)
    return df



def load_feature_meta(meta_path: Path) -> tuple[pd.DataFrame, int]:
    """
    Load CC+SS metadata and infer the number of CC dimensions.

    Returns
    -------
    meta : DataFrame indexed by original_name
    n_cc_dims : int
        Number of CC base dimensions (dim_0 .. dim_{n_cc_dims-1}).
        Used to partition elementwise indices into CC vs SS.
    """
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    meta = pd.read_csv(meta_path)

    if "original_name" not in meta.columns:
        raise ValueError("feature_metadata_cc_s_full.csv must have 'original_name' column.")

    meta = meta.set_index("original_name")

    cc_dims = meta[meta["space"] == "CC"]["dimension"].dropna()
    if cc_dims.empty:
        raise ValueError("No CC dimensions found in metadata (space == 'CC').")

    n_cc_dims = int(cc_dims.max()) + 1
    return meta, n_cc_dims


# ==========================
# 2) CC vs SS grouping logic
# ==========================

def feature_group_from_name(feat_name: str, n_cc_dims: int) -> str:
    """
    Map an elementwise feature name like "euc_elem_3141" or "cos_elem_42"
    to a group: 'CC' or 'SS', using the same logic as make_fig3.py.

    - parse the trailing index: e.g. "euc_elem_3141" -> 3141
    - if idx < n_cc_dims -> CC
    - else               -> SS
    """
    name = feat_name.lower()

    if not (name.startswith("cos_elem_") or name.startswith("euc_elem_")):
        # In exp06b all features should be elementwise; if not, mark as Unknown.
        return "Unknown"

    try:
        idx = int(name.split("_")[-1])
    except ValueError:
        return "Unknown"

    return "SS" if idx >= n_cc_dims else "CC"


def add_group_column(df_imp: pd.DataFrame, n_cc_dims: int) -> pd.DataFrame:
    df = df_imp.copy()
    df["group"] = df["feature"].apply(lambda f: feature_group_from_name(f, n_cc_dims))
    return df


# ==========================
# 3) Summary computations
# ==========================

def summarize_cc_vs_ss(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-group totals:
    - number of features
    - total normalized gain
    - fraction of total normalized gain
    """
    if "importance_gain_norm" not in df.columns or "group" not in df.columns:
        raise ValueError("DataFrame must have 'importance_gain_norm' and 'group' columns.")

    grouped = (
        df.groupby("group")
        .agg(
            n_features=("feature", "count"),
            total_gain_norm=("importance_gain_norm", "sum"),
        )
        .reset_index()
    )

    total_gain = grouped["total_gain_norm"].sum()
    grouped["fraction_of_total_gain"] = grouped["total_gain_norm"] / total_gain

    # Sort: most important group first
    grouped = grouped.sort_values("total_gain_norm", ascending=False).reset_index(drop=True)
    return grouped


def summarize_topk_counts(df: pd.DataFrame, ks=(20, 50, 100)) -> pd.DataFrame:
    """
    For each k in ks, count how many CC vs SS features appear in the top-k.
    """
    rows = []
    for k in ks:
        topk = df.head(k)
        counts = topk["group"].value_counts()
        for group, n in counts.items():
            rows.append({"k": k, "group": group, "n_features": int(n)})

    if not rows:
        return pd.DataFrame(columns=["k", "group", "n_features"])

    out = pd.DataFrame(rows).sort_values(["k", "group"]).reset_index(drop=True)
    return out


# ==========================
# 4) Main
# ==========================

def main():
    print("\n=== CC vs SS feature-group importance analysis for exp06b (PHO-CV1) ===\n")

    print("Loading feature importances from:", FI_PATH)
    df_imp = load_importances(FI_PATH)
    print("Feature importance shape:", df_imp.shape)

    print("Loading feature metadata from:", FEATURE_META_PATH)
    meta, n_cc_dims = load_feature_meta(FEATURE_META_PATH)
    print(f"Inferred number of CC base dimensions: n_cc_dims = {n_cc_dims}")

    print("\nAssigning group = CC vs SS for each feature...")
    df_imp = add_group_column(df_imp, n_cc_dims)

    # Basic sanity check: all elementwise features should map to CC or SS.
    group_counts = df_imp["group"].value_counts(dropna=False)
    print("\nGroup label counts:")
    print(group_counts)

    # ---- Summary 1: total gain per group ----
    summary = summarize_cc_vs_ss(df_imp)
    print("\nTotal normalized gain per group (CC vs SS):")
    print(summary)

    print("\nSaving group summary CSV to:", OUT_SUMMARY_PATH)
    summary.to_csv(OUT_SUMMARY_PATH, index=False)

    # ---- Summary 2: top-k group counts ----
    topk_df = summarize_topk_counts(df_imp, ks=(20, 50, 100))
    print("\nTop-k group counts (how many CC vs SS in top 20 / 50 / 100):")
    print(topk_df)

    print("Saving top-k counts CSV to:", OUT_TOPK_PATH)
    topk_df.to_csv(OUT_TOPK_PATH, index=False)

    print("\nDone.\n")
    print("Use:")
    print(f"  - {OUT_SUMMARY_PATH.name} for CC vs SS contribution numbers in Results/Discussion")
    print(f"  - {OUT_TOPK_PATH.name} for describing top-k feature composition (e.g. “almost all top-20 are CC”).")


if __name__ == "__main__":
    main()
