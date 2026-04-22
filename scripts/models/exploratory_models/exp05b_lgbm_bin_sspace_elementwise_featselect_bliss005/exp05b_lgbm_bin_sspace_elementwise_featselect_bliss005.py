#!/usr/bin/env python3
"""
Experiment: exp05b_lgbm_bin_sspace_elementwise_featselect_bliss005

Config
- model: LightGBM (used only for feature importance ranking; not a final predictive model)
- task: feature selection for binary classification (synergy vs antagonism)
- feature_design: elementwise similarity (cos_elem_* and euc_elem_* from CC+S-space base features)
- sspace: enabled
- nested_cv: disabled
- cv_scheme: cv1_single
  - single held-out split with drug-pair disjointness (GroupShuffleSplit; groups=Drug Pair)
  - no inner folds, no nested CV
- bliss neutrality cutoff: ±0.05
  - labels are recomputed from Bliss in this script via `classify_interaction(bliss, additivity_cutoff=0.05)`
  - rows labeled neutral are excluded

Goal
Select a reduced set of elementwise features (from CC+S-space) under a stricter neutrality cutoff (±0.05),
and export both the reduced feature matrix and a mapping from selected elementwise features back to the
originating base feature for interpretability and reuse in later experiments.

Feature selection procedure (train split only)
1) Variance filter: drop zero-variance features.
2) Correlation prefilter: keep features with |corr(feature, y)| >= corr_min (fallback: keep all if none pass).
3) Model-based filter: fit a fixed LightGBM classifier on the training split and keep the top fraction of
   features ranked by feature_importances_.

Outputs
- elementwise_features_filtered_cv1.csv: metadata + selected elementwise features for kept rows
- selected_features_cv1.txt: newline-separated selected feature names
- selected_elementwise_feature_mapping.csv: elementwise feature → base feature metadata

Data integrity note
All preprocessing of raw inputs (missing values, dtypes, and column validation) is performed upstream.
This script assumes the processed inputs are clean and consistent; it recomputes interaction labels from
Bliss using the ±0.05 cutoff as part of the experiment definition.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from halo.paths import FEATURES, CC_FEATURES, SS_FEATURES, PROCESSED, MODEL_RESULTS
from halo.mappers.feature_mapper import FeatureMapper
from halo.shared_utils.data_io import classify_interaction


def main():
    # ==========================
    # 0) Basic config
    # ==========================
    SCHEME = "CV1"

    corr_min = 0.01         # minimal |corr(feature, y)| to keep
    keep_top_frac = 0.30    # keep top 30% of features by importance

    cc_path = CC_FEATURES / "cc_features_concat_25x128.csv"
    ss_path = SS_FEATURES / "sspace.csv"
    combos_path = PROCESSED / "halo_training_dataset.csv"

    out_dir = MODEL_RESULTS / "exp05b_lgbm_bin_sspace_elementwise_featselect_bliss005"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== EXP05b: elementwise feature selection ===\n")

    rng = np.random.default_rng(42)

    # ==========================
    # 1) Load dataset + elementwise features
    # ==========================
    cc_df = pd.read_csv(cc_path).copy()
    ss_df = pd.read_csv(ss_path).copy()
    combinations_df = pd.read_csv(combos_path).copy()

    features_cc_s = (
        cc_df
        .merge(ss_df, on="inchikey", how="inner", suffixes=("", "_s"))
    )

    df = FeatureMapper().elementwise_similarity(combinations_df, features_cc_s)

    # binary only
    df['Interaction Type'] = df['Bliss Score'].apply(
        lambda x: classify_interaction(x, additivity_cutoff=0.05))
    df = df[df['Interaction Type'].isin(['synergy', 'antagonism'])].copy()
    print(df['Interaction Type'].value_counts())

    # ==========================
    # 2) Feature columns
    # ==========================
    drop_cols = [
        "Drug A", "Drug B",
        "Drug A Inchikey", "Drug B Inchikey",
        "Strain", "Specie", "Bliss Score",
        "Interaction Type", "Source", "Drug Pair",
    ]
    feat_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feat_cols].copy()
    y = df["Interaction Type"].copy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    pairs = df["Drug Pair"].astype(str).values

    # ==========================
    # 3) CV1 split
    # ==========================
    def make_split_cv1():
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tr_idx, te_idx = next(gss.split(X, y_enc, groups=pairs))
        return tr_idx, te_idx

    tr_idx, te_idx = make_split_cv1()

    X_tr = X.iloc[tr_idx].reset_index(drop=True)
    X_te = X.iloc[te_idx].reset_index(drop=True)
    y_tr = y_enc[tr_idx]
    y_te = y_enc[te_idx]

    # ==========================
    # 4) Step A: variance + correlation filters
    # ==========================
    print("\n--- Step A: variance + correlation filters ---")

    var_series = X_tr.var()
    kept_after_var = [c for c in feat_cols if var_series[c] > 0.0]

    kept_after_corr = []
    y_tr_s = pd.Series(y_tr)

    for col in kept_after_var:
        corr = X_tr[col].corr(y_tr_s)
        if corr is not None and np.isfinite(corr) and abs(corr) >= corr_min:
            kept_after_corr.append(col)

    if not kept_after_corr:
        kept_after_corr = kept_after_var.copy()

    # ==========================
    # 5) Step B: LGBM importance filter
    # ==========================
    print("\n--- Step B: LightGBM importance filter ---")

    fs_feat_cols = kept_after_corr

    m_fs = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        random_state=777,
        n_jobs=1,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=15,
        min_data_in_leaf=200,
        feature_fraction=0.4,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=50.0,
        lambda_l1=0.0,
        max_bin=127,
        min_gain_to_split=0.05,
    )

    m_fs.fit(X_tr[fs_feat_cols], y_tr)

    importances = m_fs.feature_importances_
    feat_imp = pd.Series(importances, index=fs_feat_cols).sort_values(ascending=False)

    n_fs = len(feat_imp)
    n_keep = max(1, int(n_fs * keep_top_frac))
    selected_features = feat_imp.index[:n_keep].tolist()
    print(f'Number of selected features in exp05b: {len(selected_features)}')

    # ==========================
    # 6) Save reduced dataset
    # ==========================
    meta_cols = [
        "Drug A", "Drug B",
        "Drug A Inchikey", "Drug B Inchikey",
        "Strain", "Specie",
        "Bliss Score",
        "Interaction Type",
        "Source", "Drug Pair",
    ]

    df_out = df[meta_cols + selected_features].copy()
    out_csv = out_dir / f"elementwise_features_filtered_{SCHEME.lower()}.csv"
    df_out.to_csv(out_csv, index=False)

    # save list only
    feat_txt = out_dir / f"selected_features_{SCHEME.lower()}.txt"
    with open(feat_txt, "w") as f:
        for col in selected_features:
            f.write(col + "\n")

    print("\nSaved filtered dataset & feature list.")

    # ==========================
    # 7) Build mapping from elementwise -> original CC/S feature
    # ==========================
    print("\n--- Building elementwise→base mapping ---")

    # original base feature order used inside elementwise_similarity
    ignore = {"drug", "inchikey", "level"}
    features_cols = [c for c in features_cc_s.columns if c not in ignore]

    # load metadata
    meta_path = FEATURES / "feature_metadata_cc_s_full.csv"
    feature_meta = pd.read_csv(meta_path).set_index("original_name")

    rows = []
    for col in selected_features:    
        # "cos_elem_2219" or "euc_elem_556", fixed names
        parts = col.split("_")
        kind = parts[0]          # "cos" or "euc"
        idx = int(parts[-1])     # 2219, 556, ...

        base_name = features_cols[idx]
        meta_row = feature_meta.loc[base_name]

        rows.append({
            "elementwise_feature": col,
            "metric": "cosine" if kind == "cos" else "euclidean",
            "base_feature": base_name,
            "space": meta_row["space"],
            "dimension": meta_row["dimension"],
        })

    mapping_df = pd.DataFrame(rows)
    mapping_path = out_dir / "selected_elementwise_feature_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)

    print("Saved mapping to:", mapping_path)
    print("\n=== EXP05b DONE ===\n")


if __name__ == "__main__":
    main()
