#!/usr/bin/env python3
"""
Experiment: build_elementwise_reduced_full_for_all_possible_pairs

Config
- feature source: Chemical Checker only (no S-space)
- feature_design: elementwise similarity (cos_elem_* and euc_elem_*)
- selected feature set: derived from exp05d-style feature selection on labeled training data
  - labels recomputed from Bliss using neutrality cutoff ±0.05
  - cv_scheme for selection: cv1_single (single drug-pair-disjoint split; selection on train only)
- output: reduced elementwise feature matrix for all unordered pairs among the candidate drug set
  (used for novel-pair scoring downstream)

Goal
1) Run CC-only elementwise feature selection on labeled combinations (synergy vs antagonism, ±0.05 cutoff).
2) Enumerate all unordered pairs among the unique compounds and compute CC-only elementwise features.
3) Subset to the selected feature list and write a reduced all-pairs matrix for downstream ranking.

Data integrity note
All preprocessing (missing values, dtypes, column validation, and base CC feature construction) is performed
upstream. This script assumes clean, validated CC features and combination metadata. Interaction labels are
recomputed from Bliss using ±0.05 as part of the feature-selection definition.
"""

import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from halo.paths import CC_FEATURES, PROCESSED, MODEL_RESULTS
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
    combos_path = PROCESSED / "halo_training_dataset.csv"

    out_dir = MODEL_RESULTS / "exp05d_lgbm_bin_nosspace_elementwise_featselect_bliss005"
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print("\n=== EXP05d: feature selection + all possible pairs ===\n")

    rng = np.random.default_rng(42)

    # ==========================
    # 1) Load CC features & labeled combos
    # ==========================
    cc_df = pd.read_csv(cc_path).copy()
    features_cc = cc_df.copy()

    combos_df = pd.read_csv(combos_path).copy()
    print("Loaded labeled combinations df shape:", combos_df.shape)

    # Some pipelines already have a "Drug Pair" column; if missing, create it
    if "Drug Pair" not in combos_df.columns:
        combos_df["Drug Pair"] = combos_df.apply(
            lambda r: f"{r['Drug A']} + {r['Drug B']}", axis=1
        )

    # ==========================
    # 1a) Elementwise features for LABELED combos (for feature selection)
    # ==========================
    print("\n--- Computing elementwise CC-only features for LABELED combos (for FS) ---")
    df_labeled = FeatureMapper().elementwise_similarity(combos_df, features_cc)

    # binary only (for feature selection)
    df_labeled["Interaction Type"] = df_labeled["Bliss Score"].apply(
        lambda x: classify_interaction(x, additivity_cutoff=0.05)
    )
    df_labeled = df_labeled[
        df_labeled["Interaction Type"].isin(["synergy", "antagonism"])
    ].copy()
    print(df_labeled["Interaction Type"].value_counts())

    # ==========================
    # 2) Feature columns (like original exp05d)
    # ==========================
    drop_cols = [
        "Drug A", "Drug B",
        "Drug A Inchikey", "Drug B Inchikey",
        "Strain", "Specie", "Bliss Score",
        "Interaction Type", "Source", "Drug Pair",
    ]
    feat_cols = [c for c in df_labeled.columns if c not in drop_cols]

    X = df_labeled[feat_cols].copy()
    y = df_labeled["Interaction Type"].copy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    pairs = df_labeled["Drug Pair"].astype(str).values

    # ==========================
    # 3) CV1 split (same logic as exp05b)
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
    print("\n--- Step A: variance + correlation filters (CC-only) ---")

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
    print("\n--- Step B: LightGBM importance filter (CC-only) ---")

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
    print(f"Number of selected CC-only features in exp05d (labeled): {len(selected_features)}")

    # ==========================
    # 6) Build ALL POSSIBLE PAIRS by Inchikey
    # ==========================
    print("\n--- Building all possible pairs from unique Inchikeys ---")

    # Get drugs & inchikeys from combos_df
    drugs_A = combos_df[["Drug A", "Drug A Inchikey"]].rename(
        columns={"Drug A": "Drug", "Drug A Inchikey": "Inchikey"}
    )
    drugs_B = combos_df[["Drug B", "Drug B Inchikey"]].rename(
        columns={"Drug B": "Drug", "Drug B Inchikey": "Inchikey"}
    )

    drugs_all_raw = pd.concat([drugs_A, drugs_B], ignore_index=True)

    # Deduplicate by Inchikey (one canonical name per Inchikey)
    drugs_all = (
        drugs_all_raw
        .dropna(subset=["Inchikey"])
        .sort_values("Drug")           # deterministic choice of name
        .drop_duplicates(subset="Inchikey", keep="first")
        .reset_index(drop=True)
    )

    print("Number of unique Inchikeys (unique compounds):", len(drugs_all))

    inchikeys = sorted(drugs_all["Inchikey"].astype(str).tolist())
    ik_to_name = dict(zip(drugs_all["Inchikey"].astype(str), drugs_all["Drug"].astype(str)))

    # Generate all unordered pairs of Inchikeys (A < B lexicographically)
    rows = []
    for ik_a, ik_b in itertools.combinations(inchikeys, 2):
        da = ik_to_name[ik_a]
        db = ik_to_name[ik_b]

        # Inchikey-based pair ID (order-invariant)
        pair_id = "||".join(sorted([ik_a, ik_b]))

        rows.append(
            {
                "Drug A": da,
                "Drug B": db,
                "Drug A Inchikey": ik_a,
                "Drug B Inchikey": ik_b,
                "Pair_ID": pair_id,
                "Strain": np.nan,
                "Specie": np.nan,
                "Bliss Score": np.nan,
                "Interaction Type": np.nan,
                "Source": np.nan,
            }
        )

    combinations_allpairs = pd.DataFrame(rows)

    # Human-readable Drug Pair name (based on canonical names)
    combinations_allpairs["Drug Pair"] = combinations_allpairs.apply(
        lambda r: f"{r['Drug A']} + {r['Drug B']}", axis=1
    )

    print("All-pairs combinations df shape:", combinations_allpairs.shape)

    # ==========================
    # 7) Elementwise features for ALL POSSIBLE PAIRS
    # ==========================
    print("\n--- Computing elementwise CC-only features for ALL possible pairs ---")
    df_allpairs = FeatureMapper().elementwise_similarity(combinations_allpairs, features_cc)

    # ==========================
    # 8) Save reduced ALL-PAIRS dataset
    # ==========================
    meta_cols = [
        "Drug A", "Drug B",
        "Drug A Inchikey", "Drug B Inchikey",
        "Pair_ID",
        "Strain", "Specie",
        "Bliss Score",
        "Interaction Type",
        "Source", "Drug Pair",
    ]

    for col in meta_cols:
        if col not in df_allpairs.columns:
            df_allpairs[col] = np.nan

    df_out_allpairs = df_allpairs[meta_cols + selected_features].copy()

    out_csv = out_dir / "elementwise_features_filtered_all_possible_pairs_cv1_cc_only.csv"
    out_csv.mkdir(parents=True, exist_ok=True)
    df_out_allpairs.to_csv(out_csv, index=False)

    print("\nSaved ALL-PAIRS reduced file to:", out_csv)
    print("\n=== EXP05d for all possible pairs DONE ===\n")


if __name__ == "__main__":
    main()



