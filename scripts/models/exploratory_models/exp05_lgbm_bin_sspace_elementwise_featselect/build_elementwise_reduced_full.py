#!/usr/bin/env python3
"""
Experiment: build_elementwise_reduced_full

Config
- purpose: export a reduced elementwise feature matrix for reuse in downstream tasks
- feature_design: elementwise similarity (cos_elem_* and euc_elem_*)
- sspace: enabled (CC merged with S-space per drug via `inchikey`)
- feature set source: selected_features_cv1.txt produced by exp05 (selected on CV1 train split for bin_clas)

Goal
Recompute the full elementwise feature matrix for all combinations (including neutral class and full
Bliss values), then subset to the elementwise features selected in exp05. This produces a consistent
reduced feature space that can be reused for multi-class classification and regression experiments.

Outputs
- elementwise_features_filtered_cv1_full.csv: metadata + selected elementwise features for all rows

Data integrity note
All preprocessing (missing values, dtypes, column validation, and label construction from Bliss using the
±0.1 cutoff) is performed upstream. This script assumes the processed inputs are clean and consistent.
"""

import pandas as pd

from halo.paths import CC_FEATURES, SS_FEATURES, PROCESSED, MODEL_RESULTS
from halo.mappers.feature_mapper import FeatureMapper

cc_path = CC_FEATURES / "cc_features_concat_25x128.csv"
ss_path = SS_FEATURES / "sspace.csv"
combos_path = PROCESSED / "halo_training_dataset.csv"

# ==========================
# 1) Load original data
# ==========================
cc_df = pd.read_csv(cc_path).copy()
ss_df = pd.read_csv(ss_path).copy()
combinations_df = pd.read_csv(combos_path).copy()

features_cc_s = (
    cc_df
    .merge(ss_df, on="inchikey", how="inner", suffixes=("", "_s"))
)

# ==========================
# 2) Full elementwise matrix for ALL combos (3 classes + Bliss)
# ==========================
df_full = FeatureMapper().elementwise_similarity(combinations_df, features_cc_s)

# ==========================
# 3) Load selected feature names from exp05
# ==========================
feat_list_path = MODEL_RESULTS / "exp05_lgbm_bin_sspace_elementwise_featselect" / "selected_features_cv1.txt"

with open(feat_list_path) as f:
    selected_features = [line.strip() for line in f if line.strip() and not line.startswith("#")]

meta_cols = [
    "Drug A", "Drug B",
    "Drug A Inchikey", "Drug B Inchikey",
    "Strain", "Specie",
    "Bliss Score",
    "Interaction Type",
    "Source", "Drug Pair",
]

df_out = df_full[meta_cols + selected_features].copy()

out_path = MODEL_RESULTS / "exp05_lgbm_bin_sspace_elementwise_featselect" / "elementwise_features_filtered_cv1_full.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(out_path, index=False)
print("Wrote full reduced matrix to:", out_path)
print("Shape:", df_out.shape)
print(df_out["Interaction Type"].value_counts())
