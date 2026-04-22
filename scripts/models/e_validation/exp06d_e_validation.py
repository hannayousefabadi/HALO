#!/usr/bin/env python3
"""
External validation for EXP06d model on Chandrasekaran dataset.

Pipeline:

1. Load EXP06d training matrix:
   - CC-only elementwise features already selected by EXP05d
   - file: elementwise_features_filtered_cv1_cc_only.csv

2. Load best hyperparameters from EXP06d:
   - file: best_params_cv1.json

3. Run 5-fold CV1 with those fixed params (no nested search) for
   internal sanity and to keep a CV1 baseline consistent with EXP06d.

4. Train a final LightGBM model on ALL training rows.

5. External evaluation on Chandrasekaran (EV1-derived) dataset:
   - Input: chan_cleaned_data.csv (Drug A/B, Inchikeys, alpha, Interaction Type, etc.)
   - Build full CC-only elementwise features via FeatureMapper.elementwise_similarity.
   - Subset those features to the exact same feat_cols used in EXP06d.
   - Run final model, produce predictions, metrics, ROC/PR curve data
     and confusion matrix for plotting.

Expected external base file (chan_cleaned_data.csv) columns:
    'Drug A', 'Drug B', 'Drug A Inchikey', 'Drug B Inchikey',
    'Drug Pair' (optional, will be created if missing),
    'Experimental Interaction Score', 'Interaction Type'
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report,
)

# ==========================
# Paths and Basic config
# ==========================

from halo.paths import MODEL_RESULTS, INTERIM, CC_FEATURES
from halo.mappers.feature_mapper import FeatureMapper


SCHEME = "CV1"

# Training features (same as original EXP06d)
filtered_path = MODEL_RESULTS / "exp05d_lgbm_bin_nosspace_elementwise_featselect_bliss005" / "elementwise_features_filtered_cv1_cc_only.csv"

# Original EXP06d result dir (where best_params_cv1.json lives)
exp06d_out = MODEL_RESULTS / "exp06d_lgbm_bin_nosspace_elementwise_reduced_nestedcv_bliss005"

ext_out = MODEL_RESULTS / "e_validation" / "external_eval_chandrasekaran"
ext_out.mkdir(parents=True, exist_ok=True)

best_params_path = exp06d_out / "best_params_cv1.json"

external_base_path = INTERIM / "source_d_chandrasekaran" / "chandrasekaran_cleaned_data.csv"

# Path to CC single-compound features (same as EXP05d)
cc_path = CC_FEATURES / "cc_features_concat_25x128.csv"


# ==========================
# 1) Load training data & features (EXP06d training matrix)
# ==========================

if not filtered_path.exists():
    raise FileNotFoundError(f"Training CSV not found at: {filtered_path}")

df = pd.read_csv(filtered_path).copy()
print("Loaded reduced training df shape:", df.shape)
print(df["Interaction Type"].value_counts())

# Keep only synergy vs antagonism
df = df[df["Interaction Type"].isin(["synergy", "antagonism"])].copy()
print("\nAfter filtering to synergy/antagonism:", df.shape)
print(df["Interaction Type"].value_counts())

drop_cols = [
    "Drug A",
    "Drug B",
    "Drug A Inchikey",
    "Drug B Inchikey",
    "Strain",
    "Specie",
    "Bliss Score",
    "Score",
    "Method",
    "Interaction Type",
    "Source",
    "Drug Pair"
]
feat_cols = [c for c in df.columns if c not in drop_cols]

X = df[feat_cols].copy()
y_text = df["Interaction Type"].copy()

le = LabelEncoder()
y_enc = le.fit_transform(y_text)

pairs = df["Drug Pair"].astype(str).values
n = len(df)

print(f"\nTotal samples: {n}")
print(f"Feature columns (CC-only reduced): {len(feat_cols)}")

# mapping int -> label
inv_label_map = {
    int(code): cls for cls, code in zip(le.classes_, le.transform(le.classes_))
}
synergy_code = le.transform(["synergy"])[0]
ant_code = le.transform(["antagonism"])[0]


# ==========================
# 2) Load best hyperparameters
# ==========================

if not best_params_path.exists():
    raise FileNotFoundError(f"best_params JSON not found at: {best_params_path}")

with open(best_params_path) as f:
    best_params_data = json.load(f)

best_params = best_params_data["best_params"]
print("\nLoaded best_params from:", best_params_path)
print(best_params)


# ==========================
# 3) Outer splits (CV1) – 5-fold Drug Pair grouping
# ==========================

def make_splits_cv1(n_splits=5, verbose=True):
    """5-fold outer CV over Drug Pair groups (CV1)."""
    try:
        outer_cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=42
        )
        split_gen = outer_cv.split(X, y_enc, groups=pairs)
    except TypeError:
        outer_cv = GroupKFold(n_splits=n_splits)
        split_gen = outer_cv.split(X, y_enc, groups=pairs)

    splits = []
    for fold_idx, (tr_idx, te_idx) in enumerate(split_gen, 1):
        splits.append((tr_idx, te_idx))
        if verbose:
            print("=" * 72)
            print(f"CV1 outer fold {fold_idx}/{n_splits} (Drug Pair grouping):")
            print(f"Train size: {len(tr_idx)} ({len(tr_idx) / n * 100:.2f}%)")
            print(f"Test size : {len(te_idx)} ({len(te_idx) / n * 100:.2f}%)")
            print(f"Test + Train set: {len(tr_idx) + len(te_idx)}")
            print("-" * 72)
    return splits


outer_splits = make_splits_cv1(n_splits=5, verbose=True)


# ==========================
# 4) Run outer CV with FIXED best_params (no nested search)
# ==========================

lgb.register_logger(
    type(
        "SilentLogger",
        (),
        {
            "info": lambda *a, **k: None,
            "warning": lambda *a, **k: None,
        },
    )()
)

fold_results = []
cm_total = None
all_test_dfs = []
all_train_dfs = []

for fold_idx, (tr_idx, te_idx) in enumerate(outer_splits, 1):
    print("\n" + "#" * 72)
    print(f"########## OUTER FOLD {fold_idx}/{len(outer_splits)} ##########")
    print("#" * 72 + "\n")

    X_tr = X.iloc[tr_idx].reset_index(drop=True)
    X_te = X.iloc[te_idx].reset_index(drop=True)
    y_tr = y_enc[tr_idx]
    y_te = y_enc[te_idx]

    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)

    # ---- Train LightGBM with fixed best_params ----
    m_final = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=4000,
        random_state=777,
        n_jobs=4,
        **best_params,
    )
    m_final.fit(X_tr, y_tr)

    pos_idx = np.flatnonzero(m_final.classes_ == synergy_code)[0]

    # ---- Evaluate on held-out test for this fold ----
    p_te = m_final.predict_proba(X_te)[:, pos_idx]
    y_pred = (p_te >= 0.5).astype(int)

    y_te_bin = (y_te == synergy_code).astype(int)

    accuracy_test = accuracy_score(y_te, y_pred)
    f1_macro_test = f1_score(y_te, y_pred, average="macro")
    f1_weighted_test = f1_score(y_te, y_pred, average="weighted")
    roc_auc_test = roc_auc_score(y_te_bin, p_te)

    print(f"\n=== Held-out Test (fold {fold_idx}) ===")
    print(f"ROC AUC : {roc_auc_test:.3f}")
    print(f"Acc     : {accuracy_test:.3f}")
    print(f"F1 (w)  : {f1_weighted_test:.3f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_te, y_pred))
    print(
        "\nReport:\n",
        classification_report(y_te, y_pred, target_names=le.classes_),
    )

    # Train-set overfitting check
    p_tr = m_final.predict_proba(X_tr)[:, pos_idx]
    y_tr_pred = (p_tr >= 0.5).astype(int)
    y_tr_bin = (y_tr == synergy_code).astype(int)

    accuracy_train = accuracy_score(y_tr, y_tr_pred)
    f1_weighted_train = f1_score(y_tr, y_tr_pred, average="weighted")
    roc_auc_train = roc_auc_score(y_tr_bin, p_tr)

    print("\n=== Overfitting check (fold", fold_idx, ") ===")
    print("Train AUC:", round(roc_auc_train, 3), "| Test AUC:", round(roc_auc_test, 3))
    print("Train Acc:", round(accuracy_train, 3), "| Test Acc:", round(accuracy_test, 3))
    print(
        "Train F1w:", round(f1_weighted_train, 3),
        "| Test F1w:", round(f1_weighted_test, 3),
    )

    # store per-fold metrics
    fold_results.append(
        dict(
            fold=fold_idx,
            roc_auc_test=roc_auc_test,
            accuracy_test=accuracy_test,
            f1_weighted_test=f1_weighted_test,
            roc_auc_train=roc_auc_train,
            accuracy_train=accuracy_train,
            f1_weighted_train=f1_weighted_train,
            n_train=len(tr_idx),
            n_test=len(te_idx),
        )
    )

    # store predictions for later analysis (internal)
    test_out_fold = pd.DataFrame(
        {
            "fold": fold_idx,
            "index": df_te.index,
            "Drug_Pair": df_te["Drug Pair"].astype(str),
            "Strain": df_te["Strain"].astype(str),
            "y_true_int": y_te,
            "y_true_label": [inv_label_map[int(v)] for v in y_te],
            "y_pred_int": y_pred,
            "y_pred_label": [inv_label_map[int(v)] for v in y_pred],
            "p_synergy": p_te,
        }
    )
    train_out_fold = pd.DataFrame(
        {
            "fold": fold_idx,
            "index": df_tr.index,
            "Drug_Pair": df_tr["Drug Pair"].astype(str),
            "Strain": df_tr["Strain"].astype(str),
            "y_true_int": y_tr,
            "y_true_label": [inv_label_map[int(v)] for v in y_tr],
            "y_pred_int": y_tr_pred,
            "y_pred_label": [inv_label_map[int(v)] for v in y_tr_pred],
            "p_synergy": p_tr,
        }
    )
    all_test_dfs.append(test_out_fold)
    all_train_dfs.append(train_out_fold)

    # accumulate confusion matrix
    order = ["antagonism", "synergy"]
    order_idx = le.transform(order)
    cm = confusion_matrix(y_te, y_pred, labels=order_idx)
    cm_total = cm if cm_total is None else cm_total + cm

# Save internal CV predictions/metrics (optional but nice to have)
test_out_all = pd.concat(all_test_dfs, ignore_index=True)
train_out_all = pd.concat(all_train_dfs, ignore_index=True)
test_out_all.to_csv(ext_out / "internal_test_predictions_cv1.csv", index=False)
train_out_all.to_csv(ext_out / "internal_train_predictions_cv1.csv", index=False)

metrics_per_fold_df = pd.DataFrame(fold_results)
metrics_per_fold_df.to_csv(ext_out / "internal_metrics_per_fold_cv1.csv", index=False)
print("\nSaved internal CV metrics & predictions to:", ext_out)


# ==========================
# 5) Train FINAL model on ALL training data
# ==========================

final_model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=4000,
    random_state=777,
    n_jobs=4,
    **best_params,
)
final_model.fit(X, y_enc)
pos_idx_final = np.flatnonzero(final_model.classes_ == synergy_code)[0]
print("\nTrained FINAL model on all training data.")


# ==========================
# 6) Build elementwise CC features for external set (EXP05d-style)
# ==========================

if not external_base_path.exists():
    raise FileNotFoundError(f"External base dataset not found at: {external_base_path}")
if not cc_path.exists():
    raise FileNotFoundError(f"CC features file not found at: {cc_path}")

ext_base = pd.read_csv(external_base_path).copy()
print("\nLoaded external base dataset:", external_base_path)
print("Shape:", ext_base.shape)

# Ensure required columns exist
required_cols = ["Drug A", "Drug B", "Drug A Inchikey", "Drug B Inchikey"]
missing_req = [c for c in required_cols if c not in ext_base.columns]
if missing_req:
    raise ValueError(f"External base dataset is missing required columns: {missing_req}")

# Normalise inchikeys (defensive)
ext_base["Drug A Inchikey"] = ext_base["Drug A Inchikey"].astype(str).str.upper().str.strip()
ext_base["Drug B Inchikey"] = ext_base["Drug B Inchikey"].astype(str).str.upper().str.strip()

# Ensure Drug Pair exists
if "Drug Pair" not in ext_base.columns:
    ext_base["Drug Pair"] = ext_base.apply(
        lambda x: "::".join(sorted([x["Drug A Inchikey"], x["Drug B Inchikey"]])),
        axis=1,
    )

# ---- Remove external pairs that appear in training (leakage guard) ----
# train_pairs = set(df["Drug Pair"].astype(str).values)

# before = len(ext_base)
# overlap_mask = ext_base["Drug Pair"].astype(str).isin(train_pairs)
# n_overlap = int(overlap_mask.sum())

# ext_base = ext_base.loc[~overlap_mask].copy()
# after = len(ext_base)

# print(f"[Leakage guard] External rows before: {before}")
# print(f"[Leakage guard] Overlapping pairs removed: {n_overlap}")
# print(f"[Leakage guard] External rows after: {after}")
# # ---- External drug coverage after removing overlapping PAIRS ----
# # (Uses Drug A/Drug B columns, plus Inchikey-based count for sanity)

# # unique drugs by name
# drug_names = pd.concat(
#     [ext_base["Drug A"].astype(str).str.strip(),
#      ext_base["Drug B"].astype(str).str.strip()],
#     ignore_index=True,
# ).replace("", np.nan).dropna().unique()

# # unique drugs by inchikey (more robust if names vary)
# drug_inchikeys = pd.concat(
#     [ext_base["Drug A Inchikey"].astype(str).str.strip(),
#      ext_base["Drug B Inchikey"].astype(str).str.strip()],
#     ignore_index=True,
# ).replace("", np.nan).dropna().unique()

# print(f"[Leakage guard] Unique drugs remaining (by name)    : {len(drug_names)}")
# print(f"[Leakage guard] Unique drugs remaining (by Inchikey): {len(drug_inchikeys)}")


# Load CC single-compound features
cc_df = pd.read_csv(cc_path).copy()

# Build full elementwise CC-only feature matrix for external set
fm = FeatureMapper()
ext_elem = fm.elementwise_similarity(ext_base, cc_df)

print("Elementwise external matrix shape (before label filtering):", ext_elem.shape)

# Expect Interaction Type already as 'synergy'/'antagonism' in ext_elem
if "Interaction Type" not in ext_elem.columns:
    raise ValueError("External elementwise matrix lacks 'Interaction Type' column.")

ext_elem = ext_elem[ext_elem["Interaction Type"].isin(["synergy", "antagonism"])].copy()
print("External elementwise after filtering to synergy/antagonism:", ext_elem.shape)

# ==========================
# 7) Align features & predict on external set
# ==========================

# Check that external elementwise has all the feat_cols used in training
missing_in_ext = set(feat_cols) - set(ext_elem.columns)
if missing_in_ext:
    raise ValueError(
        "External elementwise dataset is missing feature columns used in training. "
        f"Example missing cols: {sorted(list(missing_in_ext))[:10]}"
    )

X_ext = ext_elem[feat_cols].copy()
y_ext_text = ext_elem["Interaction Type"].copy()
y_ext = le.transform(y_ext_text)
y_ext_bin = (y_ext == synergy_code).astype(int)

print("\nExternal set size (after alignment):", len(ext_elem))

# Predict
p_synergy_ext = final_model.predict_proba(X_ext)[:, pos_idx_final]
y_pred_ext = (p_synergy_ext >= 0.5).astype(int)
y_pred_ext_label = [inv_label_map[int(v)] for v in y_pred_ext]

ext_elem["y_true_int"] = y_ext
ext_elem["y_true_label"] = y_ext_text.values
ext_elem["p_synergy"] = p_synergy_ext
ext_elem["y_pred_int"] = y_pred_ext
ext_elem["y_pred_label"] = y_pred_ext_label

# Confusion matrix & scalar metrics
cm_ext = confusion_matrix(y_ext, y_pred_ext, labels=[ant_code, synergy_code])
tn, fp, fn, tp = cm_ext.ravel()

acc_ext = accuracy_score(y_ext, y_pred_ext)
f1_ext = f1_score(y_ext, y_pred_ext)
try:
    auc_ext = roc_auc_score(y_ext_bin, p_synergy_ext)
except ValueError:
    auc_ext = float("nan")

print("\n=== External evaluation (Chandrasekaran) ===")
print("n =", len(ext_elem))
print("Accuracy:", acc_ext)
print("F1      :", f1_ext)
print("ROC AUC :", auc_ext)
print("Confusion matrix [[TN, FP], [FN, TP]]:\n", cm_ext)
print(
    "\nReport:\n",
    classification_report(y_ext, y_pred_ext, target_names=le.classes_),
)

# ==========================
# 8) Save everything needed for plotting
# ==========================

# Per-pair predictions
ext_pred_path = ext_out / "external_predictions_chandrasekaran.csv"
ext_elem.to_csv(ext_pred_path, index=False)
print("\nSaved external per-pair predictions to:", ext_pred_path)

# Scalar metrics
metrics_ext = pd.DataFrame(
    [{
        "dataset": "chandrasekaran_external",
        "n": len(ext_elem),
        "accuracy": acc_ext,
        "f1": f1_ext,
        "roc_auc": auc_ext,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }]
)
metrics_ext_path = ext_out / "external_metrics_chandrasekaran.csv"
metrics_ext.to_csv(metrics_ext_path, index=False)
print("Saved external summary metrics to:", metrics_ext_path)

# ROC + PR curves (for plotting script)
fpr, tpr, roc_thr = roc_curve(y_ext_bin, p_synergy_ext)
prec, rec, pr_thr = precision_recall_curve(y_ext_bin, p_synergy_ext)

roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thr})
pr_df = pd.DataFrame({"recall": rec, "precision": prec})
pr_thr_df = pd.DataFrame({"threshold": pr_thr})  # len = len(prec)-1

roc_df.to_csv(ext_out / "external_roc_curve_chandrasekaran.csv", index=False)
pr_df.to_csv(ext_out / "external_pr_curve_chandrasekaran.csv", index=False)
pr_thr_df.to_csv(ext_out / "external_pr_thresholds_chandrasekaran.csv", index=False)

print("Saved ROC and PR curve data to:", ext_out)

# Confusion matrix table
cm_df = pd.DataFrame(
    cm_ext,
    index=["true_antagonism", "true_synergy"],
    columns=["pred_antagonism", "pred_synergy"],
)
cm_df.to_csv(ext_out / "external_confusion_matrix_chandrasekaran.csv")

print("\n=== External evaluation script DONE ===")
