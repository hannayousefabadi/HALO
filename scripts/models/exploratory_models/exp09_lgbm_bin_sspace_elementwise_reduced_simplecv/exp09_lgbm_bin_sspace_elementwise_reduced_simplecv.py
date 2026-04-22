#!/usr/bin/env python3
"""
Experiment: exp09_lgbm_bin_sspace_elementwise_reduced_simplecv (HALO-Base / standard CV)

Config
- task: binary classification (synergy vs antagonism)
- feature_design: reduced elementwise similarity features (selected in exp05; includes CC + S-space dimensions)
- use_sspace: true (S-space is already included in the reduced feature table; no feature recomputation here)
- cv_scheme: standard stratified split + standard stratified CV (no grouping by drug pair)
- nested_cv: false (intentionally non-nested; model selection and evaluation are not strictly separated)
- purpose: provide an optimistic baseline / “upper-bound” estimate under lenient validation assumptions

Training / selection
- outer split: single stratified train/test split (80/20)
- hyperparameter tuning: RandomizedSearchCV with 5-fold StratifiedKFold on the training split
- model: LightGBM classifier (objective="binary"); metrics reported via shared_utils (F1-focused search)

Outputs
- console: best hyperparameters, test-set classification metrics, and an overfitting report
- no fold-aggregated artifacts are produced (this script is a baseline comparator)

**Data integrity note:**
All preprocessing (NA handling, dtype enforcement, column validation, etc.)
was completed in the preprocessing scripts.
This notebook assumes clean, validated input data.
"""


import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    RandomizedSearchCV
)
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# ==========================
# Paths
# ==========================
from halo.paths import MODEL_RESULTS
from halo.shared_utils.metrics import classification_metrics, overfitting_report


def main():
    print("\n=== EXP09: Simple CV + Reduced Elementwise Features (Binary) ===\n")

    # ==========================
    # 1) Load reduced feature file from exp05
    # ==========================
    reduced_path = MODEL_RESULTS / "exp05_lgbm_bin_sspace_elementwise_featselect" / "elementwise_features_filtered_cv1.csv"

    if not reduced_path.exists():
        raise FileNotFoundError(f"Reduced feature file not found: {reduced_path}")

    df = pd.read_csv(reduced_path).copy()
    print("Loaded df:", df.shape)
    print(df["Interaction Type"].value_counts())

    # ==========================
    # 2) Keep binary classes only
    # ==========================
    df = df[df["Interaction Type"].isin(["synergy", "antagonism"])].copy()
    print("\nFiltered (binary classes):", df.shape)

    # ==========================
    # 3) Feature columns
    # ==========================
    drop_cols = [
        "Drug A", "Drug B",
        "Drug A Inchikey", "Drug B Inchikey",
        "Strain", "Specie",
        "Bliss Score",
        "Interaction Type",
        "Source", "Drug Pair",
    ]

    feat_cols = [c for c in df.columns if c not in drop_cols]
    print("Feature columns:", len(feat_cols))

    X = df[feat_cols].copy()
    y = df["Interaction Type"].copy()

    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("Classes:", list(le.classes_))

    # ==========================
    # 4) Simple CV train/test split (LEAKY ON PURPOSE)
    # ==========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )

    # ==========================
    # 5) LightGBM baseline + randomized search
    # ==========================
    base_clf = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_jobs=-1,
        random_state=42,
    )

    param_dist = {
        "learning_rate": [0.02, 0.05],
        "n_estimators": [300, 600],
        "max_depth": [3, 4, 5, 6],
        "num_leaves": [7, 15, 31],
        "min_child_samples": [50, 100, 200],
        "feature_fraction": [0.3, 0.4, 0.6, 0.8],
        "subsample": [0.6, 0.8],
        "subsample_freq": [1],
        "lambda_l1": [0.0, 0.1, 1.0, 5.0],
        "lambda_l2": [0.0, 0.1, 1.0, 5.0],
        "max_bin": [63, 127, 255],
        "min_split_gain": [0.0, 0.05, 0.1],
    }

    print("\n--- Running RandomizedSearchCV (simple CV) ---\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_dist,
        n_iter=40,
        cv=cv,
        scoring="f1",
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print("\nBest parameters:", search.best_params_)

    # ==========================
    # 6) Final evaluation
    # ==========================
    y_pred = best_model.predict(X_test)
    y_score = best_model.predict_proba(X_test)  # full (n_samples, 2) matrix

    print("\n=== TEST METRICS (LEAKY SIMPLE CV) ===")
    classification_metrics(
        y_test,
        y_pred,
        y_score=y_score,
        class_names=le.classes_,
    )

    # ==========================
    # 7) Overfitting report
    # ==========================
    print("\n=== Overfitting Report ===")
    overfitting_report(
        best_model,
        X_train, y_train,
        X_test, y_test,
        task="classification",
        average="macro"
    )

    print("\n=== EXP09 DONE ===\n")


if __name__ == "__main__":
    main()
