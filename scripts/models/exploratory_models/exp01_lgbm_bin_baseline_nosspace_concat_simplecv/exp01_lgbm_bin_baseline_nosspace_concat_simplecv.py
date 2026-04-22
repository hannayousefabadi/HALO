#!/usr/bin/env python3
"""
Experiment: exp01_lgbm_bin_baseline_nosspace_concat_simplecv

Config
- model: LightGBM (LGBMClassifier)
- task: binary classification (synergy vs antagonism)
- feature_design: concatenation (drugA + drugB CC features)
- sspace: disabled
- CV:
  - nested_cv: disabled
  - evaluation split: single stratified train/test split
  - hyperparameter search: RandomizedSearchCV with StratifiedKFold (5-fold) on train only
- label rule: bliss neutrality cutoff ±0.1 (applied during preprocessing; this script assumes labels are already finalized)

Data integrity note
All preprocessing (missing values, dtypes, column validation, label construction, and filtering)
is performed upstream in preprocessing notebooks/scripts. This script assumes the processed
inputs are clean and consistent.

Class encoding
The binary target is encoded as {0,1} where 1 corresponds to synergy and is treated as the
positive class for F1 and ROC-AUC.
"""
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import lightgbm as lgb

from halo.paths import CC_FEATURES, PROCESSED
from halo.mappers.feature_mapper import FeatureMapper
from halo.shared_utils.data_io import features_and_target, basic_split
from halo.shared_utils.metrics import classification_metrics, overfitting_report


def main():

    print("\n=== EXP01 ===\n")

    # ==========================
    # 1) load data 
    # ==========================
    cc_path = CC_FEATURES / "cc_features_concat_25x128.csv"
    combos_path = PROCESSED / "halo_training_dataset.csv"

    cc_df = pd.read_csv(cc_path).copy()
    combinations_df = pd.read_csv(combos_path).copy()

    print(f"Loaded cc_df: {cc_df.shape}")
    print(f"Loaded combos: {combinations_df.shape}")

    # ==========================
    # 2) Feature mapping 
    # ==========================
    mapper = FeatureMapper()
    df = mapper.concatenation(combinations_df, cc_df)

    print("Feature frame:", df.shape)
    print(df.head())

    # ==========================
    # 3) Encoding 
    # ==========================
    X, y_encoded, class_names = features_and_target(
        df,
        task='bin_clas',
        strain_as_feature=False,
        top_n_strains=None
    )

    X_train, X_test, y_train, y_test = basic_split(
        X, y_encoded, stratify=True
    ) 

    # pos = Counter(y_train)[1]
    # neg = Counter(y_train)[0]

    print("Class names:", class_names)
    # print(f"Train positives={pos}, negatives={neg}")

    """
    **Class encoding note:**  
    The `LabelEncoder` assigns labels alphabetically:
    - `antagonism → 0`  
    - `synergy → 1`   
    Therefore, **`synergy` is the positive class (label 1)** used in ROC-AUC and F1-score metrics.
    """

    # ==========================    
    # 4) Base model 
    # ==========================
    base_clf = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        n_jobs=-1,
        random_state=42
    )

    # ==========================
    # 5) Hyperparam search
    # ==========================
    param_dist = {
        'learning_rate': [0.02, 0.05],
        'n_estimators': [300, 600],
        'max_depth': [6, 8, 10],
        'num_leaves': [15, 31, 63],
        'min_child_samples': [20, 40, 80],
        'min_child_weight': [1e-2, 1e-1],
        'min_split_gain': [0.0, 0.1, 0.5],
        'feature_fraction': [0.6, 0.8],
        'subsample': [0.6, 0.8],
        'subsample_freq': [1],
        'lambda_l1': [0.0, 0.1, 1.0, 5.0],
        'lambda_l2': [0.0, 0.1, 1.0, 5.0],
        'max_bin': [63, 127, 255],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_dist,
        n_iter=30,
        cv=cv,
        scoring='f1',
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    print("\n--- Running RandomizedSearchCV ---\n")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print("\nBest params:", search.best_params_)

    # ==========================
    # 6) Evaluate 
    # ==========================
    y_pred = best_model.predict(X_test)
    y_score = best_model.predict_proba(X_test)[:, 1]

    print("\n--- Classification Metrics ---")
    classification_metrics(y_test, y_pred, y_score, class_names)

    print("\n--- Overfitting Report ---")
    overfitting_report(
        best_model,
        X_train, y_train,
        X_test, y_test,
        task='classification',
        average='macro'
    )

    print("\n=== EXP01 DONE ===\n")


if __name__ == "__main__":
    main()
