#!/usr/bin/env python3
"""
HALO sensitivity analysis

- Bliss neutrality cutoff this script tests: [0.2, 0.1, 0.05, 0.02, 0.0]

Data integrity note
All preprocessing (missing values, dtypes, column validation, and label construction from Bliss using the
±0.1 cutoff) is performed upstream in preprocessing notebooks/scripts. This script assumes the processed
inputs are clean and consistent and that `Interaction Type` already reflects that cutoff.
"""

import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from halo.paths import CC_FEATURES, PROCESSED, RESULTS
from halo.mappers.feature_mapper import FeatureMapper
from halo.shared_utils.data_io import classify_interaction


# ==========================
# Helper functions
# ==========================
def select_features_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feat_cols: list[str],
    corr_min: float = 0.01,
    keep_top_frac: float = 0.30,
) -> list[str]:
    """
    Feature selection performed using training data only.

    Steps
    1) drop zero-variance features
    2) correlation prefilter using |corr(feature, y)| >= corr_min
       fallback: keep all variance-filtered features if none pass
    3) LightGBM importance ranking and keep top fraction
    """
    var_series = X_train.var()
    kept_after_var = [c for c in feat_cols if var_series[c] > 0.0]

    if len(kept_after_var) == 0:
        raise ValueError("No features remained after variance filtering.")

    kept_after_corr = []
    y_train_s = pd.Series(y_train, index=X_train.index)

    for col in kept_after_var:
        corr = X_train[col].corr(y_train_s)
        if corr is not None and np.isfinite(corr) and abs(corr) >= corr_min:
            kept_after_corr.append(col)

    if not kept_after_corr:
        kept_after_corr = kept_after_var.copy()

    fs_model = lgb.LGBMClassifier(
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

    fs_model.fit(X_train[kept_after_corr], y_train)

    feat_imp = pd.Series(
        fs_model.feature_importances_,
        index=kept_after_corr
    ).sort_values(ascending=False)

    n_keep = max(1, int(len(feat_imp) * keep_top_frac))
    selected_features = feat_imp.index[:n_keep].tolist()
    return selected_features


def cutoff_to_str(c):
    """
    Convert cutoff float → safe filename string.
    """
    return f"{c:.2f}".replace(".", "p")

# Bliss cutoffs to iterate through
cutoff_values = [0.2, 0.1, 0.05, 0.02, 0.0]


def main(c_value: float):
    # ==========================
    # 0) Basic config
    # ==========================
    SCHEME = "CV1"
    corr_min = 0.01
    keep_top_frac = 0.30

    cc_path = CC_FEATURES / "cc_features_concat_25x128.csv"
    combos_path = PROCESSED / "halo_training_dataset.csv"

    out_dir = RESULTS / "sensitivity_analysis" 
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "\n=== EXP06d ===\n"
    )
    # print("Using scheme:", SCHEME)
    print("Output dir :", out_dir)

    # ==========================
    # 1) Load full dataset and build elementwise features
    # ==========================
    cc_df = pd.read_csv(cc_path).copy()
    combinations_df = pd.read_csv(combos_path).copy()

    features_cc = cc_df.copy()
    df = FeatureMapper().elementwise_similarity(combinations_df, features_cc)
    print("Full df shape:", df.shape)
    
    # enforce a new Bliss cutoff in each iteration
    current_cutoff = c_value
    c_str = cutoff_to_str(c_value)

    df["Interaction Type"] = df["Bliss Score"].apply(
        lambda x: classify_interaction(x, additivity_cutoff=current_cutoff)
    )
    df = df[df["Interaction Type"].isin(["synergy", "antagonism"])].copy()
    print("\nAfter filtering to synergy/antagonism:", df.shape)
    print(df["Interaction Type"].value_counts())
    # ==========================
    # 2) Feature columns
    # ==========================
    drop_cols = [
        "Drug A",
        "Drug B",
        "Drug A Inchikey",
        "Drug B Inchikey",
        "Strain",
        "Specie",
        "Bliss Score",
        "Interaction Type",
        "Source",
        "Drug Pair",
    ]
    feat_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feat_cols].copy()

    y = df["Interaction Type"].copy()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    pairs = df["Drug Pair"].astype(str).values
    strains = df["Strain"].astype(str).values
    n = len(df)

    rng = np.random.default_rng(42)

    # print(f"\nTotal samples: {n}")
    # print(f"Full feature columns (CC-only): {len(feat_cols)}")

    inv_label_map = {
        int(code): cls for cls, code in zip(le.classes_, le.transform(le.classes_))
    }

    # ==========================
    # 3) Outer splits (CV1)
    # ==========================
    def make_splits_cv1(n_splits=5, verbose=True):
        """
        Split data into outer test and outer train under Drug Pair held-out scheme (CV1)
        over 5 folds.
        
        Returns:
            tr_idx (np.ndarray of positions),
            te_idx (np.ndarray of positions)
        """
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
            # if verbose:
            #     print("=" * 72)
            #     print(f"CV1 outer fold {fold_idx}/{n_splits} (Drug Pair grouping):")
            #     print(f"Train size: {len(tr_idx)} ({len(tr_idx) / n * 100:.2f}%)")
            #     print(f"Test size : {len(te_idx)} ({len(te_idx) / n * 100:.2f}%)")
            #     print(f"Test + Train set: {len(tr_idx) + len(te_idx)}")
            #     print("-" * 72)
        return splits

    def make_split_cv2(
        strain_col: str = "Strain",
        pair_col: str = "Drug Pair",
        min_frac: float = 0.16,
        max_frac: float = 0.20,
        lambda_penalty: float = 1.0,
        top_k_print: int = 5,
        verbose: bool = True,
    ):
        """
        Kept for completeness; not used when SCHEME == 'CV1'
        """
        S_all = df[strain_col].astype(str).values
        P_all = df[pair_col].astype(str).values
        strains_uni = sorted(np.unique(S_all).tolist())
        n_total = len(df)
        min_target = int(round(min_frac * n_total))
        max_target = int(round(max_frac * n_total))

        if verbose:
            print("=" * 72)
            print("CV2 Strain + Drug Pair grouping:")
            print(f"Total rows: {n_total} | Strain levels: {len(strains_uni)}")
            print(
                f"Target kept test rows ∈ "
                f"[{min_target} ({min_frac:.0%}), {max_target} ({max_frac:.0%})]"
            )

        def eval_subset(S_test_set):
            if not S_test_set:
                return 0, 0, n_total, set(), -np.inf

            mask_s = np.isin(S_all, list(S_test_set))
            P_test_set = set(P_all[mask_s])
            test_mask = mask_s & np.isin(P_all, list(P_test_set))
            train_mask = (~mask_s) & (~np.isin(P_all, list(P_test_set)))

            kept = int(test_mask.sum())
            train = int(train_mask.sum())
            dropped = n_total - (kept + train)
            score = kept - lambda_penalty * dropped
            return kept, train, dropped, P_test_set, score

        candidates = []
        for r in range(1, len(strains_uni) + 1):
            for subset in itertools.combinations(strains_uni, r):
                S_test = set(subset)
                kept, train, dropped, P_test, score = eval_subset(S_test)
                if min_target <= kept <= max_target:
                    candidates.append(
                        {
                            "S_test": S_test,
                            "P_test": P_test,
                            "kept": kept,
                            "train": train,
                            "dropped": dropped,
                            "score": score,
                        }
                    )

        if not candidates:
            if verbose:
                print("-" * 72)
                print("No subsets hit the target kept-test band.")
                print("=" * 72)
            return np.arange(n_total), np.array([], dtype=int), {
                "reason": "no_candidate",
                "test_strains": set(),
                "test_pairs": set(),
                "train_strains": set(strains_uni),
                "train_pairs": set(np.unique(P_all).tolist()),
                "kept_test_rows": 0,
                "kept_train_rows": n_total,
                "dropped_rows": 0,
                "params": dict(
                    min_frac=min_frac,
                    max_frac=max_frac,
                    lambda_penalty=lambda_penalty,
                ),
            }

        candidates.sort(key=lambda c: (c["dropped"], -c["kept"], -c["score"]))
        best = candidates[0]

        S_test_best = best["S_test"]
        P_test_best = best["P_test"]

        test_mask = np.isin(S_all, list(S_test_best)) & np.isin(P_all, list(P_test_best))
        train_mask = (~np.isin(S_all, list(S_test_best))) & (~np.isin(P_all, list(P_test_best)))

        te_idx = np.where(test_mask)[0]
        tr_idx = np.where(train_mask)[0]
        dropped_rows = n_total - (te_idx.size + tr_idx.size)

        S_train_best = set(strains_uni) - set(S_test_best)
        P_train_best = set(np.unique(P_all).tolist()) - set(P_test_best)

        info = dict(
            mode="bruteforce_strains",
            test_strains=set(S_test_best),
            train_strains=S_train_best,
            test_pairs=set(P_test_best),
            train_pairs=P_train_best,
            kept_test_rows=int(te_idx.size),
            kept_train_rows=int(tr_idx.size),
            dropped_rows=int(dropped_rows),
            params=dict(
                min_frac=min_frac,
                max_frac=max_frac,
                lambda_penalty=lambda_penalty,
            ),
            top_candidates=candidates[:top_k_print],
        )
        return tr_idx, te_idx, info

    if SCHEME == "CV1":
        outer_splits = make_splits_cv1(n_splits=5, verbose=True)
        info = None
    elif SCHEME == "CV2":
        tr_idx, te_idx, info = make_split_cv2()
        print("test strains:", info["test_strains"])
        outer_splits = [(tr_idx, te_idx)]
    else:
        raise ValueError("SCHEME must be 'CV1' or 'CV2'")

    # ==========================
    # 4) Inner CV (nested) + outer loop
    # ==========================
    class SilentLogger:
        def info(self, msg):
            pass

        def warning(self, msg):
            pass

    lgb.register_logger(SilentLogger())

    fold_results = []
    all_test_dfs = []
    per_fold_best = []

    synergy_code = le.transform(["synergy"])[0]
    ant_code = le.transform(["antagonism"])[0]

    def sample_one_params():
        max_depth = 3
        leaves_map = {3: [7, 9, 15]}

        return dict(
            boosting_type="gbdt",
            learning_rate=float(rng.choice([0.02, 0.03])),
            max_depth=max_depth,
            num_leaves=int(rng.choice(leaves_map[max_depth])),
            min_data_in_leaf=int(rng.choice([200, 300])),
            feature_fraction=float(rng.choice([0.30, 0.40, 0.50])),
            bagging_fraction=float(rng.choice([0.60, 0.70, 0.80])),
            bagging_freq=1,
            lambda_l2=float(10 ** rng.uniform(1.4, 1.9)),
            lambda_l1=float(rng.choice([0.0, 0.1, 0.5])),
            max_bin=int(rng.choice([63, 127])),
            min_gain_to_split=float(rng.choice([0.05, 0.10, 0.20])),
        )

    for fold_idx, (tr_idx, te_idx) in enumerate(outer_splits, 1):
        print("\n" + "#" * 72)
        print(f"########## OUTER FOLD {fold_idx}/{len(outer_splits)} ##########")
        print("#" * 72 + "\n")

        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        X_te = X.iloc[te_idx].reset_index(drop=True)
        y_tr = y_enc[tr_idx]
        y_te = y_enc[te_idx]
        grp_tr = pairs[tr_idx]

        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_te = df.iloc[te_idx].reset_index(drop=True)

        param_samples = [sample_one_params() for _ in range(32)]

        # The inner CV split, 3 folds, grouped by drug-pair
        try:
            inner_cv = StratifiedGroupKFold(
                n_splits=3, shuffle=True, random_state=111
            )

            def inner_splitter():
                return inner_cv.split(X_tr, y_tr, groups=grp_tr)

        except Exception:
            inner_cv = GroupKFold(n_splits=3)

            def inner_splitter():
                return inner_cv.split(X_tr, y_tr, groups=grp_tr)

        # inner-CV hyperparameters scores and feature selection on INNER-TRAIN-ONLY
        def cv_acc_for_params(params):
            """
            Perform a full inner cross‑validation loop for a given hyperparameter
            configuration.

            For each inner fold:
            - Fit the feature selector only on the inner‑train split.
            - Select features for that fold.
            - Train a LightGBM model with the provided hyperparameters.
            - Evaluate accuracy on the inner‑validation split.

            Returns the mean validation accuracy across inner folds, used for
            comparing and ranking hyperparameter configurations in the nested CV.

            It is trying to answer: “If we used this hyperparameter configuration, 
            how well would the pipeline generalize?”
            """
            accs = []
            for tr_f, val_f in inner_splitter():
                Xf_tr = X_tr.iloc[tr_f].reset_index(drop=True)
                Xf_val = X_tr.iloc[val_f].reset_index(drop=True)
                yf_tr = y_tr[tr_f]
                yf_val = y_tr[val_f]

                selected_inner = select_features_lgbm(
                    X_train=Xf_tr,
                    y_train=yf_tr,
                    feat_cols=feat_cols,
                    corr_min=corr_min,
                    keep_top_frac=keep_top_frac,
                )

                Xf_tr_sel = Xf_tr[selected_inner]
                Xf_val_sel = Xf_val[selected_inner]

                m = lgb.LGBMClassifier(
                    objective="binary",
                    n_estimators=4000,
                    random_state=777,
                    n_jobs=4,
                    **params,
                )
                m.fit(
                    Xf_tr_sel,
                    yf_tr,
                    eval_set=[(Xf_val_sel, yf_val)],
                    eval_metric="binary_logloss",
                    callbacks=[
                        lgb.early_stopping(200, False),
                        lgb.log_evaluation(0),
                    ],
                )

                y_pred_fold = m.predict(Xf_val_sel)
                accs.append(accuracy_score(yf_val, y_pred_fold))
            return float(np.mean(accs))

        print(
            "\n--- Nested CV (outer fold "
            f"{fold_idx}): inner search over {len(param_samples)} configs ---"
        )
        scores = [(cv_acc_for_params(ps), ps) for ps in param_samples]
        scores.sort(reverse=True, key=lambda t: t[0])
        best_acc, best_params = scores[0]
        print(
            f"Best inner-CV ACC (fold {fold_idx}): {best_acc:.3f}\n"
            f"Best params: {best_params}"
        )

        per_fold_best.append(
            {
                "fold": fold_idx,
                "best_acc_inner_cv": float(best_acc),
                "best_params": best_params,
            }
        )

        # ==========================
        # 5) Final refit on outer-train with outer-train-only feature selection
        # ==========================
        # feature selection on FULL outer-train only
        selected_outer = select_features_lgbm(
            X_train=X_tr,
            y_train=y_tr,
            feat_cols=feat_cols,
            corr_min=corr_min,
            keep_top_frac=keep_top_frac,
        )

        X_tr_sel = X_tr[selected_outer]
        X_te_sel = X_te[selected_outer]

        # final refit on full outer-train 
        m_final = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=4000,
            random_state=777,
            n_jobs=4,
            **best_params,
        )
        m_final.fit(X_tr_sel, y_tr)

        pos_idx = np.flatnonzero(m_final.classes_ == synergy_code)[0]

        # ==========================
        # 6) Final evaluation on held-out outer test
        # ==========================
        p_te = m_final.predict_proba(X_te_sel)[:, pos_idx]
        y_pred = (p_te >= 0.5).astype(int)

        y_te_bin = (y_te == synergy_code).astype(int)

        accuracy_test = accuracy_score(y_te, y_pred)
        f1_macro_test = f1_score(y_te, y_pred, average="macro")
        f1_weighted_test = f1_score(y_te, y_pred, average="weighted")
        roc_auc_test = roc_auc_score(y_te_bin, p_te)

        print(f"\n=== Held-out Test (fold {fold_idx}) ===")
        print(f"accuracy_test={accuracy_test:.4f}")
        print(f"roc_auc_test={roc_auc_test:.4f}")
        print(f"f1_macro_test={f1_macro_test:.4f}")
        print(f"f1_weighted_test={f1_weighted_test:.4f}")
    
        print("\nConfusion matrix:\n", confusion_matrix(y_te, y_pred))
        print(
            "\nReport:\n",
            classification_report(y_te, y_pred, target_names=le.classes_),
        )

        prec, rec, f1s, _ = precision_recall_fscore_support(
            y_te,
            y_pred,
            labels=[ant_code, synergy_code],
        )
        precision_antag, precision_syn = prec
        recall_antag, recall_syn = rec
        f1_antag, f1_syn = f1s


        # ==========================
        # 7) Save test scores for this fold
        # ==========================
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
        all_test_dfs.append(test_out_fold)

        order = ["antagonism", "synergy"]

        fold_results.append(
            dict(
                fold=fold_idx,
                roc_auc_test=roc_auc_test,
                accuracy_test=accuracy_test,
                f1_weighted_test=f1_weighted_test,
                f1_macro_test=f1_macro_test,
                precision_antag=precision_antag,
                recall_antag=recall_antag,
                f1_antag=f1_antag,
                precision_syn=precision_syn,
                recall_syn=recall_syn,
                f1_syn=f1_syn,
                n_train=len(tr_idx),
                n_test=len(te_idx),
            )
        )

    # ==========================
    # 8) Aggregate scores across folds
    # ==========================
    test_out_all = pd.concat(all_test_dfs, ignore_index=True)
    test_out_path = out_dir / f"test_predictions_{SCHEME.lower()}_{c_str}.csv"

    test_out_all.to_csv(test_out_path, index=False)
    print("Saved aggregated test predictions to:", test_out_path)

    def mean_std(key):
        vals = [fr[key] for fr in fold_results]
        return float(np.mean(vals)), float(np.std(vals))

    acc_mean, acc_std = mean_std("accuracy_test")
    f1w_mean, f1w_std = mean_std("f1_weighted_test")
    f1m_mean, f1m_std = mean_std("f1_macro_test")
    auc_mean, auc_std = mean_std("roc_auc_test")

    print("\n" + "=" * 72)
    print(f"=== Summary over {len(fold_results)} CV1 outer folds ===")
    print(f"roc_auc_test_mean={auc_mean:.4f}  roc_auc_test_std={auc_std:.4f}")
    print(f"accuracy_test_mean={acc_mean:.4f}  accuracy_test_std={acc_std:.4f}")
    print(f"f1_weighted_test_mean={f1w_mean:.4f}  f1_weighted_test_std={f1w_std:.4f}")
    print(f"f1_macro_test_mean={f1m_mean:.4f}  f1_macro_test_std={f1m_std:.4f}")
    print("=" * 72)

    metrics_test = {
        "scheme": SCHEME,
        "bliss cutoff": c_value,
        "n_folds": len(fold_results),
        "accuracy_test": acc_mean,
        "accuracy_test_std": acc_std,
        "f1_macro_test": f1m_mean,
        "f1_macro_test_std": f1m_std,
        "f1_weighted_test": f1w_mean,
        "f1_weighted_test_std": f1w_std,
        "roc_auc_test": auc_mean,
        "roc_auc_test_std": auc_std,
        "n_test": int(sum(fr["n_test"] for fr in fold_results)),
    }
    metrics_test_df = pd.DataFrame([metrics_test])
    metrics_test_path = out_dir / f"metrics_test_{SCHEME.lower()}_{c_str}.csv"
    metrics_test_df.to_csv(metrics_test_path, index=False)
    print("Saved aggregated test metrics to:", metrics_test_path)

    print("\n=== EXP06d DONE ===\n")

# ==========================
# Run sensitivity
# ==========================
if __name__ == "__main__":
    for c_value in cutoff_values:
        print(f"\n ############### Running analysis with Bliss cutoff: {c_value} ###############")
        main(c_value)