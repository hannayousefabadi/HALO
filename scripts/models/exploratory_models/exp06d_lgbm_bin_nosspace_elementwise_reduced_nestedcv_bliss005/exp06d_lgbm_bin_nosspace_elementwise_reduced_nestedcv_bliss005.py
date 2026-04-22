#!/usr/bin/env python3
"""
Experiment: exp06d_lgbm_bin_nosspace_elementwise_reduced_nestedcv_bliss005 (HALO-CV1)

Config
- task: binary classification (synergy vs antagonism) with Bliss additivity cutoff ±0.05
- feature_design: reduced elementwise similarity features (selected in exp05d), CC-only
- use_sspace: false (no strain-space/S-space features)
- cv_scheme: CV1 (drug-pair held-out)
- outer_cv: 5-fold grouped CV by Drug Pair (StratifiedGroupKFold if available, else GroupKFold)
- inner_cv: 3-fold grouped CV by Drug Pair for hyperparameter search (nested inside each outer fold)
- model: LightGBM (binary objective), “synergy” treated as the positive class for ROC AUC
- outputs:
    - per-fold metrics and mean±std summaries (test + train)
    - per-fold and aggregated confusion matrices
    - per-fold and averaged feature importances (split + gain, with normalized gain)
    - per-fold train/test prediction tables
    - best hyperparameters (per-fold + best-overall) saved as JSON

**Data integrity note:**
All preprocessing (NA handling, dtype enforcement, column validation, etc.)
was completed in the preprocessing scripts.
This notebook assumes clean, validated input data.
"""

import json
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

from halo.paths import MODEL_RESULTS

def main():
    # ==========================
    # 0) Basic config
    # ==========================
    SCHEME = "CV1"  # keep same scheme as exp03/exp06b for fair comparison

    # path to the reduced CC-only elementwise dataset produced by exp05d
    filtered_path = MODEL_RESULTS / "exp05d_lgbm_bin_nosspace_elementwise_featselect_bliss005" / "elementwise_features_filtered_cv1_cc_only.csv"

    out_dir = MODEL_RESULTS / "exp06d_lgbm_bin_nosspace_elementwise_reduced_nestedcv_bliss005"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "\n=== EXP06d: LGBM bin + reduced CC-only elementwise from exp05d "
        "(CV1, bliss=±0.05) + nested CV, 5-fold outer ===\n"
    )
    print("Using scheme:", SCHEME)
    print("Input  file:", filtered_path)
    print("Output dir :", out_dir)

    # ==========================
    # 1) Load reduced dataset
    # ==========================
    if not filtered_path.exists():
        raise FileNotFoundError(f"Filtered CSV not found at: {filtered_path}")

    df = pd.read_csv(filtered_path).copy()
    print("Loaded reduced df shape:", df.shape)
    print(df["Interaction Type"].value_counts())

    # safety: keep only synergy vs antagonism (should already be true)
    df = df[df["Interaction Type"].isin(["synergy", "antagonism"])].copy()
    print("\nAfter filtering to synergy/antagonism:", df.shape)
    print(df["Interaction Type"].value_counts())

    # ==========================
    # 2) Feature columns (reduced elementwise only)
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
    strains = df["Strain"].astype(str).values  # kept only for info / CV2 if needed
    n = len(df)

    rng = np.random.default_rng(42)

    print(f"\nTotal samples: {n}")
    print(f"Reduced feature columns (CC-only): {len(feat_cols)}")

    # inverse mapping from encoded labels to text
    inv_label_map = {
        int(code): cls for cls, code in zip(le.classes_, le.transform(le.classes_))
    }

    # ==========================
    # 3) Outer splits (CV1 / CV2) – 5-fold outer CV for CV1
    # ==========================
    def make_splits_cv1(n_splits=5, verbose=True):
        """
        5-fold outer CV over Drug Pair groups (CV1).
        Uses StratifiedGroupKFold if available, falls back to GroupKFold.
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
            if verbose:
                print("=" * 72)
                print(f"CV1 outer fold {fold_idx}/{n_splits} (Drug Pair grouping):")
                print(f"Train size: {len(tr_idx)} ({len(tr_idx) / n * 100:.2f}%)")
                print(f"Test size : {len(te_idx)} ({len(te_idx) / n * 100:.2f}%)")
                print(f"Test + Train set: {len(tr_idx) + len(te_idx)}")
                print("-" * 72)
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
        Exhaustive strain subset search (CV2: held out by Strain + Drug Pair)
        (kept here for completeness; not used when SCHEME == 'CV1')
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
                print(
                    "No subsets hit the target kept-test band. "
                    "You can widen [min_frac, max_frac] or allow manual S_test."
                )
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

        if verbose:
            print("-" * 72)
            print(
                f"Valid candidates in band: {len(candidates)} | "
                f"Showing top {min(top_k_print, len(candidates))}"
            )
            print(
                "rank | #S | #P | kept(%) | dropped(%) | score | S_test (truncated)"
            )
            for i, c in enumerate(candidates[:top_k_print], 1):
                kept_pct = 100.0 * c["kept"] / n_total
                drop_pct = 100.0 * c["dropped"] / n_total
                s_preview = ", ".join(list(sorted(c["S_test"]))[:3])
                if len(c["S_test"]) > 3:
                    s_preview += ", …"
                print(
                    f"{i:>4} | {len(c['S_test']):>2} | {len(c['P_test']):>3} | "
                    f"{kept_pct:6.2f} | {drop_pct:9.2f} | {c['score']:>7.1f} | {s_preview}"
                )

        best = candidates[0]
        S_test_best = best["S_test"]
        P_test_best = best["P_test"]

        test_mask = np.isin(S_all, list(S_test_best)) & np.isin(
            P_all, list(P_test_best)
        )
        train_mask = (~np.isin(S_all, list(S_test_best))) & (
            ~np.isin(P_all, list(P_test_best))
        )

        te_idx = np.where(test_mask)[0]
        tr_idx = np.where(train_mask)[0]
        dropped_rows = n_total - (te_idx.size + tr_idx.size)

        S_train_best = set(strains_uni) - set(S_test_best)
        P_train_best = set(np.unique(P_all).tolist()) - set(P_test_best)

        if verbose:
            print("-" * 72)
            print("Chosen subset (BEST):")
            print(f"#Test Strains: {len(S_test_best)} | #Test Pairs: {len(P_test_best)}")
            print(
                f"Train rows: {tr_idx.size} ({tr_idx.size/n_total*100:.2f}%)"
            )
            print(f"Test  rows: {te_idx.size} ({te_idx.size/n_total*100:.2f}%)")
            print(
                f"Dropped rows: {dropped_rows} "
                f"({dropped_rows/n_total*100:.2f}%)"
            )
            print(f"Test + Train set: {len(tr_idx) + len(te_idx)}")
            print(
                f"Overlap Strains? {len(S_train_best & S_test_best)} (expect 0)"
            )
            print(f"Overlap Pairs?   {len(P_train_best & P_test_best)} (expect 0)")
            print("=" * 72)

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

    # Decide outer splits
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

    # containers
    fold_results = []
    cm_total = None
    all_test_dfs = []
    all_train_dfs = []
    per_fold_importances = []
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
            # moderate regularization (NOT heavy)
            min_data_in_leaf=int(rng.choice([200, 300])),
            feature_fraction=float(rng.choice([0.30, 0.40, 0.50])),
            bagging_fraction=float(rng.choice([0.60, 0.70, 0.80])),
            bagging_freq=1,
            lambda_l2=float(10 ** rng.uniform(1.4, 1.9)),   # ~25–80
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

        # ---- Inner CV (nested) ----
        param_samples = [sample_one_params() for _ in range(32)]

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

        def cv_acc_for_params(params):
            accs = []
            for tr_f, val_f in inner_splitter():
                Xf_tr, Xf_val = X_tr.iloc[tr_f], X_tr.iloc[val_f]
                yf_tr, yf_val = y_tr[tr_f], y_tr[val_f]

                m = lgb.LGBMClassifier(
                    objective="binary",
                    n_estimators=4000,
                    random_state=777,
                    n_jobs=4,
                    **params,
                )
                m.fit(
                    Xf_tr,
                    yf_tr,
                    eval_set=[(Xf_val, yf_val)],
                    eval_metric="binary_logloss",
                    callbacks=[
                        lgb.early_stopping(200, False),  # was 200
                        lgb.log_evaluation(0),
                    ],
                )

                y_pred_fold = m.predict(Xf_val)
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
        # 5) Final refit on outer-train (this fold)
        # ==========================
        m_final = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=4000,
            random_state=777,
            n_jobs=4,
            **best_params,
        )
        m_final.fit(X_tr, y_tr)

        pos_idx = np.flatnonzero(m_final.classes_ == synergy_code)[0]

        # ==========================
        # 5b) Save feature importances (per fold)
        # ==========================
        booster = m_final.booster_
        imp_split = booster.feature_importance(importance_type="split")
        imp_gain = booster.feature_importance(importance_type="gain")

        feat_importance_df = pd.DataFrame(
            {
                "feature": feat_cols,
                "importance_split": imp_split,
                "importance_gain": imp_gain,
            }
        )
        feat_importance_df["importance_gain_norm"] = (
            feat_importance_df["importance_gain"]
            / feat_importance_df["importance_gain"].sum()
        )

        fi_fold_path = out_dir / f"feature_importances_{SCHEME.lower()}_fold{fold_idx}.csv"
        feat_importance_df.to_csv(fi_fold_path, index=False)
        print("Saved per-fold feature importances to:", fi_fold_path)

        # keep for later averaging
        feat_importance_df["fold"] = fold_idx
        per_fold_importances.append(feat_importance_df)

        # ==========================
        # 6) Final evaluation on held-out test (this fold)
        # ==========================
        p_te = m_final.predict_proba(X_te)[:, pos_idx]
        y_pred = (p_te >= 0.5).astype(int)

        # make sure "synergy" is the positive class for AUC
        y_te_bin = (y_te == synergy_code).astype(int)

        # Global metrics (this fold)
        accuracy_test = accuracy_score(y_te, y_pred)
        f1_macro_test = f1_score(y_te, y_pred, average="macro")
        f1_weighted_test = f1_score(y_te, y_pred, average="weighted")
        roc_auc_test = roc_auc_score(y_te_bin, p_te)

        print(f"\n=== Held-out Test (CC-only, fold {fold_idx}) ===")
        print(f"ROC AUC : {roc_auc_test:.3f}")
        print(f"Acc     : {accuracy_test:.3f}")
        print(f"F1 (w)  : {f1_weighted_test:.3f}")
        print("\nConfusion matrix:\n", confusion_matrix(y_te, y_pred))
        print(
            "\nReport:\n",
            classification_report(y_te, y_pred, target_names=le.classes_),
        )

        # Per-class metrics
        syn_code = synergy_code
        prec, rec, f1s, _ = precision_recall_fscore_support(
            y_te,
            y_pred,
            labels=[ant_code, syn_code],
        )
        precision_antag, precision_syn = prec
        recall_antag, recall_syn = rec
        f1_antag, f1_syn = f1s

        print("\n--- Metrics for Log (outer fold", fold_idx, ") ---")
        print(f"accuracy_test={accuracy_test:.4f}")
        print(f"f1_macro_test={f1_macro_test:.4f}")
        print(f"f1_weighted_test={f1_weighted_test:.4f}")
        print(f"roc_auc_test={roc_auc_test:.4f}")
        print(f"precision_antag={precision_antag:.4f}")
        print(f"recall_antag={recall_antag:.4f}")
        print(f"f1_antag={f1_antag:.4f}")
        print(f"precision_syn={precision_syn:.4f}")
        print(f"recall_syn={recall_syn:.4f}")
        print(f"f1_syn={f1_syn:.4f}")

        # ==========================
        # 7) Overfitting check (this fold)
        # ==========================
        p_tr = m_final.predict_proba(X_tr)[:, pos_idx]
        y_tr_pred = (p_tr >= 0.5).astype(int)
        y_tr_bin = (y_tr == synergy_code).astype(int)

        accuracy_train = accuracy_score(y_tr, y_tr_pred)
        f1_macro_train = f1_score(y_tr, y_tr_pred, average="macro")
        f1_weighted_train = f1_score(y_tr, y_tr_pred, average="weighted")
        roc_auc_train = roc_auc_score(y_tr_bin, p_tr)

        print("\n=== Overfitting check (CC-only, outer fold", fold_idx, ") ===")
        print(
            "Train AUC:",
            round(roc_auc_train, 3),
            "| Test AUC:",
            round(roc_auc_test, 3),
        )
        print(
            "Train Acc:",
            round(accuracy_train, 3),
            "| Test Acc:",
            round(accuracy_test, 3),
        )
        print(
            "Train F1w:",
            round(f1_weighted_train, 3),
            "| Test F1w:",
            round(f1_weighted_test, 3),
        )

        # ==========================
        # 7b) SAVE train/test-level info for this fold
        # ==========================
        # test predictions for this fold
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

        # train predictions for this fold
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
        all_train_dfs.append(train_out_fold)

        # ---- Confusion matrix for this fold (for inspection) ----
        order = ["antagonism", "synergy"]
        order_idx = le.transform(order)
        cm = confusion_matrix(y_te, y_pred, labels=order_idx)

        if cm_total is None:
            cm_total = cm.copy()
        else:
            cm_total += cm

        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=order)
        disp.plot(cmap="Blues", ax=ax, values_format="d")
        ax.set_title(
            f"Confusion Matrix — True vs Predicted Interaction Type "
            f"(CC-only reduced features, fold {fold_idx})",
            fontsize=13,
            pad=15,
        )
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        plt.tight_layout()
        fig_path = out_dir / f"confusion_matrix_{SCHEME.lower()}_fold{fold_idx}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print("\nSaved per-fold confusion matrix plot to:", fig_path)

        # store fold metrics (for per-fold panel + summary)
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
                roc_auc_train=roc_auc_train,
                accuracy_train=accuracy_train,
                f1_weighted_train=f1_weighted_train,
                f1_macro_train=f1_macro_train,
                n_train=len(tr_idx),
                n_test=len(te_idx),
            )
        )

    # ==========================
    # 8) Aggregate across folds
    # ==========================

    # 8a) Best params JSON (keep same filename, richer content)
    best_params_path = out_dir / f"best_params_{SCHEME.lower()}.json"
    # choose overall "best" fold for top-level best_params
    best_overall = max(per_fold_best, key=lambda d: d["best_acc_inner_cv"])
    with open(best_params_path, "w") as f:
        json.dump(
            {
                "scheme": SCHEME,
                "n_folds": len(per_fold_best),
                "best_acc_inner_cv": float(best_overall["best_acc_inner_cv"]),
                "best_params": best_overall["best_params"],
                "per_fold": per_fold_best,
            },
            f,
            indent=2,
        )
    print("Saved best params (incl. per-fold) to:", best_params_path)

    # 8b) Feature importances averaged over folds
    if per_fold_importances:
        fi_all = pd.concat(per_fold_importances, ignore_index=True)
        agg_cols = {}
        for col in fi_all.columns:
            if col in ["feature", "fold"]:
                continue
            if pd.api.types.is_numeric_dtype(fi_all[col]):
                agg_cols[col] = "mean"

        fi_mean = (
            fi_all.groupby("feature", as_index=False)
            .agg(agg_cols)
            .sort_values("importance_gain", ascending=False)
            .reset_index(drop=True)
        )
        # recompute normalized gain
        total_gain = fi_mean["importance_gain"].sum()
        if total_gain > 0:
            fi_mean["importance_gain_norm"] = fi_mean["importance_gain"] / total_gain

        fi_path = out_dir / f"feature_importances_{SCHEME.lower()}.csv"
        fi_mean.to_csv(fi_path, index=False)
        print("Saved averaged feature importances to:", fi_path)

    # 8c) Concatenate test/train predictions across folds
    test_out_all = pd.concat(all_test_dfs, ignore_index=True)
    train_out_all = pd.concat(all_train_dfs, ignore_index=True)

    test_out_path = out_dir / f"test_predictions_{SCHEME.lower()}.csv"
    train_out_path = out_dir / f"train_predictions_{SCHEME.lower()}.csv"

    test_out_all.to_csv(test_out_path, index=False)
    train_out_all.to_csv(train_out_path, index=False)

    print("Saved aggregated test predictions to:", test_out_path)
    print("Saved aggregated train predictions to:", train_out_path)

    # 8d) Per-fold metrics CSV (for your panel)
    metrics_per_fold_df = pd.DataFrame(fold_results)
    metrics_per_fold_path = out_dir / f"metrics_per_fold_{SCHEME.lower()}.csv"
    metrics_per_fold_df.to_csv(metrics_per_fold_path, index=False)
    print("Saved per-fold metrics to:", metrics_per_fold_path)

    # 8e) Aggregated metrics (Option A: mean ± std over folds)
    def mean_std(key):
        vals = [fr[key] for fr in fold_results]
        return float(np.mean(vals)), float(np.std(vals))

    acc_mean, acc_std = mean_std("accuracy_test")
    f1w_mean, f1w_std = mean_std("f1_weighted_test")
    f1m_mean, f1m_std = mean_std("f1_macro_test")
    auc_mean, auc_std = mean_std("roc_auc_test")

    print("\n" + "=" * 72)
    print(f"=== Summary over {len(fold_results)} outer folds (CV1, CC-only) ===")
    print(f"roc_auc_test_mean={auc_mean:.4f}  roc_auc_test_std={auc_std:.4f}")
    print(f"accuracy_test_mean={acc_mean:.4f}  accuracy_test_std={acc_std:.4f}")
    print(f"f1_weighted_test_mean={f1w_mean:.4f}  f1_weighted_test_std={f1w_std:.4f}")
    print(f"f1_macro_test_mean={f1m_mean:.4f}  f1_macro_test_std={f1m_std:.4f}")
    print("=" * 72)

    # Per-class metrics (mean ± std over folds)
    prec_a_mean, prec_a_std = mean_std("precision_antag")
    rec_a_mean,  rec_a_std  = mean_std("recall_antag")
    f1_a_mean,   f1_a_std   = mean_std("f1_antag")

    prec_s_mean, prec_s_std = mean_std("precision_syn")
    rec_s_mean,  rec_s_std  = mean_std("recall_syn")
    f1_s_mean,   f1_s_std   = mean_std("f1_syn")

    print("Per-class metrics over outer folds:")
    print(
        f"ANTAGONISM: "
        f"precision_mean={prec_a_mean:.4f} (±{prec_a_std:.4f}), "
        f"recall_mean={rec_a_mean:.4f} (±{rec_a_std:.4f}), "
        f"f1_mean={f1_a_mean:.4f} (±{f1_a_std:.4f})"
    )
    print(
        f"SYNERGY   : "
        f"precision_mean={prec_s_mean:.4f} (±{prec_s_std:.4f}), "
        f"recall_mean={rec_s_mean:.4f} (±{rec_s_std:.4f}), "
        f"f1_mean={f1_s_mean:.4f} (±{f1_s_std:.4f})"
    )
    print("=" * 72)


    # Save aggregated test metrics (keep filename, add *_std + n_folds)
    metrics_test = {
        "scheme": SCHEME,
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
    metrics_test_path = out_dir / f"metrics_test_{SCHEME.lower()}.csv"
    metrics_test_df.to_csv(metrics_test_path, index=False)
    print("Saved aggregated test metrics to:", metrics_test_path)

    # Similarly aggregate train metrics
    auc_tr_mean, auc_tr_std = mean_std("roc_auc_train")
    acc_tr_mean, acc_tr_std = mean_std("accuracy_train")
    f1w_tr_mean, f1w_tr_std = mean_std("f1_weighted_train")
    f1m_tr_mean, f1m_tr_std = mean_std("f1_macro_train")

    metrics_train = {
        "scheme": SCHEME,
        "n_folds": len(fold_results),
        "accuracy_train": acc_tr_mean,
        "accuracy_train_std": acc_tr_std,
        "f1_macro_train": f1m_tr_mean,
        "f1_macro_train_std": f1m_tr_std,
        "f1_weighted_train": f1w_tr_mean,
        "f1_weighted_train_std": f1w_tr_std,
        "roc_auc_train": auc_tr_mean,
        "roc_auc_train_std": auc_tr_std,
        "n_train": int(sum(fr["n_train"] for fr in fold_results)),
    }
    metrics_train_df = pd.DataFrame([metrics_train])
    metrics_train_path = out_dir / f"metrics_train_{SCHEME.lower()}.csv"
    metrics_train_df.to_csv(metrics_train_path, index=False)
    print("Saved aggregated train metrics to:", metrics_train_path)

    # 8f) Aggregated confusion matrix → SAVE
    order = ["antagonism", "synergy"]
    order_idx = le.transform(order)
    cm_all = cm_total if cm_total is not None else np.zeros((2, 2), dtype=int)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_all, display_labels=order)
    disp.plot(cmap="Blues", ax=ax, values_format="d")

    ax.set_title(
        "Confusion Matrix — Aggregated over outer folds "
        "(CC-only reduced features, CV1)",
        fontsize=13,
        pad=15,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

    plt.tight_layout()
    # keep original filename for “final” confusion matrix
    fig_path = out_dir / f"confusion_matrix_{SCHEME.lower()}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print("\nSaved aggregated confusion matrix plot to:", fig_path)
    print("\n=== EXP06d DONE ===\n")


if __name__ == "__main__":
    main()
