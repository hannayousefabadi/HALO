#!/usr/bin/env python3
"""
Experiment: exp07_lgbm_multi_sspace_elementwise_reduced_nestedcv (HALO-S-CV1, multiclass)

Config
- task: multiclass classification (antagonism / neutral / synergy) using Bliss additivity cutoff ±0.10
- feature_design: reduced elementwise similarity features (selected in exp05; includes CC + S-space dimensions)
- use_sspace: true (S-space is already included in the reduced feature table; no feature recomputation here)
- cv_scheme: CV1 by default (drug-pair held-out via GroupShuffleSplit, 80/20); optional CV2 (strain + pair disjoint)
- nested_cv: true
    - inner_cv: 3-fold grouped CV by Drug Pair (StratifiedGroupKFold if available, else GroupKFold)
    - model selection: 32 sampled LightGBM configurations ranked by mean inner-CV macro-F1
- model: LightGBM multiclass (objective="multiclass", num_class=3)
- evaluation: held-out outer test set; reports macro-OVR ROC AUC, accuracy, macro-F1, weighted-F1, and per-class precision/recall/F1
- outputs: saved confusion matrix plot for the selected scheme (confusion_matrix_{scheme}.png) plus console-logged metrics

**Data integrity note:**
All preprocessing (NA handling, dtype enforcement, column validation, etc.)
was completed in the preprocessing scripts.
This notebook assumes clean, validated input data.
"""


import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    GroupShuffleSplit,
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
    precision_recall_fscore_support
)

from halo.paths import MODEL_RESULTS

def main():
    # ==========================
    # 0) Basic config
    # ==========================
    SCHEME = "CV1"  # or "CV2"

    # Path to exp05 output file
    filtered_path = MODEL_RESULTS / "exp05_lgbm_bin_sspace_elementwise_featselect" / "elementwise_features_filtered_cv1_full.csv"

    out_dir = MODEL_RESULTS / "exp07_lgbm_multi_sspace_elementwise_reduced_nestedcv"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== EXP07: MULTI-CLASS + reduced elementwise features ===")
    print("Using reduced CSV:", filtered_path)
    print("SCHEME:", SCHEME)
    print("Output:", out_dir)

    rng = np.random.default_rng(42)

    # ==========================
    # 1) Load exp05 reduced feature matrix
    # ==========================
    if not filtered_path.exists():
        raise FileNotFoundError(f"Filtered CSV not found: {filtered_path}")

    df = pd.read_csv(filtered_path).copy()
    print("\nLoaded df:", df.shape)
    print(df["Interaction Type"].value_counts())

    # keep 3 classes
    df = df[df["Interaction Type"].isin(["synergy", "antagonism", "neutral"])].copy()
    print("\nAfter filtering 3 classes:", df.shape)

    # ==========================
    # 2) Define feature columns
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

    print("\nReduced feature columns:", len(feat_cols))

    X = df[feat_cols].copy()
    y = df["Interaction Type"].copy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    pairs = df["Drug Pair"].astype(str).values
    strains = df["Strain"].astype(str).values
    n = len(df)

    print("Classes:", list(le.classes_))
    print("Total samples:", n)

    # ==========================
    # 3) Outer CV split (CV1 / CV2)
    # ==========================
    def make_split_cv1(verbose=True):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tr_idx, te_idx = next(gss.split(X, y_enc, groups=pairs))
        if verbose:
            print("=" * 72)
            print("CV1 split by Drug Pair")
            print("Train:", len(tr_idx), "Test:", len(te_idx))
            print("=" * 72)
        return tr_idx, te_idx

    def make_split_cv2(
        strain_col="Strain",
        pair_col="Drug Pair",
        min_frac=0.16,
        max_frac=0.20,
        lambda_penalty=1.0,
        top_k_print=5,
        verbose=True
    ):
        # exact copy of exp03 logic
        S_all = df[strain_col].astype(str).values
        P_all = df[pair_col].astype(str).values
        strains_uni = sorted(np.unique(S_all).tolist())
        n_total = len(df)
        min_target = int(round(min_frac * n_total))
        max_target = int(round(max_frac * n_total))

        if verbose:
            print("=" * 72)
            print("CV2: Strain + Drug Pair split")
            print(f"Rows: {n_total}  Strains: {len(strains_uni)}")

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
                    candidates.append(dict(
                        S_test=S_test,
                        P_test=P_test,
                        kept=kept,
                        train=train,
                        dropped=dropped,
                        score=score
                    ))

        if not candidates:
            print("\nCV2 could not find a valid split.")
            return np.arange(n_total), np.array([], dtype=int), None

        candidates.sort(key=lambda c: (c["dropped"], -c["kept"], -c["score"]))
        best = candidates[0]

        S_test_best = best["S_test"]
        P_test_best = best["P_test"]

        test_mask = np.isin(S_all, list(S_test_best)) & np.isin(P_all, list(P_test_best))
        train_mask = (~np.isin(S_all, list(S_test_best))) & (~np.isin(P_all, list(P_test_best)))

        te_idx = np.where(test_mask)[0]
        tr_idx = np.where(train_mask)[0]

        if verbose:
            print("=" * 72)
            print("Chosen CV2 split:")
            print("#Train:", len(tr_idx), "#Test:", len(te_idx))
            print("=" * 72)

        return tr_idx, te_idx, best

    if SCHEME == "CV1":
        tr_idx, te_idx = make_split_cv1()
        info = None
    else:
        tr_idx, te_idx, info = make_split_cv2()

    # ==========================
    # 4) Prepare nested CV folds
    # ==========================
    class SilentLogger:
        def info(self, msg): pass
        def warning(self, msg): pass

    lgb.register_logger(SilentLogger())

    X_tr = X.iloc[tr_idx].reset_index(drop=True)
    X_te = X.iloc[te_idx].reset_index(drop=True)
    y_tr = y_enc[tr_idx]
    y_te = y_enc[te_idx]
    grp_tr = pairs[tr_idx]

    # ==========================
    # 5) Hyperparameter search
    # ==========================
    def sample_one_params():
        max_depth = 3
        leaves_map = {3: [7, 15]}
        return dict(
            boosting_type=rng.choice(["gbdt", "dart"], p=[0.6, 0.4]),
            learning_rate=float(rng.choice([0.02, 0.03, 0.04, 0.05])),
            max_depth=max_depth,
            num_leaves=int(rng.choice(leaves_map[max_depth])),
            min_data_in_leaf=200,
            feature_fraction=float(rng.choice([0.30, 0.40])),
            bagging_fraction=float(rng.choice([0.60, 0.80])),
            bagging_freq=1,
            lambda_l2=float(10 ** rng.uniform(1.2, 1.7)),
            lambda_l1=float(rng.choice([0.0, 0.1, 0.5])),
            max_bin=int(rng.choice([63, 127])),
            min_gain_to_split=float(rng.choice([0.05, 0.10, 0.20])),
        )

    param_samples = [sample_one_params() for _ in range(32)]

    try:
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=111)
        def inner_splitter():
            return inner_cv.split(X_tr, y_tr, groups=grp_tr)
    except Exception:
        inner_cv = GroupKFold(n_splits=3)
        def inner_splitter():
            return inner_cv.split(X_tr, y_tr, groups=grp_tr)

    def cv_acc_for_params(params):
        scores = []
        for tr_f, val_f in inner_splitter():
            Xf_tr, Xf_val = X_tr.iloc[tr_f], X_tr.iloc[val_f]
            yf_tr, yf_val = y_tr[tr_f], y_tr[val_f]

            m = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=4000,
                random_state=777,
                n_jobs=4,
                **params,
            )
            m.fit(
                Xf_tr,
                yf_tr,
                eval_set=[(Xf_val, yf_val)],
                eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(200, False)],
            )

            probs = m.predict_proba(Xf_val)
            y_pred = np.argmax(probs, axis=1)
            scores.append(f1_score(yf_val, y_pred, average="macro"))

        return float(np.mean(scores))
    

    print("\n--- Inner CV hyperparameter search ---")
    scores = [(cv_acc_for_params(ps), ps) for ps in param_samples]
    scores.sort(reverse=True)
    best_acc, best_params = scores[0]
    print("Best inner-CV ACC:", round(best_acc, 3))
    print("Best params:", best_params)

    # ==========================
    # 6) Final refit
    # ==========================
    m_final = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=4000,
        random_state=777,
        n_jobs=4,
        **best_params,
    )
    m_final.fit(X_tr, y_tr)

    # ==========================
    # 7) Final evaluation
    # ==========================
    probs_te = m_final.predict_proba(X_te)
    y_pred = np.argmax(probs_te, axis=1)

    # ---- Global metrics (compute once) ----
    roc_auc_test = roc_auc_score(
        y_te,
        probs_te,
        multi_class="ovr",
        average="macro",
        labels=m_final.classes_,
    )
    accuracy_test = accuracy_score(y_te, y_pred)
    f1_weighted_test = f1_score(y_te, y_pred, average="weighted")
    f1_macro_test = f1_score(y_te, y_pred, average="macro")

    print("\n=== Held-out Test Metrics ===")
    print(f"Macro ROC-AUC: {roc_auc_test:.3f}")
    print(f"Acc          : {accuracy_test:.3f}")
    print(f"F1 (weighted): {f1_weighted_test:.3f}")
    print(f"F1 (macro)   : {f1_macro_test:.3f}")

    print("\nConfusion matrix:\n", confusion_matrix(y_te, y_pred))
    print(
        "\nClassification Report:\n",
        classification_report(y_te, y_pred, target_names=le.classes_),
    )

    # ---- Per-class metrics (antagonism, neutral, synergy) ----
    ant_code = le.transform(["antagonism"])[0]
    neu_code = le.transform(["neutral"])[0]
    syn_code = le.transform(["synergy"])[0]

    prec, rec, f1s, _ = precision_recall_fscore_support(
        y_te,
        y_pred,
        labels=[ant_code, neu_code, syn_code],
    )

    precision_antag, precision_neutral, precision_syn = prec
    recall_antag, recall_neutral, recall_syn = rec
    f1_antag, f1_neutral, f1_syn = f1s

    # ---- Log-friendly lines (for grep / parsing) ----
    print("\n--- Metrics for Log ---")
    print(f"accuracy_test={accuracy_test:.4f}")
    print(f"f1_macro_test={f1_macro_test:.4f}")
    print(f"f1_weighted_test={f1_weighted_test:.4f}")
    print(f"roc_auc_test={roc_auc_test:.4f}")

    print(f"precision_antag={precision_antag:.4f}")
    print(f"recall_antag={recall_antag:.4f}")
    print(f"f1_antag={f1_antag:.4f}")

    print(f"precision_neutral={precision_neutral:.4f}")
    print(f"recall_neutral={recall_neutral:.4f}")
    print(f"f1_neutral={f1_neutral:.4f}")

    print(f"precision_syn={precision_syn:.4f}")
    print(f"recall_syn={recall_syn:.4f}")
    print(f"f1_syn={f1_syn:.4f}")


    # ==========================
    # 8) Overfitting check
    # ==========================
    probs_tr = m_final.predict_proba(X_tr)
    y_tr_pred = np.argmax(probs_tr, axis=1)

    auc_tr = roc_auc_score(
        y_tr,
        probs_tr,
        multi_class="ovr",
        average="macro",
        labels=m_final.classes_,
    )
    acc_tr = accuracy_score(y_tr, y_tr_pred)
    f1_macro_tr = f1_score(y_tr, y_tr_pred, average="macro")

    print("\n=== Overfitting check ===")
    print(f"Train AUC: {auc_tr:.3f} | Test AUC: {roc_auc_test:.3f}")
    print(f"Train Acc: {acc_tr:.3f} | Test Acc: {accuracy_test:.3f}")
    print(f"Train F1 macro: {f1_macro_tr:.3f} | Test F1 macro: {f1_macro_test:.3f}")


    # ==========================
    # 9) Confusion matrix → SAVE
    # ==========================
    order = ["antagonism", "neutral", "synergy"]
    order_idx = le.transform(order)
    cm = confusion_matrix(y_te, y_pred, labels=order_idx)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=order)
    disp.plot(cmap="Blues", ax=ax)

    plt.title("Confusion Matrix — Multi-class (reduced elementwise)")
    plt.tight_layout()

    fig_path = out_dir / f"confusion_matrix_{SCHEME.lower()}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print("\nSaved confusion matrix:", fig_path)
    print("\n=== EXP07 DONE ===")


if __name__ == "__main__":
    main()
