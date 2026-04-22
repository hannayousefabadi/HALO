#!/usr/bin/env python3
"""
Experiment: exp06_lgbm_bin_sspace_elementwise_reduced_nestedcv

Config
- model: LightGBM (LGBMClassifier)
- task: binary classification (synergy vs antagonism)
- feature_design: reduced elementwise similarity (subset of cos_elem_* and euc_elem_* selected in exp05)
- sspace: enabled (selection was performed on CC+S-space elementwise features; features are already baked into input)
- nested_cv: enabled
- cv_scheme: cv1
  - outer split: drug-pair disjoint (GroupShuffleSplit; groups=Drug Pair)
  - inner CV: 3-fold grouped CV (StratifiedGroupKFold fallback GroupKFold; groups=Drug Pair) for hyperparameter search
- bliss neutrality cutoff: ±0.1 (labels assumed created upstream; input is synergy/antagonism only)

Feature set provenance
The reduced feature list is taken from exp05 (feature selection performed on a CV1 train split).
This script trains and evaluates models using that fixed reduced feature space.

Data integrity note
All preprocessing (missing values, dtypes, column validation, and label construction from Bliss using the
±0.1 cutoff) is performed upstream. This script assumes the input CSV is clean and contains only valid,
numeric reduced features plus metadata.
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

# ==========================
# Paths
# ==========================
from halo.paths import MODEL_RESULTS


def main():
    # ==========================
    # 0) Basic config
    # ==========================
    SCHEME = "CV1"  # keep same scheme as exp03 for fair comparison

    # path to the reduced elementwise dataset produced by exp05
    filtered_path = MODEL_RESULTS / "exp05_lgbm_bin_sspace_elementwise_featselect" / "elementwise_features_filtered_cv1.csv"

    out_dir = MODEL_RESULTS / "exp06_lgbm_bin_sspace_elementwise_reduced_nestedcv"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== EXP06: LGBM bin + reduced elementwise (from exp05) + nested CV ===\n")
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
    print(f"Reduced feature columns: {len(feat_cols)}")

    # ==========================
    # 3) Outer splits (CV1 / CV2) – same logic as exp03
    # ==========================
    def make_split_cv1(verbose=True):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tr_idx, te_idx = next(gss.split(X, y_enc, groups=pairs))

        if verbose:
            print("=" * 72)
            print("CV1: Drug Pair grouping:")
            print(f"Train size: {len(tr_idx)}")
            print(f"Train size fraction: {round((len(tr_idx) / len(df)), 2) * 100}")
            print(f"Test size: {len(te_idx)}")
            print(f"Test size fraction: {round((len(te_idx) / len(df)), 2) * 100}")
            print(f"Test + Train set: {len(tr_idx) + len(te_idx)}")
            print("-" * 72)
        return tr_idx, te_idx

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

    if SCHEME == "CV1":
        tr_idx, te_idx = make_split_cv1()
        info = None
    elif SCHEME == "CV2":
        tr_idx, te_idx, info = make_split_cv2()
        print("test strains:", info["test_strains"])
    else:
        raise ValueError("SCHEME must be 'CV1' or 'CV2'")

    # ==========================
    # 4) Inner CV (nested)
    # ==========================
    class SilentLogger:
        def info(self, msg):
            pass

        def warning(self, msg):
            pass

    lgb.register_logger(SilentLogger())

    X_tr = X.iloc[tr_idx].reset_index(drop=True)
    X_te = X.iloc[te_idx].reset_index(drop=True)
    y_tr = y_enc[tr_idx]
    y_te = y_enc[te_idx]
    grp_tr = pairs[tr_idx]

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
                    lgb.early_stopping(200, False),
                    lgb.log_evaluation(0),
                ],
            )

            y_pred = m.predict(Xf_val)
            accs.append(accuracy_score(yf_val, y_pred))
        return float(np.mean(accs))

    print("\n--- Nested CV: inner search over", len(param_samples), "configs ---")
    scores = [(cv_acc_for_params(ps), ps) for ps in param_samples]
    scores.sort(reverse=True, key=lambda t: t[0])
    best_acc, best_params = scores[0]
    print("Best inner-CV ACC:", round(best_acc, 3), "\nBest params:", best_params)

    # ==========================
    # 5) Final refit on outer-train
    # ==========================
    m_final = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=4000,
        random_state=777,
        n_jobs=4,
        **best_params,
    )
    m_final.fit(X_tr, y_tr)

    synergy_code = le.transform(["synergy"])[0]
    pos_idx = np.flatnonzero(m_final.classes_ == synergy_code)[0]

    # ==========================
    # 6) Final evaluation on held-out test
    # ==========================
    p_te = m_final.predict_proba(X_te)[:, pos_idx]
    y_pred = (p_te >= 0.5).astype(int)

    # make sure "synergy" is the positive class for AUC
    y_te_bin = (y_te == synergy_code).astype(int)

    # ---- Global metrics (compute once) ----
    accuracy_test = accuracy_score(y_te, y_pred)
    f1_macro_test = f1_score(y_te, y_pred, average="macro")
    f1_weighted_test = f1_score(y_te, y_pred, average="weighted")
    roc_auc_test = roc_auc_score(y_te_bin, p_te)

    print("\n=== Held-out Test ===")
    print(f"ROC AUC : {roc_auc_test:.3f}")
    print(f"Acc     : {accuracy_test:.3f}")
    print(f"F1 (w)  : {f1_weighted_test:.3f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_te, y_pred))
    print(
        "\nReport:\n",
        classification_report(y_te, y_pred, target_names=le.classes_),
    )

    # ---- Per-class metrics (antagonism, synergy) ----
    ant_code = le.transform(["antagonism"])[0]
    syn_code = synergy_code

    prec, rec, f1s, _ = precision_recall_fscore_support(
        y_te,
        y_pred,
        labels=[ant_code, syn_code],
    )

    precision_antag, precision_syn = prec
    recall_antag, recall_syn = rec
    f1_antag, f1_syn = f1s

    # ---- Log-friendly lines (for grep / parsing) ----
    print("\n--- Metrics for Log ---")
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
    # 7) Overfitting check
    # ==========================
    p_tr = m_final.predict_proba(X_tr)[:, pos_idx]
    y_tr_pred = (p_tr >= 0.5).astype(int)
    y_tr_bin = (y_tr == synergy_code).astype(int)

    print("\n=== Overfitting check ===")
    print(
        "Train AUC:",
        round(roc_auc_score(y_tr_bin, p_tr), 3),
        "| Test AUC:",
        round(roc_auc_test, 3),
    )
    print(
        "Train Acc:",
        round(accuracy_score(y_tr, y_tr_pred), 3),
        "| Test Acc:",
        round(accuracy_test, 3),
    )
    print(
        "Train F1w:",
        round(f1_score(y_tr, y_tr_pred, average="weighted"), 3),
        "| Test F1w:",
        round(f1_weighted_test, 3),
    )

    # ==========================
    # 8) Confusion matrix plot → SAVE
    # ==========================
    order = ["antagonism", "synergy"]
    order_idx = le.transform(order)
    cm = confusion_matrix(y_te, y_pred, labels=order_idx)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=order)
    disp.plot(cmap="Blues", ax=ax, values_format="d")

    ax.set_title(
        "Confusion Matrix — True vs Predicted Interaction Type (reduced features)",
        fontsize=13,
        pad=15,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

    plt.tight_layout()
    fig_path = out_dir / f"confusion_matrix_{SCHEME.lower()}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print("\nSaved confusion matrix plot to:", fig_path)
    print("\n=== EXP06 DONE ===\n")


if __name__ == "__main__":
    main()
