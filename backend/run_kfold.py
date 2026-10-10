"""
run_kfold.py — 5-Fold Stratified Cross-Validation (no_leakage feature set)
===========================================================================
Trains and evaluates all 5 classifiers using StratifiedKFold(n_splits=5,
shuffle=True, random_state=42) on the no_leakage feature set only.

Per-fold procedure
──────────────────
  • SMOTE applied inside each fold on the training split only
  • LR / SVC:   features scaled with StandardScaler fit on the fold's train split
  • SVC:        uses a stratified sub-sample of the fold's training data
                (SVC_CV_SAMPLE per fold) to keep runtime tractable;
                the sub-sample size is stored in the output
  • DT / RF / XGB: raw (unscaled) SMOTE-resampled training features

Metrics reported per model
──────────────────────────
  roc_auc_mean / roc_auc_std
  f1_mean      / f1_std
  precision_mean / precision_std
  recall_mean    / recall_std

Output saved to:
  models/metrics.json        — aggregated mean ± std under key "kfold_no_leakage"
  data/kfold_raw_scores.json — raw per-fold scores for each model
All pre-existing keys in metrics.json are preserved unchanged.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ─── Paths ────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(__file__)
DATASET_PATH = os.path.join(_HERE, "..", "AIML Dataset.csv")
MODELS_DIR   = os.path.join(_HERE, "..", "models")
DATA_DIR     = os.path.join(_HERE, "..", "data")

# SVC sample size per fold (pre-SMOTE train slice); same rationale as main train.py
SVC_CV_SAMPLE    = 10_000   # per-fold sample — smaller than full-train 50K because
                             # each fold train split is ~80% of the already-sampled data;
                             # 10K keeps each fold under ~10 s on a laptop CPU

LEAKAGE_FEATURES = ["errorBalanceOrig", "errorBalanceDest"]
N_SPLITS         = 5
RANDOM_STATE     = 42


# ─── 1. Load & engineer features ─────────────────────────────────────────────
def load_and_engineer(path: str, sample_size: int = 200_000):
    print(f"[kfold] Loading dataset …")
    df = pd.read_csv(path)
    fraud     = df[df["isFraud"] == 1]
    non_fraud = df[df["isFraud"] == 0].sample(
        n=min(sample_size, len(df[df["isFraud"] == 0])), random_state=RANDOM_STATE
    )
    df = (
        pd.concat([fraud, non_fraud])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    print(f"[kfold] Sampled {len(df):,} rows  (fraud rate: {df['isFraud'].mean():.4%})")

    le = LabelEncoder()
    df["type_enc"] = le.fit_transform(df["type"])

    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["origZeroBalance"]  = (df["oldbalanceOrg"] == 0).astype(int)
    df["destZeroBalance"]  = (df["oldbalanceDest"] == 0).astype(int)
    df["amountRatio"]      = df["amount"] / (df["oldbalanceOrg"] + 1)

    feature_cols_with = [
        "step", "type_enc", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "errorBalanceOrig", "errorBalanceDest",
        "origZeroBalance", "destZeroBalance",
        "amountRatio",
    ]
    feature_cols_no = [c for c in feature_cols_with if c not in LEAKAGE_FEATURES]

    X = df[feature_cols_no].values
    y = df["isFraud"].values
    return X, y


# ─── 2. Build model factories (fresh instance each fold) ─────────────────────
def make_models(scale_pos: float):
    """Return a dict of (name, model, needs_scaling) tuples."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced",
                               random_state=RANDOM_STATE),
            True,   # needs StandardScaler
        ),
        "svc": (
            SVC(kernel="rbf", probability=True, class_weight="balanced",
                random_state=RANDOM_STATE, cache_size=500),
            True,
        ),
        "decision_tree": (
            DecisionTreeClassifier(max_depth=10, class_weight="balanced",
                                   random_state=RANDOM_STATE),
            False,
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1,
                                   random_state=RANDOM_STATE,
                                   class_weight="balanced"),
            False,
        ),
        "xgboost": (
            XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=scale_pos,
                          eval_metric="logloss",
                          random_state=RANDOM_STATE, n_jobs=-1),
            False,
        ),
    }


# ─── 3. Run K-Fold CV ─────────────────────────────────────────────────────────
def run_kfold(X: np.ndarray, y: np.ndarray) -> dict:
    skf   = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                            random_state=RANDOM_STATE)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)

    # Accumulate per-fold scores: {model_key: {metric: [fold1, fold2, ...]}}
    scores = {
        mk: {"roc_auc": [], "f1": [], "precision": [], "recall": []}
        for mk in ("logistic_regression", "svc", "decision_tree",
                   "random_forest", "xgboost")
    }

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n[kfold] -- Fold {fold_idx}/{N_SPLITS} --------------------------")
        X_tr_raw, X_te = X[train_idx], X[test_idx]
        y_tr_raw, y_te = y[train_idx], y[test_idx]

        # SMOTE on fold training split
        X_tr, y_tr = smote.fit_resample(X_tr_raw, y_tr_raw)
        print(f"[kfold]   SMOTE -> train {X_tr.shape[0]:,} "
              f"(fraud={y_tr.sum():,})")

        # Class weight ratio for XGBoost (computed from SMOTE'd train set)
        scale_pos = float((y_tr == 0).sum()) / float((y_tr == 1).sum())

        # Scaled versions for LR / SVC
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr)
        X_te_sc  = scaler.transform(X_te)

        # SVC sub-sample — stratified slice of the pre-SMOTE fold train
        n_svc = min(SVC_CV_SAMPLE, len(X_tr_raw))
        _, X_svc, _, y_svc = train_test_split(
            X_tr_raw, y_tr_raw,
            test_size=n_svc / len(X_tr_raw),
            random_state=RANDOM_STATE, stratify=y_tr_raw,
        )
        scaler_svc = StandardScaler()
        X_svc_sc   = scaler_svc.fit_transform(X_svc)
        X_te_svc   = scaler_svc.transform(X_te)
        print(f"[kfold]   SVC sample: {len(X_svc):,} rows (from fold train split)")

        models = make_models(scale_pos)

        for mk, (model, needs_scale) in models.items():
            if mk == "svc":
                model.fit(X_svc_sc, y_svc)
                prob = model.predict_proba(X_te_svc)[:, 1]
            elif needs_scale:
                model.fit(X_tr_sc, y_tr)
                prob = model.predict_proba(X_te_sc)[:, 1]
            else:
                model.fit(X_tr, y_tr)
                prob = model.predict_proba(X_te)[:, 1]

            pred = (prob >= 0.5).astype(int)
            roc  = float(roc_auc_score(y_te, prob))
            p, r, f1, _ = precision_recall_fscore_support(
                y_te, pred, average="binary", zero_division=0
            )
            scores[mk]["roc_auc"].append(roc)
            scores[mk]["f1"].append(float(f1))
            scores[mk]["precision"].append(float(p))
            scores[mk]["recall"].append(float(r))
            print(f"[kfold]   {mk:<22}  "
                  f"roc={roc:.4f}  f1={f1:.4f}  "
                  f"prec={p:.4f}  rec={r:.4f}")

    # Aggregate: mean ± std over 5 folds
    results = {}
    for mk, fold_scores in scores.items():
        results[mk] = {
            "roc_auc_mean":   round(float(np.mean(fold_scores["roc_auc"])),   4),
            "roc_auc_std":    round(float(np.std( fold_scores["roc_auc"])),   4),
            "f1_mean":        round(float(np.mean(fold_scores["f1"])),        4),
            "f1_std":         round(float(np.std( fold_scores["f1"])),        4),
            "precision_mean": round(float(np.mean(fold_scores["precision"])), 4),
            "precision_std":  round(float(np.std( fold_scores["precision"])), 4),
            "recall_mean":    round(float(np.mean(fold_scores["recall"])),    4),
            "recall_std":     round(float(np.std( fold_scores["recall"])),    4),
            "n_splits":       N_SPLITS,
            "feature_set":    "no_leakage",
        }
        if mk == "svc":
            results[mk]["svc_cv_sample_size"] = SVC_CV_SAMPLE

    return results, scores


# ─── 4. Merge into metrics.json and save raw scores ──────────────────────────
def save_kfold_results(kfold_results: dict, raw_scores: dict):
    # Aggregated results → models/metrics.json
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    all_metrics["kfold_no_leakage"] = kfold_results

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[kfold] Aggregated results saved → {metrics_path}  (key: kfold_no_leakage)")

    # Raw per-fold scores → data/kfold_raw_scores.json
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_path = os.path.join(DATA_DIR, "kfold_raw_scores.json")
    raw_out = {
        "n_splits":          N_SPLITS,
        "feature_set":       "no_leakage",
        "svc_cv_sample_size": SVC_CV_SAMPLE,
        "per_fold_scores":   {
            mk: {metric: [round(v, 6) for v in vals]
                 for metric, vals in fold_scores.items()}
            for mk, fold_scores in raw_scores.items()
        },
    }
    with open(raw_path, "w") as f:
        json.dump(raw_out, f, indent=2)
    print(f"[kfold] Raw per-fold scores saved  → {raw_path}")


# ─── 5. Pretty-print summary ──────────────────────────────────────────────────
def print_kfold_summary(results: dict):
    col_w = [22, 14, 14, 14, 14]
    header = (
        f"{'Model':<{col_w[0]}} {'ROC-AUC':>{col_w[1]}} "
        f"{'F1':>{col_w[2]}} {'Precision':>{col_w[3]}} {'Recall':>{col_w[4]}}"
    )
    sep = "-" * sum(col_w + [4])
    display = {
        "logistic_regression": "LogisticRegression",
        "svc":                 f"SVC (n={SVC_CV_SAMPLE:,}/fold)",
        "decision_tree":       "DecisionTree",
        "random_forest":       "RandomForest",
        "xgboost":             "XGBoost",
    }

    print(f"\n{'='*len(sep)}")
    print(f"  5-Fold CV · no_leakage feature set · mean ± std")
    print(f"{'='*len(sep)}")
    print(header)
    print(sep)
    for mk in ("logistic_regression", "svc", "decision_tree",
               "random_forest", "xgboost"):
        d = results[mk]
        print(
            f"{display[mk]:<{col_w[0]}} "
            f"{d['roc_auc_mean']:.4f}±{d['roc_auc_std']:.4f}  "
            f"{d['f1_mean']:.4f}±{d['f1_std']:.4f}  "
            f"{d['precision_mean']:.4f}±{d['precision_std']:.4f}  "
            f"{d['recall_mean']:.4f}±{d['recall_std']:.4f}"
        )
    print(sep)


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y = load_and_engineer(DATASET_PATH)
    kfold_results, raw_scores = run_kfold(X, y)
    save_kfold_results(kfold_results, raw_scores)
    print_kfold_summary(kfold_results)
    print("\n[kfold] Done.")
