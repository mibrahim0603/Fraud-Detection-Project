"""
train.py — Fraud Detection Model Training Pipeline
Reads AIML Dataset.csv, preprocesses data, trains 5 classifiers on two
feature sets (with_leakage and no_leakage), evaluates on a held-out test set,
and saves all artifacts to /models.

Leakage disclosure
──────────────────
Two engineered features use post-transaction balance states that would NOT be
available at real decision time in a production system:
  • errorBalanceOrig  = newbalanceOrig + amount − oldbalanceOrg
  • errorBalanceDest  = oldbalanceDest + amount − newbalanceDest
These features are highly predictive (SHAP confirms they dominate) because they
perfectly encode whether a fraud-type drain occurred. The reported metrics
(ROC-AUC ~0.9997 on with_leakage) should therefore be understood as an upper
bound on what a model could achieve in a truly online setting. The no_leakage
results give a realistic estimate of deployable performance.
See metrics.json key "leakage_note" for the full disclosure.

Models trained
──────────────
  1. LogisticRegression     (max_iter=1000, class_weight="balanced")
  2. SVC                    (RBF kernel, probability=True; trained on
                             stratified 50 000-row sample to avoid O(n²) cost)
  3. DecisionTreeClassifier (max_depth=10, class_weight="balanced")
  4. RandomForestClassifier (100 trees, max_depth=15)     ← saved to .pkl
  5. XGBClassifier          (200 estimators, depth 6)     ← saved to .pkl

All models trained on BOTH feature sets with the SAME 80/20 stratified split.
SMOTE applied to training split only (except SVC which uses its own sample).
random_state=42 everywhere.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
    average_precision_score,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "AIML Dataset.csv")
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")

LEAKAGE_FEATURES  = ["errorBalanceOrig", "errorBalanceDest"]
SVC_SAMPLE_SIZE   = 50_000   # stratified sample used for SVC to avoid O(n²) kernel cost


# ─── 1. Load & sample ────────────────────────────────────────────────────────
def load_data(path: str, sample_size: int = 200_000) -> pd.DataFrame:
    print(f"[INFO] Loading dataset from {path} ...")
    df = pd.read_csv(path)
    print(f"[INFO] Full dataset shape: {df.shape}")
    fraud     = df[df["isFraud"] == 1]
    non_fraud = df[df["isFraud"] == 0].sample(
        n=min(sample_size, len(df[df["isFraud"] == 0])), random_state=42
    )
    df_sampled = (
        pd.concat([fraud, non_fraud])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    print(f"[INFO] Sampled dataset shape: {df_sampled.shape}")
    print(f"[INFO] Fraud rate: {df_sampled['isFraud'].mean():.4%}")
    return df_sampled


# ─── 2. Feature Engineering ──────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame):
    """
    Returns df with new columns, and all four feature lists, and LabelEncoder.

    Feature sets
    ─────────────
    with_leakage    : all 12 features (post-tx balance states included)
    no_leakage      : 10 features — removes errorBalanceOrig / errorBalanceDest
    strict          : 3 features  — step, type_enc, amount only
    pre_transaction : 5 features  — step, type_enc, amount, oldbalanceOrg, oldbalanceDest
                      (only values knowable before a transaction settles)
    """
    df = df.copy()
    le = LabelEncoder()
    df["type_enc"] = le.fit_transform(df["type"])

    # Leaky features — kept for with_leakage set
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    # Safe features
    df["origZeroBalance"] = (df["oldbalanceOrg"] == 0).astype(int)
    df["destZeroBalance"] = (df["oldbalanceDest"] == 0).astype(int)
    df["amountRatio"]     = df["amount"] / (df["oldbalanceOrg"] + 1)

    feature_cols_with = [
        "step", "type_enc", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "errorBalanceOrig", "errorBalanceDest",
        "origZeroBalance", "destZeroBalance",
        "amountRatio",
    ]
    feature_cols_no      = [f for f in feature_cols_with if f not in LEAKAGE_FEATURES]
    feature_cols_strict  = ["step", "type_enc", "amount"]
    feature_cols_pre_tx  = ["step", "type_enc", "amount", "oldbalanceOrg", "oldbalanceDest"]

    return df, feature_cols_with, feature_cols_no, feature_cols_strict, feature_cols_pre_tx, le


# ─── 3. Metrics helper ───────────────────────────────────────────────────────
def compute_metrics(name: str, y_test, prob, pred) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="binary", zero_division=0
    )
    return {
        "roc_auc":       round(float(roc_auc_score(y_test, prob)), 4),
        "avg_precision": round(float(average_precision_score(y_test, prob)), 4),
        "precision":     round(float(p),  4),
        "recall":        round(float(r),  4),
        "f1":            round(float(f1), 4),
        "confusion_matrix":       confusion_matrix(y_test, pred).tolist(),
        "classification_report":  classification_report(y_test, pred, output_dict=True),
    }


# ─── 4. Train one feature set ────────────────────────────────────────────────
def train_feature_set(
    df: pd.DataFrame,
    feature_cols: list,
    label: str,
    X_test_full: np.ndarray,   # test slice from the FULL feature set (sliced below)
    y_test: np.ndarray,
    col_index: dict,           # maps feature name → column index in X_test_full
) -> dict:
    """
    Trains all 5 models on `feature_cols`, evaluates on the matching columns of
    the shared test split.  Returns per-model metric dicts.
    """
    # Extract the right columns from the shared test array
    col_idx  = [col_index[c] for c in feature_cols]
    X_test   = X_test_full[:, col_idx]

    X        = df[feature_cols].values
    y        = df["isFraud"].values
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # (test set is taken from X_test_full already)

    # SMOTE on training data
    print(f"  [SMOTE] {label} ...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_tr, y_tr = smote.fit_resample(X_train, y_train)
    print(f"  [SMOTE] Done — train size: {X_tr.shape[0]}, fraud: {y_tr.sum()}")

    # Scale once for LR and SVC (tree models don't need it but it doesn't hurt)
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_test)

    results = {}
    scale_pos = (y_tr == 0).sum() / (y_tr == 1).sum()

    # ── 1. Logistic Regression ────────────────────────────────────────────────
    print(f"  [LR] {label} ...")
    lr = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )
    lr.fit(X_tr_sc, y_tr)
    prob = lr.predict_proba(X_te_sc)[:, 1]
    pred = lr.predict(X_te_sc)
    results["logistic_regression"] = compute_metrics("LR", y_test, prob, pred)
    print(f"  [LR]   ROC-AUC={results['logistic_regression']['roc_auc']}  "
          f"F1={results['logistic_regression']['f1']}")

    # ── 2. SVC (stratified sample to avoid O(n²) kernel cost) ────────────────
    print(f"  [SVC] {label} — stratified sample of {SVC_SAMPLE_SIZE:,} rows ...")
    # Sample from the ORIGINAL (pre-SMOTE) training data to keep class ratio real
    n_svc  = min(SVC_SAMPLE_SIZE, len(X_train))
    _, X_svc_raw, _, y_svc_raw = train_test_split(
        X_train, y_train, test_size=n_svc / len(X_train),
        random_state=42, stratify=y_train,
    )
    scaler_svc  = StandardScaler()
    X_svc_sc    = scaler_svc.fit_transform(X_svc_raw)
    X_te_svc_sc = scaler_svc.transform(X_test)
    svc = SVC(kernel="rbf", probability=True, class_weight="balanced",
              random_state=42, cache_size=500)
    svc.fit(X_svc_sc, y_svc_raw)
    prob = svc.predict_proba(X_te_svc_sc)[:, 1]
    pred = svc.predict(X_te_svc_sc)
    results["svc"] = compute_metrics("SVC", y_test, prob, pred)
    results["svc"]["svc_train_sample_size"] = int(len(X_svc_raw))
    print(f"  [SVC]  ROC-AUC={results['svc']['roc_auc']}  "
          f"F1={results['svc']['f1']}  (sample={len(X_svc_raw):,})")

    # ── 3. Decision Tree ──────────────────────────────────────────────────────
    print(f"  [DT]  {label} ...")
    dt = DecisionTreeClassifier(
        max_depth=10, class_weight="balanced", random_state=42
    )
    dt.fit(X_tr, y_tr)
    prob = dt.predict_proba(X_test)[:, 1]
    pred = dt.predict(X_test)
    results["decision_tree"] = compute_metrics("DT", y_test, prob, pred)
    print(f"  [DT]   ROC-AUC={results['decision_tree']['roc_auc']}  "
          f"F1={results['decision_tree']['f1']}")

    # ── 4. Random Forest ──────────────────────────────────────────────────────
    print(f"  [RF]  {label} ...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=15, n_jobs=-1,
        random_state=42, class_weight="balanced"
    )
    rf.fit(X_tr, y_tr)
    prob = rf.predict_proba(X_test)[:, 1]
    pred = rf.predict(X_test)
    results["random_forest"] = compute_metrics("RF", y_test, prob, pred)
    print(f"  [RF]   ROC-AUC={results['random_forest']['roc_auc']}  "
          f"F1={results['random_forest']['f1']}")

    # ── 5. XGBoost ───────────────────────────────────────────────────────────
    print(f"  [XGB] {label} ...")
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    xgb.fit(X_tr, y_tr)
    prob = xgb.predict_proba(X_test)[:, 1]
    pred = xgb.predict(X_test)
    results["xgboost"] = compute_metrics("XGB", y_test, prob, pred)
    print(f"  [XGB]  ROC-AUC={results['xgboost']['roc_auc']}  "
          f"F1={results['xgboost']['f1']}")

    # ── Ensemble (RF + XGB average) ───────────────────────────────────────────
    rf_prob  = rf.predict_proba(X_test)[:, 1]
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    ens_prob = (rf_prob + xgb_prob) / 2
    ens_pred = (ens_prob >= 0.5).astype(int)
    results["ensemble"] = compute_metrics("ENS", y_test, ens_prob, ens_pred)
    print(f"  [ENS]  ROC-AUC={results['ensemble']['roc_auc']}  "
          f"F1={results['ensemble']['f1']}")

    # Feature importances (RF + XGB average, only for tree models)
    fi_rf  = rf.feature_importances_
    fi_xgb = xgb.feature_importances_
    fi_avg = ((fi_rf + fi_xgb) / 2).tolist()
    results["feature_importances"] = dict(zip(feature_cols, fi_avg))

    return rf, xgb, results


# ─── 5. Save artifacts ───────────────────────────────────────────────────────
def save_artifacts(rf, xgb, le, feature_cols_with, all_metrics):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,   exist_ok=True)

    # Always save the with_leakage RF+XGB — used by live prediction + SHAP
    joblib.dump(rf,  os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(xgb, os.path.join(MODELS_DIR, "xgboost.pkl"))
    joblib.dump(le,  os.path.join(MODELS_DIR, "label_encoder.pkl"))

    with open(os.path.join(MODELS_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols_with, f)

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"[INFO] Artifacts saved to {MODELS_DIR}")


# ─── 6. Save EDA snapshot ────────────────────────────────────────────────────
def save_eda_snapshot(df_full: pd.DataFrame):
    os.makedirs(DATA_DIR, exist_ok=True)
    eda = {
        "total_transactions": int(len(df_full)),
        "fraud_count":        int(df_full["isFraud"].sum()),
        "non_fraud_count":    int((df_full["isFraud"] == 0).sum()),
        "fraud_rate":         round(float(df_full["isFraud"].mean()), 6),
        "transaction_types":  df_full["type"].value_counts().to_dict(),
        "fraud_by_type":      df_full.groupby("type")["isFraud"].sum().to_dict(),
        "fraud_rate_by_type": df_full.groupby("type")["isFraud"].mean().round(6).to_dict(),
        "amount_stats": {
            "overall":   df_full["amount"].describe().round(2).to_dict(),
            "fraud":     df_full[df_full["isFraud"] == 1]["amount"].describe().round(2).to_dict(),
            "non_fraud": df_full[df_full["isFraud"] == 0]["amount"].describe().round(2).to_dict(),
        },
        "step_distribution": df_full.groupby("step")["isFraud"].sum().to_dict(),
        "hourly_fraud": {
            str(k): int(v)
            for k, v in df_full.groupby(df_full["step"] % 24)["isFraud"].sum().items()
        },
    }
    with open(os.path.join(DATA_DIR, "eda_snapshot.json"), "w") as f:
        json.dump(eda, f, indent=2)
    print(f"[INFO] EDA snapshot saved to {DATA_DIR}/eda_snapshot.json")


# ─── 7. Summary table ────────────────────────────────────────────────────────
def print_summary(all_metrics: dict):
    model_keys = [
        "logistic_regression", "svc", "decision_tree",
        "random_forest", "xgboost", "ensemble",
    ]
    display = {
        "logistic_regression": "LogisticRegression",
        "svc":                 "SVC (sample)",
        "decision_tree":       "DecisionTree",
        "random_forest":       "RandomForest",
        "xgboost":             "XGBoost",
        "ensemble":            "Ensemble (RF+XGB)",
    }
    col_w = [20, 10, 10, 10, 10, 8]
    header = (
        f"{'Model':<{col_w[0]}} {'ROC-AUC':>{col_w[1]}} "
        f"{'AvgPrec':>{col_w[2]}} {'Precision':>{col_w[3]}} "
        f"{'Recall':>{col_w[4]}} {'F1':>{col_w[5]}}"
    )
    sep = "-" * sum(col_w + [5])

    for fs_key, fs_label in [
        ("with_leakage",    "WITH leakage features"),
        ("no_leakage",      "WITHOUT leakage features (deployable)"),
        ("strict",          "STRICT — step, type_enc, amount only"),
        ("pre_transaction", "PRE-TRANSACTION — step, type_enc, amount, oldbalanceOrg, oldbalanceDest"),
    ]:
        fs = all_metrics.get(fs_key, {})
        print(f"\n{'='*len(sep)}")
        print(f"  Feature set: {fs_label}")
        print(f"{'='*len(sep)}")
        print(header)
        print(sep)
        for mk in model_keys:
            m = fs.get(mk)
            if m is None:
                continue
            svc_note = f" [{m['svc_train_sample_size']:,}]" if "svc_train_sample_size" in m else ""
            name = display[mk] + svc_note
            print(
                f"{name:<{col_w[0]}} "
                f"{m['roc_auc']:>{col_w[1]}.4f} "
                f"{m['avg_precision']:>{col_w[2]}.4f} "
                f"{m['precision']:>{col_w[3]}.4f} "
                f"{m['recall']:>{col_w[4]}.4f} "
                f"{m['f1']:>{col_w[5]}.4f}"
            )
        print(sep)

    # Delta table
    print(f"\n{'='*len(sep)}")
    print("  Leakage impact: metric drop when leaky features are removed (with - no_leakage)")
    print(f"{'='*len(sep)}")
    print(header)
    print(sep)
    for mk in model_keys:
        m_with = all_metrics.get("with_leakage", {}).get(mk)
        m_no   = all_metrics.get("no_leakage",   {}).get(mk)
        if m_with is None or m_no is None:
            continue
        name = display[mk]
        vals = [
            m_with['roc_auc']       - m_no['roc_auc'],
            m_with['avg_precision'] - m_no['avg_precision'],
            m_with['precision']     - m_no['precision'],
            m_with['recall']        - m_no['recall'],
            m_with['f1']            - m_no['f1'],
        ]
        row = f"{name:<{col_w[0]}}"
        for v, w in zip(vals, col_w[1:]):
            row += f" {v:+{w}.4f}"
        print(row)
    print(sep)

    # ── K-Fold CV summary (printed only when key is present) ─────────────────
    kf_keys = [
        "logistic_regression", "svc", "decision_tree",
        "random_forest", "xgboost",
    ]
    kf_display = {
        "logistic_regression": "LogisticRegression",
        "svc":                 "SVC (CV sample)",
        "decision_tree":       "DecisionTree",
        "random_forest":       "RandomForest",
        "xgboost":             "XGBoost",
    }
    kf_col_w = [22, 16, 16, 16, 16]
    kf_header = (
        f"{'Model':<{kf_col_w[0]}} {'ROC-AUC':>{kf_col_w[1]}} "
        f"{'F1':>{kf_col_w[2]}} {'Precision':>{kf_col_w[3]}} "
        f"{'Recall':>{kf_col_w[4]}}"
    )
    kf_sep = "-" * sum(kf_col_w + [4])

    for kf_key, kf_label in [
        ("kfold_no_leakage",      "no_leakage feature set"),
        ("kfold_strict",          "strict feature set"),
        ("kfold_pre_transaction", "pre_transaction feature set"),
    ]:
        kf = all_metrics.get(kf_key)
        if not kf:
            continue
        print(f"\n{'='*len(kf_sep)}")
        print(f"  5-Fold CV · {kf_label} · mean ± std")
        print(f"{'='*len(kf_sep)}")
        print(kf_header)
        print(kf_sep)
        for mk in kf_keys:
            d = kf.get(mk)
            if d is None:
                continue
            print(
                f"{kf_display[mk]:<{kf_col_w[0]}} "
                f"{d['roc_auc_mean']:.4f}±{d['roc_auc_std']:.4f}  "
                f"{d['f1_mean']:.4f}±{d['f1_std']:.4f}  "
                f"{d['precision_mean']:.4f}±{d['precision_std']:.4f}  "
                f"{d['recall_mean']:.4f}±{d['recall_std']:.4f}"
            )
        print(kf_sep)


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_sampled = load_data(DATASET_PATH)
    df_sampled, feature_cols_with, feature_cols_no, feature_cols_strict, feature_cols_pre_tx, le = engineer_features(df_sampled)

    # EDA snapshot from full CSV
    print("[INFO] Building EDA snapshot from full dataset ...")
    df_full = pd.read_csv(DATASET_PATH)
    save_eda_snapshot(df_full)
    del df_full

    # Single shared 80/20 stratified split on the FULL feature set
    X_all = df_sampled[feature_cols_with].values
    y_all = df_sampled["isFraud"].values
    _, X_test_full, _, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    # col_index maps feature name -> column position in X_test_full
    col_index = {c: i for i, c in enumerate(feature_cols_with)}

    # ── with_leakage ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TRAINING: with_leakage feature set")
    print("="*60)
    rf_wl, xgb_wl, results_wl = train_feature_set(
        df_sampled, feature_cols_with, "with_leakage",
        X_test_full, y_test, col_index,
    )

    # ── no_leakage ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TRAINING: no_leakage feature set")
    print("="*60)
    rf_nl, xgb_nl, results_nl = train_feature_set(
        df_sampled, feature_cols_no, "no_leakage",
        X_test_full, y_test, col_index,
    )

    # ── strict ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TRAINING: strict feature set  (step, type_enc, amount)")
    print("="*60)
    rf_st, xgb_st, results_st = train_feature_set(
        df_sampled, feature_cols_strict, "strict",
        X_test_full, y_test, col_index,
    )

    # ── pre_transaction ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TRAINING: pre_transaction feature set  (step, type_enc, amount, oldbalanceOrg, oldbalanceDest)")
    print("="*60)
    rf_pt, xgb_pt, results_pt = train_feature_set(
        df_sampled, feature_cols_pre_tx, "pre_transaction",
        X_test_full, y_test, col_index,
    )

    # ── Build final metrics.json ──────────────────────────────────────────────
    # Preserve existing keys (GNN, kfold) written by other scripts
    existing_extras = {}
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            existing = json.load(f)
        for key in ("gnn", "kfold_no_leakage", "kfold_strict", "kfold_pre_transaction"):
            if key in existing:
                existing_extras[key] = existing[key]

    leakage_features = LEAKAGE_FEATURES
    fi_with = results_wl.get("feature_importances", {})
    all_metrics = {
        "with_leakage":    results_wl,
        "no_leakage":      results_nl,
        "strict":          results_st,
        "pre_transaction": results_pt,
        "feature_importances": fi_with,  # kept at top level for SHAP page backwards compat
        "leakage_note": {
            "affected_features": leakage_features,
            "explanation": (
                "These features are derived from post-transaction balance states "
                "(newbalanceOrig, oldbalanceDest) which are not available before a "
                "transaction is settled. In a real-time deployment they must be "
                "removed or estimated. The reported with_leakage ROC-AUC / F1 metrics "
                "are an upper bound — use no_leakage metrics for realistic deployment estimates."
            ),
            "leaky_fi_share": round(
                sum(fi_with.get(f, 0) for f in leakage_features), 4
            ),
        },
    }
    all_metrics.update(existing_extras)

    # Also expose the with_leakage RF/XGB/ensemble at top level for
    # backwards-compatibility with dashboard's 📊 Model Metrics page
    for key in ("random_forest", "xgboost", "ensemble"):
        all_metrics[key] = results_wl[key]

    save_artifacts(rf_wl, xgb_wl, le, feature_cols_with, all_metrics)

    print_summary(all_metrics)

    print("\n[DONE] Training complete.")
    print(f"  with_leakage    Ensemble ROC-AUC : {results_wl['ensemble']['roc_auc']}")
    print(f"  with_leakage    Ensemble F1      : {results_wl['ensemble']['f1']}")
    print(f"  no_leakage      Ensemble ROC-AUC : {results_nl['ensemble']['roc_auc']}")
    print(f"  no_leakage      Ensemble F1      : {results_nl['ensemble']['f1']}")
    print(f"  strict          Ensemble ROC-AUC : {results_st['ensemble']['roc_auc']}")
    print(f"  strict          Ensemble F1      : {results_st['ensemble']['f1']}")
    print(f"  pre_transaction Ensemble ROC-AUC : {results_pt['ensemble']['roc_auc']}")
    print(f"  pre_transaction Ensemble F1      : {results_pt['ensemble']['f1']}")
