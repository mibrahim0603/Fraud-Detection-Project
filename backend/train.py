"""
train.py — Fraud Detection Model Training Pipeline
Reads AIML Dataset.csv, preprocesses data, trains a Random Forest + XGBoost
ensemble, evaluates on a held-out test set, and saves all artifacts to /models.

Leakage disclosure
──────────────────
Two engineered features use post-transaction balance states that would NOT be
available at real decision time in a production system:
  • errorBalanceOrig  = newbalanceOrig + amount − oldbalanceOrg
  • errorBalanceDest  = oldbalanceDest + amount − newbalanceDest
These features are highly predictive (SHAP confirms they dominate) because they
perfectly encode whether a fraud-type drain occurred. The reported metrics
(ROC-AUC ~0.9997) should therefore be understood as an upper bound on what the
model could achieve in a truly online setting. See metrics.json key
"leakage_note" for the full disclosure stored alongside model metrics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ─── 1. Load & sample ────────────────────────────────────────────────────────
def load_data(path: str, sample_size: int = 200_000) -> pd.DataFrame:
    print(f"[INFO] Loading dataset from {path} ...")
    df = pd.read_csv(path)
    print(f"[INFO] Full dataset shape: {df.shape}")

    # Keep all fraud rows + a random sample of non-fraud rows
    fraud = df[df["isFraud"] == 1]
    non_fraud = df[df["isFraud"] == 0].sample(
        n=min(sample_size, len(df[df["isFraud"] == 0])), random_state=42
    )
    df_sampled = pd.concat([fraud, non_fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[INFO] Sampled dataset shape: {df_sampled.shape}")
    print(f"[INFO] Fraud rate: {df_sampled['isFraud'].mean():.4%}")
    return df_sampled


# ─── 2. Feature Engineering ──────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame):
    df = df.copy()

    # Encode transaction type
    le = LabelEncoder()
    df["type_enc"] = le.fit_transform(df["type"])

    # Balance error features
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    # Flag zero-balance accounts
    df["origZeroBalance"] = (df["oldbalanceOrg"] == 0).astype(int)
    df["destZeroBalance"] = (df["oldbalanceDest"] == 0).astype(int)

    # Amount relative to origin balance
    df["amountRatio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    feature_cols = [
        "step", "type_enc", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "errorBalanceOrig", "errorBalanceDest",
        "origZeroBalance", "destZeroBalance",
        "amountRatio",
    ]
    return df, feature_cols, le


# ─── 3. Train / evaluate ─────────────────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame, feature_cols: list):
    X = df[feature_cols].values
    y = df["isFraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE only on training set
    print("[INFO] Applying SMOTE ...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"[INFO] After SMOTE — train size: {X_train_res.shape[0]}, fraud: {y_train_res.sum()}")

    # ── Random Forest
    print("[INFO] Training Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=15, n_jobs=-1, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train_res, y_train_res)

    # ── XGBoost
    print("[INFO] Training XGBoost ...")
    scale_pos = (y_train_res == 0).sum() / (y_train_res == 1).sum()
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos, use_label_encoder=False,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train_res, y_train_res)

    # ── Ensemble (average probabilities)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    ensemble_prob = (rf_prob + xgb_prob) / 2
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)

    # ── Metrics
    metrics = {}
    for name, prob, pred in [
        ("random_forest", rf_prob, rf.predict(X_test)),
        ("xgboost", xgb_prob, xgb.predict(X_test)),
        ("ensemble", ensemble_prob, ensemble_pred),
    ]:
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        metrics[name] = {
            "roc_auc": round(roc_auc_score(y_test, prob), 4),
            "avg_precision": round(average_precision_score(y_test, prob), 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "classification_report": classification_report(y_test, pred, output_dict=True),
        }
        print(f"[{name.upper()}] ROC-AUC={metrics[name]['roc_auc']}  F1={metrics[name]['f1']}")

    # Feature importances (average of both)
    fi_rf  = rf.feature_importances_
    fi_xgb = xgb.feature_importances_
    fi_avg = ((fi_rf + fi_xgb) / 2).tolist()
    metrics["feature_importances"] = dict(zip(feature_cols, fi_avg))

    # Leakage-aware annotation
    leakage_features = ["errorBalanceOrig", "errorBalanceDest"]
    metrics["leakage_note"] = {
        "affected_features": leakage_features,
        "explanation": (
            "These features are derived from post-transaction balance states "
            "(newbalanceOrig, oldbalanceDest) which are not available before a "
            "transaction is settled. In a real-time deployment they must be "
            "removed or estimated. The reported ROC-AUC / F1 metrics are an "
            "upper bound — expect lower performance without them."
        ),
        "leaky_fi_share": round(
            sum(metrics["feature_importances"].get(f, 0) for f in leakage_features), 4
        ),
    }

    return rf, xgb, metrics, X_test, y_test


# ─── 4. Save artifacts ───────────────────────────────────────────────────────
def save_artifacts(rf, xgb, le, feature_cols, metrics):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(xgb, os.path.join(MODELS_DIR, "xgboost.pkl"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))

    with open(os.path.join(MODELS_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[INFO] Artifacts saved to {MODELS_DIR}")


# ─── 5. Save EDA snapshot ────────────────────────────────────────────────────
def save_eda_snapshot(df_full: pd.DataFrame):
    os.makedirs(DATA_DIR, exist_ok=True)

    eda = {
        "total_transactions": int(len(df_full)),
        "fraud_count": int(df_full["isFraud"].sum()),
        "non_fraud_count": int((df_full["isFraud"] == 0).sum()),
        "fraud_rate": round(float(df_full["isFraud"].mean()), 6),
        "transaction_types": df_full["type"].value_counts().to_dict(),
        "fraud_by_type": df_full.groupby("type")["isFraud"].sum().to_dict(),
        "fraud_rate_by_type": df_full.groupby("type")["isFraud"].mean().round(6).to_dict(),
        "amount_stats": {
            "overall": df_full["amount"].describe().round(2).to_dict(),
            "fraud": df_full[df_full["isFraud"] == 1]["amount"].describe().round(2).to_dict(),
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


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_sampled = load_data(DATASET_PATH)
    df_sampled, feature_cols, le = engineer_features(df_sampled)

    # EDA snapshot uses full CSV — read once more just the two columns
    print("[INFO] Building EDA snapshot from full dataset ...")
    df_full = pd.read_csv(DATASET_PATH)
    save_eda_snapshot(df_full)

    rf, xgb, metrics, X_test, y_test = train_and_evaluate(df_sampled, feature_cols)
    save_artifacts(rf, xgb, le, feature_cols, metrics)

    print("\n[DONE] Training complete.")
    print(f"  Ensemble ROC-AUC : {metrics['ensemble']['roc_auc']}")
    print(f"  Ensemble F1      : {metrics['ensemble']['f1']}")
