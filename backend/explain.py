"""
explain.py — SHAP Explanation Wrapper
Computes SHAP values for a single transaction using the XGBoost model
(TreeExplainer is fast and exact for tree-based models).

Exported functions:
  explain_transaction(tx_dict, xgb_model, le, feature_cols)
      → dict with shap_values (list), base_value, feature_names, prediction

  explain_batch(records, xgb_model, le, feature_cols)
      → list of explanation dicts

  get_global_shap(X_sample, xgb_model, feature_cols)
      → mean absolute SHAP values per feature (global importance)

Leakage note:
  The features errorBalanceOrig and errorBalanceDest are derived from
  balance columns that are themselves post-transaction states — in a real
  deployment those would not be available at decision time.  SHAP values
  on these features should be interpreted with that caveat in mind.
"""

import numpy as np
import shap


# ─── Feature vector builder (mirrors train.py / dashboard.py) ─────────────────
def _build_feature_vector(tx: dict, le) -> np.ndarray:
    type_enc  = int(le.transform([tx["type"].upper()])[0])
    amount    = float(tx["amount"])
    old_orig  = float(tx.get("oldbalanceOrg",  0))
    new_orig  = float(tx.get("newbalanceOrig", 0))
    old_dest  = float(tx.get("oldbalanceDest", 0))
    new_dest  = float(tx.get("newbalanceDest", 0))
    step      = int(tx.get("step",  1))
    err_orig  = new_orig + amount - old_orig
    err_dest  = old_dest + amount - new_dest
    orig_zero = int(old_orig == 0)
    dest_zero = int(old_dest == 0)
    amt_ratio = amount / (old_orig + 1)
    return np.array([[
        step, type_enc, amount,
        old_orig, new_orig, old_dest, new_dest,
        err_orig, err_dest, orig_zero, dest_zero, amt_ratio,
    ]], dtype=np.float32)


# ─── Cached explainer factory (call once per session) ─────────────────────────
_explainer_cache = {}

def _get_explainer(xgb_model):
    key = id(xgb_model)
    if key not in _explainer_cache:
        _explainer_cache[key] = shap.TreeExplainer(xgb_model)
    return _explainer_cache[key]


# ─── Single transaction explanation ───────────────────────────────────────────
def explain_transaction(tx: dict, xgb_model, le, feature_cols: list) -> dict:
    """
    Returns:
      {
        "feature_names":  [...],
        "feature_values": [...],
        "shap_values":    [...],   # per-feature SHAP for fraud class
        "base_value":     float,   # E[f(x)] — model's average output
        "prediction":     float,   # XGBoost probability
        "leakage_warning": str,
      }
    """
    x = _build_feature_vector(tx, le)
    explainer = _get_explainer(xgb_model)

    shap_vals = explainer.shap_values(x)      # (1, n_features) for XGB binary
    base_val  = float(explainer.expected_value)
    pred_prob = float(xgb_model.predict_proba(x)[0, 1])

    # shap_vals may be a list (multi-class) or array (binary)
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]   # fraud class
    else:
        sv = shap_vals[0]

    leakage_features = {"errorBalanceOrig", "errorBalanceDest"}
    leakage_present  = any(f in leakage_features for f in feature_cols)

    return {
        "feature_names":  feature_cols,
        "feature_values": [round(float(v), 4) for v in x[0]],
        "shap_values":    [round(float(v), 6) for v in sv],
        "base_value":     round(base_val, 6),
        "prediction":     round(pred_prob, 4),
        "leakage_warning": (
            "⚠️ Features 'errorBalanceOrig' and 'errorBalanceDest' use post-transaction "
            "balance states, which are not available at real decision time. "
            "Their SHAP values reflect in-sample signal, not deployable signal."
        ) if leakage_present else None,
    }


# ─── Batch explanation ────────────────────────────────────────────────────────
def explain_batch(records: list, xgb_model, le, feature_cols: list) -> list:
    results = []
    for tx in records:
        try:
            results.append(explain_transaction(tx, xgb_model, le, feature_cols))
        except Exception as e:
            results.append({"error": str(e)})
    return results


# ─── Global SHAP (mean |SHAP| over a background sample) ──────────────────────
def get_global_shap(X_sample: np.ndarray, xgb_model, feature_cols: list) -> dict:
    """
    X_sample : (n, n_features) numpy array
    Returns dict mapping feature_name -> mean |SHAP value|
    """
    explainer = _get_explainer(xgb_model)
    shap_vals = explainer.shap_values(X_sample)

    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    else:
        sv = shap_vals

    mean_abs = np.abs(sv).mean(axis=0)
    return {
        "feature_names":   feature_cols,
        "mean_abs_shap":   [round(float(v), 6) for v in mean_abs],
        "leakage_warning": (
            "⚠️ 'errorBalanceOrig' and 'errorBalanceDest' use post-transaction balance states."
        ),
    }
