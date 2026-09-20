"""
app.py — Fraud Detection REST API (Flask)
Endpoints:
  GET  /health                  — liveness probe
  GET  /api/metrics             — model evaluation metrics + leakage note
  GET  /api/eda                 — EDA snapshot
  POST /api/predict             — single transaction prediction
  POST /api/predict/batch       — batch transaction prediction (JSON array)
  GET  /api/feature_importance  — sorted feature importances
  POST /api/shap                — SHAP explanation for one transaction
  GET  /api/graph/rings         — top fraud rings (nodes + edges)
  GET  /api/graph/nodes         — all graph nodes with metadata
  GET  /api/gnn/scores          — per-node GNN fraud scores + metrics
"""

import os
import json
import sys
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Allow imports from backend/ when running as python backend/app.py
sys.path.insert(0, os.path.dirname(__file__))
from explain import explain_transaction

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")

app = Flask(__name__)
CORS(app)

# ─── Load artifacts once at startup ──────────────────────────────────────────
_rf = _xgb = _le = _feature_cols = _metrics = _eda = None
_graph_rings = _graph_nodes = _gnn_scores = None


def load_artifacts():
    global _rf, _xgb, _le, _feature_cols, _metrics, _eda
    global _graph_rings, _graph_nodes, _gnn_scores

    _rf  = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    _xgb = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))
    _le  = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    with open(os.path.join(MODELS_DIR, "feature_cols.json")) as f:
        _feature_cols = json.load(f)
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        _metrics = json.load(f)
    with open(os.path.join(DATA_DIR, "eda_snapshot.json")) as f:
        _eda = json.load(f)

    # Graph artifacts (optional — only if graph_builder.py was run)
    rings_path = os.path.join(DATA_DIR, "fraud_rings.json")
    nodes_path = os.path.join(DATA_DIR, "graph_nodes.json")
    gnn_path   = os.path.join(DATA_DIR, "gnn_node_scores.json")
    if os.path.exists(rings_path):
        with open(rings_path) as f:
            _graph_rings = json.load(f)
    if os.path.exists(nodes_path):
        with open(nodes_path) as f:
            _graph_nodes = json.load(f)
    if os.path.exists(gnn_path):
        with open(gnn_path) as f:
            _gnn_scores = json.load(f)


def build_features(tx: dict) -> np.ndarray:
    """Convert a raw transaction dict into the model feature vector."""
    type_enc = int(_le.transform([tx["type"].upper()])[0])
    amount      = float(tx["amount"])
    old_orig    = float(tx.get("oldbalanceOrg", 0))
    new_orig    = float(tx.get("newbalanceOrig", 0))
    old_dest    = float(tx.get("oldbalanceDest", 0))
    new_dest    = float(tx.get("newbalanceDest", 0))
    step        = int(tx.get("step", 1))

    err_orig    = new_orig + amount - old_orig
    err_dest    = old_dest + amount - new_dest
    orig_zero   = int(old_orig == 0)
    dest_zero   = int(old_dest == 0)
    amt_ratio   = amount / (old_orig + 1)

    return np.array([[
        step, type_enc, amount,
        old_orig, new_orig,
        old_dest, new_dest,
        err_orig, err_dest,
        orig_zero, dest_zero,
        amt_ratio,
    ]])


def predict_one(tx: dict) -> dict:
    x = build_features(tx)
    rf_prob  = float(_rf.predict_proba(x)[0, 1])
    xgb_prob = float(_xgb.predict_proba(x)[0, 1])
    ens_prob = (rf_prob + xgb_prob) / 2
    return {
        "fraud_probability":  round(ens_prob, 4),
        "is_fraud":           ens_prob >= 0.5,
        "risk_level":         "HIGH" if ens_prob >= 0.7 else "MEDIUM" if ens_prob >= 0.4 else "LOW",
        "rf_probability":     round(rf_prob, 4),
        "xgb_probability":    round(xgb_prob, 4),
    }


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": _rf is not None})


@app.route("/api/metrics")
def metrics():
    if _metrics is None:
        return jsonify({"error": "Models not loaded"}), 503
    return jsonify(_metrics)


@app.route("/api/eda")
def eda():
    if _eda is None:
        return jsonify({"error": "EDA data not available"}), 503
    return jsonify(_eda)


@app.route("/api/predict", methods=["POST"])
def predict():
    if _rf is None:
        return jsonify({"error": "Models not loaded"}), 503

    data = request.get_json(force=True)
    required = ["type", "amount"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    valid_types = list(_le.classes_)
    if data["type"].upper() not in valid_types:
        return jsonify({"error": f"Invalid type. Must be one of: {valid_types}"}), 400

    try:
        result = predict_one(data)
        result["input"] = data
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    if _rf is None:
        return jsonify({"error": "Models not loaded"}), 503

    data = request.get_json(force=True)
    if not isinstance(data, list):
        return jsonify({"error": "Expected a JSON array of transactions"}), 400

    results = []
    for i, tx in enumerate(data):
        try:
            res = predict_one(tx)
            res["index"] = i
            results.append(res)
        except Exception as e:
            results.append({"index": i, "error": str(e)})

    return jsonify({"count": len(results), "predictions": results})


@app.route("/api/feature_importance")
def feature_importance():
    if _metrics is None:
        return jsonify({"error": "Models not loaded"}), 503
    fi = _metrics.get("feature_importances", {})
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    return jsonify({"feature_importances": sorted_fi})


@app.route("/api/shap", methods=["POST"])
def shap_explain():
    if _xgb is None:
        return jsonify({"error": "Models not loaded"}), 503

    data = request.get_json(force=True)
    required = ["type", "amount"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    valid_types = list(_le.classes_)
    if data["type"].upper() not in valid_types:
        return jsonify({"error": f"Invalid type. Must be one of: {valid_types}"}), 400

    try:
        result = explain_transaction(data, _xgb, _le, _feature_cols)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph/rings")
def graph_rings():
    if _graph_rings is None:
        return jsonify({"error": "Graph not built. Run backend/graph_builder.py first."}), 503
    limit = int(request.args.get("limit", 10))
    return jsonify({"rings": _graph_rings[:limit], "total": len(_graph_rings)})


@app.route("/api/graph/nodes")
def graph_nodes():
    if _graph_nodes is None:
        return jsonify({"error": "Graph not built. Run backend/graph_builder.py first."}), 503
    return jsonify({"nodes": _graph_nodes, "count": len(_graph_nodes)})


@app.route("/api/gnn/scores")
def gnn_scores():
    if _gnn_scores is None:
        return jsonify({"error": "GNN not trained. Run backend/train_gnn.py first."}), 503
    return jsonify(_gnn_scores)


# ─── Bootstrap ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_artifacts()
    print("[INFO] Fraud Detection API running on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
