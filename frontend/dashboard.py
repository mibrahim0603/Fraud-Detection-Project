"""
dashboard.py — Fraud Detection Streamlit Frontend (Standalone)
Loads models directly from disk — no Flask API required.
Deploy on Streamlit Community Cloud as-is.

Pages:
  1. 🏠 Overview        — KPIs, class distribution, EDA charts
  2. 📊 Model Metrics   — ROC-AUC, F1, confusion matrices, feature importance
  3. 🔍 Live Prediction — Single + batch transaction inference (in-process)
  4. 📖 About           — Dataset & methodology
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─── Paths (work both locally and on Streamlit Cloud) ────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.abspath(os.path.join(_HERE, ".."))
MODELS_DIR  = os.path.join(_ROOT, "models")
DATA_DIR    = os.path.join(_ROOT, "data")

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding-bottom: 0.3rem;
    }
    .risk-high   { color: #e74c3c; font-weight: 700; font-size: 1.1rem; }
    .risk-medium { color: #f39c12; font-weight: 700; font-size: 1.1rem; }
    .risk-low    { color: #27ae60; font-weight: 700; font-size: 1.1rem; }
    .stButton>button { border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─── Load all artifacts once (cached for the session) ────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    rf  = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    xgb = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))
    le  = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    with open(os.path.join(MODELS_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    return rf, xgb, le, feature_cols


@st.cache_data(show_spinner=False)
def load_metrics():
    path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_eda():
    path = os.path.join(DATA_DIR, "eda_snapshot.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def build_features(tx: dict, le) -> np.ndarray:
    type_enc  = int(le.transform([tx["type"].upper()])[0])
    amount    = float(tx["amount"])
    old_orig  = float(tx.get("oldbalanceOrg", 0))
    new_orig  = float(tx.get("newbalanceOrig", 0))
    old_dest  = float(tx.get("oldbalanceDest", 0))
    new_dest  = float(tx.get("newbalanceDest", 0))
    step      = int(tx.get("step", 1))
    err_orig  = new_orig + amount - old_orig
    err_dest  = old_dest + amount - new_dest
    orig_zero = int(old_orig == 0)
    dest_zero = int(old_dest == 0)
    amt_ratio = amount / (old_orig + 1)
    return np.array([[
        step, type_enc, amount,
        old_orig, new_orig, old_dest, new_dest,
        err_orig, err_dest, orig_zero, dest_zero, amt_ratio,
    ]])


def predict_one(tx: dict, rf, xgb, le) -> dict:
    x        = build_features(tx, le)
    rf_prob  = float(rf.predict_proba(x)[0, 1])
    xgb_prob = float(xgb.predict_proba(x)[0, 1])
    ens_prob = (rf_prob + xgb_prob) / 2
    return {
        "fraud_probability": round(ens_prob, 4),
        "is_fraud":          ens_prob >= 0.5,
        "risk_level":        "HIGH" if ens_prob >= 0.7 else "MEDIUM" if ens_prob >= 0.4 else "LOW",
        "rf_probability":    round(rf_prob, 4),
        "xgb_probability":   round(xgb_prob, 4),
    }


# ─── Load everything ─────────────────────────────────────────────────────────
try:
    rf, xgb, le, feature_cols = load_models()
    models_ok = True
except Exception as e:
    models_ok = False
    model_error = str(e)

metrics = load_metrics()
eda     = load_eda()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<h1 style="text-align:center;font-size:2rem;">🔐</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align:center;font-size:1.05rem;">Fraud Detection</h2>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Model Metrics", "🔍 Live Prediction", "📖 About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    if models_ok:
        st.success("✅ Models loaded")
    else:
        st.error("❌ Models not found")

    st.caption("AIML Dataset · 6.3M transactions")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="main-header">🔐 Fraud Detection Dashboard</p>', unsafe_allow_html=True)
    st.caption("Real-time financial fraud detection powered by Random Forest + XGBoost ensemble")
    st.markdown("---")

    if not eda:
        st.error("EDA data not found. Make sure `data/eda_snapshot.json` is committed to your repo.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{eda['total_transactions']:,}")
    col2.metric("Fraudulent",         f"{eda['fraud_count']:,}")
    col3.metric("Fraud Rate",         f"{eda['fraud_rate']*100:.4f}%")
    col4.metric("Legitimate",         f"{eda['non_fraud_count']:,}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Transaction Type Distribution")
        tx_types = eda.get("transaction_types", {})
        fig = px.pie(
            names=list(tx_types.keys()), values=list(tx_types.values()),
            hole=0.45, color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(margin=dict(t=20, b=20), height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Fraud Count by Transaction Type")
        fbt = eda.get("fraud_by_type", {})
        fig = px.bar(
            x=list(fbt.keys()), y=list(fbt.values()),
            color=list(fbt.values()), color_continuous_scale="Reds",
            labels={"x": "Type", "y": "Fraud Count"},
        )
        fig.update_layout(coloraxis_showscale=False, height=320, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.subheader("Fraud Rate by Type (%)")
        frbt = eda.get("fraud_rate_by_type", {})
        fig = px.bar(
            x=list(frbt.keys()), y=[v * 100 for v in frbt.values()],
            color=[v * 100 for v in frbt.values()], color_continuous_scale="OrRd",
            labels={"x": "Type", "y": "Fraud Rate (%)"},
        )
        fig.update_layout(coloraxis_showscale=False, height=280, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        st.subheader("Hourly Fraud Pattern")
        hourly = eda.get("hourly_fraud", {})
        hours  = list(range(24))
        counts = [hourly.get(str(h), 0) for h in hours]
        fig = px.line(x=hours, y=counts, markers=True,
                      labels={"x": "Hour of Day", "y": "Fraud Count"})
        fig.update_traces(line_color="#e74c3c", marker_color="#e74c3c")
        fig.update_layout(height=280, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Amount Statistics: Fraud vs Legitimate")
    amt = eda.get("amount_stats", {})
    if amt:
        fs  = amt.get("fraud", {})
        ns  = amt.get("non_fraud", {})
        keys   = ["mean", "50%", "75%", "max"]
        labels = ["Mean", "Median", "75th Pct", "Max"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Fraud",      x=labels, y=[fs.get(k, 0) for k in keys], marker_color="#e74c3c"))
        fig.add_trace(go.Bar(name="Legitimate", x=labels, y=[ns.get(k, 0) for k in keys], marker_color="#3498db"))
        fig.update_layout(barmode="group", height=300, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL METRICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Metrics":
    st.markdown('<p class="main-header">📊 Model Performance Metrics</p>', unsafe_allow_html=True)
    st.caption("Evaluation on 20% held-out test set. SMOTE applied only on training data.")
    st.markdown("---")

    if not metrics:
        st.error("Metrics not found. Make sure `models/metrics.json` is committed to your repo.")
        st.stop()

    model_names   = ["random_forest", "xgboost", "ensemble"]
    display_names = {"random_forest": "Random Forest", "xgboost": "XGBoost", "ensemble": "Ensemble (Avg)"}

    st.subheader("Model Comparison")
    rows = []
    for m in model_names:
        d = metrics.get(m, {})
        rows.append({
            "Model":         display_names[m],
            "ROC-AUC":       d.get("roc_auc", "-"),
            "Avg Precision": d.get("avg_precision", "-"),
            "Precision":     d.get("precision", "-"),
            "Recall":        d.get("recall", "-"),
            "F1 Score":      d.get("f1", "-"),
        })
    df_s = pd.DataFrame(rows).set_index("Model")
    st.dataframe(df_s.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

    st.subheader("Metric Comparison Chart")
    mk = ["roc_auc", "avg_precision", "precision", "recall", "f1"]
    ml = ["ROC-AUC", "Avg Precision", "Precision", "Recall", "F1"]
    fig = go.Figure()
    for i, m in enumerate(model_names):
        d = metrics.get(m, {})
        fig.add_trace(go.Bar(
            name=display_names[m], x=ml,
            y=[d.get(k, 0) for k in mk],
            marker_color=["#3498db", "#e67e22", "#e74c3c"][i],
        ))
    fig.update_layout(barmode="group", yaxis_range=[0.95, 1.0], height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confusion Matrices")
    cm_cols = st.columns(3)
    for i, m in enumerate(model_names):
        cm = metrics.get(m, {}).get("confusion_matrix", [[0, 0], [0, 0]])
        with cm_cols[i]:
            st.caption(display_names[m])
            fig = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                x=["Not Fraud", "Fraud"], y=["Not Fraud", "Fraud"],
                color_continuous_scale="Blues",
            )
            fig.update_layout(height=250, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance (Ensemble Average)")
    fi = metrics.get("feature_importances", {})
    if fi:
        fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        fig = px.bar(
            x=[v for _, v in fi_sorted], y=[k for k, _ in fi_sorted],
            orientation="h",
            labels={"x": "Importance", "y": "Feature"},
            color=[v for _, v in fi_sorted], color_continuous_scale="Blues",
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False, height=400, margin=dict(t=10),
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Live Prediction":
    st.markdown('<p class="main-header">🔍 Live Transaction Prediction</p>', unsafe_allow_html=True)
    st.caption("Predictions run directly in-browser via the loaded ensemble model.")
    st.markdown("---")

    if not models_ok:
        st.error(f"Models could not be loaded: {model_error}")
        st.stop()

    tab_single, tab_batch = st.tabs(["Single Transaction", "Batch Prediction"])

    # ── Single ────────────────────────────────────────────────────────────────
    with tab_single:
        st.subheader("Enter Transaction Details")

        # Scenario defaults
        defaults = {
            "type": "CASH_OUT", "amount": 1000.0, "step": 1,
            "old_orig": 5000.0, "new_orig": 4000.0,
            "old_dest": 0.0,    "new_dest": 1000.0,
        }

        c1, c2 = st.columns(2)
        with c1:
            tx_type  = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"],
                                    index=["CASH_OUT","TRANSFER","PAYMENT","CASH_IN","DEBIT"].index(defaults["type"]))
            amount   = st.number_input("Amount ($)",            min_value=0.01, value=defaults["amount"],   step=100.0)
            step     = st.number_input("Step (hour)",           min_value=1,    value=defaults["step"],     max_value=744)
        with c2:
            old_orig = st.number_input("Origin Old Balance ($)",      min_value=0.0, value=defaults["old_orig"], step=100.0)
            new_orig = st.number_input("Origin New Balance ($)",      min_value=0.0, value=defaults["new_orig"], step=100.0)
            old_dest = st.number_input("Destination Old Balance ($)", min_value=0.0, value=defaults["old_dest"], step=100.0)
            new_dest = st.number_input("Destination New Balance ($)", min_value=0.0, value=defaults["new_dest"], step=100.0)

        st.markdown("**Quick Scenarios** — paste typical transactions:")
        s1, s2, s3 = st.columns(3)
        with s1:
            if st.button("🚨 Suspicious Transfer"):
                st.info("Tip: Set Type=TRANSFER, Amount=181000, OldOrigBal=181000, NewOrigBal=0, OldDestBal=0, NewDestBal=0")
        with s2:
            if st.button("✅ Normal Payment"):
                st.info("Tip: Set Type=PAYMENT, Amount=500, OldOrigBal=20000, NewOrigBal=19500")
        with s3:
            if st.button("⚠️ High-Risk Cash Out"):
                st.info("Tip: Set Type=CASH_OUT, Amount=500000, OldOrigBal=500000, NewOrigBal=0")

        if st.button("🔍 Predict", use_container_width=True, type="primary"):
            payload = {
                "type": tx_type, "amount": amount, "step": step,
                "oldbalanceOrg": old_orig, "newbalanceOrig": new_orig,
                "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
            }
            result = predict_one(payload, rf, xgb, le)

            st.markdown("---")
            prob  = result["fraud_probability"]
            risk  = result["risk_level"]
            rc    = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}[risk]

            r1, r2, r3 = st.columns(3)
            r1.metric("Fraud Probability", f"{prob*100:.2f}%")
            r2.metric("Decision", "🚨 FRAUD" if result["is_fraud"] else "✅ LEGITIMATE")
            with r3:
                st.markdown(f"**Risk Level:** <span class='{rc}'>{risk}</span>", unsafe_allow_html=True)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(prob * 100, 2),
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Fraud Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#e74c3c" if prob >= 0.5 else "#27ae60"},
                    "steps": [
                        {"range": [0, 40],   "color": "#d4edda"},
                        {"range": [40, 70],  "color": "#fff3cd"},
                        {"range": [70, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50},
                },
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            with st.expander("Model breakdown"):
                mc1, mc2 = st.columns(2)
                mc1.metric("Random Forest Prob", f"{result['rf_probability']*100:.2f}%")
                mc2.metric("XGBoost Prob",       f"{result['xgb_probability']*100:.2f}%")

    # ── Batch ─────────────────────────────────────────────────────────────────
    with tab_batch:
        st.subheader("Batch Transaction Prediction")
        st.markdown("Upload a CSV with columns: `type, amount, step, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest`")

        sample_csv = (
            "type,amount,step,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest\n"
            "CASH_OUT,181000,1,181000,0,21182,0\n"
            "PAYMENT,9839.64,1,170136,160296.36,0,0\n"
            "TRANSFER,10000,5,50000,40000,0,10000\n"
        )
        st.download_button("📥 Download Sample CSV", sample_csv,
                           file_name="sample_fraud_input.csv", mime="text/csv")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df_input = pd.read_csv(uploaded)
            st.dataframe(df_input, use_container_width=True)

            if st.button("🔍 Run Batch Prediction", type="primary"):
                results = []
                with st.spinner(f"Predicting {len(df_input)} transactions…"):
                    for _, row in df_input.iterrows():
                        try:
                            results.append(predict_one(row.to_dict(), rf, xgb, le))
                        except Exception as e:
                            results.append({"fraud_probability": None, "is_fraud": None,
                                            "risk_level": "ERROR", "error": str(e)})

                df_result = df_input.copy()
                df_result["fraud_probability"] = [r.get("fraud_probability") for r in results]
                df_result["is_fraud"]          = [r.get("is_fraud")          for r in results]
                df_result["risk_level"]        = [r.get("risk_level")        for r in results]

                fraud_count = df_result["is_fraud"].sum()
                b1, b2, b3 = st.columns(3)
                b1.metric("Total",            len(df_result))
                b2.metric("Flagged as Fraud", int(fraud_count))
                b3.metric("Fraud Rate",       f"{fraud_count/len(df_result)*100:.1f}%")

                st.dataframe(
                    df_result.style.apply(
                        lambda row: ["background-color:#f8d7da" if row["is_fraud"] else "" for _ in row],
                        axis=1,
                    ),
                    use_container_width=True,
                )
                st.download_button("⬇️ Download Results", df_result.to_csv(index=False),
                                   file_name="fraud_predictions.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📖 About":
    st.markdown('<p class="main-header">📖 About This Project</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
## Dataset
**AIML Dataset** — synthetic financial transaction log with **6,362,620 transactions** across 30 days (744 steps).

| Feature | Description |
|---|---|
| `step` | Hour of simulation (1–744) |
| `type` | CASH_OUT · PAYMENT · TRANSFER · CASH_IN · DEBIT |
| `amount` | Transaction amount (USD) |
| `oldbalanceOrg / newbalanceOrig` | Origin account balance before/after |
| `oldbalanceDest / newbalanceDest` | Destination account balance before/after |
| `isFraud` | **Target label** (1 = fraud) |

---

## Methodology

### Class Imbalance
Only **0.129%** of transactions are fraudulent. **SMOTE** is applied to the training split only.

### Feature Engineering
| Feature | Formula |
|---|---|
| `errorBalanceOrig` | `newOrig + amount − oldOrig` |
| `errorBalanceDest` | `oldDest + amount − newDest` |
| `origZeroBalance` | `1 if oldOrig == 0` |
| `destZeroBalance` | `1 if oldDest == 0` |
| `amountRatio` | `amount / (oldOrig + 1)` |

### Models
| Model | ROC-AUC | F1 |
|---|---|---|
| Random Forest | 0.9997 | 0.9976 |
| XGBoost | 0.9998 | 0.9963 |
| **Ensemble** | **0.9997** | **0.9970** |

---

## Tech Stack
Python · scikit-learn · XGBoost · imbalanced-learn · Streamlit · Plotly · pandas · numpy · joblib
""")
