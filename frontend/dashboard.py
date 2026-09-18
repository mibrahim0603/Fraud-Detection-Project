"""
dashboard.py — Fraud Detection Streamlit Frontend
Pages:
  1. 🏠 Overview        — KPIs, class distribution, EDA charts
  2. 📊 Model Metrics   — ROC-AUC, F1, confusion matrices, feature importance
  3. 🔍 Live Prediction — Single + batch transaction inference via Flask API
  4. 📖 About           — Dataset & methodology
"""

import json
import os
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ─── Config ──────────────────────────────────────────────────────────────────
API_BASE    = "http://localhost:5000"
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")

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
    .metric-card {
        background: #f7f8fa; border-radius: 10px;
        padding: 1rem 1.2rem; border-left: 4px solid #e74c3c;
    }
    .risk-high   { color: #e74c3c; font-weight: 700; font-size: 1.1rem; }
    .risk-medium { color: #f39c12; font-weight: 700; font-size: 1.1rem; }
    .risk-low    { color: #27ae60; font-weight: 700; font-size: 1.1rem; }
    .stButton>button { border-radius: 8px; font-weight: 600; }
    div[data-testid="stSidebar"] { background: #1a1a2e; }
    div[data-testid="stSidebar"] .css-1d391kg { color: white; }
    .sidebar-logo { text-align:center; padding:1rem 0; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_eda():
    path = os.path.join(DATA_DIR, "eda_snapshot.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    try:
        r = requests.get(f"{API_BASE}/api/eda", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


@st.cache_data(ttl=300)
def load_metrics():
    path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    try:
        r = requests.get(f"{API_BASE}/api/metrics", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def api_predict(payload: dict):
    try:
        r = requests.post(f"{API_BASE}/api/predict", json=payload, timeout=10)
        return r.json(), r.status_code
    except requests.ConnectionError:
        return {"error": "Cannot connect to API. Make sure the Flask server is running on port 5000."}, 503


def api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json()
    except Exception:
        return {"status": "unreachable"}


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo"><h1 style="color:white;font-size:1.8rem;">🔐</h1></div>', unsafe_allow_html=True)
    st.markdown('<h2 style="color:white;text-align:center;font-size:1.1rem;">Fraud Detection</h2>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Model Metrics", "🔍 Live Prediction", "📖 About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<p style="color:#aaa;font-size:0.8rem;text-align:center;">AIML Dataset · 6.3M transactions</p>', unsafe_allow_html=True)

    # API health indicator
    health = api_health()
    status_color = "#27ae60" if health.get("status") == "ok" else "#e74c3c"
    status_text  = "API Online" if health.get("status") == "ok" else "API Offline"
    st.markdown(
        f'<div style="text-align:center;margin-top:0.5rem;">'
        f'<span style="background:{status_color};color:white;padding:3px 12px;border-radius:12px;font-size:0.75rem;">{status_text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="main-header">🔐 Fraud Detection Dashboard</p>', unsafe_allow_html=True)
    st.caption("Real-time financial fraud detection powered by Random Forest + XGBoost ensemble")
    st.markdown("---")

    eda = load_eda()
    if not eda:
        st.error("EDA data not available. Run `python backend/train.py` first.")
        st.stop()

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{eda['total_transactions']:,}")
    with col2:
        st.metric("Fraudulent", f"{eda['fraud_count']:,}", delta=None)
    with col3:
        st.metric("Fraud Rate", f"{eda['fraud_rate']*100:.4f}%")
    with col4:
        st.metric("Legitimate", f"{eda['non_fraud_count']:,}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Transaction Type Distribution")
        tx_types = eda.get("transaction_types", {})
        fig_pie = px.pie(
            names=list(tx_types.keys()),
            values=list(tx_types.values()),
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie.update_layout(margin=dict(t=20, b=20), showlegend=True, height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Fraud Count by Transaction Type")
        fraud_by_type = eda.get("fraud_by_type", {})
        fig_bar = px.bar(
            x=list(fraud_by_type.keys()),
            y=list(fraud_by_type.values()),
            color=list(fraud_by_type.values()),
            color_continuous_scale="Reds",
            labels={"x": "Type", "y": "Fraud Count"},
        )
        fig_bar.update_layout(coloraxis_showscale=False, height=320, margin=dict(t=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("Fraud Rate by Type")
        fr_by_type = eda.get("fraud_rate_by_type", {})
        fig_rate = px.bar(
            x=list(fr_by_type.keys()),
            y=[v * 100 for v in fr_by_type.values()],
            labels={"x": "Type", "y": "Fraud Rate (%)"},
            color=[v * 100 for v in fr_by_type.values()],
            color_continuous_scale="OrRd",
        )
        fig_rate.update_layout(coloraxis_showscale=False, height=280, margin=dict(t=20))
        st.plotly_chart(fig_rate, use_container_width=True)

    with col_right2:
        st.subheader("Hourly Fraud Pattern (Hour of Day)")
        hourly = eda.get("hourly_fraud", {})
        hours  = list(range(24))
        counts = [hourly.get(str(h), 0) for h in hours]
        fig_line = px.line(
            x=hours, y=counts,
            labels={"x": "Hour of Day", "y": "Fraud Count"},
            markers=True,
        )
        fig_line.update_traces(line_color="#e74c3c", marker_color="#e74c3c")
        fig_line.update_layout(height=280, margin=dict(t=20))
        st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Amount Statistics: Fraud vs Legitimate")
    amt_stats = eda.get("amount_stats", {})
    if amt_stats:
        fraud_stats  = amt_stats.get("fraud", {})
        normal_stats = amt_stats.get("non_fraud", {})
        stat_keys    = ["mean", "50%", "75%", "max"]
        labels       = ["Mean", "Median", "75th Pct", "Max"]

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Fraud", x=labels,
            y=[fraud_stats.get(k, 0) for k in stat_keys],
            marker_color="#e74c3c",
        ))
        fig_comp.add_trace(go.Bar(
            name="Legitimate", x=labels,
            y=[normal_stats.get(k, 0) for k in stat_keys],
            marker_color="#3498db",
        ))
        fig_comp.update_layout(barmode="group", height=300, margin=dict(t=10))
        st.plotly_chart(fig_comp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL METRICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Metrics":
    st.markdown('<p class="main-header">📊 Model Performance Metrics</p>', unsafe_allow_html=True)
    st.caption("Evaluation on 20% held-out test set. SMOTE applied only on training data.")
    st.markdown("---")

    metrics = load_metrics()
    if not metrics:
        st.error("Metrics not available. Run `python backend/train.py` first.")
        st.stop()

    # Summary comparison table
    st.subheader("Model Comparison")
    model_names = ["random_forest", "xgboost", "ensemble"]
    display_names = {"random_forest": "Random Forest", "xgboost": "XGBoost", "ensemble": "Ensemble (Avg)"}
    summary_data = []
    for m in model_names:
        d = metrics.get(m, {})
        summary_data.append({
            "Model":         display_names[m],
            "ROC-AUC":       d.get("roc_auc", "-"),
            "Avg Precision": d.get("avg_precision", "-"),
            "Precision":     d.get("precision", "-"),
            "Recall":        d.get("recall", "-"),
            "F1 Score":      d.get("f1", "-"),
        })
    df_summary = pd.DataFrame(summary_data).set_index("Model")
    st.dataframe(df_summary.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

    # Metric bar charts
    st.subheader("Metric Comparison Chart")
    metric_keys   = ["roc_auc", "avg_precision", "precision", "recall", "f1"]
    metric_labels = ["ROC-AUC", "Avg Precision", "Precision", "Recall", "F1"]
    fig_metrics   = go.Figure()
    colors = ["#3498db", "#e67e22", "#e74c3c"]
    for i, m in enumerate(model_names):
        d = metrics.get(m, {})
        fig_metrics.add_trace(go.Bar(
            name=display_names[m],
            x=metric_labels,
            y=[d.get(k, 0) for k in metric_keys],
            marker_color=colors[i],
        ))
    fig_metrics.update_layout(barmode="group", yaxis_range=[0.95, 1.0], height=350)
    st.plotly_chart(fig_metrics, use_container_width=True)

    # Confusion matrices
    st.subheader("Confusion Matrices")
    cm_cols = st.columns(3)
    for i, m in enumerate(model_names):
        cm = metrics.get(m, {}).get("confusion_matrix", [[0, 0], [0, 0]])
        with cm_cols[i]:
            st.caption(display_names[m])
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                x=["Not Fraud", "Fraud"],
                y=["Not Fraud", "Fraud"],
                color_continuous_scale="Blues",
            )
            fig_cm.update_layout(height=250, margin=dict(t=10, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

    # Feature Importance
    st.subheader("Feature Importance (Ensemble Average)")
    fi = metrics.get("feature_importances", {})
    if fi:
        fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        fig_fi = px.bar(
            x=[v for _, v in fi_sorted],
            y=[k for k, _ in fi_sorted],
            orientation="h",
            labels={"x": "Importance", "y": "Feature"},
            color=[v for _, v in fi_sorted],
            color_continuous_scale="Blues",
        )
        fig_fi.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False, height=400, margin=dict(t=10),
        )
        st.plotly_chart(fig_fi, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Live Prediction":
    st.markdown('<p class="main-header">🔍 Live Transaction Prediction</p>', unsafe_allow_html=True)
    st.caption("Submit a transaction to the Flask API and receive real-time fraud probability.")
    st.markdown("---")

    tab_single, tab_batch = st.tabs(["Single Transaction", "Batch Prediction"])

    # ── Single
    with tab_single:
        st.subheader("Enter Transaction Details")

        col1, col2 = st.columns(2)
        with col1:
            tx_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"])
            amount  = st.number_input("Amount ($)", min_value=0.01, value=1000.00, step=100.0)
            step    = st.number_input("Step (hour)", min_value=1, max_value=744, value=1)
        with col2:
            old_orig = st.number_input("Origin Old Balance ($)", min_value=0.0, value=5000.0, step=100.0)
            new_orig = st.number_input("Origin New Balance ($)", min_value=0.0, value=4000.0, step=100.0)
            old_dest = st.number_input("Destination Old Balance ($)", min_value=0.0, value=0.0, step=100.0)
            new_dest = st.number_input("Destination New Balance ($)", min_value=0.0, value=1000.0, step=100.0)

        # Pre-fill suspicious scenario
        st.markdown("**Quick Scenarios:**")
        scenario_col1, scenario_col2, scenario_col3 = st.columns(3)
        load_scenario = None
        with scenario_col1:
            if st.button("🚨 Suspicious Transfer"):
                load_scenario = "suspicious"
        with scenario_col2:
            if st.button("✅ Normal Payment"):
                load_scenario = "normal"
        with scenario_col3:
            if st.button("⚠️ High-Risk Cash Out"):
                load_scenario = "highrisk"

        if load_scenario == "suspicious":
            st.info("Loaded: Suspicious TRANSFER — origin drained to 0, destination untouched (classic flag).")
        elif load_scenario == "normal":
            st.info("Loaded: Normal small PAYMENT.")
        elif load_scenario == "highrisk":
            st.info("Loaded: High-value CASH_OUT from zero-balance account.")

        if st.button("🔍 Predict", use_container_width=True, type="primary"):
            payload = {
                "type":           tx_type,
                "amount":         amount,
                "step":           step,
                "oldbalanceOrg":  old_orig,
                "newbalanceOrig": new_orig,
                "oldbalanceDest": old_dest,
                "newbalanceDest": new_dest,
            }
            with st.spinner("Analysing transaction..."):
                result, status = api_predict(payload)
                time.sleep(0.3)

            if status != 200:
                st.error(f"API Error: {result.get('error', 'Unknown error')}")
            else:
                st.markdown("---")
                r1, r2, r3 = st.columns(3)
                prob = result["fraud_probability"]
                risk = result["risk_level"]
                risk_class = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}[risk]

                with r1:
                    st.metric("Fraud Probability", f"{prob*100:.2f}%")
                with r2:
                    st.metric("Ensemble Decision", "🚨 FRAUD" if result["is_fraud"] else "✅ LEGITIMATE")
                with r3:
                    st.markdown(f"**Risk Level:** <span class='{risk_class}'>{risk}</span>", unsafe_allow_html=True)

                # Gauge chart
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
                    with mc1:
                        st.metric("Random Forest Prob", f"{result['rf_probability']*100:.2f}%")
                    with mc2:
                        st.metric("XGBoost Prob", f"{result['xgb_probability']*100:.2f}%")

    # ── Batch
    with tab_batch:
        st.subheader("Batch Transaction Prediction")
        st.markdown("Upload a CSV with columns: `type, amount, step, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest`")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        # Sample CSV generator
        if st.button("📥 Download Sample CSV Template"):
            sample_csv = (
                "type,amount,step,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest\n"
                "CASH_OUT,181000,1,181000,0,21182,0\n"
                "PAYMENT,9839.64,1,170136,160296.36,0,0\n"
                "TRANSFER,10000,5,50000,40000,0,10000\n"
            )
            st.download_button("⬇️ Save sample.csv", sample_csv, file_name="sample_fraud_input.csv", mime="text/csv")

        if uploaded:
            df_input = pd.read_csv(uploaded)
            st.dataframe(df_input, use_container_width=True)

            if st.button("🔍 Run Batch Prediction", type="primary"):
                records = df_input.to_dict(orient="records")
                with st.spinner(f"Predicting {len(records)} transactions..."):
                    try:
                        r = requests.post(f"{API_BASE}/api/predict/batch", json=records, timeout=30)
                        batch_result = r.json()
                    except Exception as e:
                        st.error(f"API error: {e}")
                        st.stop()

                preds = batch_result.get("predictions", [])
                df_result = df_input.copy()
                df_result["fraud_probability"] = [p.get("fraud_probability", None) for p in preds]
                df_result["is_fraud"]           = [p.get("is_fraud", None) for p in preds]
                df_result["risk_level"]         = [p.get("risk_level", None) for p in preds]

                st.success(f"Predicted {len(preds)} transactions.")
                fraud_count = df_result["is_fraud"].sum()
                b1, b2, b3 = st.columns(3)
                b1.metric("Total", len(df_result))
                b2.metric("Flagged as Fraud", int(fraud_count))
                b3.metric("Fraud Rate", f"{fraud_count/len(df_result)*100:.1f}%")

                st.dataframe(
                    df_result.style.apply(
                        lambda row: ["background-color: #f8d7da" if row["is_fraud"] else "" for _ in row],
                        axis=1,
                    ),
                    use_container_width=True,
                )

                csv_out = df_result.to_csv(index=False)
                st.download_button("⬇️ Download Results CSV", csv_out, file_name="fraud_predictions.csv", mime="text/csv")


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
    | `type` | Transaction type: CASH_OUT, PAYMENT, TRANSFER, CASH_IN, DEBIT |
    | `amount` | Transaction amount (USD) |
    | `nameOrig` | Originating account |
    | `oldbalanceOrg` | Balance before transaction (origin) |
    | `newbalanceOrig` | Balance after transaction (origin) |
    | `nameDest` | Destination account |
    | `oldbalanceDest` | Balance before transaction (destination) |
    | `newbalanceDest` | Balance after transaction (destination) |
    | `isFraud` | **Target label** (1 = fraud, 0 = legitimate) |
    | `isFlaggedFraud` | System flag (not used as feature) |

    ---

    ## Methodology

    ### Feature Engineering
    - **Balance Error (Origin):** `newbalanceOrig + amount − oldbalanceOrg` — non-zero values signal balance manipulation
    - **Balance Error (Dest):** `oldbalanceDest + amount − newbalanceDest` — detects laundering
    - **Zero Balance Flags:** Binary indicators for accounts with zero balance
    - **Amount Ratio:** `amount / (oldbalanceOrg + 1)` — relative transaction size

    ### Class Imbalance
    The dataset is highly imbalanced: only **0.129%** of transactions are fraudulent.
    **SMOTE** (Synthetic Minority Oversampling Technique) is applied to the training set only.

    ### Models
    | Model | Role |
    |---|---|
    | Random Forest | 100 trees, max depth 15, balanced class weights |
    | XGBoost | 200 estimators, depth 6, lr 0.1, scale_pos_weight |
    | **Ensemble** | Average of both models' predicted probabilities |

    ### Results
    | Model | ROC-AUC | F1 Score |
    |---|---|---|
    | Random Forest | 0.9997 | 0.9976 |
    | XGBoost | 0.9998 | 0.9963 |
    | **Ensemble** | **0.9997** | **0.9970** |

    ---

    ## Architecture

    ```
    AIML Dataset.csv
         │
         ▼
    backend/train.py  ──► models/ (RF, XGBoost, LabelEncoder, metrics.json)
                      ──► data/   (eda_snapshot.json)
         │
    backend/app.py    ──► Flask REST API  :5000
         │
    frontend/dashboard.py ──► Streamlit UI  :8501
    ```

    ---

    ## Tech Stack
    - **Python 3.x**
    - **scikit-learn** — Random Forest, preprocessing, metrics
    - **XGBoost** — Gradient boosted trees
    - **imbalanced-learn** — SMOTE oversampling
    - **Flask + Flask-CORS** — REST API backend
    - **Streamlit** — Python-first frontend framework
    - **Plotly** — Interactive charts
    - **pandas / numpy** — Data processing
    """)
