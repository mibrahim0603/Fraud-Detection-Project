# 🔐 Fraud Detection System

A full-stack financial fraud detection application using **Random Forest + XGBoost ensemble + GraphSAGE GNN** trained on the **AIML Dataset** (6.3M transactions). Features a Flask REST API, a Streamlit dashboard with SHAP explainability and fraud ring visualisation.

> **Honest evaluation note:** The GNN previously reported F1=0.991 due to two compounding issues — label-leaking features (`fraud_sent`/`fraud_recv`) and no train/test split. Both are now fixed: leaking features removed, ring-based split applied, metrics evaluated on held-out rings only. Honest GNN test F1 = **0.525**.

## 🌐 Live Demo

**[https://fraud-detection-project-hurba2b9rwbrdtg73xvpbb.streamlit.app/](https://fraud-detection-project-hurba2b9rwbrdtg73xvpbb.streamlit.app/)**

---

## ⚠️ Data Leakage Disclosure

Two engineered features use **post-transaction** balance states unavailable at real decision time:
- `errorBalanceOrig` = `newbalanceOrig + amount − oldbalanceOrg`
- `errorBalanceDest` = `oldbalanceDest + amount − newbalanceDest`

These features carry **~41.8% of combined feature importance** (confirmed by SHAP). The reported ROC-AUC / F1 metrics are therefore an **upper bound**. This is documented in `models/metrics.json` under `leakage_note` and surfaced in the 🧠 SHAP Explainer page.

---

## 📊 Model Performance

| Model | ROC-AUC | F1 Score | Notes |
|---|---|---|---|
| Random Forest | 0.9997 | 0.9976 | 100 trees, depth 15, SMOTE |
| XGBoost | 0.9998 | 0.9963 | 200 estimators, depth 6 |
| **Ensemble** | **0.9997** | **0.9970** | Avg RF + XGB probability |
| GraphSAGE (numpy) | — | **0.525** | Ring-based split, leakage-safe features, held-out test rings |

---

## 🗂 Project Structure

```
Fraud Detection Project/
│
├── AIML Dataset.csv               ← Source dataset (6.3M rows)
│
├── backend/
│   ├── train.py                   ← RF + XGBoost training, leakage-aware metrics
│   ├── app.py                     ← Flask REST API (predict, SHAP, graph endpoints)
│   ├── graph_builder.py           ← Builds account-transaction graph + fraud rings
│   ├── train_gnn.py               ← 2-layer GraphSAGE (pure NumPy), saves gnn_model.pkl
│   └── explain.py                 ← SHAP TreeExplainer wrapper for XGBoost
│
├── frontend/
│   └── dashboard.py               ← 6-page Streamlit dashboard
│
├── models/
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── label_encoder.pkl
│   ├── feature_cols.json
│   ├── metrics.json               ← includes leakage_note
│   └── gnn_model.pkl               ← GraphSAGE weights (joblib format)
│
├── data/
│   ├── eda_snapshot.json
│   ├── fraud_rings.json           ← Top-20 fraud rings (nodes + edges)
│   ├── graph_nodes.json           ← All account node metadata
│   ├── gnn_features.json          ← Node feature matrix for GNN training
│   └── gnn_node_scores.json       ← Per-node GNN fraud probabilities
│
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install pandas scikit-learn xgboost imbalanced-learn flask flask-cors streamlit plotly joblib shap networkx
```

### 2. Train RF + XGBoost Models

```bash
python backend/train.py
```

Outputs: `models/random_forest.pkl`, `models/xgboost.pkl`, `models/metrics.json` (with `leakage_note`), `data/eda_snapshot.json`

### 3. Build the Transaction Graph

```bash
python backend/graph_builder.py
```

Outputs: `data/fraud_rings.json`, `data/graph_nodes.json`, `data/gnn_features.json`

### 4. Train the GraphSAGE GNN

```bash
python backend/train_gnn.py
```

Outputs: `models/gnn_model.pkl`, `data/gnn_node_scores.json`

### 5. Start the Flask API

```bash
python backend/app.py
```

API available at `http://localhost:5000`

### 6. Launch the Streamlit Dashboard

```bash
streamlit run frontend/dashboard.py
```

Dashboard opens at `http://localhost:8501`

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/api/eda` | EDA statistics snapshot |
| GET | `/api/metrics` | Model metrics + leakage note |
| GET | `/api/feature_importance` | Sorted feature importances |
| POST | `/api/predict` | Single transaction prediction |
| POST | `/api/predict/batch` | Batch prediction (JSON array) |
| POST | `/api/shap` | SHAP explanation for one transaction |
| GET | `/api/graph/rings` | Top fraud rings (nodes + edges) |
| GET | `/api/graph/nodes` | All account node metadata |
| GET | `/api/gnn/scores` | Per-node GNN fraud scores |

---

## 🖥 Dashboard Pages (6)

| Page | Description |
|---|---|
| 🏠 Overview | KPI cards, EDA charts, hourly fraud pattern, amount stats |
| 📊 Model Metrics | Comparison table, confusion matrices, feature importance |
| 🔍 Live Prediction | Single + batch inference with fraud probability gauge |
| 🧠 SHAP Explainer | Waterfall chart, leakage disclosure, global SHAP importance |
| 🕸 Fraud Rings | Interactive account-transaction graph + GNN score overlay |
| 📖 About | Dataset, leakage disclosure, methodology, tech stack |

---

## 🧠 Feature Engineering

| Feature | Formula |
|---|---|
| `errorBalanceOrig` | `newbalanceOrig + amount − oldbalanceOrg` |
| `errorBalanceDest` | `oldbalanceDest + amount − newbalanceDest` |
| `origZeroBalance` | `1 if oldbalanceOrg == 0 else 0` |
| `destZeroBalance` | `1 if oldbalanceDest == 0 else 0` |
| `amountRatio` | `amount / (oldbalanceOrg + 1)` |

---

## 🕸 GNN Split Methodology

| Aspect | Detail |
|---|---|
| **Split type** | Ring-based (whole fraud rings held out) |
| **Train/test** | 80% rings train · 20% rings test · 80/20 random for normal nodes |
| **Why not random node split?** | Co-conspirators share edges — random split lets training neighbours leak fraud signal to test nodes via message passing |
| **Message passing scope** | Full graph (transductive), but loss/gradients on train-masked nodes only |
| **Features (safe)** | `total_sent`, `total_recv`, `n_tx`, `out_degree`, `in_degree` |
| **Features removed** | ~~`fraud_sent`~~, ~~`fraud_recv`~~ — direct label aggregates; caused inflated F1=0.991 |
| **Honest test F1** | **0.525** on held-out rings |

---

## 🛠 Tech Stack

- **Python 3.x**
- **scikit-learn** — Random Forest, metrics, preprocessing
- **XGBoost** — Gradient boosted trees
- **imbalanced-learn** — SMOTE oversampling
- **SHAP** — TreeExplainer for XGBoost (waterfall + global importance)
- **networkx** — Graph construction + connected-component fraud ring detection
- **Flask + Flask-CORS** — REST API backend
- **Streamlit** — Python-native frontend
- **Plotly** — Interactive charts + network graph visualisation
- **pandas / numpy** — Data processing
- **joblib** — Model serialisation
