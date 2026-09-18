# 🔐 Fraud Detection System

A full-stack financial fraud detection application using **Random Forest + XGBoost ensemble** trained on the **AIML Dataset** (6.3M transactions). Features a Python Flask REST API backend and a Streamlit interactive dashboard frontend.

---

## 📊 Model Performance

| Model | ROC-AUC | F1 Score | Precision | Recall |
|---|---|---|---|---|
| Random Forest | 0.9997 | 0.9976 | — | — |
| XGBoost | 0.9998 | 0.9963 | — | — |
| **Ensemble** | **0.9997** | **0.9970** | — | — |

---

## 🗂 Project Structure

```
Fraud Detection Project/
│
├── AIML Dataset.csv          ← Source dataset (6.3M rows)
│
├── backend/
│   ├── train.py              ← Data pipeline, feature engineering, model training
│   └── app.py                ← Flask REST API server
│
├── frontend/
│   └── dashboard.py          ← Streamlit interactive dashboard
│
├── models/                   ← Generated after training
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── label_encoder.pkl
│   ├── feature_cols.json
│   └── metrics.json
│
├── data/                     ← Generated after training
│   └── eda_snapshot.json
│
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install pandas scikit-learn xgboost imbalanced-learn flask flask-cors streamlit plotly joblib
```

### 2. Train the Model

```bash
python backend/train.py
```

This will:
- Load and sample the AIML Dataset
- Engineer features (balance errors, zero-balance flags, amount ratio)
- Apply SMOTE oversampling on the training split
- Train Random Forest + XGBoost models
- Save model artifacts to `models/`
- Save an EDA snapshot to `data/`

Expected training time: **3–8 minutes** depending on hardware.

### 3. Start the Flask API

```bash
python backend/app.py
```

API will be available at `http://localhost:5000`

### 4. Launch the Streamlit Dashboard

Open a second terminal and run:

```bash
streamlit run frontend/dashboard.py
```

Dashboard will open at `http://localhost:8501`

---

## 🔌 API Reference

### Health Check
```
GET /health
```

### EDA Snapshot
```
GET /api/eda
```

### Model Metrics
```
GET /api/metrics
```

### Single Prediction
```
POST /api/predict
Content-Type: application/json

{
  "type": "CASH_OUT",
  "amount": 181000,
  "step": 1,
  "oldbalanceOrg": 181000,
  "newbalanceOrig": 0,
  "oldbalanceDest": 21182,
  "newbalanceDest": 0
}
```

**Response:**
```json
{
  "fraud_probability": 0.995,
  "is_fraud": true,
  "risk_level": "HIGH",
  "rf_probability": 0.99,
  "xgb_probability": 1.0
}
```

### Batch Prediction
```
POST /api/predict/batch
Content-Type: application/json

[
  { "type": "CASH_OUT", "amount": 181000, ... },
  { "type": "PAYMENT",  "amount": 9839.64, ... }
]
```

---

## 🖥 Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Overview | KPI cards, transaction type distribution, fraud by type, hourly fraud pattern, amount comparison |
| 📊 Model Metrics | Model comparison table, metric bar charts, confusion matrices, feature importance |
| 🔍 Live Prediction | Single transaction inference with gauge chart + batch CSV upload & download |
| 📖 About | Dataset description, methodology, architecture, tech stack |

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

## 🛠 Tech Stack

- **Python 3.x**
- **scikit-learn** — Random Forest, metrics, preprocessing
- **XGBoost** — Gradient boosted trees
- **imbalanced-learn** — SMOTE oversampling
- **Flask + Flask-CORS** — REST API
- **Streamlit** — Python-native frontend
- **Plotly** — Interactive visualisations
- **pandas / numpy** — Data processing
- **joblib** — Model serialisation
