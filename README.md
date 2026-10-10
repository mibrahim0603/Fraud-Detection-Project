# 🔐 Fraud Detection System

A full-stack financial fraud detection application using **Random Forest + XGBoost ensemble + GraphSAGE GNN** trained on the **AIML Dataset** (6.3M transactions). Features a Flask REST API, a Streamlit dashboard with SHAP explainability and fraud ring visualisation.

> **Research Paper:** *Hybrid Fraud Detection: Ensemble Classification and Graph Neural Networks for Transaction and Ring-Based Fraud Identification* (draft, MJCET)

> **Honest evaluation note:** The GNN previously reported F1=0.991 due to two compounding issues — label-leaking features (`fraud_sent`/`fraud_recv`) and no train/test split. Both are now fixed: leaking features removed, ring-based split applied, metrics evaluated on held-out rings only. Honest GNN overall test: ROC-AUC **0.6971** · Avg-Precision **0.6212** · Precision **67.7%** · Recall **42.9%** · F1 **0.525**.

## 🌐 Live Demo

**[https://fraud-detection-project-hurba2b9rwbrdtg73xvpbb.streamlit.app/](https://fraud-detection-project-hurba2b9rwbrdtg73xvpbb.streamlit.app/)**

---

## ⚠️ Data Leakage Disclosure

The dataset documentation states that fraudulent transactions are cancelled and that these four columns must not be used for fraud detection: **`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest` and `newbalanceDest`**.

Two engineered features are derived directly from these columns and are therefore also tainted:

- `errorBalanceOrig` = `newbalanceOrig + amount − oldbalanceOrg`
- `errorBalanceDest` = `oldbalanceDest + amount − newbalanceDest`

Together these two features hold **41.8% of the equal-weight RF+XGB average feature importance** (`leaky_fi_share` in `models/metrics.json`). The **Full** and **Error-removed** results in the tables below are therefore **upper bounds on deployable performance**, not realistic estimates.

Removing only the two error features (Error-removed set) barely changed RF/XGB performance — `newbalanceOrig` absorbed the same signal. Performance only falls to a realistic level when **all four balance columns are removed**. The **Strict** set (`step`, `type`, `amount`) is the only feature set that fully avoids all four columns and reflects truly deployable performance.

The leakage note is embedded in `models/metrics.json` under the key `leakage_note` and surfaced on the 🧠 SHAP Explainer page.

---

## 📐 Evaluation Methodology

| Aspect | Detail |
|---|---|
| **Working dataset** | All 8,213 fraud rows + 200,000 randomly sampled legitimate rows (`random_state=42`); ~208,213 total |
| **Train/test split** | 80/20 stratified on `isFraud`; test set ≈ 41,643 rows at ~3.95% fraud prevalence |
| **SMOTE** | `k_neighbors=5`, `sampling_strategy="auto"` (minority → majority count); applied to training split only |
| **Cross-validation** | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`; SMOTE applied inside each fold on the fold training split only |
| **SVC subsampling** | Pre-SMOTE stratified subsample: 50,000 rows (single split), 10,000 rows per fold; `class_weight="balanced"` |
| **Prevalence note** | Test set fraud rate ≈ 3.95% vs. 0.129% in the full 6.3M-row dataset — precision figures are optimistic at real-world prevalence |
| **Random state** | 42 everywhere |

---

## 📊 Model Performance

### Feature Sets

| Label | Features used |
|---|---|
| **Full** | `step`, `type_enc`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `errorBalanceOrig`, `errorBalanceDest`, `origZeroBalance`, `destZeroBalance`, `amountRatio` (12 features) |
| **Error-removed** | Full minus `errorBalanceOrig` and `errorBalanceDest` (10 features) |
| **Pre-transaction** | `step`, `type_enc`, `amount`, `oldbalanceOrg`, `oldbalanceDest` (5 features — values knowable before settlement) |
| **Strict** | `step`, `type_enc`, `amount` (3 features — fully avoids all balance columns) |

> ⚠️ Full and Error-removed include tainted balance columns — see Data Leakage Disclosure above.

---

### Single-split results (80/20 stratified, ~41,643 test rows)

#### Full (with_leakage) — upper bound

| Model | ROC-AUC | Avg Prec | Precision | Recall | F1 |
|---|---|---|---|---|---|
| LogisticRegression | 0.9795 | 0.8583 | 0.4442 | 0.9105 | 0.5971 |
| SVC | 0.9965 | 0.9440 | 0.4039 | 0.9884 | 0.5734 |
| DecisionTree | 0.9989 | 0.9939 | 0.9891 | 0.9970 | 0.9930 |
| RandomForest | 0.9997 | 0.9986 | 0.9982 | 0.9970 | 0.9976 |
| XGBoost | 0.9998 | 0.9985 | 0.9963 | 0.9963 | 0.9963 |
| **Ensemble (RF+XGB)** | **0.9997** | **0.9986** | **0.9976** | **0.9963** | **0.9970** |

#### Error-removed (no_leakage) — upper bound

| Model | ROC-AUC | Avg Prec | Precision | Recall | F1 |
|---|---|---|---|---|---|
| LogisticRegression | 0.9779 | 0.8575 | 0.4426 | 0.9008 | 0.5935 |
| SVC | 0.9894 | 0.8536 | 0.5278 | 0.9191 | 0.6705 |
| DecisionTree | 0.9991 | 0.9964 | 0.9927 | 0.9970 | 0.9948 |
| RandomForest | 0.9996 | 0.9986 | 0.9982 | 0.9970 | 0.9976 |
| XGBoost | 0.9997 | 0.9985 | 0.9963 | 0.9963 | 0.9963 |
| **Ensemble (RF+XGB)** | **0.9997** | **0.9986** | **0.9963** | **0.9963** | **0.9963** |

#### Pre-transaction — partial leakage removal

| Model | ROC-AUC | Avg Prec | Precision | Recall | F1 |
|---|---|---|---|---|---|
| LogisticRegression | 0.8457 | 0.3511 | 0.1733 | 0.7231 | 0.2796 |
| SVC | 0.9777 | 0.5915 | 0.4397 | 0.8959 | 0.5899 |
| DecisionTree | 0.9935 | 0.8974 | 0.4616 | 0.9817 | 0.6280 |
| RandomForest | 0.9984 | 0.9781 | 0.6950 | 0.9848 | 0.8149 |
| XGBoost | 0.9990 | 0.9880 | 0.8061 | 0.9921 | 0.8895 |
| **Ensemble (RF+XGB)** | **0.9989** | **0.9862** | **0.7837** | **0.9903** | **0.8750** |

#### Strict — closest to deployable (synthetic data, 3.95% fraud in the test set)

| Model | ROC-AUC | Avg Prec | Precision | Recall | F1 |
|---|---|---|---|---|---|
| LogisticRegression | 0.8434 | 0.3332 | 0.1766 | 0.7188 | 0.2835 |
| SVC | 0.9263 | 0.4135 | 0.2358 | 0.8460 | 0.3688 |
| DecisionTree | 0.9519 | 0.6794 | 0.2992 | 0.8375 | 0.4409 |
| RandomForest | 0.9603 | 0.7408 | 0.3739 | 0.8217 | 0.5139 |
| XGBoost | 0.9686 | 0.7753 | 0.4098 | 0.8411 | 0.5511 |
| **Ensemble (RF+XGB)** | **0.9676** | **0.7677** | **0.4063** | **0.8338** | **0.5464** |

---

### 5-fold cross-validation (mean ± std)

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` · SMOTE inside each fold · models: LogReg, SVC, DecisionTree, RandomForest, XGBoost · no ensemble (CV does not save combined weights).

#### Error-removed (kfold_no_leakage)

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| LogisticRegression | 0.9758 ± 0.0009 | 0.6011 ± 0.0056 | 0.4528 ± 0.0064 | 0.8939 ± 0.0051 |
| SVC | 0.9672 ± 0.0029 | 0.7645 ± 0.0070 | 0.7698 ± 0.0104 | 0.7597 ± 0.0160 |
| DecisionTree | 0.9990 ± 0.0005 | 0.9944 ± 0.0012 | 0.9915 ± 0.0025 | 0.9973 ± 0.0013 |
| RandomForest | 0.9996 ± 0.0002 | 0.9973 ± 0.0008 | 0.9981 ± 0.0010 | 0.9966 ± 0.0018 |
| XGBoost | 0.9996 ± 0.0002 | 0.9964 ± 0.0015 | 0.9962 ± 0.0022 | 0.9966 ± 0.0016 |

#### Pre-transaction (kfold_pre_transaction)

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| LogisticRegression | 0.8408 ± 0.0044 | 0.2789 ± 0.0030 | 0.1734 ± 0.0021 | 0.7123 ± 0.0106 |
| SVC | 0.9654 ± 0.0018 | 0.7380 ± 0.0053 | 0.7457 ± 0.0314 | 0.7321 ± 0.0187 |
| DecisionTree | 0.9944 ± 0.0007 | 0.6516 ± 0.0118 | 0.4887 ± 0.0121 | 0.9777 ± 0.0057 |
| RandomForest | 0.9984 ± 0.0004 | 0.8267 ± 0.0070 | 0.7121 ± 0.0108 | 0.9854 ± 0.0034 |
| XGBoost | 0.9991 ± 0.0003 | 0.8895 ± 0.0080 | 0.8059 ± 0.0125 | 0.9927 ± 0.0014 |

#### Strict (kfold_strict)

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| LogisticRegression | 0.8369 ± 0.0038 | 0.2819 ± 0.0032 | 0.1760 ± 0.0023 | 0.7084 ± 0.0075 |
| SVC | 0.9124 ± 0.0039 | 0.1675 ± 0.0139 | 0.9747 ± 0.0107 | 0.0917 ± 0.0084 |
| DecisionTree | 0.9497 ± 0.0036 | 0.4615 ± 0.0155 | 0.3212 ± 0.0150 | 0.8213 ± 0.0108 |
| RandomForest | 0.9590 ± 0.0021 | 0.5028 ± 0.0066 | 0.3635 ± 0.0055 | 0.8149 ± 0.0092 |
| XGBoost | 0.9684 ± 0.0014 | 0.5535 ± 0.0028 | 0.4139 ± 0.0032 | 0.8351 ± 0.0074 |

---

### GraphSAGE GNN (ring-based split)

Graph built from 500 fraud + 1,000 normal transactions. Ring-based split: 2,396 train nodes, 597 test nodes.

| Segment | ROC-AUC | Avg Prec | Precision | Recall | F1 | Notes |
|---|---|---|---|---|---|---|
| Overall (test nodes) | 0.6971 | 0.6212 | 0.6774 | 0.4286 | 0.5250 | TP=84, FP=40, FN=112, TN=361 |
| Ring-involved accounts | N/A¹ | N/A¹ | 1.0000 | 0.2500 | 0.4000 | 8 nodes, all fraud, 4 held-out rings; TP=2, FP=0, FN=6 |
| Isolated accounts | 0.7024 | 0.6198 | 0.6721 | 0.4362 | 0.5290 | 589 nodes; TP=82, FP=40, FN=106, TN=361 |

> ¹ ROC-AUC undefined — all 8 ring-account test nodes are fraud (single class). Precision=1.0 means every ring-account flagged was a true positive; low recall reflects small, sparse ring clusters.

**GNN limitations:**
- A node is labelled fraud if it **sent or received** any fraudulent transaction (not necessarily the originator).
- No separate validation set is used during training — the epoch saved is the one with the **lowest training loss**; the test set is used only once at the end.
- Only **4 fraud rings** were held out for testing (8 nodes total), so ring-level precision/recall are indicative, not statistically robust.

---

## 🗂 Project Structure

```
Fraud Detection Project/
│
├── AIML Dataset.csv               ← Source dataset (6.3M rows)
│
├── backend/
│   ├── train.py                   ← RF + XGBoost training; four feature sets; leakage-aware metrics
│   ├── run_kfold.py               ← 5-fold stratified CV for no_leakage, strict, pre_transaction sets
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
│   ├── metrics.json               ← all holdout + CV results + leakage_note + GNN metrics
│   └── gnn_model.pkl              ← GraphSAGE weights (joblib format)
│
├── data/
│   ├── eda_snapshot.json
│   ├── fraud_rings.json           ← Top-20 fraud rings (nodes + edges)
│   ├── graph_nodes.json           ← All account node metadata
│   ├── graph_edges.json           ← All transaction edges with features
│   ├── gnn_features.json          ← Node feature matrix for GNN training
│   ├── gnn_node_scores.json       ← Per-node GNN fraud probabilities
│   └── kfold_raw_scores.json      ← Raw per-fold CV scores for all three CV feature sets
│
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train RF + XGBoost Models (all four feature sets)

```bash
python backend/train.py
```

Outputs: `models/random_forest.pkl`, `models/xgboost.pkl`, `models/metrics.json` (keys: `with_leakage`, `no_leakage`, `strict`, `pre_transaction`, `leakage_note`), `data/eda_snapshot.json`

### 3. Run 5-Fold Cross-Validation

```bash
python backend/run_kfold.py
```

Outputs: `models/metrics.json` (adds keys `kfold_no_leakage`, `kfold_strict`, `kfold_pre_transaction`), `data/kfold_raw_scores.json`

### 4. Build the Transaction Graph

```bash
python backend/graph_builder.py
```

Outputs: `data/fraud_rings.json`, `data/graph_nodes.json`, `data/graph_edges.json`, `data/gnn_features.json`

### 5. Train the GraphSAGE GNN

```bash
python backend/train_gnn.py
```

Outputs: `models/gnn_model.pkl`, `data/gnn_node_scores.json`; merges GNN metrics into `models/metrics.json`

### 6. Start the Flask API

```bash
python backend/app.py
```

API available at `http://localhost:5000`

### 7. Launch the Streamlit Dashboard

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
| 📊 Model Metrics | Feature-set selector (Full / Error-removed / Pre-transaction / Strict), all-model comparison table, confusion matrices, feature importance bar chart, 5-fold CV section with selectbox and mean ± std table |
| 🔍 Live Prediction | Single + batch inference with fraud probability gauge |
| 🧠 SHAP Explainer | Waterfall chart, leakage disclosure, global SHAP importance |
| 🕸 Fraud Rings | Interactive account-transaction graph + GNN score overlay + segmented metrics |
| 📖 About | Dataset, leakage disclosure, methodology, tech stack |

---

## 🧠 Feature Engineering

| Feature | Formula | Leakage? |
|---|---|---|
| `errorBalanceOrig` | `newbalanceOrig + amount − oldbalanceOrg` | ⚠️ Yes |
| `errorBalanceDest` | `oldbalanceDest + amount − newbalanceDest` | ⚠️ Yes |
| `origZeroBalance` | `1 if oldbalanceOrg == 0 else 0` | ⚠️ Indirect¹ |
| `destZeroBalance` | `1 if oldbalanceDest == 0 else 0` | ⚠️ Indirect¹ |
| `amountRatio` | `amount / (oldbalanceOrg + 1)` | ⚠️ Indirect¹ |

> ¹ These features use `oldbalanceOrg` or `oldbalanceDest` — present in the Error-removed and no_leakage sets but excluded from the Strict set.

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
| **Node label rule** | Fraud if the account sent **or** received any fraudulent transaction |
| **Epoch selection** | Weights saved at epoch with lowest **training** loss (no validation set) |
| **Overall test metrics** | ROC-AUC **0.6971** · Avg-Prec **0.6212** · Precision **67.7%** · Recall **42.9%** · F1 **0.525** |
| **Ring-involved accounts** | 8 nodes · Precision **100%** · Recall **25%** · F1 **0.400** · ROC-AUC N/A (all fraud, single class) |
| **Isolated accounts** | 589 nodes · ROC-AUC **0.7024** · Avg-Prec **0.6198** · Precision **67.2%** · Recall **43.6%** · F1 **0.529** |
| **Ring vs isolated delta** | F1: ring −0.129 vs isolated — only 4 held-out rings (8 nodes) limits topology signal measurement |

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
