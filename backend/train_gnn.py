"""
train_gnn.py — GraphSAGE Node Classifier (pure NumPy, no PyTorch required)
Trains a 2-layer GraphSAGE on the account-transaction graph built by graph_builder.py.
Saves the trained weight matrices to models/gnn_model.pt (joblib format).

Architecture:
  Input (5 features) → SAGE-Layer-1 (64 hidden) → ReLU → SAGE-Layer-2 (32) → ReLU
  → Linear → Sigmoid → fraud probability per node

GraphSAGE aggregation (mean):
  h_v^(k) = W · CONCAT( h_v^(k-1),  MEAN_{u ∈ N(v)}[ h_u^(k-1) ] )
"""

import os
import json
import joblib
import numpy as np

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ─── Hyper-parameters ────────────────────────────────────────────────────────
HIDDEN1    = 64
HIDDEN2    = 32
LR         = 0.01
EPOCHS     = 200
SEED       = 42
np.random.seed(SEED)


# ─── Activation / helpers ────────────────────────────────────────────────────
def relu(x):
    return np.maximum(0.0, x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def bce_loss(probs, labels):
    eps = 1e-7
    p   = np.clip(probs, eps, 1 - eps)
    return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))


def normalise(X: np.ndarray) -> np.ndarray:
    mu  = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mu) / std, mu, std


# ─── Build neighbour lookup from COO edge_index ──────────────────────────────
def build_adj(edge_index, num_nodes):
    adj = [[] for _ in range(num_nodes)]
    for src, dst in edge_index:
        adj[dst].append(src)   # in-neighbours (who sends to me)
        adj[src].append(dst)   # out-neighbours (who I send to)
    return adj


# ─── GraphSAGE mean aggregation (single layer forward) ───────────────────────
def sage_layer(H, adj, W):
    """
    H   : (N, d_in)
    adj : list of neighbour index lists
    W   : (2*d_in, d_out)
    Returns (N, d_out)
    """
    N    = H.shape[0]
    agg  = np.zeros_like(H)
    for v in range(N):
        nbrs = adj[v]
        if nbrs:
            agg[v] = H[np.array(nbrs)].mean(axis=0)
        else:
            agg[v] = H[v]                  # self-loop fallback
    concat = np.concatenate([H, agg], axis=1)   # (N, 2*d_in)
    return relu(concat @ W)


# ─── Forward pass ────────────────────────────────────────────────────────────
def forward(X, adj, W1, W2, Wout, b_out):
    h1   = sage_layer(X,  adj, W1)              # (N, HIDDEN1)
    h2   = sage_layer(h1, adj, W2)              # (N, HIDDEN2)
    logit = h2 @ Wout + b_out                   # (N, 1)
    return sigmoid(logit.squeeze(1)), h1, h2


# ─── Training ────────────────────────────────────────────────────────────────
def train(gnn_data: dict):
    X_raw    = np.array(gnn_data["features"],  dtype=np.float32)
    y        = np.array(gnn_data["labels"],    dtype=np.float32)
    ei       = gnn_data["edge_index"]
    N, d_in  = X_raw.shape

    X, mu, std = normalise(X_raw)
    adj        = build_adj(ei, N)

    # ── Weight initialisation (Xavier) ───────────────────────────────────────
    def xavier(rows, cols):
        limit = np.sqrt(6.0 / (rows + cols))
        return np.random.uniform(-limit, limit, (rows, cols)).astype(np.float32)

    W1   = xavier(2 * d_in,   HIDDEN1)
    W2   = xavier(2 * HIDDEN1, HIDDEN2)
    Wout = xavier(HIDDEN2,    1)
    b_out = np.zeros(1, dtype=np.float32)

    # Class weight for imbalanced labels
    n_pos  = y.sum()
    n_neg  = N - n_pos
    pos_w  = n_neg / (n_pos + 1e-8)

    best_loss = np.inf
    best_weights = None

    print(f"[GNN] Nodes={N}, Fraud={int(n_pos)}, Edges={len(ei)}")
    print(f"[GNN] Training {EPOCHS} epochs …")

    for epoch in range(EPOCHS):
        # ── Forward ──────────────────────────────────────────────────────────
        probs, h1, h2 = forward(X, adj, W1, W2, Wout, b_out)

        # Weighted BCE
        eps = 1e-7
        p   = np.clip(probs, eps, 1 - eps)
        w   = np.where(y == 1, pos_w, 1.0)
        loss = -np.mean(w * (y * np.log(p) + (1 - y) * np.log(1 - p)))

        if loss < best_loss:
            best_loss    = loss
            best_weights = (W1.copy(), W2.copy(), Wout.copy(), b_out.copy())

        # ── Backprop (analytical gradients through sigmoid+linear head) ──────
        # dL/d_logit
        d_logit = w * (probs - y) / N          # (N,)

        # Wout, b_out gradients
        grad_Wout = h2.T @ d_logit[:, None]    # (HIDDEN2, 1)
        grad_bout = d_logit.sum(keepdims=True)

        # Backprop into h2
        d_h2 = d_logit[:, None] @ Wout.T      # (N, HIDDEN2)
        d_h2[h2 <= 0] = 0                     # ReLU gate

        # Backprop through sage_layer-2
        # h2 = relu(concat2 @ W2),  concat2 = [h1, agg1]
        # We approximate dL/dW2 numerically-free via concat2 reconstruction
        agg1    = np.zeros_like(h1)
        for v in range(N):
            nbrs = adj[v]
            agg1[v] = h1[np.array(nbrs)].mean(axis=0) if nbrs else h1[v]
        concat2 = np.concatenate([h1, agg1], axis=1)
        grad_W2 = concat2.T @ d_h2

        # Backprop into h1 (through W2 only, skip neighbour grad for speed)
        # W2 is (2*HIDDEN1, HIDDEN2); take first HIDDEN1 rows (self half)
        d_h1 = d_h2 @ W2[:HIDDEN1, :].T       # (N, HIDDEN1)
        d_h1[h1 <= 0] = 0

        # Backprop through sage_layer-1
        agg0    = np.zeros_like(X)
        for v in range(N):
            nbrs = adj[v]
            agg0[v] = X[np.array(nbrs)].mean(axis=0) if nbrs else X[v]
        concat1 = np.concatenate([X, agg0], axis=1)
        grad_W1 = concat1.T @ d_h1

        # ── Gradient update (SGD) ────────────────────────────────────────────
        W1    -= LR * grad_W1
        W2    -= LR * grad_W2
        Wout  -= LR * grad_Wout
        b_out -= LR * grad_bout

        if epoch % 40 == 0 or epoch == EPOCHS - 1:
            preds  = (probs >= 0.5).astype(int)
            tp = int(((preds == 1) & (y == 1)).sum())
            fp = int(((preds == 1) & (y == 0)).sum())
            fn = int(((preds == 0) & (y == 1)).sum())
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1   = 2 * prec * rec / (prec + rec + 1e-8)
            print(f"  Epoch {epoch:3d}  loss={loss:.4f}  prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}")

    W1, W2, Wout, b_out = best_weights

    # ── Final evaluation ─────────────────────────────────────────────────────
    probs_final, _, _ = forward(X, adj, W1, W2, Wout, b_out)
    preds_final = (probs_final >= 0.5).astype(int)
    tp = int(((preds_final == 1) & (y == 1)).sum())
    fp = int(((preds_final == 1) & (y == 0)).sum())
    fn = int(((preds_final == 0) & (y == 1)).sum())
    tn = int(((preds_final == 0) & (y == 0)).sum())
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)

    print(f"\n[GNN] Final — TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"[GNN] Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

    # ── Node embeddings for visualisation (h2 layer) ─────────────────────────
    _, _, embeddings = forward(X, adj, W1, W2, Wout, b_out)
    node_scores = {
        gnn_data["node_list"][i]: round(float(probs_final[i]), 4)
        for i in range(N)
    }

    return {
        "W1": W1, "W2": W2, "Wout": Wout, "b_out": b_out,
        "mu": mu, "std": std,
        "d_in": d_in, "hidden1": HIDDEN1, "hidden2": HIDDEN2,
        "metrics": {
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "node_scores": node_scores,
        "embeddings":  embeddings.tolist(),
        "node_list":   gnn_data["node_list"],
    }


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gnn_features_path = os.path.join(DATA_DIR, "gnn_features.json")
    if not os.path.exists(gnn_features_path):
        print("[GNN] gnn_features.json not found — run graph_builder.py first.")
        raise SystemExit(1)

    with open(gnn_features_path) as f:
        gnn_data = json.load(f)

    model = train(gnn_data)

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, "gnn_model.pt")
    joblib.dump(model, out_path)

    # Persist node scores to data/
    with open(os.path.join(DATA_DIR, "gnn_node_scores.json"), "w") as f:
        json.dump({
            "node_scores": model["node_scores"],
            "metrics":     model["metrics"],
        }, f, indent=2)

    print(f"\n[GNN] Model saved -> {out_path}")
    print(f"[GNN] Node scores -> {DATA_DIR}/gnn_node_scores.json")
