"""
train_gnn.py — GraphSAGE Node Classifier (pure NumPy, no PyTorch required)

Split methodology: RING-BASED (transductive, inductive-style evaluation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Fraud rings are identified first (connected components containing ≥1 fraud edge).
  • 20% of rings (whole rings) are held out as the test set — no ring straddles the
    split boundary, so no test node's label can leak through shared edges to training.
  • Normal (non-fraud) nodes are split 80/20 by random node index.
  • The full graph is used for message passing (transductive), but loss and
    gradient updates are computed ONLY over train-mask nodes.
  • Evaluation metrics are reported ONLY over test-mask nodes.

Why ring-based rather than random node split?
  Random node splits are unsuitable for fraud graphs: a fraudster node might be
  in the test set while its co-conspirator (sharing an edge) is in the train set.
  The GNN would see the fraud signal propagated from the training neighbour during
  message passing, inflating test performance. Ring-based splitting prevents this by
  keeping entire fraud clusters on one side of the split.

Leakage-safe features (no label-derived inputs):
  Original features  fraud_sent, fraud_recv  have been REMOVED — they are
  direct counts of fraud edges and trivially encode the node label.
  Safe features used:
    • total_sent   — total amount sent across all transactions
    • total_recv   — total amount received
    • n_tx         — total number of transactions (in + out)
    • out_degree   — number of outgoing transactions
    • in_degree    — number of incoming transactions

Architecture: 2-layer GraphSAGE (mean aggregation) → sigmoid output
"""

import os
import json
import joblib
import numpy as np

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

HIDDEN1    = 64
HIDDEN2    = 32
LR         = 0.01
EPOCHS     = 300
SEED       = 42
TEST_RING_FRAC  = 0.20   # fraction of fraud rings held out for testing
TEST_NORMAL_FRAC = 0.20  # fraction of normal nodes held out for testing
np.random.seed(SEED)


# ─── Activations ─────────────────────────────────────────────────────────────
def relu(x):    return np.maximum(0.0, x)
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def normalise(X, mu=None, std=None):
    if mu is None:
        mu  = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
    return (X - mu) / std, mu, std


# ─── Adjacency list ───────────────────────────────────────────────────────────
def build_adj(edge_index, num_nodes):
    adj = [[] for _ in range(num_nodes)]
    for src, dst in edge_index:
        adj[dst].append(src)
        adj[src].append(dst)
    return adj


# ─── GraphSAGE layer ──────────────────────────────────────────────────────────
def sage_layer(H, adj, W):
    N   = H.shape[0]
    agg = np.zeros_like(H)
    for v in range(N):
        nbrs = adj[v]
        agg[v] = H[np.array(nbrs)].mean(axis=0) if nbrs else H[v]
    return relu(np.concatenate([H, agg], axis=1) @ W)


def forward(X, adj, W1, W2, Wout, b_out):
    h1    = sage_layer(X,  adj, W1)
    h2    = sage_layer(h1, adj, W2)
    logit = h2 @ Wout + b_out
    return sigmoid(logit.squeeze(1)), h1, h2


# ─── Ring-based train / test split ───────────────────────────────────────────
def ring_split(gnn_data: dict, fraud_rings: list):
    """
    Returns train_mask, test_mask (boolean arrays of shape [N]).

    Strategy:
      1. Collect all node indices that belong to fraud rings.
      2. Shuffle the rings and hold out TEST_RING_FRAC of them.
         Every node in a held-out ring goes to the test set.
      3. Remaining fraud-ring nodes go to the train set.
      4. Normal nodes (not in any fraud ring) are split randomly 80/20.
    """
    N          = gnn_data["num_nodes"]
    node_index = gnn_data["node_index"]
    labels     = np.array(gnn_data["labels"])

    # Build ring → node-index sets
    ring_node_sets = []
    for ring in fraud_rings:
        ids = set()
        for n in ring["nodes"]:
            nid = n["id"]
            if nid in node_index:
                ids.add(node_index[nid])
        if ids:
            ring_node_sets.append(ids)

    # Shuffle rings, hold out last TEST_RING_FRAC
    rng = np.random.RandomState(SEED)
    rng.shuffle(ring_node_sets)
    n_test_rings = max(1, int(len(ring_node_sets) * TEST_RING_FRAC))
    test_ring_sets  = ring_node_sets[-n_test_rings:]
    train_ring_sets = ring_node_sets[:-n_test_rings]

    test_ring_nodes  = set().union(*test_ring_sets)  if test_ring_sets  else set()
    train_ring_nodes = set().union(*train_ring_sets) if train_ring_sets else set()
    all_ring_nodes   = test_ring_nodes | train_ring_nodes

    # Normal nodes (not in any fraud ring) — random 80/20
    normal_nodes = [i for i in range(N) if i not in all_ring_nodes]
    rng.shuffle(normal_nodes)
    n_test_normal    = max(1, int(len(normal_nodes) * TEST_NORMAL_FRAC))
    test_normal_set  = set(normal_nodes[-n_test_normal:])
    train_normal_set = set(normal_nodes[:-n_test_normal])

    train_mask = np.zeros(N, dtype=bool)
    test_mask  = np.zeros(N, dtype=bool)

    for i in train_ring_nodes:  train_mask[i] = True
    for i in train_normal_set:  train_mask[i] = True
    for i in test_ring_nodes:   test_mask[i]  = True
    for i in test_normal_set:   test_mask[i]  = True

    print(f"[GNN] Ring split: {len(ring_node_sets)} fraud rings "
          f"({len(ring_node_sets)-n_test_rings} train / {n_test_rings} test)")
    print(f"[GNN] Train nodes: {train_mask.sum()} "
          f"(fraud={labels[train_mask].sum():.0f}, "
          f"normal={(~labels[train_mask].astype(bool)).sum()})")
    print(f"[GNN] Test  nodes: {test_mask.sum()} "
          f"(fraud={labels[test_mask].sum():.0f}, "
          f"normal={(~labels[test_mask].astype(bool)).sum()})")

    # Verify no node is in both sets
    assert not (train_mask & test_mask).any(), "Mask overlap — split is broken"

    return train_mask, test_mask


# ─── Training ────────────────────────────────────────────────────────────────
def train(gnn_data: dict, fraud_rings: list):
    X_raw = np.array(gnn_data["features"], dtype=np.float32)
    y     = np.array(gnn_data["labels"],   dtype=np.float32)
    ei    = gnn_data["edge_index"]
    N     = X_raw.shape[0]

    # Normalise using train-set statistics only
    train_mask, test_mask = ring_split(gnn_data, fraud_rings)
    X_norm, mu, std = normalise(X_raw[train_mask])
    X, _, _         = normalise(X_raw, mu, std)   # apply same scale to all nodes

    adj = build_adj(ei, N)

    def xavier(r, c):
        lim = np.sqrt(6.0 / (r + c))
        return np.random.uniform(-lim, lim, (r, c)).astype(np.float32)

    d_in  = X.shape[1]
    W1    = xavier(2 * d_in,    HIDDEN1)
    W2    = xavier(2 * HIDDEN1, HIDDEN2)
    Wout  = xavier(HIDDEN2,     1)
    b_out = np.zeros(1, dtype=np.float32)

    # Class weight computed from TRAIN set only
    y_tr  = y[train_mask]
    n_pos = y_tr.sum()
    n_neg = train_mask.sum() - n_pos
    pos_w = n_neg / (n_pos + 1e-8)

    best_loss, best_weights = np.inf, None

    print(f"[GNN] Training {EPOCHS} epochs on train-masked nodes "
          f"(message passing uses full graph) ...")

    for epoch in range(EPOCHS):
        probs, h1, h2 = forward(X, adj, W1, W2, Wout, b_out)

        # Loss ONLY over train nodes
        p_tr   = np.clip(probs[train_mask], 1e-7, 1 - 1e-7)
        w_tr   = np.where(y_tr == 1, pos_w, 1.0)
        loss   = -np.mean(w_tr * (y_tr * np.log(p_tr) + (1 - y_tr) * np.log(1 - p_tr)))

        if loss < best_loss:
            best_loss    = loss
            best_weights = (W1.copy(), W2.copy(), Wout.copy(), b_out.copy())

        # Gradients w.r.t. ALL nodes (message passing is global), zero out test
        d_logit       = np.zeros(N, dtype=np.float32)
        d_logit_train = w_tr * (probs[train_mask] - y_tr) / train_mask.sum()
        d_logit[train_mask] = d_logit_train

        grad_Wout = h2.T @ d_logit[:, None]
        grad_bout = d_logit.sum(keepdims=True)

        d_h2 = d_logit[:, None] @ Wout.T
        d_h2[h2 <= 0] = 0

        agg1    = np.zeros_like(h1)
        for v in range(N):
            nbrs = adj[v]
            agg1[v] = h1[np.array(nbrs)].mean(axis=0) if nbrs else h1[v]
        concat2 = np.concatenate([h1, agg1], axis=1)
        grad_W2 = concat2.T @ d_h2

        d_h1 = d_h2 @ W2[:HIDDEN1, :].T
        d_h1[h1 <= 0] = 0

        agg0    = np.zeros_like(X)
        for v in range(N):
            nbrs = adj[v]
            agg0[v] = X[np.array(nbrs)].mean(axis=0) if nbrs else X[v]
        concat1 = np.concatenate([X, agg0], axis=1)
        grad_W1 = concat1.T @ d_h1

        W1    -= LR * grad_W1
        W2    -= LR * grad_W2
        Wout  -= LR * grad_Wout
        b_out -= LR * grad_bout

        if epoch % 60 == 0 or epoch == EPOCHS - 1:
            preds_tr = (probs[train_mask] >= 0.5).astype(int)
            tp = int(((preds_tr == 1) & (y_tr == 1)).sum())
            fp = int(((preds_tr == 1) & (y_tr == 0)).sum())
            fn = int(((preds_tr == 0) & (y_tr == 1)).sum())
            p_ = tp / (tp + fp + 1e-8)
            r_ = tp / (tp + fn + 1e-8)
            f1 = 2 * p_ * r_ / (p_ + r_ + 1e-8)
            print(f"  Epoch {epoch:3d}  loss={loss:.4f}  "
                  f"train prec={p_:.3f}  rec={r_:.3f}  F1={f1:.3f}")

    W1, W2, Wout, b_out = best_weights

    # ── Evaluate on HELD-OUT TEST NODES ONLY ─────────────────────────────────
    probs_final, _, h2_final = forward(X, adj, W1, W2, Wout, b_out)
    y_te      = y[test_mask]
    p_te      = probs_final[test_mask]
    preds_te  = (p_te >= 0.5).astype(int)

    tp = int(((preds_te == 1) & (y_te == 1)).sum())
    fp = int(((preds_te == 1) & (y_te == 0)).sum())
    fn = int(((preds_te == 0) & (y_te == 1)).sum())
    tn = int(((preds_te == 0) & (y_te == 0)).sum())
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)

    print(f"\n[GNN] TEST SET (held-out rings) — TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"[GNN] Test Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

    node_scores = {
        gnn_data["node_list"][i]: round(float(probs_final[i]), 4)
        for i in range(N)
    }

    return {
        "W1": W1, "W2": W2, "Wout": Wout, "b_out": b_out,
        "mu": mu.tolist(), "std": std.tolist(),
        "d_in": d_in, "hidden1": HIDDEN1, "hidden2": HIDDEN2,
        "split": {
            "method":           "ring-based",
            "test_ring_frac":   TEST_RING_FRAC,
            "test_normal_frac": TEST_NORMAL_FRAC,
            "train_nodes":      int(train_mask.sum()),
            "test_nodes":       int(test_mask.sum()),
            "note": (
                "Loss and gradients computed over train-masked nodes only. "
                "Message passing uses the full graph (transductive). "
                "Fraud rings are kept whole — no ring straddles the split. "
                "Features fraud_sent and fraud_recv were removed to prevent "
                "label leakage (they are direct aggregates of the target label)."
            ),
        },
        "metrics": {
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "evaluated_on": "held-out test nodes only (ring-based split)",
        },
        "node_scores": node_scores,
        "embeddings":  h2_final.tolist(),
        "node_list":   gnn_data["node_list"],
    }


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gnn_path   = os.path.join(DATA_DIR, "gnn_features.json")
    rings_path = os.path.join(DATA_DIR, "fraud_rings.json")

    for p in [gnn_path, rings_path]:
        if not os.path.exists(p):
            print(f"[GNN] Missing: {p} — run graph_builder.py first.")
            raise SystemExit(1)

    with open(gnn_path)   as f: gnn_data    = json.load(f)
    with open(rings_path) as f: fraud_rings = json.load(f)

    model = train(gnn_data, fraud_rings)

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, "gnn_model.pt")
    joblib.dump(model, out_path)

    with open(os.path.join(DATA_DIR, "gnn_node_scores.json"), "w") as f:
        json.dump({
            "node_scores": model["node_scores"],
            "metrics":     model["metrics"],
            "split":       model["split"],
        }, f, indent=2)

    print(f"\n[GNN] Model saved -> {out_path}")
    print(f"[GNN] Node scores -> {DATA_DIR}/gnn_node_scores.json")
