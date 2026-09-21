"""
graph_builder.py — Account-Transaction Graph Builder
Builds a directed bipartite graph from the AIML Dataset:
  - Account nodes  (nameOrig / nameDest)
  - Transaction edges with features (amount, type, isFraud, step)

Outputs:
  data/graph_nodes.json   — node metadata (id, type, total_sent, total_recv, fraud_count)
  data/graph_edges.json   — edge list (src, dst, amount, type, isFraud, step)
  data/fraud_rings.json   — connected components that contain ≥1 fraud transaction
  data/gnn_features.json  — per-node feature matrix + labels for GNN training
"""

import os
import json
import numpy as np
import pandas as pd
import networkx as nx

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "AIML Dataset.csv")
DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")

# ── Sampling limits (keeps memory and render time reasonable) ─────────────────
MAX_FRAUD_TX   = 500    # fraud transactions to include in graph
MAX_NORMAL_TX  = 1000   # normal transactions to include (for context)
MAX_RING_NODES = 80     # max nodes per exported fraud ring


def build_graph(path: str = DATASET_PATH):
    print("[graph] Loading dataset …")
    df = pd.read_csv(path)

    fraud_df  = df[df["isFraud"] == 1].copy()
    normal_df = df[df["isFraud"] == 0].sample(
        n=min(MAX_NORMAL_TX, len(df[df["isFraud"] == 0])), random_state=42
    )
    sample = pd.concat([fraud_df.head(MAX_FRAUD_TX), normal_df]).reset_index(drop=True)
    print(f"[graph] Working sample: {len(sample)} transactions "
          f"({sample['isFraud'].sum()} fraud, {(sample['isFraud']==0).sum()} normal)")

    G = nx.DiGraph()

    # ── Add account nodes ────────────────────────────────────────────────────
    all_accounts = set(sample["nameOrig"]) | set(sample["nameDest"])
    for acc in all_accounts:
        G.add_node(acc, node_type="account")

    # ── Add directed edges (one per transaction) ─────────────────────────────
    for _, row in sample.iterrows():
        G.add_edge(
            row["nameOrig"], row["nameDest"],
            amount   = float(row["amount"]),
            tx_type  = str(row["type"]),
            is_fraud = int(row["isFraud"]),
            step     = int(row["step"]),
        )

    print(f"[graph] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Compute per-node aggregate features ──────────────────────────────────
    # NOTE: fraud_sent / fraud_recv are stored in node_meta for ring visualisation
    # but are NOT included in the GNN feature matrix (they directly encode the label).
    node_meta = {}
    for node in G.nodes():
        out_edges  = list(G.out_edges(node, data=True))
        in_edges   = list(G.in_edges(node, data=True))
        total_sent = sum(d["amount"] for _, _, d in out_edges)
        total_recv = sum(d["amount"] for _, _, d in in_edges)
        fraud_sent = sum(d["is_fraud"] for _, _, d in out_edges)
        fraud_recv = sum(d["is_fraud"] for _, _, d in in_edges)
        out_degree = len(out_edges)
        in_degree  = len(in_edges)
        n_tx       = out_degree + in_degree
        is_fraud_node = int((fraud_sent + fraud_recv) > 0)
        node_meta[node] = {
            "id":            node,
            "total_sent":    round(total_sent, 2),
            "total_recv":    round(total_recv, 2),
            "fraud_sent":    fraud_sent,    # kept for ring viz only, NOT in GNN features
            "fraud_recv":    fraud_recv,    # kept for ring viz only, NOT in GNN features
            "out_degree":    out_degree,
            "in_degree":     in_degree,
            "n_tx":          n_tx,
            "is_fraud_node": is_fraud_node,
        }

    # ── Fraud rings via weakly-connected components ──────────────────────────
    undirected = G.to_undirected()
    components = list(nx.connected_components(undirected))
    fraud_rings = []
    for comp in components:
        subg  = G.subgraph(comp)
        fraud_edges = [(u, v, d) for u, v, d in subg.edges(data=True) if d["is_fraud"]]
        if not fraud_edges:
            continue
        nodes_trimmed = list(comp)[:MAX_RING_NODES]
        subg_trimmed  = G.subgraph(nodes_trimmed)
        ring = {
            "nodes": [
                {
                    "id":    n,
                    "fraud": node_meta[n]["is_fraud_node"],
                    "n_tx":  node_meta[n]["n_tx"],
                }
                for n in nodes_trimmed
            ],
            "edges": [
                {
                    "source":    u,
                    "target":    v,
                    "amount":    d["amount"],
                    "type":      d["tx_type"],
                    "is_fraud":  d["is_fraud"],
                }
                for u, v, d in subg_trimmed.edges(data=True)
            ],
            "fraud_tx_count": len(fraud_edges),
            "total_tx_count": subg.number_of_edges(),
            "node_count":     len(comp),
        }
        fraud_rings.append(ring)

    fraud_rings.sort(key=lambda r: r["fraud_tx_count"], reverse=True)
    print(f"[graph] Found {len(fraud_rings)} fraud rings")

    # ── Node feature matrix for GNN ──────────────────────────────────────────
    node_list   = sorted(node_meta.keys())
    node_index  = {n: i for i, n in enumerate(node_list)}
    features    = []
    labels      = []
    for node in node_list:
        m = node_meta[node]
        # Leakage-safe features only — fraud_sent / fraud_recv excluded
        features.append([
            m["total_sent"],
            m["total_recv"],
            float(m["n_tx"]),
            float(m["out_degree"]),
            float(m["in_degree"]),
        ])
        labels.append(m["is_fraud_node"])

    # Edge index (COO format)
    edge_index = [[node_index[u], node_index[v]] for u, v in G.edges()]

    gnn_data = {
        "node_list":   node_list,
        "node_index":  node_index,
        "features":    features,
        "labels":      labels,
        "edge_index":  edge_index,
        "num_nodes":   len(node_list),
        "num_edges":   len(edge_index),
        "fraud_nodes": sum(labels),
    }

    # ── Persist ──────────────────────────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(os.path.join(DATA_DIR, "graph_nodes.json"), "w") as f:
        json.dump(list(node_meta.values()), f)

    edges_out = [
        {"source": u, "target": v, **{k: v2 for k, v2 in d.items()}}
        for u, v, d in G.edges(data=True)
    ]
    with open(os.path.join(DATA_DIR, "graph_edges.json"), "w") as f:
        json.dump(edges_out, f)

    with open(os.path.join(DATA_DIR, "fraud_rings.json"), "w") as f:
        json.dump(fraud_rings[:20], f, indent=2)   # top-20 rings

    with open(os.path.join(DATA_DIR, "gnn_features.json"), "w") as f:
        json.dump(gnn_data, f)

    print(f"[graph] Artifacts written to {DATA_DIR}")
    return G, node_meta, fraud_rings, gnn_data


if __name__ == "__main__":
    build_graph()
