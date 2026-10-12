import argparse
import json
import os

import networkx as nx
from scripts.seq_curriculum import load_graph
import numpy as np
import pandas as pd

from scripts.output_diagnostic import select, load_rows
from scripts.seq_curriculum import make_ec


def analyze_graph(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()

    degrees = np.array([d for _, d in G.degree()], dtype=float)

    # Undirected, unweighted structural statistics
    clustering = nx.average_clustering(G)

    components = list(nx.connected_components(G))
    largest_cc = max(len(c) for c in components) / n

    # Normalized Laplacian eigenvalues: sample smallest ones
    try:
        import scipy.sparse.linalg as sla

        L = nx.normalized_laplacian_matrix(G).astype(float)
        k = min(16, n - 2)

        eig = sla.eigsh(
            L,
            k=k,
            which="SM",
            return_eigenvectors=False,
            tol=1e-3
        )
        eig = np.sort(eig)

        lambda2 = float(eig[1]) if len(eig) > 1 else np.nan
        spectral_mean = float(np.mean(eig))

    except Exception as e:
        print("Spectral warning:", str(e))
        lambda2 = np.nan
        spectral_mean = np.nan

    return {
        "nodes": n,
        "edges": m,
        "avg_degree": degrees.mean(),
        "degree_std": degrees.std(),
        "degree_cv": degrees.std() / max(degrees.mean(), 1e-12),
        "clustering": clustering,
        "largest_cc": largest_cc,
        "lambda2": lambda2,
        "spectral_mean16": spectral_mean,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="dataset/splits/manifest.json"
    )
    parser.add_argument(
        "--output",
        default="results/topology_diagnostic.csv"
    )
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    groups = {
        "TRAIN_ER": select(manifest, "ER", "train"),
        "TRAIN_SF": select(manifest, "SF", "train"),
        "TRAIN_SBM": select(manifest, "SBM", "train"),
        "OOD_WS": select(manifest, "WS", "test_family"),
    }

    results = []

    for name, rows in groups.items():
        print(f"\n{name}: {len(rows)} graphs", flush=True)

        for i, row in enumerate(rows):
            path = row["path"]

            # Dataset files contain NetworkX graph objects.
            G, _ = load_graph(path)

            stats = analyze_graph(G)
            stats["dataset"] = name
            stats["path"] = path
            results.append(stats)

            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(rows)}", flush=True)

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    cols = [
        "avg_degree", "degree_cv", "clustering",
        "largest_cc", "lambda2", "spectral_mean16"
    ]

    print("\n=== TOPOLOGY COMPARISON ===")
    print(df.groupby("dataset")[cols].mean().round(4).to_string())
    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()
