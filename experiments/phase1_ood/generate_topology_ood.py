import json
import hashlib
import argparse
from pathlib import Path

import networkx as nx
import numpy as np

from scripts.generate_synthetic import save_gset, make_sf, make_sbm


def graph_hash(G):
    edges = sorted((min(u, v), max(u, v)) for u, v in G.edges())
    payload = json.dumps([G.number_of_nodes(), edges], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def generate(family, n, seed):
    if family == "SBM":
        G = make_sbm(n, 10, 82.0 / n, 2.0 / n, seed)
    elif family == "WS":
        G = nx.connected_watts_strogatz_graph(n, 10, 0.1, seed=seed)
    elif family == "GEO":
        radius = np.sqrt(10 / (np.pi * n))
        G = nx.random_geometric_graph(n, radius, seed=seed)
    elif family == "SF":
        G = make_sf(n, 5, seed)
    else:
        raise ValueError(family)

    G = nx.convert_node_labels_to_integers(nx.Graph(G))
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/phase1_ood/data_topology")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--per-family", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.out)
    manifest = root / "manifest.json"

    if manifest.exists():
        raise FileExistsError(manifest)

    rows = []
    seen = set()

    for family_id, family in enumerate(("SBM", "WS", "GEO", "SF")):
        for i in range(args.per_family):
            seed = 9300000 + family_id * 100000 + i
            G = generate(family, args.n, seed)

            digest = graph_hash(G)
            if digest in seen:
                raise RuntimeError("Duplicate graph")
            seen.add(digest)

            path = root / family / f"{family}_n{args.n}_s{seed}.txt"
            if path.exists():
                raise FileExistsError(path)

            path.parent.mkdir(parents=True, exist_ok=True)
            save_gset(G, str(path))

            rows.append({
                "split": "test_topology_ood",
                "family": family,
                "n": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "seed": seed,
                "path": str(path),
                "mean_degree": 2 * G.number_of_edges() / G.number_of_nodes(),
                "components": nx.number_connected_components(G),
                "clustering": nx.average_clustering(G),
                "graph_hash": digest,
            })

            print(f"GENERATED {family} {i+1}/{args.per_family}", flush=True)

    manifest.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"SAVED {manifest}: {len(rows)} graphs")


if __name__ == "__main__":
    main()
