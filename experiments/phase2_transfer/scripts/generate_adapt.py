import json
import argparse
from pathlib import Path

import networkx as nx

from scripts.generate_synthetic import save_gset
from experiments.phase1_ood.generate_topology_ood import (
    generate, graph_hash
)

FAMILIES = ("SBM", "WS", "GEO", "SF")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="experiments/phase2_transfer/data_adapt"
    )
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--per-family", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.out)
    manifest = root / "manifest.json"

    if manifest.exists():
        raise FileExistsError(manifest)

    existing_hashes = set()

    for existing in (
        "dataset/protocol2026_v2/manifest.json",
        "experiments/phase1_ood/data_topology_v2/manifest.json",
    ):
        with open(existing) as f:
            existing_hashes.update(
                r["graph_hash"] for r in json.load(f)
            )

    rows = []

    for family_id, family in enumerate(FAMILIES):
        for i in range(args.per_family):
            seed = 9500000 + family_id * 100000 + i
            G = generate(family, args.n, seed)
            h = graph_hash(G)

            if h in existing_hashes:
                raise RuntimeError(f"Graph leakage: {family} {seed}")

            existing_hashes.add(h)

            path = root / family / f"{family}_n{args.n}_s{seed}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)

            if path.exists():
                raise FileExistsError(path)

            save_gset(G, str(path))

            rows.append({
                "split": "adapt_topology",
                "family": family,
                "n": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "seed": seed,
                "path": str(path),
                "mean_degree": 2 * G.number_of_edges() / G.number_of_nodes(),
                "components": nx.number_connected_components(G),
                "clustering": nx.average_clustering(G),
                "graph_hash": h,
            })

            print(f"GENERATED {family} {i+1}/{args.per_family}",
                  flush=True)

    manifest.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"SAVED {manifest}: {len(rows)} graphs")

if __name__ == "__main__":
    main()
