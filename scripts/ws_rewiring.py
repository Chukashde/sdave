import argparse
import csv
import json
import os
import tempfile

import networkx as nx
import numpy as np
import torch

from scripts.seq_curriculum import G, make_ec, build_net, fwd
from scripts.g1_diag import spectral, make_x
from scripts.exact_threshold import exact_best_cut, cut_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/multi_family_seed1_step100000.pt")
    ap.add_argument("--output", default="results/ws_rewiring.csv")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--degree", type=int, default=10)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model", default="pigatv2")
    ap.add_argument("--config", default="configs/maxcut_main.yaml")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    ec = make_ec(a)
    net = build_net(a, ec)

    ckpt = torch.load(
        a.checkpoint,
        map_location=ec["device"],
        weights_only=True
    )

    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    net.load_state_dict(ckpt)
    net.eval()

    probabilities = [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
    results = []

    for prob in probabilities:
        print(f"\nRewiring probability: {prob}", flush=True)

        for seed in range(1, 6):
            graph = nx.watts_strogatz_graph(
                a.n, a.degree, prob, seed=seed
            )

            clustering = nx.average_clustering(graph)

            # Write an edge-list compatible with the project's
            # graph loader. Confirm format before full experiment.
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as tmp:
                path = tmp.name
                tmp.write(
                    f"{graph.number_of_nodes()} {graph.number_of_edges()}\n"
                )
                for u, v in graph.edges():
                    tmp.write(f"{u + 1} {v + 1} 1\n")

            try:
                Gk = G(path, ec)
                U = spectral(Gk.g, a.k)
                x = make_x(Gk, "clean", a.dim, U, None, ec)

                with torch.no_grad():
                    pred = fwd(net, Gk, x)
                    fixed = cut_at(Gk, pred, 0.5)
                    median = cut_at(
                        Gk, pred, torch.median(pred).item()
                    )

                exact, selected_fraction = exact_best_cut(Gk, pred)

                results.append({
                    "rewiring": prob,
                    "seed": seed,
                    "n": a.n,
                    "edges": graph.number_of_edges(),
                    "clustering": clustering,
                    "fixed": fixed,
                    "median": median,
                    "exact": exact,
                    "selected_fraction": selected_fraction
                })

                print(
                    f"seed={seed} "
                    f"clustering={clustering:.3f} "
                    f"fixed={fixed:.3f} "
                    f"median={median:.3f} "
                    f"exact={exact:.3f}",
                    flush=True
                )

            finally:
                os.unlink(path)

    os.makedirs(os.path.dirname(a.output), exist_ok=True)

    with open(a.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n=== SUMMARY ===")

    import pandas as pd
    df = pd.DataFrame(results)
    print(
        df.groupby("rewiring")[
            ["clustering", "fixed", "median", "exact"]
        ].agg(["mean", "std"]).round(4).to_string()
    )

    print("\nSaved:", a.output)


if __name__ == "__main__":
    main()
