import csv
import json
import os
import argparse
import numpy as np

from pigat.data.loaders import load_graph
from bls_baseline import bls, cut_value, total_edge_weight


def load_manifest(path):
    with open(path, "r") as f:
        return json.load(f)


def select_graphs(manifest):
    rows = []

    for r in manifest:
        family = r["family"]
        split = r["split"]
        n = r["n"]

        # IID SBM
        if family == "SBM" and split == "test_iid":
            rows.append(r)

        # Size OOD SBM
        elif family == "SBM" and split == "test_size":
            rows.append(r)

        # Topology OOD
        elif family in ["RR", "WS", "GEO"] and split == "test_family":
            rows.append(r)

    return rows


def dataset_name(row):
    family = row["family"]
    split = row["split"]
    n = row["n"]

    if family == "SBM" and split == "test_iid":
        return "IID_SBM"

    if family == "SBM" and split == "test_size":
        return f"SIZE_{n}"

    if split == "test_family":
        return f"OOD_{family}"

    return f"{family}_{split}_{n}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="dataset/splits/manifest.json"
    )

    parser.add_argument(
        "--time_limit",
        type=float,
        default=60.0
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5]
    )

    parser.add_argument(
        "--output",
        default="results/bls_results.csv"
    )

    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    graphs = select_graphs(manifest)

    os.makedirs(
        os.path.dirname(args.output),
        exist_ok=True
    )

    print()
    print("========================================")
    print("BLS BATCH EXPERIMENT")
    print("========================================")
    print("Graphs     :", len(graphs))
    print("Seeds      :", args.seeds)
    print("Time/Run   :", args.time_limit, "sec")
    print("Total Runs :", len(graphs) * len(args.seeds))
    print("Output     :", args.output)
    print("========================================")
    print()

    results = []

    total_runs = len(graphs) * len(args.seeds)
    run_number = 0

    for graph_index, row in enumerate(graphs, start=1):

        path = row["path"]
        name = dataset_name(row)

        print()
        print("########################################")
        print(
            f"GRAPH {graph_index}/{len(graphs)}"
        )
        print("Dataset :", name)
        print("Path    :", path)
        print("########################################")

        g, info = load_graph(path)

        wsum = total_edge_weight(g)

        for seed in args.seeds:

            run_number += 1

            print()
            print(
                f"[RUN {run_number}/{total_runs}] "
                f"{name} | seed={seed}"
            )

            result = bls(
                g,
                seed=seed,
                time_limit=args.time_limit
            )

            best_cut = result["best_cut"]

            verified_cut = cut_value(
                g,
                result["best_x"]
            )

            assert abs(
                best_cut - verified_cut
            ) < 1e-8, (
                f"Cut mismatch: "
                f"{best_cut} vs {verified_cut}"
            )

            ratio = best_cut / wsum

            print(
                f"cut={best_cut:.1f} | "
                f"ratio={ratio * 100:.2f}% | "
                f"time={result['runtime_sec']:.2f}s"
            )

            record = {
                "dataset": name,
                "family": row["family"],
                "split": row["split"],
                "n": row["n"],
                "edges": row["edges"],
                "graph_seed": row["seed"],
                "path": path,
                "bls_seed": seed,
                "cut": best_cut,
                "cut_ratio": ratio,
                "runtime_sec": result["runtime_sec"],
                "iterations": result["iterations"],
                "local_optima": result["local_optima"],
                "improvements": result["improvements"],
                "A1": result["A1"],
                "A2": result["A2"],
                "B": result["B"]
            }

            results.append(record)

            # Save after every run.
            # If SSH disconnects, completed results remain.
            with open(
                args.output,
                "w",
                newline=""
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=record.keys()
                )

                writer.writeheader()
                writer.writerows(results)

    print()
    print("========================================")
    print("EXPERIMENT COMPLETE")
    print("========================================")
    print("Runs:", len(results))
    print("Saved:", args.output)
    print()

    # --------------------------------------------
    # Summary
    # --------------------------------------------

    datasets = sorted(
        set(r["dataset"] for r in results)
    )

    print("SUMMARY")
    print("----------------------------------------")

    for name in datasets:

        values = [
            r["cut_ratio"] * 100.0
            for r in results
            if r["dataset"] == name
        ]

        mean = np.mean(values)
        std = np.std(
            values,
            ddof=1
        ) if len(values) > 1 else 0.0

        print(
            f"{name:12s} "
            f"{mean:6.2f} ± {std:5.2f}% "
            f"(n={len(values)})"
        )

    print("========================================")


if __name__ == "__main__":
    main()
