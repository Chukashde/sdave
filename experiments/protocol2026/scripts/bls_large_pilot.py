import csv
import json

from pigat.data.loaders import load_graph
from scripts.bls_baseline import bls, cut_value

with open("dataset/protocol2026/manifest.json") as f:
    manifest = json.load(f)

rows = [
    next(
        r for r in manifest
        if r["split"] == "train"
        and r["family"] == family
        and r["n"] == 500
    )
    for family in ["RRG", "ER"]
]

output = "experiments/protocol2026/results/bls_large_pilot.csv"

with open(output, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "family", "n", "path", "seed",
            "time_limit", "cut", "runtime_sec"
        ]
    )
    writer.writeheader()

    for row in rows:
        graph, _ = load_graph(row["path"])

        for seconds in [5, 15, 30]:
            for seed in [1, 2]:
                result = bls(
                    graph,
                    seed=seed,
                    time_limit=seconds,
                    L0=None,
                    T=1000,
                    P0=0.8,
                    Q=0.5,
                )

                cut = cut_value(graph, result["best_x"])

                assert abs(
                    cut - result["best_cut"]
                ) < 1e-8

                writer.writerow({
                    "family": row["family"],
                    "n": row["n"],
                    "path": row["path"],
                    "seed": seed,
                    "time_limit": seconds,
                    "cut": cut,
                    "runtime_sec": result["runtime_sec"],
                })
                f.flush()

                print(
                    f"{row['family']} n={row['n']} "
                    f"time={seconds}s seed={seed} "
                    f"cut={cut}",
                    flush=True
                )

print("Saved:", output)
