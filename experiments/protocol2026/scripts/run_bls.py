import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

from pigat.data.loaders import load_graph
from scripts.bls_baseline import bls, cut_value, total_edge_weight


FIELDS = [
    "split", "family", "n", "edges",
    "path", "graph_seed", "bls_seed",
    "cut", "cut_ratio", "runtime_sec",
    "iterations", "local_optima", "improvements",
]

SOLVER_PARAMS = {
    "L0": None,
    "T": 1000,
    "P0": 0.8,
    "Q": 0.5,
}


def sha256(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--manifest",
        default="dataset/protocol2026_v2/manifest.json"
    )
    ap.add_argument("--split", default="test_iid")
    ap.add_argument(
        "--families", nargs="+", default=["RRG", "ER"]
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--seeds", nargs="+", type=int,
        default=[1, 2, 3, 4, 5]
    )
    ap.add_argument("--time_limit", type=float, default=30.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--output",
        default="experiments/protocol2026/results/bls_v2_iid.csv"
    )

    args = ap.parse_args()

    if args.time_limit <= 0:
        ap.error("--time_limit must be positive")
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")
    if len(args.seeds) != len(set(args.seeds)):
        ap.error("--seeds must be unique")
    if len(args.families) != len(set(args.families)):
        ap.error("--families must be unique")

    with open(args.manifest) as f:
        manifest = json.load(f)

    selected = [
        row for row in manifest
        if row["split"] == args.split
        and row["family"] in args.families
    ]

    if args.limit is not None:
        selected = [
            row
            for family in args.families
            for row in [
                r for r in selected if r["family"] == family
            ][:args.limit]
        ]

    if not selected:
        raise ValueError("No graphs selected")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(str(output) + ".meta.json")

    config = {
        "manifest_sha256": sha256(args.manifest),
        "split": args.split,
        "families": args.families,
        "limit": args.limit,
        "seeds": args.seeds,
        "time_limit": args.time_limit,
        "solver_params": SOLVER_PARAMS,
        "selected_graphs": [
            {
                "path": r["path"],
                "graph_hash": r["graph_hash"],
            }
            for r in selected
        ],
        "solver_sha256": sha256("scripts/bls_baseline.py"),
    }

    completed = set()

    if output.exists():
        if not args.resume:
            raise FileExistsError(
                f"{output} exists. Use --resume."
            )

        if not metadata_path.exists():
            raise RuntimeError(
                "Missing metadata; cannot safely resume."
            )

        saved_config = json.loads(
            metadata_path.read_text()
        )

        if saved_config != config:
            raise RuntimeError(
                "Configuration changed. "
                "Use a different output filename."
            )

        with open(output, newline="") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames != FIELDS:
                raise RuntimeError(
                    "CSV header mismatch."
                )

            for row in reader:
                key = (
                    row["path"],
                    int(row["bls_seed"])
                )

                if key in completed:
                    raise RuntimeError(
                        f"Duplicate result: {key}"
                    )

                completed.add(key)

        print(
            f"RESUME: {len(completed)} completed runs",
            flush=True
        )

    else:
        if metadata_path.exists():
            raise RuntimeError(
                "Metadata exists without CSV. "
                "Choose another output path."
            )

        metadata_path.write_text(
            json.dumps(config, indent=2)
        )

        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=FIELDS
            )
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())

    expected_keys = {
        (row["path"], seed)
        for row in selected
        for seed in args.seeds
    }

    if not completed.issubset(expected_keys):
        raise RuntimeError(
            "CSV contains unexpected graph/seed pairs."
        )

    total = len(expected_keys)
    start = time.perf_counter()

    with open(output, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=FIELDS
        )

        for row in selected:
            pending = [
                seed for seed in args.seeds
                if (row["path"], seed) not in completed
            ]

            if not pending:
                continue

            graph, _ = load_graph(row["path"])
            wsum = total_edge_weight(graph)

            for seed in pending:
                result = bls(
                    graph,
                    seed=seed,
                    time_limit=args.time_limit,
                    **SOLVER_PARAMS,
                )

                verified = cut_value(
                    graph, result["best_x"]
                )

                assert abs(
                    result["best_cut"] - verified
                ) < 1e-8, (
                    f"BLS verification failed: "
                    f"{row['path']}"
                )

                writer.writerow({
                    "split": row["split"],
                    "family": row["family"],
                    "n": row["n"],
                    "edges": row["edges"],
                    "path": row["path"],
                    "graph_seed": row["seed"],
                    "bls_seed": seed,
                    "cut": verified,
                    "cut_ratio": (
                        verified / wsum if wsum else 0
                    ),
                    "runtime_sec": result["runtime_sec"],
                    "iterations": result["iterations"],
                    "local_optima": result["local_optima"],
                    "improvements": result["improvements"],
                })

                f.flush()
                os.fsync(f.fileno())

                completed.add((row["path"], seed))

                print(
                    f"[{len(completed)}/{total}] "
                    f"{row['family']} n={row['n']} "
                    f"seed={seed} "
                    f"cut={verified:.1f} "
                    f"time={result['runtime_sec']:.2f}s",
                    flush=True,
                )

    print(f"\nSaved: {output}")
    print(
        f"Completed: {len(completed)}/{total}"
    )
    print(
        f"Elapsed: "
        f"{time.perf_counter() - start:.1f}s"
    )


if __name__ == "__main__":
    main()
