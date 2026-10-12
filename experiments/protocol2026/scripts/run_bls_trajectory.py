import argparse
import csv
import json
from pathlib import Path

from pigat.data.loaders import load_graph
from experiments.protocol2026.scripts.bls_trajectory import bls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--time_limit", type=float, default=30.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.time_limit <= 0:
        ap.error("--time_limit must be positive")

    graph, _ = load_graph(args.graph)

    result = bls(
        graph,
        seed=args.seed,
        time_limit=args.time_limit
    )

    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed_sec", "best_cut"])
        writer.writerows(result["trajectory"])

    print(json.dumps({
        "graph": args.graph,
        "seed": args.seed,
        "best_cut": result["best_cut"],
        "runtime_sec": result["runtime_sec"],
        "trajectory_points": len(result["trajectory"]),
        "output": str(out)
    }, indent=2))


if __name__ == "__main__":
    main()
