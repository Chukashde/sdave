import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import networkx as nx

from scripts.generate_synthetic import save_gset


COUNTS = {
    "train": 1000,
    "val": 100,
    "test_iid": 100,
}

FAMILIES = ["RRG", "ER"]

SPLIT_SEEDS = {
    "train": 1_000_000,
    "val": 2_000_000,
    "test_iid": 3_000_000,
}

FAMILY_OFFSETS = {
    "RRG": 0,
    "ER": 100_000,
}


def generate_graph(family, n, seed):
    if family == "RRG":
        graph = nx.random_regular_graph(3, n, seed=seed)

    elif family == "ER":
        p = 4.0 / (n - 1)
        graph = nx.erdos_renyi_graph(n, p, seed=seed)

    else:
        raise ValueError(f"Unknown family: {family}")

    graph = nx.convert_node_labels_to_integers(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    return graph


def graph_hash(graph):
    edges = sorted(
        (min(u, v), max(u, v))
        for u, v in graph.edges()
    )
    payload = json.dumps(
        [graph.number_of_nodes(), edges],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="dataset/protocol2026"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate only N graphs per family per split for testing"
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    seen = set()

    for split, count in COUNTS.items():
        if args.limit is not None:
            count = min(count, args.limit)

        for family in FAMILIES:
            folder = out / split / family
            folder.mkdir(parents=True, exist_ok=True)

            for i in range(count):
                seed = (
                    SPLIT_SEEDS[split]
                    + FAMILY_OFFSETS[family]
                    + i
                )

                # Even node counts, valid for 3-regular graphs.
                n = 100 + 2 * (i % 201)

                graph = generate_graph(family, n, seed)

                assert graph.number_of_nodes() == n
                assert not nx.number_of_selfloops(graph)

                if family == "RRG":
                    assert all(
                        degree == 3
                        for _, degree in graph.degree()
                    )

                fingerprint = graph_hash(graph)

                assert fingerprint not in seen, (
                    f"Duplicate graph: {split}/{family}/{i}"
                )
                seen.add(fingerprint)

                name = f"{family}_n{n}_s{seed}.txt"
                path = folder / name

                if path.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite: {path}"
                    )

                save_gset(graph, str(path))

                degrees = [
                    degree for _, degree in graph.degree()
                ]

                manifest.append({
                    "split": split,
                    "family": family,
                    "n": n,
                    "edges": graph.number_of_edges(),
                    "seed": seed,
                    "path": str(path),
                    "mean_degree": sum(degrees) / n,
                    "components": nx.number_connected_components(graph),
                    "clustering": nx.average_clustering(graph),
                    "graph_hash": fingerprint,
                })

            print(
                f"{split:10s} {family:4s}: "
                f"{count} graphs generated"
            )

    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {manifest_path}"
        )

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    csv_path = out / "graph_statistics.csv"
    if csv_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {csv_path}"
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(manifest[0].keys())
        )
        writer.writeheader()
        writer.writerows(manifest)

    print("\n=== DATASET SUMMARY ===")

    for (split, family), count in sorted(
        Counter(
            (row["split"], row["family"])
            for row in manifest
        ).items()
    ):
        print(f"{split:10s} {family:4s}: {count}")

    print(f"\nTotal graphs: {len(manifest)}")
    print(f"Manifest: {manifest_path}")
    print(f"Statistics: {csv_path}")


if __name__ == "__main__":
    main()
