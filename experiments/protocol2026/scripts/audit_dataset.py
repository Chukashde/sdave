import json
from collections import defaultdict
from pathlib import Path

import networkx as nx


def read_gset(path):
    with open(path) as f:
        n, m = map(int, f.readline().split())

        graph = nx.Graph()
        graph.add_nodes_from(range(n))

        for line in f:
            u, v, w = line.split()
            graph.add_edge(int(u) - 1, int(v) - 1, weight=float(w))

    assert graph.number_of_edges() == m, path
    return graph


def main():
    with open("dataset/protocol2026/manifest.json") as f:
        rows = json.load(f)

    stats = defaultdict(list)

    for row in rows:
        path = Path(row["path"])
        assert path.exists(), path

        graph = read_gset(path)

        n = graph.number_of_nodes()
        m = graph.number_of_edges()

        assert n == row["n"]
        assert m == row["edges"]
        assert nx.number_of_selfloops(graph) == 0

        if row["family"] == "RRG":
            assert all(d == 3 for _, d in graph.degree())

        components = nx.number_connected_components(graph)
        assert components == row["components"]

        stats[(row["split"], row["family"])].append({
            "n": n,
            "edges": m,
            "components": components,
            "degree": 2 * m / n,
        })

    print("\n=== DATASET INTEGRITY REPORT ===")

    for key, values in sorted(stats.items()):
        count = len(values)
        disconnected = sum(
            x["components"] > 1 for x in values
        )
        mean_degree = sum(
            x["degree"] for x in values
        ) / count

        print(
            f"{key}: graphs={count}, "
            f"mean_degree={mean_degree:.4f}, "
            f"disconnected={disconnected}"
        )

    print("\nALL DATASET FILES VALIDATED")


if __name__ == "__main__":
    main()
