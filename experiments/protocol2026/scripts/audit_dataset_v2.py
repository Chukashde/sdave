import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

import networkx as nx
from experiments.protocol2026.scripts.generate_dataset_v2 import graph_hash


ROOT = Path("dataset/protocol2026_v2")
MANIFEST = ROOT / "manifest.json"

EXPECTED = {
    ("train", "RRG"): 1000,
    ("train", "ER"): 1000,
    ("val", "RRG"): 100,
    ("val", "ER"): 100,
    ("test_iid", "RRG"): 100,
    ("test_iid", "ER"): 100,
}


def read_gset(path):
    with open(path) as f:
        n, m = map(int, f.readline().split())

        g = nx.Graph()
        g.add_nodes_from(range(n))

        for line in f:
            u, v, w = line.split()
            u, v = int(u) - 1, int(v) - 1

            assert 0 <= u < n
            assert 0 <= v < n
            assert u != v
            assert not g.has_edge(u, v)

            g.add_edge(u, v, weight=float(w))

    assert g.number_of_edges() == m
    return g



def main():
    with open(MANIFEST) as f:
        rows = json.load(f)

    assert len(rows) == 2400

    stats = defaultdict(list)
    hashes = set()

    for row in rows:
        path = Path(row["path"])
        assert path.is_file(), path

        g = read_gset(path)
        n = g.number_of_nodes()
        m = g.number_of_edges()

        assert n == row["n"], path
        assert m == row["edges"], path
        assert 100 <= n <= 500, path
        assert n % 2 == 0, path

        if row["family"] == "RRG":
            assert all(d == 3 for _, d in g.degree()), path

        components = nx.number_connected_components(g)
        assert components == row["components"], path

        h = graph_hash(g)
        assert h == row["graph_hash"], path
        assert h not in hashes, f"Duplicate graph: {path}"
        hashes.add(h)

        stats[(row["split"], row["family"])].append({
            "n": n,
            "degree": 2 * m / n,
            "components": components,
        })

    assert set(stats) == set(EXPECTED)

    print("\n=== DATASET V2 FINAL AUDIT ===")

    for key, expected_count in sorted(EXPECTED.items()):
        values = stats[key]

        assert len(values) == expected_count

        sizes = [v["n"] for v in values]
        disconnected = sum(
            v["components"] > 1 for v in values
        )

        print(
            f"{key}: "
            f"count={len(values)}, "
            f"n={min(sizes)}-{max(sizes)}, "
            f"mean_n={sum(sizes)/len(sizes):.1f}, "
            f"mean_degree={sum(v['degree'] for v in values)/len(values):.3f}, "
            f"disconnected={disconnected}"
        )

    print(f"\nUnique graph hashes: {len(hashes)}")
    print("DATASET V2 FINAL AUDIT PASSED")


if __name__ == "__main__":
    main()
