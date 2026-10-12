import csv
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np

from pigat.data.loaders import load_graph


manifest = Path("dataset/protocol2026_v2/manifest.json")
output = Path("experiments/protocol2026/results/graph_covariates.csv")

rows = json.loads(manifest.read_text())

fields = [
    "path", "split", "family", "n", "edges",
    "mean_degree", "degree_cv", "density",
    "clustering", "components",
    "diameter", "largest_component_diameter"
]

if output.exists():
    raise FileExistsError(f"Output already exists: {output}")

output.parent.mkdir(parents=True, exist_ok=True)

with output.open("x", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for i, row in enumerate(rows, 1):
        g, _ = load_graph(row["path"])

        n = g.number_of_nodes()
        m = g.number_of_edges()

        degrees = np.array(
            [d for _, d in g.degree()],
            dtype=float
        )

        mean_degree = float(degrees.mean()) if n else 0.0
        degree_cv = (
            float(degrees.std() / mean_degree)
            if mean_degree > 0 else 0.0
        )

        components = nx.number_connected_components(g)

        if n == 0:
            diameter = None
            largest_diameter = None
        elif components == 1:
            diameter = nx.diameter(g)
            largest_diameter = diameter
        else:
            diameter = None
            largest = max(
                nx.connected_components(g),
                key=len
            )
            largest_diameter = nx.diameter(g.subgraph(largest))

        writer.writerow({
            "path": row["path"],
            "split": row["split"],
            "family": row["family"],
            "n": n,
            "edges": m,
            "mean_degree": mean_degree,
            "degree_cv": degree_cv,
            "density": nx.density(g),
            "clustering": nx.average_clustering(g),
            "components": components,
            "diameter": diameter,
            "largest_component_diameter": largest_diameter
        })

        if i % 100 == 0:
            print(f"Processed {i}/{len(rows)}", flush=True)

print("Saved:", output)
