import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.seq_curriculum import G, make_ec
from scripts.g1_diag import spectral


class Args:
    model = "pigatv2"
    config = "configs/maxcut_main.yaml"
    device = "cpu"
    dim = 32
    layers = 3
    heads = 1
    lr = 1e-3


def main():
    manifest = json.loads(
        Path("dataset/protocol2026_v2/manifest.json").read_text()
    )

    graphs = [
        r for r in manifest
        if r["split"] == "test_iid"
        and r["family"] == "RRG"
    ]

    assert len(graphs) == 100

    output = Path(
        "experiments/protocol2026/results/spectral_rrg_iid.csv"
    )

    if output.exists():
        raise FileExistsError(output)

    ec = make_ec(Args())

    results = []

    for i, row in enumerate(graphs, 1):
        graph = G(row["path"], ec)

        _, eigenvector = spectral(graph, 16)

        # Median threshold: spectral bipartition
        labels = torch.tensor(
            eigenvector >= np.median(eigenvector),
            dtype=torch.bool,
            device=graph.src.device
        )

        crossing = labels[graph.src] != labels[graph.dst]

        cut = float(graph.w[crossing].sum().item())
        ratio = cut / graph.wsum

        results.append({
            "path": row["path"],
            "family": row["family"],
            "n": row["n"],
            "edges": row["edges"],
            "graph_seed": row["seed"],
            "cut": cut,
            "cut_ratio": ratio
        })

        if i % 20 == 0:
            print(f"Evaluated {i}/100 graphs", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    values = np.array([r["cut_ratio"] for r in results])

    print(f"\nSpectral mean: {values.mean()*100:.3f}%")
    print(f"Spectral std:  {values.std(ddof=1)*100:.3f} pp")
    print(f"Min: {values.min()*100:.3f}%")
    print(f"Max: {values.max()*100:.3f}%")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
