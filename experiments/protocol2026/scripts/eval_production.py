import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.seq_curriculum import G, make_ec, build_net, fwd
from experiments.protocol2026.scripts.train_generalization import load_graphs


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cut_value(graph, bits):
    return float(
        (graph.w * (bits[graph.src] != bits[graph.dst])).sum().item()
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset/protocol2026_v2/manifest.json")
    ap.add_argument("--checkpoint-dir", default="experiments/protocol2026/checkpoints/production")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default="test_iid")
    args = ap.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    manifest = json.loads(Path(args.manifest).read_text())
    manifest_hash = sha256(args.manifest)

    models = ["pigcn_jk", "pigat_jk", "pigatv2_jk"]
    families = ["RRG", "ER"]
    checkpoint_dir = Path(args.checkpoint_dir)

    expected = [
        (model, family, seed)
        for model in models
        for family in families
        for seed in range(1, 6)
    ]

    # Бүх checkpoint бэлэн болсон эсэхийг шалгана.
    missing = [
        f"{model}_{family}_seed{seed}.pt"
        for model, family, seed in expected
        if not (checkpoint_dir / f"{model}_{family}_seed{seed}.pt").exists()
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)} checkpoints missing. "
            "Wait for production training to finish."
        )

    test_rows = {
        family: sorted(
            [
                r for r in manifest
                if r["split"] == args.split and r["family"] == family
            ],
            key=lambda r: (r["n"], r["seed"])
        )
        for family in families
    }

    for family in families:
        if len(test_rows[family]) != 100:
            raise RuntimeError(
                f"Expected 100 {family} test graphs; "
                f"found {len(test_rows[family])}"
            )

    graph_hashes = {r["path"]: r["graph_hash"] for r in manifest}
    results = []

    for model, train_family, seed in expected:
        name = f"{model}_{train_family}_seed{seed}"
        path = checkpoint_dir / f"{name}.pt"

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["args"]

        assert cfg["model"] == model
        assert cfg["family"] == train_family
        assert int(cfg["seed"]) == seed
        assert cfg["loss"] == "qubo"
        assert cfg["feature_mode"] == "random"
        assert int(cfg["dim"]) == 17
        assert ckpt["manifest_sha256"] == manifest_hash

        train_val = set(ckpt["train_paths"]) | set(ckpt["val_paths"])
        train_val_hashes = {graph_hashes[p] for p in train_val}

        cfg = argparse.Namespace(**cfg)
        cfg.device = args.device

        ec = make_ec(cfg)
        net = build_net(cfg, ec)
        net.load_state_dict(ckpt["model_state"], strict=True)
        net.eval()

        for test_family in families:
            rows = test_rows[test_family]
            test_paths = {r["path"] for r in rows}
            test_hashes = {r["graph_hash"] for r in rows}

            assert not (train_val & test_paths), "PATH LEAKAGE"
            assert not (train_val_hashes & test_hashes), "GRAPH HASH LEAKAGE"

            # Яг training-ийн deterministic feature generation.
            graphs = load_graphs(rows, ec, cfg.k, cfg)

            with torch.no_grad():
                for row, graph, x in graphs:
                    p = fwd(net, graph, x)

                    b05 = p >= 0.5
                    bmedian = p >= p.median()

                    order = torch.argsort(p, stable=True)
                    brank = torch.zeros_like(p, dtype=torch.bool)
                    brank[order[graph.n // 2:]] = True

                    denominator = graph.w.abs().sum().clamp_min(1e-12).item()

                    results.append({
                        "model": model,
                        "train_family": train_family,
                        "test_family": test_family,
                        "train_seed": seed,
                        "split": args.split,
                        "path": row["path"],
                        "graph_seed": row["seed"],
                        "n": row["n"],
                        "edges": row["edges"],
                        "regime": "iid" if train_family == test_family else "cross_family",
                        "cut05": cut_value(graph, b05),
                        "cutmedian": cut_value(graph, bmedian),
                        "cutrank50": cut_value(graph, brank),
                        "cut05_normalized": cut_value(graph, b05) / denominator,
                        "cutmedian_normalized": cut_value(graph, bmedian) / denominator,
                        "cutrank50_normalized": cut_value(graph, brank) / denominator,
                        "p_std": float(p.std().item()),
                    })

            print(
                f"DONE {name} -> {test_family}: {len(rows)} graphs",
                flush=True
            )

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"ZERO-SHOT EVALUATION COMPLETED: {len(results)} rows")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
