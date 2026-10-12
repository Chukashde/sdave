import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.seq_curriculum import G, make_ec, build_net, fwd
from scripts.g1_diag import spectral, make_x


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cut_ratio(graph, labels):
    labels = labels.to(graph.src.device)
    crossing = labels[graph.src] != labels[graph.dst]
    return float(graph.w[crossing].sum().item() / graph.wsum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset/protocol2026_v2/manifest.json")
    ap.add_argument("--split", default="test_iid")
    ap.add_argument("--family", default="RRG")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    output = Path(args.output)

    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    manifest = json.loads(Path(args.manifest).read_text())

    graphs = [
        r for r in manifest
        if r["split"] == args.split
        and r["family"] == args.family
    ]

    graphs.sort(key=lambda r: (r["n"], r["seed"]))

    if not graphs:
        raise RuntimeError("No test graphs found")

    ckpt_dir = Path("experiments/protocol2026/checkpoints")

    checkpoints = [
        ckpt_dir / "pilot_rrg_gatv2_qubo_2k.pt",
        *[
            ckpt_dir / f"pilot_rrg_gatv2_qubo_seed{s}.pt"
            for s in range(2, 6)
        ],
    ]

    loaded = []

    for seed, path in enumerate(checkpoints, 1):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["args"]

        assert cfg["seed"] == seed
        assert cfg["family"] == args.family
        assert cfg["model"] == "pigatv2"
        assert cfg["loss"] == "qubo"
        assert ckpt["manifest_sha256"] == sha256(args.manifest)

        train_val = set(ckpt["train_paths"]) | set(ckpt["val_paths"])
        test_paths = {r["path"] for r in graphs}

        assert not (train_val & test_paths), "PATH LEAKAGE"

        hashes = {r["path"]: r["graph_hash"] for r in manifest}

        used_hashes = {hashes[p] for p in train_val}
        test_hashes = {r["graph_hash"] for r in graphs}

        assert not (used_hashes & test_hashes), "GRAPH HASH LEAKAGE"

        config_args = argparse.Namespace(**cfg)
        config_args.device = args.device

        ec = make_ec(config_args)
        net = build_net(config_args, ec)

        net.load_state_dict(ckpt["model_state"], strict=True)
        net.eval()

        loaded.append((seed, cfg, ec, net))

    fields = [
        "model", "loss", "train_seed", "split", "family",
        "path", "graph_seed", "n", "edges",
        "cut05", "rank50", "p_std"
    ]

    results = []

    # Нэг graph-ийг нэг удаа load хийж, 5 model-д ашиглана.
    for i, row in enumerate(graphs, 1):
        for seed, cfg, ec, net in loaded:
            torch.manual_seed(seed)
            np.random.seed(seed)

            with torch.no_grad():
                graph = G(row["path"], ec)

                U, _ = spectral(graph, cfg["k"])

                x = make_x(
                    graph, "clean", cfg["dim"],
                    U, None, ec
                )

                p = fwd(net, graph, x)

                labels05 = p >= 0.5

                labels_rank = torch.zeros(
                    graph.n,
                    dtype=torch.bool,
                    device=p.device
                )

                order = torch.argsort(p, stable=True)
                labels_rank[order[graph.n // 2:]] = True

                results.append({
                    "model": cfg["model"],
                    "loss": cfg["loss"],
                    "train_seed": seed,
                    "split": args.split,
                    "family": args.family,
                    "path": row["path"],
                    "graph_seed": row["seed"],
                    "n": row["n"],
                    "edges": row["edges"],
                    "cut05": cut_ratio(graph, labels05),
                    "rank50": cut_ratio(graph, labels_rank),
                    "p_std": float(p.std().item()),
                })

        if i % 10 == 0 or i == len(graphs):
            print(f"Evaluated {i}/{len(graphs)} graphs", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} rows to {output}")

    for seed in range(1, 6):
        subset = [r for r in results if r["train_seed"] == seed]
        cut = np.mean([r["cut05"] for r in subset]) * 100
        rank = np.mean([r["rank50"] for r in subset]) * 100

        print(
            f"seed={seed} "
            f"cut05={cut:.3f}% "
            f"rank50={rank:.3f}%"
        )

    means = [
        np.mean([
            r["cut05"] for r in results
            if r["train_seed"] == seed
        ]) * 100
        for seed in range(1, 6)
    ]

    print(
        f"\nIID CUT@0.5: "
        f"{np.mean(means):.3f} ± "
        f"{np.std(means, ddof=1):.3f}%"
    )


if __name__ == "__main__":
    main()
