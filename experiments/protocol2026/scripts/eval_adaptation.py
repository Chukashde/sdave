import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from scripts.seq_curriculum import make_ec, build_net, fwd
from experiments.protocol2026.scripts.train_generalization import (
    load_graphs, qubo_loss
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(net, graph, x):
    net.eval()
    with torch.no_grad():
        p = fwd(net, graph, x)
        bits = p >= 0.5
        return float(
            (graph.w * (bits[graph.src] != bits[graph.dst]))
            .sum().item()
        )


def run(net, graph, x, steps, lr, weight_decay):
    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, weight_decay=weight_decay
    )
    results = []
    start = time.perf_counter()

    for step in range(1, max(steps) + 1):
        net.train()
        p = fwd(net, graph, x)
        loss = qubo_loss(p, graph)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step in steps:
            if x.is_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            cut = evaluate(net, graph, x)
            results.append((step, cut, elapsed))

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset/protocol2026_v2/manifest.json")
    ap.add_argument("--checkpoint-dir", default="experiments/protocol2026/checkpoints/production")
    ap.add_argument("--reference", default="experiments/protocol2026/results/production/bls_test_reference_200.csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--graph-limit", type=int, default=None)
    ap.add_argument("--checkpoint-limit", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--family", choices=["RRG", "ER"], default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, nargs="+", default=[10, 100, 1000])
    args = ap.parse_args()

    if sorted(set(args.steps)) != sorted(args.steps) or min(args.steps) < 1:
        raise ValueError("Steps must be positive and unique")

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text())
    tests = sorted(
        (r for r in manifest if r["split"] == "test_iid"),
        key=lambda r: (r["family"], r["n"], r["seed"])
    )
    if len(tests) != 200:
        raise RuntimeError(f"Expected 200 test graphs, got {len(tests)}")

    with open(args.reference, newline="") as f:
        refs = {r["path"]: float(r["reference_cut"])
                for r in csv.DictReader(f)}

    if set(refs) != {r["path"] for r in tests}:
        raise RuntimeError("Test reference coverage mismatch")

    if args.graph_limit is not None:
        tests = tests[:args.graph_limit]

    jobs = [
        (model, family, seed)
        for model in ("pigcn_jk", "pigat_jk", "pigatv2_jk")
        for family in ("RRG", "ER")
        for seed in range(1, 6)
    ]
    if args.model is not None:
        jobs = [j for j in jobs if j[0] == args.model]
    if args.family is not None:
        jobs = [j for j in jobs if j[1] == args.family]
    if args.seed is not None:
        jobs = [j for j in jobs if j[2] == args.seed]
    if args.checkpoint_limit is not None:
        jobs = jobs[:args.checkpoint_limit]
    if not jobs:
        raise RuntimeError("No checkpoints selected")

    fields = [
        "model", "train_family", "test_family", "train_seed",
        "path", "regime", "steps", "cut05", "reference_cut",
        "cut_over_bls", "adaptation_sec"
    ]

    with output.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for model, family, seed in jobs:
            checkpoint = (
                Path(args.checkpoint_dir) /
                f"{model}_{family}_seed{seed}.pt"
            )
            ckpt = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            cfg = ckpt["args"]

            assert cfg["model"] == model
            assert cfg["family"] == family
            assert int(cfg["seed"]) == seed
            assert cfg["loss"] == "qubo"
            assert cfg["feature_mode"] == "random"
            assert ckpt["manifest_sha256"] == digest(args.manifest)

            train_val = set(ckpt["train_paths"]) | set(ckpt["val_paths"])
            selected_tests = {r["path"] for r in tests}
            assert not train_val & selected_tests

            ns = argparse.Namespace(**cfg)
            ns.device = args.device
            ec = make_ec(ns)

            for row in tests:
                graph_row, graph, x = load_graphs(
                    [row], ec, ns.k, ns
                )[0]

                for regime in ("fine_tune", "from_scratch"):
                    # Deterministic initialization for scratch runs.
                    init_key = (
                        f"{model}:{family}:{seed}:"
                        f"{row['graph_hash']}:{regime}"
                    )
                    init_seed = int(
                        hashlib.sha256(init_key.encode()).hexdigest()[:8],
                        16
                    )
                    random.seed(init_seed)
                    np.random.seed(init_seed)
                    torch.manual_seed(init_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(init_seed)

                    net = build_net(ns, ec)

                    if regime == "fine_tune":
                        net.load_state_dict(
                            ckpt["model_state"], strict=True
                        )

                    results = run(
                        net, graph, x, args.steps,
                        ns.lr, ns.weight_decay
                    )

                    for step, cut, elapsed in results:
                        writer.writerow({
                            "model": model,
                            "train_family": family,
                            "test_family": row["family"],
                            "train_seed": seed,
                            "path": row["path"],
                            "regime": regime,
                            "steps": step,
                            "cut05": cut,
                            "reference_cut": refs[row["path"]],
                            "cut_over_bls": cut / refs[row["path"]],
                            "adaptation_sec": elapsed,
                        })

                    f.flush()

                del graph, x

            print(
                f"DONE {model}_{family}_seed{seed}",
                flush=True
            )

    print("ADAPTATION COMPLETED:", output)


if __name__ == "__main__":
    main()
