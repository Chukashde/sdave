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

MODELS = ("pigcn_jk", "pigat_jk", "pigatv2_jk")
SOURCES = ("ER", "RRG")
TARGETS = ("SBM", "WS", "GEO", "SF")

FIELDS = [
    "model", "train_family", "train_seed", "test_family",
    "adapt_graphs", "steps", "path", "graph_seed",
    "n", "edges", "cut05", "cutrank50",
    "cut05_normalized", "cutrank50_normalized",
    "adaptation_sec"
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(net, test_graphs):
    net.eval()
    results = []

    with torch.no_grad():
        for row, graph, x in test_graphs:
            p = fwd(net, graph, x).reshape(-1)

            if p.numel() != graph.n:
                raise RuntimeError("Unexpected model output size")

            bits05 = p >= 0.5

            # Same stable ordering principle as Phase1.
            order = torch.argsort(p, stable=True)
            bits_rank = torch.zeros(
                graph.n, device=p.device, dtype=torch.bool
            )
            bits_rank[order[graph.n // 2:]] = True

            def cut(bits):
                return float((
                    graph.w *
                    (bits[graph.src] != bits[graph.dst])
                ).sum().item())

            c05 = cut(bits05)
            crank = cut(bits_rank)
            denom = float(graph.w.abs().sum().item())

            results.append((row, c05, crank, denom))

    return results


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--adapt-manifest",
        default="experiments/phase2_transfer/data_adapt/manifest.json"
    )
    ap.add_argument(
        "--test-manifest",
        default="experiments/phase1_ood/data_topology_v2/manifest.json"
    )
    ap.add_argument(
        "--phase0-manifest",
        default="dataset/protocol2026_v2/manifest.json"
    )
    ap.add_argument(
        "--checkpoint-dir",
        default="experiments/protocol2026/checkpoints/production"
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--source", choices=SOURCES)
    ap.add_argument("--target", choices=TARGETS)
    ap.add_argument("--seed", type=int)
    ap.add_argument(
        "--graph-counts", type=int, nargs="+", default=[1, 5, 20]
    )
    ap.add_argument(
        "--steps", type=int, nargs="+", default=[10, 100, 1000]
    )
    ap.add_argument("--test-limit", type=int)
    args = ap.parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("CUDA unavailable")

    if any(x <= 0 or x > 20 for x in args.graph_counts):
        raise ValueError("graph-counts must be between 1 and 20")

    if any(s <= 0 for s in args.steps):
        raise ValueError("steps must be positive")

    counts = sorted(set(args.graph_counts))
    steps = sorted(set(args.steps))

    adapt_rows = json.loads(Path(args.adapt_manifest).read_text())
    test_rows = json.loads(Path(args.test_manifest).read_text())
    phase0_rows = json.loads(Path(args.phase0_manifest).read_text())

    assert len(adapt_rows) == 80
    assert len(test_rows) == 40
    assert all(r["split"] == "adapt_topology" for r in adapt_rows)
    assert all(r["split"] == "test_topology_ood" for r in test_rows)

    a_hashes = {r["graph_hash"] for r in adapt_rows}
    t_hashes = {r["graph_hash"] for r in test_rows}
    p_hashes = {r["graph_hash"] for r in phase0_rows}

    assert len(a_hashes) == 80
    assert len(t_hashes) == 40
    assert not (a_hashes & t_hashes)
    assert not (a_hashes & p_hashes)
    assert not (t_hashes & p_hashes)

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    jobs = [
        (model, source, seed)
        for model in MODELS
        for source in SOURCES
        for seed in range(1, 6)
        if (args.model is None or model == args.model)
        and (args.source is None or source == args.source)
        and (args.seed is None or seed == args.seed)
    ]

    targets = [
        family for family in TARGETS
        if args.target is None or family == args.target
    ]

    with output.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for model, source, seed in jobs:
            ckpt_path = (
                Path(args.checkpoint_dir) /
                f"{model}_{source}_seed{seed}.pt"
            )

            ckpt = torch.load(
                ckpt_path, map_location="cpu", weights_only=False
            )
            cfg = ckpt["args"]

            assert cfg["model"] == model
            assert cfg["family"] == source
            assert int(cfg["seed"]) == seed
            assert cfg["feature_mode"] == "random"
            assert cfg["loss"] == "qubo"
            assert ckpt["manifest_sha256"] == sha256(
                args.phase0_manifest
            )

            ns = argparse.Namespace(**cfg)
            ns.device = args.device
            ec = make_ec(ns)

            for target in targets:
                train_rows = sorted(
                    (r for r in adapt_rows if r["family"] == target),
                    key=lambda r: r["seed"]
                )
                eval_rows = sorted(
                    (r for r in test_rows if r["family"] == target),
                    key=lambda r: r["seed"]
                )

                assert len(train_rows) == 20
                assert len(eval_rows) == 10

                if args.test_limit is not None:
                    eval_rows = eval_rows[:args.test_limit]

                # Preserve Phase0 deterministic feature generation.
                train_graphs = load_graphs(
                    train_rows[:max(counts)], ec, ns.k, ns
                )
                test_graphs = load_graphs(
                    eval_rows, ec, ns.k, ns
                )

                for graph_count in counts:
                    selected = train_graphs[:graph_count]

                    # Every graph-count condition starts from
                    # the same pretrained checkpoint.
                    init_key = (
                        f"phase2:{model}:{source}:{seed}:"
                        f"{target}:{graph_count}"
                    )
                    init_seed = int(
                        hashlib.sha256(init_key.encode())
                        .hexdigest()[:8], 16
                    )
                    seed_everything(init_seed)

                    net = build_net(ns, ec)
                    net.load_state_dict(
                        ckpt["model_state"], strict=True
                    )

                    optimizer = torch.optim.Adam(
                        net.parameters(),
                        lr=ns.lr,
                        weight_decay=ns.weight_decay
                    )

                    start = time.perf_counter()

                    for step in range(1, max(steps) + 1):
                        # Deterministic round-robin sampling.
                        _, graph, x = selected[
                            (step - 1) % graph_count
                        ]

                        net.train()
                        p = fwd(net, graph, x)
                        loss = qubo_loss(p, graph)

                        if not torch.isfinite(loss):
                            raise RuntimeError(
                                f"Non-finite loss at step {step}"
                            )

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        if step not in steps:
                            continue

                        if args.device == "cuda":
                            torch.cuda.synchronize()

                        elapsed = time.perf_counter() - start
                        evaluated = evaluate(net, test_graphs)

                        for row, c05, crank, denom in evaluated:
                            writer.writerow({
                                "model": model,
                                "train_family": source,
                                "train_seed": seed,
                                "test_family": target,
                                "adapt_graphs": graph_count,
                                "steps": step,
                                "path": row["path"],
                                "graph_seed": row["seed"],
                                "n": row["n"],
                                "edges": row["edges"],
                                "cut05": c05,
                                "cutrank50": crank,
                                "cut05_normalized": c05 / denom,
                                "cutrank50_normalized": crank / denom,
                                "adaptation_sec": elapsed,
                            })

                        f.flush()

                    del net, optimizer

                del train_graphs, test_graphs

                print(
                    f"DONE {model} {source} seed={seed} "
                    f"target={target}",
                    flush=True
                )

    print("COMPLETED:", output)


if __name__ == "__main__":
    main()
