import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.seq_curriculum import make_ec, build_net, fwd
from experiments.protocol2026.scripts.train_generalization import load_graphs


ROOT = Path("experiments/phase1_ood")
ORIGINAL = Path("dataset/protocol2026_v2/manifest.json")
DENSITY = ROOT / "data_density/manifest.json"
CKPTS = Path("experiments/protocol2026/checkpoints/production")
OUTPUT = ROOT / "results/threshold_calibration.csv"

THRESHOLDS = [round(i / 100, 2) for i in range(5, 96, 5)]
MODELS = ("pigcn_jk", "pigat_jk", "pigatv2_jk")
FAMILIES = ("ER", "RRG")

FIELDS = [
    "model", "train_family", "train_seed",
    "test_family", "target_degree", "path", "graph_seed",
    "split", "method", "threshold",
    "cut", "cut_edges", "frac_positive",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cut(graph, bits):
    return (
        graph.w * (bits[graph.src] != bits[graph.dst])
    ).sum().item()


def evaluate(net, ec, cfg, row):
    _, graph, x = load_graphs([row], ec, cfg.k, cfg)[0]

    with torch.inference_mode():
        p = fwd(net, graph, x).reshape(-1)

        if not torch.isfinite(p).all():
            raise RuntimeError(f"Non-finite probabilities: {row['path']}")

        denom = graph.w.abs().sum().item()

        # Fixed threshold candidates.
        fixed = []
        for t in THRESHOLDS:
            bits = p >= t
            fixed.append((t, cut(graph, bits)))

        # Existing decoding methods.
        median_bits = p >= p.median()

        order = torch.argsort(p, stable=True)
        rank_bits = torch.zeros_like(p, dtype=torch.bool)
        rank_bits[order[graph.n // 2:]] = True

        methods = {
            "fixed05": (0.5, p >= 0.5),
            "median": (float(p.median().item()), median_bits),
            "rank50": (float("nan"), rank_bits),
        }

        result = {}
        for name, (t, bits) in methods.items():
            c = cut(graph, bits)
            result[name] = (
                t, c, c / max(denom, 1e-12),
                bits.float().mean().item()
            )

        # Only diagnostics: best test-graph threshold.
        best_t, best_c = max(
            fixed, key=lambda item: (item[1], -item[0])
        )
        result["oracle"] = (
            best_t, best_c,
            best_c / max(denom, 1e-12),
            (p >= best_t).float().mean().item()
        )

    result["_scores"] = (
        p.detach().cpu(),
        graph.src.detach().cpu(),
        graph.dst.detach().cpu(),
        graph.w.detach().cpu(),
        denom
    )
    return result, fixed


def write_row(writer, model, family, seed, row, split,
              method, threshold, cut_value, ratio, balance):
    writer.writerow({
        "model": model,
        "train_family": family,
        "train_seed": seed,
        "test_family": row["family"],
        "target_degree": row.get("target_degree", ""),
        "path": row["path"],
        "graph_seed": row["seed"],
        "split": split,
        "method": method,
        "threshold": threshold,
        "cut": cut_value,
        "cut_edges": ratio,
        "frac_positive": balance,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint-limit", type=int, default=0)
    ap.add_argument("--test-limit", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--output", default=str(OUTPUT))
    args = ap.parse_args()

    torch.set_num_threads(1)

    original = json.loads(ORIGINAL.read_text())
    density = json.loads(DENSITY.read_text())

    val = [r for r in original if r["split"] == "val"]
    test = [r for r in density if r["split"] == "test_degree_ood"]

    if len(val) != 200 or len(test) != 80:
        raise RuntimeError("Unexpected validation/test manifest size")

    if {r["graph_hash"] for r in original} & {
        r["graph_hash"] for r in test
    }:
        raise RuntimeError("Graph leakage")

    if {r["path"] for r in original} & {
        r["path"] for r in test
    }:
        raise RuntimeError("Path leakage")

    jobs = [
        (model, family, seed)
        for model in MODELS
        for family in FAMILIES
        for seed in range(1, 6)
    ]

    if args.checkpoint_limit:
        jobs = jobs[:args.checkpoint_limit]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")

    with output.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for model, family, seed in jobs:
            name = f"{model}_{family}_seed{seed}"
            ckpt = torch.load(
                CKPTS / f"{name}.pt",
                map_location="cpu",
                weights_only=False
            )

            cfg = ckpt["args"]

            assert cfg["model"] == model
            assert cfg["family"] == family
            assert int(cfg["seed"]) == seed
            assert cfg["loss"] == "qubo"
            assert cfg["feature_mode"] == "random"
            assert int(cfg["dim"]) == 17
            assert ckpt["manifest_sha256"] == digest(ORIGINAL)

            test_paths = {r["path"] for r in test}
            if (set(ckpt["train_paths"]) |
                    set(ckpt["val_paths"])) & test_paths:
                raise RuntimeError("Checkpoint/test leakage")

            cfg = argparse.Namespace(**cfg)
            cfg.device = args.device

            ec = make_ec(cfg)
            net = build_net(cfg, ec)
            net.load_state_dict(ckpt["model_state"], strict=True)
            net.eval()

            # Calibration: use validation data only.
            val_rows = sorted(
                val, key=lambda r: (r["family"], r["seed"])
            )
            if args.val_limit:
                val_rows = [
                    r for fam in FAMILIES
                    for r in [v for v in val_rows
                              if v["family"] == fam][:args.val_limit]
                ]

            scores = {t: [] for t in THRESHOLDS}

            for row in val_rows:
                _, fixed = evaluate(net, ec, cfg, row)
                for t, c in fixed:
                    # Validation graphs can differ in edge count.
                    # Normalize by total absolute edge weight.
                    # For the existing unweighted ER/RRG data,
                    # this is equal to the edge count.
                    scores[t].append(c / max(row["edges"], 1))

            best_threshold = max(
                THRESHOLDS,
                key=lambda t: (np.mean(scores[t]), -t)
            )

            print(
                f"CALIBRATED {name}: threshold={best_threshold:.2f}",
                flush=True
            )

            test_rows = sorted(
                test, key=lambda r: (r["family"],
                                    r["target_degree"], r["seed"])
            )

            if args.test_limit:
                test_rows = [
                    r for fam in FAMILIES
                    for r in [v for v in test_rows
                              if v["family"] == fam][:args.test_limit]
                ]

            for row in test_rows:
                result, fixed = evaluate(net, ec, cfg, row)

                p_cpu, src, dst, w, denominator = result.pop("_scores")
                bits = p_cpu >= best_threshold
                calibrated_cut = (
                    w * (bits[src] != bits[dst])
                ).sum().item()
                result["calibrated"] = (
                    best_threshold,
                    calibrated_cut,
                    calibrated_cut / max(denominator, 1e-12),
                    bits.float().mean().item()
                )

                for method, (t, c, ratio, balance) in result.items():
                    write_row(
                        writer, model, family, seed,
                        row, "test_degree_ood",
                        method, t, c, ratio, balance
                    )

                f.flush()

            print(f"DONE {name}", flush=True)

    print(f"SAVED {output}", flush=True)


if __name__ == "__main__":
    main()
