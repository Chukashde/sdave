import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from scripts.seq_curriculum import make_ec, build_net, fwd
from experiments.protocol2026.scripts.train_generalization import load_graphs


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default="experiments/phase1_ood/data_density/manifest.json"
    )
    ap.add_argument(
        "--original-manifest",
        default="dataset/protocol2026_v2/manifest.json"
    )
    ap.add_argument(
        "--checkpoint-dir",
        default="experiments/protocol2026/checkpoints/production"
    )
    ap.add_argument(
        "--output",
        default="experiments/phase1_ood/results/partition_balance.csv"
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    rows = json.loads(Path(args.manifest).read_text())
    original = json.loads(Path(args.original_manifest).read_text())

    if any(r["split"] != "test_degree_ood" for r in rows):
        raise ValueError("Invalid density OOD split")

    if {r["graph_hash"] for r in rows} & {
        r["graph_hash"] for r in original
    }:
        raise RuntimeError("Graph leakage")

    orig_digest = sha256(args.original_manifest)

    selected = [
        r for r in rows
        if r["target_degree"] in (3, 4, 20)
        and (
            (r["family"] == "ER" and r["target_degree"] in (4, 20))
            or
            (r["family"] == "RRG" and r["target_degree"] in (3, 20))
        )
    ]

    fields = [
        "model", "train_family", "test_family",
        "train_seed", "target_degree", "path",
        "p_mean", "p_std", "p_min", "p_max",
        "frac_ge_05", "frac_lt_001", "frac_gt_099"
    ]

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for train_family in ("ER", "RRG"):
            for seed in range(1, 6):
                model = "pigat_jk"
                name = f"{model}_{train_family}_seed{seed}"

                ckpt = torch.load(
                    Path(args.checkpoint_dir) / f"{name}.pt",
                    map_location="cpu",
                    weights_only=False
                )

                cfg = ckpt["args"]

                assert cfg["model"] == model
                assert cfg["family"] == train_family
                assert int(cfg["seed"]) == seed
                assert cfg["loss"] == "qubo"
                assert cfg["feature_mode"] == "random"
                assert int(cfg["dim"]) == 17
                assert ckpt["manifest_sha256"] == orig_digest

                if (
                    set(ckpt["train_paths"]) |
                    set(ckpt["val_paths"])
                ) & {r["path"] for r in selected}:
                    raise RuntimeError("Checkpoint path leakage")

                cfg = argparse.Namespace(**cfg)
                cfg.device = args.device

                ec = make_ec(cfg)
                net = build_net(cfg, ec)
                net.load_state_dict(
                    ckpt["model_state"], strict=True
                )
                net.eval()

                for row in selected:
                    _, graph, x = load_graphs(
                        [row], ec, cfg.k, cfg
                    )[0]

                    with torch.inference_mode():
                        p = fwd(net, graph, x)

                        values = dict(
                            p_mean=p.mean().item(),
                            p_std=p.std().item(),
                            p_min=p.min().item(),
                            p_max=p.max().item(),
                            frac_ge_05=(p >= 0.5).float().mean().item(),
                            frac_lt_001=(p < 0.01).float().mean().item(),
                            frac_gt_099=(p > 0.99).float().mean().item(),
                        )

                    writer.writerow(dict(
                        model=model,
                        train_family=train_family,
                        test_family=row["family"],
                        train_seed=seed,
                        target_degree=row["target_degree"],
                        path=row["path"],
                        **values
                    ))
                    file.flush()

                print(f"DONE {name}", flush=True)

    print(f"SAVED {output}")


if __name__ == "__main__":
    main()
