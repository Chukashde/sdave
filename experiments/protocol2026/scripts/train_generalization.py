import yaml
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from scripts.seq_curriculum import G, make_ec, build_net, fwd, norm_loss
from scripts.g1_diag import spectral, make_x


def load_graphs(rows, ec, k, args):
    graphs = []
    for row in rows:
        graph = G(row["path"], ec)

        if args.feature_mode == "random":
            # Per-graph reproducible feature draw; no global RNG changes.
            key = f'{args.seed}:{row["graph_hash"]}:random-v1'
            seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            x = torch.randn(graph.n, args.dim, generator=generator)
            x[:, 0] = graph.deg
            x = x.to(device=ec["device"], dtype=ec["dtype"])
        else:
            U, _ = spectral(graph, k)
            x = make_x(graph, "clean", args.dim, U, None, ec)

        graphs.append((row, graph, x))
    return graphs


def features(graph, x, args, ec):
    return x




def qubo_loss(p, graph):
    pu = p[graph.src]
    pv = p[graph.dst]

    expected_cut = (
        graph.w * (pu + pv - 2 * pu * pv)
    ).sum()

    denominator = graph.w.abs().sum().clamp_min(1e-12)
    return -expected_cut / denominator


def training_loss(p, graph, loss_type):
    if loss_type == "qubo":
        return qubo_loss(p, graph)
    if loss_type == "square":
        return norm_loss(p, graph)
    raise ValueError(f"Unknown loss: {loss_type}")


@torch.no_grad()
def evaluate(net, graphs, args, ec):
    net.eval()
    losses, cut05, cutmedian, cutrank, pstd = [], [], [], [], []

    for _, graph, U in graphs:
        p = fwd(net, graph, features(graph, U, args, ec))
        losses.append(training_loss(p, graph, args.loss).item())

        b05 = p >= 0.5
        bmed = p >= p.median()

        order = torch.argsort(p, stable=True)
        brank = torch.zeros_like(p, dtype=torch.bool)
        brank[order[graph.n // 2:]] = True

        def cut(bits):
            return (
                graph.w * (bits[graph.src] != bits[graph.dst])
            ).sum().item() / graph.w.abs().sum().clamp_min(1e-12).item()

        cut05.append(cut(b05))
        cutmedian.append(cut(bmed))
        cutrank.append(cut(brank))
        pstd.append(p.std().item())

    return {
        "loss": float(np.mean(losses)),
        "cut_ratio": float(np.mean(cut05)),
        "cut_median": float(np.mean(cutmedian)),
        "cut_rank50": float(np.mean(cutrank)),
        "p_std": float(np.mean(pstd)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["qubo", "square"], default="square")
    ap.add_argument(
        "--manifest",
        default="dataset/protocol2026_v2/manifest.json"
    )
    ap.add_argument("--family", choices=["RRG", "ER"], default="RRG")
    ap.add_argument("--model", choices=["pigcn", "pigcn_jk", "pigat_jk", "pigatv2", "pigatv2_jk"],
                    default="pigatv2")
    ap.add_argument("--config", default="configs/maxcut_main.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--train_limit", type=int, default=2)
    ap.add_argument("--val_limit", type=int, default=2)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--feature_mode", choices=["spectral", "random"], default="spectral")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument(
        "--out",
        default="experiments/protocol2026/checkpoints/smoke.pt"
    )
    ap.add_argument(
        "--protocol_config",
        default="experiments/protocol2026/configs/protocol_phase0.yaml"
    )
    ap.add_argument("--production", action="store_true")

    args = ap.parse_args()

    if args.production:
        with open(args.protocol_config) as f:
            protocol = yaml.safe_load(f)

        from experiments.protocol2026.scripts.protocol_config import (
            apply_training_config,
        )

        try:
            args = apply_training_config(args, protocol)
        except (ValueError, KeyError, TypeError) as exc:
            ap.error(f"Production blocked: {exc}")


    if args.steps < 1 or args.train_limit < 1 or args.val_limit < 1:
        ap.error("steps, train_limit and val_limit must be positive")
    if args.k >= args.dim:
        ap.error("--k must be smaller than --dim")
    if args.log_every < 1:
        ap.error("--log_every must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ec = make_ec(args)

    with open(args.manifest) as f:
        manifest = json.load(f)

    def select(split, limit):
        rows = [
            r for r in manifest
            if r["split"] == split and r["family"] == args.family
        ]
        rows = sorted(rows, key=lambda r: r["seed"])
        return rows[:limit]

    train_rows = select("train", args.train_limit)
    val_rows = select("val", args.val_limit)

    if len(train_rows) != args.train_limit:
        raise RuntimeError("Not enough training graphs")
    if len(val_rows) != args.val_limit:
        raise RuntimeError("Not enough validation graphs")

    train = load_graphs(train_rows, ec, args.k, args)
    val = load_graphs(val_rows, ec, args.k, args)

    net = build_net(args, ec)
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)

    print(
        f"Model={args.model} Family={args.family} "
        f"Train={len(train)} Val={len(val)} "
        f"Steps={args.steps} Device={ec['device']}",
        flush=True
    )

    start = time.perf_counter()
    history = []
    best_val_loss = float("inf")
    best_step = 0
    best_model_state = None
    stale_evaluations = 0

    for step in range(1, args.steps + 1):
        net.train()
        idx = int(rng.integers(len(train)))
        _, graph, U = train[idx]

        x = features(graph, U, args, ec)
        p = fwd(net, graph, x)
        loss = training_loss(p, graph, args.loss)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            metrics = evaluate(net, val, args, ec)
            record = {
                "step": step,
                "train_loss": float(loss.item()),
                "val_loss": metrics["loss"],
                "val_cut_ratio": metrics["cut_ratio"],
                "val_cut_median": metrics["cut_median"],
                "val_cut_rank50": metrics["cut_rank50"],
                "val_p_std": metrics["p_std"],
            }
            history.append(record)

            if metrics["loss"] < best_val_loss - 1e-8:
                best_val_loss = metrics["loss"]
                best_step = step
                best_model_state = copy.deepcopy(net.state_dict())
                stale_evaluations = 0
            else:
                stale_evaluations += 1

            if stale_evaluations >= args.patience:
                print(
                    f"EARLY STOP: step={step}, best_step={best_step}, "
                    f"best_val_loss={best_val_loss:.6f}",
                    flush=True
                )
                break
            print(
                f"step={step:5d} "
                f"train_loss={record['train_loss']:.5f} "
                f"val_loss={record['val_loss']:.5f} "
                f"cut05={record['val_cut_ratio']:.3%} "
                f"median={record['val_cut_median']:.3%} "
                f"rank50={record['val_cut_rank50']:.3%} "
                f"pstd={record['val_p_std']:.4f}",
                flush=True
            )

    if best_model_state is None:
        raise RuntimeError("No validation checkpoint was recorded")

    net.load_state_dict(best_model_state)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        raise FileExistsError(
            f"Checkpoint exists: {out}. Choose another --out."
        )

    torch.save({
        "model_state": net.state_dict(),
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "early_stopping_patience": args.patience,
        "args": vars(args),
        "train_paths": [r["path"] for r in train_rows],
        "val_paths": [r["path"] for r in val_rows],
        "manifest_sha256": hashlib.sha256(
            Path(args.manifest).read_bytes()
        ).hexdigest(),
        "history": history,
        "elapsed_sec": time.perf_counter() - start,
    }, out)

    print(f"CHECKPOINT SAVED: {out}", flush=True)
    print("TRAINING SMOKE TEST PASSED", flush=True)


if __name__ == "__main__":
    main()
