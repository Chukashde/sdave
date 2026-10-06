import argparse
import json
import numpy as np
import torch

from scripts.seq_curriculum import G, make_ec, build_net, fwd
from scripts.g1_diag import spectral, make_x


def get_loss(p, G, mode):
    d = p[G.src] - p[G.dst]
    raw = -(G.w * d * d).sum()

    if mode == "raw":
        return raw
    elif mode == "node":
        return raw / G.n
    elif mode == "edge":
        return raw / G.wsum

    raise ValueError(mode)


def grad_norm(net):
    total = 0.0

    for param in net.parameters():
        if param.grad is not None:
            total += param.grad.detach().pow(2).sum().item()

    return total ** 0.5


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--model", default="pigatv2")
    ap.add_argument("--config", default="configs/maxcut_main.yaml")
    ap.add_argument("--device", default="cuda")

    ap.add_argument(
        "--manifest",
        default="dataset/splits/manifest.json"
    )

    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    ec = make_ec(a)

    man = json.load(open(a.manifest))

    # Same graph groups used in normalization pilot
    groups = {}

    groups[500] = sorted([
        r for r in man
        if r["family"] == "SBM"
        and r["split"] == "test_size"
        and r["n"] == 500
    ], key=lambda r: r["seed"])[:10]

    groups[1000] = sorted([
        r for r in man
        if r["family"] == "SBM"
        and r["split"] == "train"
        and r["n"] == 1000
    ], key=lambda r: r["seed"])[:10]

    groups[2000] = sorted([
        r for r in man
        if r["family"] == "SBM"
        and r["split"] == "test_size"
        and r["n"] == 2000
    ], key=lambda r: r["seed"])[:10]

    # Load graphs once
    loaded = {}

    for n, rows in groups.items():
        loaded[n] = []

        for r in rows:
            Gk = G(r["path"], ec)
            U, _ = spectral(Gk, a.k)

            loaded[n].append((Gk, U))

    print()
    print("=" * 72)
    print("QUBO LOSS / GRADIENT SCALE ANALYSIS")
    print("=" * 72)

    # IMPORTANT:
    # Re-create identical model initialization for every mode.
    for mode in ["raw", "node", "edge"]:

        torch.manual_seed(a.seed)
        np.random.seed(a.seed)

        net = build_net(a, ec)
        net.train()

        print()
        print(f"LOSS MODE: {mode.upper()}")
        print("-" * 72)

        for n in [500, 1000, 2000]:

            losses = []
            grads = []

            for Gk, U in loaded[n]:

                x = make_x(
                    Gk,
                    "clean",
                    a.dim,
                    U,
                    None,
                    ec
                )

                net.zero_grad(set_to_none=True)

                p = fwd(net, Gk, x)

                loss = get_loss(
                    p,
                    Gk,
                    mode
                )

                loss.backward()

                gn = grad_norm(net)

                losses.append(loss.item())
                grads.append(gn)

            print(
                f"n={n:<5d} | "
                f"loss {np.mean(losses):12.6f} "
                f"+/- {np.std(losses):10.6f} | "
                f"grad_norm {np.mean(grads):12.6f} "
                f"+/- {np.std(grads):10.6f}"
            )


if __name__ == "__main__":
    main()
