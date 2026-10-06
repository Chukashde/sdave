import argparse
import json
import numpy as np
import torch

from scripts.seq_curriculum import G, make_ec, build_net, fwd
from scripts.g1_diag import spectral, make_x, cut


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def raw_loss(p, G):
    d = p[G.src] - p[G.dst]
    return -(G.w * d * d).sum()


def node_loss(p, G):
    d = p[G.src] - p[G.dst]
    return -(G.w * d * d).sum() / G.n


def edge_loss(p, G):
    d = p[G.src] - p[G.dst]
    return -(G.w * d * d).sum() / G.wsum


def get_loss(p, G, mode):
    if mode == "raw":
        return raw_loss(p, G)

    if mode == "node":
        return node_loss(p, G)

    if mode == "edge":
        return edge_loss(p, G)

    raise ValueError(f"Unknown loss mode: {mode}")


# ============================================================
# LOAD GRAPHS
# ============================================================

def load(rows, ec, k):
    out = []

    for r in rows:
        Gk = G(r["path"], ec)
        U, _ = spectral(Gk, k)

        out.append({
            "name": r["path"].split("/")[-1],
            "G": Gk,
            "U": U,
            "n": r["n"],
        })

    return out


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--loss",
        choices=["raw", "node", "edge"],
        required=True
    )

    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--model", default="pigatv2")
    ap.add_argument("--config", default="configs/maxcut_main.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument(
        "--manifest",
        default="dataset/splits/manifest.json"
    )

    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    ec = make_ec(a)

    net = build_net(a, ec)

    opt = torch.optim.Adam(
        net.parameters(),
        lr=a.lr
    )

    man = json.load(
        open(a.manifest)
    )

    # ========================================================
    # TRAIN SET
    #
    # n=500   -> test_size
    # n=1000  -> train
    # n=2000  -> test_size
    #
    # We deliberately build a mixed-size training set.
    # ========================================================

    rows500 = sorted(
        [
            r for r in man
            if r["family"] == "SBM"
            and r["split"] == "test_size"
            and r["n"] == 500
        ],
        key=lambda r: r["seed"]
    )

    rows1000 = sorted(
        [
            r for r in man
            if r["family"] == "SBM"
            and r["split"] == "train"
            and r["n"] == 1000
        ],
        key=lambda r: r["seed"]
    )

    rows2000 = sorted(
        [
            r for r in man
            if r["family"] == "SBM"
            and r["split"] == "test_size"
            and r["n"] == 2000
        ],
        key=lambda r: r["seed"]
    )

    # Use equal number from each size.
    #
    # Existing dataset has:
    # n=500  -> 10
    # n=2000 -> 10
    #
    # Therefore use 10 from each size.
    train500 = rows500[:10]
    train1000 = rows1000[:10]
    train2000 = rows2000[:10]

    train_rows = (
        train500
        + train1000
        + train2000
    )

    # ========================================================
    # TEST SET
    # ========================================================

    iid1000_rows = sorted(
        [
            r for r in man
            if r["family"] == "SBM"
            and r["split"] == "test_iid"
            and r["n"] == 1000
        ],
        key=lambda r: r["seed"]
    )

    size5000_rows = sorted(
        [
            r for r in man
            if r["family"] == "SBM"
            and r["split"] == "test_size"
            and r["n"] == 5000
        ],
        key=lambda r: r["seed"]
    )

    # ========================================================
    # LOAD
    # ========================================================

    print("=" * 65)
    print("LOADING MIXED-SIZE TRAINING DATA")
    print("=" * 65)

    train = load(
        train_rows,
        ec,
        a.k
    )

    iid1000 = load(
        iid1000_rows,
        ec,
        a.k
    )

    test5000 = load(
        size5000_rows,
        ec,
        a.k
    )

    print()
    print(f"LOSS MODE       : {a.loss}")
    print(f"TRAIN n=500     : {len(train500)}")
    print(f"TRAIN n=1000    : {len(train1000)}")
    print(f"TRAIN n=2000    : {len(train2000)}")
    print(f"TOTAL TRAIN     : {len(train)}")
    print(f"TEST IID n=1000 : {len(iid1000)}")
    print(f"TEST OOD n=5000 : {len(test5000)}")

    # ========================================================
    # SHOW INITIAL LOSS SCALE
    # ========================================================

    print()
    print("=" * 65)
    print("INITIAL LOSS SCALE")
    print("=" * 65)

    net.eval()

    with torch.no_grad():

        for n in [500, 1000, 2000]:

            subset = [
                item for item in train
                if item["n"] == n
            ]

            vals = []

            for item in subset:

                Gk = item["G"]
                U = item["U"]

                x = make_x(
                    Gk,
                    "clean",
                    a.dim,
                    U,
                    None,
                    ec
                )

                p = fwd(
                    net,
                    Gk,
                    x
                )

                loss = get_loss(
                    p,
                    Gk,
                    a.loss
                )

                vals.append(
                    loss.item()
                )

            print(
                f"n={n:<5d} "
                f"loss mean = {np.mean(vals):12.6f} "
                f"std = {np.std(vals):10.6f}"
            )

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("=" * 65)
    print(f"TRAINING WITH {a.loss.upper()} LOSS")
    print("=" * 65)

    net.train()

    run = []

    for step in range(a.steps):

        item = train[
            np.random.randint(
                len(train)
            )
        ]

        Gk = item["G"]
        U = item["U"]

        x = make_x(
            Gk,
            "clean",
            a.dim,
            U,
            None,
            ec
        )

        p = fwd(
            net,
            Gk,
            x
        )

        loss = get_loss(
            p,
            Gk,
            a.loss
        )

        opt.zero_grad()

        loss.backward()

        opt.step()

        run.append(
            loss.item()
        )

        if (
            step % 500 == 0
            or step == a.steps - 1
        ):

            print(
                f"step {step:<5d} "
                f"loss(avg500) "
                f"{np.mean(run[-500:]):.6f}",
                flush=True
            )

    # ========================================================
    # EVALUATION
    # ========================================================

    def evaluate(name, items):

        cut05 = []
        cutmed = []

        net.eval()

        with torch.no_grad():

            for item in items:

                Gk = item["G"]
                U = item["U"]

                x = make_x(
                    Gk,
                    "clean",
                    a.dim,
                    U,
                    None,
                    ec
                )

                p = fwd(
                    net,
                    Gk,
                    x
                )

                b05 = (
                    p >= 0.5
                ).long()

                bmed = (
                    p >= p.median()
                ).long()

                cut05.append(
                    cut(Gk, b05) * 100
                )

                cutmed.append(
                    cut(Gk, bmed) * 100
                )

        print(
            f"{name:18s} | "
            f"cut@0.5 "
            f"{np.mean(cut05):6.2f} "
            f"+/- {np.std(cut05):5.2f}% | "
            f"median "
            f"{np.mean(cutmed):6.2f} "
            f"+/- {np.std(cutmed):5.2f}% | "
            f"n={len(items)}"
        )

    print()
    print("=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    # Performance on mixed training sizes
    for n in [500, 1000, 2000]:

        subset = [
            item for item in train
            if item["n"] == n
        ]

        evaluate(
            f"TRAIN n={n}",
            subset
        )

    evaluate(
        "IID n=1000",
        iid1000
    )

    evaluate(
        "OOD n=5000",
        test5000
    )


if __name__ == "__main__":
    main()
