import argparse
import json
import numpy as np
import torch

from scripts.seq_curriculum import G, make_ec, build_net, fwd, norm_loss
from scripts.g1_diag import spectral, make_x, cut


def load(rows, ec, k):
    out = []

    for r in rows:
        Gk = G(r["path"], ec)
        U, _ = spectral(Gk, k)

        out.append((
            r["path"].split("/")[-1],
            Gk,
            U,
            r
        ))

    return out


def select_balanced_train(man, families, per_family):
    rows = []

    for family in families:
        candidates = sorted(
            [
                r for r in man
                if r["split"] == "train"
                and r["family"] == family
            ],
            key=lambda r: r["seed"]
        )

        selected = candidates[:per_family]

        print(
            f"TRAIN {family}: "
            f"{len(selected)} / {len(candidates)}"
        )

        rows.extend(selected)

    return rows


def select_test(man):
    tests = {}

    tests["IID_SBM"] = sorted(
        [
            r for r in man
            if r["family"] == "SBM"
            and r["split"] == "test_iid"
        ],
        key=lambda r: r["seed"]
    )

    tests["SIZE_500"] = sorted(
        [
            r for r in man
            if r["family"] == "SBM"
            and r["split"] == "test_size"
            and r["n"] == 500
        ],
        key=lambda r: r["seed"]
    )

    tests["SIZE_2000"] = sorted(
        [
            r for r in man
            if r["family"] == "SBM"
            and r["split"] == "test_size"
            and r["n"] == 2000
        ],
        key=lambda r: r["seed"]
    )

    tests["SIZE_5000"] = sorted(
        [
            r for r in man
            if r["family"] == "SBM"
            and r["split"] == "test_size"
            and r["n"] == 5000
        ],
        key=lambda r: r["seed"]
    )

    for family in ["RR", "WS", "GEO"]:
        tests[f"OOD_{family}"] = sorted(
            [
                r for r in man
                if r["family"] == family
                and r["split"] == "test_family"
            ],
            key=lambda r: r["seed"]
        )

    return tests


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--families",
        nargs="+",
        default=["ER", "SF", "SBM"]
    )

    ap.add_argument(
        "--per_family",
        type=int,
        default=50
    )

    ap.add_argument(
        "--manifest",
        default="dataset/splits/manifest.json"
    )

    ap.add_argument(
        "--steps",
        type=int,
        default=100000
    )

    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model", default="pigatv2")
    ap.add_argument(
        "--config",
        default="configs/maxcut_main.yaml"
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)

    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    ec = make_ec(a)
    net = build_net(a, ec)

    opt = torch.optim.Adam(
        net.parameters(),
        lr=a.lr
    )

    with open(a.manifest) as f:
        man = json.load(f)

    # -----------------------------------------
    # TRAIN
    # -----------------------------------------

    tr = select_balanced_train(
        man,
        a.families,
        a.per_family
    )

    print()
    print("Total training graphs:", len(tr))

    train = load(
        tr,
        ec,
        a.k
    )

    # -----------------------------------------
    # TEST
    # -----------------------------------------

    test_rows = select_test(man)

    tests = {}

    for name, rows in test_rows.items():
        print(
            f"TEST {name}: {len(rows)}"
        )

        tests[name] = load(
            rows,
            ec,
            a.k
        )

    # -----------------------------------------
    # TRAINING
    # -----------------------------------------

    print()
    print("==============================")
    print("MULTI-FAMILY TRAINING")
    print("==============================")
    print("Families:", a.families)
    print("Graphs:", len(train))
    print("Steps:", a.steps)
    print("Seed:", a.seed)
    print("==============================")
    print()

    net.train()

    run = []

    for s in range(a.steps):

        idx = np.random.randint(
            len(train)
        )

        _, Gk, U, _ = train[idx]

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

        loss = norm_loss(
            p,
            Gk
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        run.append(
            loss.item()
        )

        if (
            s % 500 == 0
            or s == a.steps - 1
        ):
            print(
                f"step {s} "
                f"loss_norm(avg500) "
                f"{np.mean(run[-500:]):.4f}",
                flush=True
            )

    # -----------------------------------------
    # EVALUATION
    # -----------------------------------------

    def ev(name, items):

        c5 = []
        cm = []

        net.eval()

        with torch.no_grad():

            for _, Gk, U, _ in items:

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

                c5.append(
                    cut(
                        Gk,
                        (p >= 0.5).long()
                    )
                )

                cm.append(
                    cut(
                        Gk,
                        (
                            p >= p.median()
                        ).long()
                    )
                )

        c5 = np.array(c5) * 100.0
        cm = np.array(cm) * 100.0

        print(
            f"{name:12s} | "
            f"cut@0.5 "
            f"{c5.mean():6.2f} ± "
            f"{c5.std(ddof=1):5.2f}% | "
            f"median "
            f"{cm.mean():6.2f} ± "
            f"{cm.std(ddof=1):5.2f}% | "
            f"n={len(items)}"
        )

    print()
    print("==============================")
    print("RESULTS")
    print("==============================")

    for name, items in tests.items():
        if len(items) > 0:
            ev(
                name,
                items
            )


if __name__ == "__main__":
    main()
