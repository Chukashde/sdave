import argparse
import json
import numpy as np
import torch

from scripts.seq_curriculum import G, make_ec, build_net, fwd, norm_loss
from scripts.g1_diag import spectral, make_x, cut


def load(rows, ec, k):
    out = []

    for r in rows:
        Gk = G(r['path'], ec)
        U, _ = spectral(Gk, k)

        out.append((
            r['path'].split('/')[-1],
            Gk,
            U
        ))

    return out


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument('--ntrain', type=int, default=50)
    ap.add_argument('--steps', type=int, default=10000)

    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--heads', type=int, default=1)
    ap.add_argument('--dim', type=int, default=32)

    ap.add_argument('--lr', type=float, default=1e-3)

    ap.add_argument('--model', default='pigatv2')
    ap.add_argument('--config', default='configs/maxcut_main.yaml')
    ap.add_argument('--device', default='cuda')

    ap.add_argument('--seed', type=int, default=1)

    ap.add_argument(
        '--manifest',
        default='dataset/splits/manifest.json'
    )

    a = ap.parse_args()

    # -----------------------------
    # Reproducibility
    # -----------------------------

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    # -----------------------------
    # Build model
    # -----------------------------

    ec = make_ec(a)

    net = build_net(a, ec)

    opt = torch.optim.Adam(
        net.parameters(),
        lr=a.lr
    )

    # -----------------------------
    # Load manifest
    # -----------------------------

    man = json.load(open(a.manifest))

    # =========================================================
    # TRAIN = ONLY SBM
    # =========================================================

    train_rows = sorted(
        [
            r for r in man
            if r['split'] == 'train'
            and r['family'] == 'SBM'
        ],
        key=lambda r: r['seed']
    )[:a.ntrain]

    # =========================================================
    # IID TEST
    # SBM -> unseen SBM
    # =========================================================

    iid_rows = sorted(
        [
            r for r in man
            if r['split'] == 'test_iid'
            and r['family'] == 'SBM'
        ],
        key=lambda r: r['seed']
    )

    # =========================================================
    # SIZE OOD
    # SBM -> different-size SBM
    # =========================================================

    size_rows = sorted(
        [
            r for r in man
            if r['split'] == 'test_size'
            and r['family'] == 'SBM'
        ],
        key=lambda r: r['seed']
    )

    # =========================================================
    # FAMILY / TOPOLOGY OOD
    # =========================================================

    rr_rows = sorted(
        [
            r for r in man
            if r['split'] == 'test_family'
            and r['family'] == 'RR'
        ],
        key=lambda r: r['seed']
    )

    ws_rows = sorted(
        [
            r for r in man
            if r['split'] == 'test_family'
            and r['family'] == 'WS'
        ],
        key=lambda r: r['seed']
    )

    geo_rows = sorted(
        [
            r for r in man
            if r['split'] == 'test_family'
            and r['family'] == 'GEO'
        ],
        key=lambda r: r['seed']
    )

    # -----------------------------
    # Load graphs
    # -----------------------------

    train = load(train_rows, ec, a.k)

    test_iid = load(iid_rows, ec, a.k)
    test_size = load(size_rows, ec, a.k)

    test_rr = load(rr_rows, ec, a.k)
    test_ws = load(ws_rows, ec, a.k)
    test_geo = load(geo_rows, ec, a.k)

    print()
    print("========================================")
    print("DATASET")
    print("========================================")

    print(f"TRAIN SBM : {len(train)}")
    print(f"IID SBM   : {len(test_iid)}")
    print(f"SIZE SBM  : {len(test_size)}")
    print(f"RR        : {len(test_rr)}")
    print(f"WS        : {len(test_ws)}")
    print(f"GEO       : {len(test_geo)}")

    # =========================================================
    # TRAIN
    # IMPORTANT:
    # Training happens ONLY here and ONLY on SBM/train.
    # =========================================================

    print()
    print("========================================")
    print("TRAINING ON SBM ONLY")
    print("========================================")

    net.train()

    run = []

    for s in range(a.steps):

        _, Gk, U = train[
            np.random.randint(len(train))
        ]

        x = make_x(
            Gk,
            'clean',
            a.dim,
            U,
            None,
            ec
        )

        p = fwd(net, Gk, x)

        loss = norm_loss(p, Gk)

        opt.zero_grad()
        loss.backward()
        opt.step()

        run.append(loss.item())

        if s % 500 == 0 or s == a.steps - 1:

            avg = np.mean(run[-500:])

            print(
                f"step {s} "
                f"loss_norm(avg500) {avg:.4f}",
                flush=True
            )

    # =========================================================
    # EVALUATION
    # NO TRAINING / NO OPTIMIZER STEP BELOW THIS POINT
    # =========================================================

    def evaluate(name, items):

        c05 = []
        cmedian = []

        net.eval()

        with torch.no_grad():

            for _, Gk, U in items:

                x = make_x(
                    Gk,
                    'clean',
                    a.dim,
                    U,
                    None,
                    ec
                )

                p = fwd(net, Gk, x)

                c05.append(
                    cut(
                        Gk,
                        (p >= 0.5).long()
                    )
                )

                cmedian.append(
                    cut(
                        Gk,
                        (p >= p.median()).long()
                    )
                )

        mean05 = np.mean(c05) * 100
        std05 = np.std(c05) * 100

        meanmed = np.mean(cmedian) * 100
        stdmed = np.std(cmedian) * 100

        print(
            f"{name:15s} | "
            f"cut@0.5 {mean05:5.1f} +/- {std05:4.1f}% | "
            f"median {meanmed:5.1f} +/- {stdmed:4.1f}% | "
            f"n={len(items)}"
        )

    print()
    print("========================================")
    print("GENERALIZATION RESULTS")
    print("========================================")

    evaluate(
        "TRAIN",
        train[:10]
    )

    evaluate(
        "IID SBM",
        test_iid
    )

    evaluate(
        "OOD SIZE",
        test_size
    )

    evaluate(
        "OOD RR",
        test_rr
    )

    evaluate(
        "OOD WS",
        test_ws
    )

    evaluate(
        "OOD GEO",
        test_geo
    )


if __name__ == '__main__':
    main()
