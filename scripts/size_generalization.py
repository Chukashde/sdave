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
            U,
            r['n']
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

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    ec = make_ec(a)
    net = build_net(a, ec)

    opt = torch.optim.Adam(
        net.parameters(),
        lr=a.lr
    )

    man = json.load(open(a.manifest))

    # =====================================================
    # TRAIN: SBM n=1000
    # =====================================================

    train_rows = sorted(
        [
            r for r in man
            if r['split'] == 'train'
            and r['family'] == 'SBM'
        ],
        key=lambda r: r['seed']
    )[:a.ntrain]

    # =====================================================
    # IID: SBM n=1000
    # =====================================================

    iid_rows = sorted(
        [
            r for r in man
            if r['split'] == 'test_iid'
            and r['family'] == 'SBM'
        ],
        key=lambda r: r['seed']
    )

    # =====================================================
    # SIZE OOD
    # =====================================================

    size_rows = [
        r for r in man
        if r['split'] == 'test_size'
        and r['family'] == 'SBM'
    ]

    rows_500 = sorted(
        [r for r in size_rows if r['n'] == 500],
        key=lambda r: r['seed']
    )

    rows_2000 = sorted(
        [r for r in size_rows if r['n'] == 2000],
        key=lambda r: r['seed']
    )

    rows_5000 = sorted(
        [r for r in size_rows if r['n'] == 5000],
        key=lambda r: r['seed']
    )

    train = load(train_rows, ec, a.k)
    iid = load(iid_rows, ec, a.k)

    test500 = load(rows_500, ec, a.k)
    test2000 = load(rows_2000, ec, a.k)
    test5000 = load(rows_5000, ec, a.k)

    print()
    print("=" * 60)
    print("DATASET")
    print("=" * 60)

    print(f"TRAIN n=1000 : {len(train)}")
    print(f"IID   n=1000 : {len(iid)}")
    print(f"SIZE  n=500  : {len(test500)}")
    print(f"SIZE  n=2000 : {len(test2000)}")
    print(f"SIZE  n=5000 : {len(test5000)}")

    # =====================================================
    # TRAIN
    # =====================================================

    print()
    print("=" * 60)
    print("TRAINING ON SBM n=1000 ONLY")
    print("=" * 60)

    net.train()
    run = []

    for s in range(a.steps):

        _, Gk, U, _ = train[
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
            print(
                f"step {s} "
                f"loss_norm(avg500) "
                f"{np.mean(run[-500:]):.4f}",
                flush=True
            )

    # =====================================================
    # EVALUATION
    # =====================================================

    def evaluate(name, items):

        cut05 = []
        cutmed = []

        net.eval()

        with torch.no_grad():

            for _, Gk, U, _ in items:

                x = make_x(
                    Gk,
                    'clean',
                    a.dim,
                    U,
                    None,
                    ec
                )

                p = fwd(net, Gk, x)

                b05 = (p >= 0.5).long()
                bmed = (p >= p.median()).long()

                cut05.append(
                    cut(Gk, b05) * 100
                )

                cutmed.append(
                    cut(Gk, bmed) * 100
                )

        mean05 = np.mean(cut05)
        std05 = np.std(cut05)

        meanmed = np.mean(cutmed)
        stdmed = np.std(cutmed)

        print(
            f"{name:15s} | "
            f"cut@0.5 {mean05:6.2f} +/- {std05:5.2f}% | "
            f"median {meanmed:6.2f} +/- {stdmed:5.2f}% | "
            f"n={len(items)}"
        )

        return mean05, meanmed

    print()
    print("=" * 60)
    print("SIZE GENERALIZATION RESULTS")
    print("=" * 60)

    evaluate(
        "IID n=1000",
        iid
    )

    evaluate(
        "OOD n=500",
        test500
    )

    evaluate(
        "OOD n=2000",
        test2000
    )

    evaluate(
        "OOD n=5000",
        test5000
    )


if __name__ == '__main__':
    main()
