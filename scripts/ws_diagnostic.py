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
        out.append((r['path'].split('/')[-1], Gk, U))
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
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)

    man = json.load(open(a.manifest))

    # --------------------------------
    # TRAIN: SBM only
    # --------------------------------
    train_rows = sorted(
        [
            r for r in man
            if r['split'] == 'train'
            and r['family'] == 'SBM'
        ],
        key=lambda r: r['seed']
    )[:a.ntrain]

    # --------------------------------
    # TEST: WS only
    # --------------------------------
    ws_rows = sorted(
        [
            r for r in man
            if r['split'] == 'test_family'
            and r['family'] == 'WS'
        ],
        key=lambda r: r['seed']
    )

    train = load(train_rows, ec, a.k)
    ws = load(ws_rows, ec, a.k)

    print(f"TRAIN SBM = {len(train)}")
    print(f"TEST WS   = {len(ws)}")

    # =================================
    # TRAIN ON SBM
    # =================================
    net.train()

    run = []

    for s in range(a.steps):
        _, Gk, U = train[np.random.randint(len(train))]

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

        if s % 1000 == 0 or s == a.steps - 1:
            print(
                f"step {s:5d} "
                f"loss {np.mean(run[-500:]):.4f}",
                flush=True
            )

    # =================================
    # WS DIAGNOSTIC
    # =================================
    print()
    print("=" * 100)
    print("WS DIAGNOSTIC")
    print("=" * 100)

    print(
        f"{'graph':22s} "
        f"{'cut05':>7s} "
        f"{'median':>7s} "
        f"{'pmean':>7s} "
        f"{'pstd':>7s} "
        f"{'pmin':>7s} "
        f"{'pmax':>7s} "
        f"{'ones05':>8s} "
        f"{'onesMed':>8s}"
    )

    cut05_all = []
    cutmed_all = []

    net.eval()

    with torch.no_grad():

        for name, Gk, U in ws:

            x = make_x(
                Gk,
                'clean',
                a.dim,
                U,
                None,
                ec
            )

            p = fwd(net, Gk, x)

            # threshold = 0.5
            b05 = (p >= 0.5).long()

            # threshold = median
            bmed = (p >= p.median()).long()

            c05 = cut(Gk, b05)
            cmed = cut(Gk, bmed)

            cut05_all.append(c05)
            cutmed_all.append(cmed)

            ones05 = b05.float().mean().item() * 100
            onesmed = bmed.float().mean().item() * 100

            print(
                f"{name:22s} "
                f"{c05*100:7.1f} "
                f"{cmed*100:7.1f} "
                f"{p.mean().item():7.3f} "
                f"{p.std().item():7.3f} "
                f"{p.min().item():7.3f} "
                f"{p.max().item():7.3f} "
                f"{ones05:7.1f}% "
                f"{onesmed:7.1f}%"
            )

    print("=" * 100)

    print(
        f"MEAN WS cut@0.5  = "
        f"{np.mean(cut05_all)*100:.2f}%"
    )

    print(
        f"MEAN WS median   = "
        f"{np.mean(cutmed_all)*100:.2f}%"
    )


if __name__ == '__main__':
    main()
