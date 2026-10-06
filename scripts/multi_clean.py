import argparse, json, numpy as np, torch
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
    ap.add_argument('--ntrain', type=int, default=20)
    ap.add_argument('--ntest', type=int, default=5)
    ap.add_argument('--family', default='SBM')
    ap.add_argument('--manifest', default='dataset/splits/manifest.json')
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--heads', type=int, default=1)
    ap.add_argument('--dim', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--model', default='pigatv2')
    ap.add_argument('--config', default='configs/maxcut_main.yaml')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    ec = make_ec(a)
    net = build_net(a, ec)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)

    man = json.load(open(a.manifest))
    tr = sorted([r for r in man if r['split'] == 'train' and r['family'] == a.family],
                key=lambda r: r['seed'])[:a.ntrain]
    te = sorted([r for r in man if r['split'] == 'test_iid' and r['family'] == a.family],
                key=lambda r: r['seed'])[:a.ntest]
    train, test = load(tr, ec, a.k), load(te, ec, a.k)
    print(f'train {len(train)} graph, test {len(test)} graph')

    net.train()
    run = []
    for s in range(a.steps):
        _, Gk, U = train[np.random.randint(len(train))]
        p = fwd(net, Gk, make_x(Gk, 'clean', a.dim, U, None, ec))
        loss = norm_loss(p, Gk)
        opt.zero_grad()
        loss.backward()
        opt.step()
        run.append(loss.item())
        if s % 500 == 0 or s == a.steps - 1:
            print(f'step {s} loss_norm(avg500) {np.mean(run[-500:]):.4f}', flush=True)

    def ev(name, items):
        c5, cm = [], []
        net.eval()
        with torch.no_grad():
            for _, Gk, U in items:
                p = fwd(net, Gk, make_x(Gk, 'clean', a.dim, U, None, ec))
                c5.append(cut(Gk, (p >= 0.5).long()))
                cm.append(cut(Gk, (p >= p.median()).long()))
        print(f'{name}: cut@0.5 {np.mean(c5)*100:.1f}%  cut@median {np.mean(cm)*100:.1f}%  (n={len(items)})')

    ev('train graphs (first 5)', train[:5])
    ev('UNSEEN', test)


if __name__ == '__main__':
    main()
