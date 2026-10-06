import argparse, json, numpy as np, torch
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scripts.seq_curriculum import G, make_ec, build_net, fwd, norm_loss


def cut(Gk, b):
    return (Gk.w * (b[Gk.src] != b[Gk.dst])).sum().item() / Gk.wsum


def spectral(Gk, k):
    n = Gk.n
    s = Gk.src.cpu().numpy(); d_ = Gk.dst.cpu().numpy(); w = Gk.w.cpu().numpy()
    A = sp.coo_matrix((np.r_[w, w], (np.r_[s, d_], np.r_[d_, s])), shape=(n, n)).tocsr()
    d = np.asarray(A.sum(1)).ravel()
    d[d == 0] = 1
    Dm = sp.diags(1 / np.sqrt(d))
    N = Dm @ A @ Dm
    kk = max(1, k // 2)
    _, Us = eigsh(N, k=kk, which='SA')
    _, Ul = eigsh(N, k=kk, which='LA')
    return np.concatenate([Us, Ul], 1), Us[:, 0]


def make_x(Gk, mode, dim, U, fixed, ec):
    n = Gk.n
    if mode == 'random':
        x = torch.randn(n, dim)
    elif mode == 'fixed':
        x = fixed.clone()
    elif mode == 'clean':
        k = U.shape[1]
        idx = np.abs(U).argmax(0)
        sgn = np.sign(U[idx, np.arange(k)])
        x = torch.zeros(n, dim)
        x[:, 1:1 + k] = torch.tensor(U * sgn, dtype=torch.float32) * np.sqrt(n)
    else:
        x = torch.randn(n, dim) * 0.1
        sg = torch.tensor(np.random.choice([-1.0, 1.0], size=U.shape[1]), dtype=torch.float32)
        x[:, 1:1 + U.shape[1]] = torch.tensor(U, dtype=torch.float32) * sg * np.sqrt(n)
    x[:, 0] = Gk.deg
    return x.type(ec['dtype']).to(ec['device'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['random', 'fixed', 'spectral', 'clean'], default='random')
    ap.add_argument('--train', default='dataset/splits/train/SBM_n1000_s1000000.txt')
    ap.add_argument('--manifest', default='dataset/splits/manifest.json')
    ap.add_argument('--steps', type=int, default=3000)
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

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    ec = make_ec(a)
    net = build_net(a, ec)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)

    man = json.load(open(a.manifest))
    rows = sorted([r for r in man if r['split'] == 'test_iid' and r['family'] == 'SBM'],
                  key=lambda r: r['seed'])[:3]
    graphs = [('G1(train)', G(a.train, ec))] + [(f'unseen{i+1}', G(r['path'], ec)) for i, r in enumerate(rows)]

    aux = {}
    for name, Gk in graphs:
        U, v0 = spectral(Gk, a.k)
        fixed = torch.randn(Gk.n, a.dim)
        aux[name] = (U, fixed)
        b = torch.tensor(v0 >= np.median(v0), device=Gk.src.device).long()
        print(f'{name}: spectral-baseline cut {cut(Gk, b)*100:.1f}%')

    G1 = graphs[0][1]
    U1, f1 = aux['G1(train)']
    net.train()
    for s in range(a.steps):
        x = make_x(G1, a.mode, a.dim, U1, f1, ec)
        p = fwd(net, G1, x)
        loss = norm_loss(p, G1)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 500 == 0 or s == a.steps - 1:
            print(f'step {s} loss_norm {loss.item():.4f} p.std {p.std().item():.3f}', flush=True)

    print(f'\nmode={a.mode}  graph | cut@0.5 | cut@median(best of 5) | p.std | p.min | p.max')
    net.eval()
    with torch.no_grad():
        for name, Gk in graphs:
            U, fx = aux[name]
            c5, cm, sd, mn, mx = [], [], [], [], []
            for _ in range(5):
                p = fwd(net, Gk, make_x(Gk, a.mode, a.dim, U, fx, ec))
                c5.append(cut(Gk, (p >= 0.5).long()))
                cm.append(cut(Gk, (p >= p.median()).long()))
                sd.append(p.std().item()); mn.append(p.min().item()); mx.append(p.max().item())
            print(f'{name} | {np.mean(c5)*100:.1f}% | {max(cm)*100:.1f}% | {np.mean(sd):.3f} | {min(mn):.2f} | {max(mx):.2f}')


if __name__ == '__main__':
    main()
