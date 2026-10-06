"""G1 -> G2 -> G3 ... дараалан сургаж, unseen test дээр үнэлнэ."""
import argparse, json, copy, numpy as np, torch
from scripts.run_experiment import load_config, resolve_device
from pigat.data import load_graph
from pigat.models import create_model
from pigat.training import graph_utils as dgl


def make_ec(a):
    ov = {'device': resolve_device(a.device), 'num_heads': a.heads,
          'dim_embedding': a.dim, 'learning_rate': a.lr}
    cfg = load_config(a.config, overrides=ov)
    ec = cfg.to_dict()
    ec['hidden_dim'] = [cfg.fixed_hidden_dim_size] * a.layers
    return ec


def build_net(a, ec):
    return create_model(a.model, ec).type(ec['dtype']).to(ec['device'])


class G:
    """Нэг graph: edge tensor + жин."""
    def __init__(self, path, ec, ref=None):
        g, _ = load_graph(path)
        e = np.array(list(g.edges()), dtype=np.int64).T
        w = np.array([g[u][v].get('weight', 1.0) for u, v in g.edges()])
        dev = ec['device']
        self.g, self.n, self.ec, self.ref = g, g.number_of_nodes(), ec, ref
        self.dg = dgl.from_networkx(nx_graph=g)
        self.src = torch.tensor(e[0], device=dev)
        self.dst = torch.tensor(e[1], device=dev)
        self.w = torch.tensor(w, dtype=torch.float32, device=dev)
        self.wsum = float(w.sum())
        d = torch.tensor([g.degree(i) for i in range(self.n)], dtype=torch.float32)
        self.deg = (d / d.mean().clamp(min=1e-6))

    def feats(self, dim):
        x = torch.randn(self.n, dim)
        x[:, 0] = self.deg
        return x.type(self.ec['dtype']).to(self.ec['device'])


def fwd(net, G_, x):
    p, _ = net(G_.dg, x)
    return (p[:, 0] if p.dim() > 1 else p).float()


def norm_loss(p, G_):
    # = p^T Q p / sum(w) = -mean_w (p_u - p_v)^2  in [-1, 0]
    d = p[G_.src] - p[G_.dst]
    return -(G_.w * d * d).sum() / G_.wsum


@torch.no_grad()
def cut_ratio(net, G_, dim, K=5):
    net.eval()
    best = 0.0
    for _ in range(K):
        p = fwd(net, G_, G_.feats(dim))
        b = (p >= p.median()).long()
        cut = (G_.w * (b[G_.src] != b[G_.dst])).sum().item() / G_.wsum
        best = max(best, cut)
    return best


def evaluate(net, sets, dim):
    out = {}
    for name, gs in sets.items():
        r = [cut_ratio(net, g, dim) for g in gs]
        rr = [c / g.ref for c, g in zip(r, gs)] if gs[0].ref else None
        out[name] = (float(np.mean(r)), float(np.mean(rr)) if rr else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--order', nargs='+', required=True, help='G1 G2 G3 ... graph path')
    ap.add_argument('--manifest', default='dataset/splits/manifest.json')
    ap.add_argument('--test_splits', nargs='+',
                    default=['test_iid', 'test_size', 'test_degree', 'test_family'])
    ap.add_argument('--per_split', type=int, default=10, help='split бүрээс авах graph')
    ap.add_argument('--steps', type=int, default=1000, help='graph бүр дээрх step')
    ap.add_argument('--first_steps', type=int, default=None)
    ap.add_argument('--carry_opt', action='store_true', help='Adam төлөв + step тоог дамжуулна')
    ap.add_argument('--reset', action='store_true', help='scratch baseline: үе шат бүрт шинэ init')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--heads', type=int, default=1)
    ap.add_argument('--dim', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--model', default='pigatv2')
    ap.add_argument('--config', default='configs/maxcut_main.yaml')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='results/seq_curriculum.json')
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    ec = make_ec(a)

    man = json.load(open(a.manifest))
    sets = {}
    for sp in a.test_splits:
        rows = [r for r in man if r['split'] == sp]
        # family/size бүрээс жигд сонгох
        rows = sorted(rows, key=lambda r: (r['family'], r['n'], r['seed']))[::max(1, len(rows) // a.per_split)]
        sets[sp] = [G(r['path'], ec, r['ref_ratio']) for r in rows[:a.per_split]]

    train_graphs = [G(p, ec) for p in a.order]
    net = build_net(a, ec)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    total_step, log = 0, []

    for k, Gk in enumerate(train_graphs, 1):
        if a.reset:
            net = build_net(a, ec)
            opt = torch.optim.Adam(net.parameters(), lr=a.lr)
        elif not a.carry_opt:
            opt = torch.optim.Adam(net.parameters(), lr=a.lr)   # жин л дамжина
        steps = a.first_steps if (k == 1 and a.first_steps) else a.steps

        net.train()
        for s in range(steps):
            loss = norm_loss(fwd(net, Gk, Gk.feats(a.dim)), Gk)
            opt.zero_grad(); loss.backward(); opt.step()
            total_step += 1
        last = loss.item()

        res = evaluate(net, sets, a.dim)
        seen = {f'G{j}': (cut_ratio(net, train_graphs[j - 1], a.dim), None) for j in range(1, k + 1)}
        log.append(dict(stage=k, graph=a.order[k - 1], total_step=total_step,
                        train_loss=last, test=res, seen=seen))
        print(f'\n== stage {k}: trained on {a.order[k-1].split("/")[-1]} '
              f'(total step {total_step}, loss_norm {last:.4f}) ==')
        for n_, (c, r) in res.items():
            print(f'  {n_:12s} cut/edge {c*100:5.1f}%   cut/ref {r:.3f}')
        print('  seen (forgetting): ' + '  '.join(f'{n_}={v[0]*100:.1f}%' for n_, v in seen.items()), flush=True)

    json.dump(log, open(a.out, 'w'), indent=1)
    print('saved', a.out)


if __name__ == '__main__':
    main()
