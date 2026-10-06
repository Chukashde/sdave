"""Олон граф дээр сургах (inductive): embedding нь граф бүрээс тооцогдоно."""
import argparse, json, os, random
import numpy as np
import torch

from scripts.run_experiment import load_config, resolve_device
from pigat.data import load_graph
from pigat.models import create_model
from pigat.training import graph_utils as dgl
from pigat.training import setup_environment
from pigat.sweeps.problems import MaxCutProblem
from pigat.eval.maxcut import loss_func

PROBLEM = MaxCutProblem()


def make_ec(args):
    ov = {'device': resolve_device(args.device), 'num_heads': args.heads,
          'dim_embedding': args.dim, 'learning_rate': args.lr}
    cfg = load_config(args.config, overrides=ov)
    ec = cfg.to_dict()
    ec['hidden_dim'] = [cfg.fixed_hidden_dim_size] * args.layers
    return cfg, ec


def build_net(args, ec):
    net = create_model(args.model, ec)
    return net.type(ec['dtype']).to(ec['device'])


def make_feats(nx_g, dim, ec):
    """[N, dim]: 1-р багана = degree/дундаж degree, үлдсэн нь шинэ random."""
    n = nx_g.number_of_nodes()
    deg = torch.tensor([nx_g.degree(i) for i in range(n)], dtype=torch.float32)
    x = torch.randn(n, dim)
    x[:, 0] = deg / deg.mean().clamp(min=1e-6)
    return x.type(ec['dtype']).to(ec['device'])


def cut_value(bits, edges, w):
    return float((w * (bits[edges[0]] != bits[edges[1]])).sum())


def prep(path):
    g, _ = load_graph(path)
    e = np.array([(u, v) for u, v in g.edges()], dtype=np.int64).T
    w = np.array([g[u][v].get('weight', 1.0) for u, v in g.edges()])
    return g, e, w


def forward(net, dg, x):
    p, _ = net(dg, x)
    return p[:, 0] if p.dim() > 1 else p


@torch.no_grad()
def eval_graph(net, path, args, ec, samples=1):
    """Дискрет cut / edge-ийн харьцаа. samples>1 бол хамгийн сайныг авна."""
    g, e, w = prep(path)
    dg = dgl.from_networkx(nx_graph=g)
    net.eval()
    best = 0.0
    for _ in range(samples):
        p = forward(net, dg, make_feats(g, args.dim, ec))
        bits = (p >= 0.5).cpu().numpy().astype(int)
        best = max(best, cut_value(bits, e, w) / w.sum())
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/maxcut_main.yaml')
    ap.add_argument('--model', default='pigatv2')
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--heads', type=int, default=1)
    ap.add_argument('--dim', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--steps', type=int, default=200, help='граф бүр дээрх алхам')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--manifest', default='dataset/splits/manifest.json')
    ap.add_argument('--out', default='results/multi/ckpt.pt')
    args = ap.parse_args()

    setup_environment(args.seed)
    random.seed(args.seed)
    cfg, ec = make_ec(args)
    net = build_net(args, ec)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    m = json.load(open(args.manifest))
    train = [r['path'] for r in m if r['split'] == 'train']
    val = [r['path'] for r in m if r['split'] == 'val']
    print(f'train={len(train)} val={len(val)} layers={args.layers} heads={args.heads}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_val = -1.0
    for rd in range(args.rounds):
        random.shuffle(train)
        net.train()
        for path in train:
            g, _, _ = prep(path)
            dg = dgl.from_networkx(nx_graph=g)
            q = PROBLEM.build_q(g, cfg, ec['dtype'], ec['device'])
            n = g.number_of_nodes()
            for _ in range(args.steps):
                x = make_feats(g, args.dim, ec)      # шинэ random feature
                loss = loss_func(forward(net, dg, x), q) / n
                opt.zero_grad(); loss.backward(); opt.step()
        v = float(np.mean([eval_graph(net, p, args, ec) for p in val]))
        tag = ''
        if v > best_val:
            best_val = v
            torch.save({'state': net.state_dict(), 'args': vars(args)}, args.out)
            tag = '  <- best'
        print(f'round {rd+1}/{args.rounds} | val cut ratio {v*100:.2f}%{tag}', flush=True)
    print(f'best val: {best_val*100:.2f}%  -> {args.out}')


if __name__ == '__main__':
    main()
