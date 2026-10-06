"""Үл үзсэн граф дээр үнэлж, baseline-тай харьцуулна."""
import argparse, json, glob, collections, statistics as st
import numpy as np
import torch

from scripts.train_multi import (make_ec, build_net, prep, eval_graph, cut_value)


def local_search(adj, bits):
    """1-flip: cut-ыг нэмэгдүүлэх node байхгүй болтол эргүүлнэ."""
    bits = bits.copy()
    improved = True
    while improved:
        improved = False
        for v in range(len(bits)):
            same = sum(w for u, w in adj[v] if bits[u] == bits[v])
            diff = sum(w for u, w in adj[v] if bits[u] != bits[v])
            if same > diff:
                bits[v] ^= 1; improved = True
    return bits


def greedy(adj, n, rng):
    bits = -np.ones(n, dtype=int)
    for v in rng.permutation(n):
        c0 = sum(w for u, w in adj[v] if bits[u] == 1)   # 0 гэвэл 1-тэй хөршөөр cut
        c1 = sum(w for u, w in adj[v] if bits[u] == 0)
        bits[v] = 0 if c0 >= c1 else 1
    return bits


def baselines(g, e, w, seed=0):
    n = g.number_of_nodes()
    rng = np.random.RandomState(seed)
    adj = [[] for _ in range(n)]
    for (u, v), x in zip(e.T, w):
        adj[u].append((v, x)); adj[v].append((u, x))
    tot = w.sum()
    rnd = cut_value(rng.randint(0, 2, n), e, w) / tot
    gr = cut_value(greedy(adj, n, rng), e, w) / tot
    ls = cut_value(local_search(adj, rng.randint(0, 2, n)), e, w) / tot
    return rnd, gr, ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/multi/ckpt.pt')
    ap.add_argument('--manifest', default='dataset/splits/manifest.json')
    ap.add_argument('--gset', default='dataset/Gset/G*')
    ap.add_argument('--samples', type=int, default=10, help='random feature-ийн тоо (best-of-K)')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location='cpu')
    args = argparse.Namespace(**ck['args']); args.device = a.device
    cfg, ec = make_ec(args)
    net = build_net(args, ec)
    net.load_state_dict(ck['state'])

    items = [(r['split'], r['family'], r['n'], r['path']) for r in json.load(open(a.manifest))
             if r['split'].startswith('test')]
    items += [('gset', 'Gset', 0, p) for p in sorted(glob.glob(a.gset)) if '.' not in p.split('/')[-1]]

    rows = collections.defaultdict(list)
    for split, fam, n, path in items:
        g, e, w = prep(path)
        m1 = eval_graph(net, path, args, ec, 1)
        mk = eval_graph(net, path, args, ec, a.samples)
        rnd, gr, ls = baselines(g, e, w)
        rows[(split, fam, n or g.number_of_nodes())].append((m1, mk, rnd, gr, ls))

    print('split | fam | n | #g | model(1) | model(best-K) | random | greedy | 1-flip LS | model/LS')
    for k, v in sorted(rows.items()):
        c = np.array(v).mean(0) * 100
        print(*k, len(v), *[f'{x:.2f}' for x in c], f'{c[1]/c[4]*100:.1f}%')


if __name__ == '__main__':
    main()
