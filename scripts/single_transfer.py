import argparse, glob, numpy as np, torch
from pigat.training import graph_utils as dgl
from pigat.sweeps.problems import MaxCutProblem
from pigat.eval.maxcut import loss_func
from scripts.train_multi import (make_ec, build_net, prep, make_feats,
                                 forward, cut_value)
from scripts.evaluate import baselines


def infer(net, path, args, ec, K=10):
    g, e, w = prep(path)
    dg = dgl.from_networkx(nx_graph=g)
    net.eval()
    best_t = best_m = 0.0
    with torch.no_grad():
        for _ in range(K):
            p = forward(net, dg, make_feats(g, args.dim, ec)).float().cpu().numpy()
            t = (p >= 0.5).astype(int)
            m = (p >= np.median(p)).astype(int)
            best_t = max(best_t, cut_value(t, e, w) / w.sum())
            best_m = max(best_m, cut_value(m, e, w) / w.sum())
    return best_t, best_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='dataset/synthetic/ER_n1000_0.txt')
    ap.add_argument('--tests', default='dataset/synthetic/ER_n1000_[1-4].txt')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--heads', type=int, default=1)
    ap.add_argument('--dim', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--model', default='pigatv2')
    ap.add_argument('--config', default='configs/maxcut_main.yaml')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    cfg, ec = make_ec(args)
    net = build_net(args, ec)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    g, e, w = prep(args.train)
    dg = dgl.from_networkx(nx_graph=g)
    Q = MaxCutProblem().build_q(g, cfg, ec['dtype'], ec['device'])

    net.train()
    for s in range(args.steps):
        x = make_feats(g, args.dim, ec)
        loss = loss_func(forward(net, dg, x), Q)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 200 == 0:
            print(f'step {s} loss {loss.item():.1f}', flush=True)

    print('\ngraph | seen? | cut(0.5) | cut(median) | random | greedy | LS | median/LS')
    paths = [args.train] + sorted(glob.glob(args.tests))
    for i, p in enumerate(paths):
        t, m = infer(net, p, args, ec)
        gg, ee, ww = prep(p)
        rnd, gr, ls = baselines(gg, ee, ww)
        print(f'{p.split("/")[-1]} | {"train" if i == 0 else "unseen"} | '
              f'{t*100:.1f} | {m*100:.1f} | {rnd*100:.1f} | {gr*100:.1f} | '
              f'{ls*100:.1f} | {m/ls*100:.1f}%')


if __name__ == '__main__':
    main()
