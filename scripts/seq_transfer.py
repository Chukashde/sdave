import argparse, copy, glob, numpy as np, torch
from pigat.training import graph_utils as dgl
from pigat.sweeps.problems import MaxCutProblem
from pigat.eval.maxcut import loss_func
from scripts.train_multi import (make_ec, build_net, prep, make_feats,
                                 forward, cut_value)


def train(net, path, steps, args, cfg, ec):
    g, e, w = prep(path)
    dg = dgl.from_networkx(nx_graph=g)
    Q = MaxCutProblem().build_q(g, cfg, ec['dtype'], ec['device'])
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    net.train()
    for s in range(steps):
        x = make_feats(g, args.dim, ec)
        loss = loss_func(forward(net, dg, x), Q)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def infer(net, path, args, ec, K=10):
    g, e, w = prep(path)
    dg = dgl.from_networkx(nx_graph=g)
    net.eval()
    best = 0.0
    with torch.no_grad():
        for _ in range(K):
            p = forward(net, dg, make_feats(g, args.dim, ec)).float().cpu().numpy()
            m = (p >= np.median(p)).astype(int)
            best = max(best, cut_value(m, e, w) / w.sum())
    return best * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='dataset/synthetic/ER_n1000_0.txt')
    ap.add_argument('--tests', default='dataset/synthetic/ER_n1000_[1-4].txt')
    ap.add_argument('--steps', type=int, default=3000, help='эхний graph дээрх step')
    ap.add_argument('--ft_steps', type=int, default=500, help='unseen graph дээрх fine-tune step')
    ap.add_argument('--chain', action='store_true')
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--heads', type=int, default=1)
    ap.add_argument('--dim', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--model', default='pigatv2')
    ap.add_argument('--config', default='configs/maxcut_main.yaml')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    cfg, ec = make_ec(args)
    base = build_net(args, ec)
    train(base, args.train, args.steps, args, cfg, ec)
    print(f'base trained on {args.train.split("/")[-1]}: cut {infer(base, args.train, args, ec):.1f}%')

    print('\ngraph | zero-shot | warm-start(K) | scratch(K)')
    for p in sorted(glob.glob(args.tests)):
        zs = infer(base, p, args, ec)

        warm = copy.deepcopy(base)
        train(warm, p, args.ft_steps, args, cfg, ec)
        ws = infer(warm, p, args, ec)

        scratch = build_net(args, ec)
        train(scratch, p, args.ft_steps, args, cfg, ec)
        sc = infer(scratch, p, args, ec)

        print(f'{p.split("/")[-1]} | {zs:.1f} | {ws:.1f} | {sc:.1f}', flush=True)
        if args.chain:
            base = warm


if __name__ == '__main__':
    main()
