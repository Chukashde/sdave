import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import torch
from scripts.seq_curriculum import make_ec, build_net, fwd
from experiments.protocol2026.scripts.train_generalization import load_graphs


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='experiments/phase1_ood/data_topology_v2/manifest.json')
    ap.add_argument('--checkpoint-dir', default='experiments/protocol2026/checkpoints/production')
    ap.add_argument('--original-manifest', default='dataset/protocol2026_v2/manifest.json')
    ap.add_argument('--output', default='experiments/phase1_ood/results/zero_shot_topology_v2.csv')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0, help='Limit graph count per family for smoke test')
    ap.add_argument('--checkpoint-limit', type=int, default=0, help='Limit checkpoint count for smoke test')
    args = ap.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    rows = json.loads(Path(args.manifest).read_text())
    if not rows or any(r['split'] != 'test_topology_ood' for r in rows):
        raise ValueError('Invalid OOD manifest')
    original = json.loads(Path(args.original_manifest).read_text())
    old_hashes = {r['graph_hash'] for r in original}
    if old_hashes & {r['graph_hash'] for r in rows}:
        raise RuntimeError('Graph hash leakage with Phase 0 dataset')
    original_paths = {r['path'] for r in original}
    if original_paths & {r['path'] for r in rows}:
        raise RuntimeError('Graph path leakage with Phase 0 dataset')
    orig_digest = sha256(args.original_manifest)
    jobs = [(model, family, seed)
            for model in ('pigcn_jk', 'pigat_jk', 'pigatv2_jk')
            for family in ('ER', 'RRG') for seed in range(1, 6)]
    if args.checkpoint_limit:
        jobs = jobs[:args.checkpoint_limit]
    fieldnames = ['model', 'train_family', 'test_family', 'train_seed', 'path',
                  'graph_seed', 'n', 'edges', 'cut05', 'cutmedian', 'cutrank50',
                  'cut05_normalized', 'cutmedian_normalized', 'cutrank50_normalized',
                  'p_std', 'inference_sec']
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model, train_family, seed in jobs:
            name = f'{model}_{train_family}_seed{seed}'
            checkpoint = Path(args.checkpoint_dir) / f'{name}.pt'
            ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
            cfg = ckpt['args']
            assert cfg['model'] == model and cfg['family'] == train_family
            assert int(cfg['seed']) == seed
            assert cfg['loss'] == 'qubo' and cfg['feature_mode'] == 'random'
            assert int(cfg['dim']) == 17
            assert ckpt['manifest_sha256'] == orig_digest
            if (set(ckpt['train_paths']) | set(ckpt['val_paths'])) & {r['path'] for r in rows}:
                raise RuntimeError('Checkpoint train/val path leakage')
            cfg = argparse.Namespace(**cfg)
            cfg.device = args.device
            ec = make_ec(cfg)
            net = build_net(cfg, ec)
            net.load_state_dict(ckpt['model_state'], strict=True)
            net.eval()
            for test_family in ('SBM', 'WS', 'GEO', 'SF'):
                subset = sorted((r for r in rows if r['family'] == test_family),
                                key=lambda r: (r['n'], r['seed']))
                if args.limit:
                    subset = subset[:args.limit]
                for row in subset:
                    # Load one graph at a time to keep GPU memory bounded.
                    _, graph, x = load_graphs([row], ec, cfg.k, cfg)[0]
                    if x.is_cuda:
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.inference_mode():
                        p = fwd(net, graph, x)
                        b05 = p >= 0.5
                        bmedian = p >= p.median()
                        order = torch.argsort(p, stable=True)
                        brank = torch.zeros_like(p, dtype=torch.bool)
                        brank[order[graph.n // 2:]] = True
                        denom = graph.w.abs().sum().clamp_min(1e-12)
                        def cut(bits):
                            return (graph.w * (bits[graph.src] != bits[graph.dst])).sum()
                        c05, cmed, crank = cut(b05), cut(bmedian), cut(brank)
                        values = torch.stack([c05, cmed, crank, denom, p.std()]).detach().cpu().tolist()
                    if x.is_cuda:
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    c05, cmed, crank, denominator, pstd = values
                    writer.writerow(dict(model=model, train_family=train_family,
                                         test_family=test_family, train_seed=seed,
                                         path=row['path'], graph_seed=row['seed'],
                                         n=row['n'], edges=row['edges'], cut05=c05,
                                         cutmedian=cmed, cutrank50=crank,
                                         cut05_normalized=c05 / denominator,
                                         cutmedian_normalized=cmed / denominator,
                                         cutrank50_normalized=crank / denominator,
                                         p_std=pstd, inference_sec=elapsed))
                    f.flush()
                print(f'DONE {name} -> {test_family}: {len(subset)} graphs', flush=True)
    print(f'SAVED {output}')


if __name__ == '__main__':
    main()
