#!/usr/bin/env python
"""Run a single experiment (one model, one seed, one architecture) on one graph.

A thin convenience wrapper over the sweep driver:

    python scripts/run_experiment.py --graph dataset/Gset/G14 \\
        --model pigcn --seed 1 --layers 16 16 16
"""
import argparse
from argparse import SUPPRESS

from pigat.config import load_config
from pigat.sweeps import run_sweep
from pigat.training import resolve_device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="pigat single experiment (YAML config + CLI overrides)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--config', type=str, default=None, help='Path to a YAML config file')
    p.add_argument('--graph', '--graph_file', dest='graph', required=True,
                   help='Path to the graph file (format auto-detected)')
    p.add_argument('--problem', choices=['maxcut', 'mis'], default='maxcut')
    p.add_argument('--model', choices=['pigcn', 'pigatv2'], default='pigcn')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--layers', nargs='+', type=int, default=[16],
                   help='Hidden layer sizes, e.g. --layers 16 16 16')

    # Config field overrides.
    p.add_argument('--epochs', dest='number_epochs', type=int, default=SUPPRESS)
    p.add_argument('--learning-rate', '--lr', dest='learning_rate', type=float, default=SUPPRESS)
    p.add_argument('--dropout', type=float, default=SUPPRESS)
    p.add_argument('--patience', type=int, default=SUPPRESS)
    p.add_argument('--dim-embedding', dest='dim_embedding', type=int, default=SUPPRESS)
    p.add_argument('--num-heads', dest='num_heads', type=int, default=SUPPRESS)
    p.add_argument('--attn-drop', dest='attn_drop', type=float, default=SUPPRESS)
    p.add_argument('--penalty', type=int, default=SUPPRESS)
    p.add_argument('--output-dir', dest='output_dir', type=str, default=SUPPRESS)
    p.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default=SUPPRESS)
    return p


def main():
    args = build_parser().parse_args()
    ns = vars(args)

    run_keys = {'config', 'graph', 'problem', 'model', 'seed', 'layers'}
    overrides = {k: v for k, v in ns.items() if k not in run_keys}
    if 'device' in overrides:
        overrides['device'] = resolve_device(overrides['device'])

    config = load_config(ns.get('config'), overrides=overrides)

    run_sweep(
        config=config,
        graph_path=ns['graph'],
        models=[ns['model']],
        seeds=[ns['seed']],
        problem=ns['problem'],
        architectures=[ns['layers']],
        save_every=1,
    )


if __name__ == '__main__':
    main()
