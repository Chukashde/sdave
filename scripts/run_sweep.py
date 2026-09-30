#!/usr/bin/env python
"""Run a depth sweep (models x seeds x layer depths) for one graph.

Configuration is YAML-primary with per-field CLI overrides:

    python scripts/run_sweep.py --config configs/maxcut_main.yaml \\
        --graph dataset/Gset/G14 --models pigcn pigatv2 --seeds 1 2 3

    # MIS sweep
    python scripts/run_sweep.py --config configs/mis.yaml \\
        --graph dataset/Gset/G14 --problem mis
"""
import argparse
from argparse import SUPPRESS

from pigat.config import load_config
from pigat.sweeps import run_sweep
from pigat.training import resolve_device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="pigat depth sweep (YAML config + CLI overrides)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Run selection (not part of the config object).
    p.add_argument('--config', type=str, default=None, help='Path to a YAML config file')
    p.add_argument('--graph', '--graph_file', dest='graph', required=True,
                   help='Path to the graph file (format auto-detected)')
    p.add_argument('--problem', choices=['maxcut', 'mis'], default='maxcut')
    p.add_argument('--models', nargs='+', default=['pigcn', 'pigatv2'],
                   choices=['pigcn', 'pigatv2'])
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    p.add_argument('--save-every', dest='save_every', type=int, default=10)

    # Depth range (overrides hidden_layers_range when both given).
    p.add_argument('--min-layers', dest='min_layers', type=int, default=SUPPRESS)
    p.add_argument('--max-layers', dest='max_layers', type=int, default=SUPPRESS)

    # Config field overrides (SUPPRESS => unset flags do not clobber YAML).
    p.add_argument('--epochs', dest='number_epochs', type=int, default=SUPPRESS)
    p.add_argument('--learning-rate', '--lr', dest='learning_rate', type=float, default=SUPPRESS)
    p.add_argument('--dropout', type=float, default=SUPPRESS)
    p.add_argument('--patience', type=int, default=SUPPRESS)
    p.add_argument('--dim-embedding', dest='dim_embedding', type=int, default=SUPPRESS)
    p.add_argument('--hidden-dim-size', dest='fixed_hidden_dim_size', type=int, default=SUPPRESS)
    p.add_argument('--num-heads', dest='num_heads', type=int, default=SUPPRESS)
    p.add_argument('--attn-drop', dest='attn_drop', type=float, default=SUPPRESS)
    p.add_argument('--penalty', type=int, default=SUPPRESS)
    p.add_argument('--output-dir', dest='output_dir', type=str, default=SUPPRESS)
    p.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default=SUPPRESS)
    return p


def main():
    args = build_parser().parse_args()
    ns = vars(args)

    # Separate run-selection args from config-field overrides.
    run_keys = {'config', 'graph', 'problem', 'models', 'seeds', 'save_every',
                'min_layers', 'max_layers'}
    overrides = {k: v for k, v in ns.items() if k not in run_keys}

    if 'device' in overrides:
        overrides['device'] = resolve_device(overrides['device'])

    config = load_config(ns.get('config'), overrides=overrides)

    # A min/max layer range overrides hidden_layers_range from the YAML.
    if 'min_layers' in ns and 'max_layers' in ns:
        config.hidden_layers_range = list(range(ns['min_layers'], ns['max_layers'] + 1))

    run_sweep(
        config=config,
        graph_path=ns['graph'],
        models=ns['models'],
        seeds=ns['seeds'],
        problem=ns['problem'],
        save_every=ns['save_every'],
    )


if __name__ == '__main__':
    main()
