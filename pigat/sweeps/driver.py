"""Single problem-agnostic sweep driver.

Replaces the four duplicated sweep loops (experiments/runner.py,
experiments/runner_mis.py, batch_experiment.py, run_all_graphs.py) with one
loop over models x seeds x architectures. Each cell reproduces the exact
operation order of the original run_single_experiment so cut values are
unchanged. Results are written to one superset CSV schema, and the effective
config is saved next to it.
"""
import os
from datetime import datetime
from time import time
from typing import Any, Dict, List, Optional

import dgl
import networkx as nx
import pandas as pd

from pigat.data import load_graph
from pigat.eval.smoothness import compute_smoothness_metrics
from pigat.sweeps.problems import get_problem
from pigat.training import Trainer, setup_environment


def _run_cell(config, problem, nx_graph: nx.Graph, graph_info: Dict[str, Any],
              model_name: str, hidden_dim_layers: List[int], seed: int,
              graph_name: str) -> Dict[str, Any]:
    """Run one (model, seed, architecture) cell. Operation order is identical to
    the original run_single_experiment (RNG-critical)."""
    setup_environment(seed)

    experiment_start = time()

    exp_config = config.to_dict()
    exp_config['hidden_dim'] = hidden_dim_layers
    exp_config['seed'] = seed

    dgl_graph = dgl.from_networkx(nx_graph=nx_graph)
    pass  # DGL graph stays on CPU

    q_torch = problem.build_q(nx_graph, config, exp_config['dtype'], exp_config['device'])

    trainer = Trainer(exp_config)
    net, embed, optimizer = trainer.create_model_and_optimizer(
        model_name, graph_info['n_nodes'])

    print(f"\nTraining {model_name} with {len(hidden_dim_layers)} layers...")
    train_results = trainer.train(net, embed, optimizer, dgl_graph, q_torch, verbose=False)

    eval_results = problem.evaluate(train_results['best_bitstring'], nx_graph)

    final_metrics = compute_smoothness_metrics(
        train_results['final_last_hidden'] if train_results['final_last_hidden'] is not None
        else train_results['final_bitstring'],
        nx_graph)
    best_metrics = compute_smoothness_metrics(
        train_results['best_last_hidden'] if train_results['best_last_hidden'] is not None
        else train_results['best_bitstring'],
        nx_graph)

    total_time = time() - experiment_start

    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': model_name,
        'graph_name': graph_name or config.default_graph_name,
        'seed': seed,

        # Graph info
        'n_nodes': graph_info['n_nodes'],
        'n_edges': graph_info['n_edges'],
        'graph_density': graph_info['density'],
        'is_connected': graph_info['is_connected'],
        'n_components': graph_info['n_components'],

        # Model architecture
        'dim_embedding': exp_config['dim_embedding'],
        'hidden_dim_layers': str(hidden_dim_layers),
        'num_hidden_layers': len(hidden_dim_layers),
        'hidden_dim_size': hidden_dim_layers[0] if hidden_dim_layers else 0,
        'dropout': exp_config['dropout'],
        'learning_rate': exp_config['learning_rate'],
        'num_heads': exp_config['num_heads'],
        'attn_drop': exp_config['attn_drop'],

        # Training info
        'max_epochs': exp_config['number_epochs'],
        'final_epoch': train_results['final_epoch'],
        'best_epoch': train_results['best_epoch'],
        'converged': train_results['converged'],
        'training_time': round(train_results['training_time'], 4),
        'total_time': round(total_time, 4),

        # Solution quality (shared)
        'final_loss': round(train_results['final_loss'], 6),
        'best_loss': round(train_results['best_loss'], 6),

        # Smoothness metrics (final)
        'final_dirichlet_energy': round(final_metrics['dirichlet_energy'], 4),
        'final_mad': round(final_metrics['mean_average_distance'], 4),
        'final_cosine_sim': round(final_metrics['cosine_similarity'], 4),

        # Smoothness metrics (best)
        'best_dirichlet_energy': round(best_metrics['dirichlet_energy'], 4),
        'best_mad': round(best_metrics['mean_average_distance'], 4),
        'best_cosine_sim': round(best_metrics['cosine_similarity'], 4),
    }

    # Problem-specific columns (cut_value for MAX-CUT; IS metrics for MIS).
    row.update(problem.solution_fields(eval_results, graph_info, config))

    print(problem.console_line(eval_results, graph_info))
    print(f"  Dirichlet Energy: {best_metrics['dirichlet_energy']:.4f}")
    print(f"  MAD: {best_metrics['mean_average_distance']:.4f}")

    return row


def build_architectures(config, architectures: Optional[List[List[int]]] = None) -> List[List[int]]:
    """Depth sweep architectures from config, unless explicit ones are given."""
    if architectures is not None:
        return architectures
    return [[config.fixed_hidden_dim_size] * num_layers
            for num_layers in config.hidden_layers_range]


def run_sweep(config,
              graph_path: str,
              models: List[str],
              seeds: List[int],
              problem='maxcut',
              architectures: Optional[List[List[int]]] = None,
              save_every: int = 10) -> pd.DataFrame:
    """Run a sweep over models x seeds x architectures for one graph.

    Args:
        config: PigatConfig instance.
        graph_path: path to the graph file (format auto-detected).
        models: model names, e.g. ['pigcn', 'pigatv2'].
        seeds: random seeds.
        problem: 'maxcut' or 'mis' (or a problem strategy instance).
        architectures: explicit list of hidden-dim lists; if None, a depth
            sweep is built from config.hidden_layers_range/fixed_hidden_dim_size.
        save_every: write the CSV every N cells.

    Returns:
        DataFrame of all result rows.
    """
    problem = get_problem(problem)
    architectures = build_architectures(config, architectures)

    nx_graph, graph_info = load_graph(graph_path)
    graph_name = os.path.basename(graph_path).split('.')[0]

    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(config.output_dir,
                                f'{graph_name}_{problem.csv_suffix}_{timestamp}.csv')

    # Persist the effective config next to the results for reproducibility.
    config.save_yaml(os.path.splitext(results_file)[0] + '.config.yaml')

    total = len(models) * len(seeds) * len(architectures)
    print("=" * 80)
    print(f"{problem.name.upper()} SWEEP | graph={graph_name} | {total} experiments")
    print(f"Results: {results_file}")
    print("=" * 80)

    rows: List[Dict[str, Any]] = []
    count = 0
    for model_name in models:
        for seed in seeds:
            for arch in architectures:
                count += 1
                print(f"\n[{count}/{total}] {model_name} | seed {seed} | {len(arch)} layers")
                try:
                    row = _run_cell(config, problem, nx_graph, graph_info,
                                    model_name, arch, seed, graph_name)
                    rows.append(row)
                    if count % save_every == 0:
                        pd.DataFrame(rows).to_csv(results_file, index=False)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    rows.append({
                        'model_name': model_name,
                        'seed': seed,
                        'num_hidden_layers': len(arch),
                        'error': str(e),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(results_file, index=False)
    print(f"\nSaved {len(rows)} results to {results_file}")
    return df
