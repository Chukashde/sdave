"""Problem strategies for the sweep driver.

Each strategy encapsulates the parts of an experiment that differ between
MAX-CUT and MIS: building the QUBO tensor, evaluating a solution, and the
problem-specific result-row fields. The training loop and RNG path are shared
by the driver and identical across problems.
"""
from typing import Any, Dict

import networkx as nx
import torch

from pigat.eval import maxcut as _maxcut
from pigat.eval import mis as _mis


class MaxCutProblem:
    name = 'maxcut'
    csv_suffix = 'sweep'

    def build_q(self, nx_graph: nx.Graph, config, dtype, device) -> torch.Tensor:
        q_dict = _maxcut.gen_q_dict_maxcut(nx_graph)
        return _maxcut.qubo_dict_to_torch(nx_graph, q_dict, torch_dtype=dtype, torch_device=device)

    def evaluate(self, best_bitstring, nx_graph: nx.Graph) -> Dict[str, Any]:
        return _maxcut.evaluate_maxcut_solution(best_bitstring, nx_graph)

    def solution_fields(self, eval_results, graph_info, config) -> Dict[str, Any]:
        return {'cut_value': eval_results['cut_value']}

    def console_line(self, eval_results, graph_info) -> str:
        n_edges = graph_info['n_edges']
        ratio = eval_results['cut_value'] / n_edges if n_edges > 0 else 0.0
        return f"  Cut: {eval_results['cut_value']}/{n_edges} ({ratio*100:.1f}%)"


class MISProblem:
    name = 'mis'
    csv_suffix = 'mis_sweep'

    def build_q(self, nx_graph: nx.Graph, config, dtype, device) -> torch.Tensor:
        q_dict = _mis.gen_q_dict_mis(nx_graph, penalty=config.penalty)
        return _mis.qubo_dict_to_torch(nx_graph, q_dict, torch_dtype=dtype, torch_device=device)

    def evaluate(self, best_bitstring, nx_graph: nx.Graph) -> Dict[str, Any]:
        return _mis.evaluate_mis_solution(best_bitstring, nx_graph)

    def solution_fields(self, eval_results, graph_info, config) -> Dict[str, Any]:
        fields = {
            'penalty': config.penalty,
            'independent_set_size': eval_results['independent_set_size'],
            'independence_ratio': round(eval_results['independence_ratio'], 4),
            'is_valid': eval_results['is_valid'],
            'penalty_count': eval_results['penalty_count'],
            'objective_value': eval_results['objective_value'],
        }
        if getattr(config, 'compare_vertex_cover', False):
            vc_bitstring = _mis.mis_to_vertex_cover(eval_results['bitstring'])
            vc_size = sum(vc_bitstring)
            n_nodes = graph_info['n_nodes']
            fields['vertex_cover_size'] = vc_size
            fields['vertex_cover_ratio'] = round(vc_size / n_nodes if n_nodes > 0 else 0, 4)
        return fields

    def console_line(self, eval_results, graph_info) -> str:
        n_nodes = graph_info['n_nodes']
        return (f"  Independent Set: {eval_results['independent_set_size']}/{n_nodes} "
                f"({eval_results['independence_ratio']*100:.1f}%) Valid: {eval_results['is_valid']}")


PROBLEMS = {
    'maxcut': MaxCutProblem,
    'mis': MISProblem,
}


def get_problem(name):
    """Return a problem strategy instance for a name (or pass through an instance)."""
    if isinstance(name, str):
        try:
            return PROBLEMS[name]()
        except KeyError:
            raise ValueError(f"Unknown problem {name!r} (expected one of {sorted(PROBLEMS)})")
    return name
