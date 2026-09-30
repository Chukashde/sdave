import torch
import networkx as nx
import numpy as np
from collections import defaultdict
from time import time
from typing import Dict, List, Tuple, Optional
from itertools import combinations


def gen_combinations(iterable, chunk_size):
    """
    Generator that yields chunks of combinations.
    
    Args:
        iterable: Iterable to chunk
        chunk_size: Size of each chunk
        
    Yields:
        Chunks of combinations
    """
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def gen_q_dict_mis(nx_G, penalty=2):
    """
    Helper function to generate QUBO matrix for MIS as minimization problem.
    
    Input:
        nx_G: graph as networkx graph object (assumed to be unweigthed)
    Output:
        Q_dic: QUBO as defaultdict
    """

    # Initialize our Q matrix
    Q_dic = defaultdict(int)

    # Update Q matrix for every edge in the graph
    # all off-diagonal terms get penalty
    for (u, v) in nx_G.edges:
        Q_dic[(u, v)] = penalty

    # all diagonal terms get -1
    for u in nx_G.nodes:
        Q_dic[(u, u)] = -1

    return Q_dic


def qubo_dict_to_torch(nx_graph: nx.Graph, 
                      Q: defaultdict, 
                      torch_dtype: torch.dtype = None, 
                      torch_device: str = None) -> torch.Tensor:
    """
    Convert QUBO dictionary to torch tensor.
    
    Args:
        nx_graph: NetworkX graph
        Q: QUBO dictionary
        torch_dtype: Target torch data type
        torch_device: Target device
        
    Returns:
        QUBO matrix as torch tensor
    """
    n_nodes = len(nx_graph.nodes)
    Q_mat = torch.zeros(n_nodes, n_nodes)
    
    for (x_coord, y_coord), val in Q.items():
        Q_mat[x_coord][y_coord] = val

    if torch_dtype is not None:
        Q_mat = Q_mat.type(torch_dtype)
    if torch_device is not None:
        Q_mat = Q_mat.to(torch_device)

    return Q_mat


def loss_func(probs: torch.Tensor, Q_mat: torch.Tensor) -> torch.Tensor:
    """
    Compute cost value for given probability and Q matrix.
    
    Args:
        probs: Node probabilities
        Q_mat: QUBO matrix
        
    Returns:
        Cost value
    """
    probs_ = torch.unsqueeze(probs, 1)
    cost = (probs_.T @ Q_mat @ probs_).squeeze()
    return cost


# Calculate results given bitstring and graph definition, includes check for violations
def postprocess_gnn_mis(best_bitstring, nx_graph):
    """
    helper function to postprocess MIS results

    Input:
        best_bitstring: bitstring as torch tensor
    Output:
        size_mis: Size of MIS (int)
        ind_set: MIS (list of integers)
        number_violations: number of violations of ind.set condition
    """

    # get bitstring as list
    bitstring_list = list(best_bitstring)

    # compute cost
    size_mis = sum(bitstring_list)

    # get independent set
    ind_set = set([node for node, entry in enumerate(bitstring_list) if entry == 1])
    edge_set = set(list(nx_graph.edges))

    print('Calculating violations...')
    # check for violations
    number_violations = 0
    for ind_set_chunk in gen_combinations(combinations(ind_set, 2), 100000):
        number_violations += len(set(ind_set_chunk).intersection(edge_set))

    return size_mis, ind_set, number_violations


def is_valid_independent_set(bitstring: List[int], nx_graph: nx.Graph) -> bool:
    """
    Check if a bitstring represents a valid independent set.
    
    Args:
        bitstring: Solution bitstring
        nx_graph: NetworkX graph
        
    Returns:
        True if valid independent set, False otherwise
    """
    selected_nodes = [node for node, bit in enumerate(bitstring) if bit == 1]
    
    # Check if any two selected nodes are adjacent
    for i, node1 in enumerate(selected_nodes):
        for j, node2 in enumerate(selected_nodes[i+1:], i+1):
            if nx_graph.has_edge(node1, node2):
                return False
    
    return True


def evaluate_mis_solution(bitstring: torch.Tensor, 
                         nx_graph: nx.Graph) -> Dict[str, any]:
    """
    Evaluate a Maximum Independent Set solution.
    
    Args:
        bitstring: Solution bitstring
        nx_graph: NetworkX graph
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Convert to list if tensor
    if hasattr(bitstring, 'tolist'):
        bitstring_list = bitstring.tolist()
    else:
        bitstring_list = list(bitstring)
    
    # Get selected nodes
    independent_set = [node for node, bit in enumerate(bitstring_list) if bit == 1]
    
    # Check validity
    is_valid = is_valid_independent_set(bitstring_list, nx_graph)
    
    # Calculate penalty (number of edges within the independent set)
    penalty_count = 0
    for i, node1 in enumerate(independent_set):
        for j, node2 in enumerate(independent_set[i+1:], i+1):
            if nx_graph.has_edge(node1, node2):
                penalty_count += 1
    
    # Calculate metrics
    independence_ratio = len(independent_set) / nx_graph.number_of_nodes() if nx_graph.number_of_nodes() > 0 else 0
    
    return {
        'independent_set_size': len(independent_set),
        'independent_set': independent_set,
        'is_valid': is_valid,
        'penalty_count': penalty_count,
        'independence_ratio': independence_ratio,
        'bitstring': bitstring_list,
        'objective_value': len(independent_set) if is_valid else len(independent_set) - penalty_count * 10  # Heavy penalty for invalid solutions
    }


def mis_to_vertex_cover(mis_bitstring: List[int]) -> List[int]:
    """
    Convert Maximum Independent Set solution to Vertex Cover solution.
    The complement of a maximum independent set is a minimum vertex cover.
    
    Args:
        mis_bitstring: MIS solution bitstring
        
    Returns:
        Vertex cover bitstring
    """
    return [1 - bit for bit in mis_bitstring]


def vertex_cover_to_mis(vc_bitstring: List[int]) -> List[int]:
    """
    Convert Vertex Cover solution to Maximum Independent Set solution.
    
    Args:
        vc_bitstring: Vertex cover solution bitstring
        
    Returns:
        MIS bitstring
    """
    return [1 - bit for bit in vc_bitstring]