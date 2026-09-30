import torch
import networkx as nx
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def gen_q_dict_maxcut(nx_graph: nx.Graph) -> defaultdict:
    """
    Generate QUBO matrix for MaxCut as minimization problem.
    
    Args:
        nx_graph: NetworkX graph
        weight: Edge weight (default 1)
    Returns:
        QUBO dictionary
    """
    Q_dic = defaultdict(int)
    
    for (u, v) in nx_graph.edges:
        weight = nx_graph[u][v].get('weight', 1)
        Q_dic[(u, v)] += 2 * weight
        Q_dic[(u, u)] -= weight
        Q_dic[(v, v)] -= weight
    
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


def evaluate_maxcut_solution(bitstring: torch.Tensor, 
                            nx_graph: nx.Graph) -> Dict[str, any]:
    """
    Evaluate a MaxCut solution.
    
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
    
    # Get partitions
    partition_0 = [node for node, bit in enumerate(bitstring_list) if bit == 0]
    partition_1 = [node for node, bit in enumerate(bitstring_list) if bit == 1]
    
    # Calculate cut value
    cut_value = 0
    for (u, v) in nx_graph.edges:
        if bitstring_list[u] != bitstring_list[v]:
            cut_value += nx_graph[u][v].get('weight', 1)
      
    return {
        'cut_value': cut_value,
        'partition_0': partition_0,
        'partition_1': partition_1,
        'partition_0_size': len(partition_0),
        'partition_1_size': len(partition_1),
        'bitstring': bitstring_list
    }