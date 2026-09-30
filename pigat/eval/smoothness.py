import torch
import numpy as np
import networkx as nx
from typing import Union, Optional

def dirichlet_energy(node_features, nx_graph, normalized=True):
    """
    Compute the Dirichlet energy of node features on a graph.

    Args:
        node_features: Node feature tensor or array
        nx_graph: NetworkX graph
        normalized: Whether to normalize the features (default True)

    Returns:
        Dirichlet energy value

    """
    if hasattr(node_features, 'cpu'):
        X = node_features.detach().cpu().numpy()
    else:
        X = np.array(node_features)

    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    # L2 normalize
    if normalized:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        X = X / norms

    num_edges = nx_graph.number_of_edges()
    if num_edges == 0:
        return 0.0

    energy = 0.0
    for i, j in nx_graph.edges():
        diff = X[i] - X[j]
        energy += np.dot(diff, diff)

    return energy / num_edges

'''
L2 normalize hiigeegui baij.
'''
'''
def dirichlet_energy(node_features: Union[torch.Tensor, np.ndarray], 
                     nx_graph: nx.Graph) -> float:
    """
    Compute the Dirichlet energy for node features on a graph.
    
    From Equation (2) in Rusch et al. 2023:
    E(X^n) = (1/v) * sum_{i in V} sum_{j in N_i} ||X^n_i - X^n_j||^2_2
    
    The actual measure used is the square root of this (Equation 3).
    Lower values indicate smoother features (more oversmoothing).
    
    Args:
        node_features: Node feature tensor or array
        nx_graph: NetworkX graph
        
    Returns:
        Square root of Dirichlet energy
    """
    # Convert to numpy if tensor
    if hasattr(node_features, 'cpu'):
        X = node_features.detach().cpu().numpy()
    else:
        X = np.array(node_features)
    
    # Handle both 1D and 2D features
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    
    v = nx_graph.number_of_nodes()
    energy = 0.0
    
    for i in nx_graph.nodes():
        for j in nx_graph.neighbors(i):
            # Compute squared L2 norm of difference
            diff = X[i] - X[j]
            energy += np.sum(diff ** 2)
    
    # Normalize by number of nodes
    energy = energy / v if v > 0 else 0.0
    
    # Return square root as per Equation (3)
    return np.sqrt(energy)
'''

def mean_average_distance(node_features: Union[torch.Tensor, np.ndarray], 
                          nx_graph: nx.Graph) -> float:
    """
    Compute the Mean Average Distance (MAD) for node features on a graph.
    
    From Equation (4) in Rusch et al. 2023:
    MAD(X^n) = (1/v) * sum_{i in V} sum_{j in N_i} [1 - cos_sim(X^n_i, X^n_j)]
    
    Higher values indicate more diverse features (less oversmoothing).
    Note: Can be problematic for scalar features with same sign.
    
    Args:
        node_features: Node feature tensor or array
        nx_graph: NetworkX graph
        
    Returns:
        Mean Average Distance
    """
    # Convert to numpy if tensor
    if hasattr(node_features, 'cpu'):
        X = node_features.detach().cpu().numpy()
    else:
        X = np.array(node_features)
    
    # Handle both 1D and 2D features
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    
    v = nx_graph.number_of_nodes()
    mad = 0.0
    
    for i in nx_graph.nodes():
        for j in nx_graph.neighbors(i):
            # Compute norms
            norm_i = np.linalg.norm(X[i])
            norm_j = np.linalg.norm(X[j])
            
            # Avoid division by zero
            if norm_i > 1e-10 and norm_j > 1e-10:
                # Compute cosine similarity
                cosine_sim = np.dot(X[i], X[j]) / (norm_i * norm_j)
                # MAD is 1 - cosine similarity
                mad += (1 - cosine_sim)
            else:
                # If either vector is zero, treat as maximally distant
                mad += 1.0
    
    # Normalize by number of nodes
    return mad / v if v > 0 else 0.0


def graph_cosine_similarity(node_features: Union[torch.Tensor, np.ndarray], 
                           nx_graph: nx.Graph) -> float:
    """
    Compute average cosine similarity across edges.
    
    Higher values indicate more similar features across connected nodes.
    This is complementary to MAD - high similarity means low MAD.
    
    Args:
        node_features: Node feature tensor or array
        nx_graph: NetworkX graph
        
    Returns:
        Average cosine similarity
    """
    # Convert to numpy if tensor
    if hasattr(node_features, 'cpu'):
        X = node_features.detach().cpu().numpy()
    else:
        X = np.array(node_features)
    
    # Handle both 1D and 2D features
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    
    total_similarity = 0.0
    edge_count = 0
    
    for u, v in nx_graph.edges:
        norm_u = np.linalg.norm(X[u])
        norm_v = np.linalg.norm(X[v])
        
        if norm_u > 1e-10 and norm_v > 1e-10:
            cosine_sim = np.dot(X[u], X[v]) / (norm_u * norm_v)
            total_similarity += cosine_sim
        edge_count += 1
    
    return total_similarity / edge_count if edge_count > 0 else 0.0


def compute_smoothness_metrics(node_features: Union[torch.Tensor, np.ndarray], 
                               nx_graph: nx.Graph) -> dict:
    """
    Compute all smoothness metrics for node features.
    
    Args:
        node_features: Node feature tensor or array
        nx_graph: NetworkX graph
        
    Returns:
        Dictionary with all metrics
    """
    return {
        'dirichlet_energy': dirichlet_energy(node_features, nx_graph),
        'mean_average_distance': mean_average_distance(node_features, nx_graph),
        'cosine_similarity': graph_cosine_similarity(node_features, nx_graph)
    }