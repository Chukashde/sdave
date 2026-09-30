import networkx as nx
from typing import Dict, Any, Optional, Tuple


def detect_format(filename: str) -> str:
    """Detect a graph file's format from its header line.

    Gset files start with a two-token header "n_nodes n_edges"; adjacency-list
    files start with a single-token header "n_nodes". This token-count rule is
    unambiguous for every file shipped under dataset/.

    Args:
        filename: Path to the graph file

    Returns:
        'gset' or 'adj'
    """
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = line.split()
                break
        else:
            raise ValueError(f"Empty graph file: {filename}")

    if len(tokens) == 1:
        return 'adj'
    if len(tokens) >= 2 and tokens[1].lstrip('-').isdigit():
        return 'gset'
    # Fall back to gset for any other multi-token header.
    return 'gset'


def load_graph(filename: str,
               force: Optional[str] = None) -> Tuple[nx.Graph, Dict[str, Any]]:
    """Load a graph, auto-detecting Gset vs adjacency-list format.

    Args:
        filename: Path to the graph file
        force: Optional format override, 'gset' or 'adj'. When None the format
            is detected from the header (see detect_format).

    Returns:
        Tuple of (NetworkX graph, graph information dictionary)
    """
    fmt = force or detect_format(filename)
    if fmt == 'gset':
        return read_gset(filename)
    if fmt == 'adj':
        return read_graph(filename)
    raise ValueError(f"Unknown graph format: {fmt!r} (expected 'gset' or 'adj')")


def read_gset(filename: str) -> Tuple[nx.Graph, Dict[str, Any]]:
    """
    Read a graph from GSET format file.
    
    Args:
        filename: Path to the GSET format file
        
    Returns:
        Tuple of (NetworkX graph, graph information dictionary)
    """
    nx_graph = nx.Graph()
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        # Parse header
        first_line = lines[0].strip().split()
        n_nodes = int(first_line[0])
        n_edges = int(first_line[1])
        
        # Add nodes
        nx_graph.add_nodes_from(range(n_nodes))
        
        # Parse edges
        for i in range(1, len(lines)):
            line = lines[i].strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    # GSET format uses 1-indexed nodes
                    node1 = int(parts[0]) - 1
                    node2 = int(parts[1]) - 1
                    weight = float(parts[2]) if len(parts) >= 3 else 1.0
                    nx_graph.add_edge(node1, node2, weight=weight)
        
    # Verify edge count
    actual_edges = nx_graph.number_of_edges()
    if actual_edges != n_edges:
        print(f"Warning: Expected {n_edges} edges, but read {actual_edges} edges")
    
    # Collect graph information
    graph_info = {
        'n_nodes': nx_graph.number_of_nodes(),
        'n_edges': nx_graph.number_of_edges(),
        'density': nx.density(nx_graph),
        'is_weighted': any(data.get('weight', 1) != 1 for _, _, data in nx_graph.edges(data=True)),
        'is_connected': nx.is_connected(nx_graph),
        'n_components': nx.number_connected_components(nx_graph)
    }
    
    print(f"Loaded graph with {graph_info['n_nodes']} nodes and {graph_info['n_edges']} edges")
    print(f"Density: {graph_info['density']:.4f}")
    print(f"Connected: {graph_info['is_connected']} ({graph_info['n_components']} components)")
    
    return nx_graph, graph_info

def read_graph(filename: str) -> Tuple[nx.Graph, Dict[str, Any]]:
    """
    Read a graph from adjacency list format file.
    Format:
        Line 1: number of nodes
        Line i: node_id: neighbor1 neighbor2 ...
    Args:
        filename: Path to the adjacency list format file
    Returns:
        Tuple of (NetworkX graph, graph information dictionary)
    """
    nx_graph = nx.Graph()

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse header
    n_nodes = int(lines[0].strip())
    nx_graph.add_nodes_from(range(n_nodes))

    # Parse adjacency list
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split(':')
        if len(parts) < 2:
            continue

        node_id = int(parts[0].strip())
        neighbors = parts[1].strip().split()

        for neighbor in neighbors:
            nx_graph.add_edge(node_id, int(neighbor))

    # Collect graph information
    graph_info = {
        'n_nodes': nx_graph.number_of_nodes(),
        'n_edges': nx_graph.number_of_edges(),
        'density': nx.density(nx_graph),
        'is_weighted': False,
        'is_connected': nx.is_connected(nx_graph),
        'n_components': nx.number_connected_components(nx_graph)
    }

    print(f"Loaded graph with {graph_info['n_nodes']} nodes and {graph_info['n_edges']} edges")
    print(f"Density: {graph_info['density']:.4f}")
    print(f"Connected: {graph_info['is_connected']} ({graph_info['n_components']} components)")

    return nx_graph, graph_info


def save_graph(nx_graph: nx.Graph, filename: str, format: str = 'gset') -> None:
    """
    Save a graph to file.
    
    Args:
        nx_graph: NetworkX graph to save
        filename: Output filename
        format: Output format ('gset' or 'edgelist')
    """
    if format == 'gset':
        with open(filename, 'w') as f:
            # Write header
            f.write(f"{nx_graph.number_of_nodes()} {nx_graph.number_of_edges()}\n")
            
            # Write edges (convert to 1-indexed)
            for u, v, data in nx_graph.edges(data=True):
                weight = data.get('weight', 1.0)
                f.write(f"{u + 1} {v + 1} {weight}\n")
    
    elif format == 'edgelist':
        nx.write_edgelist(nx_graph, filename, data=['weight'])
    
    else:
        raise ValueError(f"Unknown format: {format}")