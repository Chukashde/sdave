import torch


class SimpleGraph:
    """Undirected NetworkX graph -> bidirectional message-passing graph."""

    def __init__(self, nx_graph):
        self._n = nx_graph.number_of_nodes()

        edges = list(nx_graph.edges(data=True))

        if edges:
            u = [int(a) for a, b, data in edges]
            v = [int(b) for a, b, data in edges]
            w = [float(data.get("weight", 1.0)) for a, b, data in edges]

            # Forward edges followed by reverse edges.
            self._src = torch.tensor(u + v, dtype=torch.long)
            self._dst = torch.tensor(v + u, dtype=torch.long)
            self._weight = torch.tensor(w + w, dtype=torch.float32)
        else:
            self._src = torch.empty(0, dtype=torch.long)
            self._dst = torch.empty(0, dtype=torch.long)
            self._weight = torch.empty(0, dtype=torch.float32)

    def edges(self):
        return self._src, self._dst

    def edge_weight(self):
        return self._weight

    def num_nodes(self):
        return self._n

    def number_of_nodes(self):
        return self._n

    def number_of_edges(self):
        return self._src.shape[0]

    def to(self, device):
        return self


def from_networkx(nx_graph):
    return SimpleGraph(nx_graph)


DGLGraph = SimpleGraph
