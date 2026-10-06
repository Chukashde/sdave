import torch


class SimpleGraph:
    """dgl.DGLGraph-ийн оронд: зөвхөн edges() ба num_nodes() хэрэгтэй загварт зориулсан."""

    def __init__(self, nx_graph):
        self._n = nx_graph.number_of_nodes()
        edges = list(nx_graph.edges())
        if edges:
            src, dst = zip(*edges)
        else:
            src, dst = (), ()
        self._src = torch.tensor(src, dtype=torch.long)
        self._dst = torch.tensor(dst, dtype=torch.long)

    def edges(self):
        return self._src, self._dst

    def num_nodes(self):
        return self._n

    def number_of_nodes(self):
        return self._n

    def number_of_edges(self):
        return self._src.shape[0]

    def to(self, device):
        # PIGCN/PIGATv2 forward нь эдгээрийг өөрөө .to(device) хийдэг тул зүгээр өөрийгөө буцаана.
        return self


def from_networkx(nx_graph):
    return SimpleGraph(nx_graph)


# dgl.DGLGraph type hint-ийг орлуулах
DGLGraph = SimpleGraph
