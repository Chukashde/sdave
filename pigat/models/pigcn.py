import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from pigat.models.base import PIBaseGNN


def get_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class MPSGraphConv(nn.Module):
    """DGL GraphConv(norm='both', allow_zero_in_degree=True)-тэй тэнцүү, MPS-д тохирсон."""

    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_size, out_size))
        self.bias = nn.Parameter(torch.zeros(out_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, src, dst, num_nodes, x):
        ones = torch.ones(src.shape[0], device=x.device, dtype=x.dtype)
        out_deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype).index_add_(0, src, ones)
        in_deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype).index_add_(0, dst, ones)
        out_deg = out_deg.clamp(min=1).pow(-0.5)
        in_deg = in_deg.clamp(min=1).pow(-0.5)

        h = x * out_deg.unsqueeze(1)
        h = h @ self.weight
        agg = torch.zeros(num_nodes, h.shape[1], device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, h[src])
        agg = agg * in_deg.unsqueeze(1)
        return agg + self.bias


class PIGCN(PIBaseGNN):
    def __init__(self, embedding_dim: int, hidden_dim, num_classes: int,
                 dropout: float, device: str = None):
        device = device or get_device()
        super().__init__(embedding_dim, hidden_dim, num_classes, dropout, device)

        self.layers = nn.ModuleList(
            [MPSGraphConv(i, o) for i, o in self.layer_sizes]
        )
        self.to(device)

    def forward(self, g, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = g.edges()
        src, dst = src.long().to(inputs.device), dst.long().to(inputs.device)
        n = g.num_nodes()

        h, last_hidden = inputs, None
        last = len(self.layers) - 1
        for k, layer in enumerate(self.layers):
            if k < last:
                h = torch.relu(layer(src, dst, n, h))
                h = F.dropout(h, p=self.dropout_frac, training=self.training)
            else:
                last_hidden = h.clone()
                h = torch.sigmoid(layer(src, dst, n, h))
        return h, last_hidden
