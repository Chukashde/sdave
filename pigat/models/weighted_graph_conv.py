import torch
import torch.nn as nn


class WeightedGraphConv(nn.Module):
    """Signed edge-weighted GCN with absolute-degree normalization."""

    def __init__(self, in_size, out_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_size, out_size))
        self.bias = nn.Parameter(torch.zeros(out_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, src, dst, num_nodes, x, edge_weight):
        src = src.long().to(x.device)
        dst = dst.long().to(x.device)
        w = edge_weight.to(device=x.device, dtype=x.dtype)

        assert w.numel() == src.numel() == dst.numel()

        # Absolute weighted degrees support signed edge weights.
        degree = torch.zeros(
            num_nodes, device=x.device, dtype=x.dtype
        )
        degree.index_add_(0, dst, w.abs())

        inv_sqrt = degree.clamp(min=1e-12).rsqrt()

        h = x @ self.weight
        messages = (
            h[src]
            * w.unsqueeze(-1)
            * inv_sqrt[src].unsqueeze(-1)
            * inv_sqrt[dst].unsqueeze(-1)
        )

        out = torch.zeros(
            num_nodes, h.shape[1], device=x.device, dtype=x.dtype
        )
        out.index_add_(0, dst, messages)

        return out + self.bias
