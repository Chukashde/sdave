import torch
import torch.nn as nn
import torch.nn.functional as F

from pigat.models.base import PIBaseGNN
from pigat.models.pigcn import get_device
from pigat.models.weighted_graph_conv import WeightedGraphConv


class PIGCNJK(PIBaseGNN):
    def __init__(self, embedding_dim, hidden_dim, num_classes,
                 dropout, device=None):
        device = device or get_device()
        super().__init__(
            embedding_dim, hidden_dim, num_classes, dropout, device
        )

        # Message-passing layers: input -> hidden representations.
        dims = [embedding_dim] + list(hidden_dim)
        self.layers = nn.ModuleList([
            WeightedGraphConv(i, o) for i, o in zip(dims[:-1], dims[1:])
        ])

        # Jumping knowledge: input + every hidden layer.
        jk_dim = sum(dims)
        self.jk_head = nn.Linear(jk_dim, num_classes)
        self.to(device)

    def forward(self, g, inputs):
        src, dst = g.edges()
        src = src.long().to(inputs.device)
        dst = dst.long().to(inputs.device)
        n = g.num_nodes()
        edge_weight = g.edge_weight().to(
            device=inputs.device, dtype=inputs.dtype
        )

        h = inputs
        representations = [h]

        for layer in self.layers:
            h = F.relu(layer(src, dst, n, h, edge_weight))
            h = F.dropout(
                h, p=self.dropout_frac, training=self.training
            )
            representations.append(h)

        combined = torch.cat(representations, dim=-1)
        probability = torch.sigmoid(self.jk_head(combined))

        return probability, h
