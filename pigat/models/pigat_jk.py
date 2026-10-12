import torch
import torch.nn as nn
import torch.nn.functional as F

from pigat.models.base import PIBaseGNN
from pigat.models.weighted_gat_conv import WeightedGATConv


class PIGATJK(PIBaseGNN):
    def __init__(self, embedding_dim, hidden_dim, num_classes,
                 dropout, device, num_heads=1,
                 feat_drop=0.0, attn_drop=0.0):

        super().__init__(
            embedding_dim, hidden_dim, num_classes, dropout, device
        )

        self.layers = nn.ModuleList()
        dims = [embedding_dim] + list(hidden_dim)

        for k, (in_dim, out_dim) in enumerate(
            zip(dims[:-1], dims[1:])
        ):
            actual_in_dim = (
                in_dim if k == 0 else in_dim * num_heads
            )
            self.layers.append(
                WeightedGATConv(
                    actual_in_dim, out_dim, num_heads,
                    feat_drop, attn_drop
                )
            )

        # Each hidden layer produces num_heads * hidden_dim.
        jk_dim = embedding_dim + sum(
            d * num_heads for d in hidden_dim
        )
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
            h = F.elu(layer(src, dst, n, h, edge_weight)).flatten(1)
            h = F.dropout(
                h, p=self.dropout_frac, training=self.training
            )
            representations.append(h)

        combined = torch.cat(representations, dim=-1)
        probability = torch.sigmoid(self.jk_head(combined))

        return probability, h
