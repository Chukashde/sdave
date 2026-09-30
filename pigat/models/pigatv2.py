import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from pigat.models.base import PIBaseGNN


def _edge_softmax(scores, dst, num_nodes):
    # scores: [E, H]; dst: [E]. dst тус бүрээр softmax.
    # Глобал max (head бүрээр) хасна: тооны тогтвортой байдал, MPS-д scatter_reduce хэрэггүй.
    m = scores.max(dim=0, keepdim=True).values
    e = torch.exp(scores - m)
    denom = torch.zeros(num_nodes, scores.shape[1], device=scores.device,
                        dtype=scores.dtype).index_add_(0, dst, e)
    return e / (denom[dst] + 1e-16)


class MPSGATConv(nn.Module):
    def __init__(self, in_size, out_size, num_heads, feat_drop, attn_drop):
        super().__init__()
        self.H, self.D = num_heads, out_size
        self.fc = nn.Linear(in_size, out_size * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.empty(1, num_heads, out_size))
        self.attn_r = nn.Parameter(torch.empty(1, num_heads, out_size))
        self.bias = nn.Parameter(torch.zeros(num_heads * out_size))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

    def forward(self, src, dst, n, x):
        h = self.fc(self.feat_drop(x)).view(n, self.H, self.D)
        el = (h * self.attn_l).sum(-1)          # [N, H]
        er = (h * self.attn_r).sum(-1)
        e = F.leaky_relu(el[src] + er[dst], 0.2)  # [E, H]
        a = self.attn_drop(_edge_softmax(e, dst, n))
        out = torch.zeros(n, self.H, self.D, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, h[src] * a.unsqueeze(-1))
        return (out + self.bias.view(1, self.H, self.D))  # [N, H, D]


class PIGATv2(PIBaseGNN):
    def __init__(self, embedding_dim, hidden_dim, num_classes, dropout, device,
                 num_heads=4, feat_drop=0.0, attn_drop=0.0):
        super().__init__(embedding_dim, hidden_dim, num_classes, dropout, device)
        sizes = self.layer_sizes
        layers = []
        for k, (i, o) in enumerate(sizes):
            in_dim = i if k == 0 else i * num_heads
            heads = num_heads if k < len(sizes) - 1 else 1
            layers.append(MPSGATConv(in_dim, o, heads, feat_drop, attn_drop))
        self.layers = nn.ModuleList(layers)
        self.to(device)

    def forward(self, g, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = g.edges()
        src, dst = src.long().to(inputs.device), dst.long().to(inputs.device)
        n = g.num_nodes()
        h, last_hidden = inputs, None
        last = len(self.layers) - 1
        for k, layer in enumerate(self.layers):
            if k < last:
                h = F.elu(layer(src, dst, n, h)).flatten(1)
                h = F.dropout(h, p=self.dropout_frac, training=self.training)
            else:
                last_hidden = h.clone()
                h = torch.sigmoid(layer(src, dst, n, h).mean(1))
        return h, last_hidden
