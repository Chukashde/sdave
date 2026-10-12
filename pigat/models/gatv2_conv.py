import torch
import torch.nn as nn
import torch.nn.functional as F

from pigat.models.pigatv2 import _edge_softmax


class MPSGATv2Conv(nn.Module):
    def __init__(self, in_size, out_size, num_heads,
                 feat_drop=0.0, attn_drop=0.0):
        super().__init__()

        self.H = num_heads
        self.D = out_size

        self.fc_src = nn.Linear(
            in_size, num_heads * out_size, bias=False
        )
        self.fc_dst = nn.Linear(
            in_size, num_heads * out_size, bias=False
        )

        self.attn = nn.Parameter(
            torch.empty(1, num_heads, out_size)
        )

        self.bias = nn.Parameter(
            torch.zeros(num_heads * out_size)
        )

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        nn.init.xavier_uniform_(self.fc_src.weight)
        nn.init.xavier_uniform_(self.fc_dst.weight)
        nn.init.xavier_uniform_(self.attn)

    def forward(self, src, dst, n, x):
        x = self.feat_drop(x)

        hs = self.fc_src(x).view(n, self.H, self.D)
        hd = self.fc_dst(x).view(n, self.H, self.D)

        # True GATv2 dynamic attention
        e = (
            F.leaky_relu(
                hs[src] + hd[dst],
                negative_slope=0.2
            ) * self.attn
        ).sum(-1)

        a = self.attn_drop(
            _edge_softmax(e, dst, n)
        )

        out = torch.zeros(
            n, self.H, self.D,
            device=x.device,
            dtype=x.dtype
        )

        out.index_add_(
            0,
            dst,
            hs[src] * a.unsqueeze(-1)
        )

        return out + self.bias.view(1, self.H, self.D)
