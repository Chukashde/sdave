import torch
import torch.nn.functional as F

from pigat.models.pigatv2 import MPSGATConv
from pigat.models.gatv2_conv import MPSGATv2Conv
from pigat.models.weighted_attention import signed_weighted_attention


class WeightedGATConv(MPSGATConv):
    """GATv1 with signed weighted attention."""

    def forward(self, src, dst, n, x, edge_weight):
        src = src.long().to(x.device)
        dst = dst.long().to(x.device)

        h = self.fc(self.feat_drop(x)).view(n, self.H, self.D)

        el = (h * self.attn_l).sum(-1)
        er = (h * self.attn_r).sum(-1)

        e = F.leaky_relu(el[src] + er[dst], 0.2)

        a = signed_weighted_attention(
            e, src, dst, n, edge_weight
        )
        a = self.attn_drop(a)

        out = torch.zeros(
            n, self.H, self.D,
            device=x.device, dtype=x.dtype
        )
        out.index_add_(0, dst, h[src] * a.unsqueeze(-1))

        return out + self.bias.view(1, self.H, self.D)


class WeightedGATv2Conv(MPSGATv2Conv):
    """GATv2 with signed weighted dynamic attention."""

    def forward(self, src, dst, n, x, edge_weight):
        src = src.long().to(x.device)
        dst = dst.long().to(x.device)

        x = self.feat_drop(x)

        hs = self.fc_src(x).view(n, self.H, self.D)
        hd = self.fc_dst(x).view(n, self.H, self.D)

        e = (
            F.leaky_relu(
                hs[src] + hd[dst],
                negative_slope=0.2
            ) * self.attn
        ).sum(-1)

        a = signed_weighted_attention(
            e, src, dst, n, edge_weight
        )
        a = self.attn_drop(a)

        out = torch.zeros(
            n, self.H, self.D,
            device=x.device, dtype=x.dtype
        )
        out.index_add_(0, dst, hs[src] * a.unsqueeze(-1))

        return out + self.bias.view(1, self.H, self.D)
