import torch
from pigat.models.pigcn import PIGCN
from pigat.models.pigatv2 import PIGATv2


class G:
    """DGL-ийн оронд ашиглах жижиг граф: зөвхөн edges(), num_nodes() хэрэгтэй."""
    def __init__(self, n, e):
        self.n = n
        self.src = torch.randint(0, n, (e,))
        self.dst = torch.randint(0, n, (e,))

    def edges(self):
        return self.src, self.dst

    def num_nodes(self):
        return self.n


device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

embedding_dim = 16
hidden_dim = [64, 32]
num_classes = 1
dropout = 0.1

g = G(100, 500)
inputs = torch.randn(100, embedding_dim).to(device)

gcn = PIGCN(embedding_dim, hidden_dim, num_classes, dropout, device)
out, feats = gcn(g, inputs)
print("PIGCN :", out.shape, feats.shape)

gat = PIGATv2(embedding_dim, hidden_dim, num_classes, dropout, device,
              num_heads=4, feat_drop=0.0, attn_drop=0.0)
out, feats = gat(g, inputs)
print("PIGATv2:", out.shape, feats.shape)
