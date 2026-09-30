import torch
import torch.nn as nn
from typing import List, Tuple

from pigat.models._utils import pairwise


class PIBaseGNN(nn.Module):
    """Shared base for physics-inspired GNNs (PI-GCN, PI-GATv2).

    Holds the bookkeeping common to every variant (layer sizes, dropout
    fraction, device) and defines the forward interface, so depth sweeps and
    seed loops can treat the variants uniformly.

    No layer is constructed here. Each subclass builds its own layers in
    __init__ after calling super().__init__(), so weight-initialization RNG
    order is determined entirely by the subclass build loop and stays identical
    to the pre-refactor models.
    """

    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: List[int],
                 num_classes: int,
                 dropout: float,
                 device: str):
        super().__init__()
        all_layers = [embedding_dim] + hidden_dim + [num_classes]
        self.layer_sizes = list(pairwise(all_layers))
        self.dropout_frac = dropout
        self.device = device

    def forward(self, g, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
