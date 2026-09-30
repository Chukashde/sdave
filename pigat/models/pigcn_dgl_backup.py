import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from typing import Tuple
from collections import OrderedDict

from pigat.models.base import PIBaseGNN


class PIGCN(PIBaseGNN):
    """
    PI-GCN: Graph Convolutional Network for combinatorial optimization.
    """

    def __init__(self,
                 embedding_dim: int,
                 hidden_dim,
                 num_classes: int,
                 dropout: float,
                 device: str):
        """
        Initialize PI-GCN model.

        Args:
            embedding_dim: Dimension of node embeddings
            hidden_dim: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout probability
            device: Device to run model on
        """
        super().__init__(embedding_dim, hidden_dim, num_classes, dropout, device)

        # Create GCN layers
        self.layers = OrderedDict()
        for i, (in_size, out_size) in enumerate(self.layer_sizes):
            self.layers[i] = GraphConv(in_size, out_size, allow_zero_in_degree=True).to(device)

    def forward(self, g, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            g: DGL graph
            inputs: Node feature inputs

        Returns:
            Tuple of (output probabilities, last hidden layer features)
        """
        h = None
        last_hidden = None

        for k, layer in self.layers.items():
            if k == 0:
                # First layer
                h = layer(g, inputs)
                h = torch.relu(h)
                h = F.dropout(h, p=self.dropout_frac, training=self.training)
            elif 0 < k < (len(self.layers) - 1):
                # Hidden layers
                h = layer(g, h)
                h = torch.relu(h)
                h = F.dropout(h, p=self.dropout_frac, training=self.training)
            else:
                # Output layer
                last_hidden = h.clone()
                h = layer(g, h)
                h = torch.sigmoid(h)

        return h, last_hidden
