from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv as GCN
from torch_scatter import scatter_add

from torchcell.models import DeepSet


class GraphAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        node_layers: list[int],
        set_layers: list[int],
        hidden_channels: int,
        num_layers: int,
        out_channels: int | None = None,
        dropout_prob: float = 0.2,
        norm: str = "batch",
        activation: str = "relu",
        skip_node: bool = False,
        skip_set: bool = False,
        skip_mp: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.deepset = DeepSet(
            input_dim=input_dim,
            node_layers=node_layers,
            set_layers=[],
            dropout_prob=dropout_prob,
            norm=norm,
            activation=activation,
            skip_node=skip_node,
            skip_set=skip_set,
        )
        self.skip_mp = skip_mp
        self.gat_layers = nn.ModuleList()

        if node_layers:
            in_dim = node_layers[-1]
        else:
            in_dim = input_dim

        for i in range(num_layers):
            out_dim = (
                hidden_channels
                if i < num_layers - 1 or out_channels is None
                else out_channels
            )
            # Set to more stable v2
            self.gat_layers.append(GATConv(in_dim, out_dim, v2=True))
            in_dim = out_dim

        # Set layers after GAT
        set_layers.insert(0, in_dim)
        self.set_layers = nn.ModuleList(
            [
                nn.Linear(set_layers[i], set_layers[i + 1])
                for i in range(len(set_layers) - 1)
            ]
        )

    def forward(self, x, batch, edge_index):
        # Process node features
        x_node = self.deepset.node_layers_forward(x)

        # Message passing with GAT
        for layer in self.gat_layers:
            out_node = layer(x_node, edge_index)
            if self.skip_mp and x_node.shape[-1] == out_node.shape[-1]:
                out_node = out_node + x_node  # Skip connection
            x_node = out_node

        x_set = scatter_add(x_node, batch, dim=0)

        # Process aggregated set features
        for i, layer in enumerate(self.set_layers):
            out_set = layer(x_set)
            if self.deepset.skip_set and x_set.shape[-1] == out_set.shape[-1]:
                out_set = out_set + x_set
            x_set = out_set

        return x_node, x_set


def main():
    torch.autograd.set_detect_anomaly(True)

    # Model configuration
    input_dim = 10
    node_layers = [16, 16]
    set_layers = [8, 8]

    model = GraphAttention(
        input_dim,
        node_layers,
        set_layers,
        hidden_channels=16,
        num_layers=2,
        norm="batch",
        activation="gelu",
        skip_node=True,
        skip_set=True,
        skip_mp=True,
    )

    # Dummy data
    x = torch.rand(100, input_dim)
    batch = torch.cat([torch.full((20,), i, dtype=torch.long) for i in range(5)])
    edge_index = torch.stack([torch.arange(100), torch.arange(100)], dim=0)

    # Forward pass
    x_nodes, x_set = model(x, batch, edge_index)
    print(x_set.shape)
    print(x_nodes.shape)

    # Dummy target tensor
    target = torch.rand(5, set_layers[-1])

    # Mean squared error loss
    criterion = nn.MSELoss()
    loss = criterion(x_set, target)
    print("Loss:", loss.item())

    # Backpropagation
    model.zero_grad()
    loss.backward()
    print("Gradients computed successfully!")
    print(model)


if __name__ == "__main__":
    main()
