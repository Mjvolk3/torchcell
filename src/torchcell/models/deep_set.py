# src/torchcell/models/deep_set.py
# [[src.torchcell.models.deep_set]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/models/deep_set.py
# Test file: src/torchcell/models/test_deep_set.py

import torch
import torch.nn as nn
from torch_scatter import scatter_add

from torchcell.models.act import act_register


class DeepSet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        node_layers: list[int],
        set_layers: list[int],
        dropout_prob: float = 0.2,
        norm: str = "batch",
        activation: str = "relu",
        skip_node: bool = False,  # Parameter to add skip connections in node_layers
        skip_set: bool = False,  # Parameter to add skip connections in set_layers
    ):
        super().__init__()

        assert norm in ["batch", "instance", "layer"], "Invalid norm type"
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(),
            "sigmoid": nn.Sigmoid(),
        }
        assert activation in activations.keys(), "Invalid activation type"

        self.skip_node = skip_node
        self.skip_set = skip_set

        def create_block(in_dim, out_dim, norm, activation):
            block = [nn.Linear(in_dim, out_dim)]
            if norm == "batch":
                block.append(nn.BatchNorm1d(out_dim))
            elif norm == "instance":
                block.append(nn.InstanceNorm1d(out_dim, affine=True))
            elif norm == "layer":
                block.append(nn.LayerNorm(out_dim))
            block.append(activations[activation])
            return nn.Sequential(*block)

        in_dim = input_dim
        node_modules = []
        for out_dim in node_layers:
            node_modules.append(create_block(in_dim, out_dim, norm, activation))
            in_dim = out_dim
        self.node_layers = nn.ModuleList(node_modules)

        set_modules = []
        for out_dim in set_layers:
            set_modules.append(create_block(in_dim, out_dim, norm, activation))
            in_dim = out_dim
        set_modules.append(nn.Dropout(dropout_prob))
        self.set_layers = nn.ModuleList(set_modules)

    def forward(self, x, batch):
        x_node = x
        for i, layer in enumerate(self.node_layers):
            out_node = layer(x_node)
            if self.skip_node and x_node.shape[-1] == out_node.shape[-1]:
                out_node += x_node  # Skip connection
            x_node = out_node

        x_summed = scatter_add(x_node, batch, dim=0)

        x_set = x_summed
        for i, layer in enumerate(self.set_layers):
            out_set = layer(x_set)
            if self.skip_set and x_set.shape[-1] == out_set.shape[-1]:
                out_set += x_set  # Skip connection
            x_set = out_set

        return x_node, x_set


def main():
    input_dim = 10
    node_layers = [64, 32, 32, 32]
    set_layers = [16, 8, 8]
    model = DeepSet(
        input_dim,
        node_layers,
        set_layers,
        norm="layer",
        activation="gelu",
        skip_node=True,
        skip_set=True,
    )

    x = torch.rand(100, input_dim)
    batch = torch.cat([torch.full((20,), i, dtype=torch.long) for i in range(5)])
    x_nodes, x_set = model(x, batch)
    print(x_set.shape)
    print(x_nodes.shape)


if __name__ == "__main__":
    main()
