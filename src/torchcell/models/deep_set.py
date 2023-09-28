# src/torchcell/models/deep_set.py
# [[src.torchcell.models.deep_set]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/models/deep_set.py
# Test file: src/torchcell/models/test_deep_set.py

import torch
import torch.nn as nn
from torch_scatter import scatter_add


class DeepSet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        node_layers: list[int],
        set_layers: list[int],
        dropout_prob: float = 0.2,
        with_global: bool = False,
        global_activation: str = None,
    ):
        super().__init__()

        # node Layers
        node_modules = []
        in_features = input_dim
        for out_features in node_layers:
            node_modules.append(nn.Linear(in_features, out_features))
            node_modules.append(nn.BatchNorm1d(out_features))
            node_modules.append(nn.GELU())
            in_features = out_features
        self.node_layers = nn.Sequential(*node_modules)

        # Set Layers
        set_modules = []
        for out_features in set_layers:
            set_modules.append(nn.Linear(in_features, out_features))
            set_modules.append(nn.BatchNorm1d(out_features))
            set_modules.append(nn.GELU())
            in_features = out_features

        set_modules.append(nn.Dropout(dropout_prob))
        self.set_layers = nn.Sequential(*set_modules)

        # Global Predictor Layer
        self.with_global = with_global
        if with_global:
            global_modules = []
            global_modules.append(nn.Linear(in_features, 1))
            # Might want to force positive, e.g. Fitness is always positive
            if global_activation == "relu":
                self.global_activation = nn.ReLU()
                global_modules.append(self.global_activation)
            self.global_layer = nn.Sequential(*global_modules)

    def forward(self, x, batch):
        # For Node Layers
        x_nodes = x
        for layer in self.node_layers:
            if isinstance(layer, nn.BatchNorm1d) and x_nodes.size(0) == 1:
                continue  # skip batch normalization if batch size is 1
            x_nodes = layer(x_nodes)

        # Sum over nodes belonging to the same graph using scatter_add
        x_summed = scatter_add(x_nodes, batch, dim=0)

        # For Set Layers
        x_set = x_summed
        for layer in self.set_layers:
            if isinstance(layer, nn.BatchNorm1d) and x_set.size(0) == 1:
                continue  # skip batch normalization if batch size is 1
            x_set = layer(x_set)

        # For Global Layer
        if self.with_global:
            x_global = self.global_layer(x_set).squeeze(-1)
        else:
            x_global = None

        return x_nodes, x_set, x_global


def main():
    # Example usage:
    input_dim = 10
    node_layers = [64, 32]
    set_layers = [16, 8]

    model = DeepSet(
        input_dim, node_layers, set_layers
    )  # No output activation specified

    # Simulate 5 sets, each with 20 nodes, and 10 features per node
    x = torch.rand(100, input_dim)  # 100 nodes in total (5 sets * 20 nodes)

    # Create a batch vector
    # This will be [0, 0, ..., 1, 1, ..., 2, 2, ..., ..., 4, 4, ...]
    # Each number i appears 20 times, indicating that 20 nodes belong to set i
    batch = torch.cat([torch.full((20,), i, dtype=torch.long) for i in range(5)])

    x_global, x_set, x_nodes = model(x, batch)
    print(x_global)
    print(x_set)
    print(x_nodes)


if __name__ == "__main__":
    main()
