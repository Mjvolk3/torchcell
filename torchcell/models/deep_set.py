import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean

from torchcell.models.act import act_register


class DeepSet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_node_layers: int,
        num_set_layers: int,
        dropout_prob: float = 0.2,
        norm: str = "batch",
        activation: str = "relu",
        skip_node: bool = False,  # Parameter to add skip connections in node_layers
        skip_set: bool = False,  # Parameter to add skip connections in set_layers
        aggregation: str = "sum",  # Aggregation method: "sum" or "mean"
    ):
        super().__init__()

        assert norm in ["batch", "instance", "layer"], "Invalid norm type"
        assert activation in act_register.keys(), "Invalid activation type"
        assert aggregation in ["sum", "mean"], "Invalid aggregation method"

        self.skip_node = skip_node
        self.skip_set = skip_set
        self.aggregation = aggregation

        def create_block(in_dim, out_dim, norm, activation):
            block = [nn.Linear(in_dim, out_dim)]
            if norm == "batch":
                block.append(nn.BatchNorm1d(out_dim))
            elif norm == "instance":
                block.append(nn.InstanceNorm1d(out_dim, affine=True))
            elif norm == "layer":
                block.append(nn.LayerNorm(out_dim))
            block.append(act_register[activation])
            return nn.Sequential(*block)

        node_modules = []
        for i in range(num_node_layers):
            if i == 0:
                node_modules.append(
                    create_block(in_channels, hidden_channels, norm, activation)
                )
            elif i == num_node_layers - 1:
                node_modules.append(
                    create_block(hidden_channels, out_channels, norm, activation)
                )
            else:
                node_modules.append(
                    create_block(hidden_channels, hidden_channels, norm, activation)
                )
        self.node_layers = nn.ModuleList(node_modules)

        set_modules = []
        for i in range(num_set_layers):
            if i == 0:
                if num_node_layers > 0:
                    set_modules.append(
                        create_block(out_channels, hidden_channels, norm, activation)
                    )
                else:
                    set_modules.append(
                        create_block(in_channels, hidden_channels, norm, activation)
                    )
            elif i == num_set_layers - 1:
                set_modules.append(
                    create_block(hidden_channels, out_channels, norm, activation)
                )
                set_modules.append(nn.Dropout(dropout_prob))
            else:
                set_modules.append(
                    create_block(hidden_channels, hidden_channels, norm, activation)
                )
        self.set_layers = nn.ModuleList(set_modules)

    def node_layers_forward(self, x):
        """Process node features through node layers."""
        x_node = x
        for i, layer in enumerate(self.node_layers):
            out_node = layer(x_node)
            if self.skip_node and x_node.shape[-1] == out_node.shape[-1]:
                out_node = out_node + x_node  # Skip connection
            x_node = out_node
        return x_node

    def set_layers_forward(self, x_aggregated):
        """Process aggregated features through set layers."""
        x_set = x_aggregated
        for i, layer in enumerate(self.set_layers):
            out_set = layer(x_set)
            if (
                self.skip_set
                and i < len(self.set_layers) - 1
                and x_set.shape[-1] == out_set.shape[-1]
            ):
                out_set = out_set + x_set  # Skip connection
            x_set = out_set
        return x_set

    def forward(self, x, batch):
        if len(self.node_layers) > 0:
            x_node = self.node_layers_forward(x)
        else:
            x_node = x

        if self.aggregation == "sum":
            x_aggregated = scatter_add(x_node, batch, dim=0)
        elif self.aggregation == "mean":
            x_aggregated = scatter_mean(x_node, batch, dim=0)

        x_set = self.set_layers_forward(x_aggregated)
        return x_node, x_set


def main():
    torch.autograd.set_detect_anomaly(True)

    # Model configuration
    in_channels = 10
    hidden_channels = 32
    out_channels = 8
    num_node_layers = 3
    num_set_layers = 2

    model = DeepSet(
        in_channels,
        hidden_channels,
        out_channels,
        num_node_layers,
        num_set_layers,
        norm="layer",
        activation="tanh",
        skip_node=True,
        skip_set=True,
        aggregation="mean",  # Use "sum" or "mean" for aggregation
    )

    # Dummy data
    x = torch.rand(100, in_channels)
    print("x shape:", x.shape)
    batch = torch.cat([torch.full((20,), i, dtype=torch.long) for i in range(5)])

    # Forward pass
    x_nodes, x_set = model(x, batch)
    print("x_set shape:", x_set.shape)
    print("batch shape:", batch.shape)
    print("batch unique:", torch.unique(batch))
    print("x_nodes shape:", x_nodes.shape)

    # Let's assume you want to predict some values for each set.
    # So, we'll create a dummy target tensor for demonstration purposes.
    target = torch.rand(5, out_channels)

    # Simple mean squared error loss
    criterion = nn.MSELoss()
    print(x_set.shape, target.shape)
    loss = criterion(x_set, target)
    print("Loss:", loss.item())

    # Backpropagation
    model.zero_grad()
    loss.backward()
    print("Gradients computed successfully!")


if __name__ == "__main__":
    main()