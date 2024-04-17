# torchcell/models/self_attention_deep_set
# [[torchcell.models.self_attention_deep_set]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/self_attention_deep_set
# Test file: tests/torchcell/models/test_self_attention_deep_set.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from torchcell.models.act import act_register


class SelfAttention(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int = 1):
        super().__init__()
        self.query = nn.Linear(dim_in, dim_out * num_heads)
        self.key = nn.Linear(dim_in, dim_out * num_heads)
        self.value = nn.Linear(dim_in, dim_out * num_heads)
        self.dim_out = dim_out
        self.num_heads = num_heads

    def forward(self, x, batch):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(-1, self.num_heads, self.dim_out).transpose(0, 1)
        K = K.view(-1, self.num_heads, self.dim_out).transpose(0, 1)
        V = V.view(-1, self.num_heads, self.dim_out).transpose(0, 1)

        attn_weights = F.softmax(
            torch.bmm(Q, K.transpose(1, 2)) / self.dim_out**0.5, dim=-1
        )
        out = torch.bmm(attn_weights, V)
        out = out.transpose(0, 1).contiguous().view(-1, self.num_heads * self.dim_out)

        # Retrieve attention weights for a single graph
        graph_attn_weights = []
        for i in torch.unique(batch):
            mask = batch == i
            graph_attn_weights.append(attn_weights[:, mask][:, :, mask])

        return out, graph_attn_weights


class SelfAttentionDeepSet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_node_layers: int,
        num_set_layers: int,
        num_heads: int = 1,
        dropout_prob: float = 0.2,
        norm: str = "batch",
        activation: str = "relu",
        skip_node: bool = False,
        skip_set: bool = False,
    ):
        super().__init__()

        assert norm in ["batch", "instance", "layer"], "Invalid norm type"
        assert activation in act_register.keys(), "Invalid activation type"

        self.skip_node = skip_node
        self.skip_set = skip_set
        self.num_heads = num_heads

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
                    create_block(
                        hidden_channels * num_heads, out_channels, norm, activation
                    )
                )
            else:
                node_modules.append(
                    create_block(
                        hidden_channels * num_heads, hidden_channels, norm, activation
                    )
                )
        self.node_layers = nn.ModuleList(node_modules)

        set_modules = []
        for i in range(num_set_layers):
            if i == 0:
                set_modules.append(
                    create_block(out_channels, hidden_channels, norm, activation)
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

        self.self_attn = SelfAttention(
            dim_in=hidden_channels, dim_out=hidden_channels, num_heads=num_heads
        )

    def node_layers_forward(self, x, batch):
        """Process node features through node layers."""
        x_node = x
        attn_weights_list = []  # List to store attention weights from each layer
        for i, layer in enumerate(self.node_layers):
            if i > 0:
                x_node, graph_attn_weights = self.self_attn(x_node, batch)
                attn_weights_list.append(graph_attn_weights)
            out_node = layer(x_node)
            if self.skip_node and x_node.shape[-1] == out_node.shape[-1]:
                out_node = out_node + x_node  # Skip connection
            x_node = out_node
        return x_node, attn_weights_list

    def set_layers_forward(self, x_summed):
        """Process aggregated features through set layers."""
        x_set = x_summed
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
        x_node, attn_weights_list = self.node_layers_forward(x, batch)
        x_summed = scatter_add(x_node, batch, dim=0)
        x_set = self.set_layers_forward(x_summed)
        return x_node, x_set, attn_weights_list


def main():
    torch.autograd.set_detect_anomaly(True)

    # Model configuration
    in_channels = 10
    hidden_channels = 32
    out_channels = 8
    num_node_layers = 3
    num_set_layers = 2
    num_heads = 2

    model = SelfAttentionDeepSet(
        in_channels,
        hidden_channels,
        out_channels,
        num_node_layers,
        num_set_layers,
        num_heads,
        norm="layer",
        activation="tanh",
        skip_node=True,
        skip_set=True,
    )

    # Print the number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    # Dummy data
    x = torch.rand(100, in_channels)
    print("x shape:", x.shape)
    batch = torch.cat([torch.full((20,), i, dtype=torch.long) for i in range(5)])

    # Forward pass
    x_nodes, x_set, attn_weights_list = model(x, batch)
    print("x_set shape:", x_set.shape)
    print("x_nodes shape:", x_nodes.shape)
    print("Number of attention weights:", len(attn_weights_list))
    for i, graph_attn_weights in enumerate(attn_weights_list):
        print(f"Number of graphs at layer {i+1}:", len(graph_attn_weights))
        for j, attn_weights in enumerate(graph_attn_weights):
            print(
                f"Attention weights shape for graph {j+1} at layer {i+1}:",
                attn_weights.shape,
            )

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
