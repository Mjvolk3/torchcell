# torchcell/models/deep_set_transformer.py
# [[torchcell.models.deep_set_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/deep_set_transformer.py
# Test file: torchcell/models/test_deep_set_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


class SelfAttention(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int = 1):
        super().__init__()
        self.query = nn.Linear(dim_in, dim_out * num_heads)
        self.key = nn.Linear(dim_in, dim_out * num_heads)
        self.value = nn.Linear(dim_in, dim_out * num_heads)
        self.dim_out = dim_out
        self.num_heads = num_heads

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Reshaping to have the num_heads represented in the tensor
        Q = Q.view(Q.size(0), -1, self.num_heads, self.dim_out).permute(2, 0, 1, 3)
        K = K.view(K.size(0), -1, self.num_heads, self.dim_out).permute(2, 0, 1, 3)
        V = V.view(V.size(0), -1, self.num_heads, self.dim_out).permute(2, 0, 1, 3)

        attn = F.softmax(Q @ K.transpose(-2, -1) / self.dim_out**0.5, dim=-1)
        out = attn @ V
        out = out.permute(1, 2, 0, 3).contiguous()

        return (out, attn)


class DeepSetTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        node_layers: list[int],
        set_layers: list[int],
        dropout_prob: float = 0.2,
        norm: str = "batch",
        activation: str = "relu",
        skip_node: bool = False,
        skip_set: bool = False,
        num_heads: int = 1,
        is_concat_attention: bool = True,
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
        self.is_concat_attention = is_concat_attention

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
        for i, out_dim in enumerate(set_layers):
            if i == 0 and self.is_concat_attention:
                in_dim = in_dim * num_heads
            set_modules.append(create_block(in_dim, out_dim, norm, activation))
            in_dim = out_dim
        set_modules.append(nn.Dropout(dropout_prob))
        self.set_layers = nn.ModuleList(set_modules)

        # Create Set Transformer layer between node_layers and set_layers
        self.set_transformer = SelfAttention(
            dim_in=node_layers[-1], dim_out=node_layers[-1], num_heads=num_heads
        )  # Here, using the same dimension for input and output and 1 head.

    def forward(self, x, batch):
        x_node = x
        for i, layer in enumerate(self.node_layers):
            out_node = layer(x_node)
            if self.skip_node and x_node.shape[-1] == out_node.shape[-1]:
                out_node += x_node  # Skip connection
            x_node = out_node

        # Apply Set Transformer Layer here
        x_transformed, attn_weights = self.set_transformer(x_node)
        if self.is_concat_attention:
            x_transformed_cast = x_transformed.reshape(x_transformed.shape[0], 1, 1, -1)
        else:
            x_transformed_cast = x_transformed.mean(dim=2, keepdim=True)

        x_summed = scatter_add(x_transformed_cast, batch, dim=0)

        x_set = x_summed
        for i, layer in enumerate(self.set_layers):
            out_set = layer(x_set)
            if self.skip_set and x_set.shape[-1] == out_set.shape[-1]:
                out_set += x_set  # Skip connection
            x_set = out_set

        x_set = x_set.squeeze(1, 2)

        if x_set.dim() == 1:
            x_set = x_set.unsqueeze(0)

        return (
            x_node,
            x_set,
            x_transformed.squeeze(1),
            attn_weights,
        )  # returning x_transformed as well, in case you need it.


def main():
    input_dim = 10
    node_layers = [64, 32, 32, 32]
    set_layers = [16, 8, 8]

    model = DeepSetTransformer(
        input_dim,
        node_layers,
        set_layers,
        norm="layer",
        activation="gelu",
        skip_node=True,
        skip_set=True,
        num_heads=3,
        is_concat_attention=False,
    )

    x = torch.rand(100, input_dim)
    batch = torch.cat([torch.full((20,), i, dtype=torch.long) for i in range(5)])
    print("x", x.shape)
    x_nodes, x_set, x_transformed, attn_weights = model(x, batch)

    print("x_set.shape", x_set.shape)
    print("x_nodes.shape", x_nodes.shape)
    print("x_transformed.shape", x_transformed.shape)
    print("attn_weights.shape", attn_weights.shape)


if __name__ == "__main__":
    main()
