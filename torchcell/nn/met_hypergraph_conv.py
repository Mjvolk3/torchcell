# torchcell/nn/met_hypergraph_conv
# [[torchcell.nn.met_hypergraph_conv]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/nn/met_hypergraph_conv
# Test file: tests/torchcell/nn/test_met_hypergraph_conv.py

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, softmax


class StoichiometricHypergraphConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = False,
        attention_mode: str = "node",
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(flow="source_to_target", node_dim=0, **kwargs)

        assert attention_mode in ["node", "edge"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.attention_mode = attention_mode

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.lin = Linear(
                in_channels,
                heads * out_channels,
                bias=False,
                weight_initializer="glorot",
            )
            self.att = Parameter(torch.empty(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.lin = Linear(
                in_channels, out_channels, bias=False, weight_initializer="glorot"
            )

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    @disable_dynamic_shapes(required_args=["num_edges"])
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        stoich: Tensor,
        hyperedge_attr: Optional[Tensor] = None,
        num_edges: Optional[int] = None,
    ) -> Tensor:
        """
        Args:
            x (Tensor): Node feature matrix [num_nodes, in_channels]
            edge_index (Tensor): [2, num_edges] tensor representing standard PyG edge_index
            stoich (Tensor): [num_edges] tensor of stoichiometric coefficients
            hyperedge_attr (Tensor, optional): Hyperedge feature matrix
            num_edges (int, optional): Number of hyperedges
        """
        num_nodes = x.size(0)

        if num_edges is None:
            num_edges = 0
            if edge_index.numel() > 0:
                num_edges = int(edge_index[1].max()) + 1

        x = self.lin(x)

        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads, self.out_channels)
            x_i = x[edge_index[0]]
            x_j = hyperedge_attr[edge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            if self.attention_mode == "node":
                alpha = softmax(alpha, edge_index[1], num_nodes=num_edges)
            else:
                alpha = softmax(alpha, edge_index[0], num_nodes=num_nodes)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Use absolute stoichiometric coefficients for degree calculation
        D = scatter(
            torch.abs(stoich), edge_index[0], dim=0, dim_size=num_nodes, reduce="sum"
        )
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter(
            torch.abs(stoich), edge_index[1], dim=0, dim_size=num_edges, reduce="sum"
        )
        B = 1.0 / B
        B[B == float("inf")] = 0

        # Propagate messages using stoichiometric weights
        out = self.propagate(
            edge_index,
            x=x,
            norm=B,
            alpha=alpha,
            stoich=stoich,
            size=(num_nodes, num_edges),
        )
        out = self.propagate(
            edge_index.flip([0]),
            x=out,
            norm=D,
            alpha=alpha,
            stoich=stoich,
            size=(num_edges, num_nodes),
        )

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(
        self, x_j: Tensor, norm_i: Tensor, alpha: Tensor, stoich: Tensor
    ) -> Tensor:
        # Split into magnitude and sign
        magnitude = torch.abs(stoich)
        sign = torch.sign(stoich)

        # Apply magnitude for scaling but preserve sign
        out = (
            norm_i.view(-1, 1, 1)
            * magnitude.view(-1, 1, 1)
            * sign.view(-1, 1, 1)
            * x_j.view(-1, self.heads, self.out_channels)
        )

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out


# def main():
#     # Create instance
#     conv = StoichiometricHypergraphConv(in_channels=64, out_channels=32)

#     # Example data
#     x = torch.randn(4, 64)  # 4 nodes with 64 features each

#     # Create edge_index and stoichiometric coefficients separately
#     edge_index = torch.tensor(
#         [[0, 1, 2, 1, 2, 3], [0, 0, 0, 1, 1, 1]],  # Node indices  # Hyperedge indices
#         dtype=torch.long,
#     )

#     stoich = torch.tensor([-1.0, 2.0, 1.0, -2.0, -1.0, 3.0], dtype=torch.float)

#     # Forward pass
#     out = conv(x, edge_index, stoich)
#     print("Output shape:", out.shape)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create instance
    conv = StoichiometricHypergraphConv(
        in_channels=3, out_channels=2
    )  # Smaller for visualization

    # Example data - simplified for visualization
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # Node 0
            [0.0, 1.0, 0.0],  # Node 1
            [0.0, 0.0, 1.0],  # Node 2
            [1.0, 1.0, 0.0],  # Node 3
        ]
    )

    # Simple reaction: A + 2B -> C
    edge_index = torch.tensor(
        [
            [0, 1, 2],  # Node indices (A, B, C)
            [0, 0, 0],  # All part of same reaction (edge 0)
        ],
        dtype=torch.long,
    )

    stoich = torch.tensor([-1.0, -2.0, 1.0], dtype=torch.float)  # A + 2B -> C

    # Forward pass
    out = conv(x, edge_index, stoich)

    # Visualize message components
    print("\nInput Features:")
    print(x)

    print("\nStoichiometric Coefficients:")
    print("Node\tCoeff\tRole")
    for i, s in enumerate(stoich):
        role = "Product" if s > 0 else "Reactant"
        print(f"{i}\t{s:.1f}\t{role}")

    # Get normalized coefficients
    D = scatter(torch.abs(stoich), edge_index[0], dim=0, dim_size=4, reduce="sum")
    D = 1.0 / D
    D[D == float("inf")] = 0

    print("\nDegree Normalization (D):")
    print("Node\tNorm")
    for i, d in enumerate(D):
        print(f"{i}\t{d:.3f}")

    # Extract message components from a forward pass
    with torch.no_grad():
        # Get transformed features
        x_transformed = conv.lin(x)

        print("\nTransformed Features (after linear layer):")
        print(x_transformed)

        # Compute messages for one edge
        magnitude = torch.abs(stoich)
        sign = torch.sign(stoich)
        messages = (
            magnitude.view(-1, 1) * sign.view(-1, 1) * x_transformed[edge_index[0]]
        )

        print("\nMessage Components:")
        print("Node\tMagnitude\tSign\tMessage")
        for i in range(len(stoich)):
            print(f"{i}\t{magnitude[i]:.1f}\t\t{sign[i]:.1f}\t{messages[i].tolist()}")

        print("\nAggregated Message:")
        agg_message = messages.sum(0)
        print(agg_message.tolist())

        print("\nFinal Output Features:")
        print(out)


if __name__ == "__main__":
    main()
