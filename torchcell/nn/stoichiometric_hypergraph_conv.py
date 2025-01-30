# torchcell/nn/stoichiometric_hypergraph_conv
# [[torchcell.nn.stoichiometric_hypergraph_conv]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/nn/stoichiometric_hypergraph_conv
# Test file: tests/torchcell/nn/test_stoichiometric_hypergraph_conv.py


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


class StoichHypergraphConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_stoich_gated: bool = False,
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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_stoich_gated = is_stoich_gated
        self.use_attention = use_attention
        self.attention_mode = attention_mode
        self.heads = heads if use_attention else 1
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        # Main transformation
        self.lin = Linear(
            in_channels,
            heads * out_channels if use_attention else out_channels,
            bias=False,
            weight_initializer="glorot",
        )

        # Gating network
        if is_stoich_gated:
            self.gate_lin = Linear(
                in_channels, 1, bias=True, weight_initializer="glorot"
            )

        # Attention
        if use_attention:
            self.att = Parameter(torch.empty(1, heads, 2 * out_channels))
        else:
            self.register_parameter("att", None)

        # Bias
        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.is_stoich_gated:
            self.gate_lin.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        if self.bias is not None:
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
        num_nodes = x.size(0)
        num_edges = int(edge_index[1].max()) + 1 if num_edges is None else num_edges

        # Transform node features before splitting for attention
        x_transformed = self.lin(x)

        # Handle attention if enabled
        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x_transformed = x_transformed.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads, self.out_channels)
            x_i = x_transformed[edge_index[0]]
            x_j = hyperedge_attr[edge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(
                alpha,
                edge_index[1] if self.attention_mode == "node" else edge_index[0],
                num_nodes=num_edges if self.attention_mode == "node" else num_nodes,
            )
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Compute gating coefficients if enabled - using original features
        gate_values = None
        if self.is_stoich_gated:
            gate_values = torch.sigmoid(self.gate_lin(x))
            gate_values = gate_values[edge_index[0]]

        # Compute normalization coefficients
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

        # Message passing
        out = self.propagate(
            edge_index,
            x=x_transformed,
            norm=B,
            alpha=alpha,
            stoich=stoich,
            gate_values=gate_values,
            size=(num_nodes, num_edges),
        )

        # Second message passing step
        out = self.propagate(
            edge_index.flip([0]),
            x=out,
            norm=D,
            alpha=alpha,
            stoich=stoich,
            gate_values=gate_values,
            size=(num_edges, num_nodes),
        )

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(
        self,
        x_j: Tensor,
        norm_i: Tensor,
        alpha: Optional[Tensor],
        stoich: Tensor,
        gate_values: Optional[Tensor],
    ) -> Tensor:
        # Split into magnitude and sign
        magnitude = torch.abs(stoich)
        sign = torch.sign(stoich)

        # Apply gating if enabled
        if gate_values is not None:
            magnitude = magnitude * gate_values.view(-1)

        # Combine all components
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
#     conv = StoichHypergraphConv(in_channels=64, out_channels=32)

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
    torch.manual_seed(42)

    # Create instances for comparison
    conv_gated = StoichHypergraphConv(
        in_channels=3, out_channels=2, is_stoich_gated=True
    )
    conv_normal = StoichHypergraphConv(
        in_channels=3, out_channels=2, is_stoich_gated=False
    )

    # Example data
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

    print("\nInput Features:")
    print(x)

    print("\nStoichiometric Coefficients:")
    print("Node\tCoeff\tRole")
    for i, s in enumerate(stoich):
        role = "Product" if s > 0 else "Reactant"
        print(f"{i}\t{s:.1f}\t{role}")

    # Compare gated and non-gated versions
    with torch.no_grad():
        out_gated = conv_gated(x, edge_index, stoich)
        out_normal = conv_normal(x, edge_index, stoich)

        # Get normalized coefficients
        D = scatter(torch.abs(stoich), edge_index[0], dim=0, dim_size=4, reduce="sum")
        D = 1.0 / D
        D[D == float("inf")] = 0

        print("\nDegree Normalization (D):")
        print("Node\tNorm")
        for i, d in enumerate(D):
            print(f"{i}\t{d:.3f}")

        # Show transformed features for both versions
        print("\nGated Version:")
        x_transformed_gated = conv_gated.lin(x)
        print("Transformed Features (after linear layer):")
        print(x_transformed_gated)

        if conv_gated.is_stoich_gated:
            gate_values = torch.sigmoid(conv_gated.gate_lin(x[edge_index[0]]))
            print("\nGate Values:")
            for i, g in enumerate(gate_values):
                print(f"Node {i}: {g.item():.3f}")

        magnitude = torch.abs(stoich)
        sign = torch.sign(stoich)
        messages_gated = (
            magnitude.view(-1, 1)
            * sign.view(-1, 1)
            * x_transformed_gated[edge_index[0]]
        )

        print("\nGated Message Components:")
        print("Node\tMagnitude\tSign\tMessage")
        for i in range(len(stoich)):
            print(
                f"{i}\t{magnitude[i]:.1f}\t\t{sign[i]:.1f}\t{messages_gated[i].tolist()}"
            )

        print("\nGated Final Output Features:")
        print(out_gated)

        print("\nNon-gated Version:")
        x_transformed_normal = conv_normal.lin(x)
        print("Transformed Features (after linear layer):")
        print(x_transformed_normal)

        messages_normal = (
            magnitude.view(-1, 1)
            * sign.view(-1, 1)
            * x_transformed_normal[edge_index[0]]
        )

        print("\nNon-gated Message Components:")
        print("Node\tMagnitude\tSign\tMessage")
        for i in range(len(stoich)):
            print(
                f"{i}\t{magnitude[i]:.1f}\t\t{sign[i]:.1f}\t{messages_normal[i].tolist()}"
            )

        print("\nNon-gated Final Output Features:")
        print(out_normal)


if __name__ == "__main__":
    main()
