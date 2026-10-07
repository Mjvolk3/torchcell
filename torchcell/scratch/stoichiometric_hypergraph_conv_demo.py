# torchcell/scratch/stoichiometric_hypergraph_conv_demo
# [[torchcell.scratch.stoichiometric_hypergraph_conv_demo]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/scratch/stoichiometric_hypergraph_conv_demo
"""Run a small StoichHypergraphConv example and print the output shape.

Moved verbatim from torchcell/nn/stoichiometric_hypergraph_conv.py on 2026-10-06
(test campaign Phase 23) because it is a demo and not library code.

Run from the repo root:

    PYTHONPATH=$PWD python torchcell/scratch/stoichiometric_hypergraph_conv_demo.py

It builds random toy tensors in memory, so it needs no .env, no data and no GPU.
"""

import torch
from torch_geometric.utils import scatter

from torchcell.nn.stoichiometric_hypergraph_conv import StoichHypergraphConv


def main() -> None:
    """Run a small StoichHypergraphConv example and print the output shape."""
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
