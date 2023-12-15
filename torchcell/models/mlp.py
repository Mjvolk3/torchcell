# torchcell/models/mlp.py
# [[torchcell.models.mlp]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/mlp.py
# Test file: torchcell/models/test_mlp.py
from typing import List, Optional

import torch
import torch.nn as nn

from torchcell.models import act_register


class Mlp(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layer_dims: list[int],
        activations: list[str] = None,
        output_activation: str | None = None,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        if activations is None:
            activations = []

        assert len(layer_dims) - 1 == len(
            activations
        ), "Number of activations should be one less than the number of layers"

        layers = []
        in_dim = input_dim
        for i, out_dim in enumerate(
            layer_dims[:-1]
        ):  # Exclude the last layer dimension
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(act_register[activations[i]])
            in_dim = out_dim  # Update in_dim for the next layer

        # Last Layer
        layers.append(nn.Linear(in_dim, layer_dims[-1]))
        layers.append(nn.Dropout(p=dropout_p))  # Apply dropout

        if (
            output_activation
        ):  # Optionally append the output activation after the final layer
            layers.append(act_register[output_activation])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)


if __name__ == "__main__":
    # Generate Fake Data
    batch_size = 64
    input_dim = 100  # e.g. number of features
    x = torch.randn(batch_size, input_dim)

    # Define a fake model
    layer_dims = [128, 64, 32, 10]
    activations = ["relu", "relu", "relu"]
    output_activation = "sigmoid"  # Set to None if you want the last layer to be linear
    dropout_p = 0.2

    model = Mlp(
        input_dim=input_dim,
        layer_dims=layer_dims,
        activations=activations,
        output_activation=output_activation,
        dropout_p=dropout_p,
    )

    # Forward pass
    out = model(x)
    print(out)  # Print the output to check if it works

    # Fake target labels for backward pass
    targets = torch.randint(0, 2, (batch_size, 10)).float()

    # Define loss and perform a backward pass
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss = criterion(out, targets)
    loss.backward()
    optimizer.step()

    print("Loss: ", loss.item())
    ########## single
    input_dim = 100  # e.g. number of features
    layer_dims = [10]  # Only one linear layer with 10 units
    activations = []  # No intermediate activations
    output_activation = (
        None  # No activation after the linear layer, making it a purely linear head
    )
    dropout_p = 0.0  # No dropout

    network = Mlp(
        input_dim=input_dim,
        layer_dims=layer_dims,
        activations=activations,
        output_activation=output_activation,
        dropout_p=dropout_p,
    )

    # Test the network with a random input
    x = torch.randn(64, input_dim)  # 64 is the batch size
    out = network(x)
    print(out.shape)
