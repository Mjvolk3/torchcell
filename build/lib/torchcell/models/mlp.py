# torchcell/models/mlp.py
# [[torchcell.models.mlp]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/mlp.py
# Test file: torchcell/models/test_mlp.py
"""Configurable multilayer perceptron with optional normalization and activations."""

import torch
import torch.nn as nn

from torchcell.models import act_register


class Mlp(nn.Module):
    """Sequential MLP built from linear blocks with optional norm and activation."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout_prob: float = 0.0,
        norm: str | None = None,
        activation: str | None = None,
        output_activation: str | None = None,
    ):
        """Build the layer stack from the channel sizes, norm, and activation choices.

        Args:
            in_channels: Number of input features.
            hidden_channels: Width of hidden layers.
            out_channels: Number of output features.
            num_layers: Total number of linear blocks.
            dropout_prob: Dropout probability applied before the final block.
            norm: Optional normalization ("batch", "instance", or "layer").
            activation: Optional activation name from ``act_register``.
            output_activation: Optional activation applied to the final output.
        """
        super().__init__()
        assert norm in [None, "batch", "instance", "layer"], "Invalid norm type"
        assert activation in [None] + list(act_register.keys()), (
            "Invalid activation type"
        )

        def create_block(
            in_dim: int, out_dim: int, norm: str | None, activation: str | None
        ) -> nn.Sequential:
            block: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
            if norm:
                if norm == "batch":
                    block.append(nn.BatchNorm1d(out_dim))
                elif norm == "instance":
                    block.append(nn.InstanceNorm1d(out_dim, affine=True))
                elif norm == "layer":
                    block.append(nn.LayerNorm(out_dim))
            if activation:
                block.append(act_register[activation])
            return nn.Sequential(*block)

        layers: list[nn.Module] = []
        for i in range(num_layers):
            if num_layers == 1:
                # Directly map from in_channels to out_channels for a single-layer model
                layers.append(create_block(in_channels, out_channels, norm, activation))
                break
            elif i == 0:
                layers.append(
                    create_block(in_channels, hidden_channels, norm, activation)
                )
            elif i == num_layers - 1:
                layers.append(create_block(hidden_channels, out_channels, None, None))
                layers.append(nn.Dropout(dropout_prob))
            else:
                layers.append(
                    create_block(hidden_channels, hidden_channels, norm, activation)
                )

        if output_activation:
            layers.append(act_register[output_activation])

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the input through the MLP and squeeze the trailing dimension."""
        out: torch.Tensor = self.model(x)
        return out.squeeze(-1)


if __name__ == "__main__":
    # Generate Fake Data
    batch_size = 64
    input_dim = 100

    # Define a fake model
    in_channels = input_dim
    hidden_channels = 0
    out_channels = 10
    num_layers = 1
    dropout_prob = 0.2
    norm = None
    activation = None
    output_activation = "sigmoid"

    model = Mlp(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        dropout_prob=dropout_prob,
        norm=norm,
        activation=activation,
        output_activation=output_activation,
    )

    # Forward pass
    x = torch.randn(batch_size, input_dim)
    out = model(x)
    print(out.shape)

    # Fake target labels for backward pass
    targets = torch.randint(0, 2, (batch_size, out_channels)).float()

    # Define loss and perform a backward pass
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = criterion(out, targets)
    loss.backward()
    optimizer.step()
    print("Loss: ", loss.item())
