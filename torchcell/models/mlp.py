# torchcell/models/mlp.py
# [[torchcell.models.mlp]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/mlp.py
# Test file: torchcell/models/test_mlp.py

import torch
import torch.nn as nn
from torchcell.models import act_register
from typing import Optional

class Mlp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout_prob: float = 0.0,
        norm: Optional[str] = None,
        activation: Optional[str] = None,
        output_activation: Optional[str] = None,
    ):
        super().__init__()
        assert norm in [None, "batch", "instance", "layer"], "Invalid norm type"
        assert activation in [None] + list(
            act_register.keys()
        ), "Invalid activation type"

        def create_block(
            in_dim: int, out_dim: int, norm: Optional[str], activation: Optional[str]
        ) -> nn.Sequential:
            block = [nn.Linear(in_dim, out_dim)]
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

        layers = []
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
        return self.model(x).squeeze(-1)


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
