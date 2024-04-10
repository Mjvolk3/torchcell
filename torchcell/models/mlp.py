# torchcell/models/mlp.py
# [[torchcell.models.mlp]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/mlp.py
# Test file: torchcell/models/test_mlp.py

import torch
import torch.nn as nn
from torchcell.models import act_register


class Mlp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout_prob: float = 0.0,
        norm: str = "none",
        activation: str = "relu",
        output_activation: str | None = None,
    ):
        super().__init__()

        assert norm in ["none", "batch", "instance", "layer"], "Invalid norm type"
        assert activation in act_register.keys(), "Invalid activation type"

        def create_block(in_dim, out_dim, norm, activation):
            block = [nn.Linear(in_dim, out_dim)]
            if norm != "none":
                if norm == "batch":
                    block.append(nn.BatchNorm1d(out_dim))
                elif norm == "instance":
                    block.append(nn.InstanceNorm1d(out_dim, affine=True))
                elif norm == "layer":
                    block.append(nn.LayerNorm(out_dim))
            block.append(act_register[activation])
            return nn.Sequential(*block)

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(
                    create_block(in_channels, hidden_channels, norm, activation)
                )
            elif i == num_layers - 1:
                layers.append(
                    create_block(hidden_channels, out_channels, norm, activation)
                )
                layers.append(nn.Dropout(dropout_prob))
            else:
                layers.append(
                    create_block(hidden_channels, hidden_channels, norm, activation)
                )

        if output_activation:
            layers.append(act_register[output_activation])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)


if __name__ == "__main__":
    # Generate Fake Data
    batch_size = 64
    input_dim = 100

    # Define a fake model
    in_channels = input_dim
    hidden_channels = 128
    out_channels = 10
    num_layers = 4
    dropout_prob = 0.2
    norm = "none"
    activation = "relu"
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
