# torchcell/models/deep_set.py
# [[torchcell.models.deep_set]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/deep_set.py
# Test file: torchcell/models/test_deep_set.py

import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_max, scatter_mean


class SimpleLinearModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, scatter: str = "add"):
        super().__init__()
        self.scatter = scatter
        # The main linear layer for the aggregated data
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, batch):
        # Aggregate the data
        if self.scatter == "add":
            x = scatter_add(x, batch, dim=0)
        elif self.scatter == "mean":
            x = scatter_mean(x, batch, dim=0)
        elif self.scatter == "max":
            x = scatter_max(x, batch, dim=0)

        # Process aggregated data through the linear layer
        x_set = self.linear(x)

        return x_set


def main():
    torch.autograd.set_detect_anomaly(True)

    # Model configuration
    input_dim = 10
    output_dim = 8  # For the aggregated data

    model = SimpleLinearModel(input_dim, output_dim)

    # Dummy data
    x = torch.rand(100, input_dim)
    batch = torch.cat([torch.full((20,), i, dtype=torch.long) for i in range(5)])

    # Forward pass
    x_set = model(x, batch)
    print(x_set.shape)

    # Let's assume you want to predict some values for each set.
    # So, we'll create a dummy target tensor for demonstration purposes.
    target = torch.rand(5, output_dim)

    # Simple mean squared error loss
    criterion = nn.MSELoss()
    loss = criterion(x_set, target)
    print("Loss:", loss.item())

    # Backpropagation
    model.zero_grad()
    loss.backward()
    print("Gradients computed successfully!")


if __name__ == "__main__":
    main()
