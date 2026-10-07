# torchcell/scratch/linear_demo
# [[torchcell.scratch.linear_demo]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/scratch/linear_demo
"""Run a small forward/backward pass to sanity-check the model.

Moved verbatim from torchcell/models/linear.py on 2026-10-06 (test campaign Phase
23) because it is a demo and not library code.

Run from the repo root:

    PYTHONPATH=$PWD python torchcell/scratch/linear_demo.py

Needs nothing beyond the installed packages: it builds random dummy data on the
CPU and reads no environment variables.
"""

import torch
import torch.nn as nn

from torchcell.models.linear import SimpleLinearModel


def main() -> None:
    """Run a small forward/backward pass to sanity-check the model."""
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
