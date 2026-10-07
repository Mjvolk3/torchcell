"""Simple set-aggregation linear model over scatter-pooled node features."""

# torchcell/models/deep_set.py
# [[torchcell.models.deep_set]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/deep_set.py
# Test file: torchcell/models/test_deep_set.py

import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_max, scatter_mean


class SimpleLinearModel(nn.Module):
    """Scatter-aggregate node features per set, then apply a linear layer."""

    def __init__(self, input_dim: int, output_dim: int, scatter: str = "add"):
        """Set up the linear layer and the scatter aggregation mode.

        Args:
            input_dim: Dimension of each input node feature.
            output_dim: Dimension of the per-set output.
            scatter: Aggregation mode, one of "add", "mean", or "max".
        """
        super().__init__()
        self.scatter = scatter
        # The main linear layer for the aggregated data
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Aggregate node features by batch and project to the output dimension."""
        # Aggregate the data
        if self.scatter == "add":
            x = scatter_add(x, batch, dim=0)
        elif self.scatter == "mean":
            x = scatter_mean(x, batch, dim=0)
        elif self.scatter == "max":
            x = scatter_max(x, batch, dim=0)

        # Process aggregated data through the linear layer
        x_set: torch.Tensor = self.linear(x)

        return x_set


if __name__ == "__main__":
    raise SystemExit(
        "the demo main moved to torchcell/scratch/linear_demo.py on 2026-10-06"
    )
