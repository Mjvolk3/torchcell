"""Log-cosh regression loss module."""

import torch
import torch.nn as nn


class LogCoshLoss(nn.Module):
    """Log-cosh loss: a smooth, outlier-robust alternative to MSE."""

    def __init__(self, reduction: str = "mean") -> None:
        """Initialize the loss with the given reduction ('none', 'mean', or 'sum')."""
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the log-cosh of the residual, reduced per ``self.reduction``."""
        loss = torch.log(torch.cosh(input - target))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
