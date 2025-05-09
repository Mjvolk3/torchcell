import torch
import torch.nn as nn


class LogCoshLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.log(torch.cosh(input - target))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
