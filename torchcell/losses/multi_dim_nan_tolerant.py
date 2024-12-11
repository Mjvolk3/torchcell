import math
import os.path as osp
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    PearsonCorrCoef,
    SpearmanCorrCoef,
)
from tqdm import tqdm
import wandb
from torchcell.losses import WeightedMSELoss
from torchcell.viz import fitness, genetic_interaction_score
from torchcell.losses.list_mle import ListMLELoss
import torchcell
from torchmetrics import Metric
import torch
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef
import logging
import sys
from typing import Optional, Tuple
import torch.optim as optim
import torch.nn.functional as F


class MultiDimNaNTolerantL1Loss(nn.Module):
    def __init__(self):
        super(MultiDimNaNTolerantL1Loss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Compute L1 loss while properly handling NaN values.

        Args:
            y_pred (torch.Tensor): Predictions [batch_size, num_dims]
            y_true (torch.Tensor): Ground truth [batch_size, num_dims]

        Returns:
            tuple: (dim_means, mask) - Loss per dimension and validity mask
        """
        # Ensure tensors have the same shape
        assert (
            y_pred.shape == y_true.shape
        ), "Predictions and targets must have the same shape"

        # Create mask for non-NaN values
        mask = ~torch.isfloat("nan")(y_true)

        # Count valid samples per dimension
        n_valid = mask.sum(dim=0).clamp(min=1)  # Avoid division by zero

        # Zero out predictions where target is NaN to avoid gradient computation
        y_pred_masked = y_pred * mask.float()
        y_true_masked = torch.where(mask, y_true, torch.zeros_like(y_true))

        # Compute absolute error only for valid elements
        absolute_error = (y_pred_masked - y_true_masked).abs()

        # Sum errors for each dimension (only valid elements contribute)
        dim_losses = absolute_error.sum(dim=0)

        # Compute mean loss per dimension
        dim_means = dim_losses / n_valid

        return dim_means, mask


class MultiDimNaNTolerantMSELoss(nn.Module):
    def __init__(self):
        super(MultiDimNaNTolerantMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Compute MSE loss while properly handling NaN values.
        """
        device = y_pred.device

        # Ensure tensors have the same shape and device
        assert (
            y_pred.shape == y_true.shape
        ), "Predictions and targets must have the same shape"
        y_true = y_true.to(device)

        # Create mask for non-NaN values
        mask = ~torch.isnan(y_true)

        # Count valid samples per dimension
        n_valid = mask.sum(dim=0).clamp(min=1)

        # Zero out predictions where target is NaN
        y_pred_masked = y_pred * mask
        y_true_masked = y_true.masked_fill(~mask, 0)

        # Compute squared error only for valid elements
        squared_error = (y_pred_masked - y_true_masked).pow(2)

        # Sum errors for each dimension
        dim_losses = squared_error.sum(dim=0)

        # Compute mean loss per dimension
        dim_means = dim_losses / n_valid

        return dim_means.to(device), mask.to(device)


class CombinedLoss(nn.Module):
    def __init__(self, loss_type="mse", weights=None):
        super(CombinedLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == "mse":
            self.loss_fn = MultiDimNaNTolerantMSELoss()
        elif loss_type == "l1":
            self.loss_fn = MultiDimNaNTolerantL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        # Register weights as a buffer so it's automatically moved with the module
        self.register_buffer(
            "weights", torch.ones(2) if weights is None else weights / weights.sum()
        )

    def forward(self, y_pred, y_true):
        """
        Compute weighted loss while handling NaN values.
        """
        # Get device from input
        device = y_pred.device

        # Ensure inputs are on same device
        y_pred = y_pred.to(device)
        y_true = y_true.to(device)
        weights = self.weights.to(device)

        # Compute per-dimension losses and get validity mask
        dim_losses, mask = self.loss_fn(y_pred, y_true)

        # Ensure all intermediate computations stay on device
        valid_dims = mask.any(dim=0).to(device)
        weights = weights * valid_dims
        weight_sum = weights.sum().clamp(min=1e-8)

        # Compute weighted average loss (all tensors now on same device)
        weighted_loss = (dim_losses * weights).sum() / weight_sum

        return weighted_loss.to(device), dim_losses.to(device)


class MultiDimNaNTolerantCELoss(nn.Module):
    def __init__(self, num_classes: int, num_tasks: int = 2, eps: float = 1e-7):
        """
        Cross Entropy loss that handles NaN values and supports any number of classes per task.

        Args:
            num_classes: Number of classes (bins) per task
            num_tasks: Number of tasks (e.g., 2 for fitness and gene_interaction)
            eps: Small value for numerical stability
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.eps = eps

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Cross Entropy loss while handling NaN values.

        Args:
            logits: Model predictions [batch_size, num_tasks * num_classes]
            targets: Target labels [batch_size, num_tasks * num_classes]

        Returns:
            tuple: (dim_losses, mask)
        """
        device = logits.device
        batch_size = logits.shape[0]

        # Reshape inputs to separate tasks
        logits = logits.view(batch_size, self.num_tasks, self.num_classes)
        targets = targets.view(batch_size, self.num_tasks, self.num_classes)

        # Move tensors to correct device
        logits = logits.to(device)
        targets = targets.to(device)

        # Create mask for non-NaN entries
        mask = ~torch.isnan(targets).any(dim=-1)  # [batch_size, num_tasks]

        # Count valid samples per task
        valid_samples = mask.sum(dim=0).clamp(min=1)  # [num_tasks]

        # Apply log_softmax to get log probabilities
        log_probs = F.log_softmax(
            logits, dim=-1
        )  # [batch_size, num_tasks, num_classes]

        # Replace NaN targets with 0 to avoid NaN propagation
        targets = torch.nan_to_num(targets, nan=0.0)

        # Compute cross entropy loss (-sum(target * log_prob))
        loss = -(targets * log_probs).sum(dim=-1)  # [batch_size, num_tasks]

        # Mask out invalid entries
        masked_loss = loss * mask.float()

        # Compute mean loss per dimension
        dim_losses = masked_loss.sum(dim=0) / valid_samples

        return dim_losses.to(device), mask.to(device)


class CombinedCELoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_tasks: int = 2,
        weights: Optional[torch.Tensor] = None,
    ):
        """
        Combined Cross Entropy loss with dimension-wise weighting.

        Args:
            num_classes: Number of classes (bins) per task
            num_tasks: Number of tasks
            weights: Optional tensor of weights for each dimension [num_tasks]
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.loss_fn = MultiDimNaNTolerantCELoss(
            num_classes=num_classes, num_tasks=num_tasks
        )

        # Register weights as a buffer
        if weights is None:
            weights = torch.ones(num_tasks)
        self.register_buffer("weights", weights / weights.sum())

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute weighted cross entropy loss.

        Args:
            logits: Model predictions [batch_size, num_tasks * num_classes]
            targets: Target labels [batch_size, num_tasks * num_classes]

        Returns:
            tuple: (total_loss, dim_losses)
        """
        device = logits.device

        # Compute per-dimension losses and get validity mask
        dim_losses, mask = self.loss_fn(logits, targets)

        # Get mask for valid dimensions
        valid_dims = mask.any(dim=0)
        weights = self.weights.to(device)

        # Apply weights only to valid dimensions
        weights = weights * valid_dims.float()
        weight_sum = weights.sum().clamp(min=1e-8)

        dim_losses = weights * dim_losses
        # Compute weighted average loss
        total_loss = (dim_losses).sum() / weight_sum

        return total_loss.to(device), dim_losses.to(device)


if __name__ == "__main__":
    # Test case
    logits = torch.tensor(
        [
            [0.6370, -0.0156, 0.2711, -0.2273],
            [-0.2947, -0.1466, 1.2458, -0.9816],
            [-0.4593, -0.2630, 1.2785, -0.8181],
            [-0.2947, -0.1466, 1.2459, -0.9816],
        ]
    )
    y = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, float("nan"), float("nan")],
            [0.0, 1.0, float("nan"), float("nan")],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    loss = CombinedCELoss(num_classes=2, num_tasks=2)
    total_loss, dim_losses = loss(logits, y)
    print(f"Total loss: {total_loss}")
    print(f"Dimension losses: {dim_losses}")
