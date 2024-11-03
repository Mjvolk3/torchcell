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
from typing import Optional
import torch.optim as optim


class NaNTolerantCorrelation(Metric):
    def __init__(self, base_metric, default_value=torch.nan):
        super().__init__()
        self.base_metric = base_metric()
        self.default_value = default_value

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Create a mask for non-NaN values
        mask = ~torch.isnan(preds) & ~torch.isnan(target)

        # If there are any non-NaN values, update the metric
        if torch.any(mask):
            self.base_metric.update(preds[mask], target[mask])

    def compute(self):
        try:
            # Compute the metric if there were valid samples
            result = self.base_metric.compute()
        except ValueError:
            # If no valid samples, return torch.nan
            result = self.default_value
        return result


class NaNTolerantPearsonCorrCoef(NaNTolerantCorrelation):
    def __init__(self):
        super().__init__(PearsonCorrCoef)


class NaNTolerantSpearmanCorrCoef(NaNTolerantCorrelation):
    def __init__(self):
        super().__init__(SpearmanCorrCoef)


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
        mask = ~torch.isnan(y_true)

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
        mask = ~torch.isnan(y_true)

        # Count valid samples per dimension
        n_valid = mask.sum(dim=0).clamp(min=1)  # Avoid division by zero

        # Zero out predictions where target is NaN to avoid gradient computation
        y_pred_masked = y_pred * mask.float()
        y_true_masked = torch.where(mask, y_true, torch.zeros_like(y_true))

        # Compute squared error only for valid elements
        squared_error = (y_pred_masked - y_true_masked).pow(2)

        # Sum errors for each dimension (only valid elements contribute)
        dim_losses = squared_error.sum(dim=0)

        # Compute mean loss per dimension
        dim_means = dim_losses / n_valid

        return dim_means, mask


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
            "weights",
            torch.ones(2) if weights is None else weights / weights.sum()
        )

    def forward(self, y_pred, y_true):
        """
        Compute weighted loss while handling NaN values.

        Args:
            y_pred (torch.Tensor): Predictions [batch_size, num_dims]
            y_true (torch.Tensor): Ground truth [batch_size, num_dims]

        Returns:
            tuple: (total_loss, dim_losses)
        """
        # Compute per-dimension losses and get validity mask
        dim_losses, mask = self.loss_fn(y_pred, y_true)

        # Apply weights only to dimensions that have valid samples
        valid_dims = mask.any(dim=0)  # Dimensions with at least one valid sample
        # weights is already on the correct device due to register_buffer
        weights = self.weights * valid_dims.float()
        weight_sum = weights.sum().clamp(min=1e-8)  # Avoid division by zero

        # Compute weighted average loss
        weighted_loss = (dim_losses * weights).sum() / weight_sum

        return weighted_loss, dim_losses

    @torch.no_grad()
    def compute_metrics(self, y_pred, y_true):
        """
        Compute additional metrics for monitoring.
        """
        metrics = {}
        mask = ~torch.isnan(y_true)

        for dim in range(y_true.shape[1]):
            dim_mask = mask[:, dim]
            if dim_mask.any():
                dim_pred = y_pred[dim_mask, dim]
                dim_true = y_true[dim_mask, dim]
                metrics[f"dim_{dim}_samples"] = dim_mask.sum().item()
                metrics[f"dim_{dim}_mean_error"] = (
                    (dim_pred - dim_true).abs().mean().item()
                )

        return metrics


if __name__ == "__main__":
    pass
