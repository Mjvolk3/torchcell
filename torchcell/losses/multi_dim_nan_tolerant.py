# torchcell/losses/multi_dim_nan_tolerant
# [[torchcell.losses.multi_dim_nan_tolerant]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/losses/multi_dim_nan_tolerant
# Test file: tests/torchcell/losses/test_multi_dim_nan_tolerant.py
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


class SupCR(nn.Module):
    """
    Supervised Contrastive Regression (SupCR) loss that handles multiple target dimensions.
    Uses efficient O(N^2 log N) implementation with row-wise sorting and suffix sums.
    """

    def __init__(self, temperature: float = 0.1, eps: float = 1e-7) -> None:
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def _reversed_cumsum(self, x: torch.Tensor) -> torch.Tensor:
        """Compute cumulative sum in reverse order."""
        return torch.flip(torch.cumsum(torch.flip(x, dims=(0,)), dim=0), dims=(0,))

    def compute_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarities between embeddings."""
        normed = F.normalize(embeddings, p=2, dim=1)
        return torch.mm(normed, normed.t()) / self.temperature

    def compute_dimension_loss(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SupCR loss for a single dimension.

        Args:
            embeddings: Shape [batch_size, embedding_dim]
            labels: Shape [batch_size] for single dimension
        """
        device = labels.device

        # Handle NaN values
        valid_mask = ~torch.isnan(labels)
        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=device)

        valid_labels = labels[valid_mask]
        valid_embeddings = embeddings[valid_mask]
        M = valid_labels.size(0)

        # Compute pairwise matrices
        dists = torch.abs(valid_labels.unsqueeze(0) - valid_labels.unsqueeze(1))
        sims = self.compute_similarity(valid_embeddings)
        exp_sims = torch.exp(sims)

        # Skip diagonal elements
        eye_mask = ~torch.eye(M, dtype=torch.bool, device=device)

        accum_loss = torch.tensor(0.0, device=device)
        accum_count = 0

        # Process each anchor point
        for i in range(M):
            row_dists = dists[i]
            sorted_dists_i, idx_i = row_dists.sort()
            sorted_sims_i = exp_sims[i][idx_i]
            suffix_sums_i = self._reversed_cumsum(sorted_sims_i)

            insertion_positions = torch.searchsorted(
                sorted_dists_i, row_dists, side="left"
            )
            denominators = suffix_sums_i[insertion_positions].clamp(min=self.eps)
            numerators = exp_sims[i]

            row_loss = -torch.log(numerators / denominators)

            valid_j_mask = eye_mask[i]
            accum_loss += row_loss[valid_j_mask].sum()
            accum_count += valid_j_mask.sum().item()

        return accum_loss / max(accum_count, 1)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute losses for all dimensions.

        Args:
            embeddings: Shape [batch_size, embedding_dim]
            labels: Shape [batch_size, num_dims]
        """
        num_dims = labels.shape[1]
        losses = []

        for dim in range(num_dims):
            dim_loss = self.compute_dimension_loss(embeddings, labels[:, dim])
            losses.append(dim_loss)

        return torch.stack(losses)


class WeightedSupCRCell(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        weights: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
    ):
        super().__init__()
        if weights is None:
            weights = torch.ones(2)
        self.register_buffer("weights", weights / weights.sum())
        self.supcr_perturbed = SupCR(temperature=temperature, eps=eps)
        self.supcr_intact = SupCR(temperature=temperature, eps=eps)

    def forward(
        self,
        perturbed_embeddings: torch.Tensor,
        intact_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = labels.device

        # Create mask for non-NaN values
        mask = ~torch.isnan(labels)

        # Check which dimensions have ANY valid samples
        valid_dims = mask.any(dim=0)  # [num_dims]

        # If no dimensions are valid, return zero loss
        if not valid_dims.any():
            return torch.tensor(0.0, device=device), torch.zeros_like(
                valid_dims, dtype=torch.float
            )

        # Compute per-dimension losses for both embedding types
        perturbed_losses = self.supcr_perturbed(perturbed_embeddings, labels)
        intact_losses = self.supcr_intact(intact_embeddings, labels)

        # Average the losses from perturbed and intact
        dim_losses = (perturbed_losses + intact_losses) / 2

        # Zero out weights for completely NaN dimensions
        weights = self.weights * valid_dims
        weight_sum = weights.sum().clamp(min=1e-8)

        # Apply dimension weights and compute total loss
        weighted_loss = (dim_losses * weights).sum() / weight_sum

        return weighted_loss, dim_losses


class NaNTolerantL1Loss(nn.Module):
    def __init__(self):
        super(NaNTolerantL1Loss, self).__init__()

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


class NaNTolerantHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        """
        Initialize Huber loss with NaN handling.

        Args:
            delta (float): Threshold where the loss transitions from quadratic to linear.
                         Default is 1.0
        """
        super(NaNTolerantHuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        """
        Compute Huber loss while properly handling NaN values.

        Args:
            y_pred (torch.Tensor): Predictions [batch_size, num_dims]
            y_true (torch.Tensor): Ground truth [batch_size, num_dims]

        Returns:
            tuple: (dim_means, mask) - Loss per dimension and validity mask
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
        n_valid = mask.sum(dim=0).clamp(min=1)  # Avoid division by zero

        # Zero out predictions where target is NaN
        y_pred_masked = y_pred * mask
        y_true_masked = y_true.masked_fill(~mask, 0)

        # Compute absolute error
        abs_error = torch.abs(y_pred_masked - y_true_masked)

        # Apply Huber loss formula
        quadratic_mask = abs_error <= self.delta
        linear_mask = ~quadratic_mask

        # Quadratic portion: 0.5 * error^2
        quadratic_loss = 0.5 * abs_error.pow(2) * quadratic_mask.float()

        # Linear portion: delta * error - 0.5 * delta^2
        linear_loss = (
            self.delta * abs_error - 0.5 * self.delta**2
        ) * linear_mask.float()

        # Combine losses
        huber_loss = quadratic_loss + linear_loss

        # Sum errors for each dimension (only valid elements contribute)
        dim_losses = huber_loss.sum(dim=0)

        # Compute mean loss per dimension
        dim_means = dim_losses / n_valid

        return dim_means.to(device), mask.to(device)


class NaNTolerantLogCoshLoss(nn.Module):
    def __init__(self):
        super(NaNTolerantLogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Compute log(cosh) loss while properly handling NaN values.

        The log(cosh) loss is defined as:
        L = log(cosh(y_pred - y_true))

        Args:
            y_pred (torch.Tensor): Predictions [batch_size, num_dims]
            y_true (torch.Tensor): Ground truth [batch_size, num_dims]

        Returns:
            tuple: (dim_means, mask) - Loss per dimension and validity mask
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
        n_valid = mask.sum(dim=0).clamp(min=1)  # Avoid division by zero

        # Zero out predictions where target is NaN
        y_pred_masked = y_pred * mask
        y_true_masked = y_true.masked_fill(~mask, 0)

        # Compute log(cosh(x)) loss only for valid elements
        diff = y_pred_masked - y_true_masked
        logcosh = torch.log(
            torch.cosh(diff + 1e-12)
        )  # Add small epsilon for numerical stability

        # Sum errors for each dimension (only valid elements contribute)
        dim_losses = logcosh.sum(dim=0)

        # Compute mean loss per dimension
        dim_means = dim_losses / n_valid

        return dim_means.to(device), mask.to(device)


class WeightedMSELoss(nn.Module):
    def __init__(self, weights: Optional[torch.Tensor] = None):
        super().__init__()
        if weights is not None:
            self.register_buffer("weights", weights / weights.sum())
        else:
            self.weights = None

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = predictions.device
        predictions, targets = predictions.to(device), targets.to(device)

        # Create mask for non-NaN values
        mask = ~torch.isnan(targets)

        # Check which dimensions have ANY valid samples
        valid_dims = mask.any(dim=0)  # [num_dims]

        # If no dimensions are valid, return zero loss
        if not valid_dims.any():
            return torch.tensor(0.0, device=device), torch.zeros_like(
                valid_dims, dtype=torch.float
            )

        # Get count of valid samples per dimension, avoid division by zero
        valid_samples = mask.sum(dim=0).clamp(min=1)

        # Zero out predictions where target is NaN
        masked_predictions = predictions * mask
        masked_targets = targets.masked_fill(~mask, 0)

        # Compute squared error
        squared_errors = (masked_predictions - masked_targets).pow(2)

        # Average errors over valid samples for each dimension
        dim_losses = squared_errors.sum(dim=0) / valid_samples

        # Zero out weights for completely NaN dimensions
        weights = (
            self.weights * valid_dims
            if self.weights is not None
            else valid_dims.float()
        )
        weight_sum = weights.sum().clamp(min=1e-8)

        # Compute weighted average
        total_loss = (dim_losses * weights).sum() / weight_sum

        return total_loss, dim_losses


class NaNTolerantMSELoss(nn.Module):
    def __init__(self):
        super(NaNTolerantMSELoss, self).__init__()

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


class NaNTolerantQuantileLoss(nn.Module):
    """
    Quantile regression loss that handles NaN values.
    For each quantile q, the loss is:
    L = q * (y - y_pred) if y > y_pred
    L = (1-q) * (y_pred - y) if y <= y_pred
    """

    def __init__(self, quantiles: list[float]):
        """
        Args:
            quantiles: List of quantiles to predict, e.g. [0.1, 0.5, 0.9]
        """
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles))

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Handle NaN values by creating a mask
        mask = ~torch.isnan(targets)
        if not mask.any():
            return torch.tensor(0.0, device=predictions.device)

        # Filter out NaN values
        valid_predictions = predictions[mask]
        valid_targets = targets[mask]

        losses = []
        for q in self.quantiles:
            diff = valid_targets - valid_predictions
            loss = torch.max(q * diff, (q - 1) * diff)
            losses.append(loss.mean())

        return torch.stack(losses).sum()


class WeightedDistLoss(nn.Module):
    def __init__(self, num_bins=64, bandwidth=0.5, weights=None, eps=1e-7):
        """
        Weighted NaN-tolerant implementation of Dist Loss that inherently combines
        distribution alignment and prediction errors through MSE between pseudo-labels
        and pseudo-predictions.

        Args:
            num_bins: Number of bins for KDE
            bandwidth: Kernel bandwidth for KDE
            weights: Optional tensor of weights for each dimension
            eps: Small value for numerical stability
        """
        super().__init__()
        self.num_bins = num_bins
        self.bandwidth = bandwidth
        self.eps = eps

        # Register weights as a buffer
        if weights is None:
            weights = torch.ones(2)  # Default to 2 dimensions with equal weights
        self.register_buffer("weights", weights / weights.sum())

    def kde(self, x: torch.Tensor, eval_points: torch.Tensor) -> torch.Tensor:
        """
        Kernel Density Estimation with Gaussian kernel.
        Handles NaN values by excluding them from density estimation.
        """
        # Remove NaN values
        x = x[~torch.isnan(x)]
        if x.size(0) == 0:
            return torch.zeros_like(eval_points)

        # Reshape for broadcasting
        x = x.view(-1, 1)
        eval_points = eval_points.view(1, -1)

        # Compute Gaussian kernel
        kernel = torch.exp(-0.5 * ((x - eval_points) / self.bandwidth) ** 2)
        kernel = kernel.mean(dim=0)  # Average across samples

        return kernel / (kernel.sum() + self.eps)  # Normalize

    def generate_pseudo_labels(
        self, y_true: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """
        Generate pseudo-labels using KDE for current batch.

        Args:
            y_true: Ground truth values with possible NaN entries
            batch_size: Number of pseudo-labels to generate

        Returns:
            Pseudo-labels sampled from estimated distribution
        """
        # Define evaluation points based on batch statistics
        valid_vals = y_true[~torch.isnan(y_true)]
        if valid_vals.size(0) == 0:
            return torch.zeros(batch_size, device=y_true.device)

        min_val = valid_vals.min()
        max_val = valid_vals.max()
        eval_points = torch.linspace(
            min_val, max_val, self.num_bins, device=y_true.device
        )

        # Estimate density using batch data
        density = self.kde(y_true, eval_points)

        # Generate cumulative distribution
        cdf = torch.cumsum(density, 0)
        cdf = cdf / (cdf[-1] + self.eps)

        # Generate uniform samples for current batch
        u = torch.linspace(0, 1, batch_size, device=y_true.device)

        # Find bins using binary search
        inds = torch.searchsorted(cdf, u)
        inds = torch.clamp(inds, 0, self.num_bins - 1)

        # Linear interpolation between bin edges
        pseudo_labels = eval_points[inds]

        return pseudo_labels

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = y_pred.device
        y_true = y_true.to(device)

        # Create mask for non-NaN values
        mask = ~torch.isnan(y_true)

        # Check which dimensions have ANY valid samples
        valid_dims = mask.any(dim=0)  # [num_dims]

        # If no dimensions are valid, return zero loss
        if not valid_dims.any():
            return torch.tensor(0.0, device=device), torch.zeros_like(
                valid_dims, dtype=torch.float
            )

        batch_size, num_dims = y_true.shape
        dim_losses = torch.zeros(num_dims, device=device)

        # Zero out weights for completely NaN dimensions
        weights = self.weights * valid_dims
        weight_sum = weights.sum().clamp(min=1e-8)

        # Process each dimension separately
        for dim in range(num_dims):
            if not valid_dims[dim]:
                continue

            # Get values for current dimension
            true_dim = y_true[:, dim]
            pred_dim = y_pred[:, dim]

            # Generate pseudo-labels and sort predictions
            pseudo_labels = self.generate_pseudo_labels(true_dim, batch_size)
            pseudo_preds = torch.sort(pred_dim[~torch.isnan(pred_dim)])[0]

            # Pad predictions if needed
            if pseudo_preds.size(0) < batch_size:
                padding = torch.zeros(batch_size - pseudo_preds.size(0), device=device)
                pseudo_preds = torch.cat([pseudo_preds, padding])

            # Compute MSE between distributions (inherently combines alignment and prediction errors)
            dim_losses[dim] = F.mse_loss(pseudo_preds, pseudo_labels)

        # Compute weighted average loss
        weighted_loss = (dim_losses * weights).sum() / weight_sum

        return weighted_loss, dim_losses


class CombinedRegressionLoss(nn.Module):
    def __init__(
        self, loss_type="mse", weights=None, quantile_spacing=None, huber_delta=1.0
    ):
        super(CombinedRegressionLoss, self).__init__()
        self.loss_type = loss_type

        if loss_type == "mse":
            self.loss_fn = NaNTolerantMSELoss()
        elif loss_type == "l1":
            self.loss_fn = NaNTolerantL1Loss()
        elif loss_type == "logcosh":
            self.loss_fn = NaNTolerantLogCoshLoss()
        elif loss_type == "huber":
            self.loss_fn = NaNTolerantHuberLoss(delta=huber_delta)
        elif loss_type == "quantile":
            if quantile_spacing is None:
                raise ValueError("quantile_spacing must be provided for quantile loss")
            quantiles = torch.arange(quantile_spacing, 1.0, quantile_spacing).tolist()
            self.loss_fn = NaNTolerantQuantileLoss(quantiles=quantiles)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        # Register weights as a buffer so it's automatically moved with the module
        if weights is None:
            weights = torch.ones(2)
        self.register_buffer("weights", weights / weights.sum())

    def forward(self, y_pred, y_true):
        """
        Compute weighted loss while handling NaN values.
        """
        device = y_pred.device
        y_true = y_true.to(device)

        # Create mask for valid (non-NaN) values
        mask = ~torch.isnan(y_true)
        valid_dims = mask.any(dim=0)

        # Adjust weights
        weights = self.weights * valid_dims
        weight_sum = weights.sum().clamp(min=1e-8)

        # Initialize list to store per-dimension losses
        dim_losses = []

        # Compute loss for each dimension
        for dim in range(y_true.shape[1]):
            # Get mask for this dimension
            dim_mask = mask[:, dim]

            if not dim_mask.any():
                # Create zero tensor with consistent shape
                dim_losses.append(torch.tensor([0.0], device=device))
                continue

            # Get valid values for this dimension
            valid_preds = y_pred[dim_mask, dim]
            valid_targets = y_true[dim_mask, dim]

            # Compute loss based on type
            if self.loss_type == "quantile":
                valid_preds = valid_preds.unsqueeze(-1)
                valid_targets = valid_targets.unsqueeze(-1)
                dim_loss = self.loss_fn(valid_preds, valid_targets)
                # Ensure consistent shape
                dim_loss = dim_loss.reshape(1)
            else:
                valid_preds = valid_preds.unsqueeze(-1)
                valid_targets = valid_targets.unsqueeze(-1)
                dim_loss = self.loss_fn(valid_preds, valid_targets)[0].squeeze()
                # Ensure consistent shape
                dim_loss = dim_loss.reshape(1)

            dim_losses.append(dim_loss)

        # Stack dimension losses - now all tensors should have shape [1]
        dim_losses = torch.stack(dim_losses)

        # Compute weighted average loss
        weighted_loss = (dim_losses * weights).sum() / weight_sum

        return weighted_loss, dim_losses.squeeze()


#############


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


class MonotonicParameter(nn.Parameter):
    """Parameter that ensures values are monotonically increasing via cumulative softplus."""

    def __new__(cls, data=None, requires_grad=True):
        return super(MonotonicParameter, cls).__new__(cls, data, requires_grad)

    @property
    def data(self):
        # Apply softplus to ensure positive deltas, then cumsum for monotonicity
        return torch.cumsum(F.softplus(super().data), dim=-1)


class MultiDimNaNTolerantOrdinalCELoss(nn.Module):
    def __init__(self, num_classes: int, num_tasks: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_tasks

        # Initialize raw parameters with small random values
        raw_deltas = torch.randn(num_tasks, num_classes - 1) * 0.01
        self.raw_thresholds = nn.Parameter(raw_deltas)

    @property
    def thresholds(self):
        # Transform raw parameters into monotonic thresholds
        return torch.cumsum(F.softplus(self.raw_thresholds), dim=-1)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ordinal cross entropy loss with NaN handling.

        Args:
            logits: Raw model outputs [batch_size, num_tasks * (num_classes - 1)]
            targets: Target one-hot encodings [batch_size, num_tasks * num_classes]
        """
        batch_size = logits.shape[0]
        device = logits.device

        # Calculate sizes
        target_entries_per_task = targets.shape[1] // self.num_tasks  # num_classes
        logit_entries_per_task = logits.shape[1] // self.num_tasks  # num_classes - 1

        # Initialize losses and validity mask
        dim_losses = torch.zeros(self.num_tasks, device=device)
        validity_mask = torch.zeros(batch_size, self.num_tasks, device=device)

        # Get monotonic thresholds
        monotonic_thresholds = self.thresholds

        # Process each task separately
        for task in range(self.num_tasks):
            # Get task-specific slices
            target_start = task * target_entries_per_task
            target_end = (task + 1) * target_entries_per_task
            task_targets = targets[
                :, target_start:target_end
            ]  # [batch_size, num_classes]

            logit_start = task * logit_entries_per_task
            logit_end = (task + 1) * logit_entries_per_task
            task_logits = logits[
                :, logit_start:logit_end
            ]  # [batch_size, num_classes-1]

            # Create validity mask
            task_valid_mask = ~torch.isnan(task_targets).any(dim=1)
            validity_mask[:, task] = task_valid_mask

            if not task_valid_mask.any():
                continue

            # Get valid samples
            valid_logits = task_logits[task_valid_mask]
            valid_targets = task_targets[task_valid_mask]

            # Convert one-hot to indices
            target_indices = torch.argmax(valid_targets, dim=1)

            if valid_logits.size(0) == 0:
                continue

            # Create binary targets for each threshold
            binary_targets = torch.zeros(
                valid_logits.size(0), self.num_classes - 1, device=device
            )
            for i in range(self.num_classes - 1):
                binary_targets[:, i] = (target_indices > i).float()

            # Get task-specific thresholds
            thresholds = monotonic_thresholds[task]

            # Compute ordinal probabilities using sigmoid
            probs = torch.sigmoid(valid_logits - thresholds)

            # Compute binary cross entropy loss
            bce = F.binary_cross_entropy(probs, binary_targets, reduction="none")

            # Average over thresholds and samples
            task_loss = bce.mean()
            dim_losses[task] = task_loss

        return dim_losses, validity_mask.any(dim=0)


class CombinedOrdinalCELoss(nn.Module):
    """Combined ordinal classification loss with dimension-wise weighting."""

    def __init__(
        self,
        num_classes: int,
        num_tasks: int = 2,
        weights: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            num_classes: Number of ordinal classes per task
            num_tasks: Number of tasks
            weights: Optional tensor of weights for each dimension [num_tasks]
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.loss_fn = MultiDimNaNTolerantOrdinalCELoss(
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
        Compute weighted ordinal classification loss.

        Args:
            logits: Model predictions [batch_size, num_tasks * num_classes]
            targets: Target labels [batch_size, num_tasks * num_classes]

        Returns:
            tuple: (total_loss, dim_losses)
        """
        device = logits.device

        # Compute per-dimension losses and get validity mask
        dim_losses, mask = self.loss_fn(logits, targets)

        # Apply weights only to valid dimensions
        weights = self.weights.to(device)
        weights = weights * mask.float()
        weight_sum = weights.sum().clamp(min=1e-8)

        # Compute weighted average loss
        dim_losses = weights * dim_losses
        total_loss = dim_losses.sum() / weight_sum

        return total_loss.to(device), dim_losses.to(device)


class CategoricalEntropyRegLoss(nn.Module):
    """Entropy regularization loss that works with both soft and one-hot categorical targets."""

    def __init__(
        self, lambda_d: float = 0.1, lambda_t: float = 0.1, num_classes: int = 32
    ):
        super().__init__()
        self.lambda_d = lambda_d
        self.lambda_t = lambda_t
        self.num_classes = num_classes

    def compute_pairwise_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pairwise L2 distances between all points."""
        diff = x.unsqueeze(1) - x.unsqueeze(0)  # [N, N, D]
        return (diff**2).sum(dim=-1)  # [N, N]

    def compute_target_distances(self, targets: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence distances between probability distributions."""
        # Add small epsilon to avoid numerical issues
        eps = 1e-10
        num_dims = targets.shape[1] // self.num_classes
        target_dists = []

        for i in range(num_dims):
            start_idx = i * self.num_classes
            end_idx = (i + 1) * self.num_classes
            # Get probabilities for this dimension
            probs = targets[:, start_idx:end_idx]

            # For categorical (one-hot), this just ensures sum is 1
            # For soft labels, this normalizes the probabilities
            probs = probs + eps
            probs = probs / probs.sum(dim=1, keepdim=True)

            # Compute symmetric KL divergence between all pairs
            kl_div = torch.zeros(
                (probs.shape[0], probs.shape[0]), device=targets.device
            )
            for j in range(probs.shape[0]):
                p = probs[j : j + 1]  # [1, num_classes]
                q = probs  # [N, num_classes]
                # Symmetric KL divergence
                kl_pq = (p * (torch.log(p) - torch.log(q))).sum(dim=1)
                kl_qp = (q * (torch.log(q) - torch.log(p.expand_as(q)))).sum(dim=1)
                kl_div[j] = (kl_pq + kl_qp) / 2

            target_dists.append(kl_div)

        # Average across dimensions
        return torch.stack(target_dists).mean(dim=0)

    def compute_centers(self, features: torch.Tensor, targets: torch.Tensor) -> dict:
        """Compute weighted feature centers for each class in each dimension."""
        centers = {}
        num_dims = targets.shape[1] // self.num_classes

        for dim in range(num_dims):
            centers[dim] = {}
            start_idx = dim * self.num_classes
            end_idx = (dim + 1) * self.num_classes

            # Get probabilities for this dimension
            probs = targets[:, start_idx:end_idx]  # [N, num_classes]

            # For each class, compute weighted center
            for class_idx in range(self.num_classes):
                # Use class probabilities as weights
                weights = probs[:, class_idx]  # [N]

                # Only compute center if we have non-zero weights
                if weights.sum() > 0:
                    # Normalize weights
                    weights = weights / weights.sum()
                    # Compute weighted center
                    center = (features * weights.unsqueeze(1)).sum(dim=0)
                    centers[dim][class_idx] = center

        return centers

    def forward(
        self, features: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass computing diversity and tightness losses.

        Args:
            features: Node features [batch_size, feature_dim]
            targets: Probability distributions [batch_size, num_dims * num_classes]
            mask: Valid sample mask [batch_size]
        """
        device = features.device
        features, targets = features.to(device), targets.to(device)

        # Normalize features
        features_norm = F.normalize(features, p=2, dim=-1)

        # Get valid samples
        valid_features = features_norm[mask]
        valid_targets = targets[mask]

        # Compute diversity loss (Ld)
        if valid_features.size(0) > 1:
            feat_dists = self.compute_pairwise_distances(valid_features)
            target_dists = self.compute_target_distances(valid_targets)

            M = valid_features.size(0)
            diag_mask = ~torch.eye(M, dtype=torch.bool, device=device)

            diversity_loss = -(feat_dists * target_dists * diag_mask).sum() / (
                M * (M - 1)
            )
        else:
            diversity_loss = torch.tensor(0.0, device=device)

        # Compute tightness loss (Lt)
        centers = self.compute_centers(valid_features, valid_targets)
        tightness_losses = []
        num_dims = valid_targets.shape[1] // self.num_classes

        # Compute weighted tightness loss for each sample
        for dim in range(num_dims):
            start_idx = dim * self.num_classes
            end_idx = (dim + 1) * self.num_classes
            probs = valid_targets[:, start_idx:end_idx]  # [N, num_classes]

            for i in range(valid_features.size(0)):
                sample_losses = []
                # Weight the distance to each center by the sample's probability for that class
                for class_idx in range(self.num_classes):
                    if class_idx in centers[dim]:
                        diff = valid_features[i] - centers[dim][class_idx]
                        class_prob = probs[i, class_idx]
                        if class_prob > 0:
                            sample_losses.append(class_prob * (diff**2).sum())

                if sample_losses:
                    tightness_losses.append(torch.stack(sample_losses).sum())

        if tightness_losses:
            tightness_loss = torch.stack(tightness_losses).mean()
        else:
            tightness_loss = torch.tensor(0.0, device=device)

        # Compute total loss
        total_loss = self.lambda_d * diversity_loss + self.lambda_t * tightness_loss

        return total_loss, diversity_loss, tightness_loss


class MseCategoricalEntropyRegLoss(nn.Module):
    """
    Combines MSE loss with entropy regularization, tracking per-dimension losses.
    """

    def __init__(
        self,
        num_classes: int,
        num_tasks: int = 2,
        weights: Optional[torch.Tensor] = None,
        lambda_d: float = 0.1,
        lambda_t: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_tasks

        # Register weights as buffer
        if weights is None:
            weights = torch.ones(num_tasks)
        self.register_buffer("weights", weights / weights.sum())

        # Initialize MSE and entropy components
        self.mse_loss = NaNTolerantMSELoss()
        self.entropy_reg = CategoricalEntropyRegLoss(
            lambda_d=lambda_d, lambda_t=lambda_t, num_classes=num_classes
        )

        self.lambda_d = lambda_d
        self.lambda_t = lambda_t

    def forward(
        self,
        continuous_pred: torch.Tensor,
        continuous_target: torch.Tensor,
        logits: torch.Tensor,
        categorical_target: torch.Tensor,
        pooled_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass tracking all loss components.
        Returns total loss and dictionary of component losses.
        """
        device = continuous_pred.device

        # Get validity mask
        mask = ~torch.isnan(continuous_target).any(dim=1)

        # Calculate MSE loss with dimension tracking
        mse_losses, valid_mask = self.mse_loss(continuous_pred, continuous_target)

        weights = self.weights.to(device)

        # Apply weights only to valid dimensions
        weights = weights * valid_mask.float()
        weight_sum = weights.sum().clamp(min=1e-8)

        # Keep per-dimension MSE losses and compute weighted total
        dim_losses = weights * mse_losses
        mse_total = dim_losses.sum() / weight_sum

        # Calculate entropy regularization losses
        entropy_total, diversity_loss, tightness_loss = self.entropy_reg(
            pooled_features, categorical_target, mask
        )

        # Ensure all components have gradients
        total_loss = mse_total + entropy_total

        # Return total loss and all components for logging
        loss_components = {
            "total_loss": total_loss,
            "mse_total": mse_total,
            "dim_losses": dim_losses,
            "entropy_total": entropy_total,
            "diversity_loss": diversity_loss,
            "tightness_loss": tightness_loss,
        }

        return total_loss, loss_components


# if __name__ == "__main__":
#     # Test case
#     logits = torch.tensor(
#         [
#             [0.6370, -0.0156, 0.2711, -0.2273],
#             [-0.2947, -0.1466, 1.2458, -0.9816],
#             [-0.4593, -0.2630, 1.2785, -0.8181],
#             [-0.2947, -0.1466, 1.2459, -0.9816],
#         ]
#     )
#     y = torch.tensor(
#         [
#             [0.0, 1.0, 0.0, 1.0],
#             [0.0, 1.0, float("nan"), float("nan")],
#             [0.0, 1.0, float("nan"), float("nan")],
#             [0.0, 1.0, 0.0, 1.0],
#         ]
#     )
#     loss = CombinedCELoss(num_classes=2, num_tasks=2)
#     total_loss, dim_losses = loss(logits, y)
#     print(f"Total loss: {total_loss}")
#     print(f"Dimension losses: {dim_losses}")

# if __name__ == "__main__":
#     torch.manual_seed(42)

#     print("Testing Ordinal Loss with Monotonic Thresholds:")
#     for num_classes in [2, 4, 8]:
#         print(f"\nTesting with {num_classes} classes:")

#         batch_size = 4
#         num_tasks = 2

#         # Create test data
#         logits = torch.randn(batch_size, num_tasks * (num_classes - 1)) * 0.5
#         targets = torch.zeros(batch_size, num_tasks * num_classes)

#         # Create ordered targets
#         for i in range(batch_size):
#             for task in range(num_tasks):
#                 class_idx = min(i, num_classes - 1)
#                 start_idx = task * num_classes
#                 targets[i, start_idx + class_idx] = 1.0

#         # Add NaN values
#         targets[1, num_classes:] = float("nan")
#         targets[2, num_classes:] = float("nan")

#         # Test loss
#         loss_fn = CombinedOrdinalCELoss(num_classes=num_classes, num_tasks=num_tasks)
#         total_loss, dim_losses = loss_fn(logits, targets)

#         print(f"\nLoss values:")
#         print(f"Total loss: {total_loss:.4f}")
#         print(f"Dimension losses: {dim_losses}")

#         # Verify monotonicity
#         params = list(loss_fn.parameters())
#         monotonic_thresholds = loss_fn.loss_fn.thresholds
#         raw_thresholds = loss_fn.loss_fn.raw_thresholds

#         print("\nThreshold Information:")
#         print(f"Raw parameters:\n{raw_thresholds}")
#         print(f"Monotonic thresholds:\n{monotonic_thresholds}")

#         # Verify monotonicity explicitly
#         diffs = monotonic_thresholds[:, 1:] - monotonic_thresholds[:, :-1]
#         print(f"Consecutive differences:\n{diffs}")
#         print(f"All thresholds monotonically increasing: {torch.all(diffs > 0).item()}")

# if __name__ == "__main__":
#     torch.manual_seed(42)
#     print("\nTesting LogCosh Loss:")

#     # Create test data
#     batch_size = 4
#     num_tasks = 2
#     y_pred = torch.tensor([[1.0, 2.0], [0.5, 1.5], [2.0, 3.0], [1.5, 2.5]])

#     y_true = torch.tensor(
#         [
#             [1.2, 2.1],
#             [0.4, float("nan")],  # Add NaN to test NaN handling
#             [float("nan"), 3.2],
#             [1.4, 2.3],
#         ]
#     )

#     # Initialize loss function
#     loss_fn = CombinedRegressionLoss(loss_type="logcosh")

#     # Compute loss
#     total_loss, dim_losses = loss_fn(y_pred, y_true)

#     print(f"\nInput tensors:")
#     print(f"Predictions:\n{y_pred}")
#     print(f"Targets:\n{y_true}")
#     print(f"\nLoss values:")
#     print(f"Total loss: {total_loss:.4f}")
#     print(f"Dimension losses: {dim_losses}")

#     # Test with custom weights
#     weights = torch.tensor([0.7, 0.3])
#     weighted_loss_fn = CombinedRegressionLoss(loss_type="logcosh", weights=weights)
#     weighted_total_loss, weighted_dim_losses = weighted_loss_fn(y_pred, y_true)

#     print(f"\nWeighted loss values (weights={weights}):")
#     print(f"Total weighted loss: {weighted_total_loss:.4f}")
#     print(f"Weighted dimension losses: {weighted_dim_losses}")

# if __name__ == "__main__":
#     torch.manual_seed(42)
#     print("\nTesting WeightedSupCRLoss:")

#     # Create test data
#     batch_size = 4
#     embedding_dim = 8
#     num_dims = 2

#     # Generate random embeddings
#     perturbed_embeddings = torch.randn(batch_size, embedding_dim)
#     intact_embeddings = torch.randn(batch_size, embedding_dim)

#     # Create labels with some NaN values
#     labels = torch.tensor(
#         [[1.2, 2.1], [0.4, float("nan")], [float("nan"), 3.2], [1.4, 2.3]]
#     )

#     # Test unweighted loss
#     loss_fn = SupCR()
#     total_loss, dim_losses = loss_fn(perturbed_embeddings, intact_embeddings, labels)

#     print(f"\nInput shapes:")
#     print(f"Perturbed embeddings: {perturbed_embeddings.shape}")
#     print(f"Intact embeddings: {intact_embeddings.shape}")
#     print(f"Labels: {labels.shape}")
#     print(f"\nLabels:\n{labels}")
#     print(f"\nLoss values:")
#     print(f"Total loss: {total_loss:.4f}")
#     print(f"Dimension losses: {dim_losses}")

#     # Test with custom weights
#     weights = torch.tensor([0.7, 0.3])
#     weighted_loss_fn = WeightedSupCRLoss(weights=weights)
#     weighted_total_loss, weighted_dim_losses = weighted_loss_fn(
#         perturbed_embeddings, intact_embeddings, labels
#     )

#     print(f"\nWeighted loss values (weights={weights}):")
#     print(f"Total weighted loss: {weighted_total_loss:.4f}")
#     print(f"Weighted dimension losses: {weighted_dim_losses}")

#     # Test with all NaN dimension
#     labels_with_nan_dim = torch.tensor(
#         [
#             [1.2, float("nan")],
#             [0.4, float("nan")],
#             [1.8, float("nan")],
#             [1.4, float("nan")],
#         ]
#     )

#     nan_loss, nan_dim_losses = weighted_loss_fn(
#         perturbed_embeddings, intact_embeddings, labels_with_nan_dim
#     )

#     print(f"\nTesting with NaN dimension:")
#     print(f"Labels:\n{labels_with_nan_dim}")
#     print(f"Total loss: {nan_loss:.4f}")
#     print(f"Dimension losses: {nan_dim_losses}")

# if __name__ == "__main__":
#     import torch

#     # Test parameters
#     batch_size = 4
#     embedding_dim = 128
#     num_targets = 2  # Two target dimensions
#     temperature = 0.1

#     # Create test embeddings
#     perturbed_embeddings = torch.randn(batch_size, embedding_dim)
#     intact_embeddings = torch.randn(batch_size, embedding_dim)

#     # Create labels with some NaN values
#     labels = torch.tensor(
#         [[1.2, 2.1], [0.4, float("nan")], [float("nan"), 3.2], [1.4, 2.3]]
#     )

#     # Test SupCRCell loss
#     weights = torch.tensor([1.0, 1.0])  # Equal weights for both dimensions
#     loss_fn = WeightedSupCRCell(weights=weights, temperature=temperature)

#     total_loss, dim_losses = loss_fn(
#         perturbed_embeddings=perturbed_embeddings,
#         intact_embeddings=intact_embeddings,
#         labels=labels,
#     )

#     print("\n=== SupCRCell Loss Test ===")
#     print(f"Input shapes:")
#     print(f"Perturbed embeddings: {perturbed_embeddings.shape}")
#     print(f"Intact embeddings: {intact_embeddings.shape}")
#     print(f"Labels: {labels.shape}")
#     print(f"\nLabels:\n{labels}")

#     print(f"\nLoss values:")
#     print(f"Total loss: {total_loss:.4f}")
#     print(f"Dimension losses: {dim_losses}")

#     # Test with all NaN dimension
#     labels_with_nan_dim = torch.tensor(
#         [
#             [1.2, float("nan")],
#             [0.4, float("nan")],
#             [1.8, float("nan")],
#             [1.4, float("nan")],
#         ]
#     )

#     nan_loss, nan_dim_losses = loss_fn(
#         perturbed_embeddings=perturbed_embeddings,
#         intact_embeddings=intact_embeddings,
#         labels=labels_with_nan_dim,
#     )

#     print(f"\n=== Testing with NaN dimension ===")
#     print(f"Labels:\n{labels_with_nan_dim}")
#     print(f"Total loss: {nan_loss:.4f}")
#     print(f"Dimension losses: {nan_dim_losses}")

#     # Verify gradients
#     perturbed_embeddings.requires_grad_(True)
#     intact_embeddings.requires_grad_(True)

#     loss, _ = loss_fn(perturbed_embeddings, intact_embeddings, labels)
#     loss.backward()

#     print(f"\n=== Gradient Check ===")
#     print(f"Perturbed gradients exist: {perturbed_embeddings.grad is not None}")
#     print(f"Intact gradients exist: {intact_embeddings.grad is not None}")


if __name__ == "__main__":
    import torch

    # Test parameters
    batch_size = 4
    num_targets = 2  # Two target dimensions
    predictions = torch.randn(batch_size, num_targets)  # Random predictions

    # Create labels with some NaN values
    labels = torch.tensor(
        [[1.2, 2.1], [0.4, float("nan")], [float("nan"), 3.2], [1.4, 2.3]]
    )

    print("\n=== WeightedMSELoss Test ===")
    print(f"Input shapes:")
    print(f"Predictions: {predictions.shape}")
    print(f"Labels: {labels.shape}")
    print(f"\nLabels:\n{labels}")

    # Test with equal weights
    loss_fn = WeightedMSELoss()
    total_loss, dim_losses = loss_fn(predictions, labels)

    print(f"\nLoss values (equal weights):")
    print(f"Total loss: {total_loss:.4f}")
    print(f"Dimension losses: {dim_losses}")

    # Test with custom weights
    weights = torch.tensor([0.7, 0.3])  # Uneven weights for dimensions
    weighted_loss_fn = WeightedMSELoss(weights=weights)
    weighted_total, weighted_dims = weighted_loss_fn(predictions, labels)

    print(f"\nLoss values (weights=[0.7, 0.3]):")
    print(f"Total loss: {weighted_total:.4f}")
    print(f"Dimension losses: {weighted_dims}")

    # Test with all NaN dimension
    labels_with_nan_dim = torch.tensor(
        [
            [1.2, float("nan")],
            [0.4, float("nan")],
            [1.8, float("nan")],
            [1.4, float("nan")],
        ]
    )

    nan_loss, nan_dim_losses = weighted_loss_fn(predictions, labels_with_nan_dim)

    print(f"\n=== Testing with NaN dimension ===")
    print(f"Labels:\n{labels_with_nan_dim}")
    print(f"Total loss: {nan_loss:.4f}")
    print(f"Dimension losses: {nan_dim_losses}")

    # Verify gradients
    predictions.requires_grad_(True)
    loss, _ = weighted_loss_fn(predictions, labels)
    loss.backward()

    print(f"\n=== Gradient Check ===")
    print(f"Gradients exist: {predictions.grad is not None}")
    print(f"Gradient shape: {predictions.grad.shape}")
