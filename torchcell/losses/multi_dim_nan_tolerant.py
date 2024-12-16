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
    """
    Entropy regularization loss for categorical targets that encourages feature diversity
    while maintaining categorical structure through tightness.

    Works with both hard categorical (one-hot) and soft categorical (probability distribution) targets.
    """

    def __init__(
        self, lambda_d: float = 0.1, lambda_t: float = 0.1, num_classes: int = 32
    ):
        """
        Args:
            lambda_d: Weight for diversity loss term (Ld)
            lambda_t: Weight for tightness loss term (Lt)
            num_classes: Number of classes per target dimension (e.g. num_bins)
        """
        super().__init__()
        self.lambda_d = lambda_d
        self.lambda_t = lambda_t
        self.num_classes = num_classes

    def compute_pairwise_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pairwise L2 distances between all points."""
        diff = x.unsqueeze(1) - x.unsqueeze(0)  # [N, N, D]
        return (diff**2).sum(dim=-1)  # [N, N]

    def compute_target_distances(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute distances between categorical targets.
        Works with both one-hot and soft targets.
        """
        # For each target dimension, compute KL divergence
        target_dists = []
        num_dims = targets.shape[1] // self.num_classes

        for i in range(num_dims):
            start_idx = i * self.num_classes
            end_idx = (i + 1) * self.num_classes
            # Get probabilities for this dimension
            probs = targets[:, start_idx:end_idx]
            # Add small epsilon to avoid log(0)
            probs = probs + 1e-10
            probs = probs / probs.sum(dim=1, keepdim=True)

            # Compute KL divergence between all pairs
            kl_div = torch.zeros(
                (probs.shape[0], probs.shape[0]), device=targets.device
            )
            for j in range(probs.shape[0]):
                p = probs[j : j + 1]  # [1, num_classes]
                q = probs  # [N, num_classes]
                kl = (p * (torch.log(p) - torch.log(q))).sum(dim=1)
                kl_div[j] = kl

            target_dists.append(kl_div)

        # Average across dimensions
        return torch.stack(target_dists).mean(dim=0)

    def compute_centers(self, features: torch.Tensor, targets: torch.Tensor):
        """Compute feature centers for categorical targets."""
        centers = {}
        num_dims = targets.shape[1] // self.num_classes

        # Get class indices for each dimension
        class_indices = []
        for i in range(num_dims):
            start_idx = i * self.num_classes
            end_idx = (i + 1) * self.num_classes
            probs = targets[:, start_idx:end_idx]
            # For soft targets, take argmax
            class_idx = torch.argmax(probs, dim=1)
            class_indices.append(class_idx)

        class_indices = torch.stack(class_indices, dim=1)  # [N, num_dims]

        # Compute centers for each unique combination of classes
        for i in range(features.size(0)):
            target_key = tuple(class_indices[i].cpu().numpy())
            if target_key not in centers:
                # Find features with same class combination
                same_target = torch.all(class_indices == class_indices[i], dim=1)
                if same_target.sum() > 0:
                    centers[target_key] = features[same_target].mean(dim=0)

        return centers

    def forward(
        self, features: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ):
        """
        Args:
            features: Node features before prediction head [batch_size, feature_dim]
            targets: Categorical target values [batch_size, num_dims * num_classes]
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

        for i in range(valid_features.size(0)):
            target_key = tuple(
                torch.argmax(valid_targets[i].view(-1, self.num_classes), dim=1)
                .cpu()
                .numpy()
            )

            if target_key in centers:
                diff = valid_features[i] - centers[target_key]
                tightness_losses.append((diff**2).sum())

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
        self.mse_loss = MultiDimNaNTolerantMSELoss()
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

if __name__ == "__main__":
    torch.manual_seed(42)

    print("Testing Ordinal Loss with Monotonic Thresholds:")
    for num_classes in [2, 4, 8]:
        print(f"\nTesting with {num_classes} classes:")

        batch_size = 4
        num_tasks = 2

        # Create test data
        logits = torch.randn(batch_size, num_tasks * (num_classes - 1)) * 0.5
        targets = torch.zeros(batch_size, num_tasks * num_classes)

        # Create ordered targets
        for i in range(batch_size):
            for task in range(num_tasks):
                class_idx = min(i, num_classes - 1)
                start_idx = task * num_classes
                targets[i, start_idx + class_idx] = 1.0

        # Add NaN values
        targets[1, num_classes:] = float("nan")
        targets[2, num_classes:] = float("nan")

        # Test loss
        loss_fn = CombinedOrdinalCELoss(num_classes=num_classes, num_tasks=num_tasks)
        total_loss, dim_losses = loss_fn(logits, targets)

        print(f"\nLoss values:")
        print(f"Total loss: {total_loss:.4f}")
        print(f"Dimension losses: {dim_losses}")

        # Verify monotonicity
        params = list(loss_fn.parameters())
        monotonic_thresholds = loss_fn.loss_fn.thresholds
        raw_thresholds = loss_fn.loss_fn.raw_thresholds

        print("\nThreshold Information:")
        print(f"Raw parameters:\n{raw_thresholds}")
        print(f"Monotonic thresholds:\n{monotonic_thresholds}")

        # Verify monotonicity explicitly
        diffs = monotonic_thresholds[:, 1:] - monotonic_thresholds[:, :-1]
        print(f"Consecutive differences:\n{diffs}")
        print(f"All thresholds monotonically increasing: {torch.all(diffs > 0).item()}")
