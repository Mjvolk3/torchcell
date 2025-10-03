# torchcell/losses/mle_wasserstein
# [[torchcell.losses.mle_wass_supcr]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/losses/mle_wasserstein
# Test file: tests/torchcell/losses/test_mle_wass_supcr.py
# Note: File name kept as mle_wasserstein.py for compatibility, but class is MleWassSupCR


import math
from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.distributed as dist

from geomloss import SamplesLoss
from torchcell.losses.multi_dim_nan_tolerant import (
    WeightedMSELoss,
    WeightedSupCRCell,
)


class AdaptiveWeighting:
    """Manages adaptive weighting for buffer contributions during training."""

    def __init__(self, warmup_epochs: int = 100, stable_epoch: int = 500):
        self.warmup_epochs = warmup_epochs
        self.stable_epoch = stable_epoch

    def get_buffer_weight(self, epoch: int) -> float:
        """Get buffer weight based on training epoch."""
        if epoch < self.warmup_epochs:
            # Linear ramp from 0.1 to 0.3
            return 0.1 + 0.2 * (epoch / self.warmup_epochs)
        elif epoch < self.stable_epoch:
            # Sigmoid transition from 0.3 to 0.9
            progress = (epoch - self.warmup_epochs) / (
                self.stable_epoch - self.warmup_epochs
            )
            return 0.3 + 0.6 / (1 + math.exp(-10 * (progress - 0.5)))
        else:
            # Stable at 0.9
            return 0.9


class TemperatureScheduler:
    """Manages temperature scheduling for SupCR loss."""

    def __init__(
        self,
        init_temp: float = 1.0,
        final_temp: float = 0.1,
        schedule: str = "exponential",
    ):
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.schedule = schedule

    def get_temperature(self, epoch: int, max_epochs: int = 1000) -> float:
        """Get temperature based on training epoch."""
        if self.schedule == "exponential":
            return self.init_temp * (self.final_temp / self.init_temp) ** (
                epoch / max_epochs
            )
        elif self.schedule == "cosine":
            return self.final_temp + 0.5 * (self.init_temp - self.final_temp) * (
                1 + math.cos(math.pi * epoch / max_epochs)
            )
        else:
            return self.init_temp


class WeightedWassersteinLoss(nn.Module):
    """Weighted Wasserstein distance loss for multi-dimensional outputs."""

    def __init__(
        self,
        blur: float = 0.05,
        p: float = 2,
        scaling: float = 0.9,
        weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize Weighted Wasserstein Loss.

        Args:
            blur: Smoothing parameter for entropic regularization
            p: Power for the ground cost (p=1: W1, p=2: W2)
            scaling: Scaling parameter for multi-scale approach (0.5 < scaling < 1)
            weights: Optional weights for different dimensions
        """
        super().__init__()
        self.blur = blur
        self.p = p
        self.scaling = scaling

        # Initialize SamplesLoss for Wasserstein distance
        self.wasserstein = SamplesLoss(
            loss="sinkhorn",
            p=p,
            blur=blur,
            scaling=scaling,
            debias=True,  # Remove bias for better accuracy
        )

        if weights is not None:
            self.register_buffer("weights", weights)
        else:
            self.weights = None

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute weighted Wasserstein distance.

        Args:
            predictions: Predicted values [batch_size, num_dims]
            targets: Target values [batch_size, num_dims]

        Returns:
            Tuple of (total_loss, per_dimension_losses)
        """
        batch_size, num_dims = predictions.shape
        device = predictions.device

        # Handle NaN values
        valid_mask = ~torch.isnan(targets)

        # Compute per-dimension Wasserstein distances
        dim_losses = []

        for dim in range(num_dims):
            dim_valid = valid_mask[:, dim]
            if dim_valid.sum() > 1:  # Need at least 2 samples
                # Extract valid predictions and targets for this dimension
                pred_dim = predictions[dim_valid, dim:dim+1]  # Keep 2D shape
                targ_dim = targets[dim_valid, dim:dim+1]

                # Compute Wasserstein distance for this dimension
                w_dist = self.wasserstein(pred_dim, targ_dim)
                dim_losses.append(w_dist)
            else:
                # Not enough valid samples
                dim_losses.append(torch.tensor(0.0, device=device))

        # Stack dimension losses
        dim_losses_tensor = torch.stack(dim_losses)

        # Apply weights if provided
        if self.weights is not None:
            weighted_losses = dim_losses_tensor * self.weights
            total_loss = weighted_losses.sum()
        else:
            total_loss = dim_losses_tensor.mean()

        return total_loss, dim_losses_tensor


class BufferedWeightedWassersteinLoss(nn.Module):
    """Enhanced Wasserstein loss with circular buffer for small batch training."""

    def __init__(
        self,
        buffer_size: int = 256,
        blur: float = 0.05,
        p: float = 2,
        scaling: float = 0.9,
        weights: Optional[torch.Tensor] = None,
        min_samples: int = 64,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.min_samples = min_samples

        # Initialize base Wasserstein loss
        self.base_wasserstein_loss = WeightedWassersteinLoss(
            blur=blur,
            p=p,
            scaling=scaling,
            weights=weights,
        )

        # Circular buffer for predictions and targets
        num_dims = 1 if weights is None or len(weights) == 1 else len(weights)
        self.register_buffer("pred_buffer", torch.zeros(buffer_size, num_dims))
        self.register_buffer("target_buffer", torch.zeros(buffer_size, num_dims))
        self.register_buffer("buffer_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("buffer_full", torch.zeros(1, dtype=torch.bool))
        self.register_buffer("total_samples", torch.zeros(1, dtype=torch.long))

    def update_buffer(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update circular buffer with new samples."""
        batch_size = predictions.size(0)
        ptr = int(self.buffer_ptr)

        # Handle wrap-around
        if ptr + batch_size <= self.buffer_size:
            self.pred_buffer[ptr : ptr + batch_size] = predictions.detach()
            self.target_buffer[ptr : ptr + batch_size] = targets.detach()
        else:
            # Split across boundary
            first_part = self.buffer_size - ptr
            self.pred_buffer[ptr:] = predictions[:first_part].detach()
            self.target_buffer[ptr:] = targets[:first_part].detach()

            second_part = batch_size - first_part
            self.pred_buffer[:second_part] = predictions[first_part:].detach()
            self.target_buffer[:second_part] = targets[first_part:].detach()

        # Update pointer and status
        self.buffer_ptr[0] = (ptr + batch_size) % self.buffer_size
        self.total_samples[0] = min(
            self.total_samples[0] + batch_size, self.buffer_size
        )
        if self.total_samples[0] >= self.buffer_size:
            self.buffer_full[0] = True

    def get_buffer_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all valid samples from buffer."""
        if self.buffer_full:
            return self.pred_buffer, self.target_buffer
        else:
            n_samples = int(self.total_samples)
            return self.pred_buffer[:n_samples], self.target_buffer[:n_samples]

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        all_predictions: Optional[torch.Tensor] = None,
        all_targets: Optional[torch.Tensor] = None,
        buffer_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Wasserstein loss with buffer support.

        Args:
            predictions: Local batch predictions
            targets: Local batch targets
            all_predictions: Gathered predictions from all GPUs (optional)
            all_targets: Gathered targets from all GPUs (optional)
            buffer_weight: Weight for buffer contribution (0-1)
        """
        device = predictions.device

        # Update buffer with gathered data when available
        if all_predictions is not None and all_targets is not None:
            self.update_buffer(all_predictions, all_targets)
            current_preds = all_predictions
            current_targets = all_targets
        else:
            self.update_buffer(predictions, targets)
            current_preds = predictions
            current_targets = targets

        # Check if we have enough samples
        if self.total_samples < self.min_samples:
            # Not enough samples yet, return zero loss
            return torch.tensor(0.0, device=device), torch.zeros(2, device=device)

        # Get buffer samples
        buffer_preds, buffer_targets = self.get_buffer_samples()
        buffer_preds = buffer_preds.to(device)
        buffer_targets = buffer_targets.to(device)

        # Combine current batch with buffer based on buffer_weight
        if buffer_weight < 1.0:
            # Weighted combination
            n_current = current_preds.size(0)
            n_buffer = int(n_current * buffer_weight / (1 - buffer_weight))
            n_buffer = min(n_buffer, buffer_preds.size(0))

            if n_buffer > 0:
                # Random sample from buffer
                indices = torch.randperm(buffer_preds.size(0))[:n_buffer]
                combined_preds = torch.cat(
                    [current_preds, buffer_preds[indices]], dim=0
                )
                combined_targets = torch.cat(
                    [current_targets, buffer_targets[indices]], dim=0
                )
            else:
                combined_preds = current_preds
                combined_targets = current_targets
        else:
            # Use all buffer samples
            combined_preds = torch.cat([current_preds, buffer_preds], dim=0)
            combined_targets = torch.cat([current_targets, buffer_targets], dim=0)

        # Compute Wasserstein loss on combined samples
        loss, dim_losses = self.base_wasserstein_loss(combined_preds, combined_targets)

        return loss, dim_losses


class BufferedWeightedSupCRCell(nn.Module):
    """Enhanced SupCR loss with circular buffer for small batch training."""

    def __init__(
        self,
        buffer_size: int = 256,
        embedding_dim: int = 128,
        temperature: float = 0.1,
        weights: Optional[torch.Tensor] = None,
        min_samples: int = 32,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.embedding_dim = embedding_dim
        self.min_samples = min_samples

        # Initialize base SupCR loss
        self.base_supcr = WeightedSupCRCell(temperature=temperature, weights=weights)

        # Circular buffer for embeddings and labels
        num_dims = 1 if weights is None or len(weights) == 1 else len(weights)
        self.register_buffer(
            "embedding_buffer", torch.zeros(buffer_size, embedding_dim)
        )
        self.register_buffer("label_buffer", torch.zeros(buffer_size, num_dims))
        self.register_buffer("buffer_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("buffer_full", torch.zeros(1, dtype=torch.bool))
        self.register_buffer("total_samples", torch.zeros(1, dtype=torch.long))

    def update_buffer(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Update circular buffer with new samples."""
        batch_size = embeddings.size(0)
        ptr = int(self.buffer_ptr)

        # Handle wrap-around
        if ptr + batch_size <= self.buffer_size:
            self.embedding_buffer[ptr : ptr + batch_size] = embeddings.detach()
            self.label_buffer[ptr : ptr + batch_size] = labels.detach()
        else:
            # Split across boundary
            first_part = self.buffer_size - ptr
            self.embedding_buffer[ptr:] = embeddings[:first_part].detach()
            self.label_buffer[ptr:] = labels[:first_part].detach()

            second_part = batch_size - first_part
            self.embedding_buffer[:second_part] = embeddings[first_part:].detach()
            self.label_buffer[:second_part] = labels[first_part:].detach()

        # Update pointer and status
        self.buffer_ptr[0] = (ptr + batch_size) % self.buffer_size
        self.total_samples[0] = min(
            self.total_samples[0] + batch_size, self.buffer_size
        )
        if self.total_samples[0] >= self.buffer_size:
            self.buffer_full[0] = True

    def get_buffer_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all valid samples from buffer."""
        if self.buffer_full:
            return self.embedding_buffer, self.label_buffer
        else:
            n_samples = int(self.total_samples)
            return self.embedding_buffer[:n_samples], self.label_buffer[:n_samples]

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        all_embeddings: Optional[torch.Tensor] = None,
        all_labels: Optional[torch.Tensor] = None,
        buffer_weight: float = 1.0,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute SupCR loss with buffer support.

        Args:
            embeddings: Local batch embeddings
            labels: Local batch labels
            all_embeddings: Gathered embeddings from all GPUs (optional)
            all_labels: Gathered labels from all GPUs (optional)
            buffer_weight: Weight for buffer contribution (0-1)
            temperature: Optional temperature override
        """
        device = embeddings.device

        # Update buffer with gathered data when available
        if all_embeddings is not None and all_labels is not None:
            self.update_buffer(all_embeddings, all_labels)
            current_embeddings = all_embeddings
            current_labels = all_labels
        else:
            self.update_buffer(embeddings, labels)
            current_embeddings = embeddings
            current_labels = labels

        # Check if we have enough samples
        if self.total_samples < self.min_samples:
            # Not enough samples yet, return zero loss
            return torch.tensor(0.0, device=device), torch.zeros(2, device=device)

        # Update temperature if provided
        if temperature is not None:
            self.base_supcr.supcr.temperature = temperature

        # Get buffer samples
        buffer_embeddings, buffer_labels = self.get_buffer_samples()
        buffer_embeddings = buffer_embeddings.to(device)
        buffer_labels = buffer_labels.to(device)

        # For SupCR, we always want to use buffer as negative samples
        combined_embeddings = torch.cat([current_embeddings, buffer_embeddings], dim=0)
        combined_labels = torch.cat([current_labels, buffer_labels], dim=0)

        # Compute SupCR loss
        loss, dim_losses = self.base_supcr(combined_embeddings, combined_labels)

        # Scale by buffer weight to reduce influence when buffer is stale
        loss = loss * (1.0 - buffer_weight + buffer_weight * 0.5)

        return loss, dim_losses


class MleWassSupCR(nn.Module):
    """
    Composite loss combining MSE, Wasserstein distance, and SupCR with
    circular buffers, DDP synchronization, and adaptive weighting.
    """

    def __init__(
        self,
        # Component configuration
        lambda_mse: float = 1.0,
        lambda_wasserstein: float = 0.1,
        lambda_supcr: float = 0.001,
        # Wasserstein-specific parameters
        wasserstein_blur: float = 0.05,
        wasserstein_p: float = 2,
        wasserstein_scaling: float = 0.9,
        # SupCR-specific parameters
        supcr_temperature: float = 0.1,
        embedding_dim: int = 128,
        # Buffer configuration
        use_buffer: bool = True,
        buffer_size: int = 256,
        min_samples_for_wasserstein: int = 64,
        min_samples_for_supcr: int = 64,
        # DDP configuration
        use_ddp_gather: bool = True,
        gather_interval: int = 1,
        # Adaptive weighting
        use_adaptive_weighting: bool = True,
        warmup_epochs: Optional[int] = None,
        stable_epoch: Optional[int] = None,
        # Temperature scheduling
        use_temp_scheduling: bool = True,
        init_temperature: float = 1.0,
        final_temperature: float = 0.1,
        temp_schedule: str = "exponential",
        # Other parameters
        weights: Optional[torch.Tensor] = None,
        max_epochs: int = 1000,
    ):
        super().__init__()

        # Lambda weights
        self.lambda_mse = lambda_mse
        self.lambda_wasserstein = lambda_wasserstein
        self.lambda_supcr = lambda_supcr

        # Configuration flags
        self.use_buffer = use_buffer
        self.use_ddp_gather = use_ddp_gather
        self.gather_interval = gather_interval
        self.use_adaptive_weighting = use_adaptive_weighting
        self.use_temp_scheduling = use_temp_scheduling
        self.max_epochs = max_epochs

        # Initialize loss components
        self.mse_loss = WeightedMSELoss(weights=weights)

        if use_buffer:
            self.wasserstein_loss = BufferedWeightedWassersteinLoss(
                buffer_size=buffer_size,
                blur=wasserstein_blur,
                p=wasserstein_p,
                scaling=wasserstein_scaling,
                weights=weights,
                min_samples=min_samples_for_wasserstein,
            )
            self.supcr_loss = BufferedWeightedSupCRCell(
                buffer_size=buffer_size,
                embedding_dim=embedding_dim,
                temperature=supcr_temperature,
                weights=weights,
                min_samples=min_samples_for_supcr,
            )
        else:
            self.wasserstein_loss = WeightedWassersteinLoss(
                blur=wasserstein_blur,
                p=wasserstein_p,
                scaling=wasserstein_scaling,
                weights=weights,
            )
            self.supcr_loss = WeightedSupCRCell(
                temperature=supcr_temperature, weights=weights
            )

        # Initialize adaptive components with dynamic defaults
        if use_adaptive_weighting:
            actual_warmup_epochs = (
                warmup_epochs if warmup_epochs is not None else int(max_epochs * 0.1)
            )
            actual_stable_epoch = (
                stable_epoch if stable_epoch is not None else int(max_epochs * 0.5)
            )
            self.adaptive_weighting = AdaptiveWeighting(
                actual_warmup_epochs, actual_stable_epoch
            )

        if use_temp_scheduling:
            self.temp_scheduler = TemperatureScheduler(
                init_temperature, final_temperature, temp_schedule
            )

        # Tracking
        self.register_buffer("forward_count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("current_epoch", torch.zeros(1, dtype=torch.long))

    def gather_across_gpus(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors from all GPUs."""
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return tensor

        world_size = dist.get_world_size()
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        z_P: torch.Tensor,
        epoch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass computing composite loss.

        Args:
            predictions: Model predictions [batch_size, num_dims]
            targets: Target values [batch_size, num_dims]
            z_P: Perturbed embeddings for SupCR [batch_size, embedding_dim]
            epoch: Current training epoch (optional)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Update epoch if provided
        if epoch is not None:
            self.current_epoch[0] = epoch

        device = predictions.device
        batch_size = predictions.size(0)

        # Initialize loss components
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        # Get adaptive weights
        buffer_weight = 1.0
        if self.use_adaptive_weighting and self.use_buffer:
            buffer_weight = self.adaptive_weighting.get_buffer_weight(
                self.current_epoch.item()
            )
            loss_dict["buffer_weight"] = buffer_weight

        # Get temperature
        temperature = None
        if self.use_temp_scheduling:
            temperature = self.temp_scheduler.get_temperature(
                self.current_epoch.item(), self.max_epochs
            )
            loss_dict["temperature"] = temperature

        # Gather samples across GPUs if needed
        all_predictions = None
        all_targets = None
        all_embeddings = None

        if self.use_ddp_gather and self.forward_count % self.gather_interval == 0:
            all_predictions = self.gather_across_gpus(predictions)
            all_targets = self.gather_across_gpus(targets)
            all_embeddings = self.gather_across_gpus(z_P)

        # Compute MSE loss if lambda > 0
        if self.lambda_mse > 0:
            mse_val, mse_dims = self.mse_loss(predictions, targets)
            weighted_mse = self.lambda_mse * mse_val
            total_loss = total_loss + weighted_mse
            loss_dict.update(
                {
                    "mse_loss": mse_val.item(),
                    "mse_dim_losses": mse_dims,
                    "weighted_mse": weighted_mse.item(),
                }
            )
        else:
            mse_val = torch.tensor(0.0, device=device)
            mse_dims = torch.zeros(2, device=device)
            loss_dict.update(
                {
                    "mse_loss": 0.0,
                    "mse_dim_losses": mse_dims,
                    "weighted_mse": 0.0,
                }
            )

        # Compute Wasserstein loss if lambda > 0
        if self.lambda_wasserstein > 0:
            if self.use_buffer:
                wasserstein_val, wasserstein_dims = self.wasserstein_loss(
                    predictions, targets, all_predictions, all_targets, buffer_weight
                )
            else:
                wasserstein_val, wasserstein_dims = self.wasserstein_loss(
                    predictions, targets
                )

            weighted_wasserstein = self.lambda_wasserstein * wasserstein_val
            total_loss = total_loss + weighted_wasserstein
            loss_dict.update(
                {
                    "wasserstein_loss": wasserstein_val.item(),
                    "wasserstein_dim_losses": wasserstein_dims,
                    "weighted_wasserstein": weighted_wasserstein.item(),
                }
            )
        else:
            wasserstein_val = torch.tensor(0.0, device=device)
            wasserstein_dims = torch.zeros(2, device=device)
            loss_dict.update(
                {
                    "wasserstein_loss": 0.0,
                    "wasserstein_dim_losses": wasserstein_dims,
                    "weighted_wasserstein": 0.0,
                }
            )

        # Compute SupCR loss if lambda > 0
        if self.lambda_supcr > 0:
            if self.use_buffer:
                supcr_val, supcr_dims = self.supcr_loss(
                    z_P, targets, all_embeddings, all_targets, buffer_weight, temperature
                )
            else:
                supcr_val, supcr_dims = self.supcr_loss(z_P, targets)

            weighted_supcr = self.lambda_supcr * supcr_val
            total_loss = total_loss + weighted_supcr
            loss_dict.update(
                {
                    "supcr_loss": supcr_val.item(),
                    "supcr_dim_losses": supcr_dims,
                    "weighted_supcr": weighted_supcr.item(),
                }
            )
        else:
            supcr_val = torch.tensor(0.0, device=device)
            supcr_dims = torch.zeros(2, device=device)
            loss_dict.update(
                {
                    "supcr_loss": 0.0,
                    "supcr_dim_losses": supcr_dims,
                    "weighted_supcr": 0.0,
                }
            )

        # Compute normalized contributions
        if total_loss > 0:
            for key in ["mse", "wasserstein", "supcr"]:
                if f"weighted_{key}" in loss_dict:
                    loss_dict[f"norm_weighted_{key}"] = (
                        loss_dict[f"weighted_{key}"] / total_loss.item()
                    )
            # Also add dist for compatibility
            if "weighted_dist" in loss_dict:
                loss_dict["norm_weighted_dist"] = (
                    loss_dict["weighted_dist"] / total_loss.item()
                )

        # Add unweighted totals for compatibility
        loss_dict["total_weighted"] = total_loss.item()
        loss_dict["total_loss"] = total_loss.item()

        # Add unweighted normalized (for compatibility)
        total_unweighted = mse_val + wasserstein_val + supcr_val
        if total_unweighted > 0:
            loss_dict["norm_unweighted_mse"] = mse_val.item() / total_unweighted.item()
            loss_dict["norm_unweighted_wasserstein"] = (
                wasserstein_val.item() / total_unweighted.item()
            )
            loss_dict["norm_unweighted_supcr"] = (
                supcr_val.item() / total_unweighted.item()
            )
            # Also add dist for compatibility
            loss_dict["norm_unweighted_dist"] = (
                wasserstein_val.item() / total_unweighted.item()
            )

        # Update forward counter
        self.forward_count += 1

        return total_loss, loss_dict