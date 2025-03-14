from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchcell.losses.multi_dim_nan_tolerant import (
    WeightedMSELoss,
    WeightedDistLoss,
    WeightedSupCRCell,
)
import math


class ICLoss(nn.Module):
    def __init__(
        self,
        lambda_dist: float,
        lambda_supcr: float,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:
            lambda_dist: Weight for the dist loss.
            lambda_supcr: Weight for the SupCR loss.
            weights: Optional weights applied uniformly to all losses.
        """
        super().__init__()
        self.lambda_dist = lambda_dist
        self.lambda_supcr = lambda_supcr

        self.mse_loss_fn = WeightedMSELoss(weights=weights)
        self.dist_loss_fn = WeightedDistLoss(weights=weights)
        self.supcr_fn = WeightedSupCRCell(weights=weights)

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor, z_P: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        mse_loss, mse_dim_losses = self.mse_loss_fn(predictions, targets)
        dist_loss, dist_dim_losses = self.dist_loss_fn(predictions, targets)

        # Now only passing z_P
        supcr_loss, supcr_dim_losses = self.supcr_fn(z_P, targets)

        # Weighted versions (using lambda multipliers)
        weighted_mse = mse_loss
        weighted_dist = self.lambda_dist * dist_loss
        weighted_supcr = self.lambda_supcr * supcr_loss
        total_weighted = weighted_mse + weighted_dist + weighted_supcr

        # Also compute the unweighted total (without lambda multipliers)
        total_unweighted = mse_loss + dist_loss + supcr_loss

        # Normalized weighted losses
        if total_weighted != 0:
            norm_weighted = {
                "mse": weighted_mse / total_weighted,
                "dist": weighted_dist / total_weighted,
                "supcr": weighted_supcr / total_weighted,
            }
        else:
            norm_weighted = {k: 0 for k in ["mse", "dist", "supcr"]}

        if total_unweighted != 0:
            norm_unweighted = {
                "mse": mse_loss / total_unweighted,
                "dist": dist_loss / total_unweighted,
                "supcr": supcr_loss / total_unweighted,
            }
        else:
            norm_unweighted = {k: 0 for k in ["mse", "dist", "supcr"]}

        total_loss = total_weighted

        loss_dict = {
            "mse_loss": mse_loss,
            "mse_dim_losses": mse_dim_losses,
            "dist_loss": dist_loss,
            "dist_dim_losses": dist_dim_losses,
            "supcr_loss": supcr_loss,
            "supcr_dim_losses": supcr_dim_losses,
            "weighted_mse": weighted_mse,
            "weighted_dist": weighted_dist,
            "weighted_supcr": weighted_supcr,
            "total_weighted": total_weighted,
            "total_loss": total_loss,
            "norm_weighted_mse": norm_weighted["mse"],
            "norm_weighted_dist": norm_weighted["dist"],
            "norm_weighted_supcr": norm_weighted["supcr"],
            "norm_unweighted_mse": norm_unweighted["mse"],
            "norm_unweighted_dist": norm_unweighted["dist"],
            "norm_unweighted_supcr": norm_unweighted["supcr"],
        }
        return total_loss, loss_dict


class ICLossStd(nn.Module):
    def __init__(
        self,
        lambda_dist: float,
        lambda_supcr: float,
        lambda_reg: float = 0.01,
        init_sigma: float = 1.0,
        task_weights: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> None:
        """
        Enhanced ICLoss with learnable per-task standard deviations.

        Implements: L = ∑(w_t/(2σ_t²) * L_t + log σ_t) + λ∑σ_t⁻²

        where L_t is the task-specific loss (fitness or gene interaction).

        Args:
            lambda_dist: Weight for the distance preservation loss
            lambda_supcr: Weight for the supervised contrastive loss
            lambda_reg: Regularization parameter for the standard deviations
            init_sigma: Initial value for the standard deviation parameters
            task_weights: Optional fixed weights for different tasks [2]
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.lambda_dist = lambda_dist
        self.lambda_supcr = lambda_supcr
        self.lambda_reg = lambda_reg
        self.eps = eps

        # Task weights (if not provided, default to ones)
        if task_weights is None:
            self.task_weights = torch.ones(2)
        else:
            self.task_weights = task_weights

        # Learnable log standard deviations (one for fitness, one for gene interaction)
        self.log_sigma = nn.Parameter(
            torch.ones(2) * torch.log(torch.tensor(init_sigma))
        )

        # Base loss functions
        self.mse_loss_fn = WeightedMSELoss(weights=None)  # We'll apply weights manually
        self.dist_loss_fn = WeightedDistLoss(weights=None)
        self.supcr_fn = WeightedSupCRCell(weights=None)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        z_P: torch.Tensor,
        z_I: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass to compute the heteroscedastic IC loss.

        Args:
            predictions: Model predictions [batch_size, 2]
            targets: Target values [batch_size, 2]
            z_P: Perturbed difference embeddings from the model
            z_I: Intact embeddings from the model

        Returns:
            tuple: (total_loss, loss_dict)
        """
        device = predictions.device

        # Get standard deviations from learned parameters
        sigma = torch.exp(self.log_sigma).to(device)

        # Calculate base losses (without applying task weights yet)
        mse_loss, mse_dim_losses = self.mse_loss_fn(predictions, targets)
        dist_loss, dist_dim_losses = self.dist_loss_fn(predictions, targets)
        supcr_loss, supcr_dim_losses = self.supcr_fn(z_P, z_I, targets)

        # Apply lambda weights to dist and supcr losses
        weighted_mse = mse_loss
        weighted_dist = self.lambda_dist * dist_loss
        weighted_supcr = self.lambda_supcr * supcr_loss

        # Combine to get per-task loss
        # Fitness loss is index 0, gene interaction is index 1
        base_task_losses = torch.tensor(
            [
                mse_dim_losses[0]
                + self.lambda_dist * dist_dim_losses[0]
                + self.lambda_supcr * supcr_dim_losses[0],
                mse_dim_losses[1]
                + self.lambda_dist * dist_dim_losses[1]
                + self.lambda_supcr * supcr_dim_losses[1],
            ],
            device=device,
        )

        # Apply heteroscedastic weighting: (w_t/(2σ_t²) * L_t + log σ_t)
        task_weights = self.task_weights.to(device)
        weighted_losses = (
            task_weights / (2 * sigma**2 + self.eps)
        ) * base_task_losses + torch.log(sigma + self.eps)

        # Calculate regularization term: λ∑σ_t⁻²
        reg_term = self.lambda_reg * torch.sum(1.0 / (sigma**2 + self.eps))

        # Final loss
        total_loss = weighted_losses.sum() + reg_term

        # Calculate additional metrics for monitoring
        # Create masks for valid (non-NaN) values
        mask = ~torch.isnan(targets)
        fitness_mask = mask[:, 0]
        gi_mask = mask[:, 1]

        # Standard deviation of predictions
        fitness_pred = predictions[:, 0]
        gi_pred = predictions[:, 1]
        fitness_pred_std = (
            fitness_pred[fitness_mask].std()
            if fitness_mask.any()
            else torch.tensor(0.0, device=device)
        )
        gi_pred_std = (
            gi_pred[gi_mask].std()
            if gi_mask.any()
            else torch.tensor(0.0, device=device)
        )

        # Prepare output dictionary
        loss_dict = {
            # Learned parameters
            "sigma_fitness": sigma[0].item(),
            "sigma_gi": sigma[1].item(),
            # Individual losses
            "mse_loss": mse_loss.item(),
            "dist_loss": dist_loss.item(),
            "supcr_loss": supcr_loss.item(),
            # Per-task losses
            "fitness_base_loss": base_task_losses[0].item(),
            "gi_base_loss": base_task_losses[1].item(),
            "fitness_weighted_loss": weighted_losses[0].item(),
            "gi_weighted_loss": weighted_losses[1].item(),
            # Regularization
            "reg_term": reg_term.item(),
            # Weighted components
            "weighted_mse": weighted_mse.item(),
            "weighted_dist": weighted_dist.item(),
            "weighted_supcr": weighted_supcr.item(),
            # Prediction statistics
            "fitness_pred_std": fitness_pred_std.item(),
            "gi_pred_std": gi_pred_std.item(),
            # Total loss
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict
