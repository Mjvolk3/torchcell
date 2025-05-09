# torchcell/losses/dango
# [[torchcell.losses.dango]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/losses/dango
# Test file: tests/torchcell/losses/test_dango.py


import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union, List
from abc import ABC, abstractmethod
import math


class DangoLossSched(ABC):
    """
    Abstract base class for DANGO loss schedulers.
    
    All schedulers must implement a forward method that computes the total loss
    based on reconstruction loss, interaction loss, and current epoch.
    """
    
    @abstractmethod
    def forward(
        self, 
        recon_loss: torch.Tensor,
        interaction_loss: torch.Tensor,
        current_epoch: int = 0
    ) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
        """
        Compute the combined loss according to the schedule.
        
        Args:
            recon_loss: Reconstruction loss value
            interaction_loss: Interaction prediction loss value
            current_epoch: Current training epoch
            
        Returns:
            Tuple containing:
                - total_loss: The combined loss value
                - alpha: Weight of reconstruction loss (for logging)
                - weighted_recon_loss: The weighted reconstruction loss component
                - weighted_interaction_loss: The weighted interaction loss component
        """
        pass


class PreThenPost(DangoLossSched):
    """
    Schedule that uses reconstruction loss only before transition_epoch,
    then switches to interaction loss only.
    
    Args:
        transition_epoch: Epoch at which to switch from reconstruction to interaction loss
    """
    
    def __init__(self, transition_epoch: int = 10):
        self.transition_epoch = transition_epoch
    
    def forward(
        self, 
        recon_loss: torch.Tensor,
        interaction_loss: torch.Tensor,
        current_epoch: int = 0
    ) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
        """
        Compute loss based on transition epoch.
        
        Before transition: alpha = 1.0 (reconstruction only)
        After transition: alpha = 0.0 (interaction only)
        """
        if current_epoch < self.transition_epoch:
            # Before transition: use only reconstruction loss
            alpha = 1.0
            weighted_recon_loss = recon_loss
            weighted_interaction_loss = torch.zeros_like(interaction_loss)
            total_loss = weighted_recon_loss
        else:
            # After transition: use only interaction loss
            alpha = 0.0
            weighted_recon_loss = torch.zeros_like(recon_loss)
            weighted_interaction_loss = interaction_loss
            total_loss = weighted_interaction_loss
            
        return total_loss, alpha, weighted_recon_loss, weighted_interaction_loss


class LinearUntilUniform(DangoLossSched):
    """
    Schedule that linearly transitions from reconstruction-focused to uniform weighting.
    
    Args:
        transition_epoch: Epoch at which weights become uniform (0.5 each)
    """
    
    def __init__(self, transition_epoch: int = 20):
        self.transition_epoch = transition_epoch
    
    def forward(
        self, 
        recon_loss: torch.Tensor,
        interaction_loss: torch.Tensor,
        current_epoch: int = 0
    ) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
        """
        Compute loss with linearly decreasing alpha.
        
        Alpha starts at 1.0 (reconstruction only) and linearly decreases to 0.5 (uniform)
        over transition_epoch epochs. After that, it remains at 0.5.
        """
        if current_epoch >= self.transition_epoch:
            # After transition_epoch, use uniform weights
            alpha = 0.5
        else:
            # Linearly adjust from 1.0 (recon only) to 0.5 (balanced)
            progress = current_epoch / self.transition_epoch
            alpha = 1.0 - (0.5 * progress)
        
        # Calculate weighted loss components
        weighted_recon_loss = alpha * recon_loss
        weighted_interaction_loss = (1 - alpha) * interaction_loss
        
        # Combine losses with dynamic weighting
        total_loss = weighted_recon_loss + weighted_interaction_loss
        
        return total_loss, alpha, weighted_recon_loss, weighted_interaction_loss


class LinearUntilFlipped(DangoLossSched):
    """
    Schedule that linearly transitions from reconstruction-focused to interaction-focused.
    
    Args:
        transition_epoch: Epoch at which the transition completes (fully flipped to interaction loss)
    """
    
    def __init__(self, transition_epoch: int = 20):
        self.transition_epoch = transition_epoch
    
    def forward(
        self, 
        recon_loss: torch.Tensor,
        interaction_loss: torch.Tensor,
        current_epoch: int = 0
    ) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
        """
        Compute loss with a linear transition from recon loss to interaction loss.
        
        Alpha linearly decreases from 1.0 (reconstruction only) to 0.0 (interaction only)
        over transition_epoch epochs.
        """
        # Calculate alpha using linear transition
        if current_epoch >= self.transition_epoch:
            # After transition_epoch, use only interaction loss
            alpha = 0.0
        else:
            # Linear decrease from 1.0 to 0.0 over transition_epoch
            alpha = 1.0 - (current_epoch / self.transition_epoch)
        
        # Calculate weighted loss components
        weighted_recon_loss = alpha * recon_loss
        weighted_interaction_loss = (1 - alpha) * interaction_loss
        
        # Combine losses with dynamic weighting
        total_loss = weighted_recon_loss + weighted_interaction_loss
        
        return total_loss, alpha, weighted_recon_loss, weighted_interaction_loss


# Dictionary mapping scheduler type names to their respective classes
SCHEDULER_MAP = {
    "PreThenPost": PreThenPost,
    "LinearUntilUniform": LinearUntilUniform,
    "LinearUntilFlipped": LinearUntilFlipped,
}


class DangoLoss(nn.Module):
    """
    Combined loss for the DANGO model with dynamic weighting between 
    reconstruction loss and interaction loss.
    
    Args:
        edge_types: List of edge types for reconstruction loss
        lambda_values: Dictionary mapping edge types to their lambda values for weighted MSE
        scheduler: DangoLossSched instance that determines loss weighting over epochs
        reduction: Reduction method for the loss ('none', 'mean', 'sum')
    """
    def __init__(
        self,
        edge_types: List[str],
        lambda_values: Dict[str, float],
        scheduler: Optional[DangoLossSched] = None,
        reduction: str = "mean"
    ) -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"Invalid reduction mode: {reduction}")
        
        self.edge_types = edge_types
        self.lambda_values = lambda_values
        self.scheduler = scheduler or PreThenPost(transition_epoch=10)
        self.reduction = reduction
        
        # Ensure scheduler is a DangoLossSched instance
        if not isinstance(self.scheduler, DangoLossSched):
            raise TypeError("scheduler must be an instance of DangoLossSched")
        
    def compute_weighted_mse_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor, lambda_value: float
    ) -> torch.Tensor:
        """
        Compute the weighted MSE loss as defined in the paper

        Args:
            predictions: Reconstructed adjacency matrix rows
            targets: Ground truth adjacency matrix rows
            lambda_value: Weight for zero entries

        Returns:
            Weighted MSE loss
        """
        # Create masks for zero and non-zero entries
        non_zero_mask = (targets != 0).float()
        zero_mask = (targets == 0).float()

        # Calculate squared differences
        squared_diff = (predictions - targets) ** 2

        # Apply weighted MSE formula
        non_zero_loss = (squared_diff * non_zero_mask).sum()
        zero_loss = lambda_value * (squared_diff * zero_mask).sum()

        # Total number of entries
        N = targets.numel()

        # Final loss
        loss = (non_zero_loss + zero_loss) / N

        return loss

    def compute_reconstruction_loss(
        self,
        reconstructions: Dict[str, torch.Tensor],
        adjacency_matrices: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the total reconstruction loss across all networks

        Args:
            reconstructions: Dictionary of reconstructed adjacency matrix rows for each edge type
            adjacency_matrices: Dictionary of ground truth adjacency matrices for each edge type

        Returns:
            Total reconstruction loss
        """
        total_loss = 0.0
        num_networks = 0

        for edge_type in self.edge_types:
            if edge_type in reconstructions and edge_type in adjacency_matrices:
                lambda_value = self.lambda_values.get(edge_type, 1.0)
                loss = self.compute_weighted_mse_loss(
                    reconstructions[edge_type],
                    adjacency_matrices[edge_type],
                    lambda_value,
                )
                total_loss += loss
                num_networks += 1

        # Average across networks
        if num_networks > 0:
            total_loss /= num_networks

        return total_loss
    
    def compute_interaction_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the log-cosh loss for interaction prediction

        Args:
            predictions: Predicted interaction scores
            targets: Ground truth interaction scores

        Returns:
            Log-cosh loss
        """
        return torch.mean(torch.log(torch.cosh(predictions - targets)))
        
    def forward(
        self, 
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reconstructions: Dict[str, torch.Tensor],
        adjacency_matrices: Dict[str, torch.Tensor],
        current_epoch: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the combined loss for DANGO model.

        Args:
            predictions: Predicted interaction scores
            targets: Ground truth interaction scores
            reconstructions: Dictionary of reconstructed adjacency matrices
            adjacency_matrices: Dictionary of ground truth adjacency matrices
            current_epoch: Current training epoch (used for dynamic weighting)

        Returns:
            Tuple containing the total loss and a dictionary with component losses
        """
        # Compute reconstruction loss
        recon_loss = self.compute_reconstruction_loss(reconstructions, adjacency_matrices)
        
        # Compute interaction loss using log-cosh
        interaction_loss = self.compute_interaction_loss(predictions, targets)
        
        # Use scheduler to compute combined loss, alpha value, and weighted components
        total_loss, alpha, weighted_recon_loss, weighted_interaction_loss = self.scheduler.forward(
            recon_loss, interaction_loss, current_epoch
        )
        
        # Create loss dictionary for logging
        loss_dict = {
            "reconstruction_loss": recon_loss,
            "interaction_loss": interaction_loss,
            "weighted_reconstruction_loss": weighted_recon_loss,
            "weighted_interaction_loss": weighted_interaction_loss,
            "alpha": torch.tensor(alpha, device=recon_loss.device)
        }
        
        return total_loss, loss_dict