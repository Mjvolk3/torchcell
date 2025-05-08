# torchcell/losses/dango
# [[torchcell.losses.dango]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/losses/dango
# Test file: tests/torchcell/losses/test_dango.py


import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union, List


class DangoLoss(nn.Module):
    """
    Combined loss for the DANGO model with dynamic weighting between 
    reconstruction loss and interaction loss.
    
    The loss gradually transitions from focusing on reconstruction to
    an even balance between reconstruction and interaction.

    Args:
        edge_types: List of edge types for reconstruction loss
        lambda_values: Dictionary mapping edge types to their lambda values for weighted MSE
        epochs_until_uniform: Number of epochs until weights become uniform (0.5 each).
                             If None, uses a fixed 0.5 weight for both losses.
        reduction: Reduction method for the loss ('none', 'mean', 'sum')
    """
    def __init__(
        self,
        edge_types: List[str],
        lambda_values: Dict[str, float],
        epochs_until_uniform: Optional[int] = None,
        reduction: str = "mean"
    ) -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"Invalid reduction mode: {reduction}")
        
        self.edge_types = edge_types
        self.lambda_values = lambda_values
        self.epochs_until_uniform = epochs_until_uniform
        self.reduction = reduction
        
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

        for edge_type in self.edge_types:
            if edge_type in reconstructions and edge_type in adjacency_matrices:
                lambda_value = self.lambda_values[edge_type]
                loss = self.compute_weighted_mse_loss(
                    reconstructions[edge_type],
                    adjacency_matrices[edge_type],
                    lambda_value,
                )
                total_loss += loss

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
        
        # Determine weights based on current epoch
        if self.epochs_until_uniform is None:
            # Fixed equal weighting
            alpha = 0.5
        else:
            if current_epoch >= self.epochs_until_uniform:
                # After epochs_until_uniform, use uniform weights
                alpha = 0.5
            else:
                # Linearly adjust from 1.0 (recon only) to 0.5 (balanced)
                progress = current_epoch / self.epochs_until_uniform
                alpha = 1.0 - (0.5 * progress)
                
        # Combine losses with dynamic weighting
        total_loss = alpha * recon_loss + (1 - alpha) * interaction_loss
        
        # Create loss dictionary for logging
        loss_dict = {
            "reconstruction_loss": recon_loss,
            "interaction_loss": interaction_loss,
            "alpha": torch.tensor(alpha, device=recon_loss.device)
        }
        
        return total_loss, loss_dict