# torchcell/losses/dcell_new.py
# [[torchcell.losses.dcell_new]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/losses/dcell_new.py
# Test file: torchcell/losses/test_dcell_new.py

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional


class DCellLoss(nn.Module):
    """
    Loss function for the DCellNew model implementation.

    This loss function:
    1. Takes the direct model outputs (predictions and outputs dictionary)
    2. Applies the main loss to the root subsystem predictions
    3. Optionally applies auxiliary losses to non-root subsystem outputs

    Args:
        alpha: Weight for auxiliary losses (default: 0.3)
        use_auxiliary_losses: Whether to use losses from non-root subsystems (default: True)
    """

    def __init__(self, alpha: float = 0.3, use_auxiliary_losses: bool = True):
        super().__init__()
        self.alpha = alpha
        self.use_auxiliary_losses = use_auxiliary_losses
        self.criterion = nn.MSELoss()

    def forward(
        self, predictions: torch.Tensor, outputs: Dict[str, Any], target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the loss for DCellNew outputs.

        Args:
            predictions: Primary predictions tensor from the model (root output)
            outputs: Dictionary of all model outputs including subsystem states
            target: Target values to predict

        Returns:
            Tuple of (total_loss, loss_components) where loss_components is a dictionary
            containing the primary_loss, auxiliary_loss, and weighted_auxiliary_loss.
        """
        # Primary loss on main predictions
        primary_loss = self.criterion(predictions, target)

        # Initialize auxiliary loss components
        auxiliary_loss = torch.tensor(0.0, device=primary_loss.device)
        weighted_auxiliary_loss = torch.tensor(0.0, device=primary_loss.device)

        # Create a dictionary to store all loss components
        loss_components = {
            "primary_loss": primary_loss.detach(),
            "auxiliary_loss": auxiliary_loss.detach(),
            "weighted_auxiliary_loss": weighted_auxiliary_loss.detach(),
        }

        # If not using auxiliary losses, return only primary loss
        if not self.use_auxiliary_losses:
            return primary_loss, loss_components

        # Get all linear outputs from subsystems
        linear_outputs = outputs.get("linear_outputs", {})

        # Compute auxiliary losses on non-root subsystems
        auxiliary_losses = []
        for subsystem_name, subsystem_output in linear_outputs.items():
            # Skip the root output since it's the same as predictions
            # The is_root check is more robust than exact equality
            is_root = subsystem_name == "GO:ROOT" or torch.equal(subsystem_output, predictions)
            if is_root:
                continue

            # Add loss for this subsystem
            aux_loss = self.criterion(subsystem_output, target)
            auxiliary_losses.append(aux_loss)

        # If no auxiliary losses, return only primary loss
        if not auxiliary_losses:
            return primary_loss, loss_components

        # Combine losses with weight alpha
        auxiliary_loss = torch.stack(auxiliary_losses).mean()
        weighted_auxiliary_loss = self.alpha * auxiliary_loss
        total_loss = primary_loss + weighted_auxiliary_loss

        # Update loss components dictionary
        loss_components["auxiliary_loss"] = auxiliary_loss.detach()
        loss_components["weighted_auxiliary_loss"] = weighted_auxiliary_loss.detach()

        return total_loss, loss_components
