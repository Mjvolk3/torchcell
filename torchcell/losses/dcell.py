# torchcell/losses/dcell.py
# [[torchcell.losses.dcell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/losses/dcell.py
# Test file: tests/torchcell/losses/test_losses_dcell.py

"""Loss for the DCell model: root MSE plus optional subsystem auxiliary losses."""

from typing import Any, Literal

import torch
import torch.nn as nn


class DCellLoss(nn.Module):
    r"""
    Loss function for the DCell model (Ma et al. 2018).

    The paper's objective (mirror ``torchcell-library/maUsingDeepLearning2018/paper.md``,
    line 196, sha256 ``ac837bc3...``) is, per sample,
    ``Loss(Linear(O^(r)), y) + alpha * sum_{t != r} Loss(Linear(O^(t)), y)``, and line
    199 states: "Loss is the squared error loss function, and $r$ is the root of the
    hierarchy" and "the parameter $\alpha$ $_ { ( = 0 . 3 ) }$ balances these two
    contributions". The auxiliary term is a SUM over every non-root subsystem, so
    ``aux_reduction="sum"`` (the default) follows the paper. ``aux_reduction="mean"``
    divides that sum by the number of non-root subsystems T - 1, an effective alpha of
    0.3 / (T - 1); every 005/006 DCell run trained before 2026-09-30 used it, so pass
    ``"mean"`` to reproduce or compare against those runs.

    The root is recognized by KEY, never by value: ``"GO:ROOT"`` and any key bound to
    the very same tensor object (``torchcell.models.dcell.DCell`` stores the root head
    under ``GO:<root index>`` and aliases it as ``GO:ROOT``). A non-root head whose
    values happen to equal the root is still counted.

    Args:
        alpha: Weight for auxiliary losses (default: 0.3, the paper's value).
        use_auxiliary_losses: Whether to use losses from non-root subsystems
            (default: True).
        aux_reduction: ``"sum"`` (the paper) or ``"mean"`` over the non-root
            subsystem MSEs (default: ``"sum"``).
    """

    def __init__(
        self,
        alpha: float = 0.3,
        use_auxiliary_losses: bool = True,
        aux_reduction: Literal["sum", "mean"] = "sum",
    ):
        """Set the auxiliary-loss weight, toggle, reduction and MSE criterion.

        Args:
            alpha: Weight applied to the auxiliary subsystem losses.
            use_auxiliary_losses: Whether to include non-root subsystem losses.
            aux_reduction: How the non-root subsystem MSEs combine, ``"sum"`` or
                ``"mean"``.

        Raises:
            ValueError: If ``aux_reduction`` is neither ``"sum"`` nor ``"mean"``.
        """
        super().__init__()
        if aux_reduction not in ("sum", "mean"):
            raise ValueError(
                f"aux_reduction must be 'sum' or 'mean', got {aux_reduction!r}"
            )
        self.alpha = alpha
        self.use_auxiliary_losses = use_auxiliary_losses
        self.aux_reduction = aux_reduction
        self.criterion = nn.MSELoss()

    def forward(
        self, predictions: torch.Tensor, outputs: dict[str, Any], target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute the loss for DCell outputs.

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

        # The root is "GO:ROOT" plus the key the model aliased to the same object
        root_output = linear_outputs.get("GO:ROOT")
        auxiliary_losses = [
            self.criterion(subsystem_output, target)
            for subsystem_name, subsystem_output in linear_outputs.items()
            if subsystem_name != "GO:ROOT" and subsystem_output is not root_output
        ]

        # If no auxiliary losses, return only primary loss
        if not auxiliary_losses:
            return primary_loss, loss_components

        # Combine losses with weight alpha
        stacked = torch.stack(auxiliary_losses)
        auxiliary_loss = (
            stacked.sum() if self.aux_reduction == "sum" else stacked.mean()
        )
        weighted_auxiliary_loss = self.alpha * auxiliary_loss
        total_loss = primary_loss + weighted_auxiliary_loss

        # Update loss components dictionary
        loss_components["auxiliary_loss"] = auxiliary_loss.detach()
        loss_components["weighted_auxiliary_loss"] = weighted_auxiliary_loss.detach()

        return total_loss, loss_components
