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

    The paper's objective (mirror ``torchcell-library/maUsingDeepLearning2018/paper.md``
    line 196, sha256
    ``ac837bc358ea4969a72789e66e31380bfdcbd7b8aea98dec47b108ee21d2070c``) is
    ``(1/N) sum_i (Loss(Linear(O_i^(r)), y_i) + alpha * sum_{t != r}
    Loss(Linear(O_i^(t)), y_i)) + lambda ||W||_2``, and line 199 states: "Loss is the
    squared error loss function, and $r$ is the root of the hierarchy" and "the
    parameter $\alpha$ $_ { ( = 0 . 3 ) }$ balances these two contributions". This
    class computes the first two terms; the ``lambda ||W||_2`` term is the optimizer's
    weight decay (``weight_decay`` in the 005/006 configs, 1e-6), and the mirror text
    gives no value for lambda ("determined by four-fold cross-validation").

    ``aux_reduction="sum"`` is the paper's sum over every non-root subsystem.
    ``aux_reduction="mean"`` divides that sum by the number of counted subsystems,
    an effective alpha of 0.3 / (T - 1). There is no default: real runs exist under
    both regimes. Neither value reproduces the loss of the 005/006 runs made before
    2026-09-30 through ``torchcell.trainers.int_dcell``: those passed ``[B, 1]``
    predictions and targets against ``[B]`` heads, so each auxiliary MSE broadcast
    over a ``[B, B]`` grid and the root head (``GO:<root index>``) was counted as an
    auxiliary term as well (issue #578 records the affected runs).

    The root is recognized by its DECLARED key, never by value or object identity:
    ``outputs["root_key"]`` (``DCell`` and ``DCellOpt`` set it to ``GO:<root index>``)
    and the alias ``"GO:ROOT"``. A non-root head whose values equal the root is
    counted. Shapes must match the target exactly; nothing is broadcast.

    Args:
        alpha: Weight for auxiliary losses (default: 0.3, the paper's value).
        use_auxiliary_losses: Whether to use losses from non-root subsystems
            (default: True).
        aux_reduction: ``"sum"`` (the paper) or ``"mean"`` over the non-root
            subsystem MSEs; required, keyword-only.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        use_auxiliary_losses: bool = True,
        *,
        aux_reduction: Literal["sum", "mean"],
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

    @staticmethod
    def _check_shape(name: str, output: torch.Tensor, target: torch.Tensor) -> None:
        """Refuse an output whose shape differs from the target's (no broadcast)."""
        if output.shape != target.shape:
            raise ValueError(
                f"DCellLoss: {name} has shape {tuple(output.shape)} but the target "
                f"has shape {tuple(target.shape)}; shapes must match exactly"
            )

    def forward(
        self, predictions: torch.Tensor, outputs: dict[str, Any], target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute the loss for DCell outputs.

        Args:
            predictions: Primary predictions tensor from the model (root output)
            outputs: Model outputs; ``linear_outputs`` maps term keys to head
                outputs and ``root_key`` names the root's own key
            target: Target values to predict, the same shape as ``predictions``

        Returns:
            Tuple of (total_loss, loss_components) where loss_components is a dictionary
            containing the primary_loss, auxiliary_loss, and weighted_auxiliary_loss.

        Raises:
            ValueError: If a prediction or counted head differs in shape from the
                target, or ``linear_outputs`` is given without ``root_key``.
        """
        self._check_shape("predictions", predictions, target)
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
        if linear_outputs and "root_key" not in outputs:
            raise ValueError(
                "DCellLoss: outputs has 'linear_outputs' but no 'root_key'; the "
                "model must declare which head is the root"
            )

        # The root is skipped by its declared key and the GO:ROOT alias only
        root_keys = {"GO:ROOT", outputs.get("root_key")}
        auxiliary_losses = []
        for subsystem_name, subsystem_output in linear_outputs.items():
            if subsystem_name in root_keys:
                continue
            self._check_shape(subsystem_name, subsystem_output, target)
            auxiliary_losses.append(self.criterion(subsystem_output, target))

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
