# torchcell/losses/isomorphic_cell_loss
# [[torchcell.losses.isomorphic_cell_loss]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/losses/isomorphic_cell_loss
# Test file: tests/torchcell/losses/test_isomorphic_cell_loss.py


from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchcell.losses.multi_dim_nan_tolerant import (
    WeightedMSELoss,
    WeightedDistLoss,
    WeightedSupCRCell,
)


class ICLoss(nn.Module):
    def __init__(
        self,
        lambda_dist: float,
        lambda_supcr: float,
        lambda_cell: float,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:
            lambda_dist: Weight for the dist loss.
            lambda_supcr: Weight for the SupCR loss.
            lambda_cell: Weight for the cell loss.
            weights: Optional weights applied uniformly to all losses.
        """
        super().__init__()
        self.lambda_dist = lambda_dist
        self.lambda_supcr = lambda_supcr
        self.lambda_cell = lambda_cell

        self.mse_loss_fn = WeightedMSELoss(weights=weights)
        self.dist_loss_fn = WeightedDistLoss(weights=weights)
        self.supcr_fn = WeightedSupCRCell(weights=weights)
        # Use a separate instance of WeightedMSELoss for the cell loss
        # TODO mask them
        self.cell_loss_fn = WeightedMSELoss(weights=None)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        z_W: torch.Tensor,
        z_P: torch.Tensor,
        z_I: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        mse_loss, mse_dim_losses = self.mse_loss_fn(predictions, targets)
        dist_loss, dist_dim_losses = self.dist_loss_fn(predictions, targets)
        supcr_loss, supcr_dim_losses = self.supcr_fn(z_P, z_I, targets)
        cell_loss, cell_dim_losses = self.cell_loss_fn(z_I, z_W + z_P)

        total_loss = (
            mse_loss
            + self.lambda_dist * dist_loss
            + self.lambda_supcr * supcr_loss
            + self.lambda_cell * cell_loss
        )

        loss_dict = {
            "mse_loss": mse_loss,
            "mse_dim_losses": mse_dim_losses,
            "dist_loss": dist_loss,
            "dist_dim_losses": dist_dim_losses,
            "supcr_loss": supcr_loss,
            "supcr_dim_losses": supcr_dim_losses,
            "cell_loss": cell_loss,
            "cell_dim_losses": cell_dim_losses,
            "total_loss": total_loss,
        }
        return total_loss, loss_dict
