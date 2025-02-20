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
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        z_P: torch.Tensor,
        z_I: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        mse_loss, mse_dim_losses = self.mse_loss_fn(predictions, targets)
        dist_loss, dist_dim_losses = self.dist_loss_fn(predictions, targets)
        supcr_loss, supcr_dim_losses = self.supcr_fn(z_P, z_I, targets)

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
