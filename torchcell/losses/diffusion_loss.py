# torchcell/losses/diffusion_loss.py
# [[torchcell.losses.diffusion_loss]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/losses/diffusion_loss
# Test file: tests/torchcell/losses/test_diffusion_loss.py

"""Strict diffusion loss with optional explicit x0 supervision."""

from typing import TYPE_CHECKING, Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:

    class _DiffusionDecoder(Protocol):
        """Structural interface for the model's diffusion decoder."""

        num_timesteps: int

        def forward_diffusion(
            self, targets: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]: ...

        def denoise(
            self,
            x_t: torch.Tensor,
            context: torch.Tensor,
            t: torch.Tensor,
            *,
            predict_x0: bool,
        ) -> torch.Tensor: ...

    class _DiffusionModel(Protocol):
        """Structural interface required of the wrapped diffusion model."""

        diffusion_decoder: "_DiffusionDecoder"

        def compute_diffusion_loss(
            self, targets: torch.Tensor, context: torch.Tensor, *, t_mode: str
        ) -> torch.Tensor: ...


class DiffusionLoss(nn.Module):
    """Strict diffusion loss for conditional decoders.

    What it does:
      • computes the model's diffusion loss
      • (optionally) adds explicit x0 supervision at sampled timesteps

    No auxiliary terms, no silent fallbacks.
    """

    # Typed view of the wrapped model (still an ``nn.Module`` submodule at
    # runtime; the structural type only documents the hooks used below).
    model: "_DiffusionModel"

    def __init__(
        self,
        model: nn.Module,
        *,
        lambda_diffusion: float = 1.0,
        lambda_x0: float = 0.0,
        x0_loss: nn.Module | None = None,
    ) -> None:
        """Configure loss weights and validate the model exposes diffusion hooks."""
        super().__init__()
        if lambda_diffusion <= 0.0 and lambda_x0 <= 0.0:
            raise ValueError("At least one of {lambda_diffusion, lambda_x0} > 0.")
        # cast is a no-op at runtime; the assignment still registers ``model`` as
        # an nn.Module submodule. The structural type enables the hook calls below.
        self.model = cast("_DiffusionModel", model)
        self.lambda_diffusion = float(lambda_diffusion)
        self.lambda_x0 = float(lambda_x0)
        self.x0_loss = x0_loss  # defaults to MSE if None

        # Hard checks: we want failures to be loud, not hidden.
        if not hasattr(model, "compute_diffusion_loss"):
            raise AttributeError("model must implement compute_diffusion_loss")
        if not hasattr(model, "diffusion_decoder"):
            raise AttributeError("model must have diffusion_decoder")

    def _sample_t(self, B: int, device: torch.device, t_mode: str) -> torch.Tensor:
        T = int(self.model.diffusion_decoder.num_timesteps)
        if t_mode == "zero":
            return torch.zeros(B, dtype=torch.long, device=device)
        if t_mode == "partial":
            hi = max(1, T // 10)
            return torch.randint(0, hi, (B,), device=device)
        return torch.randint(0, T, (B,), device=device)  # "full" / default

    def forward(
        self,
        predictions: torch.Tensor,  # kept for API compatibility; unused
        targets: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        epoch: int | None = None,  # kept for API comp.
        t_mode: str = "full",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the weighted diffusion (and optional x0) loss and a breakdown."""
        if context is None:
            raise ValueError("context (conditioning) tensor is required.")

        loss_dict: dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=targets.device, dtype=targets.dtype)

        if self.lambda_diffusion > 0.0:
            diff = self.model.compute_diffusion_loss(targets, context, t_mode=t_mode)
            loss_dict["diffusion_loss"] = diff
            total = total + self.lambda_diffusion * diff

        if self.lambda_x0 > 0.0:
            B = targets.shape[0]
            t = self._sample_t(B, targets.device, t_mode)
            noise = torch.randn_like(targets)
            x_t, _ = self.model.diffusion_decoder.forward_diffusion(targets, t, noise)
            x0_hat = self.model.diffusion_decoder.denoise(
                x_t, context, t, predict_x0=True
            )
            x0_term = (
                self.x0_loss(x0_hat, targets)
                if self.x0_loss is not None
                else F.mse_loss(x0_hat, targets)
            )
            loss_dict["x0_mse"] = x0_term
            total = total + self.lambda_x0 * x0_term

        loss_dict["total_loss"] = total
        return total, loss_dict
