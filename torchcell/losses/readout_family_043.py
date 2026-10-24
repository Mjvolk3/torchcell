# torchcell/losses/readout_family_043.py
# [[torchcell.losses.readout_family_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/losses/readout_family_043.py
# Test file: tests/torchcell/losses/test_readout_family_043.py
"""The 043 loss family beyond HL-Gauss / pinball / MSE (GilaHyper sweep, 2026-10-24).

``ReadoutLossFamily043`` extends :class:`torchcell.models.cell_model_043.ReadoutLoss043`
with two kinds that need to know which normalization GROUP a row belongs to (a dataset
under ``by: dataset``, an experiment-ontology phenotype profile under ``by: profile``).
For the three legacy kinds it calls the parent unchanged, so a legacy run is the same
computation.

``mle``
    Heteroscedastic Gaussian negative log-likelihood. The head emits ``[..., 2]``: a mean
    ``m`` and a log-variance ``s``. ``s`` is CLAMPED to ``[-7, 7]``
    (``MLE_LOGVAR_CLAMP``, a variance between 9e-4 and 1.1e3 in target units) before use,
    so a collapsing variance cannot drive the loss to minus infinity and an exploding one
    cannot flatten the gradient of the mean. The loss is the full NLL,
    ``0.5 * (log 2 pi + s + (y - m)^2 exp(-s))``, so its value is a true per-entry NLL.
    The point prediction is ``m``.

``binned_quantile``
    HL-Gauss over EQUAL-MASS bins: per group, ``K`` bins whose edges are the quantiles
    ``k / K`` of that group's TRAINING targets (:func:`fit_quantile_bins`; the validation
    and test rows never enter). A target ``y`` is mapped to its position ``u`` in
    ``[0, 1]`` by linear interpolation inside its bin (``u = (k + frac) / K``), and the
    soft target is the HL-Gauss histogram of ``u`` over ``K`` uniform bins on ``[0, 1]``
    with the same ``sigma_bins`` (smoothing in bin-index units, the equal-mass analogue of
    HL-Gauss's bin-width smoothing; open edge bins as in HL-Gauss). A target beyond the
    training range clips to the edge bin. The point prediction is the EXPECTATION
    ``sum_k softmax_k * c_k`` with ``c_k`` the mean of the group's training targets that
    fall in bin ``k`` (a bin emptied by tied values takes its edge value).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from torchcell.losses.hl_gauss import HLGaussConfig, HLGaussLoss
from torchcell.models.cell_model_043 import CellModel043Config, ReadoutLoss043

MLE_LOGVAR_CLAMP = 7.0
_LOG_2PI = math.log(2.0 * math.pi)


class QuantileBins(BaseModel):
    """Equal-mass bin edges and centers of one group, fitted on its training targets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group: str
    edges: list[float] = Field(
        description="K + 1 edges: train min, K - 1 quantiles, max"
    )
    centers: list[float] = Field(description="K per-bin means of the training targets")
    n_train_values: int = Field(ge=2)

    @model_validator(mode="after")
    def _shapes(self) -> QuantileBins:
        """``len(edges) == len(centers) + 1``, edges non-decreasing."""
        if len(self.edges) != len(self.centers) + 1:
            raise ValueError("need K + 1 edges for K centers")
        if any(b < a for a, b in zip(self.edges, self.edges[1:], strict=False)):
            raise ValueError(f"{self.group}: edges must be non-decreasing")
        return self


def fit_quantile_bins(
    group: str, values: npt.NDArray[np.floating], num_bins: int
) -> QuantileBins:
    """``K`` equal-mass bins over the finite training ``values`` of one group."""
    v = np.asarray(values, np.float64).ravel()
    v = np.sort(v[np.isfinite(v)])
    if v.size < 2:
        raise ValueError(f"{group}: {v.size} finite training values for quantile bins")
    edges = np.quantile(v, np.arange(num_bins + 1) / num_bins)
    k = np.clip(np.searchsorted(edges[1:-1], v, side="right"), 0, num_bins - 1)
    sums = np.bincount(k, weights=v, minlength=num_bins)
    counts = np.bincount(k, minlength=num_bins)
    centers = np.where(counts > 0, sums / np.maximum(counts, 1), edges[:-1])
    return QuantileBins(
        group=group,
        edges=edges.tolist(),
        centers=centers.tolist(),
        n_train_values=int(v.size),
    )


class ReadoutLossFamily043(ReadoutLoss043):
    """``ReadoutLoss043`` plus ``mle`` and ``binned_quantile`` (module docstring).

    ``groups`` is a ``[rows]`` long tensor, the normalization group of each row of the
    target (the first dimension); required by ``binned_quantile``, ignored otherwise.
    """

    bin_edges: torch.Tensor
    bin_centers: torch.Tensor

    def __init__(
        self, config: CellModel043Config, bins: Sequence[QuantileBins] | None = None
    ) -> None:
        """``bins`` (one per group, in group-id order) exactly when binned_quantile."""
        super().__init__(config)
        if (config.loss == "binned_quantile") != (bins is not None):
            raise ValueError("quantile bins are required exactly for binned_quantile")
        k = config.hl_gauss.num_bins
        if bins is not None:
            bad = [b.group for b in bins if len(b.centers) != k]
            if bad:
                raise ValueError(f"groups {bad} have bin counts other than K = {k}")
            self.register_buffer(
                "bin_edges", torch.tensor([b.edges for b in bins], dtype=torch.float32)
            )
            self.register_buffer(
                "bin_centers",
                torch.tensor([b.centers for b in bins], dtype=torch.float32),
            )
            self.unit_hl = HLGaussLoss(
                HLGaussConfig(
                    num_bins=k, low=0.0, high=1.0, sigma_bins=config.hl_gauss.sigma_bins
                )
            )
            self.group_names = [b.group for b in bins]

    # ---- binned_quantile helpers -------------------------------------------------
    def _row_table(
        self, table: torch.Tensor, groups: torch.Tensor | None, target_ndim: int
    ) -> torch.Tensor:
        """``table[groups]`` shaped to broadcast against a ``target_ndim`` target."""
        if groups is None:
            raise ValueError("binned_quantile needs the group of every row")
        rows = table[groups.to(table.device)]  # [R, W]
        return rows.view(rows.shape[0], *([1] * (target_ndim - 1)), rows.shape[-1])

    def unit_position(
        self, target: torch.Tensor, groups: torch.Tensor | None
    ) -> torch.Tensor:
        """``u`` in ``[0, 1]`` of each target in its group's bins (NaN stays NaN)."""
        edges = self._row_table(self.bin_edges, groups, target.dim())
        k_bins = edges.shape[-1] - 1
        measured = ~torch.isnan(target)
        safe = torch.where(measured, target, torch.zeros_like(target)).float()
        t = safe.unsqueeze(-1)
        k = (t >= edges[..., 1:-1]).sum(-1).clamp_max(k_bins - 1)  # [...]
        edges_b = edges.expand(*safe.shape, k_bins + 1)
        lo = torch.gather(edges_b, -1, k.unsqueeze(-1)).squeeze(-1)
        hi = torch.gather(edges_b, -1, (k + 1).unsqueeze(-1)).squeeze(-1)
        frac = ((safe - lo) / (hi - lo).clamp_min(1e-12)).clamp(0.0, 1.0)
        u = (k.to(frac.dtype) + frac) / k_bins
        return torch.where(measured, u, torch.full_like(u, float("nan")))

    # ---- the objective -------------------------------------------------------------
    def elementwise(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        groups: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Loss per entry (zero where missing) and the measured mask, ``target.shape``."""
        if self.kind == "binned_quantile":
            return self.unit_hl.elementwise(pred, self.unit_position(target, groups))
        if self.kind != "mle":
            return super().elementwise(pred, target)
        if tuple(pred.shape) != (*target.shape, 2):
            raise ValueError(
                f"mle needs pred {(*target.shape, 2)}, got {tuple(pred.shape)}"
            )
        measured = ~torch.isnan(target)
        safe = torch.where(measured, target, torch.zeros_like(target)).float()
        mean = pred[..., 0].float()
        logvar = pred[..., 1].float().clamp(-MLE_LOGVAR_CLAMP, MLE_LOGVAR_CLAMP)
        elem = 0.5 * (_LOG_2PI + logvar + (safe - mean) ** 2 * torch.exp(-logvar))
        return elem * measured.to(elem.dtype), measured

    def point(
        self, pred: torch.Tensor, groups: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Expectation (binned), the mean (mse, mle), or the median quantile (pinball)."""
        if self.kind == "mle":
            return pred[..., 0].float()
        if self.kind == "binned_quantile":
            centers = self._row_table(self.bin_centers, groups, pred.dim() - 1)
            return (torch.softmax(pred.float(), dim=-1) * centers).sum(-1)
        return super().point(pred)

    def log_variance(self, pred: torch.Tensor) -> torch.Tensor:
        """Mle only: the clamped log-variance head, ``[...]``."""
        if self.kind != "mle":
            raise ValueError("log_variance is the mle head")
        return pred[..., 1].float().clamp(-MLE_LOGVAR_CLAMP, MLE_LOGVAR_CLAMP)
