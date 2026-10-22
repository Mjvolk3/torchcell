# torchcell/losses/hl_gauss.py
# [[torchcell.losses.hl_gauss]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/losses/hl_gauss.py
# Test file: tests/torchcell/losses/test_hl_gauss.py
"""HL-Gauss binned regression and the per-dataset target standardization of 043.

Design: ``notes-tex/multimodal/043-datasets-primer/sections/output-design.tex``,
"Losses" (decision O4). Two pieces:

``HLGaussLoss``
    The histogram loss with a Gaussian target (Farebrother et al. 2024). A standardized
    target ``z`` becomes a soft target over ``K`` bins on ``[low, high]``: the mass a
    normal ``N(z, sigma^2)`` puts in each bin, with the two edge bins OPEN (the first bin
    integrates from ``-inf``, the last to ``+inf``), so the target sums to one exactly
    and a target outside ``[low, high]`` still has a well-defined histogram. The loss is
    the cross-entropy between that target and ``softmax(logits)``. The point prediction
    is the expectation over the bin centers (the open edge bins use their nominal
    midpoints), the uncertainty is the entropy of the predicted histogram, and the
    classification reading is accuracy within one bin.

    Missing entries are ``NaN`` in the target. The soft target is built from a ZERO-FILLED
    safe target and the elementwise loss is multiplied by the measured mask, so no ``NaN``
    reaches the forward value or the backward pass. Any leading shape works: ``[B, G, K]``
    per-gene logits against ``[B, G]`` targets, ``[B, K]`` scalars against ``[B]``.

``StandardizationStats``
    Per dataset, the statistics that map a transformed target ``y`` to ``z``: an optional
    per-gene center fitted over the training rows (the panel datasets, centered per gene
    over their training isolates or segregants), then a pooled mean and standard
    deviation over every training value of the dataset, ``z = (y - c_g - mu) / s``. A
    gene with no training value has no center and its ``z`` is ``NaN`` (no target), never
    a substituted zero. The record is pydantic and pinned by the sha256 of its canonical
    JSON, so a run logs exactly which statistics it trained against.

Defaults are the decision table's: ``K = 32`` bins over ``[-4, 4]`` z (width 0.25),
``sigma = 0.75`` bin widths.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field, model_validator


class HLGaussConfig(BaseModel):
    """Bins and smoothing of the HL-Gauss loss (decision O4)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    num_bins: int = Field(default=32, ge=2)
    low: float = -4.0
    high: float = 4.0
    sigma_bins: float = Field(
        default=0.75, gt=0.0, description="Gaussian width in bin widths"
    )

    @model_validator(mode="after")
    def _ordered(self) -> HLGaussConfig:
        """``low < high``."""
        if not self.low < self.high:
            raise ValueError(f"need low < high, got {self.low} .. {self.high}")
        return self

    @property
    def bin_width(self) -> float:
        """Width of one bin in z units."""
        return (self.high - self.low) / self.num_bins

    @property
    def sigma(self) -> float:
        """Gaussian width in z units."""
        return self.sigma_bins * self.bin_width


class HLGaussLoss(nn.Module):
    """Cross-entropy against a Gaussian-smoothed histogram target, with readouts."""

    edges: torch.Tensor
    centers: torch.Tensor

    def __init__(self, config: HLGaussConfig | None = None) -> None:
        """Register the interior bin edges and the bin centers."""
        super().__init__()
        self.config = config or HLGaussConfig()
        k = self.config.num_bins
        edges = torch.linspace(self.config.low, self.config.high, k + 1)
        self.register_buffer("edges", edges)
        self.register_buffer("centers", 0.5 * (edges[:-1] + edges[1:]))

    @property
    def num_bins(self) -> int:
        """``K``."""
        return self.config.num_bins

    def soft_target(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(probs [..., K], measured [...])`` for a target with ``NaN`` = missing.

        ``probs`` of a missing entry is the histogram of a zero target (finite, and
        multiplied away by the mask downstream).
        """
        measured = ~torch.isnan(target)
        safe = torch.where(measured, target, torch.zeros_like(target)).float()
        inner = self.edges[1:-1].to(safe.device)
        sigma = self.config.sigma
        # CDF of N(safe, sigma^2) at the interior edges; the outer edges are -inf / +inf.
        cdf = 0.5 * (
            1.0 + torch.erf((inner - safe.unsqueeze(-1)) / (sigma * math.sqrt(2.0)))
        )
        zeros = torch.zeros_like(cdf[..., :1])
        ones = torch.ones_like(cdf[..., :1])
        full = torch.cat([zeros, cdf, ones], dim=-1)
        probs = (full[..., 1:] - full[..., :-1]).clamp_min(0.0)
        return probs, measured

    def elementwise(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-entropy per entry (zero where missing) and the measured mask.

        ``logits`` is ``target.shape + (K,)``.
        """
        if tuple(logits.shape) != (*target.shape, self.num_bins):
            raise ValueError(
                f"logits must be {(*target.shape, self.num_bins)}, "
                f"got {tuple(logits.shape)}"
            )
        probs, measured = self.soft_target(target)
        log_q = F.log_softmax(logits.float(), dim=-1)
        ce = -(probs * log_q).sum(dim=-1)
        return ce * measured.to(ce.dtype), measured

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy over the measured entries; a connected zero if none."""
        ce, measured = self.elementwise(logits, target)
        count = measured.sum()
        if int(count) == 0:
            return logits.sum() * 0.0
        return ce.sum() / count

    def expectation(self, logits: torch.Tensor) -> torch.Tensor:
        """Point prediction ``sum_k softmax_k * center_k`` in z units, shape ``[...]``."""
        q = torch.softmax(logits.float(), dim=-1)
        return (q * self.centers.to(q.device)).sum(dim=-1)

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Entropy of the predicted histogram (nats), shape ``[...]``."""
        log_q = F.log_softmax(logits.float(), dim=-1)
        return -(log_q.exp() * log_q).sum(dim=-1)

    def target_bin(self, target: torch.Tensor) -> torch.Tensor:
        """Bin index of each (finite) target; values outside clip to the edge bins."""
        inner = self.edges[1:-1].to(target.device).contiguous()
        return torch.bucketize(target.contiguous(), inner, right=False)

    def accuracy_within_one(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(hits, measured)``: argmax bin within one bin of the target's bin."""
        measured = ~torch.isnan(target)
        safe = torch.where(measured, target, torch.zeros_like(target))
        hit = (logits.argmax(dim=-1) - self.target_bin(safe)).abs() <= 1
        return hit & measured, measured


# --------------------------------------------------------------------------- #
# Per-dataset standardization
# --------------------------------------------------------------------------- #
def _finite_or_none(values: npt.NDArray[np.float64]) -> list[float | None]:
    return [float(v) if np.isfinite(v) else None for v in values.tolist()]


class DatasetStandardization(BaseModel):
    """``z = (y - center_g - mu) / sd`` for one dataset, fitted on its training rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    transform: str = Field(description="what preceded the standardization")
    center_per_gene: bool
    gene_center: list[float | None] | None = Field(
        default=None, description="per-gene center; None where no training value"
    )
    mu: float
    sd: float = Field(gt=0.0)
    n_train_records: int = Field(ge=1)
    n_train_values: int = Field(ge=2)

    @model_validator(mode="after")
    def _center_present(self) -> DatasetStandardization:
        """A per-gene center exactly when ``center_per_gene``."""
        if self.center_per_gene != (self.gene_center is not None):
            raise ValueError("gene_center must be given iff center_per_gene")
        return self

    def center_array(self, width: int) -> npt.NDArray[np.float64]:
        """The per-gene center as an array (NaN = no center), zeros when not centered."""
        if self.gene_center is None:
            return np.zeros(width, dtype=np.float64)
        if len(self.gene_center) != width:
            raise ValueError(
                f"{self.dataset}: center has {len(self.gene_center)} entries, not {width}"
            )
        return np.asarray(
            [np.nan if c is None else c for c in self.gene_center], dtype=np.float64
        )

    def apply(self, y: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
        """Standardize ``y`` (``[n, width]`` or ``[width]``); NaN stays NaN."""
        arr = np.asarray(y, dtype=np.float64)
        center = self.center_array(arr.shape[-1])
        return ((arr - center - self.mu) / self.sd).astype(np.float32)

    def invert(self, z: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        """Back to the transformed scale: ``mu + sd * z + center_g``."""
        arr = np.asarray(z, dtype=np.float64)
        return self.mu + self.sd * arr + self.center_array(arr.shape[-1])


def fit_standardization(
    dataset: str,
    y_train: npt.NDArray[np.floating],
    center_per_gene: bool,
    transform: str,
) -> DatasetStandardization:
    """Fit on the training rows ``y_train`` ``[n_records, width]`` (NaN = missing).

    With ``center_per_gene`` the center of a gene is its mean over the training rows
    that measure it; the pooled ``mu`` and ``sd`` are then taken over every centered
    training value (``mu`` is 0 up to rounding). A dataset with fewer than two finite
    training values, or a zero spread, raises.
    """
    arr = np.asarray(y_train, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{dataset}: y_train must be [n, width], got {arr.shape}")
    center: npt.NDArray[np.float64] | None = None
    if center_per_gene:
        with np.errstate(invalid="ignore"):
            counts = np.isfinite(arr).sum(axis=0)
            sums = np.nansum(arr, axis=0)
            center = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        arr = arr - center
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        raise ValueError(f"{dataset}: {finite.size} finite training values")
    mu = float(finite.mean())
    sd = float(finite.std())
    if not sd > 0.0:
        raise ValueError(f"{dataset}: zero spread over the training values")
    return DatasetStandardization(
        dataset=dataset,
        transform=transform,
        center_per_gene=center_per_gene,
        gene_center=None if center is None else _finite_or_none(center),
        mu=mu,
        sd=sd,
        n_train_records=int(arr.shape[0]),
        n_train_values=int(finite.size),
    )


class StandardizationStats(BaseModel):
    """The per-dataset standardizations of one run, pinned by sha256."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split_version: str
    split_store_sha256: str
    datasets: dict[str, DatasetStandardization]

    def canonical_json(self) -> str:
        """Sorted-key JSON, the bytes the sha256 is taken over."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True)

    def sha256(self) -> str:
        """sha256 of :meth:`canonical_json`."""
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def get(self, dataset: str) -> DatasetStandardization:
        """The statistics of one dataset; an unknown one raises."""
        if dataset not in self.datasets:
            raise KeyError(f"no standardization for {dataset}")
        return self.datasets[dataset]

    @classmethod
    def fit(
        cls,
        split_version: str,
        split_store_sha256: str,
        rows: Sequence[tuple[str, npt.NDArray[np.floating], bool, str]],
    ) -> StandardizationStats:
        """From ``(dataset, y_train, center_per_gene, transform)`` per dataset."""
        return cls(
            split_version=split_version,
            split_store_sha256=split_store_sha256,
            datasets={
                name: fit_standardization(name, y, center, transform)
                for name, y, center, transform in rows
            },
        )
