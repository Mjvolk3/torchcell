# torchcell/losses/latent_align.py
# [[torchcell.losses.latent_align]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/losses/latent_align
"""Latent alignment: shape a strain's latent vector by the similarity of its labels.

The inductive morphology rounds (019, v24 and v25) sit on a plateau where every strain
gets the same readout vector and the head predicts the per-feature marginal. The data
loss is nearly flat there (the 19-quantile pinball moves 0.195 to 0.187 across an escape
that lifts the training Pearson from 0.06 to 0.34), so nothing pushes the latents apart.
This auxiliary term does: for every pair of supervised strains in a batch it asks that
the cosine similarity of their latent vectors match the Pearson correlation of their
label vectors (the 116 CalMorph features, or the measured proteins). Two strains with
alike phenotypes should sit close, two with opposite phenotypes should point apart. The
marginal solution, all latents equal, scores a cosine of one for every pair against a
spread of label correlations, so the term penalizes the plateau from the first step.

The label correlation is computed over the features both strains have measured, so a
sparse vector target (NaN where unmeasured) needs no imputation; a pair with fewer than
``min_overlap`` shared features is left out.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from torchcell.models.equivariant_cell_graph_transformer import pooled_perturbed

# Which [batch, d] vector the loss shapes. ``cls`` is the perturbed CLS (needs
# ``model.perturb_cls``), ``zs`` the pool over the strain's perturbed gene tokens (the
# PerturbationHead input), ``cls_zs`` their concatenation, the input of the M_zs head.
SOURCES = ("cls", "zs", "cls_zs")
EPS = 1e-8


def masked_pairwise_pearson(
    targets: torch.Tensor, min_overlap: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pearson correlation between every pair of rows over their shared finite features.

    Args:
        targets: [n, F] label vectors; NaN marks an unmeasured feature.
        min_overlap: a pair with fewer shared finite features than this is invalid.

    Returns:
        r: [n, n] correlations (0 where invalid), valid: [n, n] bool.
    """
    finite = torch.isfinite(targets)
    t = torch.where(finite, targets, torch.zeros_like(targets))
    m = finite.to(t.dtype)
    n = m @ m.T
    n_safe = n.clamp_min(1.0)
    # Sums over the features finite in BOTH rows: t_i is zero where i is unmeasured and
    # m_j masks the features j has not measured.
    sx = t @ m.T
    sxx = (t * t) @ m.T
    sxy = t @ t.T
    mx = sx / n_safe
    my = mx.T
    cov = sxy / n_safe - mx * my
    vx = (sxx / n_safe - mx * mx).clamp_min(0.0)
    vy = vx.T
    valid = (n >= min_overlap) & (vx > EPS) & (vy > EPS)
    r = cov / torch.sqrt(vx * vy).clamp_min(EPS)
    return torch.where(valid, r, torch.zeros_like(r)), valid


def latent_align_loss(
    latent: torch.Tensor,
    targets: torch.Tensor,
    row_mask: torch.Tensor,
    *,
    min_overlap: int = 10,
    center: bool = True,
) -> tuple[torch.Tensor, int]:
    """Mean squared gap between latent cosine similarity and label correlation over pairs.

    Args:
        latent: [B, d] one vector per strain in the batch.
        targets: [B, F] the head's (normalized) label vectors, NaN where unmeasured.
        row_mask: [B] bool, True for the strains that carry this head's label.
        min_overlap: minimum shared measured features for a pair to count.
        center: subtract the batch mean of the supervised latents before the cosine, so
            a shared offset common to every strain does not set the similarity.

    Returns:
        loss: a scalar connected to ``latent`` (zero, unconnected, when no pair counts);
        n_pairs: the number of pairs the mean runs over.
    """
    rows = row_mask.nonzero(as_tuple=True)[0]
    if rows.numel() < 2:
        return latent.new_zeros(()), 0
    u = latent[rows]
    if center:
        u = u - u.mean(dim=0, keepdim=True)
    u = F.normalize(u, dim=-1, eps=EPS)
    cos = u @ u.T
    r, valid = masked_pairwise_pearson(targets[rows].detach(), min_overlap)
    iu = torch.triu_indices(rows.numel(), rows.numel(), offset=1, device=latent.device)
    keep = valid[iu[0], iu[1]]
    n_pairs = int(keep.sum().item())
    if n_pairs == 0:
        return latent.new_zeros(()), 0
    gap = cos[iu[0], iu[1]][keep] - r[iu[0], iu[1]][keep]
    return (gap * gap).mean(), n_pairs


def strain_latent(
    reps: dict[str, Any],
    perturbation_indices: torch.Tensor,
    perturbation_indices_batch: torch.Tensor,
    source: str,
    pooling: str,
) -> torch.Tensor:
    """The [batch, d] strain vector the alignment loss shapes, read from the model's reps.

    ``cls`` is ``reps["h_CLS_pert"]`` and raises without ``model.perturb_cls`` (the shared
    wild-type CLS carries no strain information to align); ``zs`` pools the perturbed gene
    tokens with the PerturbationHead's ``pooling``; ``cls_zs`` concatenates the two.
    """
    if source not in SOURCES:
        raise ValueError(f"latent_align.source={source!r}, expected one of {SOURCES}")
    parts: list[torch.Tensor] = []
    if source in ("cls", "cls_zs"):
        h = reps["h_CLS_pert"]
        if h is None:
            raise ValueError(
                "latent_align.source 'cls' reads the perturbed CLS, which needs "
                "model.perturb_cls=true"
            )
        parts.append(h)
    if source in ("zs", "cls_zs"):
        parts.append(
            pooled_perturbed(
                reps["H_genes_pert"],
                perturbation_indices,
                perturbation_indices_batch,
                pooling,
            )
        )
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
