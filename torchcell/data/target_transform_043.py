# torchcell/data/target_transform_043.py
# [[torchcell.data.target_transform_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/target_transform_043.py
# Test file: tests/torchcell/data/test_target_transform_043.py
"""Target preprocessing of the 043 trainer beyond the per-dataset z-score (2026-10-24).

The legacy pipeline maps a released value ``y`` to ``z = (y - c_g - mu_d) / sd_d`` with
the statistics of its dataset ``d`` fitted on the training rows
(:class:`torchcell.losses.hl_gauss.StandardizationStats`). Everything here is a SECOND,
monotone stage on top of that ``z``, so the legacy metrics (computed against ``z``) stay
comparable across preprocessing choices: the trainer learns ``t = f_g(u)`` with
``u = mu_d + sd_d * z = y - c_g`` (the released value after the per-gene center of a
panel dataset) and maps every prediction back with ``f_g^-1`` before scoring.

Kinds (``data.target_transform``):

``zscore``
    ``t = (u - mu_g) / sd_g`` over the group's pooled training ``u``. Under
    ``by: dataset`` this is the legacy ``z`` and the trainer skips this module entirely
    (bit-identical); under ``by: profile`` the datasets sharing a phenotype profile share
    one standardization.
``quantile_normal``
    ``t = Phi^-1(F_g(u))``: ``F_g`` is the group's empirical CDF of the pooled TRAINING
    ``u`` on a grid of ``J = 2001`` quantile levels ``(j + 0.5) / J``, linear between grid
    points, tied grid values merged to their mean level; outside the training range it
    clips to the end levels, so ``|t| <= Phi^-1(1 - 0.5 / J) = 3.48``.
``none``
    ``t = u``, the released (log) scale after the per-gene center; no group statistics.

Groups (``data.target_stats_by``): ``dataset`` (legacy; Sameith Dm reads as Kemmeren) or
``profile`` = the record's experiment-ontology PHENOTYPE profile: the sorted names of
the annotated vocabulary nodes whose ancestors include ``Phenotype`` (the phenotype leaf
class, its technology parents and closed value nodes such as
``SingleCellExpressionPhenotype.measurement_type:scrnaseq_umi_count``). The vocabulary
has no phenotype unit nodes, so units do not enter the key. No dataset name is read.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from scipy.special import ndtr, ndtri

from torchcell.data.experiment_ontology import ExperimentOntologyVocabulary

TransformKind = Literal["zscore", "quantile_normal", "none"]
StatsBy = Literal["dataset", "profile"]
PHENOTYPE_ROOT = "Phenotype"
QUANTILE_GRID = 2001


def phenotype_lane(vocab: ExperimentOntologyVocabulary) -> npt.NDArray[np.bool_]:
    """``[n_nodes]``: True for nodes with ``Phenotype`` among their ancestors."""
    parents = {n.name: n.parents for n in vocab.nodes}

    def ancestors(name: str) -> set[str]:
        out: set[str] = set()
        stack = [name]
        while stack:
            for p in parents.get(stack.pop(), []):
                if p not in out:
                    out.add(p)
                    stack.append(p)
        return out

    return np.asarray([PHENOTYPE_ROOT in ancestors(n.name) for n in vocab.nodes], bool)


def profile_keys(
    vocab: ExperimentOntologyVocabulary, annotations: npt.NDArray[np.floating]
) -> list[str]:
    """The phenotype profile key of each annotation row ``[n_profiles, n_nodes]``."""
    lane = phenotype_lane(vocab)
    names = vocab.names
    keys = []
    for row in np.asarray(annotations):
        active = np.nonzero((row != 0) & lane)[0]
        if active.size == 0:
            raise ValueError("an annotation with no phenotype node has no profile")
        keys.append("|".join(sorted(names[i] for i in active)))
    return keys


class GroupTargetMap(BaseModel):
    """``t = f_g(u)`` of one group, fitted on its training values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group: str
    kind: TransformKind
    mu: float = 0.0
    sd: float = Field(default=1.0, gt=0.0)
    grid_u: list[float] = Field(default_factory=list)
    grid_p: list[float] = Field(default_factory=list)
    n_train_values: int = Field(ge=0)

    def forward(self, u: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        """``u`` -> ``t`` (NaN stays NaN)."""
        x: npt.NDArray[np.float64] = np.asarray(u, np.float64)
        if self.kind == "none":
            return x
        if self.kind == "zscore":
            return (x - self.mu) / self.sd
        p = np.interp(x, self.grid_u, self.grid_p)
        return np.where(np.isfinite(x), ndtri(p), np.nan)

    def inverse(self, t: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        """``t`` -> ``u``."""
        x: npt.NDArray[np.float64] = np.asarray(t, np.float64)
        if self.kind == "none":
            return x
        if self.kind == "zscore":
            return self.mu + self.sd * x
        out = np.interp(ndtr(x), self.grid_p, self.grid_u)
        return np.where(np.isfinite(x), out, np.nan)


def fit_group_map(
    group: str, kind: TransformKind, u_train: npt.NDArray[np.floating]
) -> GroupTargetMap:
    """Fit ``f_g`` on the finite training ``u`` of one group."""
    v = np.asarray(u_train, np.float64).ravel()
    v = v[np.isfinite(v)]
    if kind == "none":
        return GroupTargetMap(group=group, kind=kind, n_train_values=int(v.size))
    if v.size < 2:
        raise ValueError(f"{group}: {v.size} finite training values")
    if kind == "zscore":
        sd = float(v.std())
        if not sd > 0.0:
            raise ValueError(f"{group}: zero spread over the training values")
        return GroupTargetMap(
            group=group, kind=kind, mu=float(v.mean()), sd=sd, n_train_values=v.size
        )
    levels = (np.arange(QUANTILE_GRID) + 0.5) / QUANTILE_GRID
    grid = np.quantile(v, levels)
    uniq, inv = np.unique(grid, return_inverse=True)
    p = np.bincount(inv, weights=levels) / np.bincount(inv)
    if uniq.size < 2:
        raise ValueError(f"{group}: a constant training distribution")
    return GroupTargetMap(
        group=group,
        kind=kind,
        grid_u=uniq.tolist(),
        grid_p=p.tolist(),
        n_train_values=int(v.size),
    )


class TargetTransform043(BaseModel):
    """Every group's map, in group-id order, pinned by sha256."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: TransformKind
    by: StatsBy
    groups: list[GroupTargetMap]

    def canonical_json(self) -> str:
        """Sorted-key JSON, the bytes the sha256 is taken over."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True)

    def sha256(self) -> str:
        """sha256 of :meth:`canonical_json`."""
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def by_name(self) -> dict[str, GroupTargetMap]:
        """Group name -> map."""
        return {g.group: g for g in self.groups}

    def _rows(
        self, x: npt.NDArray[np.floating], group: npt.NDArray[np.integer], forward: bool
    ) -> npt.NDArray[np.float64]:
        out = np.full(np.shape(x), np.nan, np.float64)
        for g in np.unique(group):
            sel = group == g
            m = self.groups[int(g)]
            out[sel] = m.forward(x[sel]) if forward else m.inverse(x[sel])
        return out

    def z_to_t(
        self,
        z: npt.NDArray[np.floating],
        group: npt.NDArray[np.integer],
        mu_d: npt.NDArray[np.floating],
        sd_d: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.float32]:
        """Rows of legacy ``z`` (first axis = row; ``group``/``mu_d``/``sd_d`` per row)."""
        z = np.asarray(z, np.float64)
        shape = (-1,) + (1,) * (z.ndim - 1)
        u = np.asarray(mu_d).reshape(shape) + np.asarray(sd_d).reshape(shape) * z
        return self._rows(u, np.asarray(group), forward=True).astype(np.float32)

    def t_to_z(
        self,
        t: npt.NDArray[np.floating],
        group: npt.NDArray[np.integer],
        mu_d: npt.NDArray[np.floating],
        sd_d: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.float32]:
        """Inverse of :meth:`z_to_t` (predictions back to legacy ``z`` for scoring)."""
        t = np.asarray(t, np.float64)
        shape = (-1,) + (1,) * (t.ndim - 1)
        u = self._rows(t, np.asarray(group), forward=False)
        mu = np.asarray(mu_d).reshape(shape)
        sd = np.asarray(sd_d).reshape(shape)
        return ((u - mu) / sd).astype(np.float32)
