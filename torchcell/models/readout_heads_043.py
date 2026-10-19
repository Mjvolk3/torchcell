# torchcell/models/readout_heads_043.py
# [[torchcell.models.readout_heads_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/readout_heads_043
"""Output-side prototype for experiment 043: typed readout heads off the perturbed graph.

Four pieces, each importable on its own and configured by a pydantic model:

``ExperimentTypeEmbedding``
    One learned vector per ontology phenotype class (``MicroarrayExpressionPhenotype``,
    ``ProteinAbundancePhenotype``, ...). ``mode="flat"`` is one row per leaf class;
    ``mode="dag"`` sums one row per node of the class's ontology path (for example
    ``Phenotype > ExpressionPhenotype > MicroarrayExpressionPhenotype``), so two leaves
    under one parent share the parent's row. The same module serves the dataset token
    and the environment token (both flat).

``PerGeneReadout``
    The owner's "read off the node itself": a prediction for gene ``g`` is a function of
    that gene's own perturbed state ``H_g`` and the record's tokens, through ONE head
    shared by every dataset of a readout type, followed by a per-dataset affine
    ``a_k * f(.) + b_k`` (``a_k`` starts at 1, ``b_k`` at 0). The affine of dataset ``k``
    receives gradient only from rows of dataset ``k``.

``GlobalScalarReadout``
    One scalar per record from ``[h_CLS ; z_S ; tokens]`` (Vaishnav 2022's sort-seq level).

``MaskedTypedLoss``
    Per readout type, MSE or the pinball loss over a quantile grid, reduced over the
    MEASURED entries only. A missing entry is ``NaN`` in the target; it contributes no
    value and no gradient (the target is replaced by zero under the mask BEFORE the
    elementwise loss, so no ``NaN`` reaches the backward pass).

Nothing here falls back: a dataset id outside ``n_datasets``, a class name without a
path in ``dag`` mode, or a readout type without a configured loss raises.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field, model_validator

from torchcell.losses.distributional import DEFAULT_NUM_QUANTILES, pinball

OntologyMode = Literal["flat", "dag"]
LossKind = Literal["mse", "pinball"]


class ExperimentTypeEmbeddingConfig(BaseModel):
    """Which classes get a vector, and whether a class is one row or its ontology path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    classes: tuple[str, ...] = Field(min_length=1)
    hidden_dim: int = Field(gt=0)
    mode: OntologyMode = "flat"
    dag_paths: dict[str, tuple[str, ...]] | None = None

    @model_validator(mode="after")
    def _paths_cover_classes(self) -> ExperimentTypeEmbeddingConfig:
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("classes repeat a name")
        if self.mode == "flat":
            if self.dag_paths is not None:
                raise ValueError("dag_paths is only read in mode='dag'")
            return self
        if self.dag_paths is None:
            raise ValueError(
                "mode='dag' needs dag_paths, one root-first path per class"
            )
        for name in self.classes:
            path = self.dag_paths.get(name)
            if path is None:
                raise ValueError(f"no ontology path for class {name!r}")
            if len(path) == 0 or path[-1] != name:
                raise ValueError(f"the path of {name!r} must end in the class itself")
        return self


class ExperimentTypeEmbedding(nn.Module):
    """A learned token per ontology class, flat or summed along the class's path."""

    path_index: torch.Tensor
    path_mask: torch.Tensor

    def __init__(self, config: ExperimentTypeEmbeddingConfig) -> None:
        """Build the node table and the per-class path index from ``config``."""
        super().__init__()
        self.config = config
        self.class_to_index = {name: i for i, name in enumerate(config.classes)}
        if config.mode == "flat":
            self.node_names: tuple[str, ...] = config.classes
            path_index = torch.arange(len(config.classes)).unsqueeze(1)
            path_mask = torch.ones(len(config.classes), 1, dtype=torch.bool)
        else:
            assert config.dag_paths is not None
            nodes: list[str] = []
            for name in config.classes:
                for node in config.dag_paths[name]:
                    if node not in nodes:
                        nodes.append(node)
            self.node_names = tuple(nodes)
            node_index = {node: i for i, node in enumerate(nodes)}
            depth = max(len(config.dag_paths[name]) for name in config.classes)
            path_index = torch.zeros(len(config.classes), depth, dtype=torch.long)
            path_mask = torch.zeros(len(config.classes), depth, dtype=torch.bool)
            for i, name in enumerate(config.classes):
                for j, node in enumerate(config.dag_paths[name]):
                    path_index[i, j] = node_index[node]
                    path_mask[i, j] = True
        self.register_buffer("path_index", path_index)
        self.register_buffer("path_mask", path_mask)
        self.table = nn.Embedding(len(self.node_names), config.hidden_dim)

    def node_index(self, node: str) -> int:
        """Row of ``node`` (a class or an ancestor) in the embedding table."""
        return self.node_names.index(node)

    def forward(self, class_ids: torch.Tensor) -> torch.Tensor:
        """``class_ids`` [B] of indices into ``config.classes`` -> [B, hidden_dim]."""
        if class_ids.dim() != 1:
            raise ValueError(f"class_ids must be [B], got {tuple(class_ids.shape)}")
        if bool((class_ids < 0).any()) or bool(
            (class_ids >= len(self.config.classes)).any()
        ):
            raise IndexError("class id outside the configured classes")
        rows = self.table(self.path_index[class_ids])  # [B, depth, d]
        mask = self.path_mask[class_ids].unsqueeze(-1).to(rows.dtype)
        return cast(torch.Tensor, (rows * mask).sum(dim=1))


class PerGeneReadoutConfig(BaseModel):
    """The shared head over a gene's own state, and the per-dataset affine on top."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hidden_dim: int = Field(gt=0)
    n_datasets: int = Field(gt=0)
    token_dim: int = Field(ge=0, default=0)
    head_hidden: int = Field(ge=0, default=0)
    param_dim: int = Field(gt=0, default=1)
    dropout: float = Field(ge=0.0, lt=1.0, default=0.0)
    per_dataset_affine: bool = True


class PerGeneReadout(nn.Module):
    r"""``y_{b,g} = a_{k(b)} f([H_{b,g} ; t_b]) + b_{k(b)}`` for every gene node ``g``.

    ``f`` is one affine map (``head_hidden == 0``) or a two-layer MLP shared by every
    dataset; ``k(b)`` is the dataset of record ``b``. With ``param_dim > 1`` the head
    emits that many values per gene (quantile knots) and the affine acts on all of them.
    """

    def __init__(self, config: PerGeneReadoutConfig) -> None:
        """Build the shared head and, when configured, the per-dataset affine."""
        super().__init__()
        self.config = config
        in_dim = config.hidden_dim + config.token_dim
        if config.head_hidden == 0:
            self.head: nn.Module = nn.Linear(in_dim, config.param_dim)
        else:
            self.head = nn.Sequential(
                nn.Linear(in_dim, config.head_hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.head_hidden, config.param_dim),
            )
        self.scale: nn.Parameter | None = None
        self.shift: nn.Parameter | None = None
        if config.per_dataset_affine:
            self.scale = nn.Parameter(torch.ones(config.n_datasets, config.param_dim))
            self.shift = nn.Parameter(torch.zeros(config.n_datasets, config.param_dim))

    def forward(
        self,
        H: torch.Tensor,
        dataset_ids: torch.Tensor,
        tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``H`` [B, N, d], ``dataset_ids`` [B], ``tokens`` [B, token_dim] or None.

        Returns [B, N] when ``param_dim == 1``, else [B, N, param_dim].
        """
        if H.dim() != 3 or H.shape[-1] != self.config.hidden_dim:
            raise ValueError(f"H must be [B, N, {self.config.hidden_dim}]")
        batch, n_genes, _ = H.shape
        if dataset_ids.shape != (batch,):
            raise ValueError("dataset_ids must be [B]")
        if bool((dataset_ids < 0).any()) or bool(
            (dataset_ids >= self.config.n_datasets).any()
        ):
            raise IndexError("dataset id outside n_datasets")
        if self.config.token_dim == 0:
            if tokens is not None:
                raise ValueError("token_dim is 0; pass no tokens")
            x = H
        else:
            if tokens is None or tokens.shape != (batch, self.config.token_dim):
                raise ValueError(f"tokens must be [B, {self.config.token_dim}]")
            x = torch.cat([H, tokens.unsqueeze(1).expand(-1, n_genes, -1)], dim=-1)
        out = self.head(x)  # [B, N, param_dim]
        if self.scale is not None and self.shift is not None:
            a = self.scale[dataset_ids].unsqueeze(1)  # [B, 1, param_dim]
            b = self.shift[dataset_ids].unsqueeze(1)
            out = a * out + b
        if self.config.param_dim == 1:
            return cast(torch.Tensor, out.squeeze(-1))
        return cast(torch.Tensor, out)


class GlobalScalarReadoutConfig(BaseModel):
    """One scalar per record from the CLS, the perturbed-gene summary and the tokens."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hidden_dim: int = Field(gt=0)
    token_dim: int = Field(ge=0, default=0)
    head_hidden: int = Field(gt=0)
    param_dim: int = Field(gt=0, default=1)
    dropout: float = Field(ge=0.0, lt=1.0, default=0.0)
    use_pert_summary: bool = True


class GlobalScalarReadout(nn.Module):
    """``y_b = MLP([h_b ; z_{S,b} ; t_b])``, the Vaishnav 2022 promoter-level readout."""

    def __init__(self, config: GlobalScalarReadoutConfig) -> None:
        """Build the two-layer MLP over the concatenated inputs."""
        super().__init__()
        self.config = config
        in_dim = (
            config.hidden_dim * (1 + int(config.use_pert_summary)) + config.token_dim
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, config.head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, config.param_dim),
        )

    def forward(
        self,
        h_cls: torch.Tensor,
        z_s: torch.Tensor | None = None,
        tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``h_cls`` [B, d], ``z_s`` [B, d] when configured, ``tokens`` [B, token_dim]."""
        if h_cls.dim() != 2 or h_cls.shape[-1] != self.config.hidden_dim:
            raise ValueError(f"h_cls must be [B, {self.config.hidden_dim}]")
        parts = [h_cls]
        if self.config.use_pert_summary:
            if z_s is None or z_s.shape != h_cls.shape:
                raise ValueError("use_pert_summary needs z_s of the shape of h_cls")
            parts.append(z_s)
        elif z_s is not None:
            raise ValueError("use_pert_summary is False; pass no z_s")
        if self.config.token_dim > 0:
            if tokens is None or tokens.shape != (
                h_cls.shape[0],
                self.config.token_dim,
            ):
                raise ValueError(f"tokens must be [B, {self.config.token_dim}]")
            parts.append(tokens)
        elif tokens is not None:
            raise ValueError("token_dim is 0; pass no tokens")
        out = self.mlp(torch.cat(parts, dim=-1))
        if self.config.param_dim == 1:
            return cast(torch.Tensor, out.squeeze(-1))
        return cast(torch.Tensor, out)


class MaskedTypedLossConfig(BaseModel):
    """The loss per readout type, its weight, and the quantile grid for pinball."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    loss_by_type: dict[str, LossKind] = Field(min_length=1)
    weight_by_type: dict[str, float] | None = None
    num_quantiles: int = Field(gt=0, default=DEFAULT_NUM_QUANTILES)

    @model_validator(mode="after")
    def _weights_name_known_types(self) -> MaskedTypedLossConfig:
        if self.weight_by_type is None:
            return self
        unknown = set(self.weight_by_type) - set(self.loss_by_type)
        if unknown:
            raise ValueError(f"weights for types without a loss: {sorted(unknown)}")
        return self

    def weight(self, readout_type: str) -> float:
        """The loss weight of a readout type; 1 when none is configured."""
        if self.weight_by_type is None:
            return 1.0
        return self.weight_by_type.get(readout_type, 1.0)


class MaskedTypedLoss(nn.Module):
    """Sum over readout types of the weighted mean loss over measured entries."""

    taus: torch.Tensor

    def __init__(self, config: MaskedTypedLossConfig) -> None:
        """Register the even quantile grid ``k / (K + 1)`` for the pinball loss."""
        super().__init__()
        self.config = config
        k = config.num_quantiles
        self.register_buffer(
            "taus", (torch.arange(k, dtype=torch.float32) + 1) / (k + 1)
        )

    def elementwise(
        self, readout_type: str, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Elementwise loss and the measured mask, shaped like ``target``.

        ``pred`` is ``target.shape`` for MSE and ``target.shape + (K,)`` for pinball; the
        pinball is averaged over the quantile axis so both return ``target.shape``.
        """
        kind = self.config.loss_by_type.get(readout_type)
        if kind is None:
            raise KeyError(f"no loss configured for readout type {readout_type!r}")
        measured = ~torch.isnan(target)
        safe_target = torch.where(measured, target, torch.zeros_like(target))
        if kind == "mse":
            if pred.shape != target.shape:
                raise ValueError(
                    f"{readout_type}: mse needs pred {tuple(target.shape)}, "
                    f"got {tuple(pred.shape)}"
                )
            elem = (pred - safe_target) ** 2
        else:
            expected = (*target.shape, self.config.num_quantiles)
            if pred.shape != expected:
                raise ValueError(
                    f"{readout_type}: pinball needs pred {expected}, got {tuple(pred.shape)}"
                )
            elem = pinball(pred, safe_target, self.taus).mean(dim=-1)
        return elem, measured

    def forward(
        self, preds: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``preds`` and ``targets`` keyed by readout type; returns (total, per type)."""
        if set(preds) != set(targets):
            raise KeyError(
                f"preds and targets name different types: {sorted(preds)} vs "
                f"{sorted(targets)}"
            )
        if not preds:
            raise ValueError("no readout type to score")
        per_type: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        for readout_type in sorted(preds):
            elem, measured = self.elementwise(
                readout_type, preds[readout_type], targets[readout_type]
            )
            w = measured.to(elem.dtype)
            count = w.sum()
            if float(count) == 0.0:
                value = elem.sum() * 0.0
            else:
                value = (elem * w).sum() / count
            per_type[readout_type] = value
            term = self.config.weight(readout_type) * value
            total = term if total is None else total + term
        assert total is not None
        return total, per_type
