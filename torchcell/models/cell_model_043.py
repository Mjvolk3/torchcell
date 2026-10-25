# torchcell/models/cell_model_043.py
# [[torchcell.models.cell_model_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/cell_model_043.py
# Test file: tests/torchcell/models/test_cell_model_043.py
"""The 043 cell model: composite frozen tokens -> 019 trunk -> mark operator -> heads.

Design: ``notes-tex/multimodal/043-datasets-primer/sections/input-design.tex`` and
``output-design.tex`` (decisions I1 to I8, O1 to O4 of ``tables/t13-decisions-rows.tex``).
The pieces, in forward order:

``load_vector_table``
    Reads one encoder's ``alleles/<encoder>.h5`` (``sha256`` + fp16 ``embedding``; the
    S288C seed rows live in the same tables, ``source = s288c_table``) and returns the
    rows of the requested keys, in order. A key with no row raises ``KeyError``: there is
    no fallback vector.

``CompositeToken``
    The composite frozen gene token (I1): per encoder block a frozen fp16 table on the
    device, a LayerNorm and a projection to ``d``, summed over the four blocks (FUDT
    upstream 1,003 bp, FUDT downstream 300 bp, ProtT5-XL, CaLM). A slot with no key
    (key row ``-1``) carries a learned embedding of its typed gap
    (``torchcell.data.representation_043.GapKind``) for that block instead of a vector;
    the gap is never filled with another sequence's vector.

``PositionTerm``
    I5: the chromosome (or replicon) embedding scaled by the chromosome copy number
    relative to the basal ploidy, Fourier features of the relative midpoint, and
    ``log2(c + 1) - 1`` of the gene copy number ``c`` (0 at one copy, finite at zero).
    Off-graph tokens have no reference coordinate and use a ``replicon`` row instead of a
    chromosome (``ACCESSORY_ORF``, ``REPORTER_CONSTRUCT``); their midpoint features are
    zeroed. An unknown copy number (``NaN``: ploidy unreported, or not released) drops
    its term and adds the learned embedding of the position gap kind
    (``PositionGap``).

``CellTrunk043``
    The 019 encoder: a learned CLS token followed by the gene tokens through a stack of
    ``GraphRegularizedTransformerLayer`` (imported from
    ``torchcell.models.equivariant_cell_graph_transformer``) with the hard graph
    attention mask on the configured layers, one gene-gene relation per head (019's
    ``attention_mask.head_graphs``).

``MarkOperator043``
    The perturbation operator over marks. Every gene (and the CLS) of the record's
    reference context queries the record's marks: on-graph marks are the trunk state of
    the marked gene plus a projection of the mark token (the record's composite token of
    that gene, its change type, its allele parent, its position); off-graph marks
    (accessory ORFs, the Vaishnav reporter) are the projection alone. One learned NULL
    key with a zero value is always present, so a record without marks attends to it
    and the softmax over a single real mark is not degenerate (019's null sink). The
    residual is the trunk's post-norm residual, ``LN(h + attn)`` then ``LN(x + FFN(x))``.
    In eval mode the operator records how large its branches are against the context it updates
    (:attr:`MarkOperator043.last_norms`). ``z_S`` is the mean of the mark keys (zero for a
    record without marks).

``CellModel043``
    Binds the above and the heads of O1/O3: a gene-level head ``f([H_g ; t])`` shared by
    every gene-level dataset and a record-level head over ``[h_CLS ; z_S ; t]``
    (``PerGeneReadout`` and ``GlobalScalarReadout`` of ``readout_heads_043``), where ``t``
    concatenates the experiment-type path token, the dataset token and the environment
    token, plus a learned per-dataset bias of the head's output width (under HL-Gauss a
    bias vector on the K logits, a learned prior histogram per dataset). The dataset id
    enters only at the head: never the trunk, never the operator.

    ``conditioning = "ontology"`` (experiment 043, 2026-10-23, copied from 030's
    ``dataset_token.source: ontology``): ``t`` is instead the
    :class:`torchcell.data.experiment_ontology.OntologyDagEncoder` token of the record's
    experiment ontology annotation (``Batch043.ontology``, ``[B, n_nodes]``), of the same
    width ``token_dim``, and the two bias tables become ``nn.Linear(token_dim, P)`` maps
    of that token (default PyTorch init). ``dataset_ids``, ``env_ids`` and
    ``exp_type_ids`` are not read by ``forward``: the model cannot tell two records
    apart unless their annotations differ.

Route A context caching (I3). Each distinct reference context of a batch (S288C, or an
isolate under route A) is encoded ONCE per batch, and every record of that context reads
the same trunk output. The frozen inputs are cached on the device for the whole run as
the compact per-encoder tables of every key the run's records use; a context is a
``[G, 4]`` index into them, so caching a context costs ``4 G`` integers, not ``G x 3,328``
fp16 values. The table bytes are reported by :meth:`CompositeToken.table_bytes`. The
trunk output itself is not cached across batches because the trunk trains.

Marks per record (the operator's ``M``). The design has no cap. :func:`select_marks`
implements the cap used ONLY by the smoke: when a record has more than ``cap`` marks
(Boocock segregants, median 3,351; route B isolates about 5,000), keep every mark whose
change type is ``absent``, ``deletion`` or ``loss_of_function`` (in precedence order),
then a fixed random subset of the rest drawn with the record's own seed, so the subset
is the same in every epoch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field, model_validator

from torchcell.data.experiment_ontology import (
    ExperimentOntologyVocabulary,
    OntologyDagEncoder,
)
from torchcell.data.representation_043 import GapKind, PositionGap
from torchcell.losses.distributional import DEFAULT_NUM_QUANTILES, pinball
from torchcell.losses.hl_gauss import HLGaussConfig, HLGaussLoss
from torchcell.models.equivariant_cell_graph_transformer import (
    GraphRegularizedTransformerLayer,
    query_chunked_sdpa,
)
from torchcell.models.genotype_tokens_043 import (
    CHANGE_PRECEDENCE,
    CHANGE_TYPE_INDEX,
    ChangeType,
)
from torchcell.models.readout_heads_043 import (
    ExperimentTypeEmbedding,
    ExperimentTypeEmbeddingConfig,
    GlobalScalarReadout,
    GlobalScalarReadoutConfig,
    PerGeneReadout,
    PerGeneReadoutConfig,
)

# mle and binned_quantile (043 sweep, 2026-10-24) are computed by
# torchcell.losses.readout_family_043.ReadoutLossFamily043, which needs per-row groups.
LossKind = Literal["hlgauss", "pinball", "mse", "mle", "binned_quantile"]

#: The composite encoders in key-column order, and their widths.
ENCODER_DIMS: tuple[int, int, int, int] = (768, 768, 1024, 768)
N_CHROMOSOMES = 17  # 0 = mitochondrion, 1-16 nuclear
N_GAP_KINDS = len(GapKind)
N_POSITION_GAPS = len(PositionGap)


class Replicon(IntEnum):
    """Rows of the position table after the 17 chromosomes, for off-graph tokens."""

    ACCESSORY_ORF = N_CHROMOSOMES  # a pangenome ORF with no reference coordinate
    REPORTER_CONSTRUCT = N_CHROMOSOMES + 1  # the Vaishnav 2022 YFP reporter construct


N_POSITION_ROWS = N_CHROMOSOMES + len(Replicon)


# --------------------------------------------------------------------------- #
# Frozen vectors
# --------------------------------------------------------------------------- #
def load_vector_table(
    h5_path: str, keys: npt.NDArray[np.bytes_] | Sequence[str]
) -> torch.Tensor:
    """``[len(keys), dim]`` fp16 rows of ``h5_path`` for the sha256 ``keys``, in order.

    The whole ``embedding`` dataset is read once (contiguous reads are fast; a point
    selection over a million rows is not) and the requested rows gathered. A key with no
    row raises ``KeyError`` naming how many are missing.
    """
    import h5py

    want = np.asarray(keys, dtype="S64")
    with h5py.File(h5_path, "r") as f:
        have = np.asarray(f["sha256"][:], dtype="S64")
        order = np.argsort(have, kind="stable")
        sorted_have = have[order]
        if want.size == 0:
            dim = int(f["embedding"].shape[1])
            return torch.zeros((0, dim), dtype=torch.float16)
        pos = np.searchsorted(sorted_have, want)
        pos_c = np.minimum(pos, max(sorted_have.size - 1, 0))
        hit = (
            sorted_have[pos_c] == want
            if sorted_have.size
            else np.zeros(want.size, dtype=bool)
        )
        if not hit.all():
            missing = want[~hit][:3]
            raise KeyError(
                f"{int((~hit).sum())} of {want.size} keys have no vector in {h5_path}: "
                f"{[m.decode() for m in missing]}"
            )
        rows = order[pos_c]
        emb = np.asarray(f["embedding"][:], dtype=np.float16)
    return torch.from_numpy(emb[rows])


def select_marks(
    change_types: npt.NDArray[np.integer], cap: int | None, seed: int
) -> npt.NDArray[np.int64]:
    """Indices of the marks kept under the smoke cap (module docstring); all if under it.

    Returned sorted, so the kept marks keep their stored order.
    """
    n = int(change_types.shape[0])
    if cap is None or n <= cap:
        return np.arange(n, dtype=np.int64)
    priority = [
        CHANGE_TYPE_INDEX[c]
        for c in CHANGE_PRECEDENCE
        if c in (ChangeType.absent, ChangeType.deletion, ChangeType.loss_of_function)
    ]
    first: list[int] = []
    for c in priority:
        first.extend(np.nonzero(change_types == c)[0].tolist())
    first = first[:cap]
    taken = np.zeros(n, dtype=bool)
    taken[first] = True
    rest = np.nonzero(~taken)[0]
    rng = np.random.default_rng(seed)
    extra = rng.choice(rest, size=cap - len(first), replace=False)
    return np.sort(np.concatenate([np.asarray(first, dtype=np.int64), extra]))


# --------------------------------------------------------------------------- #
# Configuration and batch containers
# --------------------------------------------------------------------------- #
class CellModel043Config(BaseModel):
    """Widths, vocabularies and the loss of :class:`CellModel043`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hidden_dim: int = Field(default=144, gt=0)
    num_heads: int = Field(default=9, gt=0)
    num_layers: int = Field(default=4, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    mask_layers: tuple[int, ...] = (1,)
    head_graphs: dict[int, str] = Field(default_factory=dict)
    fourier_frequencies: int = Field(default=8, gt=0)
    operator_ffn_mult: int = Field(default=4, gt=0)
    trunk_ffn_mult: int = Field(
        default=4, gt=0, description="trunk FFN width as a multiple of hidden_dim"
    )
    token_dropout: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="training only: the composite token of each gene of each distinct "
        "context is zeroed with this probability and the kept ones scaled by 1/(1-p) "
        "(the position term stays); 0 draws nothing (043 regularization, 2026-10-24)",
    )
    n_parents: int = Field(ge=1, description="allele-parent rows; row 0 = no parent")
    n_datasets: int = Field(gt=0)
    n_envs: int = Field(gt=0)
    exp_type_classes: tuple[str, ...] = Field(min_length=1)
    exp_type_mode: Literal["flat", "dag"] = "dag"
    exp_type_paths: dict[str, tuple[str, ...]] | None = None
    type_token_dim: int = Field(default=16, gt=0)
    env_token_dim: int = Field(default=16, gt=0)
    dataset_token_dim: int = Field(default=8, gt=0)
    conditioning: Literal["name", "ontology"] = Field(
        default="name",
        description="name: type path + dataset + environment embeddings and per-dataset "
        "bias tables; ontology: the OntologyDagEncoder token of the record's annotation "
        "(width token_dim) and linear bias maps of it; no dataset id enters forward",
    )
    ontology_state_dim: int = Field(default=16, gt=0)
    head_hidden: int = Field(default=128, ge=0)
    loss: LossKind = "hlgauss"
    hl_gauss: HLGaussConfig = Field(default_factory=HLGaussConfig)
    num_quantiles: int = Field(default=DEFAULT_NUM_QUANTILES, gt=0)
    attention_query_chunk: int | None = Field(
        default=None,
        gt=0,
        description="query block size of every trunk and operator SDPA "
        "(query_chunked_sdpa); None = one call. Memory only: the attention is the same",
    )

    @model_validator(mode="after")
    def _check(self) -> CellModel043Config:
        """Heads divide the width; masked layers exist; graphs name existing heads."""
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        bad = [layer for layer in self.mask_layers if not 0 <= layer < self.num_layers]
        if bad:
            raise ValueError(f"mask_layers {bad} outside 0..{self.num_layers - 1}")
        bad_heads = [h for h in self.head_graphs if not 0 <= h < self.num_heads]
        if bad_heads:
            raise ValueError(f"head_graphs names heads {bad_heads}")
        if self.mask_layers and not self.head_graphs:
            raise ValueError("mask_layers without head_graphs masks nothing")
        return self

    @property
    def token_dim(self) -> int:
        """Width of ``t = [t_type ; t_dataset ; t_env]``."""
        return self.type_token_dim + self.dataset_token_dim + self.env_token_dim

    @property
    def param_dim(self) -> int:
        """Output width per prediction: K bins, the quantile count, 2 (mean and
        log-variance) for mle, or 1.
        """
        if self.loss in ("hlgauss", "binned_quantile"):
            return self.hl_gauss.num_bins
        if self.loss == "mle":
            return 2
        if self.loss == "pinball":
            return self.num_quantiles
        return 1


@dataclass(frozen=True)
class ContextInputs:
    """The distinct reference contexts of a batch, ``[C, G, ...]``.

    ``key_rows`` index the compact device tables (``-1`` = gap, with its kind in
    ``key_gap``); ``chrom_copy`` / ``gene_copy`` are ``NaN`` where unknown.
    """

    key_rows: torch.Tensor  # [C, G, 4] long
    key_gap: torch.Tensor  # [C, G, 4] long
    chrom_copy: torch.Tensor  # [C, G] float
    gene_copy: torch.Tensor  # [C, G] float
    position_gap: torch.Tensor  # [C, G] long


@dataclass(frozen=True)
class MarkInputs:
    """The marks of each record, padded to ``[B, M]`` (``valid`` False = padding)."""

    valid: torch.Tensor  # [B, M] bool
    on_graph: torch.Tensor  # [B, M] bool
    gene: torch.Tensor  # [B, M] long, 0 where off-graph or padding
    change: torch.Tensor  # [B, M] long
    parent: torch.Tensor  # [B, M] long, 0 = no parent
    key_rows: torch.Tensor  # [B, M, 4] long
    key_gap: torch.Tensor  # [B, M, 4] long
    replicon: torch.Tensor  # [B, M] long: chromosome index, or a Replicon row
    rel_midpoint: torch.Tensor  # [B, M] float
    chrom_copy: torch.Tensor  # [B, M] float
    gene_copy: torch.Tensor  # [B, M] float
    position_gap: torch.Tensor  # [B, M] long


@dataclass(frozen=True)
class Batch043:
    """One batch: contexts, the context of each record, marks and the head ids."""

    contexts: ContextInputs
    record_context: torch.Tensor  # [B] long, row of ``contexts``
    marks: MarkInputs
    dataset_ids: torch.Tensor  # [B] long
    exp_type_ids: torch.Tensor  # [B] long
    env_ids: torch.Tensor  # [B] long
    ontology: torch.Tensor | None = None  # [B, n_nodes] float, the record's annotation

    def to(self, device: torch.device) -> Batch043:
        """A copy with every tensor on ``device``."""

        def move(obj: ContextInputs | MarkInputs) -> dict[str, torch.Tensor]:
            return {
                k: cast(torch.Tensor, v).to(device, non_blocking=True)
                for k, v in obj.__dict__.items()
            }

        return Batch043(
            contexts=ContextInputs(**move(self.contexts)),
            record_context=self.record_context.to(device, non_blocking=True),
            marks=MarkInputs(**move(self.marks)),
            dataset_ids=self.dataset_ids.to(device, non_blocking=True),
            exp_type_ids=self.exp_type_ids.to(device, non_blocking=True),
            env_ids=self.env_ids.to(device, non_blocking=True),
            ontology=None
            if self.ontology is None
            else self.ontology.to(device, non_blocking=True),
        )


# --------------------------------------------------------------------------- #
# Input side
# --------------------------------------------------------------------------- #
class CompositeToken(nn.Module):
    """Per-block LayerNorm and projection of the frozen vectors, summed (I1)."""

    def __init__(
        self, tables: Sequence[torch.Tensor], hidden_dim: int, dims: Sequence[int]
    ) -> None:
        """``tables[e]`` is the compact fp16 table of encoder ``e`` (frozen buffer)."""
        super().__init__()
        if len(tables) != len(dims):
            raise ValueError(f"{len(tables)} tables for {len(dims)} encoder blocks")
        for e, (t, dim) in enumerate(zip(tables, dims, strict=True)):
            if t.dim() != 2 or int(t.shape[1]) != dim:
                raise ValueError(f"table {e} must be [n, {dim}], got {tuple(t.shape)}")
            self.register_buffer(f"table_{e}", t.to(torch.float16), persistent=False)
        self.n_blocks = len(dims)
        self.norms = nn.ModuleList(nn.LayerNorm(d) for d in dims)
        self.projs = nn.ModuleList(nn.Linear(d, hidden_dim) for d in dims)
        self.gap_embedding = nn.Embedding(self.n_blocks * N_GAP_KINDS, hidden_dim)

    def table(self, e: int) -> torch.Tensor:
        """The compact table of block ``e``."""
        return cast(torch.Tensor, getattr(self, f"table_{e}"))

    def table_bytes(self) -> int:
        """Device bytes held by the frozen tables."""
        return sum(
            self.table(e).numel() * self.table(e).element_size()
            for e in range(self.n_blocks)
        )

    def forward(self, key_rows: torch.Tensor, key_gap: torch.Tensor) -> torch.Tensor:
        """``key_rows``/``key_gap`` ``[..., n_blocks]`` -> ``[..., d]``."""
        out: torch.Tensor | None = None
        for e in range(self.n_blocks):
            rows = key_rows[..., e]
            have = rows >= 0
            # an out-of-range row raises IndexError on CPU and a device assert on CUDA
            v = self.table(e)[rows.clamp_min(0)].float()
            proj = self.projs[e](self.norms[e](v))
            gap = self.gap_embedding(key_gap[..., e] + e * N_GAP_KINDS)
            term = torch.where(have.unsqueeze(-1), proj, gap)
            out = term if out is None else out + term
        assert out is not None
        return out


class PositionTerm(nn.Module):
    """Chromosome/replicon x copy, Fourier midpoint, gene copy, position gaps (I5)."""

    freqs: torch.Tensor

    def __init__(self, hidden_dim: int, n_frequencies: int) -> None:
        """Embedding tables and the two linear maps."""
        super().__init__()
        self.position_embedding = nn.Embedding(N_POSITION_ROWS, hidden_dim)
        self.register_buffer(
            "freqs",
            2.0 * math.pi * (2.0 ** torch.arange(n_frequencies, dtype=torch.float32)),
        )
        self.fourier = nn.Linear(2 * n_frequencies, hidden_dim)
        self.copy_proj = nn.Linear(1, hidden_dim)
        self.gap_embedding = nn.Embedding(N_POSITION_GAPS, hidden_dim)

    def forward(
        self,
        replicon: torch.Tensor,
        rel_midpoint: torch.Tensor,
        chrom_copy: torch.Tensor,
        gene_copy: torch.Tensor,
        position_gap: torch.Tensor,
        on_graph: torch.Tensor,
    ) -> torch.Tensor:
        """Every input ``[...]`` -> ``[..., d]``."""
        chrom_known = torch.isfinite(chrom_copy)
        scale = torch.where(chrom_known, chrom_copy, torch.zeros_like(chrom_copy))
        term = self.position_embedding(replicon) * scale.unsqueeze(-1)
        angle = rel_midpoint.float().unsqueeze(-1) * self.freqs
        four = torch.cat([angle.sin(), angle.cos()], dim=-1)
        term = term + self.fourier(four) * on_graph.unsqueeze(-1).to(term.dtype)
        copy_known = torch.isfinite(gene_copy)
        safe = torch.where(copy_known, gene_copy, torch.zeros_like(gene_copy))
        feat = (torch.log2(safe.clamp_min(0.0) + 1.0) - 1.0) * copy_known.to(safe.dtype)
        term = term + self.copy_proj(feat.unsqueeze(-1))
        return cast(torch.Tensor, term + self.gap_embedding(position_gap))


def build_head_mask(
    edge_index_by_relation: Mapping[str, torch.Tensor],
    head_graphs: Mapping[int, str],
    num_heads: int,
    n_genes: int,
) -> torch.Tensor:
    """``[heads, G+1, G+1]`` bool, True where attention is allowed (019's mask).

    A head with no graph stays free; a graph head sees its symmetric edges, a self-loop
    per gene, and the CLS row and column. A relation not in ``edge_index_by_relation``
    raises.
    """
    mask = torch.ones(num_heads, n_genes + 1, n_genes + 1, dtype=torch.bool)
    for head, rel in head_graphs.items():
        if rel not in edge_index_by_relation:
            raise KeyError(
                f"head {head} names graph {rel!r}; available: "
                f"{sorted(edge_index_by_relation)}"
            )
        ei = edge_index_by_relation[rel]
        if ei.numel() and int(ei.max()) >= n_genes:
            raise IndexError(f"{rel}: edge index outside {n_genes} genes")
        m = torch.zeros(n_genes + 1, n_genes + 1, dtype=torch.bool)
        m[ei[0] + 1, ei[1] + 1] = True
        m[ei[1] + 1, ei[0] + 1] = True
        m.fill_diagonal_(True)
        m[0, :] = True
        m[:, 0] = True
        mask[head] = m
    return mask


class CellTrunk043(nn.Module):
    """CLS + gene tokens through the 019 transformer layers with the graph mask."""

    head_mask: torch.Tensor | None

    def __init__(
        self, config: CellModel043Config, head_mask: torch.Tensor | None
    ) -> None:
        """``head_mask`` ``[heads, G+1, G+1]`` is required when layers are masked."""
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.layers = nn.ModuleList(
            GraphRegularizedTransformerLayer(
                d,
                config.num_heads,
                config.dropout,
                config.attention_query_chunk,
                ffn_mult=config.trunk_ffn_mult,
            )
            for _ in range(config.num_layers)
        )
        if config.mask_layers and head_mask is None:
            raise ValueError("mask_layers are configured but no head_mask was given")
        if head_mask is not None:
            self.register_buffer("head_mask", head_mask, persistent=False)
        else:
            self.head_mask = None
        self.masked = set(config.mask_layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """``X`` ``[C, G, d]`` -> ``[C, G+1, d]`` with the CLS at position 0."""
        h = torch.cat([self.cls_token.expand(X.shape[0], -1, -1), X], dim=1)
        if self.head_mask is not None and self.head_mask.shape[-1] != h.shape[1]:
            raise ValueError("head_mask does not match the gene count")
        for i, layer in enumerate(self.layers):
            mask = self.head_mask if i in self.masked else None
            h, _ = layer(h, return_attention=False, head_mask=mask)
        return h


class MarkOperator043(nn.Module):
    """Every context token queries the record's marks plus a null key.

    Post-norm residual, as the trunk (module docstring). In eval mode ``last_norms``
    holds, for the last forward, the mean over records and tokens of

    * ``attn``: the attention branch as added, ``|attn| / |H|``;
    * ``ffn``: the FFN branch as added, ``|FFN(H1)| / |H1|``;
    * ``update``: the whole change, ``|H_out - H| / |H|``;

    ``attn`` is the operator-liveness statistic the trainer logs as ``operator/relative_norm``.
    """

    def __init__(self, config: CellModel043Config) -> None:
        """Q/K/V/O projections, the null key, the FFN and the two post-norms."""
        super().__init__()
        d, h = config.hidden_dim, config.num_heads
        self.num_heads = h
        self.head_dim = d // h
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.null_key = nn.Parameter(torch.randn(h, self.head_dim) * 0.02)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * config.operator_ffn_mult),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d * config.operator_ffn_mult, d),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.p_drop = config.dropout
        self.query_chunk = config.attention_query_chunk
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.last_norms: dict[str, torch.Tensor] = {}

    @staticmethod
    def _ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
        """Mean over records and tokens of the row-norm ratio ``|num| / |den|``."""
        n = num.float().norm(dim=-1)
        dd = den.float().norm(dim=-1).clamp_min(1e-12)
        return cast(torch.Tensor, (n / dd).mean().detach())

    def forward(
        self, H: torch.Tensor, keys: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """``H`` ``[B, N, d]``, ``keys`` ``[B, M, d]``, ``valid`` ``[B, M]`` -> ``[B, N, d]``."""
        B, N, d = H.shape
        M = keys.shape[1]
        h, hd = self.num_heads, self.head_dim
        q = self.q_proj(H).view(B, N, h, hd).transpose(1, 2)
        k = self.k_proj(keys).view(B, M, h, hd).transpose(1, 2)
        v = self.v_proj(keys).view(B, M, h, hd).transpose(1, 2)
        null_k = self.null_key.to(k.dtype).view(1, h, 1, hd).expand(B, -1, -1, -1)
        k = torch.cat([null_k, k], dim=2)
        v = torch.cat([v.new_zeros(B, h, 1, hd), v], dim=2)
        allowed = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=valid.device), valid], dim=1
        ).view(B, 1, 1, M + 1)
        attn = query_chunked_sdpa(
            q, k, v, allowed, self.p_drop if self.training else 0.0, self.query_chunk
        )
        out = self.out_proj(attn.transpose(1, 2).reshape(B, N, d))
        a = self.dropout(out)
        H1 = self.norm1(H + a)
        f = self.dropout(self.ffn(H1))
        H2 = self.norm2(H1 + f)
        if not self.training:
            self.last_norms = {
                "attn": self._ratio(a, H),
                "ffn": self._ratio(f, H1),
                "update": self._ratio(H2 - H, H),
            }
        return cast(torch.Tensor, H2)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CellModel043Output:
    """``gene`` ``[Bg, G(, P)]`` for the gene-level rows, ``record`` ``[Br(, P)]``."""

    gene: torch.Tensor
    record: torch.Tensor
    h_cls: torch.Tensor  # [B, d]
    z_s: torch.Tensor  # [B, d]


class CellModel043(nn.Module):
    """Composite tokens -> trunk per distinct context -> mark operator -> level heads."""

    chrom_index: torch.Tensor
    rel_midpoint: torch.Tensor

    def __init__(
        self,
        config: CellModel043Config,
        tables: Sequence[torch.Tensor],
        chrom_index: torch.Tensor,
        rel_midpoint: torch.Tensor,
        head_mask: torch.Tensor | None,
        encoder_dims: Sequence[int] = ENCODER_DIMS,
        ontology_vocabulary: ExperimentOntologyVocabulary | None = None,
    ) -> None:
        """``chrom_index``/``rel_midpoint`` ``[G]`` are the reference positions;
        ``ontology_vocabulary`` is required (and only allowed) under
        ``conditioning = "ontology"``.
        """
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.n_genes = int(chrom_index.shape[0])
        self.register_buffer("chrom_index", chrom_index.long(), persistent=False)
        self.register_buffer("rel_midpoint", rel_midpoint.float(), persistent=False)
        self.composite = CompositeToken(tables, d, encoder_dims)
        self.position = PositionTerm(d, config.fourier_frequencies)
        self.trunk = CellTrunk043(config, head_mask)
        self.change_embedding = nn.Embedding(len(ChangeType), d)
        self.parent_embedding = nn.Embedding(config.n_parents, d, padding_idx=0)
        self.mark_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.operator = MarkOperator043(config)
        self.ontology = config.conditioning == "ontology"
        if self.ontology != (ontology_vocabulary is not None):
            raise ValueError(
                "an ontology vocabulary is required exactly when conditioning is ontology"
            )
        if ontology_vocabulary is not None:
            self.onto_token = OntologyDagEncoder(
                ontology_vocabulary, config.ontology_state_dim, config.token_dim
            )
        else:
            self.type_token = ExperimentTypeEmbedding(
                ExperimentTypeEmbeddingConfig(
                    classes=config.exp_type_classes,
                    hidden_dim=config.type_token_dim,
                    mode=config.exp_type_mode,
                    dag_paths=config.exp_type_paths,
                )
            )
            self.dataset_token = nn.Embedding(
                config.n_datasets, config.dataset_token_dim
            )
            self.env_token = nn.Embedding(config.n_envs, config.env_token_dim)
        p = config.param_dim
        self.gene_head = PerGeneReadout(
            PerGeneReadoutConfig(
                hidden_dim=d,
                n_datasets=config.n_datasets,
                token_dim=config.token_dim,
                head_hidden=config.head_hidden,
                param_dim=p,
                dropout=config.dropout,
                per_dataset_affine=False,
            )
        )
        self.record_head = GlobalScalarReadout(
            GlobalScalarReadoutConfig(
                hidden_dim=d,
                token_dim=config.token_dim,
                head_hidden=max(config.head_hidden, 1),
                param_dim=p,
                dropout=config.dropout,
                use_pert_summary=True,
            )
        )
        if self.ontology:
            # the per-dataset bias tables become maps of the ontology token (default
            # PyTorch init; no zero init, no gate)
            self.gene_bias_map = nn.Linear(config.token_dim, p)
            self.record_bias_map = nn.Linear(config.token_dim, p)
        else:
            self.gene_bias = nn.Parameter(torch.zeros(config.n_datasets, p))
            self.record_bias = nn.Parameter(torch.zeros(config.n_datasets, p))

    def context_tokens(self, ctx: ContextInputs) -> torch.Tensor:
        """``[C, G, d]`` composite plus position for each distinct context."""
        C, G = ctx.chrom_copy.shape
        if G != self.n_genes:
            raise ValueError(f"contexts carry {G} genes, the model {self.n_genes}")
        tok = self.composite(ctx.key_rows, ctx.key_gap)
        p_drop = self.config.token_dropout
        if self.training and p_drop > 0.0:
            keep = torch.rand(C, G, 1, device=tok.device) >= p_drop
            tok = tok * keep.to(tok.dtype) / (1.0 - p_drop)
        pos = self.position(
            self.chrom_index.unsqueeze(0).expand(C, -1),
            self.rel_midpoint.unsqueeze(0).expand(C, -1),
            ctx.chrom_copy,
            ctx.gene_copy,
            ctx.position_gap,
            torch.ones(C, G, dtype=torch.bool, device=tok.device),
        )
        return cast(torch.Tensor, tok + pos)

    def mark_keys(self, marks: MarkInputs, H_rec: torch.Tensor) -> torch.Tensor:
        """``[B, M, d]``: trunk state of on-graph marks plus the projected mark token."""
        tok = self.composite(marks.key_rows, marks.key_gap)
        tok = tok + self.position(
            marks.replicon,
            marks.rel_midpoint,
            marks.chrom_copy,
            marks.gene_copy,
            marks.position_gap,
            marks.on_graph,
        )
        tok = (
            tok
            + self.change_embedding(marks.change)
            + self.parent_embedding(marks.parent)
        )
        state = torch.gather(
            H_rec, 1, (marks.gene + 1).unsqueeze(-1).expand(-1, -1, H_rec.shape[-1])
        )
        keys = self.mark_proj(tok) + state * marks.on_graph.unsqueeze(-1).to(
            state.dtype
        )
        return cast(torch.Tensor, keys * marks.valid.unsqueeze(-1).to(keys.dtype))

    def tokens(self, batch: Batch043, rows: torch.Tensor) -> torch.Tensor:
        """``t = [t_type ; t_dataset ; t_env]`` for ``rows`` of the batch, or under
        ontology conditioning the encoder token of the rows' annotations.
        """
        if self.ontology:
            if batch.ontology is None:
                raise ValueError("ontology conditioning needs Batch043.ontology")
            return cast(torch.Tensor, self.onto_token(batch.ontology[rows]))
        return torch.cat(
            [
                self.type_token(batch.exp_type_ids[rows]),
                self.dataset_token(batch.dataset_ids[rows]),
                self.env_token(batch.env_ids[rows]),
            ],
            dim=-1,
        )

    def forward(
        self, batch: Batch043, gene_rows: torch.Tensor, record_rows: torch.Tensor
    ) -> CellModel043Output:
        """Heads for the gene-level rows and the record-level rows of the batch."""
        X = self.context_tokens(batch.contexts)
        H_ctx = self.trunk(X)  # [C, G+1, d], one pass per distinct context
        H_rec = H_ctx[batch.record_context]  # [B, G+1, d]
        keys = self.mark_keys(batch.marks, H_rec)
        H_pert = self.operator(H_rec, keys, batch.marks.valid)
        valid = batch.marks.valid.unsqueeze(-1).to(keys.dtype)
        z_s = (keys * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        h_cls = H_pert[:, 0]
        p = self.config.param_dim
        if gene_rows.numel():
            t = self.tokens(batch, gene_rows)
            if self.ontology:
                # the head's dataset argument only feeds its per-dataset affine, which
                # this model never builds (per_dataset_affine=False); a constant id
                # keeps the dataset id out of forward
                head_ids = torch.zeros_like(gene_rows)
                bias = self.gene_bias_map(t)
            else:
                head_ids = batch.dataset_ids[gene_rows]
                bias = self.gene_bias[batch.dataset_ids[gene_rows]]
            gene = self.gene_head(H_pert[gene_rows, 1:], head_ids, t)
            gene = gene + (bias.unsqueeze(1) if p > 1 else bias)
        else:
            gene = H_pert.new_zeros(
                (0, self.n_genes, p) if p > 1 else (0, self.n_genes)
            )
        if record_rows.numel():
            t = self.tokens(batch, record_rows)
            record = self.record_head(h_cls[record_rows], z_s[record_rows], t)
            bias = (
                self.record_bias_map(t)
                if self.ontology
                else self.record_bias[batch.dataset_ids[record_rows]]
            )
            record = record + (bias if p > 1 else bias.squeeze(-1))
        else:
            record = H_pert.new_zeros((0, p) if p > 1 else (0,))
        return CellModel043Output(gene=gene, record=record, h_cls=h_cls, z_s=z_s)


# --------------------------------------------------------------------------- #
# The objective
# --------------------------------------------------------------------------- #
class ReadoutLoss043(nn.Module):
    """HL-Gauss, pinball or MSE on standardized targets, masked to measured entries."""

    taus: torch.Tensor

    def __init__(self, config: CellModel043Config) -> None:
        """The loss of ``config.loss``; quantile grid ``k / (K + 1)`` for the pinball."""
        super().__init__()
        self.kind: LossKind = config.loss
        self.hl = HLGaussLoss(config.hl_gauss)
        k = config.num_quantiles
        self.register_buffer(
            "taus", (torch.arange(k, dtype=torch.float32) + 1) / (k + 1)
        )
        self.median_index = k // 2

    def elementwise(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Loss per entry (zero where missing) and the measured mask, ``target.shape``."""
        if self.kind in ("mle", "binned_quantile"):
            raise ValueError(
                f"{self.kind} needs per-row groups: use "
                "torchcell.losses.readout_family_043.ReadoutLossFamily043"
            )
        if self.kind == "hlgauss":
            return self.hl.elementwise(pred, target)
        measured = ~torch.isnan(target)
        safe = torch.where(measured, target, torch.zeros_like(target)).float()
        if self.kind == "mse":
            if pred.shape != target.shape:
                raise ValueError(f"mse needs pred {tuple(target.shape)}")
            elem = (pred.float() - safe) ** 2
        else:
            if tuple(pred.shape) != (*target.shape, self.taus.numel()):
                raise ValueError("pinball needs pred target.shape + (K,)")
            elem = pinball(pred.float(), safe, self.taus).mean(dim=-1)
        return elem * measured.to(elem.dtype), measured

    def point(self, pred: torch.Tensor) -> torch.Tensor:
        """The point prediction in z: expectation, median quantile, or the value."""
        if self.kind in ("mle", "binned_quantile"):
            raise ValueError(f"{self.kind}: use ReadoutLossFamily043.point")
        if self.kind == "hlgauss":
            return self.hl.expectation(pred)
        if self.kind == "pinball":
            return pred[..., self.median_index].float()
        return pred.float()
