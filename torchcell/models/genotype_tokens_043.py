# torchcell/models/genotype_tokens_043.py
# [[torchcell.models.genotype_tokens_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/genotype_tokens_043.py
# Test file: tests/torchcell/models/test_genotype_tokens_043.py
"""Prototype of the 043 input side: genotype -> change-type marks -> gene tokens.

Design: ``notes-tex/multimodal/043-datasets-primer/sections/input-design.tex`` and
[[experiments.043-expression-proteome-morphology]]. Three objects:

* :class:`ChangeType`, the change-type vocabulary one gene carries under one genotype.
  ``PERTURBATION_TYPE_TO_CHANGE`` maps every ``perturbation_type`` of the 043 store onto
  it, and ``CHANGE_PRECEDENCE`` resolves a gene that carries several (a natural isolate
  typically carries a sequence variant AND a loss-of-function call on one gene).
* :class:`GenotypeTokenSet`, the marks of one strain: marked gene ids, their change
  types, the parent whose allele a marked gene carries (segregants), promoter sequences
  for promoter marks and strain-level tokens. Reference genes carry no entry. Built from
  deletions, from an isolate's perturbation list, or from a segregant's haplotype blocks
  under the definite assumption (non-reference parent, block posterior >= threshold,
  the gene contained in one block).
* :class:`GenotypeEncoder`, an ``nn.Module`` that turns a batch of token sets into the
  tensors the perturbation operator reads: the reference token matrix ``X`` from a
  pluggable :class:`SequenceEncoder`, the change-type matrix ``c``, the mask ``m``, the
  per-strain realized tokens ``X + T[c] + A[parent]`` and the flattened
  ``perturbation_indices`` / ``perturbation_indices_batch`` pair the existing
  ``EquivariantPerturbationTransform`` consumes.

Nothing here falls back: a promoter mark without a promoter encoder raises, a gene
outside the graph is reported in ``off_graph_marks`` rather than dropped, and a token
set whose lengths disagree is refused at construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol, cast, runtime_checkable

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import nn

from torchcell.datamodels.schema import HaplotypeBlock


class ChangeType(StrEnum):
    """What a gene's realized sequence is, relative to the reference, under a genotype.

    ``reference`` is the implicit state of every gene a token set does not name. The
    index of each member (``CHANGE_TYPE_INDEX``) is the row of its learned embedding.
    """

    reference = "reference"
    deletion = "deletion"
    variant = "variant"
    loss_of_function = "loss_of_function"
    absent = "absent"
    present = "present"
    promoter = "promoter"


CHANGE_TYPE_INDEX: dict[ChangeType, int] = {c: i for i, c in enumerate(ChangeType)}

#: Every ``perturbation_type`` the 043 store holds (results/store_census.json), mapped
#: onto the vocabulary. Engineered deletions of every marker all read as ``deletion``;
#: the natural absence of a reference gene is ``absent`` (same physical state, natural
#: provenance); an accessory ORF the isolate carries is ``present`` (no row in the
#: reference graph); Vaishnav's reporter is ``promoter``.
PERTURBATION_TYPE_TO_CHANGE: dict[str, ChangeType] = {
    "deletion": ChangeType.deletion,
    "kanmx_deletion": ChangeType.deletion,
    "sga_kanmx_deletion": ChangeType.deletion,
    "sga_natmx_deletion": ChangeType.deletion,
    "barcoded_kanmx_deletion": ChangeType.deletion,
    "marker_deletion": ChangeType.deletion,
    "sequence_variant": ChangeType.variant,
    "natural_loss_of_function": ChangeType.loss_of_function,
    "natural_gene_absence": ChangeType.absent,
    "natural_gene_presence": ChangeType.present,
    "variable_promoter_reporter": ChangeType.promoter,
}

#: When one gene carries several marks, the first type in this order wins. A gene that
#: is gone (engineered or natural) outranks a call on its sequence, which outranks a
#: variant allele that is still present and expressed.
CHANGE_PRECEDENCE: tuple[ChangeType, ...] = (
    ChangeType.absent,
    ChangeType.deletion,
    ChangeType.loss_of_function,
    ChangeType.variant,
    ChangeType.present,
    ChangeType.promoter,
)

#: The strain token that says "no parent": reference genes and every non-segregant mark.
NO_PARENT = "none"


class GeneCoordinate(BaseModel):
    """A gene's span on the reference, in the chromosome token the segregant loaders use.

    ``chromosome`` is ``chrI`` to ``chrXVI`` (Boocock and N'Guessan write their marker
    names and blocks that way; ``SCerevisiaeGene.chromosome`` is the integer key, so the
    caller maps once). ``start <= end`` in 1-based reference coordinates.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chromosome: str
    start: int = Field(ge=1)
    end: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> GeneCoordinate:
        """``start <= end``."""
        if self.end < self.start:
            raise ValueError(
                f"GeneCoordinate needs start <= end, got {self.start}..{self.end}"
            )
        return self


class GenotypeTokenSet(BaseModel):
    """The marks of one strain. Reference genes carry no entry.

    ``gene_ids[k]`` carries ``change_types[k]`` and the allele of ``allele_parent[k]``
    (``NO_PARENT`` unless the mark comes from a segregant block). ``promoter_sequences``
    holds the promoter of every ``promoter`` mark and nothing else. ``strain_tokens``
    are strain-level categorical tokens (``parent:RM``, ``dataset:kemmeren2014``,
    ``medium:YPD``) that attach to the strain, not to a gene.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    gene_ids: tuple[str, ...]
    change_types: tuple[ChangeType, ...]
    allele_parent: tuple[str, ...]
    promoter_sequences: dict[str, str] = Field(default_factory=dict)
    strain_tokens: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> GenotypeTokenSet:
        """One mark per gene, equal lengths, promoter sequences exactly for promoter marks."""
        n = len(self.gene_ids)
        if len(self.change_types) != n or len(self.allele_parent) != n:
            raise ValueError(
                "gene_ids, change_types and allele_parent must have one entry per mark: "
                f"{n}, {len(self.change_types)}, {len(self.allele_parent)}"
            )
        if len(set(self.gene_ids)) != n:
            raise ValueError(
                "a gene carries one mark; resolve duplicates with CHANGE_PRECEDENCE"
            )
        if ChangeType.reference in self.change_types:
            raise ValueError(
                "reference is implicit; a token set never names a reference gene"
            )
        promoter_genes = {
            g
            for g, c in zip(self.gene_ids, self.change_types)
            if c is ChangeType.promoter
        }
        if set(self.promoter_sequences) != promoter_genes:
            raise ValueError(
                "promoter_sequences must hold exactly the promoter marks: "
                f"{sorted(promoter_genes)} vs {sorted(self.promoter_sequences)}"
            )
        return self

    @property
    def n_marks(self) -> int:
        """Number of marked genes (reference genes are not counted)."""
        return len(self.gene_ids)

    @classmethod
    def from_deletions(
        cls, genes: Sequence[str], strain_tokens: Sequence[str] = ()
    ) -> GenotypeTokenSet:
        """One or more engineered deletions (Kemmeren, Sameith Dm, Messner, Ohya ...)."""
        return cls(
            gene_ids=tuple(genes),
            change_types=tuple(ChangeType.deletion for _ in genes),
            allele_parent=tuple(NO_PARENT for _ in genes),
            strain_tokens=tuple(strain_tokens),
        )

    @classmethod
    def from_perturbations(
        cls,
        marks: Sequence[tuple[str, str]],
        strain_tokens: Sequence[str] = (),
        promoter_sequences: Mapping[str, str] | None = None,
    ) -> GenotypeTokenSet:
        """From ``(systematic_gene_name, perturbation_type)`` pairs, as a record lists them.

        An isolate lists thousands; a gene named more than once keeps the mark of highest
        precedence. A ``perturbation_type`` outside ``PERTURBATION_TYPE_TO_CHANGE`` raises.
        """
        best: dict[str, ChangeType] = {}
        for gene, ptype in marks:
            change = PERTURBATION_TYPE_TO_CHANGE[ptype]
            prior = best.get(gene)
            if prior is None or CHANGE_PRECEDENCE.index(
                change
            ) < CHANGE_PRECEDENCE.index(prior):
                best[gene] = change
        genes = tuple(best)
        promoters = dict(promoter_sequences or {})
        return cls(
            gene_ids=genes,
            change_types=tuple(best[g] for g in genes),
            allele_parent=tuple(NO_PARENT for _ in genes),
            promoter_sequences=promoters,
            strain_tokens=tuple(strain_tokens),
        )

    @classmethod
    def from_segregant_blocks(
        cls,
        blocks: Sequence[HaplotypeBlock],
        gene_coordinates: Mapping[str, GeneCoordinate],
        nonreference_parent: Literal[1, 2],
        parent_token: str,
        posterior_min: float = 0.9,
        strain_tokens: Sequence[str] = (),
    ) -> GenotypeTokenSet:
        """The definite assumption for a segregant.

        A gene is marked ``variant`` with ``allele_parent = parent_token`` when one block
        of ``nonreference_parent`` with ``posterior >= posterior_min`` contains the whole
        gene (``block.start <= gene.start`` and ``gene.end <= block.end``). A gene in a
        reference-parent block, in a block below the threshold, straddling a block edge or
        in the unassigned gap between blocks stays reference: the crossover is unresolved
        by design (``HaplotypeBlock``), so nothing is guessed about it.
        """
        kept = [
            b
            for b in blocks
            if b.parent == nonreference_parent and b.posterior >= posterior_min
        ]
        by_chrom: dict[str, list[HaplotypeBlock]] = {}
        for b in kept:
            by_chrom.setdefault(b.chromosome, []).append(b)
        marked: list[str] = []
        for gene, coord in gene_coordinates.items():
            for b in by_chrom.get(coord.chromosome, ()):
                if b.start <= coord.start and coord.end <= b.end:
                    marked.append(gene)
                    break
        return cls(
            gene_ids=tuple(marked),
            change_types=tuple(ChangeType.variant for _ in marked),
            allele_parent=tuple(parent_token for _ in marked),
            strain_tokens=tuple(strain_tokens),
        )


@runtime_checkable
class SequenceEncoder(Protocol):
    """A swappable encoder ``f_theta`` from a gene's sequence to a vector.

    The prototype asks for the reference feature of every gene in graph order; a
    sequence-level encoder evaluates ``f_theta`` on the gene's realized sequence, a
    precomputed one looks the vector up.
    """

    @property
    def dim(self) -> int:
        """Width of one encoded gene."""
        ...

    def encode_genes(self, gene_ids: Sequence[str]) -> torch.Tensor:
        """``[len(gene_ids), dim]``, one row per gene, in the order given."""
        ...


@runtime_checkable
class PromoterEncoder(Protocol):
    """A swappable encoder ``g_phi`` from a promoter sequence to a vector.

    The interface Vaishnav 2022 trains against; no implementation ships with the
    prototype, and a promoter mark reaching :class:`GenotypeEncoder` without one raises.
    """

    @property
    def dim(self) -> int:
        """Width of one encoded promoter."""
        ...

    def encode_promoters(self, sequences: Sequence[str]) -> torch.Tensor:
        """``[len(sequences), dim]``, one row per promoter, in the order given."""
        ...


class PrecomputedGeneEmbedding(nn.Module):
    """A :class:`SequenceEncoder` that looks up precomputed per-gene vectors.

    This is what ``NodeEmbeddingBuilder`` products are (ProtT5, nucleotide transformer
    windows, FUDT, CALM, codon frequency): one frozen row per reference gene. ``table``
    is ``[n_genes, dim]`` and ``index`` maps gene id to row; an unknown gene raises.
    """

    table: torch.Tensor

    def __init__(self, table: torch.Tensor, index: Mapping[str, int]) -> None:
        """Register ``table`` as a frozen buffer and keep the gene-to-row index."""
        super().__init__()
        if table.dim() != 2:
            raise ValueError(f"table must be [n_genes, dim], got {tuple(table.shape)}")
        if len(index) != table.shape[0]:
            raise ValueError(
                f"index names {len(index)} genes but the table has {table.shape[0]} rows"
            )
        self.register_buffer("table", table)
        self.index = dict(index)

    @property
    def dim(self) -> int:
        """Width of one stored gene vector."""
        return int(self.table.shape[1])

    def encode_genes(self, gene_ids: Sequence[str]) -> torch.Tensor:
        """Rows of ``table`` for ``gene_ids``, in order; an unknown gene raises."""
        rows = torch.tensor([self.index[g] for g in gene_ids], dtype=torch.long)
        return self.table[rows.to(self.table.device)]


class GenotypeEncoderConfig(BaseModel):
    """Widths and vocabularies of :class:`GenotypeEncoder`.

    ``parent_tokens`` lists every allele-parent token a batch may carry (``NO_PARENT``
    is always row 0); ``strain_token_vocab`` every strain-level token. ``use_position``
    adds a chromosome embedding and a linear map of the normalized coordinate to the
    reference token; it needs ``positions`` at construction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    d_model: int = Field(ge=1)
    parent_tokens: tuple[str, ...] = ()
    strain_token_vocab: tuple[str, ...] = ()
    use_position: bool = False
    n_chromosomes: int = Field(
        default=17, ge=1, description="16 nuclear plus mitochondrion at 0"
    )

    @model_validator(mode="after")
    def _no_reserved_parent(self) -> GenotypeEncoderConfig:
        """Vocabularies do not repeat and do not claim the reserved parent row."""
        if NO_PARENT in self.parent_tokens:
            raise ValueError(f"{NO_PARENT!r} is reserved as parent row 0")
        if len(set(self.parent_tokens)) != len(self.parent_tokens):
            raise ValueError("parent_tokens repeat")
        if len(set(self.strain_token_vocab)) != len(self.strain_token_vocab):
            raise ValueError("strain_token_vocab repeats")
        return self


@dataclass(frozen=True)
class GenotypeBatch:
    """What the perturbation operator reads for a batch of ``B`` strains over ``|G|`` genes.

    ``X`` ``[|G|, d]`` is the reference token matrix, shared by every strain (one encoder
    pass per batch). ``change_types`` ``[B, |G|]`` holds ``CHANGE_TYPE_INDEX`` values,
    ``mask`` ``[B, |G|]`` is ``change_types != reference``, ``parent`` ``[B, |G|]`` the
    allele-parent row. ``X_strain`` ``[B, |G|, d]`` is ``X + T[c] + A[parent]``, the
    realized tokens. ``perturbation_indices`` and ``perturbation_indices_batch`` flatten
    the mask in the layout ``EquivariantPerturbationTransform.forward`` takes.
    ``strain_tokens`` ``[B, K_max, d]`` with ``strain_token_mask`` ``[B, K_max]`` (True =
    padding) are the strain-level tokens. ``off_graph_marks`` lists, per strain, every
    mark whose gene has no row in the graph (an accessory ORF, a reporter), so nothing is
    dropped without a trace.
    """

    X: torch.Tensor
    X_strain: torch.Tensor
    change_types: torch.Tensor
    mask: torch.Tensor
    parent: torch.Tensor
    perturbation_indices: torch.Tensor
    perturbation_indices_batch: torch.Tensor
    strain_tokens: torch.Tensor
    strain_token_mask: torch.Tensor
    off_graph_marks: tuple[tuple[tuple[str, ChangeType], ...], ...]


class GenotypeEncoder(nn.Module):
    """Reference tokens from a sequence encoder, plus learned marks for what changed.

    Args:
        config: widths and vocabularies.
        gene_order: the graph's gene ids in row order (``cell_graph["gene"].node_ids``).
        sequence_encoder: ``f_theta``; its ``dim`` is projected to ``d_model``.
        promoter_encoder: ``g_phi``; required by any promoter mark, otherwise unused.
        positions: ``[|G|, 2]`` with column 0 the integer chromosome (0 = mitochondrion)
            and column 1 the gene midpoint divided by the chromosome length; required
            when ``config.use_position`` is set.
    """

    def __init__(
        self,
        config: GenotypeEncoderConfig,
        gene_order: Sequence[str],
        sequence_encoder: SequenceEncoder,
        promoter_encoder: PromoterEncoder | None = None,
        positions: torch.Tensor | None = None,
    ) -> None:
        """Build the projection, the mark embeddings and the optional position terms."""
        super().__init__()
        if len(set(gene_order)) != len(gene_order):
            raise ValueError("gene_order repeats a gene id")
        self.config = config
        self.gene_order = tuple(gene_order)
        self.gene_index = {g: i for i, g in enumerate(self.gene_order)}
        self.sequence_encoder = sequence_encoder
        self.promoter_encoder = promoter_encoder
        d = config.d_model
        self.proj = nn.Linear(sequence_encoder.dim, d)
        self.change_embedding = nn.Embedding(len(ChangeType), d)
        with torch.no_grad():
            self.change_embedding.weight[
                CHANGE_TYPE_INDEX[ChangeType.reference]
            ].zero_()
        self.parent_index = {
            NO_PARENT: 0,
            **{p: i + 1 for i, p in enumerate(config.parent_tokens)},
        }
        self.parent_embedding = nn.Embedding(len(self.parent_index), d, padding_idx=0)
        self.strain_token_index = {
            t: i for i, t in enumerate(config.strain_token_vocab)
        }
        self.strain_embedding = nn.Embedding(max(len(self.strain_token_index), 1), d)
        self.positions: torch.Tensor | None
        if config.use_position:
            if positions is None or tuple(positions.shape) != (len(self.gene_order), 2):
                raise ValueError(
                    "use_position needs positions of shape "
                    f"[{len(self.gene_order)}, 2], got "
                    f"{None if positions is None else tuple(positions.shape)}"
                )
            self.register_buffer("positions", positions.clone())
            self.chromosome_embedding = nn.Embedding(config.n_chromosomes, d)
            self.coordinate_proj = nn.Linear(1, d)
        else:
            self.positions = None

    @property
    def n_genes(self) -> int:
        """Rows of the gene graph, ``|G|``."""
        return len(self.gene_order)

    def reference_tokens(self) -> torch.Tensor:
        """``X`` ``[|G|, d]``: the projected encoder feature, plus position if enabled."""
        X = cast(
            torch.Tensor, self.proj(self.sequence_encoder.encode_genes(self.gene_order))
        )
        if self.positions is not None:
            chrom = self.positions[:, 0].long()
            coord = self.positions[:, 1:2].to(X.dtype)
            X = X + self.chromosome_embedding(chrom) + self.coordinate_proj(coord)
        return X

    def forward(self, token_sets: Sequence[GenotypeTokenSet]) -> GenotypeBatch:
        """Marks of a batch of strains as the tensors of :class:`GenotypeBatch`."""
        if not token_sets:
            raise ValueError("a batch needs at least one token set")
        B, G = len(token_sets), self.n_genes
        device = self.change_embedding.weight.device
        ref = CHANGE_TYPE_INDEX[ChangeType.reference]
        c = torch.full((B, G), ref, dtype=torch.long, device=device)
        parent = torch.zeros((B, G), dtype=torch.long, device=device)
        off_graph: list[tuple[tuple[str, ChangeType], ...]] = []
        for b, ts in enumerate(token_sets):
            missing: list[tuple[str, ChangeType]] = []
            for gene, change, par in zip(
                ts.gene_ids, ts.change_types, ts.allele_parent
            ):
                if change is ChangeType.promoter and self.promoter_encoder is None:
                    raise ValueError(
                        f"strain {b}: promoter mark on {gene!r} but no promoter encoder"
                    )
                row = self.gene_index.get(gene)
                if row is None:
                    missing.append((gene, change))
                    continue
                c[b, row] = CHANGE_TYPE_INDEX[change]
                parent[b, row] = self.parent_index[par]
            off_graph.append(tuple(missing))
        mask = c != ref
        X = self.reference_tokens()
        X_strain = (
            X.unsqueeze(0) + self.change_embedding(c) + self.parent_embedding(parent)
        )
        batch_idx, gene_idx = mask.nonzero(as_tuple=True)
        strain_tokens, strain_mask = self._strain_tokens(token_sets, device)
        return GenotypeBatch(
            X=X,
            X_strain=X_strain,
            change_types=c,
            mask=mask,
            parent=parent,
            perturbation_indices=gene_idx,
            perturbation_indices_batch=batch_idx,
            strain_tokens=strain_tokens,
            strain_token_mask=strain_mask,
            off_graph_marks=tuple(off_graph),
        )

    def _strain_tokens(
        self, token_sets: Sequence[GenotypeTokenSet], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = len(token_sets)
        k_max = max(len(ts.strain_tokens) for ts in token_sets)
        ids = torch.zeros((B, k_max), dtype=torch.long, device=device)
        pad = torch.ones((B, k_max), dtype=torch.bool, device=device)
        for b, ts in enumerate(token_sets):
            for k, tok in enumerate(ts.strain_tokens):
                ids[b, k] = self.strain_token_index[tok]
                pad[b, k] = False
        tokens = self.strain_embedding(ids) * (~pad).unsqueeze(-1).to(
            self.strain_embedding.weight.dtype
        )
        return tokens, pad
