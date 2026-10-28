# torchcell/data/representation_043.py
# [[torchcell.data.representation_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/representation_043.py
# Test file: tests/torchcell/data/test_representation_043.py
"""Resolve any 043 genotype to the token inputs the model consumes, as KEYS.

Design: ``notes-tex/multimodal/043-datasets-primer/sections/input-design.tex``. Every gene
of the shared gene index gets, per composite encoder (FUDT upstream, FUDT downstream,
ProtT5, CaLM), an integer id into that encoder's key vocabulary (the sorted sha256s of
the strings the encoder reads), or ``-1`` with a typed :class:`GapKind`. Vectors are read
lazily through :class:`AlleleVectorStore`; a key whose vector is missing at read time
raises. Nothing is filled in: a gene whose allele, window or block assignment is unknown
carries its gap, and what to do with it is a modeling decision.

The resolution per genotype class:

* Engineered deletions, the reference strain (an empty perturbation list, class
  ``reference``) and the promoter reporter: context ``S288C``; every gene carries
  its S288C keys; deletions are ``deletion`` marks; the reporter is an extra token whose
  upstream key is the variant's 1,003 bp construct window. A deleted gene whose released
  name has no position in the gene index (an ORF outside ``gene_set.json``, or a
  released symbol with no standard-name owner) is an ``off_index_gene`` extra token: a
  ``deletion`` change with every key ``-1`` and gap ``NO_S288C_KEY``. The record is kept
  and the deletion stays visible to the model as a typed gap; the genome never gates a
  record. A ``released_identifier`` perturbation (the source's own gene name, not
  resolved at load time) takes its change type from ``released_kind``
  (:func:`change_type_of_perturbation`).
* Natural isolates, route ``A`` (the main design): context = the isolate; every gene
  carries the isolate's own keys (genes outside the released allele set and genes the
  record calls absent are gaps); the operator mask holds the engineered edits only (the
  HO deletion). Route ``B`` (the interim arm): context ``S288C``; S288C keys except the
  edit list, the genes whose change type is ``variant`` or ``loss_of_function``, which
  carry the isolate's keys; the operator mask holds every mark. Both routes return the
  full change-type vector under the precedence rule and the isolate's accessory ORFs as
  extra tokens.
* Segregants: context ``S288C``; the definite rule assigns a gene to a parent when one
  CONFIDENT RUN of that parent contains the whole gene (``start_run <= start_gene`` and
  ``end_gene <= end_run``). A confident run is a maximal run of consecutive blocks of one
  parent, every block at posterior ``>= posterior_min``: the loaders split a parent's run
  wherever the posterior level changes (Boocock's de novo encoding) or a marker gap
  occurs, so a single block is an encoding unit, not a haplotype. ``rule="single_block"``
  is the literal one-block form, kept to measure the difference. An assigned gene
  carries its parent's keys (BY = S288C); a gene assigned to a non-BY parent is a
  ``variant`` mark with that parent as allele parent; an unassigned gene is a gap with
  its reason (no block, a block below the posterior, or straddling runs).

Position features per gene: chromosome index (0 = mitochondrion, 1-16), relative
midpoint on the reference, chromosome copy number relative to the basal ploidy (from the
isolate descriptor's ``chromosome_copy_changes``; 1 for every other class), and gene copy
number (the pangenome matrix for an isolate or a segregant parent; 1 for S288C).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from torchcell.models.genotype_tokens_043 import (
    CHANGE_PRECEDENCE,
    CHANGE_TYPE_INDEX,
    PERTURBATION_TYPE_TO_CHANGE,
    ChangeType,
)

#: The composite encoders, in the column order of every key array.
ENCODERS: tuple[str, ...] = (
    "fudt_species_upstream",
    "fudt_species_downstream",
    "prot_t5_xl_uniref50",
    "calm",
)
N_ENC = len(ENCODERS)
S288C_CONTEXT = "S288C"
#: Chromosome token -> index; 0 is the mitochondrion.
CHROMOSOME_INDEX: dict[str, int] = {
    **{
        f"chr{r}": i + 1
        for i, r in enumerate(
            "I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI".split()
        )
    },
    "chrmt": 0,
}

IntArray = npt.NDArray[np.int64]
I8Array = npt.NDArray[np.int8]
F32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


class GapKind(IntEnum):
    """Why one (gene, encoder) slot has no key."""

    NONE = 0
    NO_ISOLATE_ALLELE = 1  # the gene is not in the context's released allele set
    LOCUS_GAP = 2  # the CDS was not located on the isolate assembly (window)
    FLANK_SHORT = 3  # the flank on the contig is shorter than the window
    SPLICE_UNVERIFIED = 4  # an intron gene's allele could not be spliced
    EXON_BOUNDARY_IN_GAP = 5
    ABSENT_IN_CONTEXT = 6  # route A: the record calls the gene absent
    SEGREGANT_UNASSIGNED = 7  # no confident run of either parent holds the gene
    NO_S288C_KEY = 8  # the S288C reference context has no key
    ACCESSORY_WITHOUT_ROW = 9  # a present accessory ORF with no row for the isolate
    MISSING_REPORTER_WINDOW = 10  # a promoter variant with no construct window
    #: the input exists but carries a character outside the encoder's alphabet (an IUPAC
    #: heterozygosity code in a CDS or window, an X/B/Z/J residue from an ambiguous
    #: codon); never resolved to a base, never embedded
    AMBIGUOUS_SEQUENCE = 11


#: Gap strings of the isolate index -> GapKind.
INDEX_GAP: dict[str, GapKind] = {
    "locus_gap": GapKind.LOCUS_GAP,
    "flank_short": GapKind.FLANK_SHORT,
    "splice_unverified": GapKind.SPLICE_UNVERIFIED,
    "exon_boundary_in_alignment_gap": GapKind.EXON_BOUNDARY_IN_GAP,
    "ambiguous_sequence": GapKind.AMBIGUOUS_SEQUENCE,
}


_ACCESSORY_GAP = (int(GapKind.ACCESSORY_WITHOUT_ROW),) * 4
#: An engineered deletion of a gene with no index position has no S288C key in any block.
_OFF_INDEX_GAP = (int(GapKind.NO_S288C_KEY),) * 4


class PositionGap(IntEnum):
    """Why a gene's position or copy-number feature is unknown."""

    NONE = 0
    COPY_NUMBER_UNRELEASED = 1  # the pangenome matrix carries no value for the gene
    PLOIDY_UNREPORTED = 2  # aneuploid chromosome of an isolate with no basal ploidy
    SEGREGANT_UNASSIGNED = 3  # gene copy of an unassigned segregant gene


class SegregantReason(IntEnum):
    """Outcome of the definite rule for one gene."""

    ASSIGNED = 0
    NO_BLOCK = 1  # no block of either parent overlaps the gene
    LOW_POSTERIOR = 2  # a block below the posterior threshold overlaps the gene
    STRADDLE = 3  # confident runs overlap the gene but none contains it


class ContextState(IntEnum):
    """A gene's state in a reference context."""

    PRESENT = 0
    ABSENT = 1  # pangenome matrix: absent
    NOT_IN_MATRIX = 2  # allele released, no presence/copy value
    NO_ALLELE = 3  # not in the released allele set


class RepresentationConfig(BaseModel):
    """Choices of the resolver."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route: Literal["A", "B"] = Field(
        default="A",
        description="isolates: A = own reference context, B = S288C plus the edit list",
    )
    segregant_posterior_min: float = Field(default=0.9, ge=0.0, le=1.0)
    segregant_rule: Literal["confident_run", "single_block"] = "confident_run"


# --------------------------------------------------------------------------- #
# Keys and vectors
# --------------------------------------------------------------------------- #
class KeyVocabulary:
    """The sorted sha256 keys of one encoder; a key's id is its position."""

    def __init__(self, keys: Sequence[str] | npt.NDArray[Any]) -> None:
        """``keys`` need not be sorted or unique; they are made so."""
        arr = np.unique(np.asarray(keys, dtype="S64"))
        self.keys: npt.NDArray[Any] = arr

    def __len__(self) -> int:
        """Keys held."""
        return int(self.keys.shape[0])

    def ids(self, keys: Sequence[str | None] | npt.NDArray[Any]) -> IntArray:
        """Ids of ``keys``; ``None`` or an empty key is -1; an unknown key raises.

        ``keys`` is a sequence of hex strings, or an ``S``-dtype array (fast path).
        """
        if isinstance(keys, np.ndarray) and keys.dtype.kind == "S":
            raw = keys.astype("S64", copy=False)
        else:
            raw = np.asarray(
                [b"" if k is None else str(k).encode() for k in keys], dtype="S64"
            )
        out = np.full(raw.shape[0], -1, dtype=np.int64)
        present = raw != b""
        if present.any():
            pos = np.searchsorted(self.keys, raw[present])
            pos_c = np.minimum(pos, len(self) - 1)
            ok = self.keys[pos_c] == raw[present]
            if not ok.all():
                bad = raw[present][~ok][:3]
                raise KeyError(f"{int((~ok).sum())} keys not in the vocabulary: {bad}")
            out[present] = pos_c
        return out

    def key(self, i: int) -> str:
        """The sha256 of id ``i``."""
        return bytes(self.keys[i]).decode()


class AlleleVectorStore:
    """Lazy reader of one encoder's ``alleles/<encoder>.h5`` table by vocabulary id.

    The id -> row map is built on the first read from the table's ``sha256`` column. A
    requested id with no row raises ``KeyError``: nothing is substituted.
    """

    def __init__(self, h5_path: str, vocab: KeyVocabulary) -> None:
        """Nothing is read until :meth:`vectors`."""
        self.h5_path = h5_path
        self.vocab = vocab
        self._row: IntArray | None = None
        self._file: Any = None

    def _open(self) -> None:
        import h5py

        self._file = h5py.File(self.h5_path, "r")
        keys = self._file["sha256"][:]
        row = np.full(len(self.vocab), -1, dtype=np.int64)
        pos = np.searchsorted(self.vocab.keys, keys)
        pos_c = np.minimum(pos, len(self.vocab) - 1)
        hit = self.vocab.keys[pos_c] == keys
        row[pos_c[hit]] = np.nonzero(hit)[0]
        self._row = row

    def vectors(self, ids: IntArray) -> npt.NDArray[np.float16]:
        """``[len(ids), dim]`` fp16; ``-1`` or an id without a row raises."""
        if self._row is None:
            self._open()
        assert self._row is not None
        if (ids < 0).any():
            raise KeyError("a gap id (-1) was passed to vectors(); mask it first")
        rows = self._row[ids]
        if (rows < 0).any():
            missing = [self.vocab.key(int(i)) for i in ids[rows < 0][:3]]
            raise KeyError(
                f"{int((rows < 0).sum())} keys have no vector in {self.h5_path}: "
                f"{missing}"
            )
        uniq, inverse = np.unique(rows, return_inverse=True)  # h5py needs increasing
        block = np.asarray(self._file["embedding"][uniq], dtype=np.float16)
        return block[inverse.reshape(-1)]


# --------------------------------------------------------------------------- #
# Change types
# --------------------------------------------------------------------------- #
_RANK = np.full(len(CHANGE_TYPE_INDEX), np.iinfo(np.int8).max, dtype=np.int64)
for _r, _c in enumerate(CHANGE_PRECEDENCE):
    _RANK[CHANGE_TYPE_INDEX[_c]] = _r
_REF = CHANGE_TYPE_INDEX[ChangeType.reference]


def change_type_of(perturbation_type: str) -> ChangeType:
    """The change type of a stored ``perturbation_type``; an unknown one raises."""
    if perturbation_type not in PERTURBATION_TYPE_TO_CHANGE:
        raise ValueError(f"unexpected perturbation_type {perturbation_type!r}")
    return PERTURBATION_TYPE_TO_CHANGE[perturbation_type]


#: ``released_kind`` of a ``released_identifier`` perturbation -> its change type. A kind
#: not listed raises: a heterozygous deletion, a tag or an insertion is not a deletion,
#: and the 043 change-type alphabet has no member for it yet.
RELEASED_KIND_TO_CHANGE: dict[str, ChangeType] = {
    "deletion": ChangeType.deletion,
    "truncation": ChangeType.deletion,
}


def change_type_of_perturbation(perturbation: Mapping[str, Any]) -> ChangeType:
    """The change type of one stored perturbation dict.

    ``released_identifier`` (``ReleasedIdentifierPerturbation``) carries the source's own
    gene name and the kind of change the source reports in ``released_kind``; every other
    type maps through :func:`change_type_of`. An unknown type or kind raises.
    """
    t = perturbation["perturbation_type"]
    if t != "released_identifier":
        return change_type_of(t)
    kind = perturbation["released_kind"]
    if kind not in RELEASED_KIND_TO_CHANGE:
        raise ValueError(
            f"released_identifier {perturbation['systematic_gene_name']!r} has "
            f"released_kind {kind!r}, which has no 043 change type"
        )
    return RELEASED_KIND_TO_CHANGE[kind]


def resolve_change_types(
    marks: Sequence[tuple[int, ChangeType]], n_genes: int
) -> I8Array:
    """Per gene, the change-type index under the precedence rule (0 = reference)."""
    out = np.full(n_genes, _REF, dtype=np.int8)
    for gene, change in marks:
        c = CHANGE_TYPE_INDEX[change]
        if out[gene] == _REF or _RANK[c] < _RANK[out[gene]]:
            out[gene] = c
    return out


# --------------------------------------------------------------------------- #
# Segregant definite rule
# --------------------------------------------------------------------------- #
class Block(BaseModel):
    """The fields of a ``HaplotypeBlock`` the rule reads."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    chromosome: str
    start: int
    end: int
    parent: Literal[1, 2]
    posterior: float


def _runs(
    blocks: Sequence[Block], posterior_min: float, single_block: bool
) -> dict[str, tuple[IntArray, IntArray, IntArray, IntArray, IntArray]]:
    """Per chromosome: confident runs (starts, ends, parents) and low blocks (starts, ends)."""
    by_chrom: dict[str, list[Block]] = {}
    for b in blocks:
        by_chrom.setdefault(b.chromosome, []).append(b)
    out = {}
    for chrom, bs in by_chrom.items():
        bs = sorted(bs, key=lambda b: b.start)
        rs: list[list[int]] = []
        low: list[tuple[int, int]] = []
        prev_conf = False
        for b in bs:
            if b.posterior < posterior_min:
                low.append((b.start, b.end))
                prev_conf = False
                continue
            if prev_conf and not single_block and rs[-1][2] == b.parent:
                rs[-1][1] = b.end
            else:
                rs.append([b.start, b.end, b.parent])
            prev_conf = True
        r = np.asarray(rs, dtype=np.int64).reshape(-1, 3)
        lo = np.asarray(low, dtype=np.int64).reshape(-1, 2)
        out[chrom] = (r[:, 0], r[:, 1], r[:, 2], lo[:, 0], lo[:, 1])
    return out


def _overlaps(
    starts: IntArray, ends: IntArray, g0: IntArray, g1: IntArray
) -> BoolArray:
    """Whether each gene [g0, g1] overlaps any of the sorted disjoint intervals."""
    if starts.size == 0:
        return np.zeros(g0.shape[0], dtype=bool)
    n_start_le_end = np.searchsorted(starts, g1, side="right")
    n_end_lt_start = np.searchsorted(ends, g0, side="left")
    return np.asarray(n_start_le_end > n_end_lt_start)


def assign_segregant_genes(
    gene_chrom: Sequence[str],
    gene_start: IntArray,
    gene_end: IntArray,
    blocks: Sequence[Block],
    posterior_min: float = 0.9,
    rule: Literal["confident_run", "single_block"] = "confident_run",
) -> tuple[I8Array, I8Array]:
    """Per gene: (parent 0/1/2, :class:`SegregantReason`) under the definite rule."""
    n = len(gene_chrom)
    parent = np.zeros(n, dtype=np.int8)
    reason = np.full(n, SegregantReason.NO_BLOCK, dtype=np.int8)
    runs = _runs(blocks, posterior_min, rule == "single_block")
    chrom_arr = np.asarray(gene_chrom)
    for chrom, (rs, re, rp, ls, le) in runs.items():
        sel = np.nonzero(chrom_arr == chrom)[0]
        if sel.size == 0:
            continue
        g0, g1 = gene_start[sel], gene_end[sel]
        contained = np.zeros(sel.size, dtype=bool)
        par: I8Array = np.zeros(sel.size, dtype=np.int8)
        if rs.size:
            k = np.searchsorted(rs, g0, side="right") - 1
            kc = np.maximum(k, 0)
            contained = (k >= 0) & (g1 <= re[kc])
            par = np.where(contained, rp[kc], 0).astype(np.int8)
        low = _overlaps(ls, le, g0, g1)
        conf = _overlaps(rs, re, g0, g1)
        rsn = np.where(
            contained,
            SegregantReason.ASSIGNED,
            np.where(
                low,
                SegregantReason.LOW_POSTERIOR,
                np.where(conf, SegregantReason.STRADDLE, SegregantReason.NO_BLOCK),
            ),
        )
        parent[sel] = par
        reason[sel] = rsn.astype(np.int8)
    return parent, reason


# --------------------------------------------------------------------------- #
# Position
# --------------------------------------------------------------------------- #
def chromosome_copy_vector(
    ploidy: int | None, changes: Sequence[Mapping[str, int]] | None
) -> tuple[F32Array, BoolArray]:
    """Relative copy number per chromosome index (17,), and which are unknown.

    ``(ploidy + change) / ploidy`` for a changed chromosome, 1 elsewhere. ``changes`` is
    ``[]`` for a euploid isolate (all 1, whatever the ploidy). A changed chromosome of an
    isolate whose ploidy is unreported is unknown (NaN, flagged). ``changes`` of None
    means the annotation itself is missing: every nuclear chromosome is unknown.
    """
    copy = np.ones(17, dtype=np.float32)
    unknown = np.zeros(17, dtype=bool)
    if changes is None:
        copy[1:] = np.nan
        unknown[1:] = True
        return copy, unknown
    for ch in changes:
        c = int(ch["chromosome"])
        if ploidy is None:
            copy[c] = np.nan
            unknown[c] = True
        else:
            copy[c] = (ploidy + int(ch["change"])) / ploidy
    return copy, unknown


class GenePositions(BaseModel):
    """Reference positions of the gene index (SGD coordinates)."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    chromosome: tuple[str, ...]
    start: IntArray
    end: IntArray
    chrom_index: I8Array
    rel_midpoint: F32Array

    @classmethod
    def build(
        cls,
        coords: Sequence[tuple[str, int, int]],
        chromosome_lengths: Mapping[str, int],
    ) -> GenePositions:
        """From (chromosome token, start, end) per gene and the chromosome lengths."""
        chrom = tuple(c for c, _, _ in coords)
        start = np.asarray([s for _, s, _ in coords], dtype=np.int64)
        end = np.asarray([e for _, _, e in coords], dtype=np.int64)
        length = np.asarray([chromosome_lengths[c] for c in chrom], dtype=np.float64)
        return cls(
            chromosome=chrom,
            start=start,
            end=end,
            chrom_index=np.asarray([CHROMOSOME_INDEX[c] for c in chrom], dtype=np.int8),
            rel_midpoint=(0.5 * (start + end) / length).astype(np.float32),
        )


# --------------------------------------------------------------------------- #
# Context tables and the resolver
# --------------------------------------------------------------------------- #
class AccessoryRow(BaseModel):
    """One accessory ORF of one isolate context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key_ids: tuple[int, int, int, int]
    gaps: tuple[int, int, int, int]
    copy_number: float | None


class ContextTables(BaseModel):
    """Per reference context (row 0 = S288C): keys, gaps, gene copy, state."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    context_ids: tuple[str, ...]
    key_ids: npt.NDArray[np.int32]  # [C, G, 4]
    key_gap: I8Array  # [C, G, 4]
    gene_copy: F32Array  # [C, G]
    state: I8Array  # [C, G]
    accessory: dict[str, dict[str, AccessoryRow]] = Field(default_factory=dict)

    def index(self, context_id: str) -> int:
        """Row of a context; an unknown one raises."""
        try:
            return self.context_ids.index(context_id)
        except ValueError as exc:
            raise KeyError(f"no reference context {context_id!r}") from exc


class ExtraToken(BaseModel):
    """An off-graph token: an accessory ORF or the promoter reporter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["accessory_orf", "reporter", "off_index_gene"]
    feature_id: str
    change_type: int
    key_ids: tuple[int, int, int, int]
    gaps: tuple[int, int, int, int]
    copy_number: float | None


class ReporterKeys(BaseModel):
    """Key ids of the reporter: one upstream window per variant, three constants."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_id: str
    upstream_by_variant: dict[str, int]
    downstream: int
    protein: int
    calm: int


class RecordRepresentation(BaseModel):
    """Everything the input side needs for one record (keys, not vectors)."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    genotype_class: Literal["reference", "deletion", "isolate", "segregant", "promoter"]
    context_id: str
    route: Literal["A", "B", "na"]
    key_ids: IntArray  # [G, 4], -1 = gap
    key_gap: I8Array  # [G, 4]
    change_type: I8Array  # [G]
    allele_parent: I8Array  # [G], -1 = none
    operator_mask: BoolArray  # [G]
    chrom_index: I8Array
    rel_midpoint: F32Array
    chrom_copy: F32Array
    gene_copy: F32Array
    position_gap: I8Array
    extra_tokens: tuple[ExtraToken, ...] = ()
    segregant_reason: I8Array | None = None
    n_marked_outside_index: int = 0

    @property
    def marked(self) -> IntArray:
        """Gene indices whose change type is not reference."""
        marked: IntArray = np.nonzero(self.change_type != _REF)[0]
        return marked


class RepresentationResolver:
    """Builds :class:`RecordRepresentation`s from parsed genotypes and context tables."""

    def __init__(
        self,
        config: RepresentationConfig,
        positions: GenePositions,
        tables: ContextTables,
        reporter: ReporterKeys | None,
        parent_vocab: Sequence[str],
    ) -> None:
        """``parent_vocab`` indexes ``allele_parent``."""
        self.config = config
        self.pos = positions
        self.tables = tables
        self.reporter = reporter
        self.parent_id = {p: i for i, p in enumerate(parent_vocab)}
        self.n_genes = int(positions.start.shape[0])
        if tables.key_ids.shape[1] != self.n_genes:
            raise ValueError("context tables and gene positions disagree on the index")
        self.s288c = tables.index(S288C_CONTEXT)

    # ---- shared pieces
    def _ctx(self, c: int) -> tuple[IntArray, I8Array]:
        return (self.tables.key_ids[c].astype(np.int64), self.tables.key_gap[c].copy())

    def _base(
        self,
        cls: Literal["reference", "deletion", "isolate", "segregant", "promoter"],
        context_id: str,
        route: Literal["A", "B", "na"],
        key_ids: IntArray,
        key_gap: I8Array,
        change: I8Array,
        mask: BoolArray,
        chrom_copy: F32Array | None = None,
        gene_copy: F32Array | None = None,
        position_gap: I8Array | None = None,
        allele_parent: I8Array | None = None,
        extra: tuple[ExtraToken, ...] = (),
        segregant_reason: I8Array | None = None,
        outside: int = 0,
    ) -> RecordRepresentation:
        g = self.n_genes
        return RecordRepresentation(
            genotype_class=cls,
            context_id=context_id,
            route=route,
            key_ids=key_ids,
            key_gap=key_gap,
            change_type=change,
            allele_parent=(
                np.full(g, -1, dtype=np.int8)
                if allele_parent is None
                else allele_parent
            ),
            operator_mask=mask,
            chrom_index=self.pos.chrom_index,
            rel_midpoint=self.pos.rel_midpoint,
            chrom_copy=np.ones(g, dtype=np.float32)
            if chrom_copy is None
            else chrom_copy,
            gene_copy=np.ones(g, dtype=np.float32) if gene_copy is None else gene_copy,
            position_gap=(
                np.zeros(g, dtype=np.int8) if position_gap is None else position_gap
            ),
            extra_tokens=extra,
            segregant_reason=segregant_reason,
            n_marked_outside_index=outside,
        )

    # ---- classes
    def deletion(
        self,
        marks: Sequence[tuple[int, ChangeType]],
        outside: int = 0,
        off_index: Sequence[tuple[str, ChangeType]] = (),
    ) -> RecordRepresentation:
        """Engineered deletions on S288C; no mark at all is the reference strain.

        ``off_index`` lists the marked genes whose released name has no index position
        (name, change type); each becomes an ``off_index_gene`` extra token with every
        key ``-1`` and gap ``NO_S288C_KEY``. ``outside`` counts them (and any other mark
        without a position) on ``n_marked_outside_index``.
        """
        if len(off_index) > outside:
            raise ValueError("more off-index tokens than marks outside the index")
        change = resolve_change_types(marks, self.n_genes)
        keys, gaps = self._ctx(self.s288c)
        extra = tuple(
            ExtraToken(
                kind="off_index_gene",
                feature_id=name,
                change_type=CHANGE_TYPE_INDEX[ct],
                key_ids=(-1, -1, -1, -1),
                gaps=_OFF_INDEX_GAP,
                # the S288C copy number of a deleted on-graph gene's token (gene_copy 1)
                copy_number=1.0,
            )
            for name, ct in off_index
        )
        return self._base(
            "deletion" if marks or outside else "reference",
            S288C_CONTEXT,
            "na",
            keys,
            gaps,
            change,
            change != _REF,
            extra=extra,
            outside=outside,
        )

    def promoter(self, variant_id: str) -> RecordRepresentation:
        """A Vaishnav reporter record: S288C keys, one reporter token."""
        if self.reporter is None:
            raise ValueError("a promoter record needs ReporterKeys")
        up = self.reporter.upstream_by_variant.get(variant_id)
        token = ExtraToken(
            kind="reporter",
            feature_id=self.reporter.feature_id,
            change_type=CHANGE_TYPE_INDEX[ChangeType.promoter],
            key_ids=(
                -1 if up is None else up,
                self.reporter.downstream,
                self.reporter.protein,
                self.reporter.calm,
            ),
            gaps=(
                int(
                    GapKind.NONE if up is not None else GapKind.MISSING_REPORTER_WINDOW
                ),
                0,
                0,
                0,
            ),
            copy_number=None,
        )
        keys, gaps = self._ctx(self.s288c)
        change = np.full(self.n_genes, _REF, dtype=np.int8)
        return self._base(
            "promoter",
            S288C_CONTEXT,
            "na",
            keys,
            gaps,
            change,
            change != _REF,
            extra=(token,),
        )

    def isolate(
        self,
        context_id: str,
        marks: Sequence[tuple[int, ChangeType]],
        ploidy: int | None,
        chromosome_copy_changes: Sequence[Mapping[str, int]] | None,
        accessory: Sequence[tuple[str, float | None]],
        outside: int = 0,
    ) -> RecordRepresentation:
        """A natural isolate under the configured route."""
        c = self.tables.index(context_id)
        change = resolve_change_types(marks, self.n_genes)
        iso_keys, iso_gaps = self._ctx(c)
        if self.config.route == "A":
            keys, gaps = iso_keys, iso_gaps
            absent = change == CHANGE_TYPE_INDEX[ChangeType.absent]
            keys[absent] = -1
            gaps[absent] = GapKind.ABSENT_IN_CONTEXT
            mask = change == CHANGE_TYPE_INDEX[ChangeType.deletion]
            ctx_id = context_id
        else:
            keys, gaps = self._ctx(self.s288c)
            edit = (change == CHANGE_TYPE_INDEX[ChangeType.variant]) | (
                change == CHANGE_TYPE_INDEX[ChangeType.loss_of_function]
            )
            keys[edit] = iso_keys[edit]
            gaps[edit] = iso_gaps[edit]
            mask = change != _REF
            ctx_id = S288C_CONTEXT
        copy_vec, unknown = chromosome_copy_vector(ploidy, chromosome_copy_changes)
        chrom_copy = copy_vec[self.pos.chrom_index]
        pgap = np.where(
            unknown[self.pos.chrom_index], PositionGap.PLOIDY_UNREPORTED, 0
        ).astype(np.int8)
        gene_copy = self.tables.gene_copy[c].copy()
        pgap = np.where(
            np.isnan(gene_copy) & (pgap == 0), PositionGap.COPY_NUMBER_UNRELEASED, pgap
        ).astype(np.int8)
        extra = []
        acc = self.tables.accessory.get(context_id, {})
        present = CHANGE_TYPE_INDEX[ChangeType.present]
        for orf, copy_number in accessory:
            row = acc.get(orf)
            extra.append(
                ExtraToken(
                    kind="accessory_orf",
                    feature_id=orf,
                    change_type=present,
                    key_ids=(-1, -1, -1, -1) if row is None else row.key_ids,
                    gaps=_ACCESSORY_GAP if row is None else row.gaps,
                    copy_number=copy_number,
                )
            )
        return self._base(
            "isolate",
            ctx_id,
            self.config.route,
            keys,
            gaps,
            change,
            mask,
            chrom_copy=chrom_copy.astype(np.float32),
            gene_copy=gene_copy,
            position_gap=pgap,
            extra=tuple(extra),
            outside=outside,
        )

    def segregant(
        self,
        blocks: Sequence[Block],
        parents: tuple[tuple[str, str | None], tuple[str, str | None]],
    ) -> RecordRepresentation:
        """A segregant: (name, Peter id or None for BY) per parent."""
        parent, reason = assign_segregant_genes(
            self.pos.chromosome,
            self.pos.start,
            self.pos.end,
            blocks,
            self.config.segregant_posterior_min,
            self.config.segregant_rule,
        )
        g = self.n_genes
        keys = np.full((g, N_ENC), -1, dtype=np.int64)
        gaps = np.full((g, N_ENC), GapKind.SEGREGANT_UNASSIGNED, dtype=np.int8)
        gene_copy = np.full(g, np.nan, dtype=np.float32)
        pgap = np.full(g, PositionGap.SEGREGANT_UNASSIGNED, dtype=np.int8)
        change = np.full(g, _REF, dtype=np.int8)
        allele_parent = np.full(g, -1, dtype=np.int8)
        variant = CHANGE_TYPE_INDEX[ChangeType.variant]
        for p, (name, peter_id) in enumerate(parents, start=1):
            if name not in self.parent_id:
                raise KeyError(
                    f"segregant parent {name!r} not in the parent vocabulary"
                )
            sel = parent == p
            c = self.s288c if peter_id is None else self.tables.index(peter_id)
            keys[sel] = self.tables.key_ids[c][sel]
            gaps[sel] = self.tables.key_gap[c][sel]
            gene_copy[sel] = self.tables.gene_copy[c][sel]
            pgap[sel] = np.where(
                np.isnan(self.tables.gene_copy[c][sel]),
                PositionGap.COPY_NUMBER_UNRELEASED,
                0,
            )
            if peter_id is not None:
                change[sel] = variant
                allele_parent[sel] = self.parent_id[name]
        return self._base(
            "segregant",
            S288C_CONTEXT,
            "na",
            keys,
            gaps,
            change,
            change != _REF,
            gene_copy=gene_copy,
            position_gap=pgap,
            allele_parent=allele_parent,
            segregant_reason=reason,
        )
