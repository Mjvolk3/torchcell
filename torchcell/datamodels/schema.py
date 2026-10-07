# torchcell/datamodels/schema
# [[torchcell.datamodels.schema]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/schema
# Test file: tests/torchcell/datamodels/test_schema.py

"""Pydantic data models for torchcell genotypes, environments, and phenotypes."""

import math
import re
from enum import StrEnum
from typing import Any, Literal, get_type_hints

from pydantic import BaseModel, Field, field_validator, model_validator
from sortedcontainers import SortedDict

from torchcell.datamodels.calmorph_labels import CALMORPH_LABELS, CALMORPH_STATISTICS
from torchcell.datamodels.pydant import ModelStrict
from torchcell.verification.sourced import ProvenanceGap, SourcedValue

# causes circular import
# from torchcell.datasets.dataset_registry import dataset_registry


class ProvenanceGapMixin(ModelStrict):
    """Mixin giving a model a ``provenance_gaps`` list of typed field-absences.

    Shared by ``Phenotype``, ``Environment``, and ``Compound`` so a field the source does
    not carry -- a phenotype ``n_samples`` a secondary curation layer dropped, an
    environment ``temperature`` YeastPhenome never recorded, a compound ``inchikey`` no
    resolver could map -- is a documented, typed ABSENCE (with a reason + ``looked_in``)
    rather than a guess or a silent None. Two invariants keep it honest and
    machine-checkable: (1) each ``gap.field`` names a real field on the concrete model
    (checked against ``model_fields``, so inherited fields resolve too); (2) a gapped
    field must be ``None`` -- you cannot both store a value and declare it missing.
    ``provenance_gaps`` itself cannot be gapped.
    """

    provenance_gaps: list[ProvenanceGap] = Field(
        default_factory=list,
        description="documented, typed ABSENCES of a sourced value for a field on this "
        "model (e.g. a phenotype n_samples, or an environment temperature the curation "
        "layer did not carry). An honest typed gap -- never a guess. The gapped field "
        "must be None (enforced below).",
    )

    @model_validator(mode="after")
    def validate_provenance_gaps(self) -> "ProvenanceGapMixin":
        """Each ProvenanceGap must name a real field on this model, and that field must
        be None (you cannot both store a value and declare it missing).
        """
        model_fields = type(self).model_fields
        for gap in self.provenance_gaps:
            if gap.field not in model_fields:
                raise ValueError(
                    f"provenance_gap field '{gap.field}' is not a field of "
                    f"{type(self).__name__}"
                )
            if gap.field == "provenance_gaps":
                raise ValueError("provenance_gaps cannot itself be gapped")
            if getattr(self, gap.field) is not None:
                raise ValueError(
                    f"field '{gap.field}' has a ProvenanceGap but is not None "
                    "(cannot both store a value and declare it missing)"
                )
        return self

    def gapped_fields(self) -> set[str]:
        """Names of the fields this record declares as typed absences."""
        return {gap.field for gap in self.provenance_gaps}


class HashableProvenanceGapMixin(ProvenanceGapMixin):
    """``ProvenanceGapMixin`` for a model that must stay hashable.

    A frozen pydantic model hashes the tuple of its field values, so a ``list`` field
    (``provenance_gaps``) makes it unhashable. ``Genotype.__eq__`` compares
    perturbations as a ``set``, so a gene-perturbation leaf that carries typed gaps
    must hash; it hashes its canonical JSON instead, which agrees with pydantic's
    field-wise ``__eq__`` (equal models dump to equal JSON). List this mixin FIRST in
    a leaf's bases: pydantic keeps the first ``__hash__`` it finds in the bases, and
    a frozen parent leaf carries a generated one.
    """

    def __hash__(self) -> int:
        """Hash the canonical JSON dump (the list fields are not hashable)."""
        return hash(self.model_dump_json())


def _require_value_or_gap(model: ProvenanceGapMixin, fields: tuple[str, ...]) -> None:
    """Raise unless every named field is set or carries a typed ``ProvenanceGap``.

    The strain-background contract (#500, #507): every element is either sourced or
    a declared absence, never a silent ``None``. An empty list counts as unset.
    """
    gapped = model.gapped_fields()
    for name in fields:
        value = getattr(model, name)
        if (value is None or value == []) and name not in gapped:
            raise ValueError(
                f"{type(model).__name__}.{name} is unset and carries no ProvenanceGap "
                "(an unsourced element must be a typed gap, never a silent None)"
            )
        if value == []:
            raise ValueError(
                f"{type(model).__name__}.{name} is an empty list; use None with a "
                "ProvenanceGap for an absent source"
            )


# Genotype
# --------------------------------------------------------------------------- #
# Strain background (issue #507, decided once for #500 #504 #505 #506).
#
# Every allele here is an edit against the S288C R64 reference assembly: the
# sequence torchcell's genome reads is S288C, so a BY4741 record differs from it at
# MAT (R64 is MATalpha), at four auxotrophic loci, and at the screened locus. The
# background is everything the strain carries that is CONSTANT across the
# collection and shared with the record's reference strain; the experiment's
# ``Genotype`` keeps only what the screen varies. It hangs off the reference genome
# (``StrainReferenceGenome``, not ``Genotype``) so pooled one-perturbation semantics,
# perturbation counts and every gene-keyed consumer are untouched, and so it is
# stored once per reference (the LMDB interns the whole reference; the graph writes
# it on the ``genome`` and ``experiment reference`` nodes), never once per record.
# Only the new classes below carry it, so no served dataset's closure moves until
# its loader opts in (``StrainEnvironmentResponseExperiment``).
# Design + worked examples: ``[[torchcell.datamodels.strain-background]]``.
# --------------------------------------------------------------------------- #
SYSTEMATIC_GENE_PATTERN = r"^(Y[A-P][LR]\d{3}[WC](-[A-Z])?|Q\d{4}|YNC[A-Q]\d{4}[WC])$"
"""A nuclear ORF / ncRNA / mitochondrial systematic name (the ``GenePerturbation``
pattern, restated so the background classes do not reach into that validator)."""


class MatingType(StrEnum):
    """Mating-type locus state. The R64 reference (S288C) is MATalpha, so a MATa
    strain differs from the reference IN SEQUENCE at MAT (chrIII), not only in label.
    """

    a = "a"
    alpha = "alpha"
    a_alpha = "a/alpha"


REFERENCE_MATING_TYPE: MatingType = MatingType.alpha
"""S288C R64 carries MATALPHA1 (YCR040W) and MATALPHA2 (YCR039C) at MAT."""


class Zygosity(StrEnum):
    """How many of a background's chromosome copies carry an allele.

    - ``haploid``: the only copy of a haploid genome.
    - ``homozygous``: both copies of a diploid.
    - ``heterozygous``: one copy of a diploid; the other copy is the R64 allele.
    """

    haploid = "haploid"
    homozygous = "homozygous"
    heterozygous = "heterozygous"


class AlleleEdit(StrEnum):
    """How a background allele's sequence differs from the R64 locus.

    - ``full_deletion``: the ORF removed with no marker left (the BY ``delta0``
      designer deletions: leu2-delta0, ura3-delta0, met15-delta0, lys2-delta0).
    - ``partial_deletion``: an internal deletion that leaves ORF flanks
      (his3-delta1).
    - ``cassette_replacement``: the ORF replaced by a cassette named in ``cassette``
      (can1-delta::STE2pr-Sp_his5, pdr1-delta::natMX).
    - ``sequence_variant``: an in-place change (a point or nonsense mutation).
    """

    full_deletion = "full_deletion"
    partial_deletion = "partial_deletion"
    cassette_replacement = "cassette_replacement"
    sequence_variant = "sequence_variant"


ALLELE_EDIT_SO: dict[AlleleEdit, tuple[str, str]] = {
    AlleleEdit.full_deletion: ("SO:0000159", "deletion"),
    AlleleEdit.partial_deletion: ("SO:0000159", "deletion"),
    AlleleEdit.cassette_replacement: ("SO:0000159", "deletion"),
    AlleleEdit.sequence_variant: ("SO:0001060", "sequence_variant"),
}
"""Sequence Ontology mechanism of each edit kind (the same pinned pairs the
gene-perturbation leaves use)."""


class GenomicSpan(ModelStrict):
    """A 1-based, end-inclusive interval on one chromosome of the R64 assembly.

    ``assembly`` names the R64 release the coordinates were read from (coordinates
    shift between releases), e.g. ``"R64-4-1"``.
    """

    chromosome: str
    start: int
    end: int
    assembly: str

    @model_validator(mode="after")
    def _check_span(self) -> "GenomicSpan":
        """Positions are 1-based and ordered; chromosome and assembly are named."""
        if self.start < 1 or self.end < self.start:
            raise ValueError(
                f"GenomicSpan needs 1 <= start <= end, got {self.start}..{self.end}"
            )
        if not self.chromosome.strip() or not self.assembly.strip():
            raise ValueError("GenomicSpan needs a chromosome and an assembly name")
        return self


class BackgroundAllele(ProvenanceGapMixin):
    """One allele a strain background carries, as an edit against R64.

    ``allele_name`` is the designation verbatim as the source writes it
    (``his3Δ1``, ``can1Δ::STE2pr-Sp_his5``); ``gene_name`` is the CURRENT R64
    standard name (``MET17`` for the ``met15Δ0`` allele); ``systematic_gene_name``
    is the R64 ORF the edit sits in. ``functional`` says whether the allele keeps the
    gene's function (False for every BY marker allele), which is what per-locus
    dosage reads. ``cassette`` is required for a ``cassette_replacement`` and
    forbidden otherwise.

    Sourcing contract: ``provenance`` (quote + sha256) or a typed ``ProvenanceGap`` on
    ``provenance``, never neither. A literature-standard allele whose source is not
    mirrored (the BY alleles before Brachmann 1998 is mirrored) is ASSERTED with a
    ``deferred_pending_source_review`` gap naming the paper that would close it, so the
    record carries the allele and says plainly that it is unverified. ``zygosity``
    likewise is set or gapped (Hoepfner's whi2 nonsense allele has no stated
    zygosity). ``deleted_span`` is optional (coordinates are rarely published).
    """

    systematic_gene_name: str
    gene_name: str
    allele_name: str
    edit: AlleleEdit
    functional: bool = Field(
        description="True if the allele keeps the gene's function; False for a null"
    )
    zygosity: Zygosity | None = Field(
        description="copies of the background carrying it; None only with a gap"
    )
    cassette: str | None = Field(
        default=None,
        description="cassette that replaced the ORF, verbatim (e.g. 'STE2pr-Sp_his5', "
        "'natMX', 'KlURA3'); set iff edit == cassette_replacement",
    )
    deleted_span: GenomicSpan | None = Field(
        default=None, description="removed R64 interval, when the source gives it"
    )
    provenance: list[SourcedValue] | None = Field(
        default=None,
        description="quotes that state this allele; None only with a gap on "
        "'provenance' (an asserted-but-unsourced allele)",
    )

    @field_validator("systematic_gene_name", mode="after")
    @classmethod
    def _validate_systematic(cls, v: str) -> str:
        """The allele sits in a real R64 feature (systematic-name pattern)."""
        if not re.match(SYSTEMATIC_GENE_PATTERN, v):
            raise ValueError(f"Invalid systematic gene name {v!r}")
        return v

    @model_validator(mode="after")
    def _check_allele(self) -> "BackgroundAllele":
        """Cassette iff cassette_replacement; provenance and zygosity set or gapped."""
        if (self.edit is AlleleEdit.cassette_replacement) != (
            self.cassette is not None
        ):
            raise ValueError(
                "cassette is required for a cassette_replacement and forbidden for "
                f"edit={self.edit.value}"
            )
        _require_value_or_gap(self, ("provenance", "zygosity"))
        return self

    @property
    def mechanism_so(self) -> tuple[str, str]:
        """(SO id, SO name) of this allele's edit."""
        return ALLELE_EDIT_SO[self.edit]

    @property
    def is_sourced(self) -> bool:
        """True when neither the allele nor its zygosity is a declared gap."""
        return not self.provenance_gaps


class StrainBackground(ProvenanceGapMixin):
    """The genome content a strain carries beyond R64, constant across a collection.

    ``name`` is the strain or lineage label the records join on (``BY4741``,
    ``BY4743``, or a descriptive name for an SGA progeny pool); it must equal
    ``StrainReferenceGenome.strain``. ``parents`` are the strains it was made from, verbatim
    (``["BY4741", "BY4742"]`` for BY4743; the SGA query and array for a progeny
    pool). ``construction`` is a one-line method statement (how the background was
    made, e.g. an SGA cross and its selections). ``reference_strain`` is the sequenced
    strain every allele is an edit against; it is pinned to S288C because the edit
    list of a non-S288C background against R64 would be thousands of variants, which
    this object is not for.

    ``mating_type`` and ``provenance`` are set or gapped (a typed absence, never a
    silent None). A haploid background is MATa or MATalpha; every allele's zygosity
    must fit the ploidy (``haploid`` in a haploid, ``homozygous``/``heterozygous`` in a
    diploid). A gene carries one allele entry, or two heterozygous entries for a
    compound heterozygote.
    """

    name: str
    reference_strain: Literal["S288C"] = "S288C"
    parents: list[str] | None = Field(
        default=None, description="strains this background was made from, verbatim"
    )
    construction: str | None = Field(
        default=None, description="how the background was made, one line"
    )
    mating_type: MatingType | None = Field(
        description="MAT locus state; None only with a gap on 'mating_type'"
    )
    ploidy: Literal["haploid", "diploid"]
    alleles: list[BackgroundAllele] = Field(
        default_factory=list,
        description="every non-R64 allele of the background, each sourced or gapped",
    )
    provenance: list[SourcedValue] | None = Field(
        default=None,
        description="quotes stating the strain name / mating type / ploidy; None only "
        "with a gap on 'provenance'",
    )

    @model_validator(mode="after")
    def _check_background(self) -> "StrainBackground":
        """Sourced-or-gapped, MAT vs ploidy, zygosity vs ploidy, one entry per locus."""
        if not self.name.strip():
            raise ValueError("StrainBackground.name cannot be empty")
        _require_value_or_gap(self, ("mating_type", "provenance"))
        if self.ploidy == "haploid" and self.mating_type is MatingType.a_alpha:
            raise ValueError("a haploid background cannot be MATa/MATalpha")
        allowed = (
            {Zygosity.haploid}
            if self.ploidy == "haploid"
            else {Zygosity.homozygous, Zygosity.heterozygous}
        )
        by_gene: dict[str, list[BackgroundAllele]] = {}
        for allele in self.alleles:
            if allele.zygosity is not None and allele.zygosity not in allowed:
                raise ValueError(
                    f"{allele.allele_name}: zygosity {allele.zygosity.value} does not "
                    f"fit a {self.ploidy} background"
                )
            by_gene.setdefault(allele.systematic_gene_name, []).append(allele)
        for gene, entries in by_gene.items():
            if len(entries) == 1:
                continue
            if len(entries) > 2 or any(
                e.zygosity is not Zygosity.heterozygous for e in entries
            ):
                raise ValueError(
                    f"{gene}: more than one allele entry is allowed only as two "
                    "heterozygous alleles of a diploid (a compound heterozygote)"
                )
        return self

    def alleles_at(self, systematic_gene_name: str) -> list[BackgroundAllele]:
        """The background's allele entries at one locus (empty = the R64 allele)."""
        return [
            a for a in self.alleles if a.systematic_gene_name == systematic_gene_name
        ]

    def functional_copies(self, systematic_gene_name: str) -> int:
        """Functional copies of a gene in the UNPERTURBED background.

        Ploidy copies minus the copies carrying a non-functional background allele:
        BY4743 ``his3Δ1/his3Δ1`` -> 0, ``LYS2/lys2Δ0`` -> 1, an untouched gene -> 2.
        Raises when an allele at the locus has a gapped zygosity, since the count is
        then not determined by the record.
        """
        copies = 1 if self.ploidy == "haploid" else 2
        for allele in self.alleles_at(systematic_gene_name):
            if allele.functional:
                continue
            if allele.zygosity is None:
                raise ValueError(
                    f"{allele.allele_name}: zygosity is a ProvenanceGap, so the "
                    "functional copy number at this locus is undetermined"
                )
            copies -= 2 if allele.zygosity is Zygosity.homozygous else 1
        return copies

    @property
    def is_fully_sourced(self) -> bool:
        """True when the background and every allele carry no declared gap."""
        return not self.provenance_gaps and all(a.is_sourced for a in self.alleles)


class ReferenceGenome(ModelStrict):
    """Reference genome identified by species and strain.

    ``ploidy`` is the genome-wide baseline copy number of the unperturbed strain
    (``"haploid"`` = 1 autosomal copy, ``"diploid"`` = 2). It lives here -- not on a
    perturbation -- because it applies to the WHOLE genome (WT and every unperturbed
    gene alike); a per-locus ``EngineeredCopyNumberPerturbation.copy_number`` records
    the deviation from this baseline (e.g. a HIP heterozygous deletion drops one
    autosomal gene from 2 -> 1 copies in a diploid). Defaults to ``"haploid"`` so all
    existing haploid datasets stay valid.

    A typed strain background is carried by the subclass ``StrainReferenceGenome``
    (#507), not by a field here: this class is in the schema closure of every served
    dataset, and any field added to it would mark every built store stale.
    """

    species: str
    strain: str
    ploidy: Literal["haploid", "diploid"] = "haploid"


class StrainReferenceGenome(ReferenceGenome):
    """A ``ReferenceGenome`` that states its typed ``StrainBackground`` (#507).

    ``background`` holds the mating type and every allele the strain carries beyond
    R64, each sourced or a typed gap. Its ``name`` equals ``strain`` and its ``ploidy``
    equals ``ploidy``, so the free ``strain`` string and the typed object cannot
    disagree. Used by ``StrainEnvironmentResponseExperimentReference``; the record
    stores it once per reference (the LMDB interns the whole reference; the graph
    writes it on the ``genome`` and ``experiment reference`` nodes).
    """

    background: StrainBackground

    @model_validator(mode="after")
    def _check_background_agrees(self) -> "StrainReferenceGenome":
        """The background names the same strain and ploidy as the reference."""
        if self.background.name != self.strain:
            raise ValueError(
                f"background name {self.background.name!r} != strain {self.strain!r}"
            )
        if self.background.ploidy != self.ploidy:
            raise ValueError(
                f"background ploidy {self.background.ploidy!r} != ploidy "
                f"{self.ploidy!r}"
            )
        return self


# --------------------------------------------------------------------------- #
# Sequence Ontology (SO) mechanism annotation.
#
# Every perturbation names the SO term for the MECHANISM by which the strain's
# genome differs from the S288C reference (deletion, insertion, SNV, ...). The id
# is a plain ``SO:NNNNNNN`` string; ``SOTerm`` is the reusable id+name record.
# We define a minimal ``SOTerm`` locally rather than import the richer one from
# ``torchcell.sequence.plasmid`` so this schema stays free of the biopython dep.
# --------------------------------------------------------------------------- #
SO_ID_PATTERN = r"^SO:\d{7}$"


def _validate_so_id(value: str) -> str:
    """Return ``value`` if it is a well-formed SO id (``SO:NNNNNNN``), else raise."""
    if not re.match(SO_ID_PATTERN, value):
        raise ValueError(f"Invalid SO id {value!r}; expected 'SO:NNNNNNN'")
    return value


class SOTerm(ModelStrict):
    """A Sequence Ontology term: a ``SO:NNNNNNN`` id paired with its name."""

    so_id: str
    name: str

    @field_validator("so_id", mode="after")
    @classmethod
    def validate_so_id(cls, v: str) -> str:
        """Enforce the ``SO:NNNNNNN`` id shape."""
        return _validate_so_id(v)


class GenePerturbation(ModelStrict):
    """Base perturbation of a single gene by systematic and common name.

    ``provenance`` records whether the difference from the S288C reference was
    ENGINEERED in the lab or arose NATURALLY in an isolate. It defaults to
    ``"engineered"`` so the many engineered perturbation types need not repeat it;
    natural types set ``provenance="natural"`` as a class default.
    """

    systematic_gene_name: str
    perturbed_gene_name: str
    provenance: str = "engineered"

    @field_validator("systematic_gene_name", mode="after")
    @classmethod
    def validate_sys_gene_name(cls, v: str) -> str:
        """Validate the systematic gene name matches an allowed feature pattern."""
        # Define named patterns for clarity based on genome feature types
        # protein-coding genes, pseudogenes, transposable_element_genes
        coding_gene_pattern = r"Y[A-P][LR]\d{3}[WC](-[A-Z])?"
        # mitochondrial genes
        mitochondrial_gene_pattern = r"Q\d{4}"
        # ncRNA_gene, rRNA_gene, snRNA_gene, snoRNA_gene, tRNA_gene, telomerase_RNA_gene
        noncoding_gene_pattern = r"YNC[A-Q]\d{4}[WC]"
        # Combine patterns
        full_pattern = f"^({coding_gene_pattern}|{mitochondrial_gene_pattern}|{noncoding_gene_pattern})$"

        if not re.match(full_pattern, v):
            raise ValueError("Invalid systematic gene name format")
        return v

    @field_validator("perturbed_gene_name", mode="after")
    @classmethod
    def validate_pert_gene_name(cls, v: str) -> str:
        """Normalize a trailing prime in the perturbed gene name to ``_prime``."""
        if v.endswith("'"):
            v = v[:-1] + "_prime"
        return v

    @field_validator("provenance", mode="after")
    @classmethod
    def validate_provenance(cls, v: str) -> str:
        """Provenance is one of the two allowed origins."""
        if v not in {"engineered", "natural"}:
            raise ValueError(f"provenance must be 'engineered' or 'natural', got {v!r}")
        return v


# --------------------------------------------------------------------------- #
# The three orthogonal AXES of a genotype difference from S288C. Each is an
# abstract base (never instantiated directly); concrete leaves below set the
# class-default ``state`` / ``mechanism_*`` / ``provenance`` for their kind.
# --------------------------------------------------------------------------- #
class PresenceAbsencePerturbation(GenePerturbation, ModelStrict):
    """AXIS 1 -- a gene is PRESENT or ABSENT relative to the reference.

    ``state`` is the neutral biological fact ("absent" spans an engineered KO and a
    natural core-loss alike); ``mechanism_so_id``/``mechanism_so_name`` name the SO
    mechanism. This ABC sets NO default for either -- every concrete child declares
    its own (deletion=absent/SO:0000159, insertion=present/SO:0000667).
    """

    state: str
    mechanism_so_id: str
    mechanism_so_name: str

    @field_validator("state", mode="after")
    @classmethod
    def validate_state(cls, v: str) -> str:
        """State is either present or absent."""
        if v not in {"present", "absent"}:
            raise ValueError(f"state must be 'present' or 'absent', got {v!r}")
        return v

    @field_validator("mechanism_so_id", mode="after")
    @classmethod
    def validate_mechanism_so_id(cls, v: str) -> str:
        """Mechanism SO id is well-formed."""
        return _validate_so_id(v)


class SequencePerturbation(GenePerturbation, ModelStrict):
    """AXIS 3 -- a sequence-level allelic change (SNP/indel/substitution allele).

    Defaults to the generic ``sequence_variant`` (SO:0001060); concrete children may
    override to a more specific term (e.g. SNV SO:0001483).
    """

    mechanism_so_id: str = "SO:0001060"
    mechanism_so_name: str = "sequence_variant"

    @field_validator("mechanism_so_id", mode="after")
    @classmethod
    def validate_mechanism_so_id(cls, v: str) -> str:
        """Mechanism SO id is well-formed."""
        return _validate_so_id(v)


class ExpressionRangeMultiplier(ModelStrict):
    """Min/max multiplier bounds on a gene's expression level."""

    min: float = Field(
        ..., description="Minimum range multiplier of gene expression levels"
    )
    max: float = Field(
        ..., description="Maximum range multiplier of gene expression levels"
    )


class CrisprConstruct(ModelStrict):
    """The engineered CRISPR machinery introduced to perturb a gene -- a first-class
    MATERIAL ENTITY (the guide RNA + Cas effector we actually put in the cell).

    Composed (field ``crispr``) onto every CRISPR perturbation leaf so the guide payload is
    defined ONCE and shared across two axes: an ACTIVE-Cas cut that deletes a gene
    (``CrisprDeletionPerturbation`` on the presence/absence axis) and a DEAD-Cas
    guide-directed effector that modulates expression (``CrisprActivationPerturbation`` /
    ``CrisprInterferencePerturbation`` on the expression axis). The CRISPR *tool* is thus
    orthogonal to the *outcome axis* -- same guide-directed machinery, different consequence
    set by the effector.

    ``effector`` is the Cas fusion (sourced, never guessed; e.g. ``SaCas9``,
    ``dSpCas9-RD1152``, ``dLbCas12a-VP``, ``dCas9-Mxi1``). ``guide_sequence`` is the short
    spacer INLINED as the perturbation identity; it is nullable so a screen that identifies
    only target genes (Mormino: spacers live upstream in the source library) can
    scaffold-and-defer. ``effector_plasmid_uri``/``_sha256`` are off-graph pointers left
    ``None`` today ("field now, plasmid later") so upgrading to full-plasmid / SBOL capture
    is a NON-breaking extension -- the record shape does not change when fidelity is upgraded.
    Design: ``[[plan.torchcell-crispr-expression-perturbation.2026.07.12]]``.
    """

    effector: str = Field(
        description="Cas effector fusion, e.g. 'SaCas9' | 'dSpCas9-RD1152' | 'dLbCas12a-VP' "
        "| 'dCas9-Mxi1' (sourced from the paper, never guessed)"
    )
    guide_sequence: str | None = Field(
        default=None,
        description="guide RNA spacer (~20 nt), inlined as identity; None if the screen "
        "released only target genes (defer to the upstream library)",
    )
    n_guides: int | None = Field(
        default=None,
        description="number of guides targeting this gene (e.g. Mormino '1-16 gRNAs/gene')",
    )
    library_pool: str | None = Field(
        default=None,
        description="the guide library sub-pool this construct was screened in, when a "
        "study runs several pools (e.g. Smith 2016 'gene_tiling_20bp' vs 'broad_tiling'). "
        "Guide-library provenance AND a strain discriminator: an identical spacer screened "
        "in two pools is two independent pooled measurements (pool-relative median-centred "
        "fitness), not one, so the pool joins the strain identity. None when a study has a "
        "single library (Lian, Mormino) -- a NON-breaking default.",
    )
    effector_plasmid_uri: str | None = Field(
        default=None,
        description="off-graph pointer into a plasmid/SBOL store for the full effector+guide "
        "cassette (future full-plasmid capture; None today)",
    )
    effector_plasmid_sha256: str | None = Field(
        default=None,
        description="sha256 of the source file the effector plasmid sequence comes from",
    )

    @model_validator(mode="after")
    def validate_plasmid_pointer(self) -> "CrisprConstruct":
        """A plasmid URI must carry its sha256 (mirror the ORF sequence-pointer invariant)."""
        if (
            self.effector_plasmid_uri is not None
            and self.effector_plasmid_sha256 is None
        ):
            raise ValueError(
                "effector_plasmid_uri requires effector_plasmid_sha256 (a pointer must be "
                "content-addressed)"
            )
        return self


class StrainConstruction(ModelStrict):
    """Where one physical deletion strain came from in its collection.

    Two strains that delete the same ORF can differ in background mutations, tags and
    behavior (Hillenmeyer: "some gene deletions were constructed more than once, in
    different batches"; Hoepfner Table S5 traces secondary mutations by construction
    Lab and Batch), so the construction record is a strain discriminator: two records
    of one ORF whose construction differs are two strains, not two replicates. All
    fields are verbatim source tokens; ``None`` = the source does not give it.
    """

    strain_accession: str | None = Field(
        default=None,
        description="collection accession of the strain (e.g. a Euroscarf 'Y0xxxx' id)",
    )
    lab: str | None = Field(
        default=None, description="constructing lab, verbatim (Hoepfner Table S5 'Lab')"
    )
    batch: str | None = Field(
        default=None,
        description="construction batch, verbatim (Hillenmeyer 'chr4_3', Table S5 'Batch')",
    )
    plate: str | None = Field(default=None, description="collection plate, verbatim")
    well: str | None = Field(
        default=None, description="collection well / row_column, verbatim"
    )

    @model_validator(mode="after")
    def _check_any(self) -> "StrainConstruction":
        """An empty construction record says nothing; leave the field None instead."""
        if all(
            v is None
            for v in (
                self.strain_accession,
                self.lab,
                self.batch,
                self.plate,
                self.well,
            )
        ):
            raise ValueError("StrainConstruction needs at least one field")
        return self


class OrfHistoryRelation(StrEnum):
    """How the ORF a strain was built against relates to the current R64 gene.

    - ``merged``: the source ORF was merged into the current gene (the strain deleted
      only the old ORF's interval, part of the current gene).
    - ``reannotated``: the same locus with shifted boundaries.
    - ``alias``: a pure renaming, same interval.
    """

    merged = "merged"
    reannotated = "reannotated"
    alias = "alias"


class ConstructedOrf(HashableProvenanceGapMixin):
    """The ORF annotation a deletion strain was BUILT against, when not the current gene.

    The YKO strains deleted ORFs as annotated around 2000. When that ORF has since
    been merged or reannotated, the perturbation's ``systematic_gene_name`` stays the
    CURRENT gene (so gene-keyed joins work) and this record states what was physically
    deleted: the source ORF name verbatim, its relation to the current gene, and the
    deleted interval. Two strains whose ``constructed_orf`` differ are different
    perturbations and must never be averaged into one record (#505 G3, #506 finding
    4). ``relation`` and ``deleted_span`` are set or gapped.
    """

    source_systematic_name: str = Field(
        description="ORF name the strain was built against, verbatim (e.g. 'YAR044W')"
    )
    relation: OrfHistoryRelation | None = Field(
        description="relation to the current gene; None only with a gap"
    )
    deleted_span: GenomicSpan | None = Field(
        description="R64 interval the cassette replaced; None only with a gap"
    )

    @model_validator(mode="after")
    def _check_constructed(self) -> "ConstructedOrf":
        """Relation and deleted span are set or carry a typed gap."""
        _require_value_or_gap(self, ("relation", "deleted_span"))
        return self


# --------------------------------------------------------------------------- #
# AXIS 1 -- presence/absence leaves.
# --------------------------------------------------------------------------- #
class DeletionPerturbation(PresenceAbsencePerturbation, ModelStrict):
    """Gene deletion via KanMX or NatMX gene replacement (engineered absence)."""

    description: str = "Deletion via KanMX or NatMX gene replacement"
    perturbation_type: Literal["deletion"] = "deletion"
    state: str = "absent"
    mechanism_so_id: str = "SO:0000159"
    mechanism_so_name: str = "deletion"
    provenance: str = "engineered"


class KanMxDeletionPerturbation(DeletionPerturbation, ModelStrict):
    """Gene deletion via KanMX gene replacement."""

    perturbation_type: Literal["kanmx_deletion"] = "kanmx_deletion"  # type: ignore[assignment]
    deletion_description: str = "Deletion via KanMX gene replacement."
    deletion_type: str = "KanMX"


class BarcodedKanMxDeletionPerturbation(
    HashableProvenanceGapMixin, KanMxDeletionPerturbation, ModelStrict
):
    """A KanMX deletion strain that carries the molecular barcode it was screened by.

    A pooled competitive-growth screen (Bar-seq, HIP/HOP) does not read colonies, it reads
    the 20-mer UPTAG/DNTAG the deletion cassette carries, so the barcode IS the strain's
    read-out identity in that assay and belongs on the perturbation rather than in a note.
    ``collection`` records WHICH physical deletion set the strain came from (Euroscarf MATa,
    the Yeast Knockout Collection HOM/HET pool, ...), because two collections can hold the
    same ORF deletion with different background mutations and different barcodes.

    This is the leaf for every whole-locus kanMX deletion a chemogenomic loader serves:
    a haploid deletion in a haploid background (Wildenhain, Vanacloig) and a homozygous
    deletion in a diploid background (HOP: Hillenmeyer hom, Hoepfner HOP); zygosity is
    read from the reference's ploidy. ``cassette`` names the exact cassette
    (``kanMX4`` for the YKO, sourced to Giaever 2014); ``barcode`` is the UPTAG (or the
    only tag a source names) and ``downtag_barcode`` the DNTAG; ``construction`` records
    the strain's accession / lab / batch / plate; ``constructed_orf`` states the
    physically deleted ORF when it is not the current gene.

    Every new field is nullable and gappable: ``None`` with a ``ProvenanceGap`` (e.g. a
    barcode table not yet mirrored) is a typed absence rather than a guess. A separate
    leaf (not fields added to ``KanMxDeletionPerturbation``) is deliberate --
    ``KanMxDeletionPerturbation`` is inside 33 served dataset closures and any field added
    to it would force a full rebuild of every one of them.
    """

    perturbation_type: Literal["barcoded_kanmx_deletion"] = "barcoded_kanmx_deletion"  # type: ignore[assignment]
    barcode: str | None = Field(
        default=None,
        description="the molecular barcode (UPTAG, or the only tag the source names) the "
        "strain is counted by in a pooled assay; None when the source released no barcode",
    )
    downtag_barcode: str | None = Field(
        default=None,
        description="the DNTAG 20-mer when the source releases both tags; None otherwise",
    )
    collection: str | None = Field(
        default=None,
        description="the physical deletion collection the strain came from, verbatim from "
        "the source (e.g. 'Euroscarf MATa deletion set'); None when unsourced",
    )
    cassette: str | None = Field(
        default=None,
        description="the exact replacement cassette, e.g. 'kanMX4' (pFA6-kanMX4, Giaever "
        "2014); None when unsourced",
    )
    construction: StrainConstruction | None = Field(
        default=None, description="accession / lab / batch / plate of this strain"
    )
    constructed_orf: ConstructedOrf | None = Field(
        default=None,
        description="the ORF the strain was built against when it is not the current "
        "gene (merged / reannotated); None when they coincide",
    )


class HeterozygousDeletionPerturbation(
    HashableProvenanceGapMixin, PresenceAbsencePerturbation, ModelStrict
):
    """One allele of a diploid replaced by a deletion cassette (HIP / het collection).

    Replaces the ``EngineeredCopyNumberPerturbation(copy_number=1,
    reference_copy_number=2)`` encoding for HIP-style data (#506 point 3). A
    heterozygous deletion is an ALLELE edit, not a dosage statement: the copy-number
    form asserted "1 of 2 working copies" even at loci where the background was
    already null (BY4743 ``his3Δ1/his3Δ1``: 0 working copies before and after). The
    functional dose is now DERIVED from this leaf plus the reference's
    ``StrainBackground`` (``heterozygous_deletion_functional_copies``).

    The gene stays PRESENT on the other allele, so ``state="present"``; the SO mechanism
    is the ``deletion`` of one allele. It is deliberately NOT a ``DeletionPerturbation``
    subclass, so an "every knockout" filter (``issubclass(_, DeletionPerturbation)``)
    does not count a heterozygote as an absent gene. ``replaced_allele`` names which
    allele the cassette replaced when the background is itself heterozygous at the
    locus (BY4743 ``LYS2/lys2Δ0``, ``MET15/met15Δ0``); set or gap it there.
    """

    description: str = (
        "Heterozygous deletion: one allele of a diploid replaced by a cassette"
    )
    perturbation_type: Literal["heterozygous_deletion"] = "heterozygous_deletion"
    state: str = "present"
    mechanism_so_id: str = "SO:0000159"
    mechanism_so_name: str = "deletion"
    provenance: str = "engineered"
    cassette: str | None = Field(
        default=None,
        description="the replacement cassette, e.g. 'kanMX4'; None only when unsourced",
    )
    barcode: str | None = Field(
        default=None, description="UPTAG (or the only tag named); None if unreleased"
    )
    downtag_barcode: str | None = Field(
        default=None, description="DNTAG 20-mer; None if unreleased"
    )
    collection: str | None = Field(
        default=None,
        description="the collection, verbatim (e.g. 'YSC1055 OpenBiosystems')",
    )
    construction: StrainConstruction | None = Field(
        default=None, description="accession / lab / batch / plate of this strain"
    )
    constructed_orf: ConstructedOrf | None = Field(
        default=None,
        description="the ORF the strain was built against when it is not the current gene",
    )
    replaced_allele: str | None = Field(
        default=None,
        description="allele the cassette replaced, verbatim (e.g. 'LYS2' or 'lys2Δ0'), "
        "when the background is heterozygous at this locus; None otherwise",
    )


class NatMxDeletionPerturbation(DeletionPerturbation, ModelStrict):
    """Gene deletion via NatMX gene replacement."""

    perturbation_type: Literal["natmx_deletion"] = "natmx_deletion"  # type: ignore[assignment]
    deletion_description: str = "Deletion via NatMX gene replacement."
    deletion_type: str = "NatMX"


class SgaKanMxDeletionPerturbation(KanMxDeletionPerturbation, ModelStrict):
    """KanMX deletion perturbation specific to SGA experiments."""

    perturbation_type: Literal["sga_kanmx_deletion"] = "sga_kanmx_deletion"  # type: ignore[assignment]
    kan_mx_description: str = (
        "KanMX Deletion Perturbation information specific to SGA experiments."
    )
    strain_id: str = Field(description="'Strain ID' in raw data.")
    kanmx_deletion_type: str = "SGA"


class SgaNatMxDeletionPerturbation(NatMxDeletionPerturbation, ModelStrict):
    """NatMX deletion perturbation specific to SGA experiments."""

    perturbation_type: Literal["sga_natmx_deletion"] = "sga_natmx_deletion"  # type: ignore[assignment]
    nat_mx_description: str = (
        "NatMX Deletion Perturbation information specific to SGA experiments."
    )
    strain_id: str = Field(description="'Strain ID' in raw data.")
    natmx_deletion_type: str = "SGA"

    # @classmethod
    # def _process_perturbation_data(cls, perturbation_data):
    #     if isinstance(perturbation_data, list):
    #         return [cls._create_perturbation_from_dict(p) for p in perturbation_data]
    #     elif isinstance(perturbation_data, dict):
    #         return cls._create_perturbation_from_dict(perturbation_data)
    #     return perturbation_data


class DampPerturbation(SequencePerturbation, ModelStrict):
    """Decreased-abundance-by-mRNA-perturbation (DAmP) allele perturbation."""

    description: str = "4-10 decreased expression via KANmx insertion at the "
    "the 3' UTR of the target gene."
    expression_range: ExpressionRangeMultiplier = Field(
        default=ExpressionRangeMultiplier(min=1 / 10.0, max=1 / 4.0),
        description="Gene expression is decreased by 4-10 fold",
    )
    perturbation_type: Literal["damp"] = "damp"


class SgaDampPerturbation(DampPerturbation, ModelStrict):
    """DAmP perturbation specific to SGA experiments."""

    damp_description: str = "Damp Perturbation information specific to SGA experiments."
    strain_id: str = Field(description="'Strain ID' in raw data.")
    damp_perturbation_type: str = "SGA"


class TsAllelePerturbation(SequencePerturbation, ModelStrict):
    """Temperature-sensitive allele perturbation via amino acid substitution."""

    description: str = (
        "Temperature sensitive allele compromised by amino acid substitution."
    )
    # seq: str = "NOT IMPLEMENTED"
    perturbation_type: Literal["temperature_sensitive_allele"] = (
        "temperature_sensitive_allele"
    )


class AllelePerturbation(SequencePerturbation, ModelStrict):
    """Generic allele perturbation via amino acid substitution."""

    description: str = (
        "Allele compromised by amino acid substitution without more generic "
        "phenotypic information specified."
    )
    # seq: str = "NOT IMPLEMENTED"
    perturbation_type: Literal["allele"] = "allele"


class SuppressorAllelePerturbation(SequencePerturbation, ModelStrict):
    """Suppressor allele that raises fitness in the presence of a perturbation."""

    description: str = (
        "suppressor allele that results in higher fitness in the presence"
        "of a perturbation, compared to the fitness of the perturbation alone."
    )
    perturbation_type: Literal["suppressor_allele"] = "suppressor_allele"


class SgaSuppressorAllelePerturbation(SuppressorAllelePerturbation, ModelStrict):
    """Suppressor allele perturbation specific to SGA experiments."""

    suppressor_description: str = (
        "Suppressor Allele Perturbation information specific to SGA experiments."
    )
    strain_id: str = Field(description="'Strain ID' in raw data.")
    suppressor_allele_perturbation_type: str = "SGA"


class SgaTsAllelePerturbation(TsAllelePerturbation, ModelStrict):
    """Temperature-sensitive allele perturbation specific to SGA experiments."""

    ts_allele_description: str = (
        "Ts Allele Perturbation information specific to SGA experiments."
    )
    strain_id: str = Field(description="'Strain ID' in raw data.")
    temperature_sensitive_allele_perturbation_type: str = "SGA"


class SgaAllelePerturbation(AllelePerturbation, ModelStrict):
    """Generic allele perturbation specific to SGA experiments."""

    allele_description: str = (
        "Ts Allele Perturbation information specific to SGA experiments."
    )
    strain_id: str = Field(description="'Strain ID' in raw data.")
    allele_perturbation_type: str = "SGA"


class ConditionalAlleleClass(StrEnum):
    """How a conditional allele of an essential gene reduces its function.

    - ``temperature_sensitive``: an amino-acid-substitution allele, null at the
      restrictive temperature (``cdc28-4``).
    - ``damp``: Decreased Abundance by mRNA Perturbation, a marker inserted in the 3'
      UTR.
    - ``promoter_replacement``: the native promoter replaced by a regulatable one
      (e.g. a tetO promoter shut off by doxycycline).

    An allele of UNKNOWN class is not a member: it is ``allele_class=None`` with a
    ``ProvenanceGap`` (one encoding of "unknown", never a fourth enum value).
    """

    temperature_sensitive = "temperature_sensitive"
    damp = "damp"
    promoter_replacement = "promoter_replacement"


class ConditionalAllelePerturbation(
    HashableProvenanceGapMixin, SequencePerturbation, ModelStrict
):
    """A conditional (hypomorphic) allele of an essential gene, class possibly unknown.

    A haploid null of an essential gene is not viable, so a screen that reports an
    essential gene as a "deletion strain" (Wildenhain: 33 such strains, #504) screened
    a conditional allele whose identity the release does not carry. This leaf records
    the gene with a typed allele class, the allele designation, the marker and the
    collection, each set or gapped: until the strain table is mirrored,
    ``allele_class=None`` with a ``deferred_pending_source_review`` gap naming the table
    that resolves it. ``allele_class`` is REQUIRED to be set or gapped. The SO mechanism
    is the generic ``sequence_variant`` (the specific edit is not yet known).
    """

    description: str = (
        "Conditional allele of an essential gene (ts, DAmP, promoter replacement)"
    )
    perturbation_type: Literal["conditional_allele"] = "conditional_allele"
    provenance: str = "engineered"
    allele_class: ConditionalAlleleClass | None = Field(
        description="how the allele reduces function; None only with a gap"
    )
    allele_name: str | None = Field(
        default=None, description="allele designation verbatim, e.g. 'cdc28-4'"
    )
    marker: str | None = Field(
        default=None, description="selection marker carried with the allele, verbatim"
    )
    collection: str | None = Field(
        default=None, description="the collection the strain came from, verbatim"
    )
    construction: StrainConstruction | None = Field(
        default=None, description="accession / lab / batch / plate of this strain"
    )

    @model_validator(mode="after")
    def _check_allele_class(self) -> "ConditionalAllelePerturbation":
        """An unknown allele class is a typed gap, never a silent None."""
        _require_value_or_gap(self, ("allele_class",))
        return self


# Change to AggregateDeletionPerturbation, or AggDeletionPerturbation
class MeanDeletionPerturbation(DeletionPerturbation, ModelStrict):
    """Deletion perturbation aggregating duplicate experiments by their mean."""

    description: str = "Mean deletion perturbation representing duplicate experiments"
    perturbation_type: Literal["mean_deletion"] = "mean_deletion"  # type: ignore[assignment]
    deletion_type: str = "mean"
    num_duplicates: int = Field(
        description="Number of duplicate experiments used to compute the mean and std."
    )


class MarkerDeletionPerturbation(DeletionPerturbation, ModelStrict):
    """Gene deletion via a selectable/auxotrophic marker other than KanMX or NatMX.

    Some drug-sensitized backgrounds delete a gene with a heterologous auxotrophic
    cassette -- e.g. the Vanacloig 3DeltaAlpha background carries ``pdr3::KlURA3`` and
    ``snq2::KlLEU2`` (Kluyveromyces lactis URA3 / LEU2 markers). Same AXIS-1
    ``state="absent"`` and SO ``deletion`` mechanism as the marker-specific leaves;
    ``marker`` names the exact cassette so the deletion is not mislabelled KanMX/NatMX.
    """

    perturbation_type: Literal["marker_deletion"] = "marker_deletion"  # type: ignore[assignment]
    marker: str = Field(
        description="selectable marker, e.g. 'KlURA3' | 'KlLEU2' | 'HIS3'"
    )
    deletion_type: str = "marker"
    strain_id: str | None = Field(
        default=None,
        description="source strain label of an RNA-barcoded / per-strain-tracked deletion, "
        "when the study tracks individual strains (e.g. Nadal-Ribelles 2025 genotype "
        "barcode 'bc_YAL012W'; replacement strains 'bc_YBR020W-1'/'-2' share a deleted ORF "
        "but are distinct strains). A strain discriminator: two records that delete the same "
        "ORF in the same environment are distinct measurements when their strain_id differs. "
        "None for backgrounds with a single strain per deletion (Vanacloig, Ohnuki) -- a "
        "NON-breaking default.",
    )


class CrisprDeletionPerturbation(DeletionPerturbation, ModelStrict):
    """Gene deletion via an ACTIVE Cas nuclease cut repaired from a homology donor.

    Model-by-state: the OUTCOME is an absent gene, so this is a THIRD deletion MECHANISM
    beside ``KanMxDeletion``/``NatMxDeletion`` (which differ from each other on exactly this
    mechanism axis), NOT a new kind of perturbation. It inherits the AXIS-1 ``state="absent"``
    and SO ``deletion`` mechanism, so ``issubclass(_, DeletionPerturbation)`` still catches
    every knockout; it additionally CARRIES the guide (the known material we introduced) via
    the shared ``crispr`` construct -- unlike a plain deletion, which would discard it.

    Motivating case -- Lian 2019 MAGIC CRISPRd (``SaCas9``): the released ``Sequence`` column
    is the guide spacer + HR donor concatenated; a loader splits it into ``crispr.guide_sequence``
    and ``donor_sequence``. NOTE on epistemics: a pooled-library screen DESIGNS the deletion
    but does not per-strain verify it -- that designed<->realized uncertainty is orthogonal to
    this leaf and handled downstream (conversion / a future certainty axis), not asserted here.
    """

    description: str = (
        "Gene deletion via an active Cas nuclease cut + homology-donor repair"
    )
    perturbation_type: Literal["crispr_deletion"] = "crispr_deletion"  # type: ignore[assignment]
    deletion_type: str = "crispr"
    crispr: CrisprConstruct = Field(
        description="the guide + active-Cas effector introduced to cut the gene"
    )
    donor_sequence: str | None = Field(
        default=None,
        description="homology-donor sequence used for scarless repair (None if not released)",
    )


class GeneAdditionPerturbation(PresenceAbsencePerturbation, ModelStrict):
    """Gain-of-function: a gene ADDED to the strain (heterologous expression or an
    extra native copy), carried on a plasmid or integrated at a chromosomal locus.

    Unlike the loss-of-function perturbations, an added gene may be HETEROLOGOUS
    (crtYB/crtI from *Xanthophyllomyces dendrorhous*; CYP76AD1/DOD from plants) and so
    has NO S. cerevisiae systematic name -- ``systematic_gene_name`` then carries the
    heterologous gene symbol and the native-name validator is relaxed for this class.
    For an extra NATIVE copy (Ozaydin BTS1) ``systematic_gene_name`` is the real
    systematic name and ``is_heterologous`` is False. ``localization`` distinguishes
    the plasmid vs chromosome context; ``plasmid_contig_id``/``locus_tag`` point at the
    raw sequence in the (future) plasmid-sequence store -- ``None`` until it lands, and
    embeddings are constructed downstream from the raw sequence, never baked here.
    Design: ``[[torchcell.datamodels.gene-addition-perturbation-design]]``.
    """

    description: str = "Gene addition (heterologous expression or extra native copy)"
    perturbation_type: Literal["gene_addition"] = "gene_addition"
    state: str = "present"
    mechanism_so_id: str = "SO:0000667"
    mechanism_so_name: str = "insertion"
    provenance: str = "engineered"
    source_organism: str = Field(
        description="organism the added gene is from, e.g. 'Xanthophyllomyces dendrorhous'"
    )
    is_heterologous: bool = Field(
        description="True if the gene is non-native to S. cerevisiae"
    )
    localization: str = Field(
        description="engineered location, e.g. 'episomal_2micron' | 'chromosomal_integration'"
    )
    construct_name: str | None = Field(
        default=None,
        description="plasmid/cassette name, e.g. 'YB/I/BTS1', 'Btx-cassette'",
    )
    integration_locus: str | None = Field(
        default=None,
        description="chromosomal integration site (integration only), e.g. 'XII-5'",
    )
    plasmid_contig_id: str | None = Field(
        default=None,
        description="pointer into the plasmid-sequence store; None until that store lands",
    )
    locus_tag: str | None = Field(
        default=None, description="feature id of the added gene on the plasmid contig"
    )
    variant: str | None = Field(
        default=None,
        description="expressed variant, e.g. 'K229L' for a feedback-resistant allele",
    )

    @field_validator("systematic_gene_name", mode="after")
    @classmethod
    def validate_sys_gene_name(cls, v: str) -> str:
        """Relax the native-name validator: an added gene may be heterologous (no yeast
        systematic name), so accept any non-empty identifier.
        """
        if not v:
            raise ValueError("systematic_gene_name must be non-empty")
        return v


class NaturalGeneAbsencePerturbation(PresenceAbsencePerturbation, ModelStrict):
    """A reference (core) gene ABSENT in a natural isolate (Caudal core-loss).

    The NATURAL counterpart of an engineered deletion: same AXIS-1 ``state="absent"``
    and SO ``deletion`` mechanism, but ``provenance="natural"``. Replaces the former
    mis-use of ``CopyNumberVariantPerturbation`` (copy_number 0) for absence -- CNV is
    now reserved for dosage of a PRESENT gene. The gene id may be a pangenome/accessory
    id, so the native-name validator is relaxed (as for ``GeneAddition``). Sequence, when
    known, is an off-graph pointer (``sequence_uri`` + ``sequence_sha256``), never inlined.
    """

    description: str = "Natural absence of a reference gene in an isolate vs S288C"
    perturbation_type: Literal["natural_gene_absence"] = "natural_gene_absence"
    state: str = "absent"
    mechanism_so_id: str = "SO:0000159"
    mechanism_so_name: str = "deletion"
    provenance: str = "natural"
    strain_id: str = Field(description="isolate id whose genome lacks this gene")
    pangenome_orf_id: str | None = Field(
        default=None, description="pangenome ORF id, when the absence is tracked there"
    )
    sequence_source: str | None = Field(
        default=None,
        description="off-graph store key / citation for the reference gene",
    )
    sequence_uri: str | None = Field(
        default=None, description="pointer into the gene-keyed sequence store"
    )
    sequence_sha256: str | None = Field(
        default=None, description="sha256 of the source file the sequence comes from"
    )

    @field_validator("systematic_gene_name", mode="after")
    @classmethod
    def validate_sys_gene_name(cls, v: str) -> str:
        """Relax: an accessory/pangenome ORF has no yeast systematic name."""
        if not v:
            raise ValueError("systematic_gene_name must be non-empty")
        return v


class NaturalGenePresencePerturbation(PresenceAbsencePerturbation, ModelStrict):
    """A non-reference (accessory) gene PRESENT in a natural isolate (Caudal accessory).

    The NATURAL counterpart of an engineered gene addition: AXIS-1 ``state="present"``
    with SO ``insertion`` mechanism and ``provenance="natural"``. Replaces the former
    mis-use of ``CopyNumberVariantPerturbation`` for accessory-presence -- CNV is now
    reserved for dosage of a PRESENT gene. ``copy_number`` records how many copies are
    present (default 1.0); ``origin`` annotates the accessory ORF's provenance with the
    field's vocabulary (``ancestral | introgression | hgt``). The native-name validator
    is relaxed; sequence is an off-graph pointer, never inlined.
    """

    description: str = "Natural presence of an accessory gene in an isolate vs S288C"
    perturbation_type: Literal["natural_gene_presence"] = "natural_gene_presence"
    state: str = "present"
    mechanism_so_id: str = "SO:0000667"
    mechanism_so_name: str = "insertion"
    provenance: str = "natural"
    strain_id: str = Field(description="isolate id whose genome carries this gene")
    copy_number: float = Field(
        default=1.0,
        description="copies of the accessory gene present (haploid basis; > 0)",
    )
    pangenome_orf_id: str | None = Field(
        default=None,
        description="pangenome ORF id for the accessory ORF, e.g. 'EC1118_1F14_0012g'",
    )

    @field_validator("copy_number", mode="after")
    @classmethod
    def validate_copy_number_positive(cls, v: float) -> float:
        """M2: this leaf means the gene IS present, so ``copy_number > 0`` (absence is
        ``NaturalGeneAbsencePerturbation``, never copy_number=0).
        """
        if v <= 0:
            raise ValueError(
                "copy_number must be > 0 for a PRESENT accessory gene "
                "(absence is NaturalGeneAbsencePerturbation)"
            )
        return v

    origin: str | None = Field(
        default=None,
        description="accessory-ORF provenance: 'ancestral' | 'introgression' | 'hgt'",
    )
    sequence_source: str | None = Field(
        default=None, description="off-graph store key / citation for the ORF sequence"
    )
    sequence_uri: str | None = Field(
        default=None, description="pointer into the pangenome ORF sequence store"
    )
    sequence_sha256: str | None = Field(
        default=None,
        description="sha256 of the source file the ORF sequence comes from",
    )

    @field_validator("systematic_gene_name", mode="after")
    @classmethod
    def validate_sys_gene_name(cls, v: str) -> str:
        """Relax: an accessory/pangenome ORF has no yeast systematic name."""
        if not v:
            raise ValueError("systematic_gene_name must be non-empty")
        return v


class SequenceVariantPerturbation(SequencePerturbation, ModelStrict):
    """SNP/indel-level allelic variation: a native gene whose sequence in this strain
    differs from the S288C reference, captured via an off-graph pointer.

    NATURAL-variation type (contrast the ENGINEERED perturbations, which carry a marker /
    construct / donor organism). Distinct from ``AllelePerturbation`` (a prose amino-acid-
    substitution description): this points at the strain's ACTUAL variant allele. The base
    systematic-name validator applies -- these are real S. cerevisiae reference genes
    (``YAL001C`` ...). The sequence is NEVER inlined: ``sequence_source`` + ``strain_id`` +
    ``sequence_uri`` identify the record in the off-graph gene-keyed store (dereferenced at
    load, ``sequence_sha256``-verified), mirroring the ``GeneAddition`` pointer pattern.

    Naming follows population-genomics usage (sequence variant = SNP + indel), NOT the
    phylogenetic "gene gain/loss" vocabulary (which asserts a lineage polarity a pairwise
    reference comparison cannot). Together with ``CopyNumberVariantPerturbation`` this lets
    a NATURAL ISOLATE be modeled as a perturbation set off S288C (~4,500 sequence variants
    per isolate). Design: ``[[torchcell.datasets.scerevisiae.caudal2024]]``.
    """

    description: str = (
        "Sequence variant (SNP/indel) of a native gene vs the S288C reference; "
        "sequence by off-graph pointer"
    )
    perturbation_type: Literal["sequence_variant"] = "sequence_variant"
    provenance: str = "natural"
    mechanism_so_id: str = "SO:0001483"
    mechanism_so_name: str = "SNV"
    strain_id: str = Field(
        description="isolate id whose allele this is, e.g. 'AAB' | 'SACE_YAU'"
    )
    sequence_source: str | None = Field(
        default=None,
        description=(
            "off-graph gene-keyed store key / citation, e.g. "
            "'peterGenomeEvolution10112018'; None until the store lands"
        ),
    )
    sequence_uri: str | None = Field(
        default=None,
        description=(
            "pointer into the gene-keyed sequence store (e.g. "
            "'<gene>.fasta#<strain_header>'); None until that store lands"
        ),
    )
    sequence_sha256: str | None = Field(
        default=None,
        description="sha256 of the source file the variant sequence is dereferenced from",
    )


class CopyNumberVariantPerturbation(GenePerturbation, ModelStrict):
    """Copy-number variation (CNV) of a PRESENT pangenome ORF relative to S288C.

    NATURAL-variation, DOSAGE axis: records the copy number of a gene that IS present
    (amplification / reduction), NOT its absence. ``copy_number`` is strictly ``> 0``
    (M2 canonical form -- absence has exactly ONE encoding, the presence/absence
    ``NaturalGeneAbsencePerturbation`` leaf; "you don't copy from zero"). Presence of an
    accessory ORF is likewise the presence/absence ``NaturalGenePresencePerturbation``
    leaf, not a CNV. This leaf is for a genuine dosage difference:

      - AMPLIFICATION -> ``copy_number`` > ``reference_copy_number``;
      - REDUCTION (still present) -> ``reference_copy_number`` > ``copy_number`` > 0.

    Contrast the ENGINEERED ``EngineeredCopyNumberPerturbation`` (same dosage axis, lab
    origin). ``origin`` annotates an accessory ORF's provenance with the field's
    mechanism vocabulary (``ancestral | introgression | hgt``). For a non-reference ORF
    the sequence is an off-graph pointer, never inlined.

    ``systematic_gene_name`` carries the S288C systematic name for a reference ORF, else the
    pangenome ORF id -- so the native-name validator is relaxed (like ``GeneAddition``).
    Design: ``[[torchcell.datasets.scerevisiae.caudal2024]]``.
    """

    description: str = (
        "Copy-number variation (incl. presence/absence) of a pangenome ORF vs S288C"
    )
    perturbation_type: Literal["copy_number_variant"] = "copy_number_variant"
    provenance: str = "natural"
    mechanism_so_id: str = "SO:0001019"
    mechanism_so_name: str = "copy_number_variation"
    copy_number: float = Field(
        description="ORF copy number in this isolate (haploid basis; non-integer allowed; strictly > 0)"
    )
    reference_copy_number: float = Field(
        default=1.0,
        description="copy number in S288C R64 (1 for a core ORF; 0 for a non-reference/accessory ORF)",
    )

    @field_validator("copy_number", mode="after")
    @classmethod
    def validate_copy_number_positive(cls, v: float) -> float:
        """M2: CNV encodes DOSAGE of a present gene, never absence -- ``copy_number > 0``.

        Absence has one canonical encoding (``NaturalGeneAbsencePerturbation``); a CNV at
        0 copies would be a second, forbidden encoding of the same state.
        """
        if v <= 0:
            raise ValueError(
                "copy_number must be > 0 (CNV is dosage of a PRESENT gene; absence is "
                "NaturalGeneAbsencePerturbation, not copy_number=0)"
            )
        return v

    strain_id: str = Field(description="isolate id, e.g. 'AAB'")
    pangenome_orf_id: str | None = Field(
        default=None,
        description="pangenome ORF id for a non-reference ORF, e.g. 'EC1118_1F14_0012g'",
    )
    origin: str | None = Field(
        default=None,
        description="accessory-ORF provenance: 'ancestral' | 'introgression' | 'hgt' (None if core/unknown)",
    )
    sequence_source: str | None = Field(
        default=None, description="off-graph store key / citation for the ORF sequence"
    )
    sequence_uri: str | None = Field(
        default=None, description="pointer into the pangenome ORF sequence store"
    )
    sequence_sha256: str | None = Field(
        default=None,
        description="sha256 of the source file the ORF sequence comes from",
    )

    @field_validator("systematic_gene_name", mode="after")
    @classmethod
    def validate_sys_gene_name(cls, v: str) -> str:
        """Relax: a non-reference accessory ORF has a pangenome id, not a yeast
        systematic name, so accept any non-empty identifier.
        """
        if not v:
            raise ValueError("systematic_gene_name must be non-empty")
        return v

    @field_validator("mechanism_so_id", mode="after")
    @classmethod
    def validate_mechanism_so_id(cls, v: str) -> str:
        """Mechanism SO id is well-formed."""
        return _validate_so_id(v)


class EngineeredCopyNumberPerturbation(GenePerturbation, ModelStrict):
    """An ENGINEERED copy-number/dosage change of a PRESENT native gene.

    The engineered counterpart of the natural ``CopyNumberVariantPerturbation``: same
    dosage axis and SO ``copy_number_variation`` mechanism, but ``provenance="engineered"``
    and referring to a REAL S288C reference gene (the base systematic-name validator
    applies -- NOT relaxed, these are not pangenome ORFs). It hangs directly off
    ``GenePerturbation`` in parallel with the natural CNV leaf (the dosage axis has no
    intermediate ABC; mirroring the existing pattern keeps the hierarchy symmetric and
    reparents nothing).

    Motivating case -- HIP/HOP chemogenomics (Hoepfner/FitDb/Lee) in a DIPLOID: a HIP
    HETEROZYGOUS deletion drops one autosomal gene from 2 -> 1 copies, i.e.
    ``EngineeredCopyNumberPerturbation(copy_number=1, reference_copy_number=2, marker="KanMX")``.
    (A HOP HOMOZYGOUS deletion is total absence and stays a ``DeletionPerturbation``.)
    ``reference_copy_number`` is the copies in the reference (= ``ReferenceGenome.ploidy``
    for an autosomal gene); ``marker`` is the optional selection cassette on the affected
    allele. The gene remains PRESENT, so ``state="present"``.
    """

    description: str = "Engineered copy-number/dosage change of a present native gene"
    perturbation_type: Literal["engineered_copy_number"] = "engineered_copy_number"
    provenance: str = "engineered"
    state: str = "present"
    mechanism_so_id: str = "SO:0001019"
    mechanism_so_name: str = "copy_number_variation"
    copy_number: float = Field(
        description="engineered target copies of the gene, e.g. 1 for a heterozygous deletion"
    )
    reference_copy_number: float = Field(
        description="copies in the reference (= ploidy for an autosomal gene, e.g. 2 in a diploid)"
    )
    marker: str | None = Field(
        default=None,
        description="selection marker on the affected allele, e.g. 'KanMX' (None if unmarked)",
    )

    @field_validator("mechanism_so_id", mode="after")
    @classmethod
    def validate_mechanism_so_id(cls, v: str) -> str:
        """Mechanism SO id is well-formed."""
        return _validate_so_id(v)

    @field_validator("copy_number", mode="after")
    @classmethod
    def validate_copy_number_positive(cls, v: float) -> float:
        """M2: dosage of a PRESENT gene -- ``copy_number > 0`` (0 copies is a total
        knockout, i.e. a ``DeletionPerturbation`` absence, not a CNV).
        """
        if v <= 0:
            raise ValueError(
                "copy_number must be > 0 (0 copies is total absence = "
                "DeletionPerturbation, not an engineered CNV)"
            )
        return v


# --------------------------------------------------------------------------- #
# AXIS 4 -- expression modulation. Engineered modulation of a gene's EXPRESSION
# level while it stays PRESENT, sequence-unedited, and copy-number-unchanged --
# effected in TRANS by a dead-Cas guide-directed effector (CRISPRi/CRISPRa). It
# fits none of axes 1-3 (not presence/absence, not DNA dosage, not a sequence
# allele), so it is its own axis. The leaf is indexed by the TARGET gene; the
# guide + effector material lives in the shared ``crispr`` construct.
# --------------------------------------------------------------------------- #
class ExpressionModulationPerturbation(GenePerturbation, ModelStrict):
    """AXIS 4 (ABC) -- engineered expression modulation of a present gene.

    The gene remains PRESENT (``state="present"``), its DNA copy number is unchanged, and its
    sequence is unedited; only expression OUTPUT changes, driven in trans by an inserted
    dead-Cas guide-directed effector. Mechanism is the guide itself (SO:0001998 ``sgRNA``).
    Concrete children set ``expression_direction`` (increased=activation, decreased=
    interference). This ABC is never instantiated directly. Design:
    ``[[plan.torchcell-crispr-expression-perturbation.2026.07.12]]``.
    """

    state: str = "present"
    provenance: str = "engineered"
    mechanism_so_id: str = "SO:0001998"
    mechanism_so_name: str = "sgRNA"
    expression_direction: str
    crispr: CrisprConstruct = Field(
        description="the guide + dead-Cas effector introduced to modulate expression"
    )

    @field_validator("mechanism_so_id", mode="after")
    @classmethod
    def validate_mechanism_so_id(cls, v: str) -> str:
        """Mechanism SO id is well-formed."""
        return _validate_so_id(v)

    @field_validator("expression_direction", mode="after")
    @classmethod
    def validate_expression_direction(cls, v: str) -> str:
        """Direction is increased (activation) or decreased (interference)."""
        if v not in {"increased", "decreased"}:
            raise ValueError(
                f"expression_direction must be 'increased' or 'decreased', got {v!r}"
            )
        return v


class CrisprActivationPerturbation(ExpressionModulationPerturbation, ModelStrict):
    """CRISPRa -- guide-directed dead-Cas activator INCREASES a present gene's expression.

    Lian 2019 MAGIC uses ``dLbCas12a-VP`` (dead Cas12a fused to an activation domain).
    """

    description: str = "CRISPR activation (increased expression of a present gene)"
    perturbation_type: Literal["crispr_activation"] = "crispr_activation"
    expression_direction: str = "increased"


class CrisprInterferencePerturbation(ExpressionModulationPerturbation, ModelStrict):
    """CRISPRi -- guide-directed dead-Cas repressor DECREASES a present gene's expression.

    Lian 2019 MAGIC uses ``dSpCas9-RD1152``; Mormino 2022 uses ``dCas9-Mxi1``.
    """

    description: str = "CRISPR interference (decreased expression of a present gene)"
    perturbation_type: Literal["crispr_interference"] = "crispr_interference"
    expression_direction: str = "decreased"


SgaPerturbationType = (
    SgaKanMxDeletionPerturbation
    | SgaNatMxDeletionPerturbation
    | SgaDampPerturbation
    | SgaTsAllelePerturbation
    | SgaSuppressorAllelePerturbation
    | SgaAllelePerturbation
)

GenePerturbationType = (
    SgaPerturbationType
    | MeanDeletionPerturbation
    | MarkerDeletionPerturbation
    | KanMxDeletionPerturbation
    | BarcodedKanMxDeletionPerturbation
    | HeterozygousDeletionPerturbation
    | ConditionalAllelePerturbation
    | NatMxDeletionPerturbation
    | CrisprDeletionPerturbation
    | GeneAdditionPerturbation
    | NaturalGeneAbsencePerturbation
    | NaturalGenePresencePerturbation
    | SequenceVariantPerturbation
    | CopyNumberVariantPerturbation
    | EngineeredCopyNumberPerturbation
    | CrisprActivationPerturbation
    | CrisprInterferencePerturbation
)


class Genotype(ModelStrict):
    """Collection of gene perturbations defining a strain's genotype."""

    perturbations: list[GenePerturbationType] = Field(description="Gene perturbation")

    @field_validator("perturbations", mode="after")
    @classmethod
    def sort_perturbations(
        cls, perturbations: list[GenePerturbationType]
    ) -> list[GenePerturbationType]:
        """Sort perturbations by gene name, type, and perturbed name for stable order."""
        return sorted(
            perturbations,
            key=lambda p: (
                p.systematic_gene_name,
                p.perturbation_type,
                p.perturbed_gene_name,
            ),
        )

    @property
    def systematic_gene_names(self) -> list[str]:
        """Return systematic gene names ordered by systematic gene name."""
        sorted_perturbations = sorted(
            self.perturbations, key=lambda p: p.systematic_gene_name
        )
        return [p.systematic_gene_name for p in sorted_perturbations]

    @property
    def perturbed_gene_names(self) -> list[str]:
        """Return perturbed gene names ordered by systematic gene name."""
        sorted_perturbations = sorted(
            self.perturbations, key=lambda p: p.systematic_gene_name
        )
        return [p.perturbed_gene_name for p in sorted_perturbations]

    @property
    def perturbation_types(self) -> list[str]:
        """Return perturbation types ordered by systematic gene name."""
        sorted_perturbations = sorted(
            self.perturbations, key=lambda p: p.systematic_gene_name
        )
        return [p.perturbation_type for p in sorted_perturbations]

    def __len__(self) -> int:
        """Return the number of perturbations."""
        return len(self.perturbations)

    # we would use set, but need serialization to be a list
    def __eq__(self, other: object) -> bool:
        """Return True if both genotypes contain the same set of perturbations."""
        if not isinstance(other, Genotype):
            return NotImplemented

        return set(self.perturbations) == set(other.perturbations)


def heterozygous_deletion_functional_copies(
    background: StrainBackground, perturbation: HeterozygousDeletionPerturbation
) -> int | None:
    """Functional copies left at the deleted locus, respecting the background.

    In a diploid background with no allele at the locus the answer is 1 (the usual
    HIP haploinsufficiency case). Where the background is already null on both copies
    (BY4743 ``his3Δ1/his3Δ1``) it is 0, before and after. Where the background is
    heterozygous (``LYS2/lys2Δ0``) it depends on which allele the cassette replaced:
    1 if ``replaced_allele`` names the null background allele, 0 if it names anything
    else (the functional copy). ``None`` means the record does not determine it: the
    background is heterozygous at the locus and ``replaced_allele`` is unset.
    """
    if background.ploidy != "diploid":
        raise ValueError("a heterozygous deletion needs a diploid background")
    gene = perturbation.systematic_gene_name
    before = background.functional_copies(gene)
    null_het = [
        a
        for a in background.alleles_at(gene)
        if not a.functional and a.zygosity is Zygosity.heterozygous
    ]
    if before == 0:
        return 0
    if not null_het:
        return before - 1
    if perturbation.replaced_allele is None:
        return None
    if perturbation.replaced_allele in {a.allele_name for a in null_het}:
        return before
    return before - 1


# Environment
# ``Media`` is defined below (after ``Compound`` / ``Concentration``, which its
# component-based form depends on) and before ``Environment`` (its only consumer).
# See the "Media as COMPONENTS" block.


class TemperatureUnit(StrEnum):
    """UO-aligned temperature units (typed, not a free string) -- G2 unit typing."""

    celsius = "Celsius"
    kelvin = "Kelvin"
    fahrenheit = "Fahrenheit"


class Temperature(BaseModel):
    """Temperature value with a typed unit (defaults to Celsius)."""

    value: float  # Renamed from scalar to value
    unit: TemperatureUnit = TemperatureUnit.celsius

    @model_validator(mode="after")
    def check_temperature(self) -> "Temperature":
        """Validate that a Celsius temperature is not below absolute zero."""
        if self.unit is TemperatureUnit.celsius and self.value < -273:
            raise ValueError("Temperature cannot be below -273 degrees Celsius")
        return self


# --------------------------------------------------------------------------- #
# Environmental-perturbation ontology (parallel to the gene-perturbation
# ontology). A GenePerturbation is an edit to the genome vs S288C; an
# EnvironmentPerturbation is an edit to the growth environment vs the base
# medium along TWO axes:
#   - SmallMoleculePerturbation -- an added chemical SPECIES (drug / acid /
#     alcohol / salt / oxidant), identified by a typed ``Compound`` and dosed at a
#     ``Concentration``.
#   - EnvironmentPhysicalPerturbation -- a neutral, scalar physical FACTOR (pH /
#     osmolarity / carbon source), NOT a compound and NOT a consequence.
# Temperature is NOT a perturbation leaf: it is carried on
# ``Environment.temperature`` (one canonical encoding, M2). A perturbation names
# the EDIT, never its phenotypic consequence (M1); "is it stress?" /
# sensitive-vs-tolerant is an ``EnvironmentResponsePhenotype`` property, and a
# compound's mode of action is a ChEBI ROLE on its ``Compound``.
# ``perturbation_type`` is the discriminator; concrete leaves set it.
# Design: ``[[torchcell.datamodels.environment-perturbation]]``.
# --------------------------------------------------------------------------- #
INCHIKEY_PATTERN = r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$"
CHEBI_ID_PATTERN = r"^CHEBI:\d+$"


class ConcentrationUnit(StrEnum):
    """UO-aligned units for a dose / physical-factor magnitude (G2 unit typing).

    A typed enum (never a free string) so 'uM' / 'µM' / 'micromolar' can never
    silently coexist across datasets. Temperature units live on ``TemperatureUnit``.
    """

    molar = "M"
    millimolar = "mM"
    micromolar = "uM"
    nanomolar = "nM"
    percent_v_v = "percent_v/v"
    percent_w_v = "percent_w/v"
    ug_per_ml = "ug/mL"
    g_per_l = "g/L"
    ph = "pH"  # dimensionless -log10[H+]; magnitude unit for a PhysicalFactor.ph edit


class DoseBasis(StrEnum):
    """How a dose was SET when the molar value is not released (dose PROVENANCE, not a
    phenotypic consequence): a target-inhibition endpoint or an explicit fixed dose.
    """

    IC30 = "IC30"
    IC50 = "IC50"
    MIC = "MIC"
    fixed = "fixed"
    reduced_from_standard = "reduced_from_standard"  # a partial drop-out: the recipe level lowered by an amount the source does not state


class PhysicalFactor(StrEnum):
    """A neutral, scalar physical/physiological environment variable.

    NOT a compound, NOT a consequence word. Temperature is deliberately ABSENT --
    it is carried on ``Environment.temperature`` (single canonical encoding, M2),
    never duplicated as a perturbation.

    - ``nutrient_dropout``: a normally-present medium nutrient is REMOVED (amino-acid or
      vitamin drop-out). The removed nutrient is named on the perturbation's ``agent``
      ``Compound`` (e.g. L-lysine), so a dropout joins on the SAME compound entity a
      dataset that ADDS that nutrient would use.
    - ``radiation``: ionizing/UV irradiation dose (``magnitude`` in the dose unit when
      released; qualitative when only 'irradiated' is reported).
    """

    ph = "pH"
    osmolarity = "osmolarity"
    carbon_source = "carbon_source"
    nitrogen_source = "nitrogen_source"
    ionic_strength = "ionic_strength"
    nutrient_dropout = "nutrient_dropout"
    radiation = "radiation"


class Compound(ProvenanceGapMixin):
    """Chemical identity of a small molecule, keyed by a canonical InChIKey.

    Identity is carried by stable, resolvable identifiers -- not the human name alone.
    ``inchikey`` (the canonical hash of the standard InChI) is the primary key;
    ``pubchem_cid`` / ``chebi_id`` are redundant cross-references; ``smiles`` / ``inchi``
    are auxiliary structure strings. ``roles`` holds ChEBI ROLE terms (the compound's
    mode of action / chemical role, e.g. 'oxidising agent') -- the ontology home for a
    compound's biological role, REPLACING the deleted ``stress_category`` consequence
    field (M1). Only ``name`` is required; identifiers are filled as SOURCED, never
    guessed (provenance discipline).
    """

    name: str = Field(description="human-readable compound name, e.g. 'isobutanol'")
    inchikey: str | None = Field(
        default=None,
        description="canonical InChIKey (14-10-1 blocks), the primary identity key",
    )
    inchi: str | None = Field(
        default=None, description="standard InChI string; None if unknown"
    )
    smiles: str | None = Field(
        default=None,
        description="canonical SMILES (auxiliary structure); None if unknown",
    )
    pubchem_cid: int | None = Field(
        default=None,
        description="PubChem CID as an integer, e.g. 6560; None if unmapped",
    )
    chebi_id: str | None = Field(
        default=None, description="ChEBI CURIE, e.g. 'CHEBI:16236'; None if unmapped"
    )
    roles: list[str] = Field(
        default_factory=list,
        description="ChEBI role terms (mode of action / chemical role); empty if unknown",
    )

    @field_validator("inchikey", mode="after")
    @classmethod
    def _validate_inchikey(cls, v: str | None) -> str | None:
        """InChIKey, when present, has the canonical 14-10-1 block form."""
        if v is not None and not re.match(INCHIKEY_PATTERN, v):
            raise ValueError(
                f"invalid InChIKey {v!r}; expected 'XXXXXXXXXXXXXX-XXXXXXXXXX-X'"
            )
        return v

    @field_validator("chebi_id", mode="after")
    @classmethod
    def _validate_chebi_id(cls, v: str | None) -> str | None:
        """ChEBI id, when present, is a well-formed ``CHEBI:NNNN`` CURIE."""
        if v is not None and not re.match(CHEBI_ID_PATTERN, v):
            raise ValueError(f"invalid ChEBI id {v!r}; expected 'CHEBI:NNNN'")
        return v

    @field_validator("pubchem_cid", mode="after")
    @classmethod
    def _validate_pubchem_cid(cls, v: int | None) -> int | None:
        """PubChem CID, when present, is a positive integer."""
        if v is not None and v < 1:
            raise ValueError(f"pubchem_cid must be a positive integer, got {v}")
        return v


class Concentration(ModelStrict):
    """A dose of an environmental agent: a numeric value+unit and/or a target-basis.

    Used for both a small-molecule ``concentration`` and a physical-factor
    ``magnitude`` (a generic value+unit record). Either a numeric ``value`` (with a
    typed ``unit``) or a ``basis`` (how the dose was set, e.g. an ``IC30`` target) must
    be present -- screens routinely fix a compound at its IC30 without releasing the
    per-compound molar value, so ``value`` may be ``None`` while ``basis=IC30``.
    """

    value: float | None = Field(
        default=None,
        description="numeric dose; None when only a target-inhibition basis is known",
    )
    unit: ConcentrationUnit | None = Field(
        default=None, description="typed UO-aligned unit; required when value is set"
    )
    basis: DoseBasis | None = Field(
        default=None,
        description="how the dose was set (dose provenance), e.g. IC30 | IC50 | fixed",
    )

    @model_validator(mode="after")
    def _check(self) -> "Concentration":
        """Require a numeric value+unit or a basis; value must be non-negative."""
        if self.value is None and self.basis is None:
            raise ValueError("Concentration needs at least a numeric value or a basis")
        if self.value is not None:
            if self.unit is None:
                raise ValueError("a numeric concentration value requires a unit")
            if self.value < 0:
                raise ValueError("concentration value must be non-negative")
        return self


class Solvent(ModelStrict):
    """The vehicle a compound was dissolved in and its final fraction in the medium.

    ``compound`` optionally carries the vehicle's typed chemical identity (a reused
    ``Compound``); ``name`` remains the plain label for the common case. A vehicle the
    source does not state (Hoepfner's concentrated stocks above the 200 uM solubility
    ceiling of its 2% DMSO normalization) is ``SmallMoleculePerturbation.solvent=None``
    with a ``ProvenanceGap`` on ``solvent``; this class is not itself a gap carrier
    because it sits in the schema closure of every small-molecule dataset (#507).
    """

    name: str = Field(description="solvent name, e.g. 'DMSO' | 'water' | 'ethanol'")
    percent: float | None = Field(
        default=None,
        description="final solvent fraction in the medium, percent v/v (e.g. 1.0 = 1%)",
    )
    compound: Compound | None = Field(
        default=None,
        description="typed chemical identity of the vehicle; None if plain",
    )


# --------------------------------------------------------------------------- #
# Media as COMPONENTS (provenance-first). A base medium resolves to a list of
# typed ``MediaComponent`` ingredients -- each a reused ``Compound`` (ChEBI /
# InChIKey / SMILES when sourced) at a ``Concentration`` -- plus deliberately
# omitted ``dropouts``. ``ComponentDefinition`` marks each ingredient's identity
# completeness so an under-characterized medium stays queryable (``open_gaps``)
# and fillable later; the amount axis is orthogonal (``concentration=None`` ==
# amount not yet sourced). ``is_synthetic`` records defined-by-construction
# (SD/SC/YNB) vs natural/complex (YPD, corn steep liquor) -- orthogonal to how
# well the composition is known (a natural medium mass-spec'd into ``defined``
# components is still ``is_synthetic=False``). Selection agents (canavanine /
# G418 / clonNAT) are COMPONENTS, not ``EnvironmentPerturbation``s: they are
# constant to the medium, not the studied edit -- a documented stopgap for the
# absent genotype x medium mechanistic layer (see the media dendron note). A
# future cobra/AMICI adapter maps components -> exchange bounds / species; those
# model conventions do NOT live here.
# Design: ``[[torchcell.datamodels.media-components]]``.
# --------------------------------------------------------------------------- #


class MediaComponentRole(StrEnum):
    """Functional role of a medium component (never a phenotypic consequence)."""

    carbon_source = "carbon_source"
    nitrogen_source = "nitrogen_source"
    amino_acid = "amino_acid"
    nucleobase = "nucleobase"
    vitamin = "vitamin"
    trace_element = "trace_element"
    bulk_salt = "bulk_salt"
    buffer = "buffer"
    selection_agent = "selection_agent"
    gelling_agent = "gelling_agent"
    complex_ingredient = "complex_ingredient"
    other = "other"


class ComponentDefinition(StrEnum):
    """Identity/composition completeness of a component (the 'what is it' axis).

    Orthogonal to the amount axis (``MediaComponent.concentration is None`` means
    the amount is not yet sourced). ``composition_deferred`` is a DEFINED sub-mix
    we have not expanded (commercial YNB, the SC 'amino-acid supplement'),
    resolvable from a cited protocol -> follow ``defers_to``.
    ``intrinsically_undefined`` is a batch-variable biological digest (peptone,
    yeast extract, corn steep liquor) that no recipe fully pins; if later
    mass-spec'd, its measured constituents become their own ``defined`` components.
    """

    defined = "defined"
    composition_deferred = "composition_deferred"
    intrinsically_undefined = "intrinsically_undefined"


class MediaComponent(ModelStrict):
    """One ingredient of a medium: a (possibly undefined) compound at a dose.

    Reuses the typed ``Compound`` (name-only when undefined; InChIKey / ChEBI /
    SMILES filled as SOURCED) and ``Concentration``. ``provenance`` is a LIST of
    ``SourcedValue`` (quote + sha256): a component's value can be corroborated by
    several papers, and the paper we READ may differ from the one that ORIGINATES
    the recipe. ``defers_to`` names cited papers holding a fuller/original
    definition not yet mirrored+quoted; following one promotes it into
    ``provenance`` and can flip ``definition`` to ``defined``.
    """

    compound: Compound
    role: MediaComponentRole
    concentration: Concentration | None = Field(
        default=None, description="amount in the final medium; None if not yet sourced"
    )
    definition: ComponentDefinition = Field(
        default=ComponentDefinition.defined,
        description="identity completeness; drives open_gaps + is_fully_characterized",
    )
    provenance: list[SourcedValue] = Field(
        default_factory=list,
        description="sourced (quote+sha256) justifications; a LIST for corroboration "
        "+ deferral-chain traceability",
    )
    defers_to: list[str] = Field(
        default_factory=list,
        description="citation_keys of papers holding a fuller/original definition not "
        "yet mirrored+quoted; follow to fill a composition_deferred gap",
    )
    note: str | None = Field(
        default=None, description="gap detail / mechanism / role rationale"
    )


class Media(ModelStrict):
    """Growth medium resolved to typed COMPONENTS (provenance-first).

    ``name`` + ``state`` remain the human label; ``is_synthetic`` (REQUIRED)
    records defined-by-construction vs natural/complex; ``components`` is the
    compositional breakdown; ``dropouts`` are auxotrophic components deliberately
    OMITTED (recorded so 'omitted' is distinguishable from 'not yet listed').
    ``is_fully_characterized`` / ``open_gaps`` make under-definition queryable.
    """

    name: str
    state: str
    is_synthetic: bool = Field(
        description="True = chemically defined by construction (SD/SC/YNB/SGA); "
        "False = natural/complex (YPD, corn steep liquor). Orthogonal to how well "
        "the composition is characterized."
    )
    base_medium: str | None = Field(
        default=None,
        description="canonical base label for grouping, e.g. 'SD_MSG' | 'YNB' | 'SC' | 'YPD'",
    )
    components: list[MediaComponent] = Field(
        default_factory=list,
        description="compositional breakdown; empty = composition not yet entered",
    )
    dropouts: list[Compound] = Field(
        default_factory=list,
        description="auxotrophic components deliberately OMITTED (e.g. -His/-Arg/-Lys/-Ura)",
    )
    provenance: list[SourcedValue] = Field(
        default_factory=list,
        description="recipe-level sourced justifications for the medium as a whole",
    )

    @field_validator("state", mode="after")
    @classmethod
    def validate_state(cls, v: str) -> str:
        """Validate that state is one of solid, liquid, or gas."""
        if v not in ["solid", "liquid", "gas"]:
            raise ValueError('state must be one of "solid", "liquid", or "gas"')
        return v

    @property
    def is_fully_characterized(self) -> bool:
        """True iff every listed component has a defined identity AND a concentration."""
        return bool(self.components) and all(
            c.definition is ComponentDefinition.defined and c.concentration is not None
            for c in self.components
        )

    @property
    def open_gaps(self) -> list[str]:
        """Component names still needing a fuller definition or a concentration."""
        return [
            c.compound.name
            for c in self.components
            if c.definition is not ComponentDefinition.defined
            or c.concentration is None
        ]


class EnvironmentPerturbation(ProvenanceGapMixin):
    """Base: a defined change to the growth environment vs the base medium.

    A gap carrier: a dose whose molar value the primary never states, or a solvent
    it never names, is declared as a typed ``ProvenanceGap`` on that field rather
    than left as a silent ``None``.
    """

    perturbation_type: str
    description: str


class SmallMoleculePerturbation(EnvironmentPerturbation, ModelStrict):
    """An added chemical SPECIES (drug, acid, alcohol, salt, oxidant, ...).

    Covers any medium-borne small molecule dosed at a ``Concentration`` or a target
    basis (e.g. IC30). Chemical identity is a typed ``Compound`` (keyed by InChIKey);
    the compound's mode of action / physiological role is a ChEBI ROLE on
    ``compound.roles`` -- there is deliberately NO consequence field (the former
    ``stress_category`` was a category error, M1). Whether the strain is sensitive or
    tolerant is an ``EnvironmentResponsePhenotype`` property, not part of the edit.
    """

    perturbation_type: Literal["small_molecule"] = "small_molecule"
    description: str = "Small-molecule compound added to the base medium"
    compound: Compound = Field(
        description="typed chemical identity of the added species"
    )
    concentration: Concentration = Field(
        description="dose (numeric value+unit and/or basis such as IC30)"
    )
    solvent: Solvent | None = Field(
        default=None,
        description="vehicle the compound was delivered in; None if dissolved directly",
    )


class EnvironmentPhysicalPerturbation(EnvironmentPerturbation, ModelStrict):
    """A neutral, scalar physical/physiological environment FACTOR (pH, osmolarity, ...).

    Use when the edit is a physical variable rather than a specific added compound --
    e.g. a shift in medium pH or osmolarity, or a change of carbon source. ``factor``
    names the neutral variable (never a consequence word); ``magnitude`` gives its
    typed value+unit when quantitative; ``agent`` optionally names the chemical species
    that realizes the factor (e.g. the salt used to set osmolarity) as a reused
    ``Compound``. Temperature is NOT here -- it lives on ``Environment.temperature``.
    When a single named compound IS the edit (NaCl, H2O2, ethanol), prefer
    ``SmallMoleculePerturbation``.
    """

    perturbation_type: Literal["environment_physical"] = "environment_physical"
    description: str = "Scalar physical/physiological environment factor"
    factor: PhysicalFactor = Field(description="the neutral physical variable changed")
    magnitude: Concentration | None = Field(
        default=None,
        description="typed value+unit of the factor; None for a purely qualitative change",
    )
    agent: Compound | None = Field(
        default=None,
        description="chemical species realizing the factor (e.g. the salt for osmolarity)",
    )


class BiologicAgentClass(StrEnum):
    """Material class of a proteinaceous / biologic agent added to the medium.

    Identity of a biologic is carried by sequence / UniProt, NOT an InChIKey -- which is
    exactly why it cannot be a ``SmallMoleculePerturbation``.

    - ``peptide``: a short (ribosomal or synthetic) peptide, e.g. an antimicrobial plant
      defensin.
    - ``protein``: a full-length protein / enzyme.
    - ``antibody``: an immunoglobulin or antibody fragment.
    - ``toxin``: a proteinaceous toxin as an ADDED agent (never a phenotypic consequence).
    """

    peptide = "peptide"
    protein = "protein"
    antibody = "antibody"
    toxin = "toxin"


class BiologicPerturbation(EnvironmentPerturbation, ModelStrict):
    """An added BIOLOGIC agent (peptide / protein / antibody / toxin).

    Use when the medium-borne agent is proteinaceous rather than a small molecule: its
    identity is a sequence / UniProt accession, not an InChIKey, so a ``Compound`` (keyed
    by InChIKey) cannot represent it. Names the EDIT (the agent ADDED); whether the strain
    is sensitive or tolerant is an ``EnvironmentResponsePhenotype`` property, never part of
    the edit (M1). Mirrors ``SmallMoleculePerturbation`` on the biologic axis.
    """

    perturbation_type: Literal["biologic"] = "biologic"
    description: str = (
        "Biologic (peptide/protein/antibody/toxin) agent added to the medium"
    )
    agent_class: BiologicAgentClass = Field(
        description="material class of the biologic agent added"
    )
    name: str = Field(description="agent name, e.g. 'plant defensin DmAMP1'")
    uniprot_id: str | None = Field(
        default=None, description="UniProt accession of the agent; None if unmapped"
    )
    sequence: str | None = Field(
        default=None, description="amino-acid sequence of the agent; None if unknown"
    )
    concentration: Concentration = Field(
        description="dose (numeric value+unit and/or basis)"
    )


EnvironmentPerturbationType = (
    SmallMoleculePerturbation | EnvironmentPhysicalPerturbation | BiologicPerturbation
)


class EndpointRule(StrEnum):
    """When a culture was read out (the endpoint is part of what was measured).

    - ``fixed_duration``: after a stated time (``Environment.duration_hours``).
    - ``fixed_generations``: after a stated number of doublings
      (``Environment.duration_generations``), e.g. serial pooled passages.
    - ``until_control_saturation``: when the vehicle-only control saturated, a
      variable time (Wildenhain: "approximately 18 h or until solvent-treated control
      cultures were saturated").
    """

    fixed_duration = "fixed_duration"
    fixed_generations = "fixed_generations"
    until_control_saturation = "until_control_saturation"


class CultureFormat(ProvenanceGapMixin):
    """The physical culture a measurement was grown in.

    Vessel, working volume, agitation, inoculum and endpoint rule are part of the
    environment: a static 100 uL microwell culture and a shaken 1.6 mL deep-well
    culture differ in aeration and in how many generations fit before saturation.
    Every field is optional; a field the source does not state can be a typed gap.
    ``vessel`` is verbatim (``"96-well plate"``, ``"24-well plate (Greiner 662102)"``).
    ``shaking_rpm`` 0.0 means static. ``inoculum_cells`` counts cells per culture;
    ``inoculum_cells_per_strain`` is the pooled-screen form (cells of each strain).
    """

    vessel: str | None = None
    working_volume_ul: float | None = None
    shaking_rpm: float | None = None
    inoculum_cells: float | None = None
    inoculum_cells_per_strain: float | None = None
    inoculum_od600: float | None = None
    endpoint: EndpointRule | None = None
    provenance: list[SourcedValue] = Field(
        default_factory=list,
        description="sourced (quote + sha256) justifications for the stated fields",
    )

    @model_validator(mode="after")
    def _check_culture(self) -> "CultureFormat":
        """Volumes, agitation and inocula are non-negative."""
        for name in (
            "working_volume_ul",
            "shaking_rpm",
            "inoculum_cells",
            "inoculum_cells_per_strain",
            "inoculum_od600",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"CultureFormat.{name} must be non-negative")
        return self


class PreCultureSource(StrEnum):
    """What the screened culture was inoculated FROM.

    - ``frozen_stock``: straight from a frozen pool, no pre-culture (Hillenmeyer's
      negative generation counts: "taken directly from the freezer").
    - ``overnight_culture``: an overnight culture of unstated phase.
    - ``log_phase_culture``: grown to log phase before treatment (Hillenmeyer's
      positive counts: "grown overnight until log phase (OD600= 2.0)").
    - ``thaw_recovery``: thawed and recovered briefly in medium (Hoepfner HOP:
      "thawed and recovered for 3 h in YPD").
    """

    frozen_stock = "frozen_stock"
    overnight_culture = "overnight_culture"
    log_phase_culture = "log_phase_culture"
    thaw_recovery = "thaw_recovery"


class PreCulture(ProvenanceGapMixin):
    """The culture step BEFORE treatment, which sets the cells' state at time zero.

    A signed generation count in a source (Hillenmeyer ``-5gen`` vs ``5gen``) encodes
    two facts: the magnitude is the treatment exposure
    (``Environment.duration_generations``), and the SIGN says whether a pre-culture
    happened (negative = ``frozen_stock``; positive = a YPD log-phase pre-culture of
    about 10 generations). ``source_label`` keeps the source token verbatim so the
    split is auditable. ``medium`` is ``None`` for a ``frozen_stock`` start.
    """

    source: PreCultureSource
    medium: Media | None = None
    generations: float | None = None
    duration_hours: float | None = None
    od600_at_transfer: float | None = None
    source_label: str | None = Field(
        default=None,
        description="the source's own token for this step, verbatim (e.g. '-5gen')",
    )
    provenance: list[SourcedValue] = Field(
        default_factory=list,
        description="sourced (quote + sha256) justifications for the stated fields",
    )

    @model_validator(mode="after")
    def _check_preculture(self) -> "PreCulture":
        """A frozen-stock start has no pre-culture medium; quantities are non-negative."""
        if self.source is PreCultureSource.frozen_stock and self.medium is not None:
            raise ValueError("a frozen_stock start has no pre-culture medium")
        for name in ("generations", "duration_hours", "od600_at_transfer"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"PreCulture.{name} must be non-negative")
        return self


class Environment(ProvenanceGapMixin):
    """Experimental environment: base medium + temperature + optional perturbations.

    ``perturbations`` mirror ``Genotype.perturbations`` on the environment axis: the
    added compounds / physical stresses applied on top of the base ``media``. It
    defaults to an empty list, so an unperturbed environment (every dataset predating
    the environment-perturbation ontology) is unchanged. ``aerobicity`` records the
    oxygen regime (anaerobic fermentation is standard for biofuel screens);
    ``duration_hours`` is the optional treatment time. ``temperature`` is optional: a
    secondary curation layer (YeastPhenome) may not carry it, in which case it is a
    ``ProvenanceGap`` (``field='temperature'``), NOT a guessed value.

    The culture protocol (#507) lives on the subclass ``CultureEnvironment``, not
    here: this class is in the schema closure of every served dataset.
    """

    media: Media
    temperature: Temperature | None = None
    perturbations: list[EnvironmentPerturbationType] = Field(
        default_factory=list,
        description="environmental perturbations (added compounds / physical stresses) "
        "on top of the base medium; empty for an unperturbed base environment",
    )
    aerobicity: str = Field(
        default="aerobic", description="'aerobic' | 'anaerobic' | 'microaerobic'"
    )
    duration_hours: float | None = Field(
        default=None, description="treatment/growth duration in hours; None if unstated"
    )
    duration_generations: float | None = Field(
        default=None,
        description="treatment/exposure duration in GENERATIONS of growth (competitive-"
        "growth screens dose exposure in doublings, not hours, e.g. Hillenmeyer 5/15/20 "
        "generations); None if not applicable. Distinct exposure durations are distinct "
        "environments, so this is part of the environment identity.",
    )

    @field_validator("aerobicity", mode="after")
    @classmethod
    def validate_aerobicity(cls, v: str) -> str:
        """Aerobicity is one of the three supported oxygen regimes."""
        if v not in {"aerobic", "anaerobic", "microaerobic"}:
            raise ValueError(
                f"aerobicity must be aerobic/anaerobic/microaerobic, got {v!r}"
            )
        return v


class CultureEnvironment(Environment):
    """An ``Environment`` that also states its culture protocol (#507 point 8).

    Same medium, temperature, perturbations, oxygen regime and duration as the base
    class (so the medium-level cross-dataset aggregate is unchanged: ``media`` is the
    same field holding the same ``Media``), plus three optional, gappable slots:

    - ``culture_format``: vessel, working volume, shaking, inoculum, endpoint rule.
    - ``pre_culture``: the step before treatment, including what a SIGNED generation
      count means (Hillenmeyer ``-5gen``: from the freezer; ``5gen``: after a YPD
      log-phase pre-culture). ``duration_generations`` always holds the magnitude.
    - ``auxotroph_supplements``: nutrients added to the base medium to complement the
      strain background's auxotrophies (His/Leu/Ura/Met for a BY strain on a minimal
      medium). They sit beside ``media`` rather than inside it, so the base medium
      stays the shared, joinable entity and the strain-specific addition is explicit.

    ``None`` means not stated; a ``ProvenanceGap`` on the field says why. A separate
    class (not fields on ``Environment``) because ``Environment`` is in the schema
    closure of every served dataset and a new field there would mark every built store
    stale; the chemogenomic experiment classes declare this type explicitly, so
    pydantic serializes these fields.
    """

    culture_format: CultureFormat | None = Field(
        default=None,
        description="vessel / volume / shaking / inoculum / endpoint; None if unstated",
    )
    pre_culture: PreCulture | None = Field(
        default=None, description="the culture step before treatment; None if unstated"
    )
    auxotroph_supplements: list[MediaComponent] | None = Field(
        default=None,
        description="nutrients added to complement the background's auxotrophies; None "
        "if unstated (a typed gap when the source implies but does not name them)",
    )


# Phenotype
class Phenotype(ProvenanceGapMixin):
    """Base phenotype describing an observed label and its graph level.

    Inherits ``provenance_gaps`` (+ its two honesty validators) from
    ``ProvenanceGapMixin`` -- a phenotype field the source does not carry (n_samples,
    an uncertainty) is a typed absence, shared with ``Environment``.
    """

    graph_level: str = Field(
        description="most natural level of graph at which phenotype is observed"
    )
    label_name: str = Field(description="name of label")
    label_statistic_name: str | None = Field(
        default=None,
        description="name of error or confidence statistic related to label",
    )

    @model_validator(mode="after")
    def validate_fields(self) -> "Phenotype":
        """Validate that graph_level is one of the supported graph levels."""
        valid_graph_levels = {
            "edge",
            "node",
            "hyperedge",
            "subgraph",
            "global",
            "metabolism",
            "gene ontology",
        }
        if self.graph_level not in valid_graph_levels:
            raise ValueError(
                f"graph_level must be one of: {', '.join(valid_graph_levels)}"
            )
        return self

    @model_validator(mode="after")
    def validate_label_fields(self) -> "Phenotype":
        """label_name / label_statistic_name must name fields on the concrete class.

        Inherited by every Phenotype subclass (replaces the formerly identical
        per-subclass copies). ``type(self).__annotations__`` is the concrete
        subclass's own field set, so each subclass is checked against its own
        declared fields.
        """
        own_fields = type(self).__annotations__
        if self.label_name not in own_fields:
            raise ValueError(
                f"label_name '{self.label_name}' must be a class attribute"
            )
        if (
            self.label_statistic_name is not None
            and self.label_statistic_name not in own_fields
        ):
            raise ValueError(
                f"label_statistic_name '{self.label_statistic_name}' "
                "must be a class attribute"
            )
        return self

    def __getitem__(self, key: str) -> Any:  # heterogeneous phenotype field values
        """Return the attribute value for the given field name."""
        return getattr(self, key)


class UncertaintyType(StrEnum):
    """What a reported uncertainty number IS, so it converts to an SE correctly.

    Rigor comes from naming the statistic: a bootstrap SD of an estimator is
    already an SE (never divide it by sqrt(n) again), whereas a sample SD of
    observations must be divided. There is deliberately NO ``unknown`` -- strict
    labelling: if the kind is unknown we do not ingest the value.
    """

    sample_sd = "sample_sd"  # SD of observations -> SE = sd / sqrt(n)
    standard_error = "standard_error"  # already SE of the mean -> use as-is
    bootstrap_se = "bootstrap_se"  # bootstrap SD of the estimator ~ SE -> as-is
    variance = "variance"  # sample variance -> SE = sqrt(var / n)
    ci95 = "ci95"  # 95% CI half-width -> SE = hw / 1.96


class SampleUnit(StrEnum):
    """What one sample in ``n_samples`` physically is. Add values as datasets need
    them (do not pre-populate). ``screen`` vs ``colony`` matters: Costanzo colonies
    are pseudoreplicates; the independent unit is the screen.
    """

    colony = "colony"
    screen = "screen"
    biological_replicate = "biological_replicate"
    technical_replicate = "technical_replicate"
    pooled = "pooled"


_Z95 = 1.959963984540054  # standard-normal two-sided 95% quantile


def derive_se(
    uncertainty: float | None,
    uncertainty_type: UncertaintyType | None,
    n_samples: int | None,
) -> float | None:
    """Derive the standard error of the mean from a reported uncertainty + its kind.

    ``standard_error``/``bootstrap_se`` -> as-is (already an SE); ``sample_sd`` ->
    sd/sqrt(n); ``variance`` -> sqrt(var/n); ``ci95`` -> half-width/1.96. n_samples
    is required for the kinds that divide. Returns None when nothing is reported.
    """
    if uncertainty is None or uncertainty_type is None:
        return None
    if uncertainty_type in (
        UncertaintyType.standard_error,
        UncertaintyType.bootstrap_se,
    ):
        return uncertainty
    if uncertainty_type is UncertaintyType.ci95:
        return uncertainty / _Z95
    if n_samples is None or n_samples < 1:
        raise ValueError(
            f"n_samples (>=1) required to derive SE from {uncertainty_type}"
        )
    if uncertainty_type is UncertaintyType.sample_sd:
        return uncertainty / math.sqrt(n_samples)
    if uncertainty_type is UncertaintyType.variance:
        return math.sqrt(uncertainty / n_samples)
    raise ValueError(f"unhandled uncertainty_type: {uncertainty_type}")


class FitnessPhenotype(Phenotype, ModelStrict):
    """Fitness phenotype with fitness value and uncertainty statistics.

    Uncertainty ontology: the source-reported number lives in ``fitness_uncertainty``
    with its ``fitness_uncertainty_type``; ``n_samples`` + ``sample_unit`` give the
    replicate design; ``fitness_se`` is the DERIVED, ML-facing standard error
    (auto-computed via ``derive_se`` when not supplied). ``fitness_std`` is
    DEPRECATED (superseded by uncertainty/type; retained until loaders migrate).
    """

    graph_level: str = "global"
    label_name: str = "fitness"
    label_statistic_name: str = "fitness_se"
    fitness: float = Field(description="ko_growth_rate/wt_growth_rate")
    fitness_se: float | None = Field(
        default=None,
        description="fitness standard error (primary uncertainty statistic)",
    )
    fitness_std: float | None = Field(
        default=None,
        description="fitness standard deviation (raw data from publication)",
    )
    n_samples: int | None = Field(
        default=None,
        description="""Number of replicate measurements of the fitness ratio.
        For experiment: n independent measurements of strain_of_interest/wt.
        For reference: n independent measurements of wt control.
        Note: numerator and denominator may have different sample sizes;
        this tracks the complete ratio measurement.""",
    )
    fitness_uncertainty: float | None = Field(
        default=None,
        description="Source-reported uncertainty number, verbatim (its meaning is "
        "given by fitness_uncertainty_type).",
    )
    fitness_uncertainty_type: UncertaintyType | None = Field(
        default=None, description="What fitness_uncertainty IS (sample_sd, ...)."
    )
    sample_unit: SampleUnit | None = Field(
        default=None,
        description="What one sample in n_samples is (colony, screen, ...).",
    )
    screen_id: str | None = Field(
        default=None,
        description="the SCREEN this measurement came from, when one publication "
        "releases the same (genotype, environment) from more than one screen with "
        "different values (Kuzmin 2020: the main diagnostic-array screen of Table S1 "
        "and the pilot genome-wide-array screens of Table S3 share 24,193 digenic "
        "crosses). Same meaning as EnvironmentResponsePhenotype.screen_id: it keeps "
        "two measurements of one strain from different screens distinguishable "
        "instead of storing two values nothing tells apart. None when the source "
        "releases one screen per measurement.",
    )

    @field_validator("fitness")
    def validate_fitness(cls, v: float) -> float:
        """Reject NaN fitness and clamp non-positive values to zero."""
        if math.isnan(v):
            raise ValueError("Fitness cannot be NaN")
        if v <= 0:
            return 0.0
        return v

    @field_validator("n_samples")
    def validate_n_samples(cls, v: int | None) -> int | None:
        """Validate that n_samples is a positive integer or None."""
        if v is not None and (not isinstance(v, int) or v < 1):
            raise ValueError(f"n_samples must be a positive integer or None, got: {v}")
        return v

    @model_validator(mode="before")
    @classmethod
    def _fill_fitness_se(cls, data: Any) -> Any:
        """Derive the ML-facing fitness_se from the reported uncertainty.

        ModelStrict is frozen, so we fill fitness_se BEFORE construction rather than
        assign after. Skips when not safely derivable (e.g. sample_sd without n);
        ``_check_uncertainty`` then raises the precise error.
        """
        if not isinstance(data, dict):
            return data
        unc = data.get("fitness_uncertainty")
        typ = data.get("fitness_uncertainty_type")
        if unc is None or typ is None or data.get("fitness_se") is not None:
            return data
        typ = UncertaintyType(typ)
        n = data.get("n_samples")
        if typ in (UncertaintyType.sample_sd, UncertaintyType.variance) and n is None:
            return data
        data["fitness_se"] = derive_se(unc, typ, n)
        return data

    @model_validator(mode="after")
    def _check_uncertainty(self) -> "FitnessPhenotype":
        """Strict invariant: no unlabelled uncertainty (reported<->type both-or-
        neither); n_samples + sample_unit required for kinds that divide.
        """
        unc, typ = self.fitness_uncertainty, self.fitness_uncertainty_type
        if (unc is None) != (typ is None):
            raise ValueError(
                "fitness_uncertainty and fitness_uncertainty_type must both be set "
                "or both be None (no unlabelled uncertainty)"
            )
        if typ in (UncertaintyType.sample_sd, UncertaintyType.variance) and (
            self.n_samples is None or self.sample_unit is None
        ):
            raise ValueError(f"n_samples and sample_unit are required for {typ}")
        return self


class GeneEssentialityPhenotype(Phenotype, ModelStrict):
    """Phenotype indicating whether a gene knockout is lethal."""

    graph_level: str = "node"
    label_name: str = "is_essential"
    is_essential: bool = Field(
        default=True, description="gene knockout leading cell death."
    )


class SyntheticLethalityPhenotype(Phenotype, ModelStrict):
    """Phenotype indicating synthetic lethality between perturbed genes."""

    graph_level: str = "edge"
    label_name: str = "is_synthetic_lethal"
    label_statistic_name: str = "synthetic_lethality_statistic_score"
    is_synthetic_lethal: bool = Field(
        default=True,
        description="synthetic lethality occurs when the combination of mutations in"
        "two or more genes leads to cell death, whereas a mutation in only one of these"
        "genes does not affect the viability of the cell.",
    )
    synthetic_lethality_statistic_score: float | None = Field(
        default=None,
        description="statistical score computed in [SynLethDB](https://synlethdb.sist.shanghaitech.edu.cn/#/",
    )


class SyntheticRescuePhenotype(Phenotype, ModelStrict):
    """Phenotype indicating one perturbation rescues another's deleterious effect."""

    graph_level: str = "edge"
    label_name: str = "is_synthetic_rescue"
    label_statistic_name: str = "synthetic_rescue_statistic_score"
    is_synthetic_rescue: bool = Field(
        default=True,
        description="synthetic rescue occurs when a mutation in one gene compensates"
        "for the deleterious effects of a mutation in another gene, thereby restoring"
        "normal function or viability to the cell",
    )
    synthetic_rescue_statistic_score: float | None = Field(
        default=None,
        description="statistical score computed in [SynLethDB](https://synlethdb.sist.shanghaitech.edu.cn/#/",
    )


class GeneInteractionPhenotype(Phenotype, ModelStrict):
    """Phenotype holding a gene interaction score and its p-value."""

    graph_level: str = "hyperedge"
    label_name: str = "gene_interaction"
    label_statistic_name: str = "gene_interaction_p_value"
    gene_interaction: float = Field(
        description="""epsilon, tau, or analogous gene interaction value.
        Computed from composite fitness phenotypes."""
    )
    gene_interaction_p_value: float | None = Field(
        default=None, description="p-value of gene interaction"
    )
    screen_id: str | None = Field(
        default=None,
        description="the SCREEN this interaction score came from, when one "
        "publication releases the same (genotype, environment) from more than one "
        "screen with different scores (Kuzmin 2020 Tables S1 and S3). Same meaning as "
        "FitnessPhenotype.screen_id; None when the source releases one screen per "
        "measurement.",
    )

    @field_validator("gene_interaction")
    def validate_fitness(cls, v: float) -> float:
        """Reject NaN gene interaction values."""
        if math.isnan(v):
            raise ValueError("Gene interaction cannot be NaN")
        return v


class CalMorphPhenotype(Phenotype, ModelStrict):
    """Phenotype holding CalMorph morphological measurements and CV statistics."""

    graph_level: str = "global"
    label_name: str = "calmorph"
    label_statistic_name: str = "calmorph_coefficient_of_variation"
    calmorph: dict[str, float] = Field(
        description="Dictionary of CalMorph base morphological measurements (281 parameters)"
    )
    calmorph_coefficient_of_variation: dict[str, float] | None = Field(
        default=None,
        description="Dictionary of coefficient of variation values for CalMorph parameters (220 parameters)",
    )

    # CALMORPH_PARAMETERS: All 501 parameters from Ohya et al. 2005
    # CALMORPH_LABELS: 281 base morphological measurements
    # CALMORPH_STATISTICS: 220 coefficient of variation parameters

    @field_validator("calmorph")
    def validate_calmorph(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate CalMorph base parameters against CALMORPH_LABELS and reject NaN."""
        if not v:
            raise ValueError("calmorph measurements cannot be empty")
        for key, value in v.items():
            if key not in CALMORPH_LABELS:
                raise ValueError(
                    f"Invalid CalMorph base parameter: {key}. "
                    f"Must be one of the 281 base parameters in CALMORPH_LABELS."
                )
            if math.isnan(value):
                raise ValueError(f"calmorph measurement {key} cannot be NaN")
        return v

    @field_validator("calmorph_coefficient_of_variation")
    def validate_cv(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        """Validate CV parameters against CALMORPH_STATISTICS and reject NaN."""
        if v is None:
            return v
        for key, value in v.items():
            if key not in CALMORPH_STATISTICS:
                raise ValueError(
                    f"Invalid CalMorph CV parameter: {key}. "
                    f"Must be one of the 220 CV parameters in CALMORPH_STATISTICS."
                )
            if math.isnan(value):
                raise ValueError(f"CV measurement {key} cannot be NaN")
        return v


class Publication(ModelStrict):
    """Publication reference identified by PubMed ID and/or DOI."""

    pubmed_id: str | None = None
    pubmed_url: str | None = None
    doi: str | None = None
    doi_url: str | None = None

    @model_validator(mode="after")
    def check_pub_info(self) -> "Publication":
        """Require at least one of PubMed ID/DOI and at least one URL."""
        if self.pubmed_id is None and self.doi is None:
            raise ValueError("At least one of PubMed ID or DOI must be provided")
        if self.pubmed_url is None and self.doi_url is None:
            raise ValueError("At least one of PubMed URL or DOI URL must be provided")
        return self


class ExperimentReference(ModelStrict):
    """Reference (wildtype/control) context for an experiment."""

    experiment_reference_type: str = "base"
    dataset_name: str
    genome_reference: ReferenceGenome
    environment_reference: Environment
    phenotype_reference: Phenotype


class Experiment(ModelStrict):
    """Base experiment pairing a genotype and environment with a phenotype."""

    experiment_type: str = "base"
    dataset_name: str
    genotype: Genotype
    environment: Environment
    phenotype: Phenotype


class FitnessExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a fitness experiment."""

    experiment_reference_type: str = "fitness"
    phenotype_reference: FitnessPhenotype


class FitnessExperiment(Experiment, ModelStrict):
    """Experiment measuring a fitness phenotype."""

    experiment_type: str = "fitness"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: FitnessPhenotype


class GeneInteractionExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a gene interaction experiment."""

    experiment_reference_type: str = "gene interaction"
    phenotype_reference: GeneInteractionPhenotype


class GeneInteractionExperiment(Experiment, ModelStrict):
    """Experiment measuring a gene interaction phenotype."""

    experiment_type: str = "gene interaction"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: GeneInteractionPhenotype


class GeneEssentialityExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a gene essentiality experiment."""

    experiment_reference_type: str = "gene essentiality"
    phenotype_reference: GeneEssentialityPhenotype


# shouldn't it jut be one gene for genotype?
class GeneEssentialityExperiment(Experiment, ModelStrict):
    """Experiment measuring a gene essentiality phenotype."""

    experiment_type: str = "gene essentiality"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: GeneEssentialityPhenotype


class SyntheticLethalityExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a synthetic lethality experiment."""

    experiment_reference_type: str = "synthetic lethality"
    phenotype_reference: SyntheticLethalityPhenotype


class SyntheticLethalityExperiment(Experiment, ModelStrict):
    """Experiment measuring a synthetic lethality phenotype."""

    experiment_type: str = "synthetic lethality"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: SyntheticLethalityPhenotype


class SyntheticRescueExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a synthetic rescue experiment."""

    experiment_reference_type: str = "synthetic rescue"
    phenotype_reference: SyntheticRescuePhenotype


class SyntheticRescueExperiment(Experiment, ModelStrict):
    """Experiment measuring a synthetic rescue phenotype."""

    experiment_type: str = "synthetic rescue"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: SyntheticRescuePhenotype


class CalMorphExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a CalMorph experiment."""

    experiment_reference_type: str = "calmorph"
    phenotype_reference: CalMorphPhenotype


class CalMorphExperiment(Experiment, ModelStrict):
    """Experiment measuring a CalMorph phenotype."""

    experiment_type: str = "calmorph"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: CalMorphPhenotype


class MicroarrayExpressionPhenotype(Phenotype, ModelStrict):
    """Microarray expression phenotype with canonical log2 ratio convention.

    CONVENTION: All expression_log2_ratio values follow the torchcell standard:
        expression_log2_ratio = log2(sample_of_interest / reference)

    Where:
        - sample_of_interest = mutant/deletion strain being studied
        - reference = wildtype or common reference pool

    This means:
        - Positive values: gene is MORE expressed in sample vs reference (upregulated)
        - Negative values: gene is LESS expressed in sample vs reference (downregulated)

    NOTE: Source data (GEO) may use different conventions (e.g., log2(reference/sample)).
    Dataset loaders MUST transform to this canonical representation for consistency.

    PRIMARY FIELDS (for BioCypher/ML):
        - expression_log2_ratio: log2 fold change relative to reference
        - expression_log2_ratio_se: Standard error of log2 ratios

    SECONDARY FIELDS (for QC/reproducibility):
        - expression: Absolute expression measurements on linear scale (reference only)
        - expression_log2_ratio_variance: Variance of log2 ratios
        - n_replicates: Number of independent biological/technical replicates
    """

    graph_level: str = "node"
    label_name: str = "expression_log2_ratio"
    label_statistic_name: str = "expression_log2_ratio_se"

    # PRIMARY FIELDS - for BioCypher/Neo4j and ML training
    expression_log2_ratio: dict[str, float] = Field(
        description=(
            "SortedDict of log2 fold change ratios relative to wildtype reference. "
            "CONVENTION: log2(sample/reference) where positive = upregulated, "
            "negative = downregulated. All datasets transformed to this standard."
        ),
        repr=False,  # Hide in repr to avoid clutter
    )
    expression_log2_ratio_se: dict[str, float] | None = Field(
        default=None,
        description=(
            "SortedDict of standard errors (SE) for log2 ratios. "
            "SE = SD / sqrt(n_replicates). "
            "None when unavailable; NaN when n_replicates = 1 (SE undefined)."
        ),
        repr=False,  # Hide in repr to avoid clutter
    )

    # SECONDARY FIELDS - for QC and reproducibility
    expression: dict[str, float] = Field(
        description=(
            "SortedDict of per-gene expression measurements on linear scale. "
            "May be raw probe intensities, background-subtracted, or normalized "
            "(e.g., quantile normalization, housekeeping gene normalization). "
            "NOT the ratio to reference - this is the sample's absolute expression."
        ),
        repr=False,  # Hide in repr to avoid clutter
    )
    expression_log2_ratio_variance: dict[str, float] | None = Field(
        default=None,
        description=(
            "SortedDict of variance for log2 ratios. Variance = SE^2 * n_replicates."
        ),
        repr=False,  # Hide in repr to avoid clutter
    )
    n_replicates: dict[str, int] = Field(
        description="SortedDict of number of independent biological/technical replicates per gene",
        repr=False,  # Hide in repr to avoid clutter
    )

    def __repr__(self) -> str:
        """Custom repr that shows summary statistics instead of full data."""
        expr_count = len(self.expression) if self.expression else 0
        log2_count = (
            len(self.expression_log2_ratio) if self.expression_log2_ratio else 0
        )
        se_count = (
            len(self.expression_log2_ratio_se) if self.expression_log2_ratio_se else 0
        )
        n_replicates_count = len(self.n_replicates) if self.n_replicates else 0

        return (
            f"MicroarrayExpressionPhenotype("
            f"expression_genes={expr_count}, "
            f"log2_ratio_genes={log2_count}, "
            f"log2_se_genes={se_count}, "
            f"n_replicates_genes={n_replicates_count})"
        )

    @field_validator("expression", mode="before")
    def convert_and_validate_expression(cls, v: Any) -> Any:  # raw pre-validation input
        """Coerce expression to a SortedDict and reject empty or infinite values."""
        if v is None:
            raise ValueError("expression measurements cannot be None")
        # Convert to SortedDict for consistent ordering
        if isinstance(v, dict) and not isinstance(v, SortedDict):
            v = SortedDict(v)
        if not v:
            raise ValueError("expression measurements cannot be empty")
        for key, value in v.items():
            # Accept any gene name, not just systematic patterns
            if math.isinf(value):
                raise ValueError(f"Invalid expression value for gene {key}: {value}")
        return v

    @field_validator("expression_log2_ratio", mode="before")
    def convert_and_validate_log2_ratio(cls, v: Any) -> Any:  # raw pre-validation input
        """Coerce log2 ratios to a SortedDict and reject empty input."""
        if v is None:
            raise ValueError("expression_log2_ratio cannot be None")
        # Convert to SortedDict for consistent ordering
        if isinstance(v, dict) and not isinstance(v, SortedDict):
            v = SortedDict(v)
        if not v:
            raise ValueError("expression_log2_ratio cannot be empty")
        # Accept any gene name keys, no validation needed
        return v

    @field_validator("expression_log2_ratio_se", mode="before")
    def convert_and_validate_log2_se(cls, v: Any) -> Any:  # raw pre-validation input
        """Coerce log2 ratio SE to a SortedDict and reject negative finite values."""
        if v is None:
            return v
        # Convert to SortedDict for consistent ordering
        if isinstance(v, dict) and not isinstance(v, SortedDict):
            v = SortedDict(v)
        # SE can be NaN or Inf when n=1
        for key, value in v.items():
            if not (math.isnan(value) or math.isinf(value)) and value < 0:
                raise ValueError(f"SE for {key} cannot be negative: {value}")
        return v

    @field_validator("expression_log2_ratio_variance", mode="before")
    def convert_and_validate_variance(cls, v: Any) -> Any:  # raw pre-validation input
        """Coerce variance to a SortedDict and reject negative non-NaN values."""
        if v is None:
            return v
        # Convert to SortedDict for consistent ordering
        if isinstance(v, dict) and not isinstance(v, SortedDict):
            v = SortedDict(v)
        # Variance must be non-negative
        for key, value in v.items():
            if not math.isnan(value) and value < 0:
                raise ValueError(f"Variance for {key} cannot be negative: {value}")
        return v

    @field_validator("n_replicates", mode="before")
    def convert_and_validate_n_replicates(
        cls, v: Any
    ) -> Any:  # raw pre-validation input
        """Coerce n_replicates to a SortedDict and require positive-integer counts."""
        if v is None:
            raise ValueError(
                "n_replicates cannot be None - it is required for SE interpretation"
            )
        # Must be a per-gene mapping. Raise a clean ValueError (not crash on .items())
        # so union resolution can skip this member when a scalar-n_replicates phenotype
        # (e.g. VisualScorePhenotype) is the real match.
        if not isinstance(v, dict):
            raise ValueError(
                f"n_replicates must be a per-gene dict for this phenotype, got {type(v).__name__}"
            )
        # Convert to SortedDict for consistent ordering
        if not isinstance(v, SortedDict):
            v = SortedDict(v)
        if not v:
            raise ValueError("n_replicates cannot be empty")
        # Validate that all n_replicates are positive integers
        for key, value in v.items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"n_replicates for {key} must be a positive integer, got: {value}"
                )
        return v

    @model_validator(mode="after")
    def validate_matching_keys(self) -> "MicroarrayExpressionPhenotype":
        """Ensure all secondary fields have same keys as expression."""
        # n_replicates must match expression keys
        if set(self.n_replicates.keys()) != set(self.expression.keys()):
            raise ValueError("n_replicates must have the same keys as expression")

        # Optional fields should match if present
        if self.expression_log2_ratio_se is not None:
            if set(self.expression_log2_ratio_se.keys()) != set(
                self.expression_log2_ratio.keys()
            ):
                raise ValueError(
                    "expression_log2_ratio_se must have the same keys as expression_log2_ratio"
                )

        if self.expression_log2_ratio_variance is not None:
            if set(self.expression_log2_ratio_variance.keys()) != set(
                self.expression_log2_ratio.keys()
            ):
                raise ValueError(
                    "expression_log2_ratio_variance must have the same keys as expression_log2_ratio"
                )

        return self


class MicroarrayExpressionExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a microarray expression experiment."""

    experiment_reference_type: str = "microarray_expression"
    phenotype_reference: MicroarrayExpressionPhenotype


class MicroarrayExpressionExperiment(Experiment, ModelStrict):
    """Experiment measuring a microarray expression phenotype."""

    experiment_type: str = "microarray_expression"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: MicroarrayExpressionPhenotype


class RNASeqExpressionPhenotype(Phenotype, ModelStrict):
    """NGS (RNA-seq) expression phenotype: ABSOLUTE per-gene expression, not a ratio.

    Distinct from ``MicroarrayExpressionPhenotype``. That family is a
    perturbation-vs-reference screen and stores ``log2(sample/reference)``. This family is a
    population/whole-transcriptome survey (Caudal 2024 pan-transcriptome of natural isolates)
    where each isolate's transcriptome is measured on its OWN genome -- there is no common
    reference to ratio against, so the stored value is the isolate's ABSOLUTE expression:
    ``expression_tpm`` (transcripts per million) with the raw ``expression_count``.
    Downstream abundance/dispersion metrics (e.g. mean log2 TPM) are DERIVED, never stored.

    Core/accessory handling: a gene ABSENT from an isolate's genome is encoded by KEY
    ABSENCE (it is simply not a key), NEVER a 0 TPM -- honest to the source, which excludes
    isolates that do not carry a given accessory gene from that gene's statistics.

    Provenance: Caudal et al. 2024, Nat. Genet. 56:1278; normalization "mean log2 of the
    normalized read counts (transcripts per million (TPM))". See
    ``[[torchcell.datasets.scerevisiae.caudal2024]]``.
    """

    graph_level: str = "node"
    label_name: str = "expression_tpm"
    label_statistic_name: str | None = None

    expression_tpm: dict[str, float] = Field(
        description=(
            "SortedDict of per-gene absolute expression in transcripts per million (TPM). "
            "Non-negative. A gene absent from the isolate's genome is omitted (no key), "
            "never stored as 0."
        ),
        repr=False,
    )
    expression_count: dict[str, int] = Field(
        description=(
            "SortedDict of per-gene raw mapped-read counts; same keys as expression_tpm."
        ),
        repr=False,
    )
    measurement_type: str = Field(
        default="rnaseq_tpm",
        description="assay/normalization tag, e.g. 'rnaseq_tpm' (batch-normalized TPM)",
    )
    n_mapped_reads: int | None = Field(
        default=None,
        description=(
            "Total clean mapped reads for the isolate (per-sample QC; Caudal kept isolates "
            "with >= 1e6 mapped reads). Per-isolate scalar, not per-gene."
        ),
    )

    def __repr__(self) -> str:
        """Summary repr instead of dumping the per-gene dicts."""
        return (
            f"RNASeqExpressionPhenotype("
            f"tpm_genes={len(self.expression_tpm) if self.expression_tpm else 0}, "
            f"count_genes={len(self.expression_count) if self.expression_count else 0})"
        )

    @field_validator("expression_tpm", mode="before")
    def convert_and_validate_tpm(cls, v: Any) -> Any:  # raw pre-validation input
        """Coerce TPM to a SortedDict; reject empty, infinite, or negative values."""
        if v is None:
            raise ValueError("expression_tpm cannot be None")
        if isinstance(v, dict) and not isinstance(v, SortedDict):
            v = SortedDict(v)
        if not v:
            raise ValueError("expression_tpm cannot be empty")
        for key, value in v.items():
            if math.isinf(value) or math.isnan(value) or value < 0:
                raise ValueError(f"Invalid TPM for gene {key}: {value}")
        return v

    @field_validator("expression_count", mode="before")
    def convert_and_validate_count(cls, v: Any) -> Any:  # raw pre-validation input
        """Coerce counts to a SortedDict; require non-negative integers."""
        if v is None:
            raise ValueError("expression_count cannot be None")
        if not isinstance(v, dict):
            raise ValueError(
                f"expression_count must be a per-gene dict, got {type(v).__name__}"
            )
        if not isinstance(v, SortedDict):
            v = SortedDict(v)
        if not v:
            raise ValueError("expression_count cannot be empty")
        for key, value in v.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"expression_count for {key} must be a non-negative integer, got: {value}"
                )
        return v

    @model_validator(mode="after")
    def validate_matching_keys(self) -> "RNASeqExpressionPhenotype":
        """expression_count must cover exactly the same genes as expression_tpm."""
        if set(self.expression_count.keys()) != set(self.expression_tpm.keys()):
            raise ValueError(
                "expression_count must have the same keys as expression_tpm"
            )
        return self


class RNASeqExpressionExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for an RNA-seq expression experiment."""

    experiment_reference_type: str = "rnaseq_expression"
    phenotype_reference: RNASeqExpressionPhenotype


class RNASeqExpressionExperiment(Experiment, ModelStrict):
    """Experiment measuring an RNA-seq (absolute TPM) expression phenotype."""

    experiment_type: str = "rnaseq_expression"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: RNASeqExpressionPhenotype


class PseudobulkExpressionPhenotype(Phenotype, ModelStrict):
    """Pseudobulk single-cell RNA-seq expression: per-gene log2 fold-change vs WT, plus
    the per-genotype single-cell summary scalars that a bulk assay cannot provide.

    Distinct from both other expression families. ``RNASeqExpressionPhenotype`` is ABSOLUTE
    TPM on an isolate's own genome (a survey, no reference to ratio against);
    ``MicroarrayExpressionPhenotype`` is a microarray log2 ratio with a per-gene
    ``expression`` linear channel + per-gene ``n_replicates``. This family is a genome-scale
    single-cell Perturb-seq collapsed to PSEUDOBULK per genotype: each non-essential-gene
    deletion's transcriptome is compared to the WILD TYPE profiled in the SAME condition,
    yielding a per-gene log2 fold-change (Nadal-Ribelles 2025; scanpy ``logfoldchanges``,
    Wilcoxon rank-sum DE on SCTransform log-normalized counts). The WT reference is log2
    fold-change 0 for every gene (``reference_centered = True``).

    The single-cell origin is preserved by two per-genotype scalars -- the WHOLE POINT of a
    pseudobulk+dispersion (rather than per-cell) representation:

      - ``dispersion``: the genotype's transcriptional HETEROGENEITY, the standard deviation
        of the scaled SVD leverage score across the genotype's cells
        (``sd_lvscore_scaledFU2``; the leverage score is z-scored against WT cells, so WT
        dispersion ~= 1). Higher = more deviated/heterogeneous expression vs WT.
      - ``n_cells``: the number of assigned single cells the pseudobulk logFC was estimated
        from (``cell_number``); a per-genotype confidence weight.

    Ragged gene sets: each mutant-vs-WT comparison first drops genes with 0 counts, so the
    tested gene set differs per genotype. A gene not tested for a genotype is KEY-ABSENT
    (no key), NEVER stored as 0 -- honest to the source, exactly as the RNA-seq family
    handles genes absent from an isolate's genome.

    Provenance: Nadal-Ribelles et al. 2025, Nat. Commun. See
    ``[[torchcell.datasets.scerevisiae.nadal_ribelles2025]]``.
    """

    graph_level: str = "node"
    label_name: str = "expression_log2_ratio"
    label_statistic_name: str | None = "dispersion"

    expression_log2_ratio: dict[str, float] = Field(
        description=(
            "SortedDict of per-gene pseudobulk log2 fold-change vs the WT profiled in the "
            "SAME condition (log2(genotype/WT); positive = up-, negative = down-regulated). "
            "Finite. A gene not tested for this genotype (0 counts, dropped pre-DE) is "
            "omitted (no key), never stored as 0."
        ),
        repr=False,
    )
    dispersion: float | None = Field(
        default=None,
        description=(
            "per-genotype transcriptional heterogeneity: the standard deviation of the "
            "scaled (WT-z-scored) SVD leverage score across the genotype's cells "
            "(source column ``sd_lvscore_scaledFU2``; WT ~= 1). None if unavailable."
        ),
    )
    n_cells: int | None = Field(
        default=None,
        description=(
            "number of assigned single cells the pseudobulk logFC was estimated from "
            "(source column ``cell_number``); a per-genotype confidence weight. None if "
            "unavailable."
        ),
    )
    measurement_type: str = Field(
        default="pseudobulk_scrnaseq_log2fc",
        description=(
            "assay/normalization tag: single-cell RNA-seq collapsed to pseudobulk, log2 "
            "fold-change vs same-condition WT (Wilcoxon DE on SCTransform log-normalized "
            "counts)."
        ),
    )

    def __repr__(self) -> str:
        """Summary repr instead of dumping the per-gene dict."""
        return (
            f"PseudobulkExpressionPhenotype(log2_ratio_genes="
            f"{len(self.expression_log2_ratio) if self.expression_log2_ratio else 0}, "
            f"dispersion={self.dispersion}, n_cells={self.n_cells})"
        )

    @field_validator("expression_log2_ratio", mode="before")
    def convert_and_validate_log2_ratio(cls, v: Any) -> Any:  # raw pre-validation input
        """Coerce log2 ratios to a SortedDict; reject empty, infinite, or NaN values."""
        if v is None:
            raise ValueError("expression_log2_ratio cannot be None")
        if isinstance(v, dict) and not isinstance(v, SortedDict):
            v = SortedDict(v)
        if not v:
            raise ValueError("expression_log2_ratio cannot be empty")
        for key, value in v.items():
            if math.isinf(value) or math.isnan(value):
                raise ValueError(f"Invalid log2 ratio for gene {key}: {value}")
        return v

    @field_validator("dispersion", mode="after")
    def validate_dispersion(cls, v: float | None) -> float | None:
        """Dispersion, when present, is a non-negative finite float."""
        if v is not None and (math.isinf(v) or math.isnan(v) or v < 0):
            raise ValueError(f"dispersion must be a non-negative finite float, got {v}")
        return v

    @field_validator("n_cells", mode="after")
    def validate_n_cells(cls, v: int | None) -> int | None:
        """n_cells, when present, is a positive integer."""
        if v is not None and v < 1:
            raise ValueError(f"n_cells must be a positive integer, got {v}")
        return v


class PseudobulkExpressionExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a pseudobulk single-cell expression experiment."""

    experiment_reference_type: str = "pseudobulk_expression"
    phenotype_reference: PseudobulkExpressionPhenotype


class PseudobulkExpressionExperiment(Experiment, ModelStrict):
    """Experiment measuring a pseudobulk single-cell (Perturb-seq) expression phenotype."""

    experiment_type: str = "pseudobulk_expression"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: PseudobulkExpressionPhenotype


class VisualScorePhenotype(Phenotype, ModelStrict):
    """Ordinal visual-inspection score as a proxy for a metabolite/product level.

    For screens where a metabolite is read out by a qualitative visual signal rather
    than a quantitative assay. Ozaydin 2005/2013: colony COLOR on a -5..+5 scale is a
    visual proxy for carotenoid (beta-carotene) accumulation -- more orange/red colony
    = more product. The score is a SUBJECTIVE ORDINAL, not a quantitative abundance;
    downstream models must treat it ordinally.

    Metabolite linkage (Yeast9): ``target_product`` names the metabolite the score is a
    proxy for (e.g. "beta-carotene"). Heterologous products (carotenoids, betalains)
    are NOT native to Yeast9, so ``target_metabolite_id`` -- an optional Yeast9
    ``s_NNNN`` metabolite id for constraint-based-model linkage -- is left ``None``
    until the metabolic-model mapping is decided; the fields here capture the data a
    CBM would need, without committing to that mapping.
    """

    graph_level: str = "global"
    label_name: str = "visual_score"
    label_statistic_name: str | None = None

    visual_score: float = Field(
        description="aggregated ordinal visual score (e.g. colony color intensity)"
    )
    visual_score_min: float | None = Field(
        default=None,
        description="min score across replicates (reproducibility; None if 1 replicate)",
    )
    n_replicates: int = Field(
        description="number of independent visually-scored replicates for this strain"
    )
    score_scale_min: int = Field(description="lower bound of the ordinal scale")
    score_scale_max: int = Field(description="upper bound of the ordinal scale")
    score_semantics: str = Field(
        description=(
            "what higher vs lower means, e.g. "
            "'higher = more orange colony = more carotenoid/beta-carotene'"
        )
    )
    target_product: str = Field(
        description="metabolite/product the score is a visual proxy for, e.g. 'beta-carotene'"
    )
    target_metabolite_id: str | None = Field(
        default=None,
        description="optional Yeast9 s_NNNN metabolite id for CBM linkage; None until modeling decided",
    )
    score_text: str | None = Field(
        default=None,
        description="non-numeric score annotations from the source (e.g. 'pet', 'tiny')",
    )
    comment_annotations: dict[str, bool] | None = Field(
        default=None,
        description=(
            "boolean annotations parsed from the source Comment column; a MIX, NOT all "
            "QC -- true QC (flag_qc_failure, flag_het_diploid), secondary growth/"
            "physiology phenotypes (flag_petite, flag_tiny, flag_slow_growth), and "
            "interpretation caveats (flag_sterile, flag_unusual_color). Do not filter "
            "records on these as if they were all quality failures."
        ),
    )

    @model_validator(mode="after")
    def validate_visual_score(self) -> "VisualScorePhenotype":
        """Enforce a coherent ordinal scale and score/replicate bounds."""
        if self.score_scale_min >= self.score_scale_max:
            raise ValueError("score_scale_min must be < score_scale_max")
        if not (self.score_scale_min <= self.visual_score <= self.score_scale_max):
            raise ValueError(
                f"visual_score {self.visual_score} outside scale "
                f"[{self.score_scale_min}, {self.score_scale_max}]"
            )
        if self.visual_score_min is not None and not (
            self.score_scale_min <= self.visual_score_min <= self.score_scale_max
        ):
            raise ValueError("visual_score_min outside the declared scale")
        if self.n_replicates < 1:
            raise ValueError("n_replicates must be >= 1")
        return self


class VisualScoreExperimentReference(ExperimentReference, ModelStrict):
    """Reference (control colony) context for a visual-score experiment."""

    experiment_reference_type: str = "visual_score"
    phenotype_reference: VisualScorePhenotype


class VisualScoreExperiment(Experiment, ModelStrict):
    """Experiment measuring a visual-score phenotype."""

    experiment_type: str = "visual_score"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: VisualScorePhenotype


class MetabolitePhenotype(Phenotype, ModelStrict):
    """Quantitative metabolite/product level(s), keyed by metabolite id.

    For assays that QUANTIFY one or more metabolite levels per strain -- e.g. Cachera
    2023 CRI-SPA (corrected colony HSV yellowness score as a proxy for the metabolite
    betaxanthin), or (later) mass-spec metabolite abundances (Zelezniak). Levels are
    keyed by metabolite id: a Yeast9 ``s_NNNN`` id where the metabolite is native, or a
    plain product name for heterologous products (carotenoids, betalains) not in Yeast9.

    ``measurement_type`` records WHAT the number is (e.g. a normalized colony-color
    score, which can be negative, vs an absolute abundance), so heterogeneous assays
    stay interpretable and are never silently compared. ``target_metabolite_ids`` maps
    the keys to Yeast9 ``s_NNNN`` ids for constraint-based-model linkage where known
    (``None`` until the mapping is decided -- capture the data, defer the modeling).
    """

    graph_level: str = "metabolism"
    label_name: str = "metabolite_level"
    label_statistic_name: str | None = "metabolite_level_se"

    metabolite_level: dict[str, float] = Field(
        description="metabolite_id -> measured level (Yeast9 s_NNNN id, or product name)"
    )
    metabolite_level_se: dict[str, float] | None = Field(
        default=None, description="metabolite_id -> standard error of the level"
    )
    n_replicates: dict[str, int] = Field(
        description="metabolite_id -> number of independent replicates"
    )
    measurement_type: str = Field(
        description=(
            "what the level number is, e.g. "
            "'cri_spa_corrected_hsv_yellowness_24h' or 'ms_abundance'"
        )
    )
    target_metabolite_ids: dict[str, str] | None = Field(
        default=None,
        description="metabolite key -> Yeast9 s_NNNN id for CBM linkage; None until decided",
    )

    @model_validator(mode="after")
    def validate_metabolite_level(self) -> "MetabolitePhenotype":
        """Require non-empty levels and consistent per-metabolite replicate keys."""
        if not self.metabolite_level:
            raise ValueError("metabolite_level cannot be empty")
        if set(self.n_replicates) != set(self.metabolite_level):
            raise ValueError("n_replicates keys must match metabolite_level keys")
        for key, n in self.n_replicates.items():
            if n < 1:
                raise ValueError(f"n_replicates for {key} must be >= 1")
        if self.metabolite_level_se is not None:
            for key, se in self.metabolite_level_se.items():
                if key not in self.metabolite_level:
                    raise ValueError(f"SE key {key} not in metabolite_level")
                if not math.isnan(se) and se < 0:
                    raise ValueError(f"SE for {key} must be non-negative")
        return self


class MetaboliteExperimentReference(ExperimentReference, ModelStrict):
    """Reference (control) context for a metabolite experiment."""

    experiment_reference_type: str = "metabolite"
    phenotype_reference: MetabolitePhenotype


class MetaboliteExperiment(Experiment, ModelStrict):
    """Experiment measuring a metabolite phenotype."""

    experiment_type: str = "metabolite"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: MetabolitePhenotype


class ProteinAbundancePhenotype(Phenotype, ModelStrict):
    """Quantitative protein-abundance profile keyed by protein (systematic ORF).

    For bottom-up proteomics that quantify per-protein abundance per strain -- e.g.
    Zelezniak 2018 SWATH-MS of kinase-knockout strains (batch-corrected, SVA-adjusted
    label-free signal). ``protein_abundance`` maps each measured protein's systematic
    ORF id to its abundance (absolute per-strain quantity on a log signal scale, NOT a
    ratio -- the WT/parent strain supplies the reference). ``measurement_type`` records
    WHAT the number is so heterogeneous proteomics assays are never silently mixed.
    """

    graph_level: str = "node"
    label_name: str = "protein_abundance"
    label_statistic_name: str | None = "protein_abundance_se"

    protein_abundance: dict[str, float] = Field(
        description="protein systematic ORF -> abundance (batch-corrected label-free signal)"
    )
    protein_abundance_se: dict[str, float] | None = Field(
        default=None, description="protein ORF -> standard error across replicates"
    )
    n_replicates: dict[str, int] = Field(
        description="protein ORF -> number of independent samples/replicates"
    )
    measurement_type: str = Field(
        description="what the level is, e.g. 'swath_ms_label_free_log_signal_sva'"
    )

    @model_validator(mode="after")
    def validate_protein_abundance(self) -> "ProteinAbundancePhenotype":
        """Require non-empty abundances and consistent per-protein replicate keys."""
        if not self.protein_abundance:
            raise ValueError("protein_abundance cannot be empty")
        if set(self.n_replicates) != set(self.protein_abundance):
            raise ValueError("n_replicates keys must match protein_abundance keys")
        for key, n in self.n_replicates.items():
            if n < 1:
                raise ValueError(f"n_replicates for {key} must be >= 1")
        if self.protein_abundance_se is not None:
            for key, se in self.protein_abundance_se.items():
                if key not in self.protein_abundance:
                    raise ValueError(f"SE key {key} not in protein_abundance")
                if not math.isnan(se) and se < 0:
                    raise ValueError(f"SE for {key} must be non-negative")
        return self


class ProteinAbundanceExperimentReference(ExperimentReference, ModelStrict):
    """Reference (control) context for a protein-abundance experiment."""

    experiment_reference_type: str = "protein_abundance"
    phenotype_reference: ProteinAbundancePhenotype


class ProteinAbundanceExperiment(Experiment, ModelStrict):
    """Experiment measuring a protein-abundance phenotype."""

    experiment_type: str = "protein_abundance"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: ProteinAbundancePhenotype


class AssayType(StrEnum):
    """HOW an environment-response readout was physically measured (experimental design).

    Orthogonal to ``MeasurementType`` (WHAT the number is): a pooled Bar-seq assay can
    yield a ``log2_ratio`` OR a ``z_score``, so the two axes cross. The free-text
    ``units`` string is NOT a substitute for this typed method axis.

    - ``pooled_competitive_growth_barcode``: pooled competitive growth read out by
      molecular barcodes (HIP/HOP, Bar-seq).
    - ``colony_size_array``: pinned colony-size array (SGA / condition-SGA).
    - ``spot_dilution``: serial-dilution spot growth on plates.
    - ``halo_zone``: halo / zone-of-inhibition (disk-diffusion) assay.
    - ``liquid_od_growth``: liquid-culture optical-density growth curve / MIC.
    - ``biosensor_readout``: a biosensor / reporter signal (e.g. fluorescence).
    - ``other``: a measured assay not covered above (record detail in ``units``).
    """

    pooled_competitive_growth_barcode = "pooled_competitive_growth_barcode"
    colony_size_array = "colony_size_array"
    spot_dilution = "spot_dilution"
    halo_zone = "halo_zone"
    liquid_od_growth = "liquid_od_growth"
    biosensor_readout = "biosensor_readout"
    other = "other"


class MeasurementType(StrEnum):
    """What an environment-response readout number IS, so heterogeneous chemogenomic
    scores are never silently compared.

    - ``log2_ratio``: log2(treatment/control) barcode abundance / fitness ratio
      (Vanacloig).
    - ``z_score``: standardized fitness/growth deviation (Wildenhain; Hoepfner z-score
      columns).
    - ``sensitivity_score``: HIP/HOP sensitivity / fitness-defect score (Hoepfner MADL,
      Hillenmeyer, Lee).
    - ``categorical``: NOMINAL qualitative call with no order among its terms
      (Auesukaree sensitive/tolerant; Mormino enhanced-signal/no-effect).
    - ``ordinal``: a RANKED grade on a source-defined scale, where the ranks are ordered
      but the spacing between them carries no meaning (Smith 2006 scores the clear zone
      4/3/2/1 = larger than wild type / wild type / less than wild type / small or not
      detectable, and growth on acetate 3/2.5/2/1). Distinct from ``categorical`` because
      the order is real (a 2 lies between a 1 and a 3) and distinct from every numeric
      member because the number is a rank, not a measured quantity: averaging it or
      comparing it to a z-score is meaningless. Like ``categorical``, it requires
      ``category``; unlike it, ``environment_response`` may carry the rank itself.
    - ``growth_rate``: absolute or normalized growth rate / doubling time.
    - ``differential_fitness``: SIGNED difference of normalized colony-size fitness in a
      test condition minus the matched reference condition (Costanzo 2021 condition-SGA:
      "the difference in colony size measured in a particular test condition versus the
      matched reference condition for each mutant"); negative = condition-hypersensitive,
      0 = fitness unchanged vs reference. Distinct from ``growth_rate`` (a rate, which is
      non-negative) -- a differential is routinely negative.
    - ``control_regression_residual``: SIGNED residual of a strain's colony size in a test
      condition regressed on the SAME strain's colony size on a matched control plate
      (Bloom 2019 segregant panels: ``residuals(lm(s.radius.mean ~ ctrl.s.radius.mean))``
      against the same-cross, same-batch YPD or YNB plate); 0 = grew exactly as predicted
      from control growth. Distinct from ``differential_fitness`` (a plain subtraction of
      two normalized fitnesses): a regression residual removes the control-plate slope, so
      the two are not comparable numbers.
    - ``colony_size``: ABSOLUTE end-point colony size (mean radius in image pixels for the
      Bloom 2019 control plates); non-negative, unnormalized, scale is the assay's own.
    """

    log2_ratio = "log2_ratio"
    z_score = "z_score"
    sensitivity_score = "sensitivity_score"
    categorical = "categorical"
    ordinal = "ordinal"
    growth_rate = "growth_rate"
    differential_fitness = "differential_fitness"
    control_regression_residual = "control_regression_residual"
    colony_size = "colony_size"


class ResponseCategory(StrEnum):
    """The controlled vocabulary a qualitative environment-response call resolves to.

    Every categorical screen invents its own words for the same small set of outcomes
    ("sensitive", "tolerant", "++", "wild type", "defective"), so a free-text category
    joins nothing: two screens reporting the same biology land in different buckets and a
    single screen's vocabulary is unknowable without reading its loader. This enum is the
    shared axis; the source's own word is kept verbatim in
    ``EnvironmentResponsePhenotype.category_label``, so nothing is lost by mapping.

    The axis is the strain's growth/readout RELATIVE TO the record's declared reference,
    in the perturbed environment. Five members form an ORDER (``enhanced`` >
    ``no_change`` > ``mildly_reduced`` > ``reduced`` > ``severely_reduced``); the
    remaining three are deliberately unordered, because a source that reports only a
    binary hit call has not measured a severity and a mapping must not invent one.

    - ``enhanced``: measurably better than the reference (Smith 2006 clear-zone 4,
      "larger than wild type"; a biosensor signal above control).
    - ``no_change``: indistinguishable from the reference (Smith 2006 score 3 "wild
      type"; Mota 2024's unlisted strains, "no detectable susceptibility").
    - ``mildly_reduced``: a slight deficit short of a clear one (Smith 2006 growth 2.5,
      the released table's undocumented intermediate grade).
    - ``reduced``: a clear but partial deficit (Smith 2006 score 2, "less than wild type"
      / "moderate"; Mota 2024 ``+``, minor-to-moderate growth inhibition).
    - ``severely_reduced``: growth or signal essentially abolished (Smith 2006 score 1,
      "small or not detectable" / "little/no growth"; Mota 2024 ``++``, total growth
      inhibition).
    - ``sensitive``: an UNGRADED hit call under stress, where the source reports only
      that the strain is affected (Auesukaree 2009's listed stress-sensitive mutants).
      Not a synonym for ``reduced``: mapping it there would assert a severity the source
      never scored.
    - ``resistant``: the ungraded opposite, a strain called tolerant of a stress that
      affects the reference.
    - ``not_determined``: the strain was screened but no call could be made (a failed or
      excluded well), which is distinct both from ``no_change`` and from absence.
    """

    enhanced = "enhanced"
    no_change = "no_change"
    mildly_reduced = "mildly_reduced"
    reduced = "reduced"
    severely_reduced = "severely_reduced"
    sensitive = "sensitive"
    resistant = "resistant"
    not_determined = "not_determined"


#: Measurement types whose readout is a CALL rather than a quantity, so a ``category`` is
#: required and a numeric ``environment_response`` is optional.
CATEGORICAL_MEASUREMENT_TYPES: frozenset[MeasurementType] = frozenset(
    {MeasurementType.categorical, MeasurementType.ordinal}
)


class EnvironmentResponsePhenotype(Phenotype, ModelStrict):
    """A strain's fitness/growth RESPONSE to an environmental perturbation.

    For chemical-genomic / stress screens where a (usually deletion) strain's fitness
    in a perturbed environment is scored relative to a control. Distinct from
    ``FitnessPhenotype`` -- a strictly positive ko/wt growth-rate RATIO that clamps
    non-positive values to 0 -- because this readout is a SIGNED score (log2 ratio,
    z-score, sensitivity score) that is routinely NEGATIVE, or a qualitative
    ``category``. ``measurement_type`` records WHAT the number is, ``assay_type`` records
    HOW it was measured (the experimental design), and ``units`` gives its human-readable
    definition -- ``units`` is NOT a substitute for the typed ``assay_type`` axis. The
    uncertainty ontology mirrors ``FitnessPhenotype``:
    ``environment_response_se`` is the DERIVED, ML-facing SE (auto-filled from the
    source-reported uncertainty + its type via ``derive_se``).

    A qualitative call is TYPED: ``category`` is a ``ResponseCategory`` on the shared
    cross-screen axis and ``category_label`` holds the source's own word for it, so a
    screen's private vocabulary stays readable without becoming the join key.
    ``screen_id`` names the screening run a measurement came from, which is what keeps
    two independent screens of the same compound at the same dose from collapsing into
    one record.
    """

    graph_level: str = "global"
    label_name: str = "environment_response"
    label_statistic_name: str | None = "environment_response_se"

    measurement_type: MeasurementType = Field(
        description="what the response number is (log2_ratio, z_score, ...)"
    )
    assay_type: AssayType | None = Field(
        default=None,
        description="HOW the response was measured (experimental design); orthogonal to "
        "measurement_type (WHAT the number is). Free-text units is NOT a substitute for "
        "this typed method axis. None until sourced; a genuine absence is a ProvenanceGap "
        "on 'assay_type'.",
    )
    environment_response: float | None = Field(
        default=None,
        description="signed numeric score (log2 ratio, z-score, sensitivity score, "
        "growth rate); None only for a purely categorical readout",
    )
    category: ResponseCategory | None = Field(
        default=None,
        description="the qualitative call on the shared ResponseCategory axis; None for "
        "purely numeric readouts",
    )
    category_label: str | None = Field(
        default=None,
        description="the source's own word or symbol for that call, verbatim (e.g. "
        "'++', 'tolerant', 'wild type', the ordinal '4'), so the mapping onto "
        "ResponseCategory is auditable and nothing the source said is lost. Requires "
        "`category`: a label with no typed call is the free-text state this axis "
        "replaces.",
    )
    environment_response_se: float | None = Field(
        default=None,
        description="standard error of the response (primary uncertainty statistic)",
    )
    environment_response_uncertainty: float | None = Field(
        default=None,
        description="source-reported uncertainty number, verbatim (meaning given by "
        "environment_response_uncertainty_type)",
    )
    environment_response_uncertainty_type: UncertaintyType | None = Field(
        default=None,
        description="what environment_response_uncertainty IS (sample_sd, ...)",
    )
    n_samples: int | None = Field(
        default=None,
        description="number of independent replicate measurements of the response",
    )
    sample_unit: SampleUnit | None = Field(
        default=None,
        description="what one sample in n_samples is (biological_replicate, ...)",
    )
    units: str | None = Field(
        default=None,
        description="human-readable definition/units of the score, e.g. "
        "'log2(inhibitor/control)'",
    )
    screen_id: str | None = Field(
        default=None,
        description="the source's own identifier for the SCREEN this measurement came "
        "from (Hoepfner 2014's internal study number, a plate/round id). A compound at "
        "one dose can be screened more than once, and once the compound name is cleaned "
        "those runs become indistinguishable: Hoepfner has 45 columns that collide on "
        "(compound, dose) alone, so the screen id is what keeps one strain x one "
        "condition L1-unique instead of silently merging independent measurements.",
    )

    @field_validator("environment_response")
    def validate_response(cls, v: float | None) -> float | None:
        """Reject NaN numeric responses (None is allowed for categorical readouts)."""
        if v is not None and math.isnan(v):
            raise ValueError("environment_response cannot be NaN")
        return v

    @field_validator("n_samples")
    def validate_n_samples(cls, v: int | None) -> int | None:
        """n_samples is a positive integer or None."""
        if v is not None and (not isinstance(v, int) or v < 1):
            raise ValueError(f"n_samples must be a positive integer or None, got: {v}")
        return v

    @model_validator(mode="before")
    @classmethod
    def _fill_response_se(cls, data: Any) -> Any:
        """Derive the ML-facing SE from the reported uncertainty (frozen -> fill first)."""
        if not isinstance(data, dict):
            return data
        unc = data.get("environment_response_uncertainty")
        typ = data.get("environment_response_uncertainty_type")
        if (
            unc is None
            or typ is None
            or data.get("environment_response_se") is not None
        ):
            return data
        typ = UncertaintyType(typ)
        n = data.get("n_samples")
        if typ in (UncertaintyType.sample_sd, UncertaintyType.variance) and n is None:
            return data
        data["environment_response_se"] = derive_se(unc, typ, n)
        return data

    @model_validator(mode="after")
    def _check(self) -> "EnvironmentResponsePhenotype":
        """Enforce numeric-vs-categorical coherence + the uncertainty invariant."""
        if self.measurement_type in CATEGORICAL_MEASUREMENT_TYPES:
            if self.category is None:
                raise ValueError(
                    f"{self.measurement_type} measurement_type requires `category`"
                )
        elif self.environment_response is None:
            raise ValueError(
                f"{self.measurement_type} requires a numeric environment_response"
            )
        if self.category_label is not None and self.category is None:
            raise ValueError(
                "category_label requires `category` (a verbatim source label with no "
                "typed call is the free-text state ResponseCategory replaces)"
            )
        unc, typ = (
            self.environment_response_uncertainty,
            self.environment_response_uncertainty_type,
        )
        if (unc is None) != (typ is None):
            raise ValueError(
                "environment_response_uncertainty and its type must both be set or "
                "both be None (no unlabelled uncertainty)"
            )
        if typ in (UncertaintyType.sample_sd, UncertaintyType.variance) and (
            self.n_samples is None or self.sample_unit is None
        ):
            raise ValueError(f"n_samples and sample_unit are required for {typ}")
        return self


class EnvironmentResponseExperimentReference(ExperimentReference, ModelStrict):
    """Reference (control) context for an environment-response experiment."""

    experiment_reference_type: str = "environment_response"
    phenotype_reference: EnvironmentResponsePhenotype


class EnvironmentResponseExperiment(Experiment, ModelStrict):
    """Experiment measuring a strain's response to an environmental perturbation."""

    experiment_type: str = "environment_response"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: EnvironmentResponsePhenotype


# --------------------------------------------------------------------------- #
# Strain-resolved environment response (#507): the chemogenomic family whose
# reference states a typed StrainBackground and whose environment states its
# culture protocol. Its own experiment_type, so the reconstruction maps resolve it
# to the classes that declare StrainReferenceGenome / CultureEnvironment
# explicitly (pydantic v2 serializes a field by its DECLARED type, so a subclass
# instance in a base-typed slot would lose its fields). The base family and every
# class it is built from are untouched, so no other served dataset's schema
# closure moves. The phenotype is the same EnvironmentResponsePhenotype.
# --------------------------------------------------------------------------- #
class StrainEnvironmentResponseExperimentReference(
    EnvironmentResponseExperimentReference, ModelStrict
):
    """Reference for a strain-resolved environment response (typed background).

    ``genome_reference`` is a ``StrainReferenceGenome`` (mating type + every background
    allele, sourced or gapped); ``environment_reference`` is a ``CultureEnvironment``.
    """

    experiment_reference_type: str = "strain_environment_response"
    # Narrowed to subclasses: this family states its background and culture protocol.
    genome_reference: StrainReferenceGenome
    environment_reference: CultureEnvironment
    phenotype_reference: EnvironmentResponsePhenotype


class StrainEnvironmentResponseExperiment(EnvironmentResponseExperiment, ModelStrict):
    """A strain's response to an environmental perturbation, with a typed background.

    The chemogenomic loaders (Vanacloig, Wildenhain, Hillenmeyer, Hoepfner) emit this
    family: the screened edit stays in ``genotype`` (a typed deletion leaf, so pooled
    one-perturbation semantics hold), the strain background rides on the reference's
    ``StrainReferenceGenome``, and the culture protocol on a ``CultureEnvironment``.
    """

    experiment_type: str = "strain_environment_response"
    environment: CultureEnvironment  # narrowed: this family states its culture protocol
    # Redeclared (same type): torchcell.data reads ``__annotations__["phenotype"]``,
    # which holds only a class's OWN annotations.
    phenotype: EnvironmentResponsePhenotype


# --------------------------------------------------------------------------- #
# Segregant (meiotic recombinant) genotypes -- a haplotype MOSAIC, not a gene edit.
#
# A segregant of a biparental cross carries no engineered perturbation and is not
# individually sequenced to a verified sequence, so neither ``Genotype`` (a list of
# gene-keyed ``GenePerturbation``s, regex-validated on ``systematic_gene_name``) nor
# ``SequenceVariantPerturbation`` (which promises a dereferenceable sequence) can
# hold it. The honest record is the strain-level mosaic settled in
# ``[[torchcell.datamodels.eqtl-data-model]]``: ``{(chr, start, end, parent, p)}``
# against two sha256-pinned parent assemblies. ``SegregantGenotype`` is therefore a
# SIBLING of ``Genotype`` (never a subclass): a ``Genotype`` subclass with an empty
# ``perturbations`` list would read as wild-type S288C to every gene-keyed consumer,
# encoding an inference as an observation. The sibling makes gene-keyed consumers
# fail loudly instead. ``Genotype`` itself is untouched, and the new names appear
# only in their own bodies and the module-level union assignments below, so no
# served dataset's schema contract moves (``torchcell/provenance/schema_deps.py``).
# --------------------------------------------------------------------------- #
class HaplotypeBlock(ModelStrict):
    """One run of consecutive markers called to the same parent on one chromosome.

    ``start``/``end`` are the reference-coordinate positions of the FIRST and LAST
    marker of the run (as the marker names carry them); the crossover lies somewhere
    in the unassigned gap between two adjacent blocks and is deliberately not
    resolved. ``posterior`` is the assignment probability in [0, 1] (1.0 for a
    released hard call); ``n_markers`` is the number of markers the run spans.
    """

    chromosome: str
    start: int
    end: int
    parent: Literal[1, 2]
    posterior: float = 1.0
    n_markers: int

    @model_validator(mode="after")
    def _check(self) -> "HaplotypeBlock":
        """Positions are ordered, the posterior is a probability, the run is non-empty."""
        if self.start < 1 or self.end < self.start:
            raise ValueError(
                f"HaplotypeBlock needs 1 <= start <= end, got {self.start}..{self.end}"
            )
        if not 0.0 <= self.posterior <= 1.0:
            raise ValueError(f"posterior must be in [0, 1], got {self.posterior}")
        if self.n_markers < 1:
            raise ValueError(f"n_markers must be >= 1, got {self.n_markers}")
        return self


class SegregantParent(ModelStrict):
    """One parent of a biparental cross, pinned to a sha256-anchored assembly.

    ``name`` is the source's label verbatim (e.g. ``BYa``, ``RMx``); ``peter_strain_id``
    is the 1011-collection (Peter 2018) strain id when the parent is one of those
    isolates (None for the S288C-derived BY); ``assembly_member`` names the assembly
    file (a member path inside the pinned 1011 assemblies tarball, or the S288C
    reference) and ``assembly_sha256`` the pinned container; ``engineered_background``
    is the parent's marker/deletion genotype as the source states it, verbatim.
    """

    name: str
    peter_strain_id: str | None = None
    assembly_member: str
    assembly_sha256: str
    engineered_background: str


class SegregantGenotype(ModelStrict):
    """Strain-level haplotype mosaic of one haploid segregant from a two-parent cross.

    ``blocks`` partition the called markers of every chromosome into parent-assigned
    runs (see ``HaplotypeBlock``); the per-marker call matrix is a derived view
    (re-expand each block at the cross's marker positions). ``call_method`` records
    how the released calls were produced (quoted from the source's code), and
    ``marker_matrix_sha256`` pins the released matrix the blocks were encoded from.
    """

    cross: str
    segregant_id: str
    parent_1: SegregantParent
    parent_2: SegregantParent
    blocks: list[HaplotypeBlock]
    call_method: str
    marker_matrix_sha256: str

    @field_validator("blocks", mode="after")
    @classmethod
    def _check_blocks(cls, blocks: list[HaplotypeBlock]) -> list[HaplotypeBlock]:
        """Per chromosome: blocks are in ascending order, non-overlapping, and alternate
        parents (two adjacent runs of the same parent would be one run).
        """
        if not blocks:
            raise ValueError("SegregantGenotype needs at least one HaplotypeBlock")
        by_chrom: dict[str, list[HaplotypeBlock]] = {}
        for block in blocks:
            by_chrom.setdefault(block.chromosome, []).append(block)
        for chrom, runs in by_chrom.items():
            for prev, cur in zip(runs, runs[1:]):
                if cur.start <= prev.end:
                    raise ValueError(
                        f"{chrom}: blocks overlap or are unordered at {prev.end} -> {cur.start}"
                    )
                if cur.parent == prev.parent:
                    raise ValueError(
                        f"{chrom}: adjacent blocks share parent {cur.parent}; merge them"
                    )
        return blocks

    @property
    def n_blocks(self) -> int:
        """Number of haplotype blocks across all chromosomes."""
        return len(self.blocks)


class SegregantGrowthExperimentReference(ExperimentReference, ModelStrict):
    """Reference (control) context for a segregant growth experiment."""

    experiment_reference_type: str = "segregant_growth"
    phenotype_reference: EnvironmentResponsePhenotype


class SegregantGrowthExperiment(Experiment, ModelStrict):
    """Growth of a haploid segregant (haplotype-mosaic genotype) in one environment."""

    experiment_type: str = "segregant_growth"
    genotype: SegregantGenotype  # type: ignore[assignment]  # sibling of Genotype, deliberately narrowed
    phenotype: EnvironmentResponsePhenotype


PhenotypeType = (
    Phenotype
    | FitnessPhenotype
    | GeneInteractionPhenotype
    | GeneEssentialityPhenotype
    | SyntheticLethalityPhenotype
    | SyntheticRescuePhenotype
    | CalMorphPhenotype
    | MicroarrayExpressionPhenotype
    | RNASeqExpressionPhenotype
    | PseudobulkExpressionPhenotype
    | VisualScorePhenotype
    | MetabolitePhenotype
    | ProteinAbundancePhenotype
    | EnvironmentResponsePhenotype
)

ExperimentType = (
    Experiment
    | FitnessExperiment
    | GeneInteractionExperiment
    | GeneEssentialityExperiment
    | SyntheticLethalityExperiment
    | SyntheticRescueExperiment
    | CalMorphExperiment
    | MicroarrayExpressionExperiment
    | RNASeqExpressionExperiment
    | PseudobulkExpressionExperiment
    | VisualScoreExperiment
    | MetaboliteExperiment
    | ProteinAbundanceExperiment
    | EnvironmentResponseExperiment
    | StrainEnvironmentResponseExperiment
    | SegregantGrowthExperiment
)

ExperimentReferenceType = (
    ExperimentReference
    | FitnessExperimentReference
    | GeneInteractionExperimentReference
    | GeneEssentialityExperimentReference
    | SyntheticLethalityExperimentReference
    | SyntheticRescueExperimentReference
    | CalMorphExperimentReference
    | MicroarrayExpressionExperimentReference
    | RNASeqExpressionExperimentReference
    | PseudobulkExpressionExperimentReference
    | VisualScoreExperimentReference
    | MetaboliteExperimentReference
    | ProteinAbundanceExperimentReference
    | EnvironmentResponseExperimentReference
    | StrainEnvironmentResponseExperimentReference
    | SegregantGrowthExperimentReference
)


EXPERIMENT_TYPE_MAP = {
    "fitness": FitnessExperiment,
    "gene interaction": GeneInteractionExperiment,
    "gene essentiality": GeneEssentialityExperiment,
    "synthetic lethality": SyntheticLethalityExperiment,
    "synthetic rescue": SyntheticRescueExperiment,
    "calmorph": CalMorphExperiment,
    "microarray_expression": MicroarrayExpressionExperiment,
    "rnaseq_expression": RNASeqExpressionExperiment,
    "pseudobulk_expression": PseudobulkExpressionExperiment,
    "visual_score": VisualScoreExperiment,
    "metabolite": MetaboliteExperiment,
    "protein_abundance": ProteinAbundanceExperiment,
    "environment_response": EnvironmentResponseExperiment,
    "strain_environment_response": StrainEnvironmentResponseExperiment,
    "segregant_growth": SegregantGrowthExperiment,
}


def environment_class_for(experiment_type: str) -> type[Environment]:
    """The ``Environment`` class the experiment class of ``experiment_type`` declares.

    Every reader that rebuilds an environment from stored JSON (the aggregation key,
    the single-pass raw stage, a flatten script) must construct the class the record's
    experiment family declares, not the base ``Environment``: the base class forbids
    extra fields, so a ``CultureEnvironment`` payload (the strain-resolved chemogenomic
    family, #507) is rejected by it, and a looser base would drop the protocol fields
    that are part of the environment's identity. Raises on an experiment class whose
    annotation is not an ``Environment`` subclass rather than guessing.
    """
    annotation = get_type_hints(EXPERIMENT_TYPE_MAP[experiment_type])["environment"]
    if not (isinstance(annotation, type) and issubclass(annotation, Environment)):
        raise TypeError(
            f"{experiment_type!r} declares environment: {annotation!r}, "
            "not an Environment subclass"
        )
    return annotation


EXPERIMENT_REFERENCE_TYPE_MAP = {
    "fitness": FitnessExperimentReference,
    "gene interaction": GeneInteractionExperimentReference,
    "gene essentiality": GeneEssentialityExperimentReference,
    "synthetic lethality": SyntheticLethalityExperimentReference,
    "synthetic rescue": SyntheticRescueExperimentReference,
    "calmorph": CalMorphExperimentReference,
    "microarray_expression": MicroarrayExpressionExperimentReference,
    "rnaseq_expression": RNASeqExpressionExperimentReference,
    "pseudobulk_expression": PseudobulkExpressionExperimentReference,
    "visual_score": VisualScoreExperimentReference,
    "metabolite": MetaboliteExperimentReference,
    "protein_abundance": ProteinAbundanceExperimentReference,
    "environment_response": EnvironmentResponseExperimentReference,
    "strain_environment_response": StrainEnvironmentResponseExperimentReference,
    "segregant_growth": SegregantGrowthExperimentReference,
}


if __name__ == "__main__":
    pass
