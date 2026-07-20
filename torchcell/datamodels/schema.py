# torchcell/datamodels/schema
# [[torchcell.datamodels.schema]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/schema
# Test file: tests/torchcell/datamodels/test_schema.py

"""Pydantic data models for torchcell genotypes, environments, and phenotypes."""

import math
import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from sortedcontainers import SortedDict

from torchcell.datamodels.calmorph_labels import CALMORPH_LABELS, CALMORPH_STATISTICS
from torchcell.datamodels.pydant import ModelStrict
from torchcell.verification.sourced import ProvenanceGap, SourcedValue

# causes circular import
# from torchcell.datasets.dataset_registry import dataset_registry


# Genotype
class ReferenceGenome(ModelStrict):
    """Reference genome identified by species and strain.

    ``ploidy`` is the genome-wide baseline copy number of the unperturbed strain
    (``"haploid"`` = 1 autosomal copy, ``"diploid"`` = 2). It lives here -- not on a
    perturbation -- because it applies to the WHOLE genome (WT and every unperturbed
    gene alike); a per-locus ``EngineeredCopyNumberPerturbation.copy_number`` records
    the deviation from this baseline (e.g. a HIP heterozygous deletion drops one
    autosomal gene from 2 -> 1 copies in a diploid). Defaults to ``"haploid"`` so all
    existing haploid datasets stay valid.
    """

    species: str
    strain: str
    ploidy: Literal["haploid", "diploid"] = "haploid"


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


class Compound(ModelStrict):
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
    ``Compound``); ``name`` remains the plain label for the common case.
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


class EnvironmentPerturbation(ModelStrict):
    """Base: a defined change to the growth environment vs the base medium."""

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


EnvironmentPerturbationType = (
    SmallMoleculePerturbation | EnvironmentPhysicalPerturbation
)


class ProvenanceGapMixin(ModelStrict):
    """Mixin giving a model a ``provenance_gaps`` list of typed field-absences.

    Shared by ``Phenotype`` and ``Environment`` so a field the source does not carry
    -- a phenotype ``n_samples`` a secondary curation layer dropped, an environment
    ``temperature`` YeastPhenome never recorded -- is a documented, typed ABSENCE (with
    a reason + ``looked_in``) rather than a guess or a silent None. Two invariants keep
    it honest and machine-checkable: (1) each ``gap.field`` names a real field on the
    concrete model (checked against ``model_fields``, so inherited fields resolve too);
    (2) a gapped field must be ``None`` -- you cannot both store a value and declare it
    missing. ``provenance_gaps`` itself cannot be gapped.
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
    2023 CRI-SPA (corrected colony fluorescence intensity as a proxy for the metabolite
    betaxanthin), or (later) mass-spec metabolite abundances (Zelezniak). Levels are
    keyed by metabolite id: a Yeast9 ``s_NNNN`` id where the metabolite is native, or a
    plain product name for heterologous products (carotenoids, betalains) not in Yeast9.

    ``measurement_type`` records WHAT the number is (e.g. a normalized fluorescence
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
            "'cri_spa_corrected_fluorescence_intensity' or 'ms_abundance'"
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


class MeasurementType(StrEnum):
    """What an environment-response readout number IS, so heterogeneous chemogenomic
    scores are never silently compared.

    - ``log2_ratio``: log2(treatment/control) barcode abundance / fitness ratio
      (Vanacloig).
    - ``z_score``: standardized fitness/growth deviation (Wildenhain; Hoepfner z-score
      columns).
    - ``sensitivity_score``: HIP/HOP sensitivity / fitness-defect score (Hoepfner MADL,
      Hillenmeyer, Lee).
    - ``categorical``: qualitative call (Auesukaree sensitive/tolerant; Mota 0/+/++).
    - ``growth_rate``: absolute or normalized growth rate / doubling time.
    - ``differential_fitness``: SIGNED difference of normalized colony-size fitness in a
      test condition minus the matched reference condition (Costanzo 2021 condition-SGA:
      "the difference in colony size measured in a particular test condition versus the
      matched reference condition for each mutant"); negative = condition-hypersensitive,
      0 = fitness unchanged vs reference. Distinct from ``growth_rate`` (a rate, which is
      non-negative) -- a differential is routinely negative.
    """

    log2_ratio = "log2_ratio"
    z_score = "z_score"
    sensitivity_score = "sensitivity_score"
    categorical = "categorical"
    growth_rate = "growth_rate"
    differential_fitness = "differential_fitness"


class EnvironmentResponsePhenotype(Phenotype, ModelStrict):
    """A strain's fitness/growth RESPONSE to an environmental perturbation.

    For chemical-genomic / stress screens where a (usually deletion) strain's fitness
    in a perturbed environment is scored relative to a control. Distinct from
    ``FitnessPhenotype`` -- a strictly positive ko/wt growth-rate RATIO that clamps
    non-positive values to 0 -- because this readout is a SIGNED score (log2 ratio,
    z-score, sensitivity score) that is routinely NEGATIVE, or a qualitative
    ``category``. ``measurement_type`` records WHAT the number is; ``units`` gives its
    human-readable definition. The uncertainty ontology mirrors ``FitnessPhenotype``:
    ``environment_response_se`` is the DERIVED, ML-facing SE (auto-filled from the
    source-reported uncertainty + its type via ``derive_se``).
    """

    graph_level: str = "global"
    label_name: str = "environment_response"
    label_statistic_name: str | None = "environment_response_se"

    measurement_type: MeasurementType = Field(
        description="what the response number is (log2_ratio, z_score, ...)"
    )
    environment_response: float | None = Field(
        default=None,
        description="signed numeric score (log2 ratio, z-score, sensitivity score, "
        "growth rate); None only for a purely categorical readout",
    )
    category: str | None = Field(
        default=None,
        description="qualitative call for categorical readouts, e.g. 'sensitive' | "
        "'tolerant' | 'no_effect'; None for numeric readouts",
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
        if self.measurement_type is MeasurementType.categorical:
            if self.category is None:
                raise ValueError("categorical measurement_type requires `category`")
        elif self.environment_response is None:
            raise ValueError(
                f"{self.measurement_type} requires a numeric environment_response"
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
}

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
}


if __name__ == "__main__":
    pass
