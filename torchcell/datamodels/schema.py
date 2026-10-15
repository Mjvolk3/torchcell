# torchcell/datamodels/schema
# [[torchcell.datamodels.schema]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/schema
# Test file: tests/torchcell/datamodels/test_schema.py

"""Pydantic data models for torchcell genotypes, environments, and phenotypes."""

import math
import posixpath
import re
from collections import Counter
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Final, Literal, Self, get_args

from pydantic import BaseModel, Field, field_validator, model_validator
from sortedcontainers import SortedDict

from torchcell.datamodels.bacterial_morphology_features import (
    MorphologyAssay,
    morphology_assay,
)
from torchcell.datamodels.calmorph_labels import CALMORPH_LABELS, CALMORPH_STATISTICS

# --- begin ADDITIVE (Puddu 2019 genome architecture): the WGS architecture family ---
from torchcell.datamodels.genome_architecture_features import (
    ArchitectureAssay,
    ArchitectureBasis,
    architecture_assay,
)

# --- end ADDITIVE (Puddu 2019 genome architecture) ---
from torchcell.datamodels.pydant import ModelStrict
from torchcell.verification.sourced import ProvenanceGap, SourcedValue

# causes circular import
# from torchcell.datasets.dataset_registry import dataset_registry


# The artifact tiers an ``ArtifactRef`` can point into (see ``torchcell.artifacts``).
ArtifactTier = Literal["raw", "genomes", "library", "objects"]
ARTIFACT_TIERS: tuple[str, ...] = get_args(ArtifactTier)
ARTIFACT_URI_SCHEME = "tc://"
ARTIFACT_MEMBER_SEPARATOR = "#"
_ARTIFACT_SHA256 = re.compile(r"[0-9a-f]{64}")


class ArtifactRef(ModelStrict):
    """A sha256-pinned pointer to one file (and optionally a member) in an artifact tier.

    The one pointer type from a graph record to bytes kept off the graph. A ref names a
    file in one of the four artifact tiers (``raw``, ``genomes``, ``library``,
    ``objects``) by its key directory and relative path, pins the file's bytes by sha256,
    and optionally names a member inside the file (a tar member, a FASTA record token, an
    h5ad obs name). The member never changes which bytes are pinned: ``sha256`` is always
    the sha256 of the FILE at ``path``.

    String form: ``tc://<tier>/<key>/<path>[#<member>]``. The string carries the location
    only; the sha256 travels beside it (``ArtifactRef.parse(text, sha256=...)``), because a
    location without a hash is not a ref. The member is everything after the FIRST ``#``,
    so a member may itself contain ``#`` (Caudal's ``<gene>.fasta#<token>``) while a path
    may not.

    Defined here, on the schema surface, so its contract is fingerprinted with the record
    classes that carry it; ``torchcell.artifacts.ref`` re-exports it with the resolver.
    """

    tier: ArtifactTier
    key: str
    path: str
    member: str | None = None
    sha256: str
    bytes: int | None = None
    media_type: str | None = None

    @field_validator("sha256")
    @classmethod
    def _sha256_is_lowercase_hex(cls, value: str) -> str:
        if not _ARTIFACT_SHA256.fullmatch(value):
            raise ValueError(
                f"sha256 must be 64 lowercase hex characters, got {value!r}"
            )
        return value

    @field_validator("key")
    @classmethod
    def _key_is_one_directory_name(cls, value: str) -> str:
        if not value or "/" in value or value in (".", ".."):
            raise ValueError(f"key must be one directory name, got {value!r}")
        if value.startswith("_"):
            raise ValueError(
                f"key {value!r} starts with '_', which names a service directory"
            )
        if ARTIFACT_MEMBER_SEPARATOR in value:
            raise ValueError(f"key must not contain '#', got {value!r}")
        return value

    @field_validator("path")
    @classmethod
    def _path_is_normalized_and_relative(cls, value: str) -> str:
        if value in ("", "."):
            raise ValueError(f"path must name a file, got {value!r}")
        if value.startswith("/"):
            raise ValueError(f"path must be relative (no leading slash), got {value!r}")
        if ".." in value.split("/"):
            raise ValueError(f"path must not contain '..', got {value!r}")
        if ARTIFACT_MEMBER_SEPARATOR in value:
            raise ValueError(f"path must not contain '#', got {value!r}")
        if posixpath.normpath(value) != value:
            raise ValueError(
                f"path must be normalized, got {value!r} "
                f"(normalized: {posixpath.normpath(value)!r})"
            )
        return value

    @field_validator("bytes")
    @classmethod
    def _bytes_is_non_negative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError(f"bytes must be non-negative, got {value}")
        return value

    @model_validator(mode="after")
    def _member_requires_a_path(self) -> Self:
        if self.member is not None and (not self.member or not self.path):
            raise ValueError("member must be a non-empty name inside the file at path")
        return self

    def __str__(self) -> str:
        """``tc://<tier>/<key>/<path>[#<member>]``."""
        base = f"{ARTIFACT_URI_SCHEME}{self.tier}/{self.key}/{self.path}"
        if self.member is None:
            return base
        return f"{base}{ARTIFACT_MEMBER_SEPARATOR}{self.member}"

    @classmethod
    def parse(
        cls,
        text: str,
        *,
        sha256: str,
        bytes: int | None = None,
        media_type: str | None = None,
    ) -> Self:
        """Build a ref from its string form plus the sha256 the string does not carry.

        Raises ``ValueError`` when ``text`` is not ``tc://<tier>/<key>/<path>[#<member>]``
        with a known tier; the field validators then apply as on direct construction.
        """
        if not text.startswith(ARTIFACT_URI_SCHEME):
            raise ValueError(f"{text!r} does not start with {ARTIFACT_URI_SCHEME!r}")
        location, sep, member = text[len(ARTIFACT_URI_SCHEME) :].partition(
            ARTIFACT_MEMBER_SEPARATOR
        )
        parts = location.split("/", 2)
        if len(parts) != 3:
            raise ValueError(f"{text!r} is not tc://<tier>/<key>/<path>[#<member>]")
        tier, key, path = parts
        if tier not in ARTIFACT_TIERS:
            raise ValueError(
                f"{text!r}: unknown tier {tier!r}; tiers are {ARTIFACT_TIERS}"
            )
        return cls.model_validate(
            {
                "tier": tier,
                "key": key,
                "path": path,
                "member": member if sep else None,
                "sha256": sha256,
                "bytes": bytes,
                "media_type": media_type,
            }
        )


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

    #753 adds the PER-KEY form of invariant (2). A dict-valued field that is sourced
    for some keys and unsourced for others (Rapp 2026's ``target_metabolite_ids``:
    1,077 features with a sourced identity, 244 merged isobaric features with 2 to 11
    candidates each and no sourced pick) cannot be ``None``, so the whole-field rule
    refused the gap and the 244 keys had to live outside the record. A gap that names
    ``keys`` is admitted beside a populated mapping, on the strictly stronger
    condition that the mapping does NOT carry any of those keys: storing a value and
    declaring it missing is still refused, now key by key.
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
            value = getattr(self, gap.field)
            # --- begin #753: per-key gaps beside a partially populated map ------- #
            if gap.keys:
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"field '{gap.field}' has a per-key ProvenanceGap but is "
                        f"{type(value).__name__}, not a mapping (a per-key gap names "
                        "keys of a dict-valued field)"
                    )
                present = sorted(key for key in gap.keys if key in value)
                if present:
                    raise ValueError(
                        f"field '{gap.field}' declares keys {present} missing but "
                        "carries them (cannot both store a value and declare it "
                        "missing)"
                    )
                continue
            # --- end #753 -------------------------------------------------------- #
            if value is not None:
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
SGD_SYSTEMATIC_GENE_PATTERN = (
    r"^(Y[A-P][LR]\d{3}[WC](-[A-Z])?|Q\d{4}|YNC[A-Q]\d{4}[WC])$"
)
"""An SGD (S. cerevisiae) nuclear ORF / ncRNA / mitochondrial systematic name (the
``GenePerturbation`` pattern, restated so the background classes do not reach into that
validator)."""


# --------------------------------------------------------------------------- #
# Bacterial gene identity (plan D1, [[plan.bacteria-ontology-genome]]).
#
# A bacterial gene is identified by its GenBank locus tag, and the NAMESPACE that
# tag belongs to is carried in its own field. ``Genotype`` derives its content
# identity from the sorted systematic names of its perturbations, so without the
# namespace a b-number and a ``PP_`` tag would be two anonymous strings in one id
# space; with it, a future host whose tags DO collide with an existing one stays
# distinguishable and no rename is needed.
#
# The tag forms below were MEASURED from the deposited GenBank flat file of each
# assembly set (``torchcell.sequence.genome.registry``), not recalled: every one of
# MG1655's 4,651, BW25113's 4,490 and KT2440's 5,786 gene features matches its
# host's pattern, the three patterns are pairwise disjoint, and none of them matches
# a yeast systematic name. ``tests/torchcell/datamodels/test_schema.py`` re-pins
# those counts and the disjointness.
# --------------------------------------------------------------------------- #
BacterialGeneNamespace = Literal[
    "ecoli_k12_mg1655_bnumber",
    "ecoli_k12_bw25113_locus_tag",
    "pputida_kt2440_locus_tag",
    "ecoli_b_rel606_locus_tag",
    # --- begin #910: E. coli K-12 W3110, read RefSeq-primary --------------------- #
    "ecoli_k12_w3110_refseq_locus_tag",
    # --- end #910 ---------------------------------------------------------------- #
]
"""The bacterial members of :data:`GeneIdNamespace`: the locus-tag space of one
deposited bacterial assembly set.

One member per deposited strain assembly set, because a ``BW25113_`` number is NOT an
MG1655 b-number (measured: joining the two by gene name leaves 45 disagreements and 230
BW25113 genes absent from MG1655, so no string surgery relates them). Since the P1
unification (ontology-unify, 2026-10-15) every gene perturbation carries
``gene_namespace: GeneIdNamespace``; this sub-vocabulary remains only as the type of
``BacterialBackgroundAllele.gene_namespace`` and of the loaders' strain-to-namespace
tables, and retires with that class (P5).
"""

GeneIdNamespace = Literal[
    "scerevisiae_s288c_sgd_systematic",
    "scerevisiae_cenpk113_7d_locus_tag",
    "scerevisiae_peter2018_pangenome_orf",
    "ecoli_k12_mg1655_bnumber",
    "ecoli_k12_bw25113_locus_tag",
    "pputida_kt2440_locus_tag",
    "ecoli_b_rel606_locus_tag",
    "ecoli_k12_w3110_refseq_locus_tag",
]
"""The HOST genome a gene perturbation's identifiers are written against (P1).

One member per identifier space a served record carries or the genomes tier deposits,
measured rather than assumed (``experiments/database/scripts/measure_gene_id_namespaces.py``
over the 51 served stores of release ``2026.10.06-4b293d34`` and the deposited yeast
sets): the five bacterial locus-tag spaces above, the SGD S288C R64 systematic names every
yeast leaf stores (``YAL001C``, ``Q0010``, ``YNCA0001W``), the ``SCEN_`` locus tags of
the deposited CEN.PK113-7D assembly (``scerevisiae_CENPK113-7D_ASM257140v2``, #990), and
the Peter 2018 pangenome ORF ids Caudal 2024 stores on its accessory-presence leaf
(``13-EC1118_1N26_0012g``, 79,016 identifiers of the served store).

The field means the host, never "which pattern the string matches": a heterologous gene
symbol on an addition leaf, a released ECK accession, a CURIE and a variant site id match
no member's pattern, and the namespace still names the host the construct was built in
or the study wrote its strains against. There is deliberately no ``released``, ``other``
or ``mixed`` member: "resolved or released" is a property of the identifier and is
stated by the leaf class, and a member that skipped the pattern check would erase the
host in the same value (constitution, principle 5).
"""

GENE_ID_NAMESPACE_PATTERNS: dict[str, str] = {
    "ecoli_k12_mg1655_bnumber": r"^b\d{4}$",
    "ecoli_k12_bw25113_locus_tag": r"^BW25113_\d{4}$",
    "pputida_kt2440_locus_tag": (
        r"^PP_(?:\d{4}|(?:5|16|23)S[A-G]|t\d{2}|tm\d{2}|mr\d{2}|r\d{2})$"
    ),
    "ecoli_b_rel606_locus_tag": r"^ECB_[rt]?\d{5}$",
    "ecoli_k12_w3110_refseq_locus_tag": r"^Y75_RS\d{5}$",
    "scerevisiae_s288c_sgd_systematic": (
        r"^(Y[A-P][LR]\d{3}[WC](-[A-Z])?|Q\d{4}|YNC[A-Q]\d{4}[WC])$"
    ),
    "scerevisiae_cenpk113_7d_locus_tag": (
        r"^SCEN_(?:[A-P]\d{5}|[QR]\d+|[A-R]trna\d{1,2}[A-Z]?)$"
    ),
    "scerevisiae_peter2018_pangenome_orf": r"^\d+-\S+$",
}
"""Identifier pattern of each :data:`GeneIdNamespace` member, measured from the
deposited files of each assembly set, pairwise disjoint (pinned by
``tests/torchcell/datamodels/test_schema.py``).

MG1655 and BW25113 are uniform four-digit tags. KT2440 is four-digit for its 5,621
protein-coding genes plus the named structural-RNA tags its annotation uses: the
rRNA operons (``PP_16SA``-``PP_16SG``, ``PP_23SA``-``PP_23SG``, ``PP_5SA``-``PP_5SG``),
75 tRNA tags (``PP_t01``-``PP_t75``), 67 ``PP_mr`` tags, ``PP_tm01`` and ``PP_r01``.
Admitting only four-digit tags would silently reject 165 real KT2440 genes. E. coli B
REL606 (GCA_000017985.1) is five-digit: 4,276 ``ECB_NNNNN``, 85 tRNA ``ECB_tNNNNN`` and
22 rRNA ``ECB_rNNNNN`` of its 4,383 gene features. E. coli K-12 W3110 (GCF_000010245.2) is
the one namespace read off a RefSeq file, because its GenBank annotation
(GCA_000010245.1, the 2006 DDBJ/NIG submission) carries 4,444 gene features and not one
``locus_tag``: all 4,531 gene and pseudogene features of the RefSeq flat file are
``Y75_RSNNNNN`` (owner route #910, 2026-10-12; measured in ``teteneva2024``).

S288C is the SGD systematic feature name: protein-coding genes, pseudogenes and
transposable-element genes (``YAL001C``, ``YAL001C-A``), mitochondrial genes
(``Q0010``) and the non-coding ``YNC`` names; 7,147 of the 7,148 gene-class features of
the R64-4-1 GFF match it (the one exception, ``YNCI0012``, carries no strand letter and
was never stored). CEN.PK113-7D (GCA_002571405.2): 5,865 ``SCEN_<chromosome><5
digits>`` tags, 21 mitochondrial ``SCEN_Q``/``SCEN_R`` tags and 285 tRNA tags of the form
``SCEN_<chromosome>trna<n><amino acid>``; all 6,171 locus tags of the deposited GenBank
match. The Peter 2018 pangenome id is ``<column number>-<ORF name>`` as the presence
matrix heads its columns, the leading ``X`` stripped by ``torchcell.sequence.natural_isolate``.
"""

BACTERIAL_LOCUS_TAG_PATTERNS: dict[str, str] = {
    namespace: GENE_ID_NAMESPACE_PATTERNS[namespace]
    for namespace in get_args(BacterialGeneNamespace)
}
"""The bacterial rows of :data:`GENE_ID_NAMESPACE_PATTERNS`, keyed as
:data:`BacterialGeneNamespace` lists them; what ``_validate_bacterial_locus_tag`` and
the loaders' ``LOCUS_TAG_PATTERNS`` read."""

BACTERIAL_LOCUS_TAG_PATTERN = (
    "^(?:"
    + "|".join(pattern[1:-1] for pattern in BACTERIAL_LOCUS_TAG_PATTERNS.values())
    + ")$"
)
"""Any bacterial namespace's tag -- what ``BacterialBackgroundAllele`` accepts.

Before P1 every bacterial perturbation leaf overrode the name validator with this
union; the leaves now share ``GenePerturbation``'s validator over every
:data:`GeneIdNamespace` member, and this union serves the one bacterial-only class left.
"""


def _validate_bacterial_locus_tag(value: str) -> str:
    """Return ``value`` if it is a locus tag of one of the bacterial namespaces.

    Used by ``BacterialBackgroundAllele`` (retired with it in P5); every gene
    perturbation leaf validates through ``_validate_host_identifier`` instead.
    """
    if not re.match(BACTERIAL_LOCUS_TAG_PATTERN, value):
        raise ValueError(
            f"Invalid bacterial locus tag {value!r}; expected one of "
            + ", ".join(
                f"{namespace} ({pattern})"
                for namespace, pattern in BACTERIAL_LOCUS_TAG_PATTERNS.items()
            )
        )
    return value


def _namespace_of_locus_tag(value: str) -> str | None:
    """The BACTERIAL namespace whose pattern ``value`` matches, or ``None``.

    What ``BacterialBackgroundAllele`` and ``DerivedIdentifierMapping`` read; ``None``
    is not a failure. The released leaf no longer refuses a locus tag (P3: a released
    record's name is whatever the source wrote, and the base namespace check keeps a
    host-form identifier on the declared host).
    """
    for namespace, pattern in BACTERIAL_LOCUS_TAG_PATTERNS.items():
        if re.match(pattern, value):
            return namespace
    return None


def _namespace_of_identifier(value: str) -> str | None:
    """The :data:`GeneIdNamespace` member whose pattern ``value`` matches, or ``None``.

    ``None`` is not a failure: a heterologous gene symbol on an addition leaf, a released
    ECK accession, a CURIE and a variant site id match no host's pattern, and the leaf
    class decides whether such an identifier is admissible.
    """
    for namespace, pattern in GENE_ID_NAMESPACE_PATTERNS.items():
        if re.match(pattern, value):
            return namespace
    return None


def _validate_host_identifier(value: str) -> str:
    """Return ``value`` if it has the form of some :data:`GeneIdNamespace` member.

    ``GenePerturbation.validate_sys_gene_name``: the identifier of a placed leaf is a
    host identifier of SOME deposited namespace; ``_check_gene_namespace`` then requires
    it to be the declared host's. A leaf whose identifiers are legitimately not host
    identifiers (an addition, a natural pangenome ORF, a released name, a variant site)
    overrides this with its own rule.
    """
    if _namespace_of_identifier(value) is None:
        raise ValueError(
            f"Invalid systematic gene name format: {value!r} matches no gene identifier "
            "namespace pattern (expected one of "
            + ", ".join(
                f"{namespace} ({pattern})"
                for namespace, pattern in GENE_ID_NAMESPACE_PATTERNS.items()
            )
            + ")"
        )
    return value


def _check_gene_namespace(model: "GenePerturbation") -> None:
    """Raise when a record's identifier belongs to a namespace other than its own.

    ``gene_namespace`` names the HOST genome the record is written against. When the
    identifier IS a host identifier (a locus tag, an SGD systematic name, a pangenome
    ORF id) the two must agree, so a KT2440 tag cannot be filed under the MG1655
    namespace and a b-number cannot be filed under S288C. When the identifier is foreign
    to every namespace (a heterologous gene symbol on an addition leaf, a released ECK
    accession) the namespace still names the host the construct was built in, and there
    is nothing to cross-check. The ``identifier_mapping`` routes are checked against the
    same namespaces.
    """
    namespace = model.gene_namespace
    tag_namespace = _namespace_of_identifier(model.systematic_gene_name)
    if tag_namespace is not None and tag_namespace != namespace:
        raise ValueError(
            f"{model.systematic_gene_name!r} is a {tag_namespace} tag but "
            f"gene_namespace is {namespace!r}"
        )
    mapping = model.identifier_mapping
    if mapping is None:
        return
    if mapping.source_identifier == model.systematic_gene_name:
        raise ValueError(
            f"identifier_mapping says {model.systematic_gene_name!r} was derived from "
            "itself; a tag the source released verbatim carries no mapping"
        )
    if mapping.route == "eck_crosswalk":
        source_namespace = _namespace_of_identifier(mapping.source_identifier)
        if source_namespace == namespace:
            raise ValueError(
                f"an eck_crosswalk crosses namespaces, but {mapping.source_identifier!r} "
                f"is already a {namespace} tag"
            )
    # --- begin #753: two routes two landed loaders could not name --------------- #
    if mapping.route == "locus_tag_synonym":
        source_namespace = _namespace_of_identifier(mapping.source_identifier)
        if source_namespace != namespace:
            raise ValueError(
                f"a locus_tag_synonym stays inside one namespace, but "
                f"{mapping.source_identifier!r} is a {source_namespace} tag while "
                f"gene_namespace is {namespace!r} (a tag of another strain is an "
                "eck_crosswalk)"
            )
    # --- end #753 ---------------------------------------------------------------- #


UNIPROT_ACCESSION_PATTERN = (
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9](-\d+)?$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}"
    r"(-\d+)?$"
)
"""A UniProtKB accession as the knowledgebase writes one, with an optional isoform
suffix. UniProt's own published regular expression, which is what the pinned GenBank
assemblies' ``/db_xref="UniProtKB/Swiss-Prot:<acc>"`` values are drawn from (Gupta 2024
reads 4,275 of them off ``ecoli_K12_MG1655_ASM584v2``)."""

# --- begin #753: two routes two landed loaders could not name ------------------- #
DerivedIdentifierRoute = Literal[
    "eck_crosswalk", "jw_synonym", "gene_symbol", "uniprot_db_xref", "locus_tag_synonym"
]
# --- end #753 ------------------------------------------------------------------- #
"""How a bacterial record's stored locus tag was reached from the identifier its source
released, when the two differ.

- ``eck_crosswalk``: the source released a locus tag of ANOTHER strain's namespace (an
  MG1655 b-number labeling a BW25113 strain), and the stored tag is the one locus of the
  pinned strain carrying the same ``ECK`` accession as a ``/gene_synonym``
  (``bacteria_common.eck_crosswalk``, the one-to-one ECK join; 4,423 pairs on the
  deposited sets, 11 of them with disagreeing numbers, which is why no string surgery
  relates the namespaces).
- ``jw_synonym``: the source released a Keio ``JW`` id, which the pinned BW25113
  annotation carries as a ``/gene_synonym`` of exactly one locus.
- ``gene_symbol``: the source released a gene name (a current symbol or a symbol the
  annotation lists as a synonym), resolved to exactly one locus by the genome's name
  layers.
- ``uniprot_db_xref``: the source keyed on a UniProtKB accession (a proteomics release
  does this as a matter of course: ``sp|A5A614|YCIZ_ECOLI``), and the stored tag is the
  one locus whose CDS feature carries that accession as a
  ``/db_xref="UniProtKB/Swiss-Prot:<acc>"``. This is the PRIMARY route for a proteomics
  dataset and it outranks the symbol layer, which Gupta 2024 measured: 3,225 of 3,262
  accessions resolve through the cross-reference, and the one row where the two layers
  disagree shows the symbol column is the imprecise one
  (``sp|P0A6E9|BIOD2_ECOLI`` carries gene name ``bioD``, whose symbol resolves to
  ``b0778`` (bioD1) while its accession resolves to ``b1593`` (bioD2)).
- ``locus_tag_synonym``: the source released a RETIRED locus tag of the pinned strain's
  OWN namespace, which the annotation carries as a ``/gene_synonym`` of exactly one
  current locus. Not an ``eck_crosswalk`` (nothing crosses strains), not a
  ``jw_synonym`` (no ``JW`` id) and not a ``gene_symbol`` (it is a tag). Rapp 2026's
  ``phnE`` is released as ``b4104``, a ``/gene_synonym`` of the pseudogene ``b4583``
  (``phnE1``) on the pinned MG1655 annotation; with no member to name that route the
  loader dropped the record, which is the one record this member recovers. Four more
  landed loaders (``babu2014``, ``butland2008``, ``girgis2009``,
  ``rapp2026_platforms``) record the same unnameable route in their drop ledgers.
"""

JW_IDENTIFIER_PATTERN = r"^JW[RS]?\d{4}$"
"""A Keio ``JW`` id as BW25113's GenBank annotation writes it, measured on
``GCA_000750555.1_ASM75055v1_genomic.gbff.gz``: 4,181 ``JW`` plus four digits, 153
``JWR`` (RNA genes) and one ``JWS``; nothing else."""


class DerivedIdentifierMapping(ModelStrict):
    """A record's locus tag was DERIVED from the identifier its source released.

    Plan section 4 ([[plan.bacteria-ontology-genome]]): a use of a crosswalk is "recorded
    on the record as a derived mapping, never silently". ``source_identifier`` is what
    the source released, verbatim; ``route`` names how the stored ``systematic_gene_name``
    was reached from it. A record whose tag the source released as stored carries no
    mapping (``None`` on the leaf), so every record built before this field existed
    reads exactly as before.
    """

    source_identifier: str = Field(
        description="the identifier the source released, verbatim (e.g. the MG1655 "
        "b-number behind a BW25113 tag, a Keio JW id, a gene symbol)"
    )
    route: DerivedIdentifierRoute = Field(
        description="how systematic_gene_name was reached from source_identifier"
    )

    @model_validator(mode="after")
    def _check_route(self) -> "DerivedIdentifierMapping":
        """The released identifier has the form its route reads."""
        identifier = self.source_identifier
        if not identifier or identifier != identifier.strip():
            raise ValueError(
                f"source_identifier must be a non-empty verbatim id, got {identifier!r}"
            )
        if self.route == "jw_synonym" and not re.match(
            JW_IDENTIFIER_PATTERN, identifier
        ):
            raise ValueError(
                f"a jw_synonym route starts from a Keio JW id ({JW_IDENTIFIER_PATTERN}), "
                f"got {identifier!r}"
            )
        is_tag = _namespace_of_locus_tag(identifier) is not None
        if self.route == "eck_crosswalk" and not is_tag:
            raise ValueError(
                "an eck_crosswalk route starts from another strain's locus tag, got "
                f"{identifier!r}"
            )
        if self.route == "gene_symbol" and is_tag:
            raise ValueError(
                f"{identifier!r} is a locus tag, not a gene symbol; a tag of another "
                "strain is an eck_crosswalk"
            )
        # --- begin #753: two routes two landed loaders could not name ----------- #
        if self.route == "uniprot_db_xref" and not re.match(
            UNIPROT_ACCESSION_PATTERN, self.uniprot_accession()
        ):
            raise ValueError(
                "a uniprot_db_xref route starts from a UniProtKB accession, bare or "
                f"in a db|ACC|ENTRY header, got {identifier!r}"
            )
        if self.route == "locus_tag_synonym" and not is_tag:
            raise ValueError(
                "a locus_tag_synonym route starts from a retired locus tag of the "
                f"pinned strain's own namespace, got {identifier!r}"
            )
        # --- end #753 ------------------------------------------------------------ #
        return self

    # --- begin #753: two routes two landed loaders could not name --------------- #
    def uniprot_accession(self) -> str:
        """The accession inside ``source_identifier``, for a ``uniprot_db_xref`` route.

        A proteomics release keys either on the bare accession (``A5A614``) or on the
        FASTA header UniProt itself writes (``sp|A5A614|YCIZ_ECOLI``). Both are stored
        verbatim, and this is the one place that reads the accession out of either, so
        a consumer joining on the annotation's ``/db_xref`` never parses the string
        itself.
        """
        parts = self.source_identifier.split("|")
        return parts[1] if len(parts) == 3 else self.source_identifier

    # --- end #753 ---------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# The assembly vocabulary: the deposited sets a record can pin (ontology-unify, P2).
#
# ``AssemblySet`` is the closed vocabulary of the genomes tier's deposited ASSEMBLY sets,
# restated here rather than imported from ``torchcell.sequence.genome.registry`` because
# ``schema.py`` sits UNDER the sequence package in the import graph
# (``torchcell.sequence`` imports ``torchcell.datamodels``), so importing the registry
# here is a cycle. Equality with the registry's own constants is pinned by
# ``test_schema.py``, which is what keeps the two from drifting. It sits above
# ``GenomicSpan`` and ``ReferenceGenome`` because both are typed by it.
# --------------------------------------------------------------------------- #
BacterialReferenceStrain = Literal[
    "MG1655",
    "BW25113",
    "KT2440",
    "REL606",
    # --- begin #910: E. coli K-12 W3110, read RefSeq-primary --------------------- #
    "W3110",
    # --- end #910 ---------------------------------------------------------------- #
]
"""The sequenced strain a bacterial background's alleles are edits against.

One member per deposited assembly set, for the same reason ``BacterialGeneNamespace``
has one: the two *E. coli* K-12 strains are not interchangeable, and a record that said
only "K-12" would not say which genome its genotype is written against.
"""

#: Assembly-set id of each bacterial reference strain. Stated here rather than imported
#: from ``torchcell.sequence.genome.registry`` because ``schema.py`` sits UNDER the
#: sequence package in the import graph (``torchcell.sequence`` imports
#: ``torchcell.datamodels``), so importing the registry here is a cycle. Equality with
#: the registry's own constants is pinned by ``test_schema.py`` instead, which is what
#: keeps the two from drifting.
BACTERIAL_ASSEMBLY_SETS: dict[str, str] = {
    "MG1655": "ecoli_K12_MG1655_ASM584v2",
    "BW25113": "ecoli_K12_BW25113_ASM75055v1",
    "KT2440": "pputida_KT2440_ASM756v2",
    "REL606": "ecoli_B_REL606_ASM1798v1",
    # --- begin #910: E. coli K-12 W3110, read RefSeq-primary --------------------- #
    "W3110": "ecoli_K12_W3110_ASM1024v1",
    # --- end #910 ---------------------------------------------------------------- #
}

BacterialAssemblySet = Literal[
    "ecoli_K12_MG1655_ASM584v2",
    "ecoli_K12_BW25113_ASM75055v1",
    "pputida_KT2440_ASM756v2",
    "ecoli_B_REL606_ASM1798v1",
    # --- begin #910: E. coli K-12 W3110, read RefSeq-primary --------------------- #
    "ecoli_K12_W3110_ASM1024v1",
    # --- end #910 ---------------------------------------------------------------- #
]
"""A deposited bacterial assembly set id: the sub-vocabulary ``BacterialStrainBackground``
pins, one per ``BacterialReferenceStrain`` (``BACTERIAL_ASSEMBLY_SETS``).
"""

YeastAssemblySet = Literal[
    "sgd_S288C_R64-4-1_20230830", "scerevisiae_CENPK113-7D_ASM257140v2"
]
"""A deposited S. cerevisiae assembly set id: the SGD S288C R64-4-1 release and the
CEN.PK113-7D GenBank assembly (ASM257140v2), each a FASTA plus GFF set in the tier."""

#: The S288C set id, bound once for the yeast cross-check in ``StrainReferenceGenome``
#: (every ``StrainBackground`` is an edit against S288C); loaders import the registry's
#: own ``SGD_S288C_R64`` constant, which ``test_schema.py`` pins equal to this.
SGD_S288C_ASSEMBLY_SET = "sgd_S288C_R64-4-1_20230830"

AssemblySet = Literal[
    "ecoli_K12_MG1655_ASM584v2",
    "ecoli_K12_BW25113_ASM75055v1",
    "pputida_KT2440_ASM756v2",
    "ecoli_B_REL606_ASM1798v1",
    "ecoli_K12_W3110_ASM1024v1",
    "sgd_S288C_R64-4-1_20230830",
    "scerevisiae_CENPK113-7D_ASM257140v2",
]
"""A deposited assembly set id, as a closed vocabulary: the union of ``BacterialAssemblySet``
and ``YeastAssemblySet``, which ``test_schema.py`` pins.

An id outside this set is rejected at construction, which is the "validated against the
registry's known ids" requirement in declarative form: a loader cannot invent a set id,
and ``resolve(assembly_set, member)`` is then guaranteed to name a set the tier knows.
The vocabulary names ASSEMBLIES only; the registry's other sets are excluded by name in
``NON_ASSEMBLY_REGISTRY_SETS``, with the reason each is not a reference a record can
be written against.
"""

#: Registry sets that are NOT an assembly a record pins, each with its reason. The
#: vocabulary test proves ``AssemblySet`` plus these keys is exactly the registry's ids,
#: so a newly deposited set must be placed on one side or the other, in writing.
NON_ASSEMBLY_REGISTRY_SETS: dict[str, str] = {
    "peter2018_1011_assemblies": "a panel of 1,011 isolate assemblies and the pangenome "
    "matrices, not one reference a record's identifiers are written against; a Peter "
    "2018 record pins the S288C release its systematic names and loss-of-function "
    "coordinates are read on",
    "hale2024_322134S_variants": "a variant table of the 3S parent against S288C, not an "
    "assembly; a Hale 2024 record pins the S288C release the variants are called on",
    "go_release_2026-08-05": "an ontology release, not a genome",
}

#: The NCBI accessions of each assembly set: the GenBank / RefSeq pair read from the
#: ``_assembly_report.txt`` member deposited in that set ("# GenBank assembly accession"
#: and "# RefSeq assembly accession"), or the one accession a set has. A record names
#: one of its own set's accessions, so a (set, accession) pair cannot disagree.
ASSEMBLY_SET_ACCESSIONS: dict[str, tuple[str, ...]] = {
    "ecoli_K12_MG1655_ASM584v2": ("GCA_000005845.2", "GCF_000005845.2"),
    "ecoli_K12_BW25113_ASM75055v1": ("GCA_000750555.1", "GCF_000750555.1"),
    "pputida_KT2440_ASM756v2": ("GCA_000007565.2", "GCF_000007565.2"),
    "ecoli_B_REL606_ASM1798v1": ("GCA_000017985.1", "GCF_000017985.1"),
    # --- begin #910: E. coli K-12 W3110, read RefSeq-primary --------------------- #
    # Read off the GCF report: the GCA report of this set still names the 2006 RefSeq
    # release GCF_000010245.1 (AC_000091.1), whose files are not in the set.
    "ecoli_K12_W3110_ASM1024v1": ("GCA_000010245.1", "GCF_000010245.2"),
    # --- end #910 ---------------------------------------------------------------- #
    # The SGD release ships no assembly report. Its sequence is NCBI assembly "R64",
    # GCA_000146045.2 (SGD, 2011-04-18; RefSeq pair GCF_000146045.2), MEASURED on
    # 2026-10-15: all 16 nuclear chromosomes of the deposited
    # S288C_reference_sequence_R64-4-1_20230830.fsa are byte-identical to the 16
    # sequences of GCA_000146045.2_R64_genomic.fna.gz; R64-2-1 through R64-4-1 revised
    # the annotation, not the sequence (registry.SGD_S288C_R64_GENBANK_ACCESSION).
    "sgd_S288C_R64-4-1_20230830": ("GCA_000146045.2", "GCF_000146045.2"),
    # Read off the deposited GCA report, which names no RefSeq accession; NCBI lists no
    # paired RefSeq assembly for it (dataset report, 2026-10-15).
    "scerevisiae_CENPK113-7D_ASM257140v2": ("GCA_002571405.2",),
}

ASSEMBLY_ACCESSION_PATTERN = r"^GC[AF]_\d{9}\.\d+$"


class GenomicSpan(ModelStrict):
    """A 1-based, end-inclusive interval on one replicon of a deposited assembly set.

    ``assembly`` is the deposited set the coordinates are on (an ``AssemblySet`` id,
    never a release name): coordinates shift between assemblies, and the set is what
    ``torchcell.sequence.genome.registry.resolve`` dereferences. A span a release read
    on SGD R64-1-1 is on ``sgd_S288C_R64-4-1_20230830``, whose 16 nuclear chromosomes
    are measured sequence-identical to GCA_000146045.2.
    """

    chromosome: str
    start: int
    end: int
    assembly: AssemblySet

    @model_validator(mode="after")
    def _check_span(self) -> "GenomicSpan":
        """Positions are 1-based and ordered; the chromosome is named."""
        if self.start < 1 or self.end < self.start:
            raise ValueError(
                f"GenomicSpan needs 1 <= start <= end, got {self.start}..{self.end}"
            )
        if not self.chromosome.strip():
            raise ValueError("GenomicSpan needs a chromosome name")
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
        if not re.match(SGD_SYSTEMATIC_GENE_PATTERN, v):
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


#: Sequence Ontology mechanism of a cassette INTEGRATION. The pair is the one the
#: gene-addition leaves already pin (``GeneAdditionPerturbation``,
#: ``NaturalGenePresencePerturbation``), reused here rather than restated: an integrated
#: cassette is an insertion of sequence absent from R64. ``ALLELE_EDIT_SO`` has no
#: insertion member, because every ``AlleleEdit`` is a deletion or an in-place variant.
CASSETTE_INTEGRATION_SO: tuple[str, str] = ("SO:0000667", "insertion")


class IntegratedCassette(ProvenanceGapMixin):
    """One cassette a strain background carries INTEGRATED at a named genomic site.

    The counterpart of ``BackgroundAllele`` on the insertion axis: an allele is an edit
    AT an R64 locus, a cassette is sequence absent from R64 placed INTO one. The
    CRISPR-AID host bAID carries
    ``Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]``, which no
    ``AlleleEdit`` describes: nothing at the Ty1 delta site is deleted, four effector
    cassettes and a marker are added.

    ``name`` is the designation verbatim as the source writes it; ``locus`` is the named
    integration site verbatim (``"Delta"`` for a Ty1 delta site, ``"X3"`` / ``"X4"`` /
    ``"XI1"`` for a pre-selected intergenic landing pad), which is what the source states
    and is frequently NOT an ORF. ``locus_systematic_gene_name`` is set only when the site
    IS an R64 ORF, and is then regex-validated like every other systematic name, so a
    landing-pad label never masquerades as a gene.

    ``elements`` are the cassette's parts verbatim and IN ORDER, so the composition is
    queryable without parsing ``name``. ``marker`` is the selection marker when the
    integration carries one (``KanMX``), ``None`` for a marker-less integration, which
    the MAGIC gRNA cassettes are.

    Sourcing contract, the same ``_require_value_or_gap`` one ``BackgroundAllele`` has:
    ``provenance`` (quote + sha256) and ``zygosity`` are each set or carry a typed
    ``ProvenanceGap``, never a silent ``None``.
    """

    name: str = Field(
        description="cassette designation verbatim, as the source writes it"
    )
    locus: str = Field(
        description="the named integration site verbatim (e.g. 'Delta', 'X4'); a landing "
        "pad label, not necessarily a gene"
    )
    locus_systematic_gene_name: str | None = Field(
        default=None,
        description="the R64 ORF the cassette sits in, when the site IS an ORF; None for "
        "an intergenic landing pad or a repeat family",
    )
    elements: list[str] = Field(
        description="the cassette's elements verbatim, in the order the source writes them"
    )
    marker: str | None = Field(
        default=None,
        description="selection marker the integration carries, verbatim (e.g. 'KanMX'); "
        "None for a marker-less integration",
    )
    zygosity: Zygosity | None = Field(
        description="copies of the background carrying it; None only with a gap"
    )
    provenance: list[SourcedValue] | None = Field(
        default=None,
        description="quotes that state this cassette; None only with a gap on "
        "'provenance'",
    )

    @field_validator("locus_systematic_gene_name", mode="after")
    @classmethod
    def _validate_locus_systematic(cls, v: str | None) -> str | None:
        """When the integration site is given as an ORF, it is a real R64 feature."""
        if v is not None and not re.match(SGD_SYSTEMATIC_GENE_PATTERN, v):
            raise ValueError(f"Invalid systematic gene name {v!r}")
        return v

    @model_validator(mode="after")
    def _check_cassette(self) -> "IntegratedCassette":
        """Name, locus and elements are stated; provenance and zygosity set or gapped."""
        if not self.name.strip():
            raise ValueError("IntegratedCassette.name cannot be empty")
        if not self.locus.strip():
            raise ValueError("IntegratedCassette.locus cannot be empty")
        if not self.elements or any(not e.strip() for e in self.elements):
            raise ValueError(
                "IntegratedCassette.elements must name at least one non-empty element"
            )
        _require_value_or_gap(self, ("provenance", "zygosity"))
        return self

    @property
    def mechanism_so(self) -> tuple[str, str]:
        """(SO id, SO name) of a cassette integration: ``SO:0000667``, insertion."""
        return CASSETTE_INTEGRATION_SO

    @property
    def is_sourced(self) -> bool:
        """True when neither the cassette nor its zygosity is a declared gap."""
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
    diploid), and so must every integration's. A gene carries one allele entry, or two
    heterozygous entries for a compound heterozygote.

    ``integrations`` carries the cassettes the strain has INTEGRATED (an engineered host
    such as bAID, whose Cas effectors sit at a Ty1 delta site). It is separate from
    ``alleles`` because an insertion at a landing pad is not an edit to an R64 ORF and
    has no ``AlleleEdit``; it defaults to empty, so every background written before it
    existed reads exactly as before.
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
    integrations: list[IntegratedCassette] = Field(
        default_factory=list,
        description="every cassette the background carries integrated at a named site, "
        "each sourced or gapped; empty for a strain that carries none",
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
        for cassette in self.integrations:
            if cassette.zygosity is not None and cassette.zygosity not in allowed:
                raise ValueError(
                    f"{cassette.name}: zygosity {cassette.zygosity.value} does not fit "
                    f"a {self.ploidy} background"
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
        """True when the background, every allele and every integration carry no gap."""
        return (
            not self.provenance_gaps
            and all(a.is_sourced for a in self.alleles)
            and all(c.is_sourced for c in self.integrations)
        )

    def integrations_at(self, locus: str) -> list[IntegratedCassette]:
        """The cassettes integrated at one named site (empty = nothing integrated)."""
        return [c for c in self.integrations if c.locus == locus]


class ReferenceGenome(HashableProvenanceGapMixin):
    """The deposited reference a record's identifiers and background edits are written
    against.

    ``species`` and ``strain`` name the host as the source writes it. ``ploidy`` is the
    genome-wide baseline copy number of the unperturbed strain (``"haploid"`` = 1
    autosomal copy, ``"diploid"`` = 2). It lives here, not on a perturbation, because it
    applies to the WHOLE genome (WT and every unperturbed gene alike); a per-locus
    ``EngineeredCopyNumberPerturbation.copy_number`` records the deviation from this
    baseline. Defaults to ``"haploid"``.

    ``assembly_set`` is the deposited set in the genomes tier (sha256-pinned FASTA plus
    GFF) that the record's gene identifiers and coordinates, and the background's
    edits, are written against. It has ONE meaning, and that meaning is NOT "this
    strain's own assembly": a BY4741 or FY4 record pins ``sgd_S288C_R64-4-1_20230830``
    because its systematic names and its background alleles are edits against the S288C
    release, exactly as a BW25113 record pins the BW25113 set because its locus tags
    are that assembly's. The bacterial cross-check (``BacterialStrainBackground``
    pins the strain's set) and the yeast one (a ``StrainBackground`` is an edit of
    S288C, so the S288C set) are the same rule read on each family.

    The pin is REQUIRED: a value, or a typed ``ProvenanceGap`` on ``assembly_set`` for
    a record whose reference is not deposited in the tier (an isolate's own
    annotation, a BL21 ``ECBD_`` tag space). Depositing the set is what closes such a
    gap, so its reason is ``deferred_pending_source_review`` with ``note`` naming the
    reference to deposit. A silent default to S288C would be a fallback; the field has
    no default, so every loader states its reference. ``assembly_accession`` is the NCBI
    accession of the pinned set (GenBank ``GCA_`` or RefSeq ``GCF_``, one of that set's
    own in ``ASSEMBLY_SET_ACCESSIONS``): required whenever a set is pinned and ``None``
    exactly when the set is gapped.

    This is the reference of a record whose strain IS the deposited reference. A typed
    strain background is carried by ``StrainReferenceGenome`` (#507), whose
    ``background`` is required. Hashable through the mixin (the canonical JSON), as it
    was before it carried a gap list.
    """

    species: str
    strain: str
    ploidy: Literal["haploid", "diploid"] = "haploid"
    assembly_set: AssemblySet | None = Field(
        description="the deposited assembly set the record's identifiers and background "
        "edits are written against (never 'this strain's own assembly'); None only "
        "with a ProvenanceGap on 'assembly_set'"
    )
    assembly_accession: str | None = Field(
        description="NCBI accession of the pinned set, GenBank 'GCA_...' or RefSeq "
        "'GCF_...', one of that set's own; None exactly when assembly_set is gapped"
    )

    @field_validator("assembly_accession", mode="after")
    @classmethod
    def _validate_accession(cls, v: str | None) -> str | None:
        """A stated accession has the NCBI assembly-accession shape."""
        if v is not None and not re.match(ASSEMBLY_ACCESSION_PATTERN, v):
            raise ValueError(
                f"invalid assembly accession {v!r}; expected 'GCA_NNNNNNNNN.N' or "
                "'GCF_NNNNNNNNN.N'"
            )
        return v

    @model_validator(mode="after")
    def _check_assembly_pin(self) -> "ReferenceGenome":
        """The set is pinned or typed-gapped; the accession is the pinned set's own."""
        _require_value_or_gap(self, ("assembly_set",))
        if self.assembly_set is None:
            if self.assembly_accession is not None:
                raise ValueError(
                    f"assembly_accession {self.assembly_accession!r} is stated but "
                    "assembly_set is gapped (an accession names a pinned set)"
                )
            return self
        if self.assembly_accession is None:
            raise ValueError(
                f"assembly set {self.assembly_set!r} is pinned but assembly_accession "
                "is None (the accession is required whenever a set is pinned)"
            )
        allowed = ASSEMBLY_SET_ACCESSIONS[self.assembly_set]
        if self.assembly_accession not in allowed:
            raise ValueError(
                f"{self.assembly_accession!r} is not an accession of "
                f"{self.assembly_set!r} (expected one of {allowed})"
            )
        return self


#: BW25113's background genotype, VERBATIM from the ``/note`` qualifier of the ``source``
#: feature of ``GCA_000750555.1_ASM75055v1_genomic.gbff.gz`` in the
#: ``ecoli_K12_BW25113_ASM75055v1`` assembly set. The full qualifier reads
#: "from B. L. Wanner laboratory; genotype: rrnB3 lacZ4787 hsdR514 (araBAD)567
#: (rhaBAD)568 rph-1"; this constant is the genotype clause of it. The source is a
#: sha256-pinned artifact in the genomes tier, so the string has provenance without a
#: paper quote, and ``test_schema.py`` re-reads it from those bytes.
BW25113_BACKGROUND_GENOTYPE = "rrnB3 lacZ4787 hsdR514 (araBAD)567 (rhaBAD)568 rph-1"

#: The same genotype split into its six lesion designations, each verbatim. Deliberately
#: NOT expanded into ``BacterialBackgroundAllele`` records here: the ``/note`` gives each
#: lesion's DESIGNATION and nothing else, while an allele record also asserts an edit
#: kind, a functional status and a locus. Three of the six name a gene BW25113's own
#: annotation still carries (measured: lacZ at BW25113_0344, hsdR at BW25113_4350, rph at
#: BW25113_3643) and the two operon deletions name genes it does NOT annotate at all
#: (araB, araA, rhaB, rhaA are absent, which is what being deleted looks like), so
#: mapping those to a locus would mean borrowing MG1655 b-numbers -- the cross-strain
#: inference this plan refuses. The loader that needs the typed alleles sources the edit
#: kinds from the BW25113 construction paper and records which lesion it mapped how.
BW25113_BACKGROUND_LESIONS: tuple[str, ...] = (
    "rrnB3",
    "lacZ4787",
    "hsdR514",
    "(araBAD)567",
    "(rhaBAD)568",
    "rph-1",
)


class BacterialBackgroundAllele(ProvenanceGapMixin):
    """One allele a bacterial strain background carries, as an edit against its assembly.

    The bacterial sibling of ``BackgroundAllele``: same sourced-or-gapped contract and
    the same ``AlleleEdit`` mechanism vocabulary, with two differences that follow from
    the host. ``systematic_gene_name`` is a locus tag validated against
    ``gene_namespace``'s pattern rather than an R64 ORF name, and there is no
    ``zygosity`` field at all: these strains are haploid, so a copy count would have one
    possible value and would be a field that cannot carry information.

    ``allele_name`` is the designation verbatim as the source writes it (``lacZ4787``,
    ``rph-1``); ``gene_name`` is the gene symbol the assembly's annotation carries
    (``lacZ``). ``functional`` says whether the allele keeps the gene's function, which
    is what a per-locus dosage read needs. ``cassette`` is required for a
    ``cassette_replacement`` and forbidden otherwise, exactly as on the yeast class.
    """

    systematic_gene_name: str
    gene_namespace: BacterialGeneNamespace
    gene_name: str
    allele_name: str
    edit: AlleleEdit
    functional: bool = Field(
        description="True if the allele keeps the gene's function; False for a null"
    )
    cassette: str | None = Field(
        default=None,
        description="cassette that replaced the gene, verbatim (e.g. 'kan', 'FRT-kan-FRT'); "
        "set iff edit == cassette_replacement",
    )
    deleted_span: GenomicSpan | None = Field(
        default=None,
        description="removed interval on the pinned assembly, when the source gives it",
    )
    provenance: list[SourcedValue] | None = Field(
        default=None,
        description="quotes that state this allele; None only with a gap on "
        "'provenance' (an asserted-but-unsourced allele)",
    )

    @field_validator("systematic_gene_name", mode="after")
    @classmethod
    def _validate_systematic(cls, v: str) -> str:
        """The allele sits at a real locus tag of one of the bacterial namespaces."""
        return _validate_bacterial_locus_tag(v)

    @model_validator(mode="after")
    def _check_allele(self) -> "BacterialBackgroundAllele":
        """Tag agrees with namespace; cassette iff cassette_replacement; sourced or gapped."""
        tag_namespace = _namespace_of_locus_tag(self.systematic_gene_name)
        if tag_namespace != self.gene_namespace:
            raise ValueError(
                f"{self.systematic_gene_name!r} is a {tag_namespace} tag but "
                f"gene_namespace is {self.gene_namespace!r}"
            )
        if (self.edit is AlleleEdit.cassette_replacement) != (
            self.cassette is not None
        ):
            raise ValueError(
                "cassette is required for a cassette_replacement and forbidden for "
                f"edit={self.edit.value}"
            )
        _require_value_or_gap(self, ("provenance",))
        return self

    @property
    def mechanism_so(self) -> tuple[str, str]:
        """(SO id, SO name) of this allele's edit."""
        return ALLELE_EDIT_SO[self.edit]

    @property
    def is_sourced(self) -> bool:
        """True when the allele carries no declared gap."""
        return not self.provenance_gaps


class BacterialStrainBackground(ProvenanceGapMixin):
    """The genome content a bacterial strain carries beyond its reference assembly.

    The bacterial sibling of ``StrainBackground``. ``name`` is the strain label the
    records join on and must equal the reference's ``strain``; ``reference_strain`` is
    the sequenced strain every allele is an edit against, and ``assembly_set`` is the
    deposited bytes that strain means. There is no ``mating_type`` and no ``ploidy``:
    both are eukaryote facts, and a haploid single-replicon genome has nothing to state.

    The motivating case is BW25113, whose lesions are given by its own deposited GenBank
    ``source`` feature (``BW25113_BACKGROUND_GENOTYPE``) rather than by a paper, and the
    production-chassis rows, which carry stacked deletions plus heterologous cassettes.
    ``construction`` is a one-line method statement; ``provenance`` is set or gapped.
    """

    name: str
    reference_strain: BacterialReferenceStrain
    assembly_set: BacterialAssemblySet
    parents: list[str] | None = Field(
        default=None, description="strains this background was made from, verbatim"
    )
    construction: str | None = Field(
        default=None, description="how the background was made, one line"
    )
    genotype_statement: str | None = Field(
        default=None,
        description="the background genotype exactly as the source writes it (e.g. "
        "BW25113's 'rrnB3 lacZ4787 hsdR514 (araBAD)567 (rhaBAD)568 rph-1'), kept "
        "verbatim beside the typed alleles so nothing the source said is lost",
    )
    alleles: list[BacterialBackgroundAllele] = Field(
        default_factory=list,
        description="every allele of the background beyond the reference assembly, each "
        "sourced or gapped; empty when the source states only a genotype string",
    )
    provenance: list[SourcedValue] | None = Field(
        default=None,
        description="quotes stating the strain name and genotype; None only with a gap "
        "on 'provenance'",
    )

    @model_validator(mode="after")
    def _check_background(self) -> "BacterialStrainBackground":
        """Named, sourced-or-gapped, consistent strain/assembly, one entry per locus."""
        if not self.name.strip():
            raise ValueError("BacterialStrainBackground.name cannot be empty")
        _require_value_or_gap(self, ("provenance",))
        expected = BACTERIAL_ASSEMBLY_SETS[self.reference_strain]
        if self.assembly_set != expected:
            raise ValueError(
                f"reference strain {self.reference_strain!r} is assembly set "
                f"{expected!r}, not {self.assembly_set!r}"
            )
        seen: set[str] = set()
        for allele in self.alleles:
            if allele.systematic_gene_name in seen:
                raise ValueError(
                    f"{allele.systematic_gene_name}: a haploid background carries one "
                    "allele entry per locus"
                )
            seen.add(allele.systematic_gene_name)
        return self

    def alleles_at(self, systematic_gene_name: str) -> list[BacterialBackgroundAllele]:
        """The background's allele entry at one locus (empty = the reference allele)."""
        return [
            a for a in self.alleles if a.systematic_gene_name == systematic_gene_name
        ]

    def functional_copies(self, systematic_gene_name: str) -> int:
        """Functional copies of a gene in the UNPERTURBED background: 1 or 0.

        Haploid, so the answer is 1 unless the background carries a non-functional
        allele at the locus.
        """
        return (
            0
            if any(not a.functional for a in self.alleles_at(systematic_gene_name))
            else 1
        )

    @property
    def is_fully_sourced(self) -> bool:
        """True when the background and every allele carry no declared gap."""
        return not self.provenance_gaps and all(a.is_sourced for a in self.alleles)


class StrainReferenceGenome(ReferenceGenome):
    """A ``ReferenceGenome`` that states its typed strain background (#507).

    ``background`` is REQUIRED and holds the content the strain carries beyond the
    pinned assembly: the eukaryotic ``StrainBackground`` (mating type, ploidy, alleles,
    integrations) or the ``BacterialStrainBackground`` (reference strain, alleles,
    genotype statement), each sourced or typed-gapped. Its ``name`` equals ``strain`` so
    the free string and the typed object cannot disagree; a ``StrainBackground`` also
    agrees on ``ploidy`` and requires the S288C set (every ``StrainBackground`` is an
    edit of S288C), and a ``BacterialStrainBackground`` pins the same ``assembly_set``
    as the reference. One field admits both families until the shared background base
    lands (P5); the union is the typed home the retired ``AssemblyReferenceGenome``'s
    optional ``background`` had, with the requiredness this class has always had.

    An ``ExperimentReference`` holds its reference as ``ReferenceGenome |
    StrainReferenceGenome``, so the background survives serialization in every
    experiment family without a narrowed clone of the pair: measured on pydantic
    2.12.3, a subclass in a base-or-subclass union dumps its own fields and revalidates
    to the subclass, because the base forbids extras. The record stores it once per
    reference (the LMDB interns the whole reference; the graph writes it on the
    ``genome`` and ``experiment reference`` nodes).
    """

    background: StrainBackground | BacterialStrainBackground

    @model_validator(mode="after")
    def _check_background_agrees(self) -> "StrainReferenceGenome":
        """The background names the same strain, ploidy and assembly as the reference."""
        if self.background.name != self.strain:
            raise ValueError(
                f"background name {self.background.name!r} != strain {self.strain!r}"
            )
        if isinstance(self.background, StrainBackground):
            if self.background.ploidy != self.ploidy:
                raise ValueError(
                    f"background ploidy {self.background.ploidy!r} != ploidy "
                    f"{self.ploidy!r}"
                )
            if self.assembly_set != SGD_S288C_ASSEMBLY_SET:
                raise ValueError(
                    f"a StrainBackground is an edit of "
                    f"{self.background.reference_strain!r}, so assembly_set must be "
                    f"{SGD_S288C_ASSEMBLY_SET!r}, not {self.assembly_set!r}"
                )
        elif self.background.assembly_set != self.assembly_set:
            raise ValueError(
                f"background pins assembly set {self.background.assembly_set!r} != "
                f"{self.assembly_set!r}"
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


#: An S288C systematic feature name: protein-coding genes, pseudogenes and
#: transposable-element genes (``YAL001C``, ``YAL001C-A``), mitochondrial genes
#: (``Q0010``) and the non-coding ``YNC`` names. The S288C row of
#: :data:`GENE_ID_NAMESPACE_PATTERNS`, compiled; the identity form of every yeast leaf
#: whose ``gene_namespace`` defaults to S288C.
_SYSTEMATIC_GENE_NAME_RE = re.compile(
    GENE_ID_NAMESPACE_PATTERNS["scerevisiae_s288c_sgd_systematic"]
)

#: The S288C member of :data:`GeneIdNamespace`, the class default of every leaf whose
#: name validator pins the S288C systematic form (the stored name's form proves the
#: namespace, so the default asserts nothing the record does not already say).
SGD_S288C_NAMESPACE: Final = "scerevisiae_s288c_sgd_systematic"


class GenePerturbation(ModelStrict):
    """Base perturbation of a single gene by systematic and common name.

    ``gene_namespace`` names the HOST genome the record's identifiers are written
    against (P1, ontology-unify 2026-10-15): the host is a fact of every record, and
    before this field it was expressible only by choosing a bacterial leaf class. It is
    REQUIRED on the base. A leaf whose name validator pins the S288C systematic form
    defaults it to :data:`SGD_S288C_NAMESPACE`, because the stored name's form already
    proves the namespace; a leaf whose validator is relaxed (an addition, a natural
    pangenome ORF, a released name, a variant site) carries no default, so the loader
    states the host. When the identifier has the form of some namespace it must be the
    declared one (``_check_gene_namespace``); when it has none, the leaf decides what is
    admissible.

    ``identifier_mapping`` records how a stored identifier was DERIVED from the one the
    source released, when the two differ; ``None`` when the source released it as stored.

    ``provenance`` records whether the difference from the reference was ENGINEERED in
    the lab or arose NATURALLY in an isolate. It defaults to ``"engineered"`` so the many
    engineered perturbation types need not repeat it; natural types set
    ``provenance="natural"`` as a class default.
    """

    systematic_gene_name: str
    perturbed_gene_name: str
    gene_namespace: GeneIdNamespace = Field(
        description="the HOST genome this record's identifiers are written against; "
        "when systematic_gene_name has the form of a namespace it must be this one"
    )
    identifier_mapping: DerivedIdentifierMapping | None = Field(
        default=None,
        description="how systematic_gene_name was derived from the identifier the "
        "source released; None when the source released it as stored",
    )
    provenance: str = "engineered"

    @field_validator("systematic_gene_name", mode="after")
    @classmethod
    def validate_sys_gene_name(cls, v: str) -> str:
        """The identifier has the form of some host namespace (see the class doc)."""
        return _validate_host_identifier(v)

    @model_validator(mode="after")
    def _identifier_agrees_with_namespace(self) -> Self:
        """A host-form identifier belongs to the declared host; mappings stay in it."""
        _check_gene_namespace(self)
        return self

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
    scaffold-and-defer. ``effector_plasmid_ref`` is an off-graph ``ArtifactRef`` left
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
    effector_plasmid_ref: ArtifactRef | None = Field(
        default=None,
        description="sha256-pinned ArtifactRef to the full effector+guide cassette in a "
        "plasmid/SBOL store (future full-plasmid capture; None today). The ref carries "
        "the file's sha256 itself.",
    )


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
    """Gene deletion via KanMX or NatMX gene replacement (engineered absence).

    The parent of every marker-specific deletion leaf. It is stored ITSELF (a
    ``GenePerturbationType`` member since #988) only where a source states an engineered
    deletion and not the cassette that made it: each child asserts a marker, so it is the
    one deletion class that claims nothing the source does not say. Such a record sets
    ``description`` to say what the source states, since the class default names KanMX
    and NatMX. First case: the HO deletion Peter 2018 Supplementary Table 1 marks on 124
    natural isolates without naming the marker.
    """

    description: str = "Deletion via KanMX or NatMX gene replacement"
    perturbation_type: Literal["deletion"] = "deletion"
    gene_namespace: GeneIdNamespace = Field(
        default=SGD_S288C_NAMESPACE,
        description="the host genome; S288C by default because this leaf's name validator pins the S288C systematic form",
    )
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
    gene_namespace: GeneIdNamespace = Field(
        default=SGD_S288C_NAMESPACE,
        description="the host genome; S288C by default because this leaf's name validator pins the S288C systematic form",
    )
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
    gene_namespace: GeneIdNamespace = Field(
        default=SGD_S288C_NAMESPACE,
        description="the host genome; S288C by default because this leaf's name validator pins the S288C systematic form",
    )


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
    gene_namespace: GeneIdNamespace = Field(
        default=SGD_S288C_NAMESPACE,
        description="the host genome; S288C by default because this leaf's name validator pins the S288C systematic form",
    )


class AllelePerturbation(SequencePerturbation, ModelStrict):
    """Generic allele perturbation via amino acid substitution."""

    description: str = (
        "Allele compromised by amino acid substitution without more generic "
        "phenotypic information specified."
    )
    # seq: str = "NOT IMPLEMENTED"
    perturbation_type: Literal["allele"] = "allele"
    gene_namespace: GeneIdNamespace = Field(
        default=SGD_S288C_NAMESPACE,
        description="the host genome; S288C by default because this leaf's name validator pins the S288C systematic form",
    )


class SuppressorAllelePerturbation(SequencePerturbation, ModelStrict):
    """Suppressor allele that raises fitness in the presence of a perturbation."""

    description: str = (
        "suppressor allele that results in higher fitness in the presence"
        "of a perturbation, compared to the fitness of the perturbation alone."
    )
    perturbation_type: Literal["suppressor_allele"] = "suppressor_allele"
    gene_namespace: GeneIdNamespace = Field(
        default=SGD_S288C_NAMESPACE,
        description="the host genome; S288C by default because this leaf's name validator pins the S288C systematic form",
    )


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
    gene_namespace: GeneIdNamespace = Field(
        default=SGD_S288C_NAMESPACE,
        description="the host genome; S288C by default because this leaf's name validator pins the S288C systematic form",
    )
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
    known, is an off-graph ``ArtifactRef`` (``sequence_ref``), never inlined.
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
    sequence_ref: ArtifactRef | None = Field(
        default=None,
        description="sha256-pinned ArtifactRef to the reference gene sequence in an "
        "artifact tier (file + member); None when no store holds it",
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
    sequence_ref: ArtifactRef | None = Field(
        default=None,
        description="sha256-pinned ArtifactRef to the ORF sequence in an artifact tier "
        "(file + member); None when no store holds it",
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
    ``sequence_ref`` identify the record in the off-graph gene-keyed store (an
    ``ArtifactRef``, dereferenced at load and verified by its sha256), mirroring the ``GeneAddition`` pointer pattern.

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
    gene_namespace: GeneIdNamespace = Field(
        default=SGD_S288C_NAMESPACE,
        description="the host genome; S288C by default because this leaf's name validator pins the S288C systematic form",
    )
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
    sequence_ref: ArtifactRef | None = Field(
        default=None,
        description=(
            "sha256-pinned ArtifactRef to the variant allele in an artifact tier, e.g. "
            "tc://genomes/peter2018_1011_assemblies/<tarball>#<gene>.fasta#<token>; "
            "None until a store holds it"
        ),
    )


# --------------------------------------------------------------------------- #
# begin #987 / #988: the natural-isolate genotype of the Peter 2018 panel.
# A natural loss-of-function call (a sibling of SequenceVariantPerturbation
# under SequencePerturbation) and the isolate descriptor ``Genotype`` carries
# beside its perturbations (ploidy, genome zygosity, aneuploidy, manipulation).
# Built for every natural-isolate loader by
# ``torchcell.datasets.scerevisiae.peter2018_genotype``.
# --------------------------------------------------------------------------- #
class NaturalLossOfFunctionCause(StrEnum):
    """Which Peter 2018 release matrix called an isolate's allele a loss of function.

    Each member names one matrix as the release describes it (the host's
    ``README.txt``, sha256 ``f48e32771b44d38e88b9c8833f6f47e1a30eb6634ffccf55be42b117dd24f025``):

    - ``nonsense_or_sift``: a 1 in ``1011LossOfFunction.xls.gz``, "loss of function
      predicted for all genes in the 1011 dataset (nonsense mutation + SIFT
      predictions)". The release folds the two predictions into one bit and does not
      say which applied, so it is one member.
    - ``homozygous_frameshift``: a 1 in ``genesMatrix_Frameshift.tab.gz``, "for each
      isolate, the presence or absence (indicated by 1 or 0, respectively) of
      homozygous frameshift is reported in each gene, based on the number of bases
      affected by indels."

    A gene called by both matrices carries two perturbations, one per call.
    """

    nonsense_or_sift = "nonsense_or_sift"
    homozygous_frameshift = "homozygous_frameshift"


class NaturalLossOfFunctionPerturbation(SequencePerturbation, ModelStrict):
    """One released loss-of-function call on a natural isolate's gene, as released.

    NATURAL-variation leaf, a SIBLING of ``SequenceVariantPerturbation`` under
    ``SequencePerturbation`` (AXIS 3, the allelic axis), not a child of it: a call is a
    statement a release matrix makes about the isolate's gene, not an allele, so it is
    never counted as a variant. It is stored whatever any other file says about the same
    gene; nothing is looked up to admit, drop or annotate it.

    One perturbation per released call: ``cause`` names the matrix and ``call_ref`` pins
    its bytes, and the row is keyed as that matrix keys it. ``release_code`` is the
    row's pangenome ORF number (``CODE``), ``release_name`` the loss-of-function
    matrix's ``NAME`` (the frameshift matrix has no such column), ``release_gene_name``
    the row's ``Gene`` and ``release_span`` its ``CH``/``START``/``STOP`` on the
    reference the release was called against. ``systematic_gene_name`` is the gene key
    the release gives the row: ``NAME`` for a loss-of-function row, the pangenome ORF
    name the release gives the code for a coded frameshift row, and ``Gene`` verbatim
    for a frameshift row released without a code (a standard name on some rows, so the
    yeast systematic-name pattern is not enforced here). ``perturbed_gene_name`` is the
    same key.

    The SO mechanism is ``loss_of_function_variant`` (SO:0002054, "A sequence variant
    whereby the gene product has diminished or abolished function.").
    """

    description: str = (
        "Natural predicted loss of function of a gene in an isolate, as one released "
        "matrix calls it"
    )
    perturbation_type: Literal["natural_loss_of_function"] = "natural_loss_of_function"
    provenance: str = "natural"
    mechanism_so_id: str = "SO:0002054"
    mechanism_so_name: str = "loss_of_function_variant"
    strain_id: str = Field(description="isolate id whose gene is called, e.g. 'AAB'")
    cause: NaturalLossOfFunctionCause = Field(
        description="the release matrix that made the call"
    )
    call_ref: ArtifactRef = Field(
        description="the sha256-pinned release matrix the call is read from"
    )
    release_code: int | None = Field(
        description="the row's pangenome ORF number (CODE); None for a frameshift row "
        "the release gives no code"
    )
    release_name: str | None = Field(
        description="the loss-of-function matrix's NAME, verbatim; None for a "
        "frameshift row (that matrix has no NAME column)"
    )
    release_gene_name: str = Field(description="the row's Gene, verbatim")
    release_span: GenomicSpan = Field(
        description="the row's coordinates (CH, START, STOP) on the reference the "
        "release was called against"
    )

    @field_validator("systematic_gene_name", mode="after")
    @classmethod
    def validate_sys_gene_name(cls, v: str) -> str:
        """Relax: the release's key, which is a standard name on some frameshift rows."""
        if not v:
            raise ValueError("systematic_gene_name must be non-empty")
        return v

    @model_validator(mode="after")
    def _check_call(self) -> "NaturalLossOfFunctionPerturbation":
        """A natural call, keyed the way its matrix keys its rows."""
        if self.provenance != "natural":
            raise ValueError("a natural loss-of-function call has provenance 'natural'")
        if self.perturbed_gene_name != self.systematic_gene_name:
            raise ValueError("perturbed_gene_name is the release's gene key")
        if self.cause is NaturalLossOfFunctionCause.nonsense_or_sift:
            if self.release_code is None or self.release_name is None:
                raise ValueError(
                    "a loss-of-function matrix row carries its CODE and NAME"
                )
            if self.systematic_gene_name != self.release_name:
                raise ValueError("a loss-of-function call is keyed by the row's NAME")
            return self
        if self.release_name is not None:
            raise ValueError("the frameshift matrix has no NAME column")
        if self.release_code is None and (
            self.systematic_gene_name != self.release_gene_name
        ):
            raise ValueError(
                "a frameshift row without a code is keyed by its Gene, verbatim"
            )
        return self


class GenomeZygosity(StrEnum):
    """A natural isolate's genome-wide zygosity, as Peter 2018 Table S1 classes it.

    Distinct from ``Zygosity`` (how many chromosome copies of a BACKGROUND carry one
    allele): this is a property of the whole genome, whether it carries heterozygous
    sites, written in Table S1's own two words.
    """

    homozygous = "homozygous"
    heterozygous = "heterozygous"


class IsolateManipulation(StrEnum):
    """A genetic manipulation a natural isolate's own annotation marks.

    - ``ho_deletion``: Peter 2018 Table S1 reads 'yes' in 'HO deletion' ("124 were
      non-natural haploid with the HO gene deleted"). The deletion itself is a
      ``DeletionPerturbation`` of HO (YDL227C) on the genotype.
    """

    ho_deletion = "ho_deletion"


class ChromosomeCopyChange(ModelStrict):
    """One aberrant chromosome of a whole-chromosome aneuploidy annotation.

    Peter 2018 Table S1 writes aneuploidy as "aneu;+/-X*Y" "with +/- the gain or loss
    of chromosome, X the number of supernumerary copies and Y the concerned
    chromosome"; ``aneu;+1*9;`` is one extra copy of chromosome IX over the isolate's
    basal ploidy.
    """

    chromosome: int = Field(ge=1, le=16, description="chromosome number, 1 to 16")
    change: int = Field(
        description="signed copies gained (+) or lost (-) relative to the basal ploidy"
    )

    @field_validator("change", mode="after")
    @classmethod
    def _nonzero(cls, v: int) -> int:
        """A copy change of zero is not an aneuploidy."""
        if v == 0:
            raise ValueError("a chromosome copy change cannot be 0")
        return v


class NaturalIsolateDescriptor(HashableProvenanceGapMixin):
    """What a natural isolate's own annotation says about its whole genome.

    Carried on ``Genotype.natural_isolate`` beside the gene-keyed perturbations, because
    none of it is gene-keyed: the basal ploidy (1n to 5n), the genome-wide zygosity,
    the whole-chromosome copy changes, and whether the isolate was genetically
    manipulated before sequencing. Read from one row of Peter 2018 Supplementary Table 1
    (``annotation_ref``), so every loader on the 1,011-genome panel describes an isolate
    identically.

    Every element is the table's value or a typed gap, never a guess:

    - ``ploidy`` is ``None`` only with a gap (the one isolate Table S1 writes ``Xn``).
    - ``chromosome_copy_changes`` is ``[]`` for a euploid isolate and ``None`` only with
      a gap (an empty 'Aneuploidies' cell).
    - ``genetic_manipulation`` is ``ho_deletion`` where Table S1 marks one; otherwise it
      is ``None`` with a gap, because the paper also says "the 93 external isolates were
      genetically manipulated and made homozygous before sequencing" and Table S1 does
      not mark which isolates those are.

    These are the isolate's genome as SEQUENCED. For a manipulated isolate the paper
    says the ploidy is not the natural one ("217 strains were genetically manipulated
    and no longer in their natural ploidy states").
    """

    isolate: str = Field(description="the isolate's standardized name, e.g. 'AAB'")
    annotation_ref: ArtifactRef = Field(
        description="the sha256-pinned workbook holding the isolate's annotation row "
        "(member: the sheet)"
    )
    genetic_manipulation: IsolateManipulation | None = Field(
        description="a manipulation the annotation marks; None only with a gap"
    )
    ploidy: int | None = Field(
        ge=1, description="basal ploidy as annotated; None only with a gap"
    )
    zygosity: GenomeZygosity = Field(description="genome-wide zygosity as annotated")
    chromosome_copy_changes: list[ChromosomeCopyChange] | None = Field(
        description="whole-chromosome gains and losses, [] for a euploid isolate; None "
        "only with a gap"
    )

    @model_validator(mode="after")
    def _sourced_or_gapped(self) -> "NaturalIsolateDescriptor":
        """Each optional element is set or gapped, and the copy changes are canonical."""
        gapped = self.gapped_fields()
        for name in ("genetic_manipulation", "ploidy", "chromosome_copy_changes"):
            if getattr(self, name) is None and name not in gapped:
                raise ValueError(
                    f"NaturalIsolateDescriptor.{name} is None and carries no "
                    "ProvenanceGap"
                )
        changes = self.chromosome_copy_changes or []
        chromosomes = [c.chromosome for c in changes]
        if len(set(chromosomes)) != len(chromosomes):
            raise ValueError("a chromosome appears twice in chromosome_copy_changes")
        return self


# --------------------------------------------------------------------------- #
# end #987 / #988
# --------------------------------------------------------------------------- #


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
    sequence_ref: ArtifactRef | None = Field(
        default=None,
        description="sha256-pinned ArtifactRef to the ORF sequence in an artifact tier "
        "(file + member); None when no store holds it",
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
    gene_namespace: GeneIdNamespace = Field(
        default=SGD_S288C_NAMESPACE,
        description="the host genome; S288C by default because this leaf's name validator pins the S288C systematic form",
    )
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
    gene_namespace: GeneIdNamespace = Field(
        default=SGD_S288C_NAMESPACE,
        description="the host genome; S288C by default because this leaf's name validator pins the S288C systematic form",
    )
    expression_direction: str = "increased"


class CrisprInterferencePerturbation(ExpressionModulationPerturbation, ModelStrict):
    """CRISPRi -- guide-directed dead-Cas repressor DECREASES a present gene's expression.

    Lian 2019 MAGIC uses ``dSpCas9-RD1152``; Mormino 2022 uses ``dCas9-Mxi1``.
    """

    description: str = "CRISPR interference (decreased expression of a present gene)"
    perturbation_type: Literal["crispr_interference"] = "crispr_interference"
    gene_namespace: GeneIdNamespace = Field(
        default=SGD_S288C_NAMESPACE,
        description="the host genome; S288C by default because this leaf's name validator pins the S288C systematic form",
    )
    expression_direction: str = "decreased"


# --------------------------------------------------------------------------- #
# begin ADDITIVE (Hackett 2020): an inducible synthetic promoter inserted upstream of a
# present yeast gene. The expression axis without CRISPR: the gene stays PRESENT, its
# coding sequence and copy number are unchanged, and a synthetic promoter (with its
# selection marker) is INSERTED between the native promoter and the start codon, so the
# native DNA is displaced rather than removed. The mechanism is therefore SO:0000667
# ``insertion``, not the ``delins`` of ``PromoterReplacementPerturbation``, which
# removes the native promoter. A new leaf and a new member of ``GenePerturbationType``.
# --------------------------------------------------------------------------- #
class InduciblePromoterInsertionPerturbation(
    ExpressionModulationPerturbation, ModelStrict
):
    """A synthetic inducible promoter inserted in front of a present gene's ORF.

    Hackett 2020 (IDEA) places each induced regulator under a beta-estradiol-responsive
    synthetic promoter: a cassette of selection marker plus promoter (``KanMX-Z3EVpr``,
    ``KanMX-Pgal1``, ``URA3::Z3EVpr``) inserted "directly between a target gene's first
    methionine residue and its native promoter to prevent the removal of any genomic
    DNA". The promoter answers a synthetic transcription factor the strain BACKGROUND
    carries (``GEV``, ``ZEV``, ``4ZEV``, ``Z3EV``); that activator is an integrated
    cassette of the ``StrainBackground``, named here in ``responsive_to`` so the edit
    says which activator drives it. The INDUCER and its dose are not here: they are a
    ``SmallMoleculePerturbation`` on the environment (one canonical encoding, as the
    bacterial ``PromoterReplacementPerturbation.is_inducible`` docstring sets out).

    ``expression_direction`` is ``"increased"``: the designed edit is induced
    overexpression, which is the direction under the inducer. The uninduced level can
    sit below the native one (Hackett: "86% of synthetic promoter-driven TF alleles have
    lower expression than native promoter-driven TF alleles"); that is a property of
    the record's environment (no inducer), not of the edit. ``crispr`` is re-declared
    optional and stays ``None``: no guide or Cas effector is involved.
    """

    description: str = (
        "Inducible synthetic promoter inserted upstream of a present gene's start codon"
    )
    perturbation_type: Literal["inducible_promoter_insertion"] = (
        "inducible_promoter_insertion"
    )
    gene_namespace: GeneIdNamespace = Field(
        default=SGD_S288C_NAMESPACE,
        description="the host genome; S288C by default because this leaf's name validator pins the S288C systematic form",
    )
    mechanism_so_id: str = "SO:0000667"
    mechanism_so_name: str = "insertion"
    expression_direction: str = "increased"
    promoter_name: str = Field(
        description="the inserted promoter part, verbatim (e.g. 'Z3EVpr', 'Pgal1', "
        "'4ZEVpr')"
    )
    cassette: str = Field(
        description="the inserted cassette as the source writes it, gene included "
        "(e.g. 'KanMX-Z3EVpr-OPI1', 'URA3::Z3EVpr-HMX1')"
    )
    marker: str = Field(
        description="selection marker the cassette carries, verbatim (e.g. 'KanMX', "
        "'URA3')"
    )
    responsive_to: str = Field(
        description="the synthetic transcription factor of the background that "
        "activates the promoter under the inducer, verbatim (e.g. 'ZEV', 'GEV')"
    )
    native_promoter_retained: bool = Field(
        description="True when the cassette was inserted between the native promoter "
        "and the start codon without removing native DNA (the native promoter is "
        "displaced upstream, not deleted)"
    )
    strain_id: str = Field(
        description="the strain carrying the edit, as the source's strain table names it"
    )
    crispr: CrisprConstruct | None = Field(  # type: ignore[assignment]
        default=None,
        description="no guide-directed machinery is involved in a promoter insertion; "
        "the field is inherited from the expression axis and stays None",
    )

    @model_validator(mode="after")
    def _check_insertion(self) -> "InduciblePromoterInsertionPerturbation":
        """The promoter, cassette, marker and activator are named; no CRISPR payload."""
        for name in (
            "promoter_name",
            "cassette",
            "marker",
            "responsive_to",
            "strain_id",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")
        if self.crispr is not None:
            raise ValueError("a promoter insertion carries no CRISPR construct")
        if self.expression_direction != "increased":
            raise ValueError(
                "an inducible promoter insertion is designed for induced "
                "overexpression; expression_direction must be 'increased'"
            )
        return self


# --------------------------------------------------------------------------- #
# end ADDITIVE (Hackett 2020)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Host-neutral perturbation leaves that were written for the bacterial hosts first
# (plan 3a, #731, #749, #792, #799) and took general names in P1 (ontology-unify,
# 2026-10-15). Each is a subclass of the axis leaf its OUTCOME belongs to, so every
# existing filter keeps working: ``issubclass(_, DeletionPerturbation)`` still catches
# a knockout, and the expression axis still holds every modulation. The host is the
# base class's ``gene_namespace``, stated by the loader; none of these leaves defaults
# it, because none of them pins one host's identifier form.
# --------------------------------------------------------------------------- #
class DeletionStrainPerturbation(DeletionPerturbation, ModelStrict):
    """A gene deletion STRAIN: the deletion plus the strain-level facts the source states.

    Serves the catalogued single-deletion rows (the Keio collection and the
    chemical-genomics screens built on it), the Keio-derived deletion pairs and the
    designed deletion strains of the engineered-chassis releases. Same AXIS-1
    ``state="absent"`` and SO ``deletion`` mechanism as the marker-specific yeast
    deletion leaves. ``collection``, ``cassette`` and ``construction`` are each stated
    verbatim when the source states them and are strain discriminators in the same sense
    they are on ``BarcodedKanMxDeletionPerturbation``: two records that delete the same
    gene from different collections are two strains, not two replicates. Unlike that
    leaf it asserts no marker (a Keio strain carries ``FRT-kan-FRT``, a scarless
    deletion none), and unlike ``DeletionPerturbation`` it can state one. Before P1 this
    was ``BacterialDeletionPerturbation``; the host is now ``gene_namespace``.
    """

    description: str = (
        "Gene deletion strain, with its collection, cassette and construction as stated"
    )
    perturbation_type: Literal["deletion_strain"] = "deletion_strain"  # type: ignore[assignment]
    deletion_type: str = "strain"
    gene_namespace: GeneIdNamespace = Field(
        description="the host genome the deletion is written against; required, no "
        "default, because this leaf pins no one host's identifier form"
    )
    collection: str | None = Field(
        default=None,
        description="the physical deletion collection the strain came from, verbatim "
        "(e.g. 'Keio collection'); None when unsourced",
    )
    cassette: str | None = Field(
        default=None,
        description="the exact replacement cassette, verbatim (e.g. 'FRT-kan-FRT'); "
        "None when unsourced",
    )
    construction: StrainConstruction | None = Field(
        default=None, description="accession / lab / batch / plate of this strain"
    )


class TransposonInsertionPerturbation(PresenceAbsencePerturbation, ModelStrict):
    """A transposon insertion that disrupts a gene, as a REALIZED genotype.

    The readout leaf of the transposon / RB-TnSeq rows. It is an AXIS-1 absence --
    the gene's function is gone -- by the mechanism of a ``transgenic_insertion``
    (SO:0001218), which is the honest pairing: the state is what happened to the gene,
    the SO term is what was done to the DNA.

    What makes it realized rather than designed is that the insertion was MAPPED: a
    pooled library is built by random insertion and then sequenced, so each strain's
    barcode and the position its insertion was mapped to are measurements, not a
    design. ``insertion_position`` is 1-based on the replicon of the pinned assembly
    and ``insertion_strand`` the orientation the mapping reported; both are ``None``
    when a study releases only gene-level calls. ``barcode`` is the random tag a
    Bar-seq / RB-TnSeq strain is counted by, which IS its read-out identity in a
    pooled assay.
    """

    description: str = (
        "Transposon insertion disrupting a gene, with its mapped insertion site"
    )
    perturbation_type: Literal["transposon_insertion"] = "transposon_insertion"
    state: str = "absent"
    mechanism_so_id: str = "SO:0001218"
    mechanism_so_name: str = "transgenic_insertion"
    provenance: str = "engineered"
    barcode: str | None = Field(
        default=None,
        description="the random barcode the strain is counted by in a pooled assay; "
        "None when the source released no per-strain tag",
    )
    insertion_position: int | None = Field(
        default=None,
        description="1-based position the insertion was mapped to on the replicon of "
        "the pinned assembly; None when only gene-level calls were released",
    )
    insertion_strand: Literal["+", "-"] | None = Field(
        default=None,
        description="orientation of the inserted element as mapped; None when unreleased",
    )
    transposon: str | None = Field(
        default=None,
        description="the transposon / delivery vector, verbatim (e.g. 'mariner Himar1', "
        "'Tn5'); None when unsourced",
    )
    library_pool: str | None = Field(
        default=None,
        description="the mutant library sub-pool this strain was screened in, when a "
        "study runs several; a pooled measurement is pool-relative, so the pool joins "
        "the strain identity",
    )

    @field_validator("mechanism_so_id", mode="after")
    @classmethod
    def validate_mechanism_so_id(cls, v: str) -> str:
        """Mechanism SO id is well-formed."""
        return _validate_so_id(v)

    @field_validator("insertion_position", mode="after")
    @classmethod
    def _validate_position(cls, v: int | None) -> int | None:
        """A mapped position is 1-based, so it is at least 1."""
        if v is not None and v < 1:
            raise ValueError(f"insertion_position is 1-based, got {v}")
        return v


class PromoterReplacementPerturbation(ExpressionModulationPerturbation, ModelStrict):
    """A gene's native promoter replaced by a characterized promoter part.

    The expression axis without CRISPR: the gene stays PRESENT, its coding sequence is
    unedited and its copy number unchanged, and what changed is the regulatory part
    driving it. The mechanism is therefore neither a plain insertion nor a plain
    deletion but SO:1000032 ``delins`` -- the native promoter removed and the new part
    put in its place -- rather than the ``sgRNA`` term the CRISPR leaves inherit.

    ``expression_direction`` is REQUIRED with no default, because a promoter swap can
    raise or lower expression depending on the part, and defaulting it would assert a
    direction the record did not state. ``crispr`` is re-declared as optional here (the
    axis base requires it for the guide-directed leaves); a promoter replacement
    introduces no guide and no Cas effector, so it is ``None``. The base class is left
    untouched, since widening ITS field would move every served CRISPR loader's closure.
    """

    description: str = "Native promoter replaced by a characterized promoter part"
    perturbation_type: Literal["promoter_replacement"] = "promoter_replacement"
    mechanism_so_id: str = "SO:1000032"
    mechanism_so_name: str = "delins"
    promoter_name: str = Field(
        description="identity of the promoter part installed, verbatim (e.g. 'Ptac', "
        "'P14g', 'ParaBAD')"
    )
    native_promoter: str | None = Field(
        default=None,
        description="the promoter it replaced, verbatim, when the source names it",
    )
    promoter_sequence: str | None = Field(
        default=None,
        description="sequence of the installed part, when released; None otherwise",
    )
    is_inducible: bool = Field(
        default=False,
        description="True when the installed part is inducible rather than "
        "constitutive. The INDUCER and its dose are deliberately not here: they are a "
        "``SmallMoleculePerturbation`` on the environment, carrying the typed "
        "``Compound``, so the molecule joins the shared compound layer and the genome "
        "edit does not restate it (one canonical encoding).",
    )
    crispr: CrisprConstruct | None = Field(  # type: ignore[assignment]
        default=None,
        description="no guide-directed machinery is involved in a promoter swap; the "
        "field is inherited from the expression axis and stays None",
    )


class HeterologousPathwayPerturbation(
    HashableProvenanceGapMixin, GeneAdditionPerturbation, ModelStrict
):
    """One heterologous gene of an expressed pathway, with its cassette and copy context.

    The engineered-chassis production rows. It inherits the gene-addition axis whole
    (``state="present"``, SO ``insertion``, the relaxed name validator a heterologous
    gene needs since it has no host locus tag) and adds the two things a production
    record cannot be read without: WHICH pathway the gene belongs to, so the genes of
    one pathway are recoverable as a set rather than as unrelated additions, and the
    copy context of the cassette carrying it.

    ``gene_namespace`` names the HOST genome the construct was built in, not the
    namespace of ``systematic_gene_name`` (which is typically a heterologous gene
    symbol), as it does on every leaf since P1; it is cross-checked only when the
    identifier IS a host identifier, which it legitimately is for an extra copy of a
    NATIVE gene in the same pathway. No default: the loader states the host.

    ``source_organism`` is stated or a typed ``ProvenanceGap`` (#915). A production
    paper can name a pathway part by its token alone (Menasalvas 2025's pIY670 writes
    ``mvaS``, ``mvaE``, ``MKmm``, ``PMDHKQ`` and ``aphA`` and never says which organism
    each came from), and the leaf previously forced a sentinel string such as
    ``"unreported"`` into the field, which reads as an organism name. The field stays
    required with no default, as ``Media.name`` does: ``None`` only beside a gap naming
    ``source_organism``, its reason and where the source was searched. The widening is
    on this leaf alone, as ``PromoterReplacementPerturbation`` re-declares ``crispr``:
    the yeast ``GeneAdditionPerturbation`` loaders state every organism, and their
    closures do not move.
    """

    description: str = (
        "Heterologous pathway gene added on a cassette, with its copy context"
    )
    perturbation_type: Literal["heterologous_pathway"] = "heterologous_pathway"  # type: ignore[assignment]
    # --- begin #915: a typed gap on the source organism ------------------------- #
    # The organism the added gene is from, as the source states it; None ONLY with a
    # ProvenanceGap(field='source_organism') when the source names none. Required, no
    # default.
    source_organism: str | None = Field(  # type: ignore[assignment]
        description="organism the added gene is from, e.g. 'Staphylococcus aureus'; "
        "None ONLY with a ProvenanceGap(field='source_organism') when the source "
        "names no organism for this part"
    )
    # --- end #915 ---------------------------------------------------------------- #
    pathway_name: str = Field(
        description="the pathway this gene is part of, verbatim (e.g. 'isoprenol via "
        "mevalonate'), so one pathway's genes form a recoverable set"
    )
    copy_number: float = Field(
        default=1.0,
        description="copies of the cassette carrying this gene (strictly > 0; absence "
        "is not a copy count)",
    )
    promoter_name: str | None = Field(
        default=None,
        description="the promoter driving this gene on the cassette, verbatim",
    )

    @field_validator("copy_number", mode="after")
    @classmethod
    def validate_copy_number_positive(cls, v: float) -> float:
        """M2: this leaf means the gene IS present, so ``copy_number > 0``."""
        if v <= 0:
            raise ValueError(
                "copy_number must be > 0 for an ADDED gene (absence is a deletion "
                "leaf, never copy_number=0)"
            )
        return v

    # --- begin #915: a typed gap on the source organism ------------------------- #
    @model_validator(mode="after")
    def _source_organism_stated_or_gapped(self) -> "HeterologousPathwayPerturbation":
        """``source_organism`` is stated or a typed gap; never a silent ``None``.

        The mixin enforces the converse (a gapped field must be ``None``).
        """
        _require_value_or_gap(self, ("source_organism",))
        return self

    # --- end #915 ---------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Bacterial CALLED-VARIANT leaves (issue #731).
#
# An evolved clone's genotype is its parent's genomic content plus the variants a
# caller found in its resequencing, and before these classes no leaf could hold one
# of those calls. Measured on the two P. putida evolved-WGS releases rather than
# designed from the model: de Siqueira 2025 Data Set S2 releases 173 Geneious calls
# over 5 sequenced clones (58 in a GenBank locus, 10 with a RefSeq tag only, 105 with
# no locus at all) and Lim 2025's ``Fig 2B_Mutation List`` releases 159 breseq rows
# over 49 clone columns (123 single-locus, 17 intergenic, 19 spanning runs of up to 53
# loci). Four blockages those loaders counted, and what closes each:
#
#   1. ``SequenceVariantPerturbation`` admits only S288C ORF names and promises a
#      dereferenceable allele sequence these papers never deposited -> a bacterial
#      leaf with the locus-tag validator and the call itself on the record.
#   2. ``BacterialBackgroundAllele.functional`` is a required bool no release states
#      for a called variant -> no ``functional`` field here at all (the perturbation
#      axis has none, by design: the edit is not its consequence).
#   3. ``BacterialStrainBackground`` permits one allele entry per locus, which refuses
#      a clone carrying two calls in one gene (Lim's ``PP_3415`` P293S + V46I in
#      A12_F53_I1) -> ``Genotype.perturbations`` has no such rule, so two calls in one
#      locus are two perturbations.
#   4. A call with no locus tag of the pinned assembly to key to -> its own leaf,
#      keyed on the SITE. Two measured reasons, typed apart by ``VariantSiteKind``:
#      the call is intergenic, or the release named a locus in an identifier space the
#      assembly does not carry (de Siqueira's 10 RefSeq-only calls name ``PP_RS21780``
#      and ``PP_RS19075``, and ``resolve_gene_name`` on the pinned GenBank assembly
#      returns ``retired``, "not found", for both).
#
# The call itself is COMPOSED (``BacterialVariantCall``) rather than repeated on each
# leaf, so what a call is is defined once and the three leaves differ only in WHERE
# the call sits: in a locus, between two loci, or spanning a run of them.
# --------------------------------------------------------------------------- #
class BacterialVariantType(StrEnum):
    """The kind of sequence change a called bacterial variant is.

    The four shapes the two releases carry, read from their own type columns rather
    than chosen: Lim's ``Mutation Type`` is ``SNP`` / ``DEL`` / ``INS`` / ``SUB``
    (measured: 100 / 47 / 11 / 1 of its 159 rows) and de Siqueira's ``Polymorphism
    Type`` is ``SNP (transition)`` 68, ``SNP (transversion)`` 47, ``Insertion`` 23,
    ``Substitution`` 21, ``Insertion (tandem repeat)`` 5, ``Deletion`` 5, ``Deletion
    (tandem repeat)`` 4. Both vocabularies collapse onto these four, and the released
    string is kept verbatim beside the enum in ``type_statement`` so the transition /
    transversion / tandem-repeat distinction is not lost.

    - ``snv``: one base replaced by one base.
    - ``insertion``: bases gained.
    - ``deletion``: bases lost (an IN-GENE deletion; a deletion that removes whole
      loci is ``SpanDeletionCallPerturbation``, on the presence/absence axis).
    - ``substitution``: a multi-base replacement (Lim's ``2 bp->CG``).
    """

    snv = "snv"
    insertion = "insertion"
    deletion = "deletion"
    substitution = "substitution"


BACTERIAL_VARIANT_TYPE_SO: dict[BacterialVariantType, tuple[str, str]] = {
    BacterialVariantType.snv: ("SO:0001483", "SNV"),
    BacterialVariantType.insertion: ("SO:0000667", "insertion"),
    BacterialVariantType.deletion: ("SO:0000159", "deletion"),
    BacterialVariantType.substitution: ("SO:1000032", "delins"),
}
"""Sequence Ontology mechanism of each variant kind.

Every pair is one the schema ALREADY pins (``SO_ALLOWED`` in
``test_ontology_all_trees.py``), so these leaves introduce no new SO term: a
multi-base replacement is the same ``delins`` a promoter replacement is, and the other
three are the deletion / insertion / SNV terms the yeast leaves use. The leaves keep
the generic ``SO:0001060 sequence_variant`` as their class-default ``mechanism_so_id``
-- true of every call whatever its kind -- and expose the specific pair as a property,
so the mechanism field cannot desync from ``variant_type``.
"""


class VariantCallMode(StrEnum):
    """Whether a frequency is a clone's within-isolate fraction or a population's.

    The distinction is load-bearing and the releases differ on it. de Siqueira and Lim
    sequenced ISOLATED clones, so a frequency below 1 means that clone is not pure at
    that site (Lim: 431 calls at 1 and 12 at 0.9; de Siqueira: 0.96, 1, ...). Choe 2019
    released POPULATION allele frequencies per timepoint, where a frequency is the
    share of the evolving population carrying the allele and no clone genotype follows
    from it without a threshold the authors never set. A consumer that cannot tell the
    two apart would read a 0.5 as half a genome in one case and half a population in
    the other.
    """

    clone = "clone"
    population = "population"


class VariantFrequencyBasis(StrEnum):
    """The scale a released variant frequency is written on, never assumed.

    Lim's matrix is a FRACTION (``1``, ``0.9``) and de Siqueira's Geneious ``Variant
    Frequency`` column is also a fraction (``0.96``, ``1``), while other releases write
    percents. The scale is a property of the release, so it is stated rather than
    inferred from whether a value exceeds 1.
    """

    fraction = "fraction"
    percent = "percent"


class VariantSiteKind(StrEnum):
    """Why a called variant is keyed on its genomic SITE rather than on a locus tag.

    Two measured cases, and the distinction matters because only one of them is a
    statement about the genome's annotation.

    - ``intergenic``: the caller placed the variant between loci and the release says
      so (105 of de Siqueira 2025's 173 calls carry no locus, and 17 of Lim 2025's 159
      read ``intergenic`` with the two flanking loci named).
    - ``locus_not_in_assembly``: the release DID name a locus, but in an identifier
      space the pinned assembly does not carry, so no locus tag of it holds the call.
      Measured: de Siqueira's 10 RefSeq-only calls name ``PP_RS21780`` (9) and
      ``PP_RS19075`` (1), and ``resolve_gene_name`` on the pinned KT2440 GenBank
      assembly (GCA_000007565.2, ASM756v2) returns ``retired``, "not found in
      GCA_000007565.2_ASM756v2; retained as given", for both. Keying such a call to a
      neighboring locus would invent a mapping our own bytes refuse.
    """

    intergenic = "intergenic"
    locus_not_in_assembly = "locus_not_in_assembly"


class BacterialVariantCall(ProvenanceGapMixin):
    """One called variant, exactly as a resequencing release states it.

    Composed onto the three variant leaves below, so the call is defined once. Every
    field is either the released cell verbatim or a typed reading of it; nothing here
    is inferred.

    ``reference_sequence`` is the replicon the coordinates are on, verbatim as the
    release writes it (Lim's ``AE015451``, de Siqueira's ``NC_002947 (2)``). It is
    required because a position is meaningless without it, and because it is what
    exposes a call made against a genome that is NOT the record's own host: Niu 2019
    aligned a BW25113 derivative to MG1655, so its calls mix evolution-acquired
    mutations with strain-background differences, and the only way a consumer can see
    that is the replicon named here disagreeing with the record's
    ``ReferenceGenome.assembly_set``.

    ``position_start`` / ``position_end`` are 1-based and end-inclusive; they are equal
    for a point change and differ for a multi-base one (measured: 53 of de Siqueira's
    173 calls have ``Maximum > Minimum``).

    ``frequency_statement`` is the released cell verbatim and is set or carries a typed
    ``ProvenanceGap`` -- never a silent ``None``. It is kept beside the numeric
    ``frequency`` because a release can write a RANGE there (de Siqueira's multi-base
    calls write ``61 -> 63``) which no float holds; the numeric field is then ``None``
    while the statement keeps what the source said. ``frequency_basis`` is required
    whenever ``frequency`` is set, so a fraction is never read as a percent.

    The frequency being part of the call, and the call part of the perturbation, means
    the same genomic change called at two frequencies in two clones is two perturbation
    identities. That is deliberate: an isolate that is 90% mutant at a site is not
    genotypically the same as one that is 100% mutant. A convergence query joins on
    ``systematic_gene_name`` and ``position_start``, not on the node id.
    """

    variant_type: BacterialVariantType
    type_statement: str = Field(
        description="the release's own type cell, verbatim (e.g. 'SNP (transition)', "
        "'DEL'), kept beside the enum so the finer distinction is not lost"
    )
    reference_sequence: str = Field(
        description="the replicon the coordinates are on, verbatim as released "
        "(e.g. 'AE015451', 'NC_002947 (2)')"
    )
    position_start: int = Field(description="1-based start of the change")
    position_end: int = Field(
        description="1-based, end-INCLUSIVE end of the change; equals position_start "
        "for a point change, and for an INSERTION may be position_start - 1, which is "
        "how a caller writes the zero-length reference interval an insertion sits in"
    )
    sequence_change: str | None = Field(
        default=None,
        description="the released change cell, verbatim (e.g. 'G->A', 'A -> G', '+C', "
        "'2 bp->CG', the delta-bp form of a deletion); None when the release gives the "
        "two alleles in their own columns instead and no fused cell",
    )
    reference_allele: str | None = Field(
        default=None,
        description="the reference base(s), verbatim, when the release gives them in "
        "their own column (de Siqueira writes '--' for an insertion site)",
    )
    alternate_allele: str | None = Field(
        default=None, description="the alternate base(s), on the same condition"
    )
    annotation: str | None = Field(
        default=None,
        description="the release's effect / detail cell, verbatim (e.g. "
        "'L784Q (CTG->CAG)', 'coding (539/1299 nt)', 'intergenic (+140/-75)')",
    )
    amino_acid_change: str | None = Field(
        default=None,
        description="the protein-level change, verbatim, when released in its own "
        "column (e.g. 'Y -> H')",
    )
    codon_change: str | None = Field(
        default=None, description="the codon-level change, verbatim (e.g. 'GGT -> GGC')"
    )
    codon_number: int | None = Field(
        default=None, description="the 1-based codon of the CDS the change sits in"
    )
    call_mode: VariantCallMode = Field(
        description="whether the frequency is a clone's within-isolate fraction or a "
        "population allele frequency"
    )
    frequency_statement: str | None = Field(
        default=None,
        description="the released frequency cell, verbatim; None only with a typed gap "
        "(Niu 2019 leaves one of its 374 rows blank)",
    )
    frequency: float | None = Field(
        default=None,
        description="the frequency as a number, when the release gives ONE value; None "
        "when it gives a range or nothing",
    )
    frequency_basis: VariantFrequencyBasis | None = Field(
        default=None,
        description="the scale `frequency` is on; required whenever frequency is set",
    )
    caller: str | None = Field(
        default=None,
        description="the variant caller and version, verbatim (e.g. 'breseq 0.33.1', "
        "'Geneious Prime'), when the release names it",
    )

    @model_validator(mode="after")
    def _check_call(self) -> "BacterialVariantCall":
        """Ordered 1-based coordinates, a sourced-or-gapped frequency, a stated basis."""
        if self.position_start < 1:
            raise ValueError(f"position_start is 1-based, got {self.position_start}")
        floor = (
            self.position_start - 1
            if self.variant_type is BacterialVariantType.insertion
            else self.position_start
        )
        if self.position_end < floor:
            raise ValueError(
                f"a call needs position_start <= position_end, got "
                f"{self.position_start}..{self.position_end}"
            )
        if not self.reference_sequence.strip():
            raise ValueError("BacterialVariantCall needs a reference_sequence")
        if not self.type_statement.strip():
            raise ValueError("BacterialVariantCall needs a type_statement")
        if self.sequence_change is not None and not self.sequence_change.strip():
            raise ValueError("sequence_change is the released cell or None, not blank")
        if (
            self.sequence_change is None
            and self.reference_allele is None
            and self.alternate_allele is None
        ):
            raise ValueError(
                "a call states WHAT changed: the release's fused change cell "
                "(sequence_change), or an allele column. One SIDE may legitimately be "
                "absent -- an insertion replaces no reference bases, and de Siqueira "
                "2025 writes '--' or an empty cell there -- but all three cannot be"
            )
        _require_value_or_gap(self, ("frequency_statement",))
        if self.frequency is None:
            if self.frequency_basis is not None:
                raise ValueError(
                    "frequency_basis names the scale of a frequency, but frequency is "
                    "None (a range or an absent cell has no scale to state)"
                )
            return self
        if self.frequency_basis is None:
            raise ValueError(
                f"frequency {self.frequency} needs a frequency_basis; a bare number "
                "cannot be read as a fraction or a percent"
            )
        ceiling = (
            1.0 if self.frequency_basis is VariantFrequencyBasis.fraction else 100.0
        )
        if not 0.0 < self.frequency <= ceiling:
            raise ValueError(
                f"frequency {self.frequency} is outside (0, {ceiling}] for basis "
                f"{self.frequency_basis.value}"
            )
        return self

    @property
    def mechanism_so(self) -> tuple[str, str]:
        """(SO id, SO name) of this call's variant kind."""
        return BACTERIAL_VARIANT_TYPE_SO[self.variant_type]

    @property
    def span_length(self) -> int:
        """Reference bases the call covers (1 for a point change, 0 for an insertion).

        Measured: 21 of de Siqueira 2025's 173 calls are insertions released with
        ``Maximum == Minimum - 1`` and ``Length`` 0, which is the zero-length interval
        between two reference bases, so the coordinates are stored as released rather
        than normalized to a one-base span the source never wrote.
        """
        return self.position_end - self.position_start + 1


def _variant_site_id(reference_sequence: str, position_start: int) -> str:
    """The identifier of a genomic SITE that is not a gene: ``<replicon>:<position>``.

    An intergenic call names no locus, so its gene-keyed identity cannot be a locus
    tag, and keying it to a flanking gene would assert the variant is IN that gene --
    exactly the inference the two loaders refused to make. The site id is DERIVED from
    the call (the replicon the release names plus the 1-based position), so it is
    stable, unique per site, and cannot be mistaken for one of the three namespaces'
    locus tags. ``GeneAdditionPerturbation`` already establishes the shape: a
    perturbation whose identifier is legitimately not a host locus tag.
    """
    return f"{reference_sequence}:{position_start}"


VARIANT_SITE_ID_PATTERN = r"^[^\s].*:\d+$"
"""The shape of a site id, so a locus tag can never be stored in one's place."""


class VariantCallPerturbation(SequencePerturbation, ModelStrict):
    """AXIS-3 base of the called-variant leaves: a sequence change the caller FOUND.

    Abstract. ``provenance`` defaults to ``"natural"`` because the motivating case is a
    variant that arose in an evolution experiment; a DESIGNED allele that the leaf also
    serves (Choe 2019 built ``MS56 cspC::cspC(G37A)``) sets ``provenance="engineered"``
    per record, which is what making provenance a free field on the root is for.

    ``mechanism_so_id`` stays the generic ``SO:0001060 sequence_variant``: it is true of
    every call whatever its kind, and the kind-specific pair is read off
    ``call.mechanism_so``, so the two cannot disagree. The host the call is written
    against is the base class's ``gene_namespace`` (no default: a call is a statement
    about one deposited assembly, named by the loader). Before P1 this family carried a
    ``Bacterial`` prefix; a caller's released call is host-neutral.
    """

    call: BacterialVariantCall = Field(
        description="the called variant, exactly as the release states it"
    )
    provenance: str = "natural"

    @property
    def mechanism_so(self) -> tuple[str, str]:
        """(SO id, SO name) of this perturbation's call."""
        return self.call.mechanism_so


class LocusVariantCallPerturbation(
    HashableProvenanceGapMixin, VariantCallPerturbation, ModelStrict
):
    """A called variant that sits INSIDE a named locus of the pinned assembly.

    The 58 de Siqueira calls carrying a GenBank locus tag, the 123 single-locus Lim
    rows, and every designed single-allele strain of the same shape. A partial deletion
    of a locus is one of these with ``call.variant_type == deletion`` and the released
    coding range in ``call.annotation`` (Lim's ``PP_2675`` reads ``coding
    (4-459/462 nt)``), NOT a presence/absence absence: a truncation leaves the locus in
    place, and asserting it is absent would state a consequence the release did not.
    """

    description: str = (
        "Called sequence variant inside a locus, with its replicon, position, change "
        "and call frequency"
    )
    perturbation_type: Literal["locus_variant_call"] = "locus_variant_call"


class SiteVariantCallPerturbation(
    HashableProvenanceGapMixin, VariantCallPerturbation, ModelStrict
):
    """A called variant keyed on its genomic SITE, because no locus tag holds it.

    ``site_kind`` says why, and the two reasons are different claims (see
    ``VariantSiteKind``): the caller placed the variant BETWEEN loci, or the release
    named a locus in an identifier space the pinned assembly does not carry. Together
    they are 122 of de Siqueira 2025's 173 calls and 17 of Lim 2025's 159, so the
    carrier is not an edge case.

    ``systematic_gene_name`` is the derived site id ``<replicon>:<position_start>``
    (checked here against the call's own replicon and position, so it cannot drift from
    the call it names), and ``perturbed_gene_name`` is the release's own label for the
    place: Lim writes the two flanking loci ``PP_4061, PP_4063``, de Siqueira writes
    nothing for an intergenic call and ``PP_RS21780`` for a RefSeq-only one.

    ``flanking_systematic_gene_names`` carries the flanking loci as resolved tags WHEN
    the release names them, in released left-to-right order, so "which gene is this
    variant upstream of" stays answerable without claiming the variant is in either.
    ``released_locus_statement`` carries the unmappable identifier verbatim, and is
    required for ``locus_not_in_assembly`` and forbidden for ``intergenic`` -- an
    intergenic call names no locus to record.
    """

    description: str = (
        "Called sequence variant keyed on its genomic site, because no locus tag of the "
        "pinned assembly holds it"
    )
    perturbation_type: Literal["site_variant_call"] = "site_variant_call"
    site_kind: VariantSiteKind = Field(
        description="why the call is keyed on its site rather than on a locus tag"
    )
    flanking_systematic_gene_names: tuple[str, ...] = Field(
        default=(),
        description="the flanking loci as resolved tags, in released order; empty when "
        "the release names no neighbor",
    )
    flanking_gene_statement: str | None = Field(
        default=None,
        description="the release's flanking-gene cell, verbatim, when it has one",
    )
    released_locus_statement: str | None = Field(
        default=None,
        description="the locus identifier the release named, verbatim, when the pinned "
        "assembly carries no tag for it (e.g. the RefSeq 'PP_RS21780'); set iff "
        "site_kind is locus_not_in_assembly",
    )

    @field_validator("systematic_gene_name", mode="after")
    @classmethod
    def validate_sys_gene_name(cls, v: str) -> str:
        """A site id, never a host identifier (one would claim the variant is in a gene)."""
        if _namespace_of_identifier(v) is not None:
            raise ValueError(
                f"{v!r} is a gene identifier of a host namespace; a site-keyed call is "
                "keyed on its site, and a call inside a locus of the pinned assembly "
                "is a LocusVariantCallPerturbation"
            )
        if not re.match(VARIANT_SITE_ID_PATTERN, v):
            raise ValueError(
                f"invalid variant site id {v!r}; expected '<replicon>:<position>'"
            )
        return v

    @model_validator(mode="after")
    def _check_site(self) -> "SiteVariantCallPerturbation":
        """The site id matches its own call; flanks are tags; the locus rule holds."""
        expected = _variant_site_id(
            self.call.reference_sequence, self.call.position_start
        )
        if self.systematic_gene_name != expected:
            raise ValueError(
                f"site id {self.systematic_gene_name!r} does not name this call's site "
                f"({expected!r})"
            )
        for tag in self.flanking_systematic_gene_names:
            if _namespace_of_identifier(tag) != self.gene_namespace:
                raise ValueError(
                    f"flanking locus {tag!r} is not a {self.gene_namespace} tag"
                )
        unmapped = self.site_kind is VariantSiteKind.locus_not_in_assembly
        if unmapped != (self.released_locus_statement is not None):
            raise ValueError(
                "released_locus_statement is required for site_kind="
                "locus_not_in_assembly and forbidden for site_kind=intergenic, got "
                f"site_kind={self.site_kind.value} with "
                f"released_locus_statement={self.released_locus_statement!r}"
            )
        if unmapped and self.flanking_systematic_gene_names:
            raise ValueError(
                "a call inside a locus the assembly does not carry has no flanking "
                "loci; those belong to an intergenic call"
            )
        return self

    @classmethod
    def site_id(cls, call: BacterialVariantCall) -> str:
        """The site id a call determines, so a loader never spells one by hand."""
        return _variant_site_id(call.reference_sequence, call.position_start)


class SpanDeletionCallPerturbation(
    HashableProvenanceGapMixin, DeletionStrainPerturbation, ModelStrict
):
    """One locus removed ENTIRELY by a called deletion that spans a run of loci.

    The 19 Lim rows whose deletion covers whole loci, up to 53 of them
    (``PP_3024``-``PP_5558``, 40,784 bp), and the single-locus rows of the same shape
    (a deletion the caller reports with no coding offset, which is how breseq says the
    locus lies wholly inside the deleted interval). It stays on the presence/absence
    axis -- these loci ARE absent -- so ``issubclass(_, DeletionPerturbation)`` still
    catches it, and it is ONE perturbation PER COVERED LOCUS rather than one per event:
    a gene-keyed consumer must see all 53 genes as absent, not only an endpoint.

    The event is recoverable as a set from ``span_designation``, which every locus of
    one deletion carries identically (the grouping ``pathway_name`` does for a
    pathway's genes), and ``span_systematic_gene_names`` lists the loci it covers in
    released order.
    """

    description: str = (
        "Locus removed entirely by a called deletion spanning a run of loci"
    )
    perturbation_type: Literal["span_deletion_call"] = "span_deletion_call"  # type: ignore[assignment]
    deletion_type: str = "called_span"
    call: BacterialVariantCall = Field(
        description="the called deletion, exactly as the release states it"
    )
    provenance: str = "natural"
    span_designation: str = Field(
        description="the designation of the deletion EVENT, identical on every locus it "
        "covers (e.g. 'AE015451:4588139 delta5,553 bp'), so the event is one query"
    )
    span_systematic_gene_names: tuple[str, ...] = Field(
        description="every locus the event removes, in released order; this record's "
        "own systematic_gene_name is one of them"
    )
    deleted_span: GenomicSpan | None = Field(
        default=None,
        description="the removed interval on the pinned assembly, when the release "
        "gives both endpoints; None when it gives only a start and a length",
    )

    @model_validator(mode="after")
    def _check_span(self) -> "SpanDeletionCallPerturbation":
        """The event names at least this locus, and the call is a deletion."""
        if self.call.variant_type is not BacterialVariantType.deletion:
            raise ValueError(
                f"a span deletion's call is a deletion, not "
                f"{self.call.variant_type.value}"
            )
        if not self.span_designation.strip():
            raise ValueError("SpanDeletionCallPerturbation needs a span_designation")
        if self.systematic_gene_name not in self.span_systematic_gene_names:
            raise ValueError(
                f"{self.systematic_gene_name} is not among the loci its own event "
                f"removes ({self.span_systematic_gene_names})"
            )
        return self

    @classmethod
    def designation(cls, call: BacterialVariantCall) -> str:
        """The event designation a call determines, so a loader never spells one."""
        return f"{call.reference_sequence}:{call.position_start} {call.sequence_change}"


# --------------------------------------------------------------------------- #
# Round-2 leaves (issues #749, #792, #799), general names since P1.
#
# Two leaves on the same contract as those above: each subclasses the axis leaf its
# OUTCOME already belongs to; the host is the base class's ``gene_namespace``. The
# third leaf of the round, ``BacterialCrisprActivationPerturbation``, differed from
# ``CrisprActivationPerturbation`` by the namespace alone and folded into it in P1.
#
# Each leaf exists because a landed loader MEASURED released rows it could not
# type, and the counts are in the PR body and the dataset notes:
#
# - ``MarkedAllelePerturbation``: Shiver 2016 drops 134 Nichols 2011
#   columns whose strains are tags and marked insertions rather than deletions
#   (#749), and Babu 2014 drops 3,420 of 42,705 released interaction pairs whose
#   donor or recipient is one of Butland 2008's 149 kan-marked, SPA-tagged
#   essential-gene strains (#792). ONE leaf serves both, because the two issues
#   describe the same construct: a selectable cassette integrated at a gene's
#   terminus with the open reading frame intact, with or without an in-frame tag.
#   Splitting them would file one physical collection under two labels.
# - ``DegronPerturbation``: the Nichols ``-DAS`` columns, whose construct
#   asserts regulated PROTEOLYSIS of the product conditional on an adaptor, which a
#   marked allele does not assert (#749).
# - ``CrisprActivationPerturbation`` (folded): Niu 2019's 57-target activation arm,
#   four fifths of that release's numeric cells (#799).
#
# Point mutants and codon substitutions are deliberately NOT here: a called or
# designed single-base allele is issue #731's leaf, and inventing a second one
# would put one kind of record in two classes.
# --------------------------------------------------------------------------- #
class MarkedAllelePerturbation(SequencePerturbation, ModelStrict):
    """A gene carrying a selectable cassette at a terminus, ORF intact.

    The leaf #749 asks for as a tagged-allele leaf and #792 as a hypomorph leaf. It is
    one class because the two releases describe one construct: a selectable marker
    integrated at the gene's own locus, optionally fusing an in-frame peptide tag, with
    the coding sequence otherwise intact. Butland 2008's 149 essential-gene strains are
    simultaneously "kan-marked", "tagged with ... a C-terminal sequential peptide
    affinity tag (SPA)" and "potentially hypomorphic", so a tagged leaf and a hypomorph
    leaf would both claim the same strain set.

    It sits on AXIS 3 (a sequence-level allelic change at the locus) for the same reason
    the yeast ``DampPerturbation`` does: a cassette in the 3' UTR is an edit to the
    gene's own sequence, not a trans-acting effector, and the gene is neither absent nor
    copy-number-changed. The mechanism is SO:0001218 ``transgenic_insertion`` -- what was
    done to the DNA is the insertion of recombinant bases -- which is the same honest
    state/mechanism pairing ``TransposonInsertionPerturbation`` carries, minus its
    ``state="absent"``: here the gene still makes a product.

    ``allele_effect`` is REQUIRED with no default, because it is a claim about the
    strain that only the source can make: a C-terminal tag on a non-essential gene may
    be a purification handle with no growth consequence, while the same cassette on an
    essential gene is how an essential gene is interrogated at all. It is three-valued
    rather than a bool so a silent release is written as ``not_stated`` instead of being
    asserted not to be hypomorphic. There is deliberately NO fold-change or
    ``expression_range`` field: Butland 2008 states the magnitude is unknown ("in the
    majority of cases the nature of the observed hypomorphic defect is unknown"), and
    the yeast DAmP leaf's 4-to-10-fold default would assert a number no bacterial source
    releases. ``tag``, ``terminus``, ``cassette`` and ``insertion_site`` are each
    optional and each stated only when a mirrored source states it, so an untagged
    marked allele and a SPA-tagged one are the same class with the fusion present or
    absent.
    """

    description: str = (
        "Gene with a selectable cassette integrated at a terminus, optionally fusing "
        "an in-frame tag; the open reading frame is intact"
    )
    perturbation_type: Literal["marked_allele"] = "marked_allele"
    mechanism_so_id: str = "SO:0001218"
    mechanism_so_name: str = "transgenic_insertion"
    cassette: str | None = Field(
        default=None,
        description="the selectable cassette integrated at the locus, verbatim (e.g. "
        "'kan'); None when the source does not name it. Optional for the same reason "
        "``DeletionStrainPerturbation.cassette`` is: a gene-perturbation leaf is not "
        "a ``ProvenanceGapMixin`` carrier, and an array whose strain construction lives "
        "in an unmirrored method paper names no marker",
    )
    insertion_site: str | None = Field(
        default=None,
        description='where in the locus the cassette went, verbatim (e.g. "3\'-UTR"); '
        "None when the source states only that the strain is marked",
    )
    tag: str | None = Field(
        default=None,
        description="the in-frame peptide tag fused to the product, verbatim "
        "(e.g. 'SPA'); None when the cassette carries no fusion",
    )
    terminus: Literal["N", "C"] | None = Field(
        default=None,
        description="which end of the product the tag is fused to, when the SOURCE "
        "states it; None when there is no fusion, and None when a release names the tag "
        "but not its end (Shiver's '-SPA' columns, whose strain construction is in an "
        "unmirrored method paper)",
    )
    allele_effect: Literal["hypomorphic", "unaffected", "not_stated"] = Field(
        description="what the SOURCE says the cassette does to the gene's function: "
        "``hypomorphic`` (reduced function or reduced transcript), ``unaffected`` (the "
        "source states the tagged allele behaves as wild type), or ``not_stated``. "
        "Required with no default, and three-valued rather than a bool, because a "
        "gene-perturbation leaf cannot carry a ``ProvenanceGap``: without an explicit "
        "``not_stated`` member, a release that is silent would have to be written as "
        "False, which asserts the allele is NOT hypomorphic"
    )
    collection: str | None = Field(
        default=None,
        description="the physical strain set this allele came from, verbatim (e.g. "
        "'SPA-tagged essential-gene collection'); None when unsourced",
    )
    construction: StrainConstruction | None = Field(
        default=None, description="accession / lab / batch / plate of this strain"
    )

    @field_validator("mechanism_so_id", mode="after")
    @classmethod
    def validate_mechanism_so_id(cls, v: str) -> str:
        """Mechanism SO id is well-formed."""
        return _validate_so_id(v)

    @model_validator(mode="after")
    def _check_fusion(self) -> "MarkedAllelePerturbation":
        """A terminus names a tag."""
        if self.tag is None and self.terminus is not None:
            raise ValueError(
                f"terminus {self.terminus!r} is stated with no tag; a terminus without "
                "a fusion names nothing"
            )
        return self


class DegronPerturbation(SequencePerturbation, ModelStrict):
    """A gene fused to a degron that directs its product to a protease.

    The Nichols 2011 ``-DAS`` columns (#749). It is a SEPARATE leaf from
    ``MarkedAllelePerturbation`` although the DNA edit is the same kind of
    in-frame cassette, because the construct asserts something the marked allele does
    not: the product is targeted for REGULATED PROTEOLYSIS, and in the ssrA/DAS system
    that degradation happens only when the adaptor is present. The adaptor's inducing
    condition is therefore part of what the strain is, and a leaf without a slot for it
    would make an induced depletion indistinguishable from an uninduced control.

    AXIS 3 and SO:0001218 ``transgenic_insertion`` for the same reason the marked-allele
    leaf carries them: the gene is present and its own sequence is what changed. The
    CONSEQUENCE (how much product is left) is a phenotype, never a field here (M1).

    ``degron`` is the one required field, because it is the one thing a release that
    states a degron at all always states. Everything else is stated when the source
    states it: two columns of one locus differing only in their degron ('imp-DAS' and
    'imp-DAS+4') are two strains on the strength of that field alone.
    """

    description: str = "Gene fused to a degron directing its product to a protease"
    perturbation_type: Literal["degron"] = "degron"
    mechanism_so_id: str = "SO:0001218"
    mechanism_so_name: str = "transgenic_insertion"
    degron: str = Field(
        description="the degron fused to the product, verbatim (e.g. 'DAS+4', "
        "'ssrA-DAS')"
    )
    terminus: Literal["N", "C"] | None = Field(
        default=None,
        description="which end of the product the degron is fused to, when the SOURCE "
        "states it. Optional for the same reason the marked-allele leaf's is: Shiver "
        "2016 releases its degron columns as a label suffix ('-DAS', '-DAS+4') and "
        "describes none of those strains' construction, and its deferral target is not "
        "in the literature mirror. A required field with no source would have to be "
        "filled with a guess, which is the exact failure "
        "``BacterialBackgroundAllele.functional`` was measured to have",
    )
    protease: str | None = Field(
        default=None,
        description="the protease the degron directs the product to, verbatim (e.g. "
        "'ClpXP'); None when the source does not name it",
    )
    adaptor: str | None = Field(
        default=None,
        description="the adaptor the degron needs to be recognized, verbatim (e.g. "
        "'SspB'); None when the source does not name it",
    )
    inducing_condition: str | None = Field(
        default=None,
        description="what must be present for the degradation to happen, verbatim "
        "(e.g. 'sspB expressed from an arabinose-inducible promoter'); None when the "
        "construct is constitutive or the source does not state it. The INDUCER "
        "molecule and its dose stay a ``SmallMoleculePerturbation`` on the "
        "environment, as on ``PromoterReplacementPerturbation``, so the compound layer "
        "is not restated here",
    )
    cassette: str | None = Field(
        default=None,
        description="the selectable cassette carrying the degron, verbatim (e.g. "
        "'kan'); None when unsourced",
    )
    collection: str | None = Field(
        default=None,
        description="the physical strain set this allele came from, verbatim; None "
        "when unsourced",
    )
    construction: StrainConstruction | None = Field(
        default=None, description="accession / lab / batch / plate of this strain"
    )

    @field_validator("mechanism_so_id", mode="after")
    @classmethod
    def validate_mechanism_so_id(cls, v: str) -> str:
        """Mechanism SO id is well-formed."""
        return _validate_so_id(v)


# --------------------------------------------------------------------------- #
# End of the round-2 leaves.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# begin ADDITIVE (#991 Vaishnav 2022, #990 Zhang 2020): the promoter as part of the
# genome's content.
#
# A promoter is an edit to the total genomic content of the cell, so it belongs on the
# genotype, not on the phenotype that reads it. Two releases blocked on that and no
# existing leaf holds either:
#
#   - Zhang 2020 re-integrates five genes as one cluster at EasyClone XII-5, each gene
#     driven by one of six promoters (30 promoters, 6^5 designs). The measured variable
#     IS which promoter drives which gene.
#   - Vaishnav 2022 measures an 80 bp variable insert in the promoter of a YFP reporter
#     on a CEN plasmid. The measured variable IS the promoter's sequence.
#
# ``PromoterReplacementPerturbation`` is the bacterial sibling: it asserts that a host
# gene's NATIVE promoter was removed and a characterized part put in its place
# (SO:1000032 ``delins``), and its required ``gene_namespace`` plus locus-tag validator
# refuse ``YKR097W``. Neither release replaces a native promoter: both ADD a gene that
# arrives with its own promoter, so both leaves sit on the gene-addition axis
# (``state="present"``, SO:0000667 ``insertion``) and inherit its relaxed name validator,
# which a heterologous reporter needs.
#
# They are two leaves rather than one because the promoter means a different thing in
# each: a named, characterized part whose sequence the release need not deposit, versus a
# sequence that is itself the measured quantity and is required. A promoter field added to
# ``GeneAdditionPerturbation`` instead would change the meaning of a SERVED class (it is in
# the closure of the metabolite, morphology and product-titer loaders), which the schema
# rules reserve for a written correctness argument; a new leaf moves no served class.
# --------------------------------------------------------------------------- #
def _validate_dna_sequence(value: str, field: str) -> str:
    """A released DNA sequence: non-empty, unambiguous, upper-case ACGT.

    Rejecting a lower-case or IUPAC-ambiguous string is deliberate: a promoter sequence
    is an identity key here (two records differing by one base are two genotypes), so a
    case-folded or degenerate spelling would silently split or merge identities. A
    release that carries ambiguity codes needs its own typed decision, not a cast.
    """
    if not value:
        raise ValueError(f"{field} cannot be empty")
    if set(value) - set("ACGT"):
        raise ValueError(
            f"{field} must be upper-case unambiguous DNA (ACGT), got {value[:32]!r}"
        )
    return value


class PromoterDrivenGeneIntegrationPerturbation(GeneAdditionPerturbation, ModelStrict):
    """A gene integrated at a named locus, driven by the promoter part named here.

    The combinatorial-promoter rows: a gene is re-integrated into the genome (an extra
    copy of a native gene, or a heterologous gene) and the cassette carrying it supplies
    the promoter, so the strain's expression of that gene is set by WHICH promoter was
    chosen. Zhang 2020 varies exactly that, five genes at a time, and the choice is the
    experiment's independent variable; without this leaf the record could say the gene
    was added but not what drives it, and 275 distinct designs would collapse onto one
    genotype.

    ``integration_locus`` is REQUIRED here (the parent leaves it optional, because a
    plasmid-borne addition has none): an integrated promoter-gene pair that does not say
    where it landed cannot be told from a second copy elsewhere, and the landing site is
    part of what sets expression. ``promoter_sequence`` is the part's sequence when the
    release deposits it and ``None`` when it names the part only -- a named part with no
    deposited sequence is the common case, and inventing one from a strain database would
    not be the release's own value.

    The INDUCER of an inducible part is not here: it is a ``SmallMoleculePerturbation``
    on the environment, the one canonical encoding the bacterial
    ``PromoterReplacementPerturbation.is_inducible`` docstring sets out.
    """

    description: str = (
        "Gene integrated at a named locus, driven by the named promoter part"
    )
    perturbation_type: Literal["promoter_driven_gene_integration"] = (
        "promoter_driven_gene_integration"  # type: ignore[assignment]
    )
    promoter_name: str = Field(
        description="the promoter part driving this gene, verbatim (e.g. 'pTEF1')"
    )
    promoter_sequence: str | None = Field(
        default=None,
        description="sequence of the promoter part, when the release deposits it; None "
        "when the release names the part without its sequence",
    )
    integration_locus: str = Field(
        description="the integration site the cassette landed at, verbatim (e.g. "
        "'EasyClone XII-5'); required on this leaf, unlike the parent"
    )
    is_inducible: bool = Field(
        default=False,
        description="True when the part is inducible rather than constitutive. The "
        "INDUCER and its dose are a SmallMoleculePerturbation on the environment, never "
        "restated here.",
    )

    @model_validator(mode="after")
    def _check_promoter(self) -> "PromoterDrivenGeneIntegrationPerturbation":
        """The promoter and the locus are named; a deposited sequence is real DNA."""
        for name in ("promoter_name", "integration_locus"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")
        if self.promoter_sequence is not None:
            _validate_dna_sequence(self.promoter_sequence, "promoter_sequence")
        return self


class VariablePromoterReporterPerturbation(GeneAdditionPerturbation, ModelStrict):
    """A reporter gene whose promoter carries a variable sequence, stored verbatim.

    The massively-parallel-reporter rows: one library member is a reporter construct
    (Vaishnav 2022: YFP plus a constitutive RFP and a URA3 marker on a low-copy CEN
    plasmid) whose promoter holds a variable insert, and that insert's SEQUENCE is the
    independent variable of the experiment. The sequence-level genotype axis is the whole
    point of the leaf, so ``promoter_sequence`` is REQUIRED and carries the released
    bases verbatim; ``variable_region_start`` / ``variable_region_end`` are the 1-based
    inclusive span of the variable part inside it, so the constant scaffold is not
    duplicated per record and the insert stays recoverable.

    ``promoter_variant_id`` is the source's own identifier of the variant (Vaishnav's
    ``origID``, e.g. ``HQ_28``). It is the label a release's design tables join on, and
    two variants with the same sequence under different design labels are distinguishable
    by it. ``scaffold_name`` is the source's name for the constant promoter context the
    insert sits in (``pTpA``), which is part of what the measured level is relative to:
    the same 80-mer in another scaffold is another construct.

    Distinct from ``PromoterDrivenGeneIntegrationPerturbation`` on the two axes that
    matter: there the promoter is a named characterized part whose sequence is optional
    and the gene is integrated at a required locus; here the promoter is a sequence that
    must be present and the construct is normally episomal, so the parent's
    ``localization`` carries the plasmid context and ``integration_locus`` stays None.
    """

    description: str = (
        "Reporter gene driven by a promoter carrying a variable released sequence"
    )
    perturbation_type: Literal["variable_promoter_reporter"] = (
        "variable_promoter_reporter"  # type: ignore[assignment]
    )
    promoter_variant_id: str = Field(
        description="the source's own identifier of this promoter variant, verbatim "
        "(e.g. Vaishnav 2022 origID 'HQ_28')"
    )
    promoter_sequence: str = Field(
        description="the released promoter sequence of this construct, verbatim "
        "(scaffold plus variable insert, as the release writes it)"
    )
    scaffold_name: str = Field(
        description="the source's name for the constant promoter scaffold the variable "
        "region sits in (e.g. 'pTpA')"
    )
    variable_region_start: int | None = Field(
        default=None,
        description="1-based inclusive start of the variable region within "
        "promoter_sequence; None when the release does not define the span",
    )
    variable_region_end: int | None = Field(
        default=None,
        description="1-based inclusive end of the variable region within "
        "promoter_sequence; None when the release does not define the span",
    )

    @model_validator(mode="after")
    def _check_variant(self) -> "VariablePromoterReporterPerturbation":
        """The variant is identified, its sequence is real DNA, and the span fits it."""
        for name in ("promoter_variant_id", "scaffold_name"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")
        _validate_dna_sequence(self.promoter_sequence, "promoter_sequence")
        start, end = self.variable_region_start, self.variable_region_end
        if (start is None) != (end is None):
            raise ValueError(
                "variable_region_start and variable_region_end must both be set or "
                "both be None (half a span does not locate the insert)"
            )
        if start is not None and end is not None:
            if start < 1:
                raise ValueError(f"variable_region_start is 1-based, got {start}")
            if end < start:
                raise ValueError(
                    f"variable_region_end {end} is before variable_region_start {start}"
                )
            if end > len(self.promoter_sequence):
                raise ValueError(
                    f"variable_region_end {end} is past the end of promoter_sequence "
                    f"({len(self.promoter_sequence)} bases)"
                )
        return self


# --------------------------------------------------------------------------- #
# end ADDITIVE (#991, #990)
# --------------------------------------------------------------------------- #


SgaPerturbationType = (
    SgaKanMxDeletionPerturbation
    | SgaNatMxDeletionPerturbation
    | SgaDampPerturbation
    | SgaTsAllelePerturbation
    | SgaSuppressorAllelePerturbation
    | SgaAllelePerturbation
)


# --------------------------------------------------------------------------- #
# P3 (ontology-unify, 2026-10-15): ONE leaf for a gene stored under the identifier its
# source released, with no lookup against any assembly or annotation.
#
# Four leaves asserted this one fact before P3: ``BacterialReleasedStrainPerturbation``
# (#691, Nichols 2011), ``BacterialReleasedCrisprPerturbation`` and
# ``BacterialReleasedTransposonPerturbation`` (#691 round 2) for the bacterial hosts,
# and ``ReleasedGeneNamePerturbation`` (#1045) for yeast. They were held apart by the
# host the loader was written for, the distinction the constitution's first principle
# refuses (owner, 2026-10-15: "like why a bacterial experiment! it seems this is
# unnecessary distinction"), and their field names drifted (``identifier_kind`` and
# ``released_kind`` against ``released_gene_name``; ``cassette`` against ``marker``; a
# required ``expression_direction`` against a ``crispr_interference`` kind member), so a
# consumer asking what the source said it did to a gene had to know which family wrote
# the row. The owner rule of 2026-10-12 is that the data is the data: the loader stores
# what the source released and the genome joins at modeling time. One leaf states that
# once, and nothing above it changes.
# --------------------------------------------------------------------------- #
class ReleasedGeneIdNamespace(StrEnum):
    """The identifier space a source's released gene id is written in."""

    ncbi_entrez_gene = "ncbi_entrez_gene"


class ReleasedKind(StrEnum):
    """What a source says it did to a gene it names only as released.

    The members are the classes a source's own label or Methods state, never a reading
    of the strain's sequence. Nichols 2011 marks its rows with a suffix (none, ``- SPA``,
    ``- DAS``, ``- Linked``, ``- Allele``, ``- Truncation``) and the paper says what each
    set of strains is; a deletion-collection symbol (Cachera 2023), a heterozygous
    column (YeastPhenome), a pooled Tn library's gene-level call (Teteneva 2024) or a
    guide table (Mormino 2022, Niu 2019) states its kind in the Methods.

    - ``deletion``: the gene (or, for a multi-gene label, each listed gene) is deleted.
    - ``heterozygous_deletion``: one allele of a diploid is deleted.
    - ``affinity_tag``: an in-frame affinity tag fused to the product (``- SPA``).
    - ``degron``: a degron fused to the product (``- DAS``).
    - ``linked_marker``: a selectable cassette linked to the wild-type allele, the
      control of an allele series (``- Linked``).
    - ``sequence_allele``: a designed allele the label states as a change (a point
      substitution, an in-frame deletion or a duplication; ``- Allele``).
    - ``truncation``: a truncated gene (``- Truncation``).
    - ``transposon_insertion``: a transposon inserted in the gene's coding region, one
      gene-level call over a pooled library's insertion mutants (Teteneva 2024: "only
      insertions in the coding gene regions, which were assumed to inactivate the
      respective gene function, were considered"; #910), or a mapped insertion whose
      site the release carries (Goodall 2018). The paper's Methods state the class,
      since a pooled release labels each row by its gene alone.
    - ``crispr_interference``: a dead-Cas repressor is guided to the gene; the record
      carries its construct in ``crispr``.
    - ``crispr_activation``: a dead-Cas activator is guided to the gene; the record
      carries its construct in ``crispr``. The direction of a released guide is its
      kind, as it is the class on the placed CRISPR leaves, so there is no second
      field for it.
    """

    deletion = "deletion"
    heterozygous_deletion = "heterozygous_deletion"
    affinity_tag = "affinity_tag"
    degron = "degron"
    linked_marker = "linked_marker"
    sequence_allele = "sequence_allele"
    truncation = "truncation"
    transposon_insertion = "transposon_insertion"
    crispr_interference = "crispr_interference"
    crispr_activation = "crispr_activation"


RELEASED_KIND_SO: dict[ReleasedKind, tuple[str, str]] = {
    ReleasedKind.deletion: ("SO:0000159", "deletion"),
    ReleasedKind.heterozygous_deletion: ("SO:0000159", "deletion"),
    ReleasedKind.affinity_tag: ("SO:0001218", "transgenic_insertion"),
    ReleasedKind.degron: ("SO:0001218", "transgenic_insertion"),
    ReleasedKind.linked_marker: ("SO:0001218", "transgenic_insertion"),
    ReleasedKind.sequence_allele: ("SO:0001060", "sequence_variant"),
    ReleasedKind.truncation: ("SO:0000159", "deletion"),
    ReleasedKind.transposon_insertion: ("SO:0001218", "transgenic_insertion"),
    ReleasedKind.crispr_interference: ("SO:0001998", "sgRNA"),
    ReleasedKind.crispr_activation: ("SO:0001998", "sgRNA"),
}
"""Sequence Ontology mechanism of each released kind, every pair one the schema
already pins: the deletion leaves' ``deletion``, the marked-allele, degron and
transposon leaves' ``transgenic_insertion``, the generic ``sequence_variant`` of the
AXIS-3 root and the expression-modulation leaves' ``sgRNA``. A truncation removes part
of the gene, which is a deletion."""

RELEASED_CRISPR_KINDS: frozenset[ReleasedKind] = frozenset(
    {ReleasedKind.crispr_interference, ReleasedKind.crispr_activation}
)
"""The kinds whose record carries a ``CrisprConstruct``."""

RELEASED_TRANSPOSON_FIELDS: tuple[str, ...] = (
    "barcode",
    "insertion_position",
    "insertion_strand",
    "transposon",
    "library_pool",
)
"""The mapped-insertion fields of :class:`TransposonInsertionPerturbation`, which only a
``transposon_insertion`` record may set."""


class ReleasedIdentifierPerturbation(
    HashableProvenanceGapMixin, GenePerturbation, ModelStrict
):
    """One gene identified only by the identifier its source released.

    ``systematic_gene_name`` is the source's own gene identifier, verbatim (Nichols
    2011: the ECK accession of the row label, ``ECK1963``; Cachera 2023: the ``gene``
    column, ``ADE2``; Goodall 2018: the ``Gene`` symbol, including the nine multi-copy
    insertion-sequence symbols no single BW25113 locus carries), and
    ``perturbed_gene_name`` the source's gene name, verbatim, through the base
    normalization. Nothing is looked up: no assembly or annotation decides whether the
    record exists and no derived identifier is stored, so the record says exactly what
    the release says and the join to a genome is a separate, later act.

    The CLASS is what states that the identifier is unresolved. ``gene_namespace``
    keeps the meaning it has on every gene perturbation, the HOST the study writes its
    strains against (Nichols 2011: "During the entire study, BW25113 is referred to as
    wild-type"; a symbol-named KanMX deletion: the S288C-derived background), and the
    loader states it, since a released identifier proves no namespace. There is
    deliberately no ``released`` member of :class:`GeneIdNamespace`: one value that
    carried both the host and "unresolved" would erase the host.

    ``identifier_kind`` is what the identifier is, as the source writes it (``'ECK
    accession'``, ``'gene symbol'``, ``'JW id'``, ``'gene column entry'``).
    ``strain_label`` is the release's whole strain, row or guide identifier when it
    releases one beyond the gene identifier (``ECK1205/1207/1209-RDLABC``, shared by
    every gene of a multi-gene strain, is what keeps two strains of one gene apart;
    a guide id keeps two guides against one target apart when the spacer alone does
    not), and ``None`` when the gene identifier is the row's whole label.
    ``released_kind`` is what the source says it did to the gene, a value or a typed gap
    (SynLethDB names the gene pair and not the perturbation that made it). A source
    that also releases a gene id stores it verbatim in ``released_gene_id`` with its
    namespace (SynLethDB: the NCBI Entrez id).

    A construct, when the kind has one, is stored as the placed leaf of its technology
    stores it. A ``crispr_interference`` or ``crispr_activation`` record carries its
    ``CrisprConstruct`` in ``crispr``, set exactly for those kinds. A
    ``transposon_insertion`` record may carry the mapped-insertion fields of
    :class:`TransposonInsertionPerturbation` (``barcode``, ``insertion_position``,
    ``insertion_strand``, ``transposon``, ``library_pool``), each verbatim from the
    release and ``None`` when the source releases only gene-level calls; here
    ``insertion_position`` is the coordinate AS RELEASED, on whatever reference the
    source mapped against, since no assembly is consulted. ``cassette`` is the
    selectable marker the source names for the strain (``'kanamycin'``, ``'KanMX'``),
    ``collection`` the strain set it names, ``strain_id`` the background strain it
    names (``'BY4741'``), each verbatim and ``None`` when unstated.

    It is NOT placed under the presence/absence, sequence or expression axis classes,
    because the axis is a property of the kind (a deletion, an affinity tag and a guide
    are all stored here); ``mechanism_so`` gives the Sequence Ontology pair of the kind,
    from :data:`RELEASED_KIND_SO`. One rule governs the name: it is whatever the source
    wrote, non-empty and verbatim. A loader places a record by its form and by what the
    source states: a host identifier whose kind the source states goes on the placed
    leaf of that kind, and every other identifier (a symbol, an accession, a label, or
    a host-form name whose kind is a gap, as SynLethDB's are) comes here. A released
    identifier that does have the form of a host namespace must be the declared host's
    (the base check), so a tag of another strain is never filed under this host.
    """

    description: str = (
        "Gene perturbation identified by the identifier its source released, not "
        "resolved against any assembly or annotation"
    )
    perturbation_type: Literal["released_identifier"] = "released_identifier"
    identifier_kind: str = Field(
        description="what systematic_gene_name is, as the source writes it (e.g. 'ECK "
        "accession', 'gene symbol', 'JW id')"
    )
    strain_label: str | None = Field(
        default=None,
        description="the source's whole strain, row or guide identifier, verbatim, when "
        "it releases one beyond the gene identifier (shared by every gene of a "
        "multi-gene strain); None when the gene identifier is the row's whole label",
    )
    released_kind: ReleasedKind | None = Field(
        description="what the source says it did to the gene; None only with a gap"
    )
    released_gene_id: str | None = Field(
        default=None,
        description="the source's gene identifier, verbatim (SynLethDB: '856882'); "
        "None when the source releases only a name",
    )
    released_gene_id_namespace: ReleasedGeneIdNamespace | None = Field(
        default=None,
        description="the identifier space of released_gene_id; set exactly when it is",
    )
    crispr: CrisprConstruct | None = Field(
        default=None,
        description="the guide + dead-Cas effector of a crispr_interference or "
        "crispr_activation record; None for every other kind",
    )
    barcode: str | None = Field(
        default=None,
        description="the random barcode a transposon_insertion strain is counted by "
        "in a pooled assay; None when the source released no per-strain tag",
    )
    insertion_position: int | None = Field(
        default=None,
        description="1-based position of a transposon insertion as released, on the "
        "reference the source mapped against; None when only gene-level calls were "
        "released",
    )
    insertion_strand: Literal["+", "-"] | None = Field(
        default=None,
        description="orientation of the inserted element as released; None when "
        "unreleased",
    )
    transposon: str | None = Field(
        default=None,
        description="the transposon / delivery vector, verbatim (e.g. 'mariner Himar1', "
        "'Tn5'); None when unsourced",
    )
    library_pool: str | None = Field(
        default=None,
        description="the mutant library sub-pool this strain was screened in, when a "
        "study runs several; a pooled measurement is pool-relative, so the pool joins "
        "the strain identity",
    )
    cassette: str | None = Field(
        default=None,
        description="the selectable marker the source names for the strain, verbatim "
        "(e.g. 'kanamycin', 'KanMX'); None when unstated",
    )
    collection: str | None = Field(
        default=None,
        description="the strain set the source names for this strain, verbatim; None "
        "when the source does not say which of its sets a strain belongs to",
    )
    strain_id: str | None = Field(
        default=None,
        description="the background strain the source names, verbatim (e.g. 'BY4741'); "
        "None when unstated",
    )

    @field_validator("systematic_gene_name", mode="after")
    @classmethod
    def validate_sys_gene_name(cls, v: str) -> str:
        """The released identifier verbatim: whatever the source wrote, non-empty."""
        if not v or v != v.strip():
            raise ValueError(
                f"systematic_gene_name must be the released identifier verbatim, got "
                f"{v!r}"
            )
        return v

    @field_validator("insertion_position", mode="after")
    @classmethod
    def _validate_position(cls, v: int | None) -> int | None:
        """A released position is 1-based, so it is at least 1."""
        if v is not None and v < 1:
            raise ValueError(f"insertion_position is 1-based, got {v}")
        return v

    @model_validator(mode="after")
    def _check_released(self) -> "ReleasedIdentifierPerturbation":
        """The labels are verbatim; an id travels with its namespace; the kind is set or
        gapped; a construct is set exactly for the kinds that have one.
        """
        _require_value_or_gap(self, ("released_kind",))
        for field in ("identifier_kind", "strain_label", "released_gene_id"):
            value = getattr(self, field)
            if value is None and field != "identifier_kind":
                continue
            if not value or value != value.strip():
                raise ValueError(f"{field} must be a non-empty verbatim string")
        if (self.released_gene_id is None) != (self.released_gene_id_namespace is None):
            raise ValueError(
                "released_gene_id and released_gene_id_namespace are set together"
            )
        if (self.released_kind in RELEASED_CRISPR_KINDS) != (self.crispr is not None):
            raise ValueError(
                "crispr is set exactly when released_kind is crispr_interference or "
                "crispr_activation"
            )
        mapped = [
            name
            for name in RELEASED_TRANSPOSON_FIELDS
            if getattr(self, name) is not None
        ]
        if mapped and self.released_kind is not ReleasedKind.transposon_insertion:
            raise ValueError(
                f"{mapped} are set but released_kind is {self.released_kind!r}; the "
                "mapped-insertion fields belong to a transposon_insertion record"
            )
        return self

    @property
    def mechanism_so(self) -> tuple[str, str] | None:
        """(SO id, SO name) of this perturbation's released kind; None when gapped."""
        if self.released_kind is None:
            return None
        return RELEASED_KIND_SO[self.released_kind]


class NamedAllelePerturbation(SequencePerturbation, ModelStrict):
    """An allele of a yeast gene that its source names and does not otherwise describe.

    SGD annotates an inviable null phenotype on an allele it names (``cdc42-111``,
    ``sac1Δ114-503``, ``hht2-L61P``) and the pinned dump carries no sequence change for
    it, so the record stores the allele name verbatim in ``allele_name`` on the gene's
    systematic name. The SO mechanism is the generic ``sequence_variant`` of the axis
    root: the specific edit is not stated. Two alleles of one gene are two
    perturbations, kept apart by the name. Unlike :class:`ConditionalAllelePerturbation`
    this leaf asserts nothing about how the allele acts.
    """

    description: str = "An allele the source names, with no sequence change stated"
    perturbation_type: Literal["named_allele"] = "named_allele"
    gene_namespace: GeneIdNamespace = Field(
        default=SGD_S288C_NAMESPACE,
        description="the host genome; S288C by default because this leaf's name validator pins the S288C systematic form",
    )
    provenance: str = "engineered"
    allele_name: str = Field(
        description="the allele's name, verbatim, e.g. 'cdc42-111'"
    )
    strain_id: str | None = Field(
        default=None,
        description="the background strain the source names, verbatim; None when "
        "unstated",
    )

    @field_validator("allele_name", mode="after")
    @classmethod
    def validate_allele_name(cls, v: str) -> str:
        """The allele name is a non-empty verbatim string."""
        if not v or v != v.strip():
            raise ValueError(
                f"allele_name must be a non-empty verbatim string, got {v!r}"
            )
        return v


# --------------------------------------------------------------------------- #
# End ADDITIVE (#1045).
# --------------------------------------------------------------------------- #


GenePerturbationType = (
    SgaPerturbationType
    | MeanDeletionPerturbation
    | MarkerDeletionPerturbation
    | DeletionPerturbation  # #988: a deletion whose marker the source does not state
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
    | NaturalLossOfFunctionPerturbation  # #987
    | CopyNumberVariantPerturbation
    | EngineeredCopyNumberPerturbation
    | CrisprActivationPerturbation
    | CrisprInterferencePerturbation
    | InduciblePromoterInsertionPerturbation  # Hackett 2020 (additive)
    | PromoterDrivenGeneIntegrationPerturbation  # Zhang 2020 (#990, additive)
    | VariablePromoterReporterPerturbation  # Vaishnav 2022 (#991, additive)
    # leaves written for the bacterial hosts first, general since P1 (ontology-unify)
    | DeletionStrainPerturbation
    | TransposonInsertionPerturbation
    | PromoterReplacementPerturbation
    | HeterologousPathwayPerturbation
    | LocusVariantCallPerturbation
    | SiteVariantCallPerturbation
    | SpanDeletionCallPerturbation
    # round-2 leaves (#749, #792); the CRISPRa leaf of #799 folded into the general one
    | MarkedAllelePerturbation
    | DegronPerturbation
    # a gene under the identifier its source released (P3, ontology-unify: the one leaf
    # that replaced the three bacterial released leaves of #691 and the yeast one of
    # #1045), and an allele under the name its source released (#1045)
    | ReleasedIdentifierPerturbation
    | NamedAllelePerturbation
)


class Genotype(ModelStrict):
    """Collection of gene perturbations defining a strain's genotype.

    ``natural_isolate`` (#988) describes a natural isolate's whole genome where the
    perturbations cannot: its ploidy, zygosity, aneuploidy and manipulation. It is
    ``None`` for every other genotype and is then left out of the dump, so a genotype
    without it serializes exactly as it did before the field existed.
    """

    perturbations: list[GenePerturbationType] = Field(description="Gene perturbation")
    natural_isolate: NaturalIsolateDescriptor | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
        description="the natural isolate's genome-wide annotation (ploidy, zygosity, "
        "aneuploidy, manipulation); None for a genotype that is not a natural isolate",
    )

    @model_validator(mode="after")
    def _one_isolate(self) -> "Genotype":
        """A natural isolate's perturbations that name a strain all name that isolate."""
        if self.natural_isolate is None:
            return self
        isolate = self.natural_isolate.isolate
        others = sorted(
            {
                str(strain)
                for p in self.perturbations
                if (strain := getattr(p, "strain_id", None)) is not None
                and strain != isolate
            }
        )
        if others:
            raise ValueError(
                f"natural isolate {isolate!r} carries perturbations of {others}"
            )
        return self

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

        return (
            set(self.perturbations) == set(other.perturbations)
            and self.natural_isolate == other.natural_isolate
        )


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

    ``percent`` is a percent the source reports WITHOUT stating v/v or w/v; never a
    substitute for ``percent_v_v`` / ``percent_w_v`` when the source states the basis.
    """

    molar = "M"
    millimolar = "mM"
    micromolar = "uM"
    nanomolar = "nM"
    percent_v_v = "percent_v/v"
    percent_w_v = "percent_w/v"
    # no molar conversion exists for a basis-free percent (the basis and a density are
    # not reported), so a consumer computing a log10 molar leaves it None
    percent = "percent"
    ug_per_ml = "ug/mL"
    g_per_l = "g/L"
    ph = "pH"  # dimensionless -log10[H+]; magnitude unit for a PhysicalFactor.ph edit
    # --- begin De Chiara 2022: a nitrogen-equivalent dose ------------------------- #
    # Milligrams of NITROGEN per litre, not of the compound: an alternative nitrogen
    # source dosed so every source supplies the same nitrogen (De Chiara 2022: "the
    # alternative nitrogen source at growth yield limiting concentrations (30 mg N
    # l-1)"). Converting it to a compound mass needs the compound's nitrogen content,
    # which the source does not do, so the dose is stored on the basis it was set on.
    mg_nitrogen_per_l = "mg_N/L"
    # --- end De Chiara 2022 ------------------------------------------------------- #
    # --- begin Niu 2019: a dimensionless ratio to a reference ---------------------- #
    # A product quantity released only as a FOLD CHANGE over the source's own reference
    # (Niu 2019: 40 pinene cells, "the ratio of pinene titer with CRISPRa and without",
    # no absolute titer anywhere in that arm). The number is dimensionless, so no
    # concentration unit is true of it and a molar conversion does not exist; it is
    # stored on the scale the source set it on, with the reference at 1.0.
    fold_change = "fold_change"
    # --- end Niu 2019 ------------------------------------------------------------- #


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
    - ``dissolved_oxygen``: a dissolved-oxygen setpoint the culture is held at
      (``magnitude`` the released value in its released unit, e.g. Public K-12's
      ``aerobic(30% DO)`` as 30 ``percent``), beside ``Environment.aerobicity``, which
      states the regime the setpoint holds (#1045).
    """

    ph = "pH"
    osmolarity = "osmolarity"
    carbon_source = "carbon_source"
    nitrogen_source = "nitrogen_source"
    ionic_strength = "ionic_strength"
    nutrient_dropout = "nutrient_dropout"
    radiation = "radiation"
    # --- begin #1045 part 3 ------------------------------------------------------ #
    dissolved_oxygen = "dissolved_oxygen"
    # --- end #1045 part 3 -------------------------------------------------------- #


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


class Media(ProvenanceGapMixin):
    """Growth medium resolved to typed COMPONENTS (provenance-first).

    ``name`` + ``state`` remain the human label; ``is_synthetic`` (REQUIRED)
    records defined-by-construction vs natural/complex; ``components`` is the
    compositional breakdown; ``dropouts`` are auxotrophic components deliberately
    OMITTED (recorded so 'omitted' is distinguishable from 'not yet listed').
    ``is_fully_characterized`` / ``open_gaps`` make under-definition queryable.

    A medium the source never states is still a ``Media`` (#901, #902, #903):
    ``Environment.media`` stays required, and each of ``name``, ``state`` and
    ``is_synthetic`` is stated or ``None`` beside a ``ProvenanceGap`` naming that
    field, its reason and where we looked (``_stated_or_gapped``). So SynLethDB's
    aggregated pairs, whose release carries no condition field, and an SGD phenotype
    annotation with no condition property store a medium whose gaps say exactly that,
    never a guessed YPD and never a free-text stand-in. A partly stated medium gaps
    only what is missing: SGD's "YPD with 2% glucose" states a name, not a physical
    state. ``media_identity`` keys a gapped medium by its gaps, so an absence never
    merges with a stated medium or with another source's absence.
    """

    # --- begin #901/#902/#903: a typed gap on the medium ------------------------ #
    # The medium's label as the source states it; None ONLY with a
    # ProvenanceGap(field='name') when the source names no medium. Required, no default.
    name: str | None
    # solid, liquid or gas; None ONLY with a ProvenanceGap(field='state') when the
    # source states no physical state. Required, no default.
    state: str | None
    is_synthetic: bool | None = Field(
        description="True = chemically defined by construction (SD/SC/YNB/SGA); "
        "False = natural/complex (YPD, corn steep liquor). Orthogonal to how well "
        "the composition is characterized. None ONLY with a ProvenanceGap"
        "(field='is_synthetic') when the source states no medium to classify."
    )
    # --- end #901/#902/#903 ------------------------------------------------------ #
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
    def validate_state(cls, v: str | None) -> str | None:
        """Validate that a stated state is one of solid, liquid, or gas."""
        if v is not None and v not in ["solid", "liquid", "gas"]:
            raise ValueError('state must be one of "solid", "liquid", or "gas"')
        return v

    # --- begin #901/#902/#903: a typed gap on the medium ------------------------ #
    @model_validator(mode="after")
    def _stated_or_gapped(self) -> "Media":
        """Each label field is stated or carries a typed gap; never a silent ``None``.

        The mixin enforces the converse (a gapped field must be ``None``).
        """
        _require_value_or_gap(self, ("name", "state", "is_synthetic"))
        return self

    @property
    def stated_name(self) -> str:
        """The name the source states; raises for a medium whose name is a typed gap.

        For code that builds or reads a medium it knows to be named (a loader's own
        library medium, an FBA expansion of one): a gapped medium has no label to
        stand in, so it is refused by name rather than spelled ``"None"``.
        """
        if self.name is None:
            raise ValueError(
                "this medium's name is a typed ProvenanceGap; it has no stated name"
            )
        return self.name

    # --- end #901/#902/#903 ------------------------------------------------------ #

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


class PhagePerturbation(EnvironmentPerturbation, ModelStrict):
    """A bacteriophage added to the culture, dosed as a multiplicity of infection.

    A virion is none of the other three environment leaves: it has no InChIKey (so not a
    ``SmallMoleculePerturbation``), it is not a scalar physical factor, and it is a
    replicating nucleoprotein particle rather than a peptide, protein, antibody or toxin
    (so not a ``BiologicPerturbation``, whose ``BiologicAgentClass`` would misstate it).
    Its identity is the phage itself: ``name`` as the source writes it, plus the NCBI
    taxon and the genome accession when the source states them.

    The dose is NOT a ``Concentration``. A multiplicity of infection is a dimensionless
    ratio of two counts (plaque-forming units per cell at the start of the assay), and a
    ``ConcentrationUnit`` member for it would make a particle-to-cell ratio look like a
    dose unit. ``multiplicity_of_infection`` carries that ratio in its own field and
    ``titer_pfu_per_ml`` the phage concentration in the culture, each in the unit its
    name states. A phage challenge always has a dose, so an unstated MOI is a typed
    ``ProvenanceGap``, never a silent ``None``. Whether the strain resisted is an
    ``EnvironmentResponsePhenotype`` property, never part of the edit (M1).
    """

    perturbation_type: Literal["phage"] = "phage"
    description: str = "Bacteriophage added to the culture"
    name: str = Field(
        description="phage name as the source gives it, verbatim (e.g. 'T4', "
        "'lambda cI857')"
    )
    family: str | None = Field(
        default=None,
        description="viral family as the source states it, verbatim (e.g. "
        "'Myoviridae'); None when unstated",
    )
    genome_type: str | None = Field(
        default=None,
        description="nucleic-acid type of the phage genome as stated (e.g. 'dsDNA')",
    )
    ncbi_taxid: int | None = Field(
        default=None,
        description="NCBI Taxonomy id of the phage, when the source names it",
    )
    genome_accession: str | None = Field(
        default=None,
        description="nucleotide accession of the phage genome, when the source names it",
    )
    multiplicity_of_infection: float | None = Field(
        default=None,
        description="the dose, unit MOI: plaque-forming units per cell at the start of "
        "the assay (dimensionless). None only with a ProvenanceGap",
    )
    titer_pfu_per_ml: float | None = Field(
        default=None,
        description="phage in the culture at the start of the assay, unit pfu/mL; None "
        "when unstated",
    )
    host_of_propagation: str | None = Field(
        default=None,
        description="the strain the phage stock was propagated on, verbatim, when stated",
    )

    @field_validator("name", mode="after")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        """A phage name is a non-empty verbatim string."""
        if not v or v != v.strip():
            raise ValueError(f"phage name must be non-empty and verbatim, got {v!r}")
        return v

    @model_validator(mode="after")
    def _check_dose(self) -> "PhagePerturbation":
        """The dose is stated or gapped; every stated count is positive and finite."""
        _require_value_or_gap(self, ("multiplicity_of_infection",))
        for field in ("multiplicity_of_infection", "titer_pfu_per_ml"):
            value = getattr(self, field)
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError(f"PhagePerturbation.{field} must be > 0, got {value}")
        if self.ncbi_taxid is not None and self.ncbi_taxid < 1:
            raise ValueError(f"ncbi_taxid must be positive, got {self.ncbi_taxid}")
        return self


# --------------------------------------------------------------------------- #
# A physical exposure whose DOSE is not a concentration (issue #749 item 3).
#
# Follows ``PhagePerturbation``'s precedent exactly: a dose that is not a
# concentration gets its own named field rather than a new ``ConcentrationUnit``
# member that would make a time look like a dose unit. Additive, a sibling of the
# other environment leaves, and nothing above it is edited.
# --------------------------------------------------------------------------- #
class PhysicalExposurePerturbation(EnvironmentPerturbation, ModelStrict):
    """A timed physical exposure (irradiation, heat shock, ...) dosed by time or fluence.

    ``EnvironmentPhysicalPerturbation`` carries a scalar factor whose magnitude is a
    ``Concentration``, which works for pH and osmolarity and fails for an exposure: a
    UV dose is an irradiance times a time, and when a release states only the time there
    is no concentration to write. Shiver 2016 releases its UV conditions as
    ``UV [12 sec] {4}`` with no irradiance anywhere in the paper or its SI, measured over
    every text-searchable mirrored file, so no fluence exists to compute.

    The fix is the one ``PhagePerturbation`` already made for the multiplicity of
    infection: the dose takes its own field in the unit its name states, instead of a
    ``ConcentrationUnit`` member (``sec``, ``J/m2``) that would make an exposure time
    look like a concentration. ``factor`` reuses the ``PhysicalFactor`` vocabulary so a
    radiation exposure and a qualitative ``EnvironmentPhysicalPerturbation(factor=
    radiation)`` name the same variable.

    An exposure always HAS a dose, so one of the three dose fields is stated or the
    absent one is a typed ``ProvenanceGap``, never a silent ``None``. The three are not
    redundant: a release may state the time alone (Shiver), the fluence alone, or the
    irradiance and the time from which a fluence follows. No cross-field arithmetic is
    asserted, because an irradiance a source reports may be nominal or time-averaged and
    multiplying it would manufacture a fluence the source never released.
    """

    perturbation_type: Literal["physical_exposure"] = "physical_exposure"
    description: str = (
        "Timed physical exposure dosed by duration, irradiance or fluence"
    )
    factor: PhysicalFactor = Field(
        description="the neutral physical variable the culture was exposed to, from the "
        "same vocabulary the scalar physical leaf uses (e.g. ``radiation``)"
    )
    exposure_duration_seconds: float | None = Field(
        default=None,
        description="how long the exposure lasted, unit seconds; None only with a "
        "ProvenanceGap",
    )
    irradiance_w_per_m2: float | None = Field(
        default=None,
        description="the delivered power per area, unit W/m^2, when the source states "
        "it; None only with a ProvenanceGap",
    )
    fluence_j_per_m2: float | None = Field(
        default=None,
        description="the delivered energy per area, unit J/m^2, when the source "
        "releases it; None only with a ProvenanceGap. NOT back-computed from the two "
        "fields above",
    )
    source_description: str | None = Field(
        default=None,
        description="the exposure apparatus and its spectrum as the source states them, "
        "verbatim (e.g. 'germicidal lamp, 254 nm'); None when unstated",
    )

    @model_validator(mode="after")
    def _check_dose(self) -> "PhysicalExposurePerturbation":
        """At least one dose field is stated or gapped, and every stated one is > 0."""
        dose_fields = (
            "exposure_duration_seconds",
            "irradiance_w_per_m2",
            "fluence_j_per_m2",
        )
        gapped = self.gapped_fields()
        if not any(
            getattr(self, field) is not None or field in gapped for field in dose_fields
        ):
            raise ValueError(
                "PhysicalExposurePerturbation states no dose and declares no "
                f"ProvenanceGap on any of {dose_fields}; an exposure always has a dose, "
                "so an unreleased one is a typed gap, never a silent None"
            )
        for field in dose_fields:
            value = getattr(self, field)
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError(
                    f"PhysicalExposurePerturbation.{field} must be > 0, got {value}"
                )
        return self


EnvironmentPerturbationType = (
    SmallMoleculePerturbation
    | EnvironmentPhysicalPerturbation
    | BiologicPerturbation
    | PhagePerturbation
    # a dose that is not a concentration (#749 item 3)
    | PhysicalExposurePerturbation
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
    - ``timed_culture``: grown for a stated time (``PreCulture.duration_hours``) whose
      growth phase the source does not state (De Chiara 2022: a 24 h YPD culture before
      transfer into PBS, a 48 h presporulation culture before sporulation medium).
      Distinct from ``overnight_culture`` because the duration is stated and is not
      overnight.
    """

    frozen_stock = "frozen_stock"
    overnight_culture = "overnight_culture"
    log_phase_culture = "log_phase_culture"
    thaw_recovery = "thaw_recovery"
    timed_culture = "timed_culture"  # De Chiara 2022 (additive)


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


# --- begin #1045 part 3: the growth phase an environment holds a culture in ---- #
class CultureGrowthPhase(StrEnum):
    """The growth phase a batch culture was harvested in, as the source names it.

    Part of the SAMPLE, not of the environment: the medium, temperature and aeration of
    an exponential and a stationary harvest of one culture are the same, and what
    differs is when the cells were taken, which is the ``harvest_optical_density``
    precedent of ``MicroarrayExpressionScorePhenotype``. ``exponential`` is the source's
    "mid-log phase".

    ``Environment.growth_phase`` (#1045) uses the same vocabulary for the other case: a
    culture HELD in a phase as the condition itself, which Schmidt 2016's starved arms
    are ("Starved cells were continuously shaken after reaching stationary phase for
    either 1 or 3 d"). There the phase is the treatment, not the harvest moment.
    """

    exponential = "exponential"
    stationary = "stationary"


# --- end #1045 part 3 --------------------------------------------------------- #


class Environment(ProvenanceGapMixin):
    """Experimental environment: base medium + temperature + optional perturbations.

    ``perturbations`` mirror ``Genotype.perturbations`` on the environment axis: the
    added compounds / physical stresses applied on top of the base ``media``. It
    defaults to an empty list, so an unperturbed environment (every dataset predating
    the environment-perturbation ontology) is unchanged. ``aerobicity`` records the
    oxygen regime (anaerobic fermentation is standard for biofuel screens);
    ``duration_hours`` is the optional treatment time. ``temperature`` is optional: a
    secondary curation layer (YeastPhenome) may not carry it, in which case it is a
    ``ProvenanceGap`` (``field='temperature'``), NOT a guessed value. ``aerobicity`` is
    a value or a typed ``ProvenanceGap`` (#1045): ``None`` only beside a gap naming it,
    when the source states no oxygen regime (Lamoureux 2023's blank ``Electron
    Acceptor`` and ``aerobicity`` cells), never a guessed ``aerobic``; ``transition`` is
    a regime that changes during the culture (Public K-12's anaerobic-aerobic
    transition series), which no single regime states. A dissolved-oxygen setpoint is
    an ``EnvironmentPhysicalPerturbation`` of ``PhysicalFactor.dissolved_oxygen``
    carrying the released value and unit, beside the regime it sets. ``growth_phase``
    is the phase a culture is held in as the condition (Schmidt 2016's stationary
    arms), ``None`` when the source states none.

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
    # --- begin #1045 part 3: aerobicity is a value or a typed gap ---------------- #
    aerobicity: str | None = Field(
        default="aerobic",
        description="'aerobic' | 'anaerobic' | 'microaerobic' | 'transition'; None only "
        "with a ProvenanceGap(field='aerobicity') when the source states no oxygen "
        "regime",
    )
    # --- end #1045 part 3 -------------------------------------------------------- #
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
    # --- begin #753: the chemostat dilution rate --------------------------------- #
    dilution_rate_per_hour: float | None = Field(
        default=None,
        description="the dilution rate of a CONTINUOUS (chemostat) culture, in h^-1; "
        "None for a batch culture. In a chemostat the dilution rate IS the controlled "
        "variable: at steady state it equals the specific growth rate and it sets the "
        "residual concentration of the growth-limiting substrate, so two cultures that "
        "differ only in dilution rate are two different environments and this is part "
        "of the environment identity, as duration_generations is. It is a typed field "
        "rather than free text on a phenotype's measurement_type (Gupta 2024) so it is "
        "queryable, and naming it is what lets a dilution-rate series be loaded at all "
        "(Ishii 2007 drops its wild-type series under culture_not_batch because its "
        "cultures differ from one another ONLY here).",
    )
    # --- end #753 ---------------------------------------------------------------- #
    # --- begin #1045 part 3: the growth phase a culture is held in -------------- #
    growth_phase: CultureGrowthPhase | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
        description="the growth phase the culture is held in as the condition (a "
        "stationary-phase starvation arm); None when the source states none, and "
        "then left out of the dump, so an environment without it serializes exactly "
        "as it did before the field existed (the Genotype.natural_isolate precedent)",
    )
    # --- end #1045 part 3 -------------------------------------------------------- #

    @field_validator("aerobicity", mode="after")
    @classmethod
    def validate_aerobicity(cls, v: str | None) -> str | None:
        """Aerobicity is one of the four oxygen regimes, or ``None`` beside a gap."""
        if v is None:
            return v
        if v not in {"aerobic", "anaerobic", "microaerobic", "transition"}:
            raise ValueError(
                "aerobicity must be aerobic/anaerobic/microaerobic/transition, got "
                f"{v!r}"
            )
        return v

    # --- begin #1045 part 3: aerobicity is a value or a typed gap ---------------- #
    @model_validator(mode="after")
    def _aerobicity_stated_or_gapped(self) -> "Environment":
        """An unstated oxygen regime is a typed gap, never a silent ``None``."""
        _require_value_or_gap(self, ("aerobicity",))
        return self

    # --- end #1045 part 3 -------------------------------------------------------- #

    # --- begin #753: the chemostat dilution rate --------------------------------- #
    @field_validator("dilution_rate_per_hour", mode="after")
    @classmethod
    def validate_dilution_rate(cls, v: float | None) -> float | None:
        """A dilution rate is finite and strictly positive, or absent."""
        if v is None:
            return v
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(
                f"dilution_rate_per_hour must be finite and positive, got {v}"
            )
        return v

    # --- end #753 ---------------------------------------------------------------- #


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
    # One segmented cell of a microscopy screen. The sample unit of a per-strain
    # morphology profile is the cell, not the well: Campos 2018 computes each mean and
    # each CV over the strain's retained segmented cells (291 +/- 116 of them) from a
    # single well, so `biological_replicate` would report 1 for a mean over hundreds of
    # measurements.
    cell = "cell"


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


# --- begin #924: the typed evidence behind a gene-essentiality call ------------ #
class EssentialityEvidenceClass(StrEnum):
    """The PHYSICAL observation a gene-essentiality call rests on (#924).

    The parent level of :class:`EssentialityEvidenceType`: two calls of one class
    observed the same kind of thing, and differ only in how it was read out or scored.
    Calls of different classes are different measurements, and Baba 2006 states the
    difference that matters for a consumer pooling them: a population method can call a
    slow grower essential, a recovery method cannot.

    - ``mutant_recovery``: a defined mutant of each gene is built one gene at a time and
      recovered as a clone, or not (Baba 2006, the Keio construction).
    - ``insertion_depletion``: a pooled random-insertion library is grown and the
      insertions that survive are mapped; a gene without enough of them is called
      essential (Gerdes 2003, Goodall 2018, Price 2018).
    - ``curated_annotation``: a database curator's record of a primary paper's
      observation (SGD's inviable null phenotypes).
    """

    mutant_recovery = "mutant_recovery"
    insertion_depletion = "insertion_depletion"
    curated_annotation = "curated_annotation"


class EssentialityEvidenceType(StrEnum):
    """The readout and rule that turn the observation into the call (#924).

    The leaf level; each member has exactly one parent in
    :data:`ESSENTIALITY_EVIDENCE_CLASS`.

    - ``construction_outcome`` (``mutant_recovery``): targeted replacement, KmR
      colonies checked by PCR; a gene is a candidate essential gene when too few
      colonies have the expected structure (Baba 2006).
    - ``genetic_footprinting`` (``insertion_depletion``): insertions mapped by nested
      PCR after outgrowth in a fermentor; essential when no disruptive insertion is
      recovered (Gerdes 2003).
    - ``tradis_insertion_index`` (``insertion_depletion``): transposon junctions
      sequenced (TraDIS); the insertion index per gene is called by a two-mode
      likelihood ratio (Goodall 2018).
    - ``rb_tnseq_coverage_and_fitness`` (``insertion_depletion``): TnSeq of an RB-TnSeq
      library; essential or important for growth when the gene got no fitness value and
      both normalized densities are under a threshold (Price 2018).
    - ``curated_null_phenotype`` (``curated_annotation``): an SGD phenotype annotation of
      a null allele as inviable.
    - ``tnseq_hmm_state`` (``insertion_depletion``): insertion junctions sequenced
      (Tn-seq); TRANSIT's hidden Markov model labels every TA site and classifies the
      gene into one of four states (Royet 2025 Table S4, #909). A graded call, so it is
      stored on ``GradedGeneEssentialityPhenotype``, never on the boolean leaf.

    A graded call (Royet 2025's HMM states, #909) is a different released quantity, so
    it is a sibling of ``GeneEssentialityPhenotype`` under ``Phenotype`` carrying the
    same descriptor, with its rule added here as a new ``insertion_depletion`` leaf.
    """

    construction_outcome = "construction_outcome"
    genetic_footprinting = "genetic_footprinting"
    tradis_insertion_index = "tradis_insertion_index"
    rb_tnseq_coverage_and_fitness = "rb_tnseq_coverage_and_fitness"
    curated_null_phenotype = "curated_null_phenotype"
    tnseq_hmm_state = "tnseq_hmm_state"  # #909: Royet 2025's TRANSIT HMM states


#: The parent of each evidence type: the DAG of the two enums, checked on every record.
ESSENTIALITY_EVIDENCE_CLASS: dict[
    EssentialityEvidenceType, EssentialityEvidenceClass
] = {
    EssentialityEvidenceType.construction_outcome: (
        EssentialityEvidenceClass.mutant_recovery
    ),
    EssentialityEvidenceType.genetic_footprinting: (
        EssentialityEvidenceClass.insertion_depletion
    ),
    EssentialityEvidenceType.tradis_insertion_index: (
        EssentialityEvidenceClass.insertion_depletion
    ),
    EssentialityEvidenceType.rb_tnseq_coverage_and_fitness: (
        EssentialityEvidenceClass.insertion_depletion
    ),
    EssentialityEvidenceType.curated_null_phenotype: (
        EssentialityEvidenceClass.curated_annotation
    ),
    EssentialityEvidenceType.tnseq_hmm_state: (
        EssentialityEvidenceClass.insertion_depletion
    ),
}


class FootprintingCall(StrEnum):
    """A genetic-footprinting call as Gerdes 2003 Table S1 writes it.

    ``N*`` is an N call whose asterisk the release does not define (accD's call is
    ``N*`` and the page says accD "is called nonessential"). ``X`` ("the gene was not
    covered") and ``?`` ("ambiguous") have no boolean, so ``GeneEssentialityPhenotype``
    refuses them; they are stored on ``GradedGeneEssentialityPhenotype`` (#909).
    """

    essential = "E"
    non_essential = "N"
    non_essential_flagged = "N*"
    not_covered = "X"
    ambiguous = "?"


class TradisCall(StrEnum):
    """A TraDIS call as Goodall 2018 Tables S1 and S4 release it (one-hot columns).

    ``unclear`` (log likelihood ratio within the threshold) has no boolean, so
    ``GeneEssentialityPhenotype`` refuses it, as it refuses the footprinting X and ?;
    ``GradedGeneEssentialityPhenotype`` stores it (#909).
    """

    essential = "essential"
    non_essential = "non_essential"
    unclear = "unclear"


# --- begin #909: the graded essentiality sibling ------------------------------ #
class HmmEssentialityState(StrEnum):
    """A TRANSIT hidden-Markov-model gene state as Royet 2025 Table S4 releases it.

    Table 1 footnote c of the publisher PDF defines the four: "State of each gene in LB
    defined by the TRANSIT software using a Hidden Markov Model: E, essential; GA,
    growth-Advantage; GD, growth-defect; NE, non-essential." Table S4 writes the
    essential state ``ES``, and ``N/A`` for the 13 genes with no TA site (no insertion
    site, so no state; measured on both sheets). Graded, so no member is a boolean:
    these calls are stored on ``GradedGeneEssentialityPhenotype``.
    """

    essential = "ES"
    growth_defect = "GD"
    non_essential = "NE"
    growth_advantage = "GA"
    not_applicable = "N/A"


# --- end #909 ---------------------------------------------------------------- #


#: The boolean each released footprinting call is; None where the call has none.
FOOTPRINTING_CALL_IS_ESSENTIAL: dict[FootprintingCall, bool | None] = {
    FootprintingCall.essential: True,
    FootprintingCall.non_essential: False,
    FootprintingCall.non_essential_flagged: False,
    FootprintingCall.not_covered: None,
    FootprintingCall.ambiguous: None,
}
#: The boolean each released TraDIS call is; None where the call has none.
TRADIS_CALL_IS_ESSENTIAL: dict[TradisCall, bool | None] = {
    TradisCall.essential: True,
    TradisCall.non_essential: False,
    TradisCall.unclear: None,
}

#: The per-record statistics each evidence type may carry. A statistic belongs to the
#: one evidence type whose release defines it, so it is optional by construction and
#: refused on any other type.
ESSENTIALITY_EVIDENCE_FIELDS: dict[EssentialityEvidenceType, tuple[str, ...]] = {
    EssentialityEvidenceType.construction_outcome: (
        "colonies_verified",
        "colonies_tested",
        "percent_verified",
    ),
    EssentialityEvidenceType.genetic_footprinting: (
        "footprinting_call",
        "footprint_insertions_aa",
        "assertion_error",
        "assertion_error_is_upper_bound",
    ),
    EssentialityEvidenceType.tradis_insertion_index: (
        "tradis_call",
        "insertion_index",
        "log_likelihood_ratio",
    ),
    EssentialityEvidenceType.rb_tnseq_coverage_and_fitness: (
        "tnseq_reads",
        "normalized_read_density",
        "central_insertion_positions",
        "normalized_insertion_density",
        "gene_gc_fraction",
        "false_discovery_rate_lower",
        "false_discovery_rate_upper",
    ),
    EssentialityEvidenceType.curated_null_phenotype: ("curated_experiment_type",),
    EssentialityEvidenceType.tnseq_hmm_state: (
        "hmm_state",
        "ta_sites",
        "hmm_sites_essential",
        "hmm_sites_growth_defect",
        "hmm_sites_non_essential",
        "hmm_sites_growth_advantage",
        "mean_insertion_rate",
        "mean_read_count",
    ),
}


class EssentialityStatementRole(StrEnum):
    """The role a sourced statement plays in a gene-essentiality call's evidence.

    - ``scoring_rule``: the rule that turns the observation into the call; its
      ``value`` is the threshold or criterion.
    - ``call_definition``: the source's own words for what its call means; on the
      boolean leaf, what ``is_essential=True`` means (Baba's "candidates for essential
      genes", Price's "essential or important for growth (nearly essential)").
    - ``caveat``: a caveat the source states about its own calls (Baba 2006 on
      population methods, Price 2018's false-discovery rate).
    - ``statistic_definition``: the source's definition of a per-record statistic the
      evidence carries (the colony counts, the assertion error, the insertion index,
      the TnSeq coverage columns).
    """

    scoring_rule = "scoring_rule"
    call_definition = "call_definition"
    caveat = "caveat"
    statistic_definition = "statistic_definition"


class EssentialityStatement(ModelStrict):
    """One verbatim statement of the source about its essentiality call, and its role.

    The quote is ``provenance``, the one name every ``SourcedValue`` field carries, so
    a generic auditor finds it without a per-class table.
    """

    role: EssentialityStatementRole
    provenance: SourcedValue = Field(
        description="the statement, verbatim with its source's sha256"
    )


class GeneEssentialityEvidence(ProvenanceGapMixin):
    """What a gene-essentiality call IS: its evidence type, rule and statistics (#924).

    ``GeneEssentialityPhenotype.is_essential`` means a different thing in every store
    that writes it: a failed construction (Baba 2006), no disruptive insertion after 23
    doublings of a pool (Gerdes 2003), the low mode of a TraDIS insertion index
    (Goodall 2018), no fitness value and low TnSeq coverage (Price 2018), or a curated
    inviable null phenotype (SGD). This descriptor states which, on every record.

    Every record carries the two-level type (:class:`EssentialityEvidenceClass` and its
    :class:`EssentialityEvidenceType` leaf, checked against
    :data:`ESSENTIALITY_EVIDENCE_CLASS`) and the source's own statements about the call,
    each verbatim with sha256 and typed by its :class:`EssentialityStatementRole`:
    exactly one ``scoring_rule`` (the rule that made the call), exactly one
    ``call_definition`` (what a True call means), any ``caveat`` the source states about
    its own calls, and the ``statistic_definition`` of each statistic the record
    carries. Only what the source states about its own method goes here.

    The per-record statistics are OPTIONAL BY CONSTRUCTION, because each belongs to one
    evidence type (:data:`ESSENTIALITY_EVIDENCE_FIELDS`) and is refused on any other:
    ``None`` means "this evidence type does not release it". A reference phenotype
    (the unperturbed parent) carries the type, rule and definition of its dataset and
    no statistics, since no call was made on it. Within its type a statistic is
    value-or-gap where the source defines it for the record (a footprinting E call's
    ``assertion_error``), and the loaders' L2 checks require the rest.

    The descriptor reads no boolean, so it is not tied to ``GeneEssentialityPhenotype``:
    that leaf maps a released call to ``is_essential`` in its own validator and refuses
    a call with no boolean. The graded sibling under the same parent, ``Phenotype``,
    is ``GradedGeneEssentialityPhenotype`` (#909): it carries this same descriptor and
    stores a categorical call as released (Royet 2025's HMM states, type
    ``tnseq_hmm_state``; Goodall's "unclear"; Gerdes's X and ?).
    """

    evidence_class: EssentialityEvidenceClass = Field(
        description="the physical observation the call rests on (the parent level)"
    )
    evidence_type: EssentialityEvidenceType = Field(
        description="the readout and rule that made the call (the leaf level)"
    )
    statements: list[EssentialityStatement] = Field(
        description="the source's own statements about the call, each verbatim with "
        "sha256 and typed by its role: exactly one scoring_rule, exactly one "
        "call_definition, the source's caveats on its own calls, and the definitions "
        "of the per-record statistics"
    )
    # construction_outcome (Baba 2006 Supplementary Table 3)
    colonies_verified: int | None = Field(
        default=None,
        description="construction_outcome: KmR colonies shown by PCR to have the "
        "expected structure",
    )
    colonies_tested: int | None = Field(
        default=None, description="construction_outcome: KmR colonies checked by PCR"
    )
    percent_verified: float | None = Field(
        default=None,
        description="construction_outcome: the released percent of the tested colonies "
        "verified (100 x verified / tested)",
    )
    # genetic_footprinting (Gerdes 2003 Table S1)
    footprinting_call: FootprintingCall | None = Field(
        default=None,
        description="genetic_footprinting: the released call letter (E, N or N* on a "
        "boolean record)",
    )
    footprint_insertions_aa: list[int] | None = Field(
        default=None,
        description="genetic_footprinting: the insertions recovered in the gene after "
        "outgrowth, in amino-acid coordinates from the translational start; empty when "
        "none was recovered",
    )
    assertion_error: float | None = Field(
        default=None,
        description="genetic_footprinting: the probability of missing the ORF by chance "
        "if insertions were random, p_o(L) = exp(-rL); value or typed gap on an E call",
    )
    assertion_error_is_upper_bound: bool | None = Field(
        default=None,
        description="genetic_footprinting: True when the release states the assertion "
        "error as a bound ('< 0.05'), False when it states the value",
    )
    # tradis_insertion_index (Goodall 2018 Tables S1 and S4)
    tradis_call: TradisCall | None = Field(
        default=None,
        description="tradis_insertion_index: the released call (essential or "
        "non_essential on a boolean record)",
    )
    insertion_index: float | None = Field(
        default=None,
        description="tradis_insertion_index: unique insertion sites in the CDS divided "
        "by the CDS length in bases",
    )
    log_likelihood_ratio: float | None = Field(
        default=None,
        description="tradis_insertion_index: log2 likelihood ratio between the two "
        "modes (negative = the essential mode)",
    )
    # rb_tnseq_coverage_and_fitness (Price 2018 Supplementary Table 1)
    tnseq_reads: int | None = Field(
        default=None,
        description="rb_tnseq_coverage_and_fitness: TnSeq reads within the gene (nReads)",
    )
    normalized_read_density: float | None = Field(
        default=None,
        description="rb_tnseq_coverage_and_fitness: reads normalized to a median of 1 "
        "across genes and for GC content (normreads)",
    )
    central_insertion_positions: int | None = Field(
        default=None,
        description="rb_tnseq_coverage_and_fitness: locations in the central 10-90% of "
        "the gene supported by at least 2 TnSeq reads (nPosCentral)",
    )
    normalized_insertion_density: float | None = Field(
        default=None,
        description="rb_tnseq_coverage_and_fitness: density of central insertions "
        "normalized to a median of 1 across genes (dens)",
    )
    gene_gc_fraction: float | None = Field(
        default=None,
        description="rb_tnseq_coverage_and_fitness: the gene's G or C fraction (GC), "
        "the input of the read normalization",
    )
    false_discovery_rate_lower: float | None = Field(
        default=None,
        description="rb_tnseq_coverage_and_fitness: the low end of the source's stated "
        "false-discovery rate of its essential list (a fraction)",
    )
    false_discovery_rate_upper: float | None = Field(
        default=None,
        description="rb_tnseq_coverage_and_fitness: the high end of that rate",
    )
    # curated_null_phenotype (SGD)
    curated_experiment_type: str | None = Field(
        default=None,
        description="curated_null_phenotype: the annotation's SGD experiment type, "
        "verbatim ('classical genetics', 'systematic mutation set')",
    )
    # tnseq_hmm_state (Royet 2025 Table S4, #909)
    hmm_state: HmmEssentialityState | None = Field(
        default=None,
        description="tnseq_hmm_state: the released state classification of the gene",
    )
    ta_sites: int | None = Field(
        default=None,
        description="tnseq_hmm_state: TA sites in the gene (Number of TA sites)",
    )
    hmm_sites_essential: int | None = Field(
        default=None, description="tnseq_hmm_state: TA sites the HMM labeled ES"
    )
    hmm_sites_growth_defect: int | None = Field(
        default=None, description="tnseq_hmm_state: TA sites the HMM labeled GD"
    )
    hmm_sites_non_essential: int | None = Field(
        default=None, description="tnseq_hmm_state: TA sites the HMM labeled NE"
    )
    hmm_sites_growth_advantage: int | None = Field(
        default=None, description="tnseq_hmm_state: TA sites the HMM labeled GA"
    )
    mean_insertion_rate: float | None = Field(
        default=None,
        description="tnseq_hmm_state: mean insertion rate within the gene (a fraction "
        "of its TA sites)",
    )
    mean_read_count: float | None = Field(
        default=None, description="tnseq_hmm_state: mean read count within the gene"
    )

    @model_validator(mode="after")
    def _check_evidence(self) -> "GeneEssentialityEvidence":
        """The class is the type's parent; statistics belong to their own type and are
        internally consistent; a footprinting E call's assertion error is value or gap.
        """
        roles = Counter(statement.role for statement in self.statements)
        for role in (
            EssentialityStatementRole.scoring_rule,
            EssentialityStatementRole.call_definition,
        ):
            if roles[role] != 1:
                raise ValueError(
                    f"the evidence states {roles[role]} {role.value} statements, not 1"
                )
        parent = ESSENTIALITY_EVIDENCE_CLASS[self.evidence_type]
        if self.evidence_class != parent:
            raise ValueError(
                f"evidence_type {self.evidence_type.value} is a "
                f"{parent.value} type, not {self.evidence_class.value}"
            )
        own = set(ESSENTIALITY_EVIDENCE_FIELDS[self.evidence_type])
        gapped = self.gapped_fields()
        for type_, names in ESSENTIALITY_EVIDENCE_FIELDS.items():
            for name in names:
                if name in own:
                    continue
                if getattr(self, name) is not None or name in gapped:
                    raise ValueError(
                        f"{name} belongs to {type_.value} evidence, not "
                        f"{self.evidence_type.value}"
                    )
        colonies = (self.colonies_verified, self.colonies_tested, self.percent_verified)
        if any(v is not None for v in colonies):
            verified, tested, percent = colonies
            if verified is None or tested is None or percent is None:
                raise ValueError(
                    "colonies_verified, colonies_tested and percent_verified are set "
                    "together"
                )
            if tested <= 0 or not 0 <= verified <= tested:
                raise ValueError(f"{verified} of {tested} colonies verified")
            if abs(percent - 100.0 * verified / tested) > 1e-9:
                raise ValueError(
                    f"percent_verified {percent} is not 100 x {verified} / {tested}"
                )
        if (self.assertion_error is None) != (
            self.assertion_error_is_upper_bound is None
        ):
            raise ValueError(
                "assertion_error and assertion_error_is_upper_bound are set together"
            )
        if self.assertion_error is not None and not 0 <= self.assertion_error <= 1:
            raise ValueError(
                f"assertion_error {self.assertion_error} is no probability"
            )
        if (
            self.footprinting_call == FootprintingCall.essential
            and self.assertion_error is None
            and "assertion_error" not in gapped
        ):
            raise ValueError(
                "an E call carries its assertion error or a typed ProvenanceGap"
            )
        if self.footprint_insertions_aa is not None and any(
            p < 0 for p in self.footprint_insertions_aa
        ):
            raise ValueError("footprint_insertions_aa holds a negative position")
        if self.insertion_index is not None and self.insertion_index < 0:
            raise ValueError(f"insertion_index {self.insertion_index} < 0")
        for name in (
            "tnseq_reads",
            "normalized_read_density",
            "central_insertion_positions",
            "normalized_insertion_density",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} {value} < 0")
        # --- begin #909: the HMM site counts are the gene's TA sites, labeled ---- #
        labeled = (
            self.hmm_sites_essential,
            self.hmm_sites_growth_defect,
            self.hmm_sites_non_essential,
            self.hmm_sites_growth_advantage,
        )
        if any(v is not None for v in labeled):
            if any(v is None for v in labeled) or self.ta_sites is None:
                raise ValueError(
                    "ta_sites and the four hmm_sites_* counts are set together"
                )
            counts = [v for v in labeled if v is not None]
            if min(counts) < 0 or sum(counts) != self.ta_sites:
                raise ValueError(
                    f"HMM site labels {counts} are not the gene's {self.ta_sites} "
                    "TA sites"
                )
        if self.hmm_state is not None and self.ta_sites is not None:
            if (self.hmm_state == HmmEssentialityState.not_applicable) != (
                self.ta_sites == 0
            ):
                raise ValueError(
                    f"HMM state {self.hmm_state.value} with {self.ta_sites} TA sites: "
                    "N/A is exactly the gene with no TA site"
                )
        if self.mean_insertion_rate is not None and not (
            0 <= self.mean_insertion_rate <= 1
        ):
            raise ValueError(
                f"mean_insertion_rate {self.mean_insertion_rate} is not a fraction"
            )
        if self.mean_read_count is not None and self.mean_read_count < 0:
            raise ValueError(f"mean_read_count {self.mean_read_count} < 0")
        # --- end #909 ----------------------------------------------------------- #
        if self.gene_gc_fraction is not None and not 0 <= self.gene_gc_fraction <= 1:
            raise ValueError(f"gene_gc_fraction {self.gene_gc_fraction} is no fraction")
        fdr = (self.false_discovery_rate_lower, self.false_discovery_rate_upper)
        if (fdr[0] is None) != (fdr[1] is None):
            raise ValueError(
                "false_discovery_rate_lower and false_discovery_rate_upper are set "
                "together"
            )
        if fdr[0] is not None and fdr[1] is not None and not 0 <= fdr[0] <= fdr[1] <= 1:
            raise ValueError(
                f"false-discovery range {fdr} is not 0 <= low <= high <= 1"
            )
        return self

    def statements_of(self, role: EssentialityStatementRole) -> list[SourcedValue]:
        """The quotes of one role, in the order the evidence lists them."""
        return [s.provenance for s in self.statements if s.role == role]

    @property
    def scoring_rule(self) -> SourcedValue:
        """The one rule that made the call."""
        (rule,) = self.statements_of(EssentialityStatementRole.scoring_rule)
        return rule

    @property
    def call_definition(self) -> SourcedValue:
        """The source's one statement of what a True call means."""
        (definition,) = self.statements_of(EssentialityStatementRole.call_definition)
        return definition


# --- end #924 ---------------------------------------------------------------- #


class GeneEssentialityPhenotype(Phenotype, ModelStrict):
    """Phenotype indicating whether a gene knockout is lethal.

    ``is_essential`` is the release's own call. What that call IS differs by store, so
    every record also carries its ``evidence`` (#924): the evidence class and type, the
    scoring rule and the source's definition of the call, quoted with sha256, and the
    per-record statistics of its type. A released call with no boolean (a footprinting
    X or ?, a TraDIS "unclear") is refused here; it is a graded call, not a boolean.
    """

    graph_level: str = "node"
    label_name: str = "is_essential"
    is_essential: bool = Field(
        default=True, description="gene knockout leading cell death."
    )
    # --- begin #924: the typed evidence behind a gene-essentiality call -------- #
    evidence: GeneEssentialityEvidence = Field(
        description="the evidence class and type, rule, definition, caveats and "
        "per-record statistics of this call (GeneEssentialityEvidence)"
    )

    @model_validator(mode="after")
    def _released_call_is_the_boolean(self) -> "GeneEssentialityPhenotype":
        """A released call letter must be the stored boolean, and must have one."""
        # --- begin #909: a graded evidence type has no boolean ------------------ #
        if self.evidence.evidence_type == EssentialityEvidenceType.tnseq_hmm_state:
            raise ValueError(
                "tnseq_hmm_state evidence calls a graded state, not a boolean; it is "
                "stored on GradedGeneEssentialityPhenotype"
            )
        # --- end #909 ----------------------------------------------------------- #
        released: list[tuple[str, bool | None]] = []
        if self.evidence.footprinting_call is not None:
            call = self.evidence.footprinting_call
            released.append(
                (
                    f"footprinting call {call.value}",
                    FOOTPRINTING_CALL_IS_ESSENTIAL[call],
                )
            )
        if self.evidence.tradis_call is not None:
            call_ = self.evidence.tradis_call
            released.append(
                (f"TraDIS call {call_.value}", TRADIS_CALL_IS_ESSENTIAL[call_])
            )
        for label, boolean in released:
            if boolean is None:
                raise ValueError(
                    f"the {label} has no boolean; a graded call is not an "
                    "is_essential record"
                )
            if boolean != self.is_essential:
                raise ValueError(f"the {label} is not is_essential={self.is_essential}")
        return self

    # --- end #924 -------------------------------------------------------------- #


# --- begin #909: the graded essentiality sibling ------------------------------ #
#: The categorical calls each evidence type releases, as the release spells them, and
#: the descriptor field that carries the typed call. A type with no categorical call
#: (a construction outcome, a coverage rule, a curated annotation) has no graded record.
GRADED_ESSENTIALITY_CALLS: dict[
    EssentialityEvidenceType, tuple[str, frozenset[str]]
] = {
    EssentialityEvidenceType.genetic_footprinting: (
        "footprinting_call",
        frozenset(call.value for call in FootprintingCall),
    ),
    EssentialityEvidenceType.tradis_insertion_index: (
        "tradis_call",
        frozenset(call.value for call in TradisCall),
    ),
    EssentialityEvidenceType.tnseq_hmm_state: (
        "hmm_state",
        frozenset(state.value for state in HmmEssentialityState),
    ),
}


class GradedGeneEssentialityPhenotype(Phenotype, ModelStrict):
    """A categorical essentiality call stored as released (#909).

    The sibling of ``GeneEssentialityPhenotype`` under ``Phenotype``: the same
    evidence (``GeneEssentialityEvidence``, unchanged), and a label that is the
    release's own category rather than a boolean. Royet 2025 Table S4 grades every
    KT2440 gene ES, GD, NE or GA (or N/A without a TA site); Goodall 2018 calls a gene
    "unclear" when its likelihood ratio sits between the two modes; Gerdes 2003 calls
    a gene X (not covered) or ? (ambiguous). None of those is a boolean, and a boolean
    store would either drop them or collapse a grade into essential / not, so they are
    stored here, verbatim.

    ``essentiality_call`` is one of the calls its evidence type releases
    (:data:`GRADED_ESSENTIALITY_CALLS`), and equals the evidence's typed call when the
    record carries one; a reference (the unperturbed parent) states the type's
    non-essential call and carries no statistics.
    """

    graph_level: str = "node"
    label_name: str = "essentiality_call"
    essentiality_call: str = Field(
        description="the release's categorical call, as the release spells it (ES, "
        "GD, NE, GA, N/A; essential, non_essential, unclear; E, N, N*, X, ?)"
    )
    evidence: GeneEssentialityEvidence = Field(
        description="the evidence class and type, the source's statements and the "
        "per-record statistics of this call (GeneEssentialityEvidence)"
    )

    @model_validator(mode="after")
    def _call_is_the_types_category(self) -> "GradedGeneEssentialityPhenotype":
        """The call is one its evidence type releases and agrees with the typed call."""
        type_ = self.evidence.evidence_type
        if type_ not in GRADED_ESSENTIALITY_CALLS:
            raise ValueError(
                f"{type_.value} evidence releases no categorical call; a graded record "
                "needs genetic_footprinting, tradis_insertion_index or tnseq_hmm_state"
            )
        field, calls = GRADED_ESSENTIALITY_CALLS[type_]
        if self.essentiality_call not in calls:
            raise ValueError(
                f"{self.essentiality_call!r} is not a {type_.value} call {sorted(calls)}"
            )
        typed = getattr(self.evidence, field)
        if typed is not None and typed.value != self.essentiality_call:
            raise ValueError(
                f"essentiality_call {self.essentiality_call!r} is not the evidence's "
                f"{field} {typed.value!r}"
            )
        return self


# --- end #909 ---------------------------------------------------------------- #


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
    """Phenotype holding a gene interaction score, its p-value and its replicate design.

    ``gene_interaction_p_value`` is a TEST of the score (the only p-value field in the
    schema); ``n_samples`` + ``sample_unit`` + ``gene_interaction_uncertainty`` + its
    type are the replicate/dispersion quartet the other quantitative phenotypes carry
    (#793), so a sourced replicate design (Babu 2014's eight colonies per pair) is on
    the record rather than beside the build. The two absences read differently and are
    meant to: a gapped p-value is "the source tested the SET, not the pair", a gapped
    ``n_samples`` is "the source states no replicate design".
    """

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

    # ----------------------------------------------------------------------- #
    # ADDITIVE (#793): the replicate-design quartet every other quantitative
    # phenotype carries (FitnessPhenotype, EnvironmentResponsePhenotype,
    # MetabolitePhenotype, ProteinAbundancePhenotype). Nothing above changes.
    # No derived SE field is added: `label_statistic_name` stays
    # `gene_interaction_p_value`, because an interaction score's released
    # statistic is a test of the score, not a dispersion of it.
    # ----------------------------------------------------------------------- #
    n_samples: int | None = Field(
        default=None,
        description="number of independent replicate measurements averaged into this "
        "interaction score (Babu 2014: eight colonies per gene pair, exactly, from two "
        "replicate screens x four biological replicate recipient colonies; Butland 2008: "
        "counted off its own raw colony sheet per record, 4 to 182). None when the "
        "source does not state a replicate design, which is then a typed ProvenanceGap.",
    )
    sample_unit: SampleUnit | None = Field(
        default=None,
        description="what one sample in n_samples is (`colony` for a pinned colony "
        "array, `screen` when the independent unit is the screen rather than the "
        "colony). Required alongside n_samples for a dispersion that divides by n.",
    )
    gene_interaction_uncertainty: float | None = Field(
        default=None,
        description="source-reported uncertainty of the interaction score, verbatim; "
        "its meaning is given by gene_interaction_uncertainty_type.",
    )
    gene_interaction_uncertainty_type: UncertaintyType | None = Field(
        default=None,
        description="what gene_interaction_uncertainty IS (sample_sd, standard_error, "
        "bootstrap_se, variance, ci95), so it converts to an SE correctly. Both or "
        "neither: an unlabelled dispersion is not ingested.",
    )
    # ----------------------------------------------------------------------- #
    # end ADDITIVE (#793)
    # ----------------------------------------------------------------------- #

    @field_validator("gene_interaction")
    def validate_fitness(cls, v: float) -> float:
        """Reject NaN gene interaction values."""
        if math.isnan(v):
            raise ValueError("Gene interaction cannot be NaN")
        return v

    # ----------------------------------------------------------------------- #
    # ADDITIVE (#793): the same two invariants the other phenotypes enforce.
    # ----------------------------------------------------------------------- #
    @field_validator("n_samples")
    def validate_n_samples(cls, v: int | None) -> int | None:
        """n_samples is a positive integer or None."""
        if v is not None and (not isinstance(v, int) or v < 1):
            raise ValueError(f"n_samples must be a positive integer or None, got: {v}")
        return v

    @model_validator(mode="after")
    def _check_uncertainty(self) -> "GeneInteractionPhenotype":
        """A reported dispersion is labelled, and one that divides by n states its n."""
        unc, typ = (
            self.gene_interaction_uncertainty,
            self.gene_interaction_uncertainty_type,
        )
        if (unc is None) != (typ is None):
            raise ValueError(
                "gene_interaction_uncertainty and its type must both be set or both be "
                "None (no unlabelled uncertainty)"
            )
        if typ in (UncertaintyType.sample_sd, UncertaintyType.variance) and (
            self.n_samples is None or self.sample_unit is None
        ):
            raise ValueError(f"n_samples and sample_unit are required for {typ}")
        return self

    # ----------------------------------------------------------------------- #
    # end ADDITIVE (#793)
    # ----------------------------------------------------------------------- #


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


class SourceType(StrEnum):
    """What KIND of document a record's source is.

    A DOI plus a URL is the right identity for a journal article and the wrong one for
    every other kind of source torchcell ingests: a dissertation, a preliminary-exam
    report and an in-house measurement have no DOI and no citable URL, so demanding one
    would either block the record or invite a fabricated identifier. Each non-journal
    source is identified instead by its TITLE plus the deposited document's
    mirror-relative path and sha256, which is the identity the provenance principle
    already treats as canonical.

    - ``journal_article``: a peer-reviewed article identified by DOI and/or PubMed ID.
    - ``dissertation``: a thesis deposited in an institutional repository or in-house.
    - ``preliminary_report``: a preliminary-exam / qualifier report, not published.
    - ``in_house``: a measurement recorded in-house with no document beyond the
      deposited data record itself.
    """

    journal_article = "journal_article"
    dissertation = "dissertation"
    preliminary_report = "preliminary_report"
    in_house = "in_house"


class Publication(ModelStrict):
    """The source document a record's values come from.

    ``source_type`` says what kind of document it is, and the identity rule follows
    from it:

    - ``journal_article`` (the default, so every record written before this field
      existed reads exactly as before) keeps the original rule: a DOI and/or a PubMed
      ID, plus at least one URL.
    - every other ``source_type`` requires ``title`` and ``identifier``, and makes
      doi/pubmed optional, because a dissertation or an in-house report has neither.
      ``identifier`` is the DEPOSITED document: its mirror-relative path plus
      ``sha256:<hex>``, e.g.
      ``"si/prelim-report.pdf sha256:<64 hex>"``. That pair is what the provenance
      principle calls canonical, and it is resolvable from our own mirror without a
      live URL. ``identifier_url`` is optional retrieval metadata, never the identity.
    """

    pubmed_id: str | None = None
    pubmed_url: str | None = None
    doi: str | None = None
    doi_url: str | None = None
    source_type: SourceType = Field(
        default=SourceType.journal_article,
        description="what kind of document the source is; the identity rule follows it",
    )
    title: str | None = Field(
        default=None,
        description="the document's title verbatim; required for a non-journal source, "
        "which has no DOI to name it by",
    )
    identifier: str | None = Field(
        default=None,
        description="for a non-journal source, the deposited document's mirror-relative "
        "path plus 'sha256:<hex>'; the canonical identity when there is no DOI",
    )
    identifier_url: str | None = Field(
        default=None,
        description="optional retrieval metadata for the deposited document; never the "
        "identity (the stored artifact plus its sha256 is)",
    )

    @model_validator(mode="after")
    def check_pub_info(self) -> "Publication":
        """A journal article needs a DOI/PMID and a URL; any other source a title + id."""
        if self.source_type is SourceType.journal_article:
            if self.pubmed_id is None and self.doi is None:
                raise ValueError("At least one of PubMed ID or DOI must be provided")
            if self.pubmed_url is None and self.doi_url is None:
                raise ValueError(
                    "At least one of PubMed URL or DOI URL must be provided"
                )
            return self
        if self.title is None or not self.title.strip():
            raise ValueError(
                f"a {self.source_type.value} source requires a title (it has no DOI to "
                "name it by)"
            )
        if self.identifier is None or not self.identifier.strip():
            raise ValueError(
                f"a {self.source_type.value} source requires an identifier: the "
                "deposited document's mirror-relative path plus 'sha256:<hex>'"
            )
        return self


class ExperimentReference(ModelStrict):
    """Reference (wildtype/control) context for an experiment."""

    experiment_reference_type: str = "base"
    dataset_name: str
    # ``ReferenceGenome | StrainReferenceGenome``, not the base alone: pydantic serializes
    # by the DECLARED type, so a ``StrainReferenceGenome`` in a ``ReferenceGenome`` slot
    # dumped without its background. The union keeps the subclass's fields on the way
    # out and revalidates to the subclass on the way in (the base forbids extras), so no
    # experiment family needs a narrowed clone of its pair to carry a background or the
    # assembly pin. Families that REQUIRE a background still narrow the slot.
    genome_reference: ReferenceGenome | StrainReferenceGenome
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


# --------------------------------------------------------------------------- #
# begin #1022 #989: the two-channel microarray parent
# One physical measurement, released four ways. A two-channel array co-hybridizes a
# sample and a reference, each labeled with its own dye, to the same spots, and every
# value a release carries is derived from that per-spot pair. What the releases CARRY
# differs, so the leaves differ (schema-change rule 5: the same technology at one level,
# diverging in the specifics, shares the parent and diverges in the leaf):
#
# - MicroarrayExpressionPhenotype (served: Kemmeren 2014, both Sameith 2015 sets): log2
#   ratio, its SE, the linear signal and a per-gene replicate count.
# - MicroarrayExpressionScorePhenotype (Hu 2007, GEO GSE4654): an error-model score of
#   the ratio and its p-value, no ratio and no signal.
# - MicroarrayExpressionRatioPhenotype (Gasch 2000): the log2 ratio alone, against a
#   reference the record names.
# - InductionTimecourseExpressionPhenotype (Hackett 2020): every processing layer of a
#   time course, both channels included.
#
# The parent carries NO field: the leaves share no released quantity (even the replicate
# count differs in kind), so any field here would add a required field to the served
# leaf. Placing the served class under it changes only that class's MRO.
# --------------------------------------------------------------------------- #
class TwoChannelMicroarrayExpressionPhenotype(Phenotype, ModelStrict):
    """Abstract parent of every two-channel microarray expression release.

    The physical measurement is one array on which a sample and a reference, labeled
    with two dyes, hybridize to the same spots: each value a release carries (a log
    ratio, its error-model score, a channel intensity) is computed from that per-spot
    pair, so it is relative to the reference the release names. Kemmeren 2014 states
    "a common reference design with wildtype (WT) reference RNA applied in dye-swap to
    each microarray", Sameith 2015 "Dual-channel 70-mer oligonucleotide arrays were
    employed with WT RNA as common reference", Hu 2007 "we used a common reference RNA
    design for all microarray hybridizations", Gasch 2000 "each Cy5-labeled sample was
    compared with a Cy3- labeled reference pool" (the paper's own text, for its heat
    shock series) and Hackett 2020 "RNA from references and samples was labeled with
    Cy3-CTP and Cy5-CTP, respectively".

    Never instantiated: a record is always one of the leaves, because what was released
    decides which fields exist. The parent declares no field of its own, so
    ``Phenotype.validate_label_fields`` (which reads the concrete class's
    ``__annotations__``) refuses any ``label_name`` it is given. The empty
    ``__annotations__`` below is what keeps that true: once any code has read
    ``ABCMeta.__annotations__``, a class with no annotations of its own resolves the
    attribute through its MRO to ``Phenotype``'s, and the parent would then accept
    ``label_name="graph_level"`` (the CPython quirk ``test_ontology_coherence.py``
    records for #640; measured here in CI's full run). Declaring the empty dict in
    the class's own namespace pins the lookup to it.

    The leaves keep their own ``label_name``, because no leaf's quantity converts into
    another's without something its release does not carry: a score back to a ratio
    needs each spot's error, a ratio up to the served leaf needs the channel signal
    and the replicate counts. The one conversion that exists is inside a leaf: a score
    leaf's p-value is a function of its score,
    ``MicroarrayExpressionScorePhenotype.p_value_of_score``.
    """

    __annotations__ = {}  # its OWN (empty) annotations; see the class docstring


# --------------------------------------------------------------------------- #
# end #1022 #989: the two-channel microarray parent
# --------------------------------------------------------------------------- #


class MicroarrayExpressionPhenotype(
    TwoChannelMicroarrayExpressionPhenotype, ModelStrict
):
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


# --------------------------------------------------------------------------- #
# MICROARRAY EXPRESSION SCORE BLOCK START (issue #1022)
# A two-color microarray release that carries a per-gene error-model SCORE (and its
# p-value) in place of a log ratio: a leaf of TwoChannelMicroarrayExpressionPhenotype,
# the sibling of MicroarrayExpressionPhenotype.
# --------------------------------------------------------------------------- #
class MicroarrayExpressionScorePhenotype(
    TwoChannelMicroarrayExpressionPhenotype, ModelStrict
):
    """Per-gene error-model expression SCORE of a strain against a reference, with its
    p-value: what a two-color microarray release carries when it releases no ratio.

    Hu, Killion and Iyer 2007 (Nat Genet 39:683; GEO GSE4654) release every TF-deletion
    transcriptome as the error model's ``X`` score, "a normalized, confidence-weighted
    measure of each ratio", beside a p-value, and nothing else: no log ratio, no
    channel intensity. ``X`` is SIGNED like the ratio it weighs (positive = higher in
    the sample than in the reference) but it is a significance statistic, not a ratio:
    on GSE4654 ``p == 2 * (1 - Phi(|X|))`` to within the release's rounding.

    Why a sibling of ``MicroarrayExpressionPhenotype`` (both are leaves of
    ``TwoChannelMicroarrayExpressionPhenotype``) and not a relaxation of it: that
    class's label ``expression_log2_ratio`` is documented as ``log2(sample/reference)``
    and its linear ``expression`` field is required, so storing a score there would
    assert a log ratio and a signal the release does not carry, and relaxing either
    field would change a served class's meaning for one release. This class holds
    exactly what was released.

    CONVERSION (score to p-value, ``p_value_of_score``): the error model's ``X`` is a
    standard-normal deviate under the null, so its p-value is the two-sided tail,
    ``p == 2 * (1 - Phi(|X|))``. Measured on the GSE4654 bytes for #1022: the identity
    holds on all 1,707,575 released (X, p) pairs to within 5e-7, the rounding of the
    release's six-decimal p column. No conversion runs the other way to a log ratio:
    that needs each spot's error, which the release does not carry.

    Key semantics follow the other dict-valued families: a gene whose spot the release
    left null for this sample is not a key, never a 0. ``expression_score_p_value``
    carries the released p-value key for key, or is None on a reference, whose score is
    the neutral 0 by definition (a reference is not tested against itself).

    ``measurement_type`` names the statistic and its pipeline; ``reference_basis``
    names the denominator in the source's own terms; ``array_platform`` names the
    array (a GEO GPL accession and its title), because a score is only comparable
    within one platform's error model. ``n_replicates`` is the number of independent
    cultures behind the strain's profile (one count per strain: the release does not
    say how many arrays entered each gene's score). ``harvest_optical_density`` is the
    culture density at harvest when the source releases it per strain.
    """

    graph_level: str = "node"
    label_name: str = "expression_score"
    label_statistic_name: str | None = "expression_score_p_value"

    expression_score: dict[str, float] = Field(
        description=(
            "SortedDict of the per-gene error-model score of the sample against the "
            "reference, as released (signed: positive = higher in the sample); a gene "
            "the release left null is not a key"
        ),
        repr=False,
    )
    expression_score_p_value: dict[str, float] | None = Field(
        default=None,
        description=(
            "SortedDict of the released per-gene p-value of the score, a probability "
            "in [0, 1], keyed exactly like expression_score; None on a reference"
        ),
        repr=False,
    )
    measurement_type: str = Field(
        description="what the score is and how it was computed, e.g. "
        "'error_model_x_score_weighted_replicate_average'"
    )
    reference_basis: str = Field(
        description="what the reference channel is, in the source's own terms"
    )
    array_platform: str = Field(
        description="the array the score was measured on, e.g. 'GEO GPL3588: UT Austin "
        "Iyer yeast 6.4k ORF-only cDNA platform'"
    )
    n_replicates: int = Field(
        description="independent cultures profiled for this strain (a strain-level "
        "count; the release does not resolve it per gene)"
    )
    harvest_optical_density: float | None = Field(
        default=None,
        description="culture optical density at harvest as released (averaged over "
        "the replicate cultures when the source says so); None when not released",
    )

    def __repr__(self) -> str:
        """Summary repr instead of dumping the per-gene dicts."""
        n_p = (
            len(self.expression_score_p_value)
            if self.expression_score_p_value is not None
            else 0
        )
        return (
            f"MicroarrayExpressionScorePhenotype(genes={len(self.expression_score)}, "
            f"p_value_genes={n_p}, n_replicates={self.n_replicates})"
        )

    @field_validator("expression_score", mode="before")
    @classmethod
    def convert_and_validate_score(cls, v: Any) -> Any:  # raw pre-validation input
        """Coerce to a SortedDict; refuse an empty map or a non-finite score."""
        if not isinstance(v, dict):
            raise ValueError(
                f"expression_score must be a per-gene dict, got {type(v).__name__}"
            )
        if not v:
            raise ValueError("expression_score cannot be empty")
        if not isinstance(v, SortedDict):
            v = SortedDict(v)
        for key, value in v.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"expression_score for {key} is not a number")
            if not math.isfinite(value):
                raise ValueError(f"expression_score for {key} is not finite: {value}")
        return v

    @field_validator("expression_score_p_value", mode="before")
    @classmethod
    def convert_and_validate_p_value(cls, v: Any) -> Any:  # raw pre-validation input
        """Coerce to a SortedDict; every p-value is a probability in [0, 1]."""
        if v is None:
            return v
        if not isinstance(v, dict):
            raise ValueError(
                "expression_score_p_value must be a per-gene dict, got "
                f"{type(v).__name__}"
            )
        if not isinstance(v, SortedDict):
            v = SortedDict(v)
        for key, value in v.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"expression_score_p_value for {key} is not a number")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"expression_score_p_value for {key} is {value}, not a "
                    "probability in [0, 1]"
                )
        return v

    @field_validator("measurement_type", "reference_basis", "array_platform")
    @classmethod
    def validate_named(cls, v: str) -> str:
        """The statistic, the denominator and the array are named, never blank."""
        if not v.strip():
            raise ValueError(
                "measurement_type, reference_basis and array_platform cannot be blank"
            )
        return v

    @field_validator("n_replicates", mode="after")
    @classmethod
    def validate_n_replicates(cls, v: int) -> int:
        """At least one culture stands behind every profile."""
        if v < 1:
            raise ValueError(f"n_replicates must be a positive integer, got {v}")
        return v

    @field_validator("harvest_optical_density", mode="after")
    @classmethod
    def validate_harvest_optical_density(cls, v: float | None) -> float | None:
        """A released density is finite and positive, or absent."""
        if v is not None and (not math.isfinite(v) or v <= 0.0):
            raise ValueError(
                f"harvest_optical_density must be finite and positive, got {v}"
            )
        return v

    @staticmethod
    def p_value_of_score(score: float) -> float:
        """The two-sided standard-normal p-value of an error-model score.

        ``2 * (1 - Phi(|score|))``, written as ``erfc(|score| / sqrt(2))`` so the tail
        keeps its precision where ``1 - Phi`` would round to 0. This is the conversion
        the released p-values obey (class docstring); a release whose p-values do not
        is not an error-model score of this kind.
        """
        if isinstance(score, bool) or not math.isfinite(score):
            raise ValueError(f"score must be a finite number, got {score!r}")
        return math.erfc(abs(score) / math.sqrt(2.0))

    @model_validator(mode="after")
    def validate_p_value_keys(self) -> "MicroarrayExpressionScorePhenotype":
        """A stored p-value map tests exactly the genes the score map carries."""
        p_values = self.expression_score_p_value
        if p_values is not None and set(p_values) != set(self.expression_score):
            raise ValueError(
                "expression_score_p_value must have the same keys as expression_score"
            )
        return self


class MicroarrayExpressionScoreExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a microarray expression-score experiment."""

    experiment_reference_type: str = "microarray_expression_score"
    phenotype_reference: MicroarrayExpressionScorePhenotype


class MicroarrayExpressionScoreExperiment(Experiment, ModelStrict):
    """Experiment measuring a microarray expression-score phenotype."""

    experiment_type: str = "microarray_expression_score"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: MicroarrayExpressionScorePhenotype


# --------------------------------------------------------------------------- #
# MICROARRAY EXPRESSION SCORE BLOCK END (issue #1022)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# begin #989: the ratio-only microarray leaf
# A two-channel release that carries the per-gene log2 ratio against a named reference
# and nothing else: no channel signal, no SE, no error-model score. The third leaf of
# TwoChannelMicroarrayExpressionPhenotype (Gasch 2000 releases 173 such arrays).
# --------------------------------------------------------------------------- #
class MicroarrayExpressionRatioPhenotype(
    TwoChannelMicroarrayExpressionPhenotype, ModelStrict
):
    """Per-gene log2 ratio of a sample against a named reference, as a two-channel
    array release carries it when it carries nothing else.

    Gasch et al. 2000 (Mol Biol Cell 11:4241) release every array as the ScanAlyze
    log2 ratio of the sample (Cy5) to the reference (Cy3): "The normalized, relative
    transcript levels (R/G ratios) are represented in log2 space (RAT2N)" (the authors'
    data page). No channel intensity, standard error or replicate count is released
    beside it, so ``MicroarrayExpressionPhenotype`` cannot hold the release: its linear
    ``expression`` and per-gene ``n_replicates`` are required, and making them optional
    would change a served class's contract for one source.

    ``expression_log2_ratio`` is ``log2(sample/reference)`` in the same orientation as
    the served leaf (positive = higher in the sample); a loader whose source uses the
    other orientation negates it. A gene whose spot the release left null is not a key,
    never a 0. A reference's own record is the neutral 0 for every key it carries.

    ``reference_basis`` names the denominator in the source's own terms, because a
    ratio means nothing without it and a release mixes kinds (the same culture before a
    shift, a culture grown at another temperature, a strain carrying the empty vector).
    ``measurement_type`` names the statistic and its pipeline; ``array_platform`` names
    the array. ``n_replicates`` is the per-gene count of measurements behind each ratio,
    or None with a typed gap when the release does not state it (Gasch: one column per
    array, and no statement of how many spots per gene each column averages).
    """

    graph_level: str = "node"
    label_name: str = "expression_log2_ratio"
    label_statistic_name: str | None = None

    expression_log2_ratio: dict[str, float] = Field(
        description=(
            "SortedDict of the per-gene log2(sample/reference) as released; a gene the "
            "release left null is not a key"
        ),
        repr=False,
    )
    measurement_type: str = Field(
        description="what the ratio is and how it was computed, e.g. "
        "'scanalyze_rat2n_normalized_log2_ratio'"
    )
    reference_basis: str = Field(
        description="what the reference channel is, in the source's own terms"
    )
    array_platform: str = Field(
        description="the array the ratio was measured on, as the source names it"
    )
    n_replicates: dict[str, int] | None = Field(
        default=None,
        description="SortedDict of the measurements behind each gene's ratio, keyed "
        "exactly like expression_log2_ratio; None only with a typed gap",
        repr=False,
    )

    def __repr__(self) -> str:
        """Summary repr instead of dumping the per-gene dicts."""
        n_rep = len(self.n_replicates) if self.n_replicates is not None else 0
        return (
            f"MicroarrayExpressionRatioPhenotype(genes={len(self.expression_log2_ratio)}"
            f", n_replicates_genes={n_rep})"
        )

    @field_validator("expression_log2_ratio", mode="before")
    @classmethod
    def convert_and_validate_ratio(cls, v: Any) -> Any:  # raw pre-validation input
        """Coerce to a SortedDict; refuse an empty map or a non-finite ratio."""
        if not isinstance(v, dict):
            raise ValueError(
                f"expression_log2_ratio must be a per-gene dict, got {type(v).__name__}"
            )
        if not v:
            raise ValueError("expression_log2_ratio cannot be empty")
        if not isinstance(v, SortedDict):
            v = SortedDict(v)
        for key, value in v.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"expression_log2_ratio for {key} is not a number")
            if not math.isfinite(value):
                raise ValueError(
                    f"expression_log2_ratio for {key} is not finite: {value}"
                )
        return v

    @field_validator("n_replicates", mode="before")
    @classmethod
    def convert_and_validate_n_replicates(
        cls, v: Any
    ) -> Any:  # raw pre-validation input
        """Coerce to a SortedDict of positive integer counts, or keep None."""
        if v is None:
            return v
        if not isinstance(v, dict):
            raise ValueError(
                f"n_replicates must be a per-gene dict, got {type(v).__name__}"
            )
        if not v:
            raise ValueError("n_replicates cannot be empty; use None with a gap")
        if not isinstance(v, SortedDict):
            v = SortedDict(v)
        for key, value in v.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"n_replicates for {key} must be a positive integer, got {value!r}"
                )
        return v

    @field_validator("measurement_type", "reference_basis", "array_platform")
    @classmethod
    def validate_named(cls, v: str) -> str:
        """The statistic, the denominator and the array are named, never blank."""
        if not v.strip():
            raise ValueError(
                "measurement_type, reference_basis and array_platform cannot be blank"
            )
        return v

    @model_validator(mode="after")
    def validate_counts(self) -> "MicroarrayExpressionRatioPhenotype":
        """Counts are sourced or gapped, and keyed exactly like the ratios."""
        _require_value_or_gap(self, ("n_replicates",))
        counts = self.n_replicates
        if counts is not None and set(counts) != set(self.expression_log2_ratio):
            raise ValueError(
                "n_replicates must have the same keys as expression_log2_ratio"
            )
        return self


class MicroarrayExpressionRatioExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a ratio-only microarray experiment."""

    experiment_reference_type: str = "microarray_expression_ratio"
    phenotype_reference: MicroarrayExpressionRatioPhenotype


class MicroarrayExpressionRatioExperiment(Experiment, ModelStrict):
    """Experiment measuring a ratio-only microarray phenotype."""

    experiment_type: str = "microarray_expression_ratio"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: MicroarrayExpressionRatioPhenotype


# --------------------------------------------------------------------------- #
# end #989: the ratio-only microarray leaf
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# begin ADDITIVE (Hackett 2020): one array of an induction time course.
#
# A two-colour microarray time course after a transcription factor is induced in a
# chemostat. Three NEW classes, siblings of the microarray family rather than
# subclasses of it: the phenotype carries layers ``MicroarrayExpressionPhenotype``
# has no field for, and a subclass would make the served ``microarray expression
# phenotype`` node class ambiguous (its discriminating properties would also be fields
# of the subclass, ``ontology_checks.phenotype_label_map``). The phenotype therefore
# names its primary layer ``log2_cleaned`` and has its own graph class, ``induction
# timecourse expression phenotype``. No served dataset's schema closure moves
# through these classes; the new ``GenePerturbationType`` member is the only change a
# served closure sees. Since #1022 #989 the phenotype is a leaf of
# ``TwoChannelMicroarrayExpressionPhenotype``, beside the served microarray leaf rather
# than under it (the same two-channel measurement, a different release).
#
# - The reference narrows ``genome_reference`` to ``StrainReferenceGenome`` (the typed
#   background, which carries the synthetic activator cassette) and
#   ``environment_reference`` to ``CultureEnvironment`` (the chemostat), as
#   ``StrainEnvironmentResponseExperimentReference`` does for #507: pydantic v2
#   serializes a field by its DECLARED type, so a subclass in a base slot loses fields.
# - The sampling time after induction is ``Environment.duration_hours``, the field the
#   served Brunk 2016 time course already uses for "hours post-induction"; no time
#   field is added.
# --------------------------------------------------------------------------- #
class InductionTimecourseExpressionPhenotype(
    TwoChannelMicroarrayExpressionPhenotype, ModelStrict
):
    """One array of an induction time course, with every released processing layer.

    PRIMARY LAYER (``label_name``): ``log2_cleaned``, the cleaned log2 ratio relative
    to the same culture at t = 0 (0 at t = 0 by construction), with the induction
    system's common response and the prominent outliers removed by the authors. Its
    statistic (``label_statistic_name``) is ``log2_noise_sd``, the authors' noise-model
    standard deviation of each cleaned observation; one array per culture and time
    point, so it is the standard error of the single observation. ``None`` with a typed
    gap when the release carries no noise model for the array.

    FURTHER LAYERS (all keyed by the same genes as ``log2_cleaned``):

    - ``sample_channel_intensity`` / ``reference_channel_intensity``: the median
      fluorescence of the sample and of the common reference RNA hybridized against it
      on the same array (linear scale).
    - ``log2_ratio_unprocessed``: log2(sample / reference channel) minus the same
      gene's value at t = 0, before cleaning.
    - ``log2_thresholded``, ``log2_selected``, ``log2_shrunken``: the hard-thresholded,
      timecourse-selected and observation-level-shrunken layers. A timecourse the
      authors' noise model does not keep is exactly 0 in them: an analysis decision,
      not a measured zero. ``None`` with a typed gap when not released.

    IDENTITY: ``timecourse_id`` is the release's key of the time course (strain, date,
    limitation, induction system), ``strain_label`` the release's strain token,
    ``experiment_date`` the release's date token and ``release_table`` the released
    table the values were read from, all verbatim. Repeated time courses of one strain
    stay distinct records through them.
    """

    graph_level: str = "node"
    label_name: str = "log2_cleaned"
    label_statistic_name: str | None = "log2_noise_sd"

    log2_cleaned: dict[str, float] = Field(
        description="cleaned log2 ratio per gene, relative to the same culture at t = 0",
        repr=False,
    )
    log2_noise_sd: dict[str, float] | None = Field(
        description="authors' noise-model standard deviation of each cleaned value; "
        "None only with a typed gap",
        repr=False,
    )
    sample_channel_intensity: dict[str, float] = Field(
        description="sample-channel median fluorescence per gene (linear scale)",
        repr=False,
    )
    reference_channel_intensity: dict[str, float] = Field(
        description="reference-channel median fluorescence per gene (linear scale)",
        repr=False,
    )
    log2_ratio_unprocessed: dict[str, float] = Field(
        description="log2(sample / reference channel) minus the gene's t = 0 value, "
        "before cleaning",
        repr=False,
    )
    log2_thresholded: dict[str, float] | None = Field(
        description="cleaned values hard-thresholded by the noise model; None only "
        "with a typed gap",
        repr=False,
    )
    log2_selected: dict[str, float] | None = Field(
        description="thresholded values kept only in timecourses passing the "
        "biological-feasibility selection; None only with a typed gap",
        repr=False,
    )
    log2_shrunken: dict[str, float] | None = Field(
        description="selected values with observation-level shrinkage toward 0; None "
        "only with a typed gap",
        repr=False,
    )
    timecourse_id: str = Field(
        description="the release's key of the time course this array belongs to, "
        "verbatim"
    )
    strain_label: str = Field(
        description="the strain token as the release writes it (e.g. 'SMY2266a')"
    )
    experiment_date: str | None = Field(
        description="the release's date token for the time course, verbatim; None "
        "only with a typed gap"
    )
    release_table: str = Field(
        description="the released table the values were read from, verbatim file name"
    )

    def __repr__(self) -> str:
        """Summarize instead of printing ~50,000 numbers."""
        return (
            f"InductionTimecourseExpressionPhenotype(timecourse_id="
            f"{self.timecourse_id!r}, genes={len(self.log2_cleaned)})"
        )

    @field_validator(
        "log2_cleaned",
        "log2_noise_sd",
        "sample_channel_intensity",
        "reference_channel_intensity",
        "log2_ratio_unprocessed",
        "log2_thresholded",
        "log2_selected",
        "log2_shrunken",
        mode="before",
    )
    @classmethod
    def _sorted_finite_layer(cls, v: Any) -> Any:  # raw pre-validation input
        """Coerce a layer to a SortedDict and refuse an empty or non-finite one."""
        if v is None:
            return v
        if isinstance(v, dict) and not isinstance(v, SortedDict):
            v = SortedDict(v)
        if not v:
            raise ValueError("an expression layer cannot be empty")
        for key, value in v.items():
            if not math.isfinite(value):
                raise ValueError(f"non-finite value for gene {key}: {value}")
        return v

    @model_validator(mode="after")
    def _check_layers(self) -> "InductionTimecourseExpressionPhenotype":
        """One gene set across layers; intensities and noise non-negative; set or gapped."""
        keys = set(self.log2_cleaned)
        layers = {
            "log2_noise_sd": self.log2_noise_sd,
            "sample_channel_intensity": self.sample_channel_intensity,
            "reference_channel_intensity": self.reference_channel_intensity,
            "log2_ratio_unprocessed": self.log2_ratio_unprocessed,
            "log2_thresholded": self.log2_thresholded,
            "log2_selected": self.log2_selected,
            "log2_shrunken": self.log2_shrunken,
        }
        for name, layer in layers.items():
            if layer is not None and set(layer) != keys:
                raise ValueError(f"{name} must have the genes of log2_cleaned")
        for name in (
            "log2_noise_sd",
            "sample_channel_intensity",
            "reference_channel_intensity",
        ):
            layer = layers[name]
            negative = [] if layer is None else [g for g, v in layer.items() if v < 0]
            if negative:
                raise ValueError(f"{name} cannot be negative, got it at {negative[:5]}")
        _require_value_or_gap(
            self,
            (
                "experiment_date",
                "log2_noise_sd",
                "log2_thresholded",
                "log2_selected",
                "log2_shrunken",
            ),
        )
        for name in ("timecourse_id", "strain_label", "release_table"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")
        return self


class InductionTimecourseExpressionExperimentReference(
    ExperimentReference, ModelStrict
):
    """The same culture at t = 0, on its typed strain background, in its chemostat."""

    experiment_reference_type: str = "induction_timecourse_expression"
    genome_reference: StrainReferenceGenome
    environment_reference: CultureEnvironment
    phenotype_reference: InductionTimecourseExpressionPhenotype


class InductionTimecourseExpressionExperiment(Experiment, ModelStrict):
    """One array of an induction time course (genotype, chemostat at a time, layers)."""

    experiment_type: str = "induction_timecourse_expression"
    environment: CultureEnvironment
    # Declared here: torchcell.data reads ``__annotations__["phenotype"]``, which holds
    # only a class's OWN annotations.
    phenotype: InductionTimecourseExpressionPhenotype


# --------------------------------------------------------------------------- #
# end ADDITIVE (Hackett 2020)
# --------------------------------------------------------------------------- #


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
    # ADDITIVE (#964, Guzman 2018): value-or-gap. A release that publishes TPM and no
    # read counts (GEO GSE117303's ``deseq2-tpm.csv``) stores its TPM with
    # ``expression_count`` None beside a ProvenanceGap on it; every other record keeps
    # its counts, so no stored record changes meaning.
    expression_count: dict[str, int] | None = Field(
        description=(
            "SortedDict of per-gene raw mapped-read counts; same keys as expression_tpm. "
            "None ONLY with a ProvenanceGap on 'expression_count' (a release with no "
            "counts)"
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
            # #964: a typed gap; validate_matching_keys requires the gap.
            return v
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
        if self.expression_count is None:
            if "expression_count" not in self.gapped_fields():
                raise ValueError(
                    "expression_count is None without a ProvenanceGap on it (a release "
                    "with no counts is a typed gap, never a silent None)"
                )
            return self
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


# --- begin #770: the protein fold-change family -------------------------------- #
class FoldChangeScale(StrEnum):
    """The scale a relative measurement is RELEASED on, which fixes its neutral value.

    A fold change is dimensionless, so the number alone does not say whether ``0.5``
    means halved (linear) or a four-fold increase (log2). The scale is therefore a
    required, typed part of the measurement rather than a word in a free-text field:
    without it, a linear ratio and a log2 ratio of the same contrast average together
    into nothing. ``neutral_value`` is the value of no change on each scale, which is
    also the value a reference phenotype carries by definition.
    """

    linear = "linear"
    log2 = "log2"
    log10 = "log10"

    @property
    def neutral_value(self) -> float:
        """The value of "no change" on this scale: 1.0 linear, 0.0 on a log scale."""
        return 1.0 if self is FoldChangeScale.linear else 0.0


class ProteinFoldChangePhenotype(Phenotype, ModelStrict):
    """Per-protein RELATIVE abundance: the ratio of a strain's proteome to a reference.

    The sibling of ``ProteinAbundancePhenotype`` on the relative axis, and not a mode of
    it: that class's docstring is explicit that its number is an "absolute per-strain
    quantity on a log signal scale, NOT a ratio", because an absolute level and a ratio
    answer different questions and must never pool. A differential-proteomics release
    ships a fold change with a test result beside it, which this class carries and the
    absolute class has no field for (``gene_interaction_p_value`` was the only p-value
    anywhere in the schema).

    ``fold_change_scale`` fixes what the number means and what "no change" is, so the
    reference phenotype is the neutral value by definition rather than a measurement.
    ``reference_basis`` names the DENOMINATOR, which one paper can vary within itself
    (Carruthers 2025's Source Data carries ``POI:Control`` beside ``dCas9:Control`` on
    one strain), so the pair (scale, basis) is what makes two columns comparable.
    ``protein_fold_change_p_value`` and its BH-adjusted companion are per protein, with
    ``p_value_adjustment_method`` naming the correction, because an unadjusted and an
    adjusted p-value of one contrast are different numbers.

    Keyed by the protein's systematic identifier in the record's namespace, with the
    same ragged-key honesty as the other dict-valued families: a protein the contrast
    did not test is simply not a key, never a 1.0 and never a 0. On the LINEAR scale a
    stored ``0.0`` is therefore a measurement and not a placeholder: the numerator fell
    below detection while the denominator did not, which is a complete loss of signal
    rather than an absent value. Yunus 2026 releases the verbatim cell ``0`` for 37 of
    its 102 single-target rows and counts them inside its own "51 of which were
    downregulated by more than 95 %" census, so refusing zero would refuse a third of
    that paper. A log scale cannot express it (``log2 0`` is not finite), which is one
    more reason the scale is a required field.
    """

    graph_level: str = "node"
    label_name: str = "protein_fold_change"
    label_statistic_name: str | None = "protein_fold_change_se"

    protein_fold_change: dict[str, float] = Field(
        description="protein systematic id -> fold change on fold_change_scale "
        "(SIGNED on a log scale; strictly positive on the linear scale)"
    )
    fold_change_scale: FoldChangeScale = Field(
        description="the scale the released number is on, which fixes its neutral value"
    )
    reference_basis: str = Field(
        description="what the denominator is, in the source's own terms, e.g. "
        "'control strain carrying a non-targeting sgRNA' or 'protein of interest over "
        "the same protein in the no-guide control'"
    )
    protein_fold_change_se: dict[str, float] | None = Field(
        default=None,
        description="protein systematic id -> standard error of the fold change on "
        "fold_change_scale",
    )
    protein_fold_change_p_value: dict[str, float] | None = Field(
        default=None,
        description="protein systematic id -> the UNADJUSTED p-value of the contrast's "
        "test for that protein, as a probability in [0, 1]. A release that publishes "
        "-log10(p) is converted by exact arithmetic (p = 10**-x), which is reversible; "
        "the test itself is named by measurement_type",
    )
    protein_fold_change_p_value_adjusted: dict[str, float] | None = Field(
        default=None,
        description="protein systematic id -> the multiple-testing-adjusted p-value, as "
        "a probability in [0, 1]; p_value_adjustment_method names the correction",
    )
    p_value_adjustment_method: str | None = Field(
        default=None,
        description="the multiple-testing correction behind "
        "protein_fold_change_p_value_adjusted, e.g. 'benjamini_hochberg'. Required "
        "whenever adjusted p-values are stored and forbidden otherwise",
    )
    n_replicates: dict[str, int] = Field(
        description="protein systematic id -> number of independent samples/replicates "
        "behind that protein's fold change"
    )
    measurement_type: str = Field(
        description="what the number is and how it was tested, e.g. "
        "'dia_top3_ratio_to_control' or "
        "'dia_log2_fold_change_paired_two_tailed_t_test'"
    )

    @model_validator(mode="after")
    def validate_protein_fold_change(self) -> "ProteinFoldChangePhenotype":
        """Non-empty, scale-consistent values, matching replicate keys, valid p-values."""
        if not self.protein_fold_change:
            raise ValueError("protein_fold_change cannot be empty")
        if set(self.n_replicates) != set(self.protein_fold_change):
            raise ValueError("n_replicates keys must match protein_fold_change keys")
        for key, n in self.n_replicates.items():
            if n < 1:
                raise ValueError(f"n_replicates for {key} must be >= 1")
        linear = self.fold_change_scale is FoldChangeScale.linear
        for key, value in self.protein_fold_change.items():
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"protein_fold_change for {key} must be finite")
            if linear and value < 0.0:
                raise ValueError(
                    f"protein_fold_change for {key} is {value}, which is not a ratio; "
                    "a linear fold change is non-negative (a log scale is signed). "
                    "Zero is a measured numerator below detection, not a missing key"
                )
        if self.protein_fold_change_se is not None:
            for key, se in self.protein_fold_change_se.items():
                if key not in self.protein_fold_change:
                    raise ValueError(f"SE key {key} not in protein_fold_change")
                if not math.isnan(se) and se < 0:
                    raise ValueError(f"SE for {key} must be non-negative")
        for name in (
            "protein_fold_change_p_value",
            "protein_fold_change_p_value_adjusted",
        ):
            mapping = getattr(self, name)
            if mapping is None:
                continue
            for key, p in mapping.items():
                if key not in self.protein_fold_change:
                    raise ValueError(f"{name} key {key} not in protein_fold_change")
                if math.isnan(p):
                    continue
                if math.isinf(p) or not 0.0 <= p <= 1.0:
                    raise ValueError(
                        f"{name} for {key} is {p}, not a probability in [0, 1]"
                    )
        adjusted = self.protein_fold_change_p_value_adjusted is not None
        if adjusted and self.p_value_adjustment_method is None:
            raise ValueError(
                "adjusted p-values name their correction: set p_value_adjustment_method"
            )
        if not adjusted and self.p_value_adjustment_method is not None:
            raise ValueError(
                "p_value_adjustment_method describes stored adjusted p-values; set "
                "protein_fold_change_p_value_adjusted or leave the method None"
            )
        return self

    def neutral_reference(self) -> dict[str, float]:
        """The denominator's own value on this record's scale, for every stored key.

        Not a measurement: a fold change's reference is its neutral value by
        definition (1.0 linear, 0.0 on a log scale), so experiment over reference
        reproduces the released number exactly and nothing is imputed.
        """
        neutral = self.fold_change_scale.neutral_value
        return dict.fromkeys(sorted(self.protein_fold_change), neutral)


class ProteinFoldChangeExperimentReference(ExperimentReference, ModelStrict):
    """Reference (control) context for a protein fold-change experiment."""

    experiment_reference_type: str = "protein_fold_change"
    phenotype_reference: ProteinFoldChangePhenotype


class ProteinFoldChangeExperiment(Experiment, ModelStrict):
    """Experiment measuring a per-protein fold change against a reference strain."""

    experiment_type: str = "protein_fold_change"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: ProteinFoldChangePhenotype


# --- end #770 ------------------------------------------------------------------- #


# --- begin #921: the phosphosite intensity family ------------------------------ #
# A mass-spectrometry proteome measurement AT A RESIDUE: the precursor signal of the
# phosphopeptides that carry one modified site, per sample. A SIBLING of
# ``ProteinAbundancePhenotype`` under ``Phenotype`` (the schema has no proteome parent
# class; inserting one would change the bases of the served protein-abundance leaf),
# not a mode of it: a site intensity moves with the site's occupancy AND with its
# protein's abundance, so keying it on a protein in the abundance leaf would put two
# quantities under one label (#921). Potel 2018 (E. coli, PXD008369) is the first
# consumer; Leutert 2023 (yeast) releases the same per-sample shape beside a
# differential one, which would be a further sibling (a ratio is a meaning change).
class PhosphoResidue(StrEnum):
    """The phosphorylated amino acid of a site, by its one-letter code.

    S, T and Y carry an O-phosphate. H carries an N-phosphate, the residue Potel 2018
    exists to measure: its first author's dissertation, Chapter V (the reprint of the
    paper), "high ΔG of hydrolysis of the phosphoramidate bond makes the
    phosphohistidine relatively" unstable, which is why acidic enrichment had missed it
    (``torchcell-raw/potelWidespreadBacterialProtein2018/paper/thesis/
    Potel.pdftotext.txt``, sha256 ``05a50657...``). The member value is the letter the
    release writes in its ``Amino acid`` column.
    """

    serine = "S"
    threonine = "T"
    tyrosine = "Y"
    histidine = "H"

    @property
    def is_phosphohistidine(self) -> bool:
        """True for the N-phosphorylated histidine, False for the three O-phosphates."""
        return self is PhosphoResidue.histidine


class ProteinIdNamespace(StrEnum):
    """The identifier space a phosphosite's protein ids are written in.

    One member today, the UniProtKB accession a proteomics search database keys on
    (Potel 2018 searched the "reviewed E. coli K12 database (Uniprot, March 2016)").
    A release keyed on another namespace (a yeast systematic ORF) is a new member with
    its own pattern, so the ids of two namespaces never compare as one.
    """

    uniprotkb_accession = "uniprotkb_accession"


PROTEIN_ID_PATTERNS: dict[ProteinIdNamespace, str] = {
    ProteinIdNamespace.uniprotkb_accession: UNIPROT_ACCESSION_PATTERN
}
"""The pattern every protein id of each namespace must match."""


class PhosphositeQuantification(StrEnum):
    """Whether a released site carries a precursor intensity in one sample.

    ``not_quantified`` is the released ``0`` of a MaxQuant intensity column: the site
    was identified in the experiment but no precursor signal was assigned to it in this
    sample. It is a status and never a value, because a zero intensity is not a
    measurement of zero phosphorylation, and never a dropped key, because the site's
    identification belongs to the record whether or not this sample quantified it.
    """

    quantified = "quantified"
    not_quantified = "not_quantified"


# ``CultureGrowthPhase`` is defined above ``Environment``, which carries it (#1045).


PHOSPHOSITE_KEY_SEPARATOR = "|"
"""Separates the three parts of a phosphosite key: proteins, residue, positions."""
PHOSPHOSITE_LIST_SEPARATOR = ";"
"""Joins the members of a multi-protein site, the release's own ``;``."""


class Phosphosite(ModelStrict):
    """One phosphosite as its release identifies it, with the evidence for the call.

    A site the search could not assign to one protein is NOT split: Potel 2018 lists 42
    of its 2,129 sites on more than one gene (``pflB;tdcE;grcA`` at ``734;738;101``;
    ``tufA;tufB`` at ``313;313`` under the one accession ``P0CE47``), and the record
    keeps the released lists in the released order. ``positions`` pairs one 1-based
    position with each gene name (the Table 2 legend: "Gene names, the gene name of
    proteins" and "Positions, the position of identified phosphosite within the protein
    associated with"), while ``protein_ids`` may be shorter, since one accession can
    stand for identical proteins of two genes.

    The identification fields are the release's per-site evidence, constant across its
    samples: the localization probability, and the posterior error probability, search
    score and delta score of the best phosphopeptide carrying the site, with whether a
    diagnostic immonium ion was seen. ``sequence_windows`` are the residues around the
    site (``_`` pads a protein terminus), one per distinct protein sequence, and each
    must center on ``residue``.
    """

    protein_ids: list[str] = Field(
        description="the protein ids the release assigns the site to, in its order"
    )
    protein_id_namespace: ProteinIdNamespace = Field(
        description="the identifier space of protein_ids"
    )
    gene_names: list[str] = Field(
        description="the gene names the release lists for the site, in its order"
    )
    positions: list[int] = Field(
        description="1-based residue position of the site in each gene's protein, "
        "paired with gene_names"
    )
    residue: PhosphoResidue = Field(description="the phosphorylated amino acid")
    sequence_windows: list[str] = Field(
        description="the released sequence window(s) centered on the site, one per "
        "distinct protein sequence; '_' pads a terminus"
    )
    localization_probability: float = Field(
        description="probability the phosphate sits on this residue, in [0, 1]"
    )
    posterior_error_probability: float = Field(
        description="posterior error probability of the best phosphopeptide carrying "
        "the site, in [0, 1]"
    )
    identification_score: float = Field(
        description="search-engine score of that best phosphopeptide (Andromeda for a "
        "MaxQuant release)"
    )
    delta_score: float = Field(
        description="score difference to the second-best peptide with a different "
        "sequence"
    )
    diagnostic_peak: bool = Field(
        description="whether a diagnostic immonium ion was observed in the spectra "
        "identifying the site"
    )

    @model_validator(mode="after")
    def _check_site(self) -> "Phosphosite":
        """Ids in their namespace, paired positions, centered windows, probabilities."""
        if not self.protein_ids:
            raise ValueError("a phosphosite names at least one protein id")
        pattern = PROTEIN_ID_PATTERNS[self.protein_id_namespace]
        for protein_id in self.protein_ids:
            if not re.match(pattern, protein_id):
                raise ValueError(
                    f"{protein_id!r} is not a {self.protein_id_namespace} id"
                )
        if not self.gene_names or any(not name for name in self.gene_names):
            raise ValueError("a phosphosite names at least one non-empty gene name")
        if len(self.positions) != len(self.gene_names):
            raise ValueError(
                f"{len(self.positions)} positions for {len(self.gene_names)} gene "
                "names; the release pairs one position with each gene"
            )
        if any(position < 1 for position in self.positions):
            raise ValueError(f"positions are 1-based, got {self.positions}")
        if not self.sequence_windows:
            raise ValueError("a phosphosite carries at least one sequence window")
        for window in self.sequence_windows:
            if len(window) % 2 == 0:
                raise ValueError(f"window {window!r} has no center residue")
            if window[len(window) // 2] != self.residue.value:
                raise ValueError(
                    f"window {window!r} does not center on {self.residue.value}"
                )
        for name in ("localization_probability", "posterior_error_probability"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} is {value}, not a probability in [0, 1]")
        for name in ("identification_score", "delta_score"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        return self

    def key(self) -> str:
        """The site's record key: released proteins, residue and positions, verbatim.

        ``P00350|S|86`` for a one-protein site and ``P09373;P42632;P68066|S|734;738;101``
        for a site the release lists on three, so a multi-protein site is one key and
        never three guessed ones.
        """
        return PHOSPHOSITE_KEY_SEPARATOR.join(
            (
                PHOSPHOSITE_LIST_SEPARATOR.join(self.protein_ids),
                self.residue.value,
                PHOSPHOSITE_LIST_SEPARATOR.join(str(p) for p in self.positions),
            )
        )


class PhosphositeIntensityPhenotype(Phenotype, ModelStrict):
    """Per-site phosphopeptide precursor intensity in ONE sample, as released.

    One record is one sample (one enrichment and injection). ``sites`` holds every site
    the release identifies, keyed by :meth:`Phosphosite.key`; ``quantification`` says
    for each of them whether this sample assigned it an intensity; and
    ``phosphosite_intensity`` holds the released number for exactly the quantified
    ones. A released ``0`` is therefore ``not_quantified`` and absent from the intensity
    map, never a stored zero and never a dropped site. The number is the instrument's
    summed precursor signal on its own arbitrary scale (``measurement_type`` names the
    acquisition and the software), so two samples compare only within one release.

    The sample is described where it is measured: ``sample_label`` is the release's own
    name for the column, ``replicate`` its replicate number, ``replicate_unit`` whether
    that replicate is a biological or a technical one, and ``harvest_growth_phase`` /
    ``harvest_optical_density`` when the culture was taken. Each of the last three is a
    value or a typed ``ProvenanceGap``: Potel 2018 calls its twelve samples "n=12
    biological replicates" in two SI legends while its reporting summary and the
    dissertation reprint call each condition's three "technical triplicates", so the
    unit is a gap until the published Online Methods are read (#922), not a pick.
    """

    graph_level: str = "node"
    label_name: str = "phosphosite_intensity"
    label_statistic_name: str | None = None

    phosphosite_intensity: dict[str, float] = Field(
        description="site key -> the released precursor intensity, for the sites this "
        "sample quantified; strictly positive"
    )
    quantification: dict[str, PhosphositeQuantification] = Field(
        description="site key -> whether this sample quantified the site, for every "
        "site the release identifies"
    )
    sites: dict[str, Phosphosite] = Field(
        description="site key -> the site's proteins, residue, positions and "
        "identification evidence"
    )
    measurement_type: str = Field(
        description="the acquisition and quantification, e.g. "
        "'fe3_imac_lc_msms_maxquant_summed_xic'"
    )
    sample_label: str = Field(
        description="the release's own name for this sample's column, verbatim"
    )
    replicate: int = Field(description="the release's replicate number, 1-based")
    replicate_unit: SampleUnit | None = Field(
        default=None,
        description="what one replicate is (biological or technical); None only with "
        "a ProvenanceGap",
    )
    harvest_growth_phase: CultureGrowthPhase | None = Field(
        default=None,
        description="the growth phase the culture was harvested in; None only with a "
        "ProvenanceGap",
    )
    harvest_optical_density: float | None = Field(
        default=None,
        description="OD600 of the culture at harvest, as the source states it; None "
        "only with a ProvenanceGap",
    )

    @model_validator(mode="after")
    def validate_phosphosite_intensity(self) -> "PhosphositeIntensityPhenotype":
        """Keys agree across the three maps; intensities positive; sample fields set."""
        if not self.sites:
            raise ValueError("sites cannot be empty")
        for key, site in self.sites.items():
            if site.key() != key:
                raise ValueError(
                    f"site key {key!r} is not its site's key {site.key()!r}"
                )
        if set(self.quantification) != set(self.sites):
            raise ValueError("quantification keys must match sites keys")
        quantified = {
            key
            for key, status in self.quantification.items()
            if status is PhosphositeQuantification.quantified
        }
        if set(self.phosphosite_intensity) != quantified:
            raise ValueError(
                "phosphosite_intensity keys must be exactly the quantified sites"
            )
        for key, value in self.phosphosite_intensity.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"phosphosite_intensity for {key} is {value}; an unquantified "
                    "site is a not_quantified status, never a stored zero"
                )
        if not self.sample_label or not self.measurement_type:
            raise ValueError("sample_label and measurement_type are non-empty")
        if self.replicate < 1:
            raise ValueError(f"replicate is 1-based, got {self.replicate}")
        od = self.harvest_optical_density
        if od is not None and (not math.isfinite(od) or od <= 0.0):
            raise ValueError(f"harvest_optical_density must be positive, got {od}")
        _require_value_or_gap(
            self, ("replicate_unit", "harvest_growth_phase", "harvest_optical_density")
        )
        return self


# --- end #921 ------------------------------------------------------------------- #


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
      columns). It ALSO covers the variance-normalized colony-size score the
      *E. coli* chemical-genomics arrays release as a "fitness-score" (Shiver 2016,
      Nichols 2011 and the Collins 2006 pipeline they are computed with), and there is
      deliberately NO ``s_score`` member for that family. The reason is the sourcing
      rather than the statistics: no mirrored text calls the released number an S-score.
      Shiver 2016 says only that "These fitness-scores represent the statistical
      significance of a change in colony size for a particular condition, with negative
      and positive fitness-scores representing sensitivity and resistance,
      respectively", and that its pipeline "built upon previous analyses [8,18]" --
      Nichols 2011 and Collins 2006, NEITHER of which is in the literature mirror
      (#691). Its own Methods name the standardizing step, "variance normalization of
      the data (to improve reproducibility of measurements between plates)", which is
      what this member's "standardized" means. A member named after a statistic whose
      definition we cannot read would be a label with no provenance behind it, so the
      S-score identification stays an explicitly labeled inference in the loader and in
      ``[[torchcell.datasets.ecoli.shiver2016]]`` (issue #749).
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
    - ``relative_growth_rate``: the RUN's wild-type generation time divided by the well's
      generation time, so 1.0 = grows like the wild type, below 1.0 = slower, above 1.0 =
      faster. Distinct from ``growth_rate``, which is an absolute rate in the assay's own
      units: this number is a dimensionless ratio against a wild-type control measured in
      the SAME run, so it is comparable across runs where a raw rate is not. Its
      denominator is the well, which is why a slower strain scores below 1.0.
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
    - ``condition_growth_ratio``: a strain's end-point growth on the TEST plate divided by
      the SAME strain's growth on the control plate of the same screen (Peter 2018: the
      "growth ratio between stress media and standard YPD medium at 30 C" of each of 971
      natural isolates); 1.0 = grows as on the control, non-negative, 0 = no colony.
      Distinct from ``relative_growth_rate`` (the denominator there is the run's WILD
      TYPE in the same condition, and the numbers are generation times), from
      ``differential_fitness`` (a signed difference, 0 at the control) and from
      ``control_regression_residual`` (a residual with the control slope removed).
    - ``cell_yield``: ABSOLUTE number of cells a colony population yields over its
      growth curve, from calibrated colony images (De Chiara 2022's Scan-o-matic
      "total cell yields (y)", released in cells). Distinct from ``colony_size``,
      which is an end-point size on an image scale (pixels): this is a cell count, and
      the source does not say whether it is the final population or the gain over the
      inoculum, so it is named for what the source calls it.
    - ``relative_fitness``: SIGNED fitness of one barcode lineage relative to the mean
      fitness of the pooled population it competed in, as a lineage-tracking estimator
      reports it from read-count trajectories (Hale 2024: PyFitSeq Malthusian fitness on
      a time axis in generations, then shifted per assay to a stated reference); 0 =
      grows like the reference. Distinct from ``log2_ratio`` (an end-point abundance
      ratio, not a fitted growth rate) and from ``relative_growth_rate`` (a ratio against
      a wild-type well, not a difference against a pooled population).
    - ``optical_density``: ABSOLUTE absorbance of a liquid culture read at a stated
      wavelength after a stated incubation (Baba 2006 Supplementary Table 3: each Keio
      mutant's OD600 after 22 h in LB and 24 h and 48 h in glucose MOPS), on the plate
      reader's own scale and unnormalized. Distinct from ``growth_rate`` (a rate fitted
      over a curve; this is one end-point reading), from ``cell_yield`` (a cell count,
      not an absorbance) and from ``colony_size`` (a solid-plate image measure).
    """

    log2_ratio = "log2_ratio"
    z_score = "z_score"
    sensitivity_score = "sensitivity_score"
    categorical = "categorical"
    ordinal = "ordinal"
    growth_rate = "growth_rate"
    relative_growth_rate = "relative_growth_rate"
    differential_fitness = "differential_fitness"
    control_regression_residual = "control_regression_residual"
    colony_size = "colony_size"
    # ADDITIVE (Peter 2018 phenome): a same-strain test-over-control growth ratio.
    condition_growth_ratio = "condition_growth_ratio"
    cell_yield = "cell_yield"  # De Chiara 2022 (additive)
    relative_fitness = "relative_fitness"  # Hale 2024
    optical_density = "optical_density"  # Baba 2006 Table 3 (additive, #927)


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

# --------------------------------------------------------------------------- #
# ABSOLUTE environment-response readouts (#776). ADDITIVE: a name for the
# measurement types that are a quantity on the assay's own scale rather than a
# response relative to a control, so the verifier's reference rule can be relaxed
# for exactly those and refuse to be relaxed for anything else.
# --------------------------------------------------------------------------- #
#: Measurement types whose number is an ABSOLUTE quantity in the assay's own units, so
#: the record's reference states the reference condition's own measured value and NOT a
#: centered 0. A relative readout (``log2_ratio``, ``z_score``, ``sensitivity_score``,
#: ``differential_fitness``, ``control_regression_residual``, ``relative_growth_rate``)
#: is 0 at the control by construction and is deliberately absent: the
#: ``reference_centered=False`` branch of the environment-response verifier requires
#: every record's type to be a member here, which is what keeps the relief from being a
#: blanket relaxation. ``colony_size`` is here because Bloom 2019's two control-plate
#: conditions release the raw colony radius; ``growth_rate`` because Caglar 2017 Table S5
#: releases a doubling time in minutes; ``cell_yield`` because De Chiara 2022 releases
#: each colony population's cell yield in cells.
ABSOLUTE_MEASUREMENT_TYPES: frozenset[MeasurementType] = frozenset(
    {
        MeasurementType.growth_rate,
        MeasurementType.colony_size,
        MeasurementType.cell_yield,  # De Chiara 2022: cells per colony population
        MeasurementType.optical_density,  # Baba 2006 Table 3: end-point OD600 (#927)
    }
)
# --------------------------------------------------------------------------- #
# end ABSOLUTE environment-response readouts (#776)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# RATIO environment-response readouts. ADDITIVE, and the same shape as the
# ABSOLUTE set above: a name for the measurement types whose control is 1.0
# rather than 0, so the verifier's reference rule can be relaxed for exactly
# those and refuse to be relaxed for anything else.
# --------------------------------------------------------------------------- #
#: Measurement types whose number is a RATIO against a control measured in the same run,
#: so an unaffected strain scores 1.0 and the record's reference states 1.0 rather than a
#: centered 0. ``relative_growth_rate`` (the run's wild type as the denominator) and
#: ``condition_growth_ratio`` (the same strain on the control plate as the denominator)
#: qualify: a difference-scaled relative readout (``log2_ratio``, ``z_score``,
#: ``differential_fitness``, ``control_regression_residual``) is 0 at its control, and an
#: absolute quantity has no control in its units at all (``ABSOLUTE_MEASUREMENT_TYPES``).
#: The verifier's ``reference_unit_scaled=True`` branch requires every record's type to be
#: a member here, which is what keeps the relief from being a blanket relaxation. Hawkins
#: 2020's mismatch-CRISPRi relative fitness (strain doublings / wild-type doublings), the
#: Bioscreen panels' wild-type-relative growth rate and Peter 2018's growth ratios over
#: the YPD 30 C plate are the readouts that need it.
RATIO_MEASUREMENT_TYPES: frozenset[MeasurementType] = frozenset(
    {MeasurementType.relative_growth_rate, MeasurementType.condition_growth_ratio}
)
#: What a ratio readout's control is by construction.
RATIO_REFERENCE_VALUE: float = 1.0
# --------------------------------------------------------------------------- #
# end RATIO environment-response readouts
# --------------------------------------------------------------------------- #


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
    # ----------------------------------------------------------------------- #
    # ADDITIVE (#776): a two-sided confidence-interval carrier, and the
    # per-replicate identifier a source that releases one row per replicate
    # curve needs. Nothing above this block changes.
    # ----------------------------------------------------------------------- #
    environment_response_lower: float | None = Field(
        default=None,
        description="the SOURCE's lower confidence limit of the response, verbatim, at "
        "`confidence_level`. The counterpart of FluxPhenotype.net_flux_lower: a "
        "two-sided bound is not a single number, so naming one side as THE statistic "
        "(feeding a half-width to UncertaintyType.ci95) would misreport it. Caglar 2017 "
        "Table S5's interval is asymmetric in 55 of 55 rows (upper-to-lower half-width "
        "ratio median 1.3187), which is what makes a half-width field unusable for it.",
    )
    environment_response_upper: float | None = Field(
        default=None,
        description="the SOURCE's upper confidence limit of the response, verbatim, at "
        "`confidence_level`. Deliberately NOT validated to lie above "
        "`environment_response`: a limit carried through a nonlinear transform can come "
        "out below the value or negative (Caglar 2017 Table S5 Glycerol replicate 1 "
        "releases 95p = -1027.769034, the image under DT = log_e 2 / slope of a slope "
        "interval that straddles zero), and repairing or dropping it would substitute "
        "our arithmetic for the released bytes. The environment-response verifier's L2 "
        "`interval_orientation` rule counts such rows against a DECLARED oracle, so the "
        "count is measured and a regression in it fails the gate.",
    )
    confidence_level: float | None = Field(
        default=None,
        description="the level the two limits are stated at, as a fraction (0.95 for a "
        "95% interval). Required whenever either limit is set -- an unlabeled interval "
        "is the state the typed uncertainty axis exists to prevent.",
    )
    replicate_id: str | None = Field(
        default=None,
        description="the source's own identifier for ONE replicate measurement of a "
        "(strain, condition), when the release is per replicate rather than an "
        "aggregate (Caglar 2017 Table S5's `replicate` column: one row per biological "
        "replicate growth curve, each with its own interval and r^2). It joins the L1 "
        "uniqueness key, so independently released replicate curves stay distinct "
        "records instead of colliding as duplicates and nothing has to be aggregated "
        "into a mean the source never released. None when the source releases one "
        "aggregated measurement per (strain, condition), which is every other dataset.",
    )
    # ----------------------------------------------------------------------- #
    # end ADDITIVE (#776)
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    # ADDITIVE (#863): a per-record test of the response, the shape of the
    # protein-fold-change family's p-value pair but scalar (one response per
    # record). A p-value is a TEST of the number, not a dispersion of it, so it
    # never feeds environment_response_uncertainty or the derived SE. Nothing
    # above this block changes.
    # ----------------------------------------------------------------------- #
    environment_response_p_value: float | None = Field(
        default=None,
        description="the UNADJUSTED p-value the source released for this response's "
        "test, as a probability in [0, 1], verbatim (a released 0 is stored as 0; it "
        "is the source's rounding of a small value, not a placeholder). The test "
        "itself is named in `units` or the dataset's provenance (Wang 2024: TRANSIT "
        "resampling, a permutation test of the log2FC). None when the source releases "
        "no test.",
    )
    environment_response_p_value_adjusted: float | None = Field(
        default=None,
        description="the multiple-testing-adjusted p-value the source released for the "
        "same test, as a probability in [0, 1], verbatim; p_value_adjustment_method "
        "names the correction",
    )
    p_value_adjustment_method: str | None = Field(
        default=None,
        description="the multiple-testing correction behind "
        "environment_response_p_value_adjusted, e.g. 'benjamini_hochberg' (the same "
        "vocabulary as ProteinFoldChangePhenotype.p_value_adjustment_method). Required "
        "whenever an adjusted p-value is stored and forbidden otherwise",
    )
    # ----------------------------------------------------------------------- #
    # end ADDITIVE (#863)
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    # ADDITIVE (Wetmore 2015 own release): the released test statistic of the
    # response. A test statistic is a TEST of the number, like the p-value above,
    # so it never feeds environment_response_uncertainty or the derived SE.
    # Nothing above this block changes.
    # ----------------------------------------------------------------------- #
    environment_response_t_statistic: float | None = Field(
        default=None,
        description="the test statistic the source released for this response, "
        "verbatim. For RB-TnSeq it is the moderated t of Wetmore 2015 (Methods (iv)): "
        "t = f / sqrt(sigma^2 + max(V_e, V_n)) with sigma = 0.1, so it is a function "
        "of the response and its standard error and is stored because the release "
        "states it, not recomputed. The statistic is named in `units` or the "
        "dataset's provenance. None when the source releases no statistic.",
    )
    # ----------------------------------------------------------------------- #
    # end ADDITIVE (Wetmore 2015 own release)
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    # ADDITIVE (#879): the released Z score of the response. Like the t
    # statistic above it is stored because the release states it, and it never
    # feeds environment_response_uncertainty or the derived SE. Nothing above
    # this block changes.
    # ----------------------------------------------------------------------- #
    environment_response_z_score: float | None = Field(
        default=None,
        description="the Z score the source released for this response, verbatim. "
        "Its definition is the source's, named in `units` or the dataset's "
        "provenance; for Wang 2018 it is equation (4), the response divided by the "
        "standard deviation of a normal fitted to the non-targeting guides' fitness, "
        "one constant per screen, so it is a released rescaling of the response and "
        "not a dispersion of it. None when the source releases no Z score.",
    )
    # ----------------------------------------------------------------------- #
    # end ADDITIVE (#879)
    # ----------------------------------------------------------------------- #

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
        elif (
            self.environment_response is None
            # ADDITIVE (#927): a numeric readout's value may be a typed
            # ProvenanceGap, which is how a reference states that the source
            # releases no reading for it (Baba 2006 Table 3 has no wild-type row).
            and "environment_response" not in self.gapped_fields()
        ):
            raise ValueError(
                f"{self.measurement_type} requires a numeric environment_response "
                "(or a ProvenanceGap on 'environment_response')"
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
        # ------------------------------------------------------------------- #
        # ADDITIVE (#776): the interval invariant. Finite limits, and a level
        # whenever either limit is set. The limits are NOT required to bracket
        # the value -- see environment_response_upper's description.
        # ------------------------------------------------------------------- #
        for name in ("environment_response_lower", "environment_response_upper"):
            limit = getattr(self, name)
            if limit is not None and not math.isfinite(limit):
                raise ValueError(f"{name} must be finite, got {limit}")
        has_limit = (
            self.environment_response_lower is not None
            or self.environment_response_upper is not None
        )
        if has_limit and self.confidence_level is None:
            raise ValueError(
                "confidence_level is required when a confidence limit is stored "
                "(an interval with no stated level is unlabelled)"
            )
        if self.confidence_level is not None and not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                f"confidence_level is a fraction in (0, 1), got {self.confidence_level}"
            )
        # ------------------------------------------------------------------- #
        # end ADDITIVE (#776)
        # ------------------------------------------------------------------- #
        # ------------------------------------------------------------------- #
        # ADDITIVE (#863): p-values are finite probabilities, and an adjusted
        # p-value names its correction (the protein-fold-change invariant).
        # ------------------------------------------------------------------- #
        for name in (
            "environment_response_p_value",
            "environment_response_p_value_adjusted",
        ):
            p_value = getattr(self, name)
            if p_value is not None and (
                not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0
            ):
                raise ValueError(f"{name} is {p_value}, not a probability in [0, 1]")
        adjusted = self.environment_response_p_value_adjusted is not None
        if adjusted and self.p_value_adjustment_method is None:
            raise ValueError(
                "an adjusted p-value names its correction: set "
                "p_value_adjustment_method"
            )
        if not adjusted and self.p_value_adjustment_method is not None:
            raise ValueError(
                "p_value_adjustment_method describes a stored adjusted p-value; set "
                "environment_response_p_value_adjusted or leave the method None"
            )
        # ------------------------------------------------------------------- #
        # end ADDITIVE (#863)
        # ------------------------------------------------------------------- #
        # ------------------------------------------------------------------- #
        # ADDITIVE (Wetmore 2015 own release): a released test statistic is a
        # finite number; a statistic needs the response it tests.
        # ------------------------------------------------------------------- #
        t_statistic = self.environment_response_t_statistic
        if t_statistic is not None:
            if not math.isfinite(t_statistic):
                raise ValueError(
                    f"environment_response_t_statistic must be finite, got {t_statistic}"
                )
            if self.environment_response is None:
                raise ValueError(
                    "environment_response_t_statistic tests a numeric response; set "
                    "environment_response"
                )
        # ------------------------------------------------------------------- #
        # end ADDITIVE (Wetmore 2015 own release)
        # ------------------------------------------------------------------- #
        # ------------------------------------------------------------------- #
        # ADDITIVE (#879): a released Z score is a finite number and scores a
        # numeric response.
        # ------------------------------------------------------------------- #
        z_score = self.environment_response_z_score
        if z_score is not None:
            if not math.isfinite(z_score):
                raise ValueError(
                    f"environment_response_z_score must be finite, got {z_score}"
                )
            if self.environment_response is None:
                raise ValueError(
                    "environment_response_z_score scores a numeric response; set "
                    "environment_response"
                )
        # ------------------------------------------------------------------- #
        # end ADDITIVE (#879)
        # ------------------------------------------------------------------- #
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
    isolates (None for the S288C-derived BY); ``assembly_ref`` is the sha256-pinned
    ``ArtifactRef`` to the parent's assembly (the 1011 assemblies tarball in the genomes
    tier with the isolate's member path, or the S288C reference sequence file of the SGD
    assembly set for BY); ``engineered_background`` is the parent's marker/deletion
    genotype as the source states it, verbatim.
    """

    name: str
    peter_strain_id: str | None = None
    assembly_ref: ArtifactRef
    engineered_background: str


class SegregantGenotype(ModelStrict):
    """Strain-level haplotype mosaic of one haploid segregant from a two-parent cross.

    ``blocks`` partition the called markers of every chromosome into runs of one
    parent and one posterior (see ``HaplotypeBlock``); the per-marker call matrix is a
    derived view (re-expand each block at the cross's marker positions). A hard-call
    release (posterior 1.0 everywhere) gives one block per parent run; a released
    posterior that changes along a run of one parent splits that run at each change. ``call_method`` records
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
        """Per chromosome: blocks are in ascending order, non-overlapping, and no two
        adjacent blocks share both parent and posterior (they would be one run).
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
                if cur.parent == prev.parent and cur.posterior == prev.posterior:
                    raise ValueError(
                        f"{chrom}: adjacent blocks share parent {cur.parent} and "
                        f"posterior {cur.posterior}; merge them"
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


# --------------------------------------------------------------------------- #
# --- begin boocock2025: single-cell expression of haplotype-mosaic segregants ---
#
# Boocock 2025 (eLife, doi 10.7554/eLife.95566) measures one-pot single-cell RNA-seq of
# meiotic segregants: each cell is genotyped from its own transcribed SNPs and its
# transcriptome is a vector of UMI counts. No existing expression leaf holds that.
# ``PseudobulkExpressionPhenotype`` is a log2 fold change against a same-condition wild
# type, and a segregant panel has no wild type; ``RNASeqExpressionPhenotype`` requires
# TPM, which a 3' UMI assay does not produce. The leaf below stores what the release
# holds, absolute UMI counts, and the pair below is the expression sibling of
# ``SegregantGrowthExperiment``. Since #1016 the leaf is also Jackson 2020's and N'Guessan
# 2025's: one class for single-cell UMI counts, its source-specific parts (cell-cycle
# stages, cassette markers) required as a value or a typed gap.
# --------------------------------------------------------------------------- #
class SingleCellExpressionPhenotype(Phenotype, ModelStrict):
    """Single-cell RNA-seq UMI counts per gene, summed over the cells of one genotype.

    One leaf for every source that releases 10x UMI counts per cell (#1016): Boocock
    2025 (segregant cells, alone or pooled per segregant), Jackson 2020 (barcoded
    deletion strains pooled per strain and condition) and N'Guessan 2025 (segregant
    strains). ``n_cells`` is how many cells the counts are summed over: 1 for a single
    cell, more for a genotype observed in several cells (pseudobulk).

    ``umi_count`` holds every gene feature with a NONZERO count. The quantified feature
    universe is pinned by ``n_features`` and ``feature_universe_sha256`` (sha256 of the
    sorted feature ids joined by newlines), so a feature of that universe that is not a
    key had zero UMIs in these cells; a gene outside the universe is not stored (not
    quantified, or outside the source's analysis set). Storing only the nonzero counts
    keeps a single cell to its ~10^3 detected genes, and the pin keeps a zero
    distinguishable from "not measured".

    ``marker_umi_count`` holds the drug-resistance cassette transcripts a release counts
    beside the genes (Jackson 2020: ``KANMX`` and ``NATMX``), every marker keyed, zeros
    included; they are not R64 features, so they never enter ``umi_count``. A release
    that counts no cassette feature apart from the genes stores ``None`` with a
    ``ProvenanceGap``.

    ``total_umis`` is the UMI total over EVERY feature the source counted in these cells:
    genes inside and outside the stored universe and cassette markers alike. It is the
    size factor the source normalizes by, never smaller than the stored counts, and
    equal to them exactly when the release's every feature is stored (Jackson 2020).
    ``cell_barcodes``, ``cell_libraries`` and ``cell_total_umis`` say which cells were
    summed: each cell's droplet barcode as released, the sequencing library (the
    source's batch) it was captured in, and its own UMI total, aligned and in release
    row order. A barcode is unique only within its library, so the (library, barcode)
    pair identifies a cell; the per-cell totals sum to ``total_umis``.

    ``cell_cycle_stage_counts`` and ``library_cell_counts`` partition the ``n_cells``
    cells by the source's cell-cycle stage label and by library, verbatim labels; the
    library partition is the count of ``cell_libraries``. A source that assigns no stage
    (Jackson 2020, N'Guessan 2025) stores ``cell_cycle_stage_counts=None`` with a
    ``ProvenanceGap``; a missing partition without the gap is refused, so a None is
    never a silent absence.

    ``measurement_type`` is the leaf's one assay tag, ``scrnaseq_umi_count``, and no
    other value validates. The leaf is the physical measurement, 10x Chromium 3' UMI
    counts, and every release it holds stores the same tag whatever its design: summing
    a strain's cells (Jackson 2020) is analysis that ``n_cells`` already records, and the
    designs diverge in the experiment classes above the leaf (``SegregantExpression*``,
    ``PseudobulkUmiCount*``), not in the tag. A single-cell UMI platform other than 10x
    Chromium 3' gets a sibling leaf under the same parent (``Phenotype``), with its own
    tag, never a second tag on this leaf.

    Provenance: Boocock et al. 2025, eLife; Jackson et al. 2020, eLife. See
    ``[[torchcell.datasets.scerevisiae.boocock2025]]`` and
    ``[[torchcell.datasets.scerevisiae.jackson2020]]``.
    """

    graph_level: str = "node"
    label_name: str = "umi_count"
    label_statistic_name: str | None = None

    umi_count: dict[str, int] = Field(
        description=(
            "SortedDict of per-gene UMI counts summed over the n_cells cells; only "
            "nonzero counts are keys, a pinned-universe feature that is not a key had 0"
        ),
        repr=False,
    )
    marker_umi_count: dict[str, int] | None = Field(
        description=(
            "SortedDict of summed UMI counts of the cassette marker transcripts the "
            "release counts beside the genes (e.g. KANMX, NATMX); every marker keyed, "
            "zeros included. None only with a ProvenanceGap"
        ),
        repr=False,
    )
    total_umis: int = Field(
        description=(
            "UMI total over every feature the source counted in the cells (genes "
            "inside and outside the stored universe, and markers); the size factor, "
            "never smaller than umi_count plus marker_umi_count"
        )
    )
    n_features: int = Field(
        description="size of the quantified feature universe the counts are taken over"
    )
    feature_universe_sha256: str = Field(
        description="sha256 of the sorted feature ids of that universe, newline-joined"
    )
    n_cells: int = Field(description="number of cells the counts are summed over")
    cell_cycle_stage_counts: dict[str, int] | None = Field(
        description="cells per source cell-cycle stage label; sums to n_cells. None "
        "only with a ProvenanceGap, for a source that labels no stage (required, so a "
        "loader states the partition or its absence, never inherits a default)"
    )
    library_cell_counts: dict[str, int] = Field(
        description="cells per sequencing library (the source's batch); sums to "
        "n_cells and equals the count of cell_libraries"
    )
    cell_barcodes: list[str] = Field(
        description="the summed cells' droplet barcodes as released, in release row "
        "order; unique within a library",
        repr=False,
    )
    cell_libraries: list[str] = Field(
        description="each summed cell's sequencing library, aligned with "
        "cell_barcodes; the (library, barcode) pair identifies the cell",
        repr=False,
    )
    cell_total_umis: list[int] = Field(
        description="per-cell UMI totals over every feature the source counted, "
        "aligned with cell_barcodes; sums to total_umis",
        repr=False,
    )
    measurement_type: Literal["scrnaseq_umi_count"] = Field(
        default="scrnaseq_umi_count",
        description="the leaf's one assay tag: 10x Chromium 3' single-cell RNA-seq UMI "
        "counts per feature; a non-10x platform is a sibling leaf, not another tag",
    )

    def __repr__(self) -> str:
        """Summary repr instead of dumping the per-gene dict."""
        return (
            f"SingleCellExpressionPhenotype(genes={len(self.umi_count)}, "
            f"total_umis={self.total_umis}, n_cells={self.n_cells})"
        )

    @field_validator("umi_count", mode="before")
    @classmethod
    def convert_and_validate_umi_count(cls, v: Any) -> Any:  # raw pre-validation input
        """Coerce to a SortedDict; refuse empty input and any count that is not >= 1."""
        if not isinstance(v, dict):
            raise ValueError(
                f"umi_count must be a per-gene dict, got {type(v).__name__}"
            )
        if not v:
            raise ValueError("umi_count cannot be empty")
        if not isinstance(v, SortedDict):
            v = SortedDict(v)
        for key, value in v.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"umi_count for {key} must be a positive integer (zeros are "
                    f"omitted, not stored), got {value!r}"
                )
        return v

    @field_validator("marker_umi_count", mode="before")
    @classmethod
    def convert_marker_count(cls, v: Any) -> Any:  # raw pre-validation input
        """Coerce to a SortedDict of non-negative integer counts; None passes through."""
        if v is None:
            return v
        if not isinstance(v, dict):
            raise ValueError(
                f"marker_umi_count must be a per-marker dict, got {type(v).__name__}"
            )
        if not isinstance(v, SortedDict):
            v = SortedDict(v)
        for key, value in v.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"marker_umi_count for {key} must be a non-negative integer, "
                    f"got {value!r}"
                )
        return v

    @model_validator(mode="after")
    def _check_totals(self) -> "SingleCellExpressionPhenotype":
        """Totals, partitions, per-cell lists and the universe pin agree."""
        if self.n_cells < 1:
            raise ValueError(f"n_cells must be a positive integer, got {self.n_cells}")
        _require_value_or_gap(self, ("cell_cycle_stage_counts", "marker_umi_count"))
        markers = self.marker_umi_count or {}
        if set(self.umi_count) & set(markers):
            raise ValueError("a feature is keyed as both a gene and a marker")
        stored = sum(self.umi_count.values()) + sum(markers.values())
        if self.total_umis < stored:
            raise ValueError(
                f"total_umis {self.total_umis} is smaller than the stored counts "
                f"{stored} (umi_count plus marker_umi_count)"
            )
        if self.n_features < len(self.umi_count):
            raise ValueError(
                f"n_features {self.n_features} is smaller than the {len(self.umi_count)} "
                "keyed features"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", self.feature_universe_sha256):
            raise ValueError("feature_universe_sha256 must be 64 lowercase hex digits")
        for field_name in ("cell_cycle_stage_counts", "library_cell_counts"):
            counts: dict[str, int] | None = getattr(self, field_name)
            if counts is None:
                continue
            if any(n < 1 for n in counts.values()):
                raise ValueError(f"{field_name} holds a non-positive count")
            if sum(counts.values()) != self.n_cells:
                raise ValueError(
                    f"{field_name} sums to {sum(counts.values())}, not n_cells "
                    f"{self.n_cells}"
                )
        for list_name in ("cell_barcodes", "cell_libraries", "cell_total_umis"):
            n_listed = len(getattr(self, list_name))
            if n_listed != self.n_cells:
                raise ValueError(
                    f"{list_name} lists {n_listed} cells for n_cells {self.n_cells}"
                )
        cells = set(zip(self.cell_libraries, self.cell_barcodes, strict=True))
        if len(cells) != self.n_cells:
            raise ValueError("a (library, barcode) cell repeats")
        if dict(Counter(self.cell_libraries)) != dict(self.library_cell_counts):
            raise ValueError("library_cell_counts is not the count of cell_libraries")
        if any(isinstance(n, bool) or n < 0 for n in self.cell_total_umis):
            raise ValueError("cell_total_umis holds a negative or boolean count")
        if sum(self.cell_total_umis) != self.total_umis:
            raise ValueError(
                f"cell_total_umis sums to {sum(self.cell_total_umis)}, not total_umis "
                f"{self.total_umis}"
            )
        return self


class SegregantExpressionExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a segregant single-cell expression experiment."""

    experiment_reference_type: str = "segregant_expression"
    phenotype_reference: SingleCellExpressionPhenotype


class SegregantExpressionExperiment(Experiment, ModelStrict):
    """Single-cell expression of a haploid segregant (haplotype-mosaic genotype)."""

    experiment_type: str = "segregant_expression"
    genotype: SegregantGenotype  # type: ignore[assignment]  # sibling of Genotype, deliberately narrowed
    phenotype: SingleCellExpressionPhenotype


# --- end boocock2025 ---


# --------------------------------------------------------------------------- #
# --- begin jackson2020: pseudobulk UMI counts of barcoded deletion strains ---
#
# Jackson 2020 (eLife 9:e51254) releases a raw single-cell UMI count matrix of 72
# transcriptionally barcoded diploid deletion strains pooled in 11 growth conditions
# (Source code 2: "gene expression values are unmodified transcript counts after
# deartifacting using UMIs"), and its own pseudobulk sums those raw counts per strain and
# condition. The phenotype is ``SingleCellExpressionPhenotype``, the one single-cell
# UMI-count leaf (#1016): Jackson's own leaf (``PseudobulkUmiCountPhenotype``) was merged
# into it after every stored value of both dev stores was measured to round-trip, with
# the cassette markers, the per-cell barcodes and totals kept. The experiment pair below
# is Jackson's own because its genotype is gene-keyed: it states the strain background on
# a ``StrainReferenceGenome`` and the culture protocol on a ``CultureEnvironment``.
# --------------------------------------------------------------------------- #
class PseudobulkUmiCountExperimentReference(ExperimentReference, ModelStrict):
    """Reference for a pseudobulk UMI-count experiment (typed strain background).

    ``genome_reference`` is a ``StrainReferenceGenome`` (the background the deletion
    strains were built in, mating type and alleles sourced or gapped);
    ``environment_reference`` is a ``CultureEnvironment``.
    """

    experiment_reference_type: str = "pseudobulk_umi_count"
    # Narrowed to subclasses: this family states its background and culture protocol.
    genome_reference: StrainReferenceGenome
    environment_reference: CultureEnvironment
    phenotype_reference: SingleCellExpressionPhenotype


class PseudobulkUmiCountExperiment(Experiment, ModelStrict):
    """Pseudobulk single-cell UMI counts of one deletion strain in one culture."""

    experiment_type: str = "pseudobulk_umi_count"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    environment: CultureEnvironment  # narrowed: this family states its culture protocol
    phenotype: SingleCellExpressionPhenotype


# --- end jackson2020 ---


# --------------------------------------------------------------------------- #
# begin ADDITIVE (Hale 2024): an ENGINEERED perturbation on a SEGREGANT background.
#
# Hale 2024 integrates a CRISPRi guide into each of 169 barcoded segregants of a
# BY4716 x 322134S cross (and into the BY parent). The strain's genome is therefore
# two different kinds of thing at once: an INFERRED haplotype mosaic (the segregant,
# a ``SegregantGenotype`` with no perturbation list) and a DESIGNED gene-keyed edit
# (the guide, a ``CrisprInterferencePerturbation``). Neither ``Genotype`` (gene-keyed
# edits on the S288C reference) nor ``SegregantGenotype`` (a mosaic with nothing
# engineered) can hold both, and folding the guide into either would lose one half:
# a ``Genotype`` would read the strain as S288C plus a knockdown, a mosaic would drop
# the knockdown. ``PerturbedSegregantGenotype`` composes the two, keeping each in the
# class that already states it honestly: the background stays a mosaic with its call
# method and matrix pin, the perturbations stay typed gene-keyed leaves.
#
# Another sibling of ``Genotype`` (never a subclass), for the reason
# ``SegregantGenotype`` is one: gene-keyed consumers must not mistake the strain for
# an S288C derivative. The new names appear only in their own bodies and the
# module-level union and map assignments, so no served class body changes.
# --------------------------------------------------------------------------- #
class PerturbedSegregantGenotype(ModelStrict):
    """Gene-keyed engineered perturbations carried on a segregant-cross background.

    ``background`` is the strain's haplotype mosaic as the source released it.
    ``background_role`` says which member of the cross the background is: a meiotic
    ``segregant``, or one of the cross's parents (``parent_1`` / ``parent_2``) when the
    panel includes a parental control transformed with the same perturbation library.
    A parent's background is the degenerate mosaic of that parent at every called
    marker; the validator enforces it, so a parent role can never mask a recombinant.
    ``perturbations`` are the designed edits (Hale 2024: one CRISPRi guide each), sorted
    exactly as ``Genotype`` sorts them so equal genotypes serialize equally.
    """

    background: SegregantGenotype = Field(
        description="the haplotype mosaic of the strain the perturbation was put into"
    )
    background_role: Literal["segregant", "parent_1", "parent_2"] = Field(
        description="whether the background is a meiotic segregant or one of the "
        "cross's two parents carried as a control"
    )
    perturbations: list[GenePerturbationType] = Field(
        description="the designed gene-keyed perturbations on that background"
    )

    @field_validator("perturbations", mode="after")
    @classmethod
    def _sort_perturbations(
        cls, perturbations: list[GenePerturbationType]
    ) -> list[GenePerturbationType]:
        """At least one perturbation (no edit is a plain ``SegregantGenotype``), sorted."""
        if not perturbations:
            raise ValueError(
                "PerturbedSegregantGenotype needs at least one perturbation; an "
                "unperturbed segregant is a SegregantGenotype"
            )
        return sorted(
            perturbations,
            key=lambda p: (
                p.systematic_gene_name,
                p.perturbation_type,
                p.perturbed_gene_name,
            ),
        )

    @model_validator(mode="after")
    def _parent_role_is_one_parent(self) -> "PerturbedSegregantGenotype":
        """A parent role's background carries that parent at every block."""
        if self.background_role == "segregant":
            return self
        parent = 1 if self.background_role == "parent_1" else 2
        mixed = [b for b in self.background.blocks if b.parent != parent]
        if mixed:
            raise ValueError(
                f"background_role {self.background_role!r} but {len(mixed)} block(s) "
                f"carry the other parent (first: {mixed[0]})"
            )
        return self

    @property
    def systematic_gene_names(self) -> list[str]:
        """Systematic names of the perturbed genes, ordered by systematic name."""
        return [p.systematic_gene_name for p in self.perturbations]

    @property
    def perturbed_gene_names(self) -> list[str]:
        """Perturbed gene names, ordered by systematic name."""
        return [p.perturbed_gene_name for p in self.perturbations]

    @property
    def perturbation_types(self) -> list[str]:
        """Perturbation types, ordered by systematic name."""
        return [p.perturbation_type for p in self.perturbations]

    def __len__(self) -> int:
        """Return the number of perturbations."""
        return len(self.perturbations)


class SegregantPerturbationExperimentReference(ExperimentReference, ModelStrict):
    """Reference (control) context for a perturbation-on-segregant experiment."""

    experiment_reference_type: str = "segregant_perturbation"
    phenotype_reference: EnvironmentResponsePhenotype


class SegregantPerturbationExperiment(Experiment, ModelStrict):
    """A perturbation on a segregant background, measured in one environment."""

    experiment_type: str = "segregant_perturbation"
    genotype: PerturbedSegregantGenotype  # type: ignore[assignment]  # sibling of Genotype, deliberately narrowed
    phenotype: EnvironmentResponsePhenotype


# --------------------------------------------------------------------------- #
# end ADDITIVE (Hale 2024)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Bacterial phenotypes (plan 3c). Three record types the fifty bacterial rows need
# and that no existing class can hold; the other eight families they need are
# REUSED unchanged, and get bacterial experiment pairs further below purely so
# their reference can pin an assembly.
#
# A new phenotype class moves 0 of 36 served dataset closures, and the unions and
# maps it is appended to are module-level assignments outside the fingerprinted
# surface.
# --------------------------------------------------------------------------- #
class ProductYieldUnit(StrEnum):
    """How a production yield was expressed. Add values as datasets need them.

    A yield is a RATIO, so it carries no concentration unit and cannot reuse
    ``ConcentrationUnit``; these three are the forms a fermentation paper reports.
    """

    g_per_g_substrate = "g/g_substrate"
    mol_per_mol_substrate = "mol/mol_substrate"
    percent_of_theoretical = "percent_of_theoretical"


class ProductivityUnit(StrEnum):
    """How a volumetric productivity was expressed. Add values as datasets need them."""

    g_per_l_per_h = "g/L/h"
    mg_per_l_per_h = "mg/L/h"


# --- begin #753: per-key censoring on a turnover measurement ------------------- #
class Censoring(StrEnum):
    """Whether a per-key measurement is an ESTIMATE or a BOUND, and on which side.

    A pulse-labeling turnover assay cannot resolve a protein that barely turns over
    within the labeling window, so the fit is capped and the released number is the cap:
    a right-censored value, not an estimate. Gupta 2024 releases 2,082 such cells over
    its 13 conditions, each marked ``*`` with the gloss "Protein total half-life was set
    to ceiling for this dilution rate". Averaging a censored value with uncensored ones
    biases the mean toward the cap, so a consumer must be able to tell them apart.

    ``uncensored`` is a member rather than being spelled by absence: a source whose
    censoring oracle is complete (Gupta's is, 0 disagreements in 61,811 cells) can state
    it for every key, and a key MISSING from the map then means the source does not say
    for that key. The two are different facts and the schema keeps them different.

    THE SIDE IS STATED ON THE QUANTITY THE SOURCE CENSORED, which is not always the
    record's primary label. Gupta 2024 caps the released total HALF-LIFE, so the true
    half-life lies above the stored one and the cell is ``right`` censored; because
    ``degradation_rate = ln 2 / half_life`` is DECREASING, that same cell bounds the
    RATE from above. A consumer reading ``right`` beside a rate alone would read it
    backwards, so a loader that stores censoring names the censored quantity in its
    ``measurement_type`` and in its note.
    """

    uncensored = "uncensored"
    right = "right"
    left = "left"


# --- end #753 ------------------------------------------------------------------- #


class ProductTiterPhenotype(Phenotype, ModelStrict):
    """How much of a named product a strain made: titer, with yield and productivity.

    The phenotype side of inverse strain design, and the one genuinely new record type
    the engineering half of the bacterial list needs. The product is a typed
    ``Compound``, NOT a free-text product name, so a titer joins the same compound
    entity a chemogenomic dataset dosing that molecule already uses and "every record
    about isoprenol" is one query rather than a string match.

    The value is a plain float with a typed ``titer_unit``, so the ML-facing label stays
    scalar, and the uncertainty ontology is ``FitnessPhenotype``'s: the source-reported
    number in ``titer_uncertainty`` with its ``titer_uncertainty_type``, the replicate
    design in ``n_samples`` + ``sample_unit``, and the DERIVED standard error in
    ``titer_se`` (auto-filled through ``derive_se``). ``product_yield`` and
    ``productivity`` are optional because most rows release only a titer, and each is
    paired with its own unit enum since neither is a concentration.

    What is deliberately NOT here: the fermentation context. The medium, the carbon
    source, the oxygen regime, the vessel and the duration are all ``Environment``
    (and ``CultureEnvironment``) slots already, and putting them on the phenotype would
    make "what was measured" carry part of "what it was grown in".

    A titer the source never RELEASED is a typed gap, not a substitute (#788). Banerjee
    2024's M9-minimal family states its denominator only as a ratio ("over 4-fold higher
    than WT (IY1245)") and never prints the parent's titer, so the reference carries
    ``titer=None`` with a ``ProvenanceGap`` naming 'titer'. Putting a family member's own
    value there instead would state a parental measurement nobody made, and
    reconstructing one from the ratio would invent a number; the three released child
    titers stay records either way. ``titer`` therefore ADMITS ``None`` without becoming
    optional: it keeps no default, so every record states it explicitly, and
    ``_check_titer`` refuses a ``None`` that no gap declares.
    This is distinct from the censoring case below, which is a value the assay bounded.

    A titer the assay could not detect is a BOUND, not a zero (#867): Thompson 2019
    states the wild type's 48 h valerolactam as "no valerolactam could be detected",
    which is a value below the assay's floor. ``titer_censoring`` carries it on the
    ``ProteinTurnoverPhenotype`` precedent (#753): ``left`` means the true titer lies
    BELOW the stored ``titer``, which then holds the floor in the culture, and such a
    titer carries no uncertainty and no standard error because a bound has neither.
    """

    graph_level: str = "global"
    label_name: str = "titer"
    label_statistic_name: str = "titer_se"

    product: Compound = Field(
        description="typed chemical identity of the product measured (InChIKey / ChEBI "
        "when sourced), the join key onto the shared compound layer"
    )
    titer: float | None = Field(
        description="measured product titer in ``titer_unit``. REQUIRED to be stated: "
        "there is no default, so every record passes it explicitly. ``None`` is "
        "admitted ONLY with a ``ProvenanceGap`` naming 'titer' -- a reference whose "
        "parental titer the source never released (see the class docstring)"
    )
    titer_unit: ConcentrationUnit = Field(
        description="typed UO-aligned unit of the titer, e.g. 'g/L' | 'mM'"
    )
    titer_se: float | None = Field(
        default=None, description="DERIVED standard error of the titer"
    )
    titer_uncertainty: float | None = Field(
        default=None,
        description="source-reported uncertainty number, verbatim (its meaning is given "
        "by titer_uncertainty_type)",
    )
    titer_uncertainty_type: UncertaintyType | None = Field(
        default=None, description="what titer_uncertainty IS (sample_sd, ...)"
    )
    n_samples: int | None = Field(
        default=None,
        description="number of independent replicate cultures the titer is over",
    )
    sample_unit: SampleUnit | None = Field(
        default=None,
        description="what one sample in n_samples is (biological_replicate, ...)",
    )
    product_yield: float | None = Field(
        default=None,
        description="product per substrate consumed; None when not released",
    )
    product_yield_unit: ProductYieldUnit | None = Field(
        default=None, description="typed unit of product_yield; required when it is set"
    )
    productivity: float | None = Field(
        default=None, description="volumetric productivity; None when not released"
    )
    productivity_unit: ProductivityUnit | None = Field(
        default=None, description="typed unit of productivity; required when it is set"
    )
    quantification_method: str | None = Field(
        default=None,
        description="how the product was quantified, verbatim (e.g. 'GC-MS', "
        "'HPLC-RID'); None when the source does not state it",
    )
    # --- begin #867: a below-detection titer is a bound, not a zero ------------- #
    titer_censoring: Censoring | None = Field(
        default=None,
        description="whether the stored titer is an ESTIMATE or a BOUND, and on which "
        "side. 'left' means the true titer lies below the stored value, which is the "
        "assay's floor expressed in the culture (a 'not detected' cell); 'right' means "
        "it lies above (a saturated assay's ceiling); 'uncensored' means the source "
        "states a quantified value. None means the source says nothing about censoring "
        "for this titer.",
    )
    # --- end #867 ---------------------------------------------------------------- #

    @field_validator("titer", mode="after")
    @classmethod
    def validate_titer(cls, v: float | None) -> float | None:
        """A titer is a measured amount: finite and non-negative, or a typed absence."""
        if v is None:
            return v
        if math.isnan(v) or math.isinf(v):
            raise ValueError("titer must be a finite number")
        if v < 0:
            raise ValueError(f"titer must be non-negative, got {v}")
        return v

    @field_validator("n_samples", mode="after")
    @classmethod
    def validate_n_samples(cls, v: int | None) -> int | None:
        """n_samples is a positive integer or None."""
        if v is not None and v < 1:
            raise ValueError(f"n_samples must be a positive integer or None, got: {v}")
        return v

    @model_validator(mode="before")
    @classmethod
    def _fill_titer_se(cls, data: Any) -> Any:
        """Derive the ML-facing titer_se from the reported uncertainty (frozen -> fill first)."""
        if not isinstance(data, dict):
            return data
        unc = data.get("titer_uncertainty")
        typ = data.get("titer_uncertainty_type")
        if unc is None or typ is None or data.get("titer_se") is not None:
            return data
        typ = UncertaintyType(typ)
        n = data.get("n_samples")
        if typ in (UncertaintyType.sample_sd, UncertaintyType.variance) and n is None:
            return data
        data["titer_se"] = derive_se(unc, typ, n)
        return data

    @model_validator(mode="after")
    def _check_titer(self) -> "ProductTiterPhenotype":
        """No unlabelled uncertainty, no unitless yield or productivity.

        An absent titer must be a DECLARED absence (#788): ``ProvenanceGapMixin``
        already refuses a gap beside a stored value, and this is the other half --
        a ``None`` titer with no gap naming it would be a silent hole rather than a
        typed one. An absent titer also carries no uncertainty and no censoring,
        because there is no value for either to be about.
        """
        if self.titer is None:
            if "titer" not in {gap.field for gap in self.provenance_gaps}:
                raise ValueError(
                    "titer is None without a ProvenanceGap naming 'titer': an absent "
                    "titer is a declared absence, never a silent one"
                )
            if self.titer_uncertainty is not None or self.titer_se is not None:
                raise ValueError(
                    "an absent titer carries no titer_uncertainty and no titer_se"
                )
            if self.titer_censoring is not None:
                raise ValueError(
                    "an absent titer carries no titer_censoring: a bound is a statement "
                    "about a value, and no value was released"
                )
        unc, typ = self.titer_uncertainty, self.titer_uncertainty_type
        if (unc is None) != (typ is None):
            raise ValueError(
                "titer_uncertainty and titer_uncertainty_type must both be set or both "
                "be None (no unlabelled uncertainty)"
            )
        if typ in (UncertaintyType.sample_sd, UncertaintyType.variance) and (
            self.n_samples is None or self.sample_unit is None
        ):
            raise ValueError(f"n_samples and sample_unit are required for {typ}")
        for value_name, unit_name in (
            ("product_yield", "product_yield_unit"),
            ("productivity", "productivity_unit"),
        ):
            value = getattr(self, value_name)
            unit = getattr(self, unit_name)
            if (value is None) != (unit is None):
                raise ValueError(
                    f"{value_name} and {unit_name} must both be set or both be None "
                    "(a bare number with no unit is not a measurement)"
                )
            if value is not None and value < 0:
                raise ValueError(f"{value_name} must be non-negative, got {value}")
        # --- begin #867: a below-detection titer is a bound, not a zero --------- #
        if self.titer_censoring in (Censoring.left, Censoring.right):
            if self.titer_uncertainty is not None or self.titer_se is not None:
                raise ValueError(
                    f"a {self.titer_censoring.value}-censored titer is a bound, so it "
                    "carries no titer_uncertainty and no titer_se"
                )
        if self.titer_censoring == Censoring.left and self.titer == 0.0:
            raise ValueError(
                "a left-censored titer stores the detection floor it lies below; a "
                "floor of 0.0 states an exact zero, which is a measurement, not a bound"
            )
        # --- end #867 ------------------------------------------------------------ #
        return self


class ProteinTurnoverPhenotype(Phenotype, ModelStrict):
    """Per-protein turnover: how fast each protein is degraded and replaced.

    Distinct from ``ProteinAbundancePhenotype``, which is a STANDING amount: a turnover
    measurement is a RATE, obtained by following a label into or out of the proteome, and
    two strains can hold the same abundance through entirely different synthesis and
    degradation rates. Keyed by the protein's locus tag in the record's namespace, with
    the same ragged-key honesty as the other dict-valued families: a protein the assay
    did not quantify is simply not a key, never a zero.

    ``degradation_rate`` is the primary label; ``half_life`` and ``synthesis_rate`` are
    optional companions a study may release instead of or beside it.
    ``measurement_type`` records WHAT the numbers are, including their time unit, so two
    assays are never silently compared.

    Two further axes (#753) carry what a turnover release routinely publishes and this
    class could not hold. ``degradation_rate_lower`` / ``degradation_rate_upper`` with
    ``confidence_level`` and ``interval_method`` store a published INTERVAL as an
    interval, reusing ``FluxPhenotype``'s lossless triple rather than inverting it into
    a standard deviation the source does not give; ``label_statistic_name`` stays
    ``degradation_rate_se`` because a replicate SE and a fitted interval are different
    statistics and a release may carry either. ``censoring`` says per key whether the
    stored number is an estimate or a BOUND: a pulse-labeling assay caps a protein that
    barely turns over, and a capped half-life averaged in with estimates biases the mean
    toward the cap.
    """

    graph_level: str = "node"
    label_name: str = "degradation_rate"
    label_statistic_name: str | None = "degradation_rate_se"

    degradation_rate: dict[str, float] = Field(
        description="protein locus tag -> degradation rate in measurement_type's unit"
    )
    degradation_rate_se: dict[str, float] | None = Field(
        default=None, description="protein locus tag -> standard error of the rate"
    )
    half_life: dict[str, float] | None = Field(
        default=None,
        description="protein locus tag -> half-life, when the source reports one",
    )
    synthesis_rate: dict[str, float] | None = Field(
        default=None,
        description="protein locus tag -> synthesis rate, when the source reports one",
    )
    n_replicates: dict[str, int] = Field(
        description="protein locus tag -> number of independent samples/replicates"
    )
    measurement_type: str = Field(
        description="what the numbers are, including the time unit, e.g. "
        "'pulse_silac_degradation_rate_per_hour'"
    )
    # --- begin #753: the published interval and the censoring flag -------------- #
    degradation_rate_lower: dict[str, float] | None = Field(
        default=None,
        description="protein locus tag -> lower bound of the published interval on the "
        "RATE scale (the scale label_name is on). Ragged by design: a key the source "
        "publishes no interval for is simply not a key here.",
    )
    degradation_rate_upper: dict[str, float] | None = Field(
        default=None,
        description="protein locus tag -> upper bound of the published interval on the "
        "RATE scale",
    )
    confidence_level: float | None = Field(
        default=None,
        description="the level the bounds are stated at, as a fraction (e.g. 0.95); "
        "None when no interval is stored, or a typed gap when bounds are stored and "
        "the source does not say what level they are",
    )
    interval_method: str | None = Field(
        default=None,
        description="HOW the interval was produced, so two studies' bounds are never "
        "silently compared, e.g. 'curve_fit_parameter_variance_t_quantile' or "
        "'bootstrap_percentile'. None when no interval is stored",
    )
    censoring: dict[str, Censoring] | None = Field(
        default=None,
        description="protein locus tag -> whether the stored value is an estimate or a "
        "BOUND, and on which side. A right-censored half-life capped at the labeling "
        "window's ceiling is a bound, and averaging it with estimates biases the mean, "
        "so it must travel with the record. None means the source reports no censoring "
        "at all; a key absent from a present map means the source does not say for "
        "that key.",
    )
    # --- end #753 ---------------------------------------------------------------- #

    @model_validator(mode="after")
    def validate_turnover(self) -> "ProteinTurnoverPhenotype":
        """Non-empty rates, matching replicate keys, non-negative rates and errors."""
        if not self.degradation_rate:
            raise ValueError("degradation_rate cannot be empty")
        if set(self.n_replicates) != set(self.degradation_rate):
            raise ValueError("n_replicates keys must match degradation_rate keys")
        for key, n in self.n_replicates.items():
            if n < 1:
                raise ValueError(f"n_replicates for {key} must be >= 1")
        for key, rate in self.degradation_rate.items():
            if math.isnan(rate) or math.isinf(rate) or rate < 0:
                raise ValueError(
                    f"degradation_rate for {key} must be finite and non-negative"
                )
        for name in ("degradation_rate_se", "half_life", "synthesis_rate"):
            mapping = getattr(self, name)
            if mapping is None:
                continue
            for key, value in mapping.items():
                if key not in self.degradation_rate:
                    raise ValueError(f"{name} key {key} not in degradation_rate")
                if not math.isnan(value) and value < 0:
                    raise ValueError(f"{name} for {key} must be non-negative")
        # --- begin #753: the published interval and the censoring flag ---------- #
        for name in ("degradation_rate_lower", "degradation_rate_upper"):
            mapping = getattr(self, name)
            if mapping is None:
                continue
            for key, value in mapping.items():
                if key not in self.degradation_rate:
                    raise ValueError(f"{name} key {key} not in degradation_rate")
                if math.isnan(value) or math.isinf(value) or value < 0:
                    raise ValueError(
                        f"{name} for {key} must be finite and non-negative"
                    )
        for key, lower in (self.degradation_rate_lower or {}).items():
            if lower > self.degradation_rate[key]:
                raise ValueError(
                    f"degradation_rate_lower for {key} exceeds the stored rate "
                    f"({lower} > {self.degradation_rate[key]})"
                )
        for key, upper in (self.degradation_rate_upper or {}).items():
            if upper < self.degradation_rate[key]:
                raise ValueError(
                    f"degradation_rate_upper for {key} is below the stored rate "
                    f"({upper} < {self.degradation_rate[key]})"
                )
        has_bounds = (
            self.degradation_rate_lower is not None
            or self.degradation_rate_upper is not None
        )
        gapped = self.gapped_fields()
        if (
            has_bounds
            and self.confidence_level is None
            and ("confidence_level" not in gapped)
        ):
            raise ValueError(
                "an interval states the level it covers: set confidence_level or "
                "carry a ProvenanceGap on it"
            )
        if (
            has_bounds
            and self.interval_method is None
            and ("interval_method" not in gapped)
        ):
            raise ValueError(
                "an interval states how it was produced: set interval_method or "
                "carry a ProvenanceGap on it"
            )
        if not has_bounds and (
            self.confidence_level is not None or self.interval_method is not None
        ):
            raise ValueError(
                "confidence_level and interval_method describe stored bounds; set "
                "degradation_rate_lower/upper or leave both None"
            )
        if self.confidence_level is not None and not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                f"confidence_level is a fraction in (0, 1), got {self.confidence_level}"
            )
        for key in self.censoring or {}:
            if key not in self.degradation_rate:
                raise ValueError(f"censoring key {key} not in degradation_rate")
        # --- end #753 ------------------------------------------------------------ #
        return self


# --- begin #857: the protein synthesis-rate family ---------------------------- #
class SynthesisRateUnit(StrEnum):
    """The unit a per-protein synthesis rate is released in, amount over time basis.

    The time basis is part of the unit and is typed rather than spelled into a string,
    because the two bases do not compare without a growth rate: a rate per GENERATION
    (Li 2014's ribosome-profiling release, "where ki has the unit of molecules per
    generation") becomes a rate per hour only through the doubling time of the culture
    it was measured in, which is why ``molecules_per_generation`` requires
    ``ProteinSynthesisRatePhenotype.generation_time_minutes``.
    """

    molecules_per_generation = "molecules_per_generation"
    molecules_per_hour = "molecules_per_hour"


class ProteinSynthesisRatePhenotype(Phenotype, ModelStrict):
    """Per-protein synthesis rate: how many molecules of each protein a cell makes.

    A sibling of ``ProteinTurnoverPhenotype``, not a relaxation of it. A turnover
    record's primary label is the DEGRADATION rate and its ``synthesis_rate`` is a
    companion admitted only on keys that carry a degradation rate; a ribosome-profiling
    release (Li 2014) measures synthesis alone and states no degradation, half-life or
    turnover for any protein. Making ``degradation_rate`` optional would let one class
    carry two different primary labels and switch ``label_name`` per record, so a
    consumer reading the turnover lane could no longer trust what the label is.

    Not a ``ProteinAbundancePhenotype`` either: a synthesis rate equals a standing copy
    number only for a stable protein in steady state, which is an assumption about the
    protein, not a measurement of it.

    Keyed by the protein's locus tag in the record's namespace, ragged like every
    dict-valued family: a protein the assay did not quantify (or quantified below its
    own read-count gate) is not a key, never a zero. ``rate_unit`` types the amount and
    the time basis; ``measurement_type`` names the assay. ``censoring`` follows
    ``ProteinTurnoverPhenotype``: None when the source states no censoring at all.
    """

    graph_level: str = "node"
    label_name: str = "synthesis_rate"
    label_statistic_name: str | None = "synthesis_rate_se"

    synthesis_rate: dict[str, float] = Field(
        description="protein locus tag -> synthesis rate in rate_unit"
    )
    synthesis_rate_se: dict[str, float] | None = Field(
        default=None,
        description="protein locus tag -> standard error of the rate; None when the "
        "source releases no per-protein uncertainty",
    )
    n_replicates: dict[str, int] | None = Field(
        default=None,
        description="protein locus tag -> number of independent samples behind the "
        "rate; None when the source releases no per-protein replicate count",
    )
    rate_unit: SynthesisRateUnit = Field(
        description="amount and time basis of every value in synthesis_rate"
    )
    generation_time_minutes: float | None = Field(
        default=None,
        description="the culture's doubling time in minutes, as the source states it; "
        "required when rate_unit is per generation, since it is the time basis",
    )
    measurement_type: str = Field(
        description="the assay that produced the numbers, e.g. "
        "'ribosome_profiling_footprint_density'"
    )
    censoring: dict[str, Censoring] | None = Field(
        default=None,
        description="protein locus tag -> whether the stored value is an estimate or a "
        "bound; None means the source reports no censoring",
    )

    @model_validator(mode="after")
    def validate_synthesis_rate(self) -> "ProteinSynthesisRatePhenotype":
        """Non-empty finite non-negative rates; companions keyed on stored proteins."""
        if not self.synthesis_rate:
            raise ValueError("synthesis_rate cannot be empty")
        for key, rate in self.synthesis_rate.items():
            if math.isnan(rate) or math.isinf(rate) or rate < 0:
                raise ValueError(
                    f"synthesis_rate for {key} must be finite and non-negative"
                )
        for key, se in (self.synthesis_rate_se or {}).items():
            if key not in self.synthesis_rate:
                raise ValueError(f"synthesis_rate_se key {key} not in synthesis_rate")
            if not math.isnan(se) and se < 0:
                raise ValueError(f"synthesis_rate_se for {key} must be non-negative")
        if self.n_replicates is not None:
            if set(self.n_replicates) != set(self.synthesis_rate):
                raise ValueError("n_replicates keys must match synthesis_rate keys")
            for key, n in self.n_replicates.items():
                if n < 1:
                    raise ValueError(f"n_replicates for {key} must be >= 1")
        for key in self.censoring or {}:
            if key not in self.synthesis_rate:
                raise ValueError(f"censoring key {key} not in synthesis_rate")
        minutes = self.generation_time_minutes
        if minutes is not None and (
            math.isnan(minutes) or math.isinf(minutes) or minutes <= 0
        ):
            raise ValueError(
                f"generation_time_minutes must be finite and positive, got {minutes}"
            )
        if (
            self.rate_unit is SynthesisRateUnit.molecules_per_generation
            and minutes is None
        ):
            raise ValueError(
                "a rate per generation states its generation time: set "
                "generation_time_minutes"
            )
        return self


# --- end #857 ------------------------------------------------------------------- #


class FluxPhenotype(Phenotype, ModelStrict):
    """Per-reaction net metabolic flux, with the confidence interval of the fit.

    A flux map is not a measurement of one reaction at a time: it is a FIT of a whole
    reaction network to labeling data, so the interval is a property of that fit and is
    stored as an interval rather than as a scalar statistic. ``label_statistic_name`` is
    therefore ``None`` -- a two-sided confidence bound is not a single number, and naming
    one of the two bounds as "the" statistic would misreport it.

    ``net_flux`` is SIGNED: a net flux through a reversible reaction can run either way,
    and the sign is the direction, so nothing here is clamped. Keys are the source's own
    reaction identifiers; ``target_reaction_ids`` maps them to a constraint-based model's
    reaction ids where that mapping is settled, and is ``None`` until it is -- the data is
    captured, the modeling choice deferred.
    """

    graph_level: str = "metabolism"
    label_name: str = "net_flux"
    label_statistic_name: str | None = None

    # ----------------------------------------------------------------------- #
    # ADDITIVE (#800, Li 2021): ``net_flux`` admits a typed ProvenanceGap, so a
    # reference whose flux map the release does not carry (Li 2021 names its
    # control strain's map in the text and figure but releases none) is a typed
    # absence rather than a reason to refuse the three released maps. The field
    # stays required on every record that has a map: None ONLY with the gap.
    # ----------------------------------------------------------------------- #
    net_flux: dict[str, float] | None = Field(
        description="reaction id -> signed net flux in measurement_type's unit. None "
        "ONLY with a ProvenanceGap on 'net_flux' (a map the source names but does not "
        "release, e.g. a control strain's)"
    )
    net_flux_lower: dict[str, float] | None = Field(
        default=None,
        description="reaction id -> lower confidence bound of the fitted flux",
    )
    net_flux_upper: dict[str, float] | None = Field(
        default=None,
        description="reaction id -> upper confidence bound of the fitted flux",
    )
    confidence_level: float | None = Field(
        default=None,
        description="the level the bounds are stated at, as a fraction (e.g. 0.95); "
        "None when the source does not say, which is then a typed gap",
    )
    measurement_type: str = Field(
        description="what the numbers are, including the unit, e.g. "
        "'c13_mfa_net_flux_mmol_per_gdcw_per_h'"
    )
    n_samples: int | None = Field(
        default=None,
        description="independent labeling experiments the fit is over (one fit, so this "
        "is a scalar, not a per-reaction count)",
    )
    sample_unit: SampleUnit | None = Field(
        default=None, description="what one sample in n_samples is"
    )
    target_reaction_ids: dict[str, str] | None = Field(
        default=None,
        description="reaction key -> constraint-based-model reaction id for CBM linkage; "
        "None until that mapping is decided",
    )
    # ----------------------------------------------------------------------- #
    # ADDITIVE (#800, Li 2021): the EXCHANGE fluxes a 13C-MFA fit releases beside
    # its net fluxes. A reversible reaction has a net flux (forward minus
    # backward, above) and an exchange flux (how much it runs both ways), and a
    # labeling fit identifies both; 13CFLUX-family tools release the exchange on
    # their own compactified scale. Stored verbatim keyed by the release's own
    # exchange ids (Li 2021: ``vAco.x``), never converted, and NOT validated to
    # lie between their bounds, so no released row is refused.
    # ----------------------------------------------------------------------- #
    exchange_flux: dict[str, float] | None = Field(
        default=None,
        description="exchange id -> the fitted exchange flux of a reversible reaction, "
        "verbatim on the scale the source releases it; None when the source releases "
        "none",
    )
    exchange_flux_lower: dict[str, float] | None = Field(
        default=None,
        description="exchange id -> lower confidence bound of the exchange flux, at "
        "confidence_level, verbatim",
    )
    exchange_flux_upper: dict[str, float] | None = Field(
        default=None,
        description="exchange id -> upper confidence bound of the exchange flux, at "
        "confidence_level, verbatim",
    )

    @model_validator(mode="after")
    def validate_flux(self) -> "FluxPhenotype":
        """Non-empty fluxes, finite values, bounds that key on and bracket the flux."""
        # --- begin #800: a gapped map carries no values and no bounds ----------- #
        if self.net_flux is None:
            if "net_flux" not in self.gapped_fields():
                raise ValueError(
                    "net_flux is None without a ProvenanceGap on 'net_flux' (an "
                    "absent map is a typed gap, never a silent None)"
                )
            for name in (
                "net_flux_lower",
                "net_flux_upper",
                "exchange_flux",
                "exchange_flux_lower",
                "exchange_flux_upper",
            ):
                if getattr(self, name) is not None:
                    raise ValueError(f"{name} is set on a gapped net_flux map")
            return self
        self._validate_exchange()
        # --- end #800 ------------------------------------------------------------ #
        if not self.net_flux:
            raise ValueError("net_flux cannot be empty")
        for key, value in self.net_flux.items():
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"net_flux for {key} must be finite")
        for name in ("net_flux_lower", "net_flux_upper"):
            mapping = getattr(self, name)
            if mapping is None:
                continue
            for key, value in mapping.items():
                if key not in self.net_flux:
                    raise ValueError(f"{name} key {key} not in net_flux")
                if math.isnan(value) or math.isinf(value):
                    raise ValueError(f"{name} for {key} must be finite")
        if self.net_flux_lower is not None and self.net_flux_upper is not None:
            for key in set(self.net_flux_lower) & set(self.net_flux_upper):
                if self.net_flux_lower[key] > self.net_flux_upper[key]:
                    raise ValueError(
                        f"net_flux bounds for {key} are inverted: "
                        f"{self.net_flux_lower[key]} > {self.net_flux_upper[key]}"
                    )
        for key, lower in (self.net_flux_lower or {}).items():
            if lower > self.net_flux[key]:
                raise ValueError(
                    f"net_flux_lower for {key} exceeds the fitted flux "
                    f"({lower} > {self.net_flux[key]})"
                )
        for key, upper in (self.net_flux_upper or {}).items():
            if upper < self.net_flux[key]:
                raise ValueError(
                    f"net_flux_upper for {key} is below the fitted flux "
                    f"({upper} < {self.net_flux[key]})"
                )
        if self.confidence_level is not None and not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                f"confidence_level is a fraction in (0, 1), got {self.confidence_level}"
            )
        if self.n_samples is not None and self.n_samples < 1:
            raise ValueError(
                f"n_samples must be a positive integer, got {self.n_samples}"
            )
        return self

    # --- begin #800 ------------------------------------------------------------- #
    def _validate_exchange(self) -> None:
        """Finite exchange values; bounds key on stored exchange ids (no bracketing)."""
        exchange = self.exchange_flux
        for name in ("exchange_flux", "exchange_flux_lower", "exchange_flux_upper"):
            mapping = getattr(self, name)
            if mapping is None:
                continue
            if name != "exchange_flux" and exchange is None:
                raise ValueError(f"{name} is set without exchange_flux")
            for key, value in mapping.items():
                if not math.isfinite(value):
                    raise ValueError(f"{name} for {key} must be finite")
                if exchange is not None and key not in exchange:
                    raise ValueError(f"{name} key {key} not in exchange_flux")

    # --- end #800 --------------------------------------------------------------- #


class ReporterReadout(StrEnum):
    """HOW a transcriptional reporter's output was physically read.

    The counterpart of ``AssayType`` for a reporter-fusion readout: the number is a
    signal from a reporter gene, and what instrument produced it decides what the
    number can be compared with. A plate-reader intensity is a well average over a
    whole population; a flow-cytometry intensity is a per-cell distribution the source
    summarized; an enzymatic activity is a rate, not an intensity. Mixing them silently
    would compare a population average with a single-cell summary.

    - ``plate_reader_fluorescence``: well fluorescence read by a microplate reader.
    - ``flow_cytometry_fluorescence``: per-cell fluorescence summarized by the source.
    - ``microscopy_fluorescence``: fluorescence quantified from images.
    - ``luminescence``: a luciferase or other light-emitting reporter.
    - ``enzymatic_activity``: a reporter enzyme's activity (e.g. Miller units).
    - ``sort_seq_fluorescence``: fluorescence read by sorting a POOLED library into
      expression bins and sequencing each bin, so a sequence's level is estimated from
      its read distribution over the bins rather than read off one well or one cell
      (a sort-seq / massively parallel reporter assay; Vaishnav 2022, de Boer 2020).
      Added for #991: the four instrument members above all return a reading per well
      or per cell, which a pooled assay never produces, so a sort-seq estimate stored
      under one of them would claim a sample unit it does not have.
    """

    plate_reader_fluorescence = "plate_reader_fluorescence"
    flow_cytometry_fluorescence = "flow_cytometry_fluorescence"
    microscopy_fluorescence = "microscopy_fluorescence"
    luminescence = "luminescence"
    enzymatic_activity = "enzymatic_activity"
    sort_seq_fluorescence = "sort_seq_fluorescence"  # ADDITIVE (#991)


class PromoterActivityPhenotype(Phenotype, ModelStrict):
    """Transcription from ONE promoter, read as the signal of a reporter fused to it.

    For transcriptional-reporter libraries (Zaslaver-style promoter-GFP collections,
    luciferase or lacZ fusion panels): the strain carries a reporter whose expression is
    driven by one promoter, and the measured number is that reporter's signal. The
    measured entity is the PROMOTER, not the reporter gene and not the strain's fitness,
    which is why this is its own family:

    - ``FitnessPhenotype`` and ``EnvironmentResponsePhenotype`` are growth readouts.
      ``EnvironmentResponsePhenotype`` additionally requires an environmental edit on
      every record (its verifier's ``environment_perturbed`` rule), so it cannot carry
      the untreated arm of a reporter time course, which is half of what such a screen
      releases.
    - the three expression families (``MicroarrayExpressionPhenotype``,
      ``RNASeqExpressionPhenotype``, ``PseudobulkExpressionPhenotype``) are genome-wide
      dicts of one strain's transcriptome. A reporter library inverts that: one promoter
      per strain, measured in many strains, and the reporter is a proxy for the
      promoter's transcription rather than a count of the native transcript.

    ``promoter_activity`` is the signal AS RELEASED, in ``activity_units``: an absolute
    intensity when the source released one, a normalized or background-subtracted value
    when the source released that, and nothing is converted here. A ratio against a
    control arm is NOT stored, because it is a function of two stored records (the
    treated and the control reading of the same promoter at the same time) and storing it
    beside them would store one measurement twice.

    ``promoter_name`` is the source's own label for the promoter, verbatim.
    ``promoter_gene`` is the gene identifier that label resolves to in the record's
    pinned assembly, which is what joins the record to a gene node; it is ``None`` when
    the label names no single gene, never a guess. ``well_id`` carries the source's own
    plate/well (or tube) identifier: a library routinely holds one promoter in several
    wells, and without it two independent readings of one promoter under one condition
    would collide into a single L1 identity instead of standing as two measurements.

    The uncertainty ontology mirrors ``FitnessPhenotype``: ``promoter_activity_se`` is
    the DERIVED, ML-facing SE, auto-filled from the source-reported uncertainty and its
    type via ``derive_se``.
    """

    graph_level: str = "node"
    label_name: str = "promoter_activity"
    label_statistic_name: str | None = "promoter_activity_se"

    promoter_activity: float = Field(
        description="the reporter signal as released, in activity_units"
    )
    promoter_activity_se: float | None = Field(
        default=None,
        description="standard error of the signal (primary uncertainty statistic)",
    )
    promoter_activity_uncertainty: float | None = Field(
        default=None,
        description="source-reported uncertainty number, verbatim (meaning given by "
        "promoter_activity_uncertainty_type)",
    )
    promoter_activity_uncertainty_type: UncertaintyType | None = Field(
        default=None,
        description="what promoter_activity_uncertainty IS (sample_sd, ...)",
    )
    n_samples: int | None = Field(
        default=None,
        description="number of independent replicate measurements of the signal",
    )
    sample_unit: SampleUnit | None = Field(
        default=None,
        description="what one sample in n_samples is (biological_replicate, ...)",
    )
    promoter_name: str = Field(
        description="the source's own label for the measured promoter, verbatim"
    )
    promoter_gene: str | None = Field(
        default=None,
        description="the gene identifier the promoter label resolves to in the pinned "
        "assembly (the join key to a gene node); None when the label names no single "
        "gene, never a guess",
    )
    readout: ReporterReadout = Field(
        description="HOW the reporter signal was read (plate reader, cytometer, ...)"
    )
    reporter_gene: str = Field(
        description="the reporter the promoter drives, verbatim (e.g. 'gfpmut2')"
    )
    activity_units: str = Field(
        description="human-readable definition and units of the signal, including "
        "whether it is background-subtracted or normalized"
    )
    well_id: str = Field(
        description="the source's own plate/well (or tube) identifier of this reading, "
        "which is what keeps two wells of one promoter label L1-distinct"
    )

    @field_validator("promoter_activity")
    def validate_activity(cls, v: float) -> float:
        """Reject a non-finite signal (NaN or infinity is not a measurement)."""
        if not math.isfinite(v):
            raise ValueError(f"promoter_activity must be finite, got {v}")
        return v

    @field_validator("n_samples")
    def validate_n_samples(cls, v: int | None) -> int | None:
        """n_samples is a positive integer or None."""
        if v is not None and (not isinstance(v, int) or v < 1):
            raise ValueError(f"n_samples must be a positive integer or None, got: {v}")
        return v

    @model_validator(mode="before")
    @classmethod
    def _fill_activity_se(cls, data: Any) -> Any:
        """Derive the ML-facing SE from the reported uncertainty (frozen -> fill first)."""
        if not isinstance(data, dict):
            return data
        unc = data.get("promoter_activity_uncertainty")
        typ = data.get("promoter_activity_uncertainty_type")
        if unc is None or typ is None or data.get("promoter_activity_se") is not None:
            return data
        typ = UncertaintyType(typ)
        n = data.get("n_samples")
        if typ in (UncertaintyType.sample_sd, UncertaintyType.variance) and n is None:
            return data
        data["promoter_activity_se"] = derive_se(unc, typ, n)
        return data

    @model_validator(mode="after")
    def _check(self) -> "PromoterActivityPhenotype":
        """Enforce the uncertainty invariant and non-empty identifiers."""
        if not self.promoter_name.strip():
            raise ValueError("promoter_name cannot be empty")
        if not self.well_id.strip():
            raise ValueError("well_id cannot be empty")
        if not self.reporter_gene.strip():
            raise ValueError("reporter_gene cannot be empty")
        if not self.activity_units.strip():
            raise ValueError("activity_units cannot be empty")
        unc, typ = (
            self.promoter_activity_uncertainty,
            self.promoter_activity_uncertainty_type,
        )
        if (unc is None) != (typ is None):
            raise ValueError(
                "promoter_activity_uncertainty and its type must both be set or both "
                "be None (no unlabelled uncertainty)"
            )
        if typ in (UncertaintyType.sample_sd, UncertaintyType.variance) and (
            self.n_samples is None or self.sample_unit is None
        ):
            raise ValueError(f"n_samples and sample_unit are required for {typ}")
        return self


# --------------------------------------------------------------------------- #
# Experiment families first landed for bacteria (plan 3b, 3c), host-neutral since the
# ontology unification (P2).
#
# Until P2 every reference below narrowed ``genome_reference`` to an
# ``AssemblyReferenceGenome`` subclass, because pydantic v2 serializes a field by its
# DECLARED type and the assembly pin lived on that subclass; the same defect forced a
# cloned ``Bacterial*`` pair beside every general family whose phenotype bacteria
# reused. The pin now lives on ``ReferenceGenome`` itself and the base slot is
# ``ReferenceGenome | StrainReferenceGenome``, so a family is one pair for every host,
# and a fitness measurement on BW25113 is a ``FitnessExperiment`` like one on BY4741.
#
# ``environment`` stays the shared ``Environment`` everywhere except the product-titer
# pair, so the cross-dataset medium-level aggregate still forms across hosts. The
# product-titer pair declares
# ``CultureEnvironment`` (the same narrowing the chemogenomic family makes) because a
# titer cannot be read without its vessel: the fermentation format is part of what was
# measured, and a ``CultureEnvironment`` in an ``Environment``-typed slot dumps without
# ``culture_format``. ``media`` is the same field holding the same ``Media``, and a
# culture environment that states no protocol has the plain environment's identity, so
# the aggregate is unchanged.
# --------------------------------------------------------------------------- #
class ProductTiterExperimentReference(ExperimentReference, ModelStrict):
    """Reference (parent-strain) context for a product-titer experiment."""

    experiment_reference_type: str = "product_titer"
    # Narrowed: the control culture states its vessel, as the experiment does.
    environment_reference: CultureEnvironment
    phenotype_reference: ProductTiterPhenotype


class ProductTiterExperiment(Experiment, ModelStrict):
    """Experiment measuring how much product a strain made."""

    experiment_type: str = "product_titer"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    environment: CultureEnvironment  # narrowed: a titer is read with its vessel
    phenotype: ProductTiterPhenotype


class ProteinTurnoverExperimentReference(ExperimentReference, ModelStrict):
    """Reference (control) context for a protein-turnover experiment."""

    experiment_reference_type: str = "protein_turnover"
    phenotype_reference: ProteinTurnoverPhenotype


class ProteinTurnoverExperiment(Experiment, ModelStrict):
    """Experiment measuring per-protein turnover rates."""

    experiment_type: str = "protein_turnover"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: ProteinTurnoverPhenotype


# --- begin #857: the protein synthesis-rate family ---------------------------- #
class ProteinSynthesisRateExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a protein synthesis-rate experiment."""

    experiment_reference_type: str = "protein_synthesis_rate"
    phenotype_reference: ProteinSynthesisRatePhenotype


class ProteinSynthesisRateExperiment(Experiment, ModelStrict):
    """Experiment measuring per-protein synthesis rates."""

    experiment_type: str = "protein_synthesis_rate"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: ProteinSynthesisRatePhenotype


# --- end #857 ------------------------------------------------------------------- #


class FluxExperimentReference(ExperimentReference, ModelStrict):
    """Reference (parent-strain) context for a flux experiment."""

    experiment_reference_type: str = "flux"
    phenotype_reference: FluxPhenotype


class FluxExperiment(Experiment, ModelStrict):
    """Experiment measuring a fitted net-flux distribution."""

    experiment_type: str = "flux"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: FluxPhenotype


class PromoterActivityExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a promoter-activity experiment.

    The reference of a reporter reading is the SAME reporter strain in the control arm
    of the same screen -- the untreated culture, read at the same time -- so the
    reference is a real measurement, not a derived baseline.
    """

    experiment_reference_type: str = "promoter_activity"
    phenotype_reference: PromoterActivityPhenotype


class PromoterActivityExperiment(Experiment, ModelStrict):
    """Experiment measuring transcription from one promoter through a reporter fusion."""

    experiment_type: str = "promoter_activity"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: PromoterActivityPhenotype


# --------------------------------------------------------------------------- #
# begin ADDITIVE (#991 Vaishnav 2022): a POOLED sequence-to-expression readout.
#
# A sort-seq / massively parallel reporter assay measures a whole library at once: the
# pooled culture is sorted into expression bins, every bin is sequenced, and each
# sequence's expression level is estimated from its own read distribution over the bins.
# ``PromoterActivityPhenotype`` (served: Mohiuddin 2022) cannot hold that, and the honest
# reason is its ``well_id``, which is REQUIRED there for a measured reason: an arrayed
# reporter library holds one promoter in several wells, so without the well two readings
# of one promoter under one condition would collide into a single L1 identity. A pooled
# library has no well at all. Making ``well_id`` optional would make the served class's
# identity rule conditional for every record that already carries one, so this is a
# SIBLING leaf under ``Phenotype`` whose identity key is the sequence the release names.
#
# The two leaves are the same technology at one level (a fluorescent reporter's signal
# stands for transcription from the promoter driving it) and genuinely different at the
# level that decides what the number can be compared with: a plate reader or cytometer
# returns one reading per well, while a sort-seq estimate is a read-weighted average of
# bin indices for one sequence out of millions. ``ReporterReadout`` gains
# ``sort_seq_fluorescence`` to name that instrument.
# --------------------------------------------------------------------------- #
class PooledSortSeqExpressionPhenotype(Phenotype, ModelStrict):
    """Expression of ONE library sequence, estimated from its reads across sort bins.

    For a sort-seq massively parallel reporter assay (Vaishnav 2022 / de Boer 2020 GPRA):
    the strain carries a reporter whose promoter holds the library sequence, the pooled
    culture is sorted into expression bins by the reporter's signal, every bin is
    sequenced, and the sequence's expression level is the average of the bins it was
    observed in, weighted by how often it was observed in each. The measured entity is
    the SEQUENCE, which is why this is its own family:

    - ``PromoterActivityPhenotype`` is an arrayed reporter library: one reading per well,
      and its required ``well_id`` is what keeps two wells of one promoter L1-distinct. A
      pooled library has no well, and the key that keeps two readings distinct is the
      source's own sequence identifier.
    - the expression families (``MicroarrayExpressionPhenotype``,
      ``RNASeqExpressionPhenotype``, ``PseudobulkExpressionPhenotype``) are genome-wide
      dicts of one strain's transcriptome. Here one construct is measured in a pool of
      millions, and the number is a reporter proxy for transcription from the sequence,
      not a count of a native transcript.

    ``expression_level`` is the estimate AS RELEASED, in ``expression_units``: nothing is
    rescaled here, because the bin scale is set by the sorter's configuration on the day
    (de Boer 2020 Methods: "The FACS configuration varied between experiments (for
    example, different laser intensities), resulting in different baseline expression
    values"), so a value is comparable within its library and not across sorts.

    ``sequence_id`` is the source's own identifier of the measured sequence, verbatim;
    the sequence itself lives on the genotype
    (``VariablePromoterReporterPerturbation.promoter_sequence``), so it is stored once.
    ``library_id`` is the source's own label for the library or sort the reading came
    from, ``None`` when the release carries none; ``read_count`` is the number of
    observations the estimate averaged, which is the release's own measure of how well
    determined it is (de Boer 2020 restricts its high-quality analysis to sequences with
    at least 100 reads).

    A release ROW whose estimate cell is empty or ``NA`` is still a record: the
    sequence was designed, cloned and listed, and what the release says about it is
    that it has no estimate. ``expression_level`` is then ``None`` beside a
    ``ProvenanceGap`` on that field quoting the cell, never a refused row and never a
    filled value. A sequence the release lists no row for is not a record at all.

    The uncertainty ontology mirrors the sibling: ``expression_level_se`` is the DERIVED,
    ML-facing SE, auto-filled from a source-reported uncertainty and its type via
    ``derive_se``. A source that releases a dispersion of unstated kind leaves both None
    with a ``ProvenanceGap``, never a typed guess.
    """

    graph_level: str = "node"
    label_name: str = "expression_level"
    label_statistic_name: str | None = "expression_level_se"

    expression_level: float | None = Field(
        description="the expression estimate as released, in expression_units; None "
        "only beside a ProvenanceGap, for a release row whose estimate cell is empty "
        "or 'NA'"
    )
    expression_level_se: float | None = Field(
        default=None,
        description="standard error of the estimate (primary uncertainty statistic)",
    )
    expression_level_uncertainty: float | None = Field(
        default=None,
        description="source-reported uncertainty number, verbatim (meaning given by "
        "expression_level_uncertainty_type)",
    )
    expression_level_uncertainty_type: UncertaintyType | None = Field(
        default=None,
        description="what expression_level_uncertainty IS (sample_sd, ...)",
    )
    n_samples: int | None = Field(
        default=None,
        description="number of independent replicate measurements of the estimate; a "
        "read is NOT a replicate (see read_count)",
    )
    sample_unit: SampleUnit | None = Field(
        default=None,
        description="what one sample in n_samples is (biological_replicate, ...)",
    )
    sequence_id: str = Field(
        description="the source's own identifier of the measured sequence, verbatim; "
        "the identity key a pooled library has in place of a well"
    )
    library_id: str | None = Field(
        default=None,
        description="the source's own label for the library or sort this reading came "
        "from (e.g. Vaishnav 2022 expt 'NBT_S288CdU_SCUra'); None when unreleased",
    )
    read_count: int | None = Field(
        default=None,
        description="observations (reads) the estimate averaged over, as released; None "
        "when the release carries no per-sequence count",
    )
    readout: ReporterReadout = Field(
        description="HOW the reporter signal was read (sort_seq_fluorescence, ...)"
    )
    reporter_gene: str = Field(
        description="the reporter the library sequence drives, verbatim (e.g. 'YFP')"
    )
    expression_units: str = Field(
        description="human-readable definition and units of the estimate, including the "
        "bin scale and what it is normalized against"
    )

    @field_validator("expression_level")
    def validate_expression_level(cls, v: float | None) -> float | None:
        """Reject a non-finite estimate (NaN or infinity is not a measurement)."""
        if v is not None and not math.isfinite(v):
            raise ValueError(f"expression_level must be finite, got {v}")
        return v

    @field_validator("n_samples")
    def validate_n_samples(cls, v: int | None) -> int | None:
        """n_samples is a positive integer or None."""
        if v is not None and (not isinstance(v, int) or v < 1):
            raise ValueError(f"n_samples must be a positive integer or None, got: {v}")
        return v

    @field_validator("read_count")
    def validate_read_count(cls, v: int | None) -> int | None:
        """A released read count is a positive integer (an estimate needs one read)."""
        if v is not None and (not isinstance(v, int) or v < 1):
            raise ValueError(f"read_count must be a positive integer or None, got: {v}")
        return v

    @model_validator(mode="before")
    @classmethod
    def _fill_expression_se(cls, data: Any) -> Any:
        """Derive the ML-facing SE from the reported uncertainty (frozen -> fill first)."""
        if not isinstance(data, dict):
            return data
        unc = data.get("expression_level_uncertainty")
        typ = data.get("expression_level_uncertainty_type")
        if unc is None or typ is None or data.get("expression_level_se") is not None:
            return data
        typ = UncertaintyType(typ)
        n = data.get("n_samples")
        if typ in (UncertaintyType.sample_sd, UncertaintyType.variance) and n is None:
            return data
        data["expression_level_se"] = derive_se(unc, typ, n)
        return data

    @model_validator(mode="after")
    def _check(self) -> "PooledSortSeqExpressionPhenotype":
        """Enforce the uncertainty invariant, non-empty identifiers, value-or-gap."""
        _require_value_or_gap(self, ("expression_level",))
        if not self.sequence_id.strip():
            raise ValueError("sequence_id cannot be empty")
        if self.library_id is not None and not self.library_id.strip():
            raise ValueError("library_id is a released label or None, never blank")
        if not self.reporter_gene.strip():
            raise ValueError("reporter_gene cannot be empty")
        if not self.expression_units.strip():
            raise ValueError("expression_units cannot be empty")
        unc, typ = (
            self.expression_level_uncertainty,
            self.expression_level_uncertainty_type,
        )
        if (unc is None) != (typ is None):
            raise ValueError(
                "expression_level_uncertainty and its type must both be set or both be "
                "None (no unlabelled uncertainty)"
            )
        if typ in (UncertaintyType.sample_sd, UncertaintyType.variance) and (
            self.n_samples is None or self.sample_unit is None
        ):
            raise ValueError(f"n_samples and sample_unit are required for {typ}")
        return self


class PooledSortSeqExpressionExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a pooled sort-seq expression experiment."""

    experiment_reference_type: str = "pooled_sort_seq_expression"
    phenotype_reference: PooledSortSeqExpressionPhenotype


class PooledSortSeqExpressionExperiment(Experiment, ModelStrict):
    """Experiment measuring one library sequence's expression in a pooled sort-seq."""

    experiment_type: str = "pooled_sort_seq_expression"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: PooledSortSeqExpressionPhenotype


# --------------------------------------------------------------------------- #
# end ADDITIVE (#991)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# begin ADDITIVE (#990 Zhang 2020): a BIOSENSOR fluorescence readout.
#
# Zhang 2020's reporter is a trpR-based biosensor: its GFP reports the strain's
# INTRACELLULAR TRYPTOPHAN, not transcription from any of the five varied promoters. Two
# existing homes would both misname it:
#
#   - ``PromoterActivityPhenotype`` (and the pooled sibling above) say the measured
#     entity is the promoter driving the reporter. Here the promoters are the genotype's
#     independent variable and the reporter answers a metabolite, so attributing the
#     signal to the promoter would assert a measurement nobody made.
#   - ``MeasurementType.biosensor_readout`` exists, but it lives on
#     ``EnvironmentResponsePhenotype``, whose verifier requires an environmental edit on
#     every record; a biosensor strain grown in one medium has none.
#
# ``MetabolitePhenotype`` is the nearest neighbour and stays what it is: a metabolite
# QUANTITY (titer, pool size) in its own units. A biosensor signal is a proxy whose
# calibration to concentration the release may never state, so storing it as a metabolite
# quantity would silently promote a fluorescence rate to a concentration. This leaf keeps
# the signal in its own units and NAMES the sensed metabolite as a typed ``Compound``, so
# the record joins the shared compound layer without claiming to be a titer. The
# conversion between the two, where a release publishes a calibration, is a conversion
# entry, not a reinterpretation of the stored number.
# --------------------------------------------------------------------------- #
class BiosensorFluorescencePhenotype(Phenotype, ModelStrict):
    """A biosensor reporter's signal, standing for the metabolite it senses.

    The strain carries a sensor (a transcription factor plus its operator, an RNA
    aptamer, an enzyme-coupled device) fused to a fluorescent reporter, and the released
    number is that reporter's signal, which responds to the intracellular level of
    ``sensed_metabolite``. ``biosensor_signal`` is the value AS RELEASED, in
    ``signal_units`` (Zhang 2020: a specific GFP synthesis rate in MFI per hour), and
    nothing is converted to a concentration here: a signal becomes a concentration only
    through a calibration the release has to publish.

    ``sensor_name`` is the device as the source names it, so two releases using different
    sensors for one metabolite stay distinguishable (their dynamic ranges differ);
    ``reporter_gene`` is the fluorophore; ``readout`` names the instrument, which decides
    what the number can be compared with (a plate-reader population average is not a
    cytometer's per-cell summary).
    """

    graph_level: str = "node"
    label_name: str = "biosensor_signal"
    label_statistic_name: str | None = "biosensor_signal_se"

    biosensor_signal: float = Field(
        description="the biosensor reporter's signal as released, in signal_units"
    )
    biosensor_signal_se: float | None = Field(
        default=None,
        description="standard error of the signal (primary uncertainty statistic)",
    )
    biosensor_signal_uncertainty: float | None = Field(
        default=None,
        description="source-reported uncertainty number, verbatim (meaning given by "
        "biosensor_signal_uncertainty_type)",
    )
    biosensor_signal_uncertainty_type: UncertaintyType | None = Field(
        default=None,
        description="what biosensor_signal_uncertainty IS (sample_sd, ...)",
    )
    n_samples: int | None = Field(
        default=None,
        description="number of independent replicate measurements of the signal",
    )
    sample_unit: SampleUnit | None = Field(
        default=None,
        description="what one sample in n_samples is (biological_replicate, ...)",
    )
    sensed_metabolite: Compound = Field(
        description="the metabolite the sensor reports, as a typed Compound (the join "
        "key into the shared compound layer); NOT a claim that a concentration was read"
    )
    sensor_name: str = Field(
        description="the biosensor device as the source names it, verbatim (e.g. "
        "'trpR-based tryptophan biosensor')"
    )
    reporter_gene: str = Field(
        description="the reporter the sensor drives, verbatim (e.g. 'GFP')"
    )
    readout: ReporterReadout = Field(
        description="HOW the reporter signal was read (plate reader, cytometer, ...)"
    )
    signal_units: str = Field(
        description="human-readable definition and units of the signal, including "
        "whether it is a rate, normalized, or background-subtracted"
    )

    @field_validator("biosensor_signal")
    def validate_signal(cls, v: float) -> float:
        """Reject a non-finite signal (NaN or infinity is not a measurement)."""
        if not math.isfinite(v):
            raise ValueError(f"biosensor_signal must be finite, got {v}")
        return v

    @field_validator("n_samples")
    def validate_n_samples(cls, v: int | None) -> int | None:
        """n_samples is a positive integer or None."""
        if v is not None and (not isinstance(v, int) or v < 1):
            raise ValueError(f"n_samples must be a positive integer or None, got: {v}")
        return v

    @model_validator(mode="before")
    @classmethod
    def _fill_signal_se(cls, data: Any) -> Any:
        """Derive the ML-facing SE from the reported uncertainty (frozen -> fill first)."""
        if not isinstance(data, dict):
            return data
        unc = data.get("biosensor_signal_uncertainty")
        typ = data.get("biosensor_signal_uncertainty_type")
        if unc is None or typ is None or data.get("biosensor_signal_se") is not None:
            return data
        typ = UncertaintyType(typ)
        n = data.get("n_samples")
        if typ in (UncertaintyType.sample_sd, UncertaintyType.variance) and n is None:
            return data
        data["biosensor_signal_se"] = derive_se(unc, typ, n)
        return data

    @model_validator(mode="after")
    def _check(self) -> "BiosensorFluorescencePhenotype":
        """Enforce the uncertainty invariant and non-empty identifiers."""
        for name in ("sensor_name", "reporter_gene", "signal_units"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")
        if not self.sensed_metabolite.name.strip():
            raise ValueError("sensed_metabolite must name a compound")
        unc, typ = (
            self.biosensor_signal_uncertainty,
            self.biosensor_signal_uncertainty_type,
        )
        if (unc is None) != (typ is None):
            raise ValueError(
                "biosensor_signal_uncertainty and its type must both be set or both be "
                "None (no unlabelled uncertainty)"
            )
        if typ in (UncertaintyType.sample_sd, UncertaintyType.variance) and (
            self.n_samples is None or self.sample_unit is None
        ):
            raise ValueError(f"n_samples and sample_unit are required for {typ}")
        return self


class BiosensorFluorescenceExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a biosensor fluorescence experiment.

    The reference is the platform strain of the same screen read with the same sensor,
    which is a real measurement rather than a derived baseline. A yeast pair, so
    ``genome_reference`` is the plain ``ReferenceGenome``.
    """

    experiment_reference_type: str = "biosensor_fluorescence"
    phenotype_reference: BiosensorFluorescencePhenotype


class BiosensorFluorescenceExperiment(Experiment, ModelStrict):
    """Experiment reading a metabolite through a biosensor's fluorescent reporter."""

    experiment_type: str = "biosensor_fluorescence"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: BiosensorFluorescencePhenotype


# --------------------------------------------------------------------------- #
# end ADDITIVE (#990)
# --------------------------------------------------------------------------- #


# --- begin #909: the graded essentiality sibling ------------------------------ #
class GradedGeneEssentialityExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a graded gene-essentiality experiment."""

    experiment_reference_type: str = "graded_gene_essentiality"
    phenotype_reference: GradedGeneEssentialityPhenotype


class GradedGeneEssentialityExperiment(Experiment, ModelStrict):
    """Categorical essentiality calls (an HMM state, an unclear TraDIS call); first
    landed for bacteria (Goodall 2018, Royet 2025), host-neutral by construction.
    """

    experiment_type: str = "graded_gene_essentiality"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: GradedGeneEssentialityPhenotype


# --- end #909 ---------------------------------------------------------------- #


# --- begin #770: the protein fold-change family -------------------------------- #
# --- end #770 ------------------------------------------------------------------- #


# --- begin #921: the phosphosite intensity family ------------------------------ #
class PhosphositeIntensityExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a phosphosite-intensity experiment."""

    experiment_reference_type: str = "phosphosite_intensity"
    phenotype_reference: PhosphositeIntensityPhenotype


class PhosphositeIntensityExperiment(Experiment, ModelStrict):
    """Phosphoproteome experiment: per-site intensities in one sample (Potel 2018 first;
    Leutert 2023 on yeast releases the same per-sample shape).
    """

    experiment_type: str = "phosphosite_intensity"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: PhosphositeIntensityPhenotype


# --- end #921 ------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# BACTERIAL MORPHOLOGY BLOCK START (issue #774)
# A multi-feature bacterial cell-morphology profile: the CalMorph shape with the
# vocabulary lifted out of the yeast imaging program and into the assay.
# --------------------------------------------------------------------------- #
class BacterialMorphologyPhenotype(Phenotype, ModelStrict):
    """A bacterial strain's cell-morphology profile from one microscopy assay.

    ``CalMorphPhenotype`` models exactly this: a dict of named per-strain values plus a
    dict of named coefficients of variation. Its vocabulary is the wrong one, and not by
    oversight: ``CALMORPH_LABELS`` and ``CALMORPH_STATISTICS`` are the 281 base and 220
    CV parameters of CalMorph, a *S. cerevisiae* image-analysis program, and that set was
    measured to be disjoint from every symbol a bacterial screen releases. Storing
    ``<L>`` under a CalMorph key would assert a measurement nobody made.

    So the shape is kept and the vocabulary becomes a property of the ASSAY.
    ``assay`` names a :class:`~torchcell.datamodels.bacterial_morphology_features.MorphologyAssay`
    in ``MORPHOLOGY_ASSAYS``, that assay's features are the only permitted keys, and each
    feature declares WHAT STATISTIC its number is. The split between the two dicts is
    that declaration and not a naming convention: a key whose statistic is
    ``coefficient_of_variation`` belongs in ``morphology_coefficient_of_variation`` and
    every other key belongs in ``morphology``, so a CV cannot be filed as a mean. A later
    bacterial imaging screen registers its own assay and widens nobody else's label set.

    ``morphology`` therefore holds several statistics side by side (a mean over the
    strain's cells, a Pearson correlation across them, a fitted intercept, a population
    fraction, an inferred relative timing), which is what the source releases in one row.
    The assay says which is which, and a consumer that pools them across keys is pooling
    quantities the source never claimed were comparable.

    A record carries the features the source DETERMINED for that strain, not necessarily
    the whole vocabulary: Campos 2018's release states that "NaN (Not a Number) values
    are attributed to non-determined fields", and 278 of its 4,227 strains have no
    nucleoid channel, so their 7 nucleoid-derived features do not exist. Demanding full
    coverage would drop 21 real features to save 7 absent ones.

    ``n_samples`` with ``sample_unit`` is the replicate design of the per-cell features:
    the number of segmented cells the means and CVs are computed over. Some features are
    computed over a stated SUBSET of those cells (Campos's ``CV_DR`` and ``rho_CD`` over
    constricted cells only, its nucleoid features over DAPI-scored cells), and the
    subset sizes are not released per strain; each such feature says so in its own
    ``note`` rather than this field pretending to cover them.
    """

    graph_level: str = "global"
    label_name: str = "morphology"
    label_statistic_name: str = "morphology_coefficient_of_variation"
    assay: str = Field(
        description="the registered morphology assay whose vocabulary names the keys "
        "(a key of torchcell.datamodels.bacterial_morphology_features.MORPHOLOGY_ASSAYS)"
    )
    morphology: dict[str, float] = Field(
        description="per-strain feature values keyed by the assay's own symbol, for "
        "every feature of the assay whose statistic is not a coefficient of variation"
    )
    morphology_coefficient_of_variation: dict[str, float] | None = Field(
        default=None,
        description="per-strain coefficients of variation keyed by the assay's own "
        "symbol, for the features whose declared statistic is a coefficient of variation",
    )
    n_samples: int | None = Field(
        default=None,
        description="number of samples the per-cell features are computed over",
    )
    sample_unit: SampleUnit | None = Field(
        default=None, description="what one sample in n_samples physically is"
    )

    @property
    def assay_vocabulary(self) -> MorphologyAssay:
        """The assay this record names, which is the authority on its keys."""
        return morphology_assay(self.assay)

    @model_validator(mode="after")
    def validate_against_the_assay(self) -> "BacterialMorphologyPhenotype":
        """Every key is a feature of the named assay, filed under its own statistic.

        The assay lookup raises on an unregistered name, so a record can never be
        written against a vocabulary this version of the schema does not hold.
        """
        assay = morphology_assay(self.assay)
        if not self.morphology:
            raise ValueError(
                f"morphology measurements cannot be empty (assay {self.assay!r})"
            )
        self._check_dict("morphology", self.morphology, assay.value_symbols, assay)
        self._check_dict(
            "morphology_coefficient_of_variation",
            self.morphology_coefficient_of_variation or {},
            assay.coefficient_of_variation_symbols,
            assay,
        )
        return self

    @staticmethod
    def _check_dict(
        field: str,
        values: dict[str, float],
        permitted: frozenset[str],
        assay: MorphologyAssay,
    ) -> None:
        """Reject a key outside ``permitted``, a key of the wrong statistic, and NaN."""
        for key, value in values.items():
            if key not in permitted:
                feature = assay.by_symbol.get(key)
                if feature is None:
                    raise ValueError(
                        f"Invalid {assay.name} morphology feature: {key}. Must be one "
                        f"of the {len(assay.features)} symbols the assay declares."
                    )
                raise ValueError(
                    f"{key} is a {feature.statistic.value} of assay {assay.name}, so it "
                    f"does not belong in {field}"
                )
            if math.isnan(value):
                raise ValueError(f"{field} measurement {key} cannot be NaN")


class MorphologyExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a cell-morphology experiment read through an assay
    vocabulary (``BacterialMorphologyPhenotype``; the phenotype's own rename is P5).
    """

    experiment_reference_type: str = "morphology"
    phenotype_reference: BacterialMorphologyPhenotype


class MorphologyExperiment(Experiment, ModelStrict):
    """A strain's cell-morphology profile from one microscopy screen."""

    experiment_type: str = "morphology"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: BacterialMorphologyPhenotype


# --------------------------------------------------------------------------- #
# BACTERIAL MORPHOLOGY BLOCK END (issue #774)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# MRNA NUMBER FRACTION BLOCK START (issue #854)
# A count-less transcriptome: each gene's share of the cell's mRNA molecules, released
# as a fraction with no read counts behind it.
# --------------------------------------------------------------------------- #
#: How far above 1 a record's fractions may sum. A released column sums to 1 over every
#: row of the release; a record keeps a subset of those rows, so its sum is at most 1,
#: and the tolerance is the release's own rounding (Balakrishnan 2022 columns sum to 1
#: within 2.3e-8, measured on the pinned workbook).
MRNA_NUMBER_FRACTION_SUM_ATOL = 1e-6


class MrnaNumberFractionPhenotype(Phenotype, ModelStrict):
    """Per-gene mRNA NUMBER FRACTION: one gene's mRNA molecules over all mRNA molecules.

    The quantity is ``psi_m,i = [mR_i] / [mR]`` with ``[mR] = sum_i [mR_i]``, as
    Balakrishnan et al. 2022 (Science 378, eabk2066) define it, estimated from RNA-seq and
    released WITHOUT the read counts it was estimated from. It is dimensionless, lies in
    [0, 1], and a released column sums to 1 over the whole transcriptome.

    Why a sibling of ``RNASeqExpressionPhenotype`` and not a relaxation of it: that class
    requires ``expression_count`` with the same keys as ``expression_tpm``, and every
    served record of it (Caudal 2024, Caglar 2017, Lamoureux 2023, Lim 2022) carries
    counts. Making the counts optional would move the schema closure of all four
    served transcriptomes for one release that has none, and storing ``psi x 1e6`` in a
    field named ``expression_tpm`` would assert a TPM pipeline (length normalization of
    read counts) that the release does not document. This class holds exactly what was
    released and implies no counts, so it is additive: no served record changes.

    Key semantics:

    - A gene that is a KEY carries the released fraction verbatim. ``0.0`` is a released
      value (no reads assigned to that gene in that library), not a missing one: the
      column sums to 1 over its zero rows too.
    - A gene that is NOT a key was not released under an identifier that names a locus
      of the pinned assembly. Because a record holds a subset of the released rows, its
      fractions sum to at most 1 (``MRNA_NUMBER_FRACTION_SUM_ATOL`` above it is refused).

    ``n_libraries`` is how many sequenced libraries the stored value is taken over: 1 for
    a released sample column, more for a reference that averages biological replicates.
    """

    graph_level: str = "node"
    label_name: str = "mrna_number_fraction"
    label_statistic_name: str | None = None

    mrna_number_fraction: dict[str, float] = Field(
        description=(
            "SortedDict of per-gene mRNA number fraction (dimensionless, in [0, 1]); a "
            "gene the release does not identify on the pinned assembly is not a key"
        ),
        repr=False,
    )
    n_libraries: int = Field(
        description="number of sequenced libraries the stored fractions are taken over"
    )
    measurement_type: str = Field(
        description="what the number is and the pipeline that produced it, e.g. "
        "'rnaseq_mrna_number_fraction'"
    )

    def __repr__(self) -> str:
        """Summary repr instead of dumping the per-gene dict."""
        return (
            f"MrnaNumberFractionPhenotype(genes={len(self.mrna_number_fraction)}, "
            f"n_libraries={self.n_libraries})"
        )

    @field_validator("mrna_number_fraction", mode="before")
    @classmethod
    def convert_and_validate_fraction(cls, v: Any) -> Any:  # raw pre-validation input
        """Coerce to a SortedDict; refuse empty, non-finite, or out-of-[0, 1] values."""
        if not isinstance(v, dict):
            raise ValueError(
                f"mrna_number_fraction must be a per-gene dict, got {type(v).__name__}"
            )
        if not v:
            raise ValueError("mrna_number_fraction cannot be empty")
        if not isinstance(v, SortedDict):
            v = SortedDict(v)
        total = 0.0
        for key, value in v.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"mrna_number_fraction for {key} is not a number")
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(
                    f"mrna_number_fraction for {key} must be a finite value in [0, 1], "
                    f"got {value}"
                )
            total += float(value)
        if total > 1.0 + MRNA_NUMBER_FRACTION_SUM_ATOL:
            raise ValueError(
                f"mrna_number_fraction sums to {total}, above 1: a subset of one "
                "transcriptome's number fractions cannot exceed the whole"
            )
        return v

    @field_validator("n_libraries", mode="after")
    @classmethod
    def validate_n_libraries(cls, v: int) -> int:
        """At least one library stands behind every stored value."""
        if v < 1:
            raise ValueError(f"n_libraries must be a positive integer, got {v}")
        return v

    @field_validator("measurement_type", mode="after")
    @classmethod
    def validate_measurement_type(cls, v: str) -> str:
        """The quantity is named, never blank."""
        if not v.strip():
            raise ValueError("measurement_type cannot be blank")
        return v


class MrnaNumberFractionExperimentReference(ExperimentReference, ModelStrict):
    """Assembly-pinned reference for an mRNA number-fraction experiment."""

    experiment_reference_type: str = "mrna_number_fraction"
    phenotype_reference: MrnaNumberFractionPhenotype


class MrnaNumberFractionExperiment(Experiment, ModelStrict):
    """A transcriptome released as per-gene mRNA number fractions, without counts."""

    experiment_type: str = "mrna_number_fraction"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: MrnaNumberFractionPhenotype


# --------------------------------------------------------------------------- #
# MRNA NUMBER FRACTION BLOCK END (issue #854)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# FITNESS CONTRAST BLOCK (issue #913). ADDITIVE: a new leaf under ``Phenotype``,
# the sibling of ``EnvironmentResponsePhenotype``, with its own experiment pair.
# Nothing above this block changes.
# --------------------------------------------------------------------------- #
class FitnessContrastPhenotype(Phenotype, ModelStrict):
    """A released significance test of one gene's fitness between two culture arms.

    Borchert 2023 releases, for every (gene, comparison sheet), a two-sample t test of
    the gene's RB-TnSeq fitness in a condition arm against a reference arm: the
    t-statistic, its p-value, the pFDR q-value and the monotonicity-adjusted q-value.
    That quadruple tests a DIFFERENCE between two arms, so it is not a property of
    either arm's per-culture fitness (``EnvironmentResponsePhenotype``), whose t and p
    fields test one culture's response; filing it there would misfile the statistic on
    one of its two arms. The released per-replicate fitness of each arm stays on the
    sibling leaf; this leaf holds only what the release states about the contrast.

    The record's ``environment`` is the condition arm and its reference's
    ``environment_reference`` the reference arm. ``contrast_label`` is the source's own
    name for the comparison (the sheet name), verbatim. Every statistic is stored as
    released (a pFDR q-value can exceed 1 before the monotonicity adjustment, and 582
    of Borchert's do), and each is value-or-gap: the reference of a contrast tests
    nothing, so its fields are typed gaps rather than invented null-test values.
    """

    graph_level: str = "global"
    label_name: str = "t_statistic"
    label_statistic_name: str | None = None

    measurement_type: str = Field(
        description="what was compared and how, e.g. 'rbtnseq_gene_fitness_two_sample_t_test'"
    )
    test_name: str = Field(
        description="the test the source names, verbatim (e.g. 'two-sample t test')"
    )
    contrast_label: str = Field(
        description="the source's own label for the comparison, verbatim (a sheet name)"
    )
    t_statistic: float | None = Field(
        description="the released test statistic, verbatim, signed as released. None "
        "ONLY with a ProvenanceGap"
    )
    p_value: float | None = Field(
        description="the released unadjusted p-value, a probability in [0, 1]. None "
        "ONLY with a ProvenanceGap"
    )
    q_value: float | None = Field(
        description="the released multiple-testing q-value, verbatim and not bounded "
        "above (a pFDR q-value can exceed 1 before a monotonicity adjustment). None "
        "ONLY with a ProvenanceGap"
    )
    q_value_method: str | None = Field(
        default=None,
        description="the correction behind q_value, e.g. 'positive_false_discovery_rate'; "
        "required whenever q_value is stored",
    )
    q_value_adjusted: float | None = Field(
        default=None,
        description="a second, adjusted q-value the source also releases, verbatim; None "
        "when the source releases no second q-value",
    )
    q_value_adjustment_method: str | None = Field(
        default=None,
        description="the adjustment behind q_value_adjusted, e.g. 'monotonicity'; "
        "required whenever q_value_adjusted is stored and forbidden otherwise",
    )
    n_replicates_condition: int | None = Field(
        default=None,
        description="biological replicates of the condition arm the test is over",
    )
    n_replicates_reference: int | None = Field(
        default=None,
        description="biological replicates of the reference arm the test is over",
    )

    @model_validator(mode="after")
    def _check_contrast(self) -> "FitnessContrastPhenotype":
        """Value-or-gap statistics, finite values, a probability p, named corrections."""
        _require_value_or_gap(self, ("t_statistic", "p_value", "q_value"))
        for name in ("t_statistic", "p_value", "q_value", "q_value_adjusted"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if self.p_value is not None and not 0.0 <= self.p_value <= 1.0:
            raise ValueError(f"p_value is {self.p_value}, not a probability in [0, 1]")
        for name in ("q_value", "q_value_adjusted"):
            value = getattr(self, name)
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if (self.q_value is not None) != (self.q_value_method is not None):
            raise ValueError(
                "q_value and q_value_method are set together or not at all"
            )
        if (self.q_value_adjusted is not None) != (
            self.q_value_adjustment_method is not None
        ):
            raise ValueError(
                "q_value_adjusted and q_value_adjustment_method are set together or not "
                "at all"
            )
        for name in ("n_replicates_condition", "n_replicates_reference"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value}")
        for name in ("measurement_type", "test_name", "contrast_label"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be blank")
        return self


class FitnessContrastExperimentReference(ExperimentReference, ModelStrict):
    """The reference arm of a fitness contrast."""

    experiment_reference_type: str = "fitness_contrast"
    phenotype_reference: FitnessContrastPhenotype


class FitnessContrastExperiment(Experiment, ModelStrict):
    """One gene's fitness tested between a condition arm and a reference arm."""

    experiment_type: str = "fitness_contrast"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: FitnessContrastPhenotype


# --------------------------------------------------------------------------- #
# FITNESS CONTRAST BLOCK END (issue #913)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# LIFE CYCLE BLOCK START (De Chiara 2022)
# Progression through the sexual cycle and survival in stationary phase, read as the
# percentage of a culture's cells in a state at a time after the culture was induced
# or entered stationary phase. A time course of one culture is one record.
# --------------------------------------------------------------------------- #
class LifeCycleMeasurementType(StrEnum):
    """What one life-cycle number counts, as a percentage.

    - ``sporulation_efficiency``: asci over all cells counted. De Chiara 2022:
      "Sporulation efficiency was estimated as the number of asci divided by the total
      cell count." A part over its whole, so it lies in [0, 100].
    - ``dyad_fraction``: two-spore asci (dyads) over asci, so a percentage of the
      SPORULATED cells, not of the culture. Undefined when a culture formed no ascus,
      which is why a record carries no dyad key at a time point with no asci.
    - ``stationary_phase_viability``: viable cells over cells at a time after the
      culture entered stationary phase (chronological life span). A percentage
      normalized to controls on the same plate can exceed 100 (De Chiara 2022: "data
      were normalized to four internal controls distributed in each plate"; 37 of its
      released values lie above 100), so it is bounded below only.
    """

    sporulation_efficiency = "sporulation_efficiency"
    dyad_fraction = "dyad_fraction"
    stationary_phase_viability = "stationary_phase_viability"


#: Measurement types that are a part over its whole, so a value above 100 is refused.
LIFE_CYCLE_BOUNDED_PERCENT: frozenset[LifeCycleMeasurementType] = frozenset(
    {
        LifeCycleMeasurementType.sporulation_efficiency,
        LifeCycleMeasurementType.dyad_fraction,
    }
)


class LifeCycleAssay(StrEnum):
    """HOW a life-cycle percentage was read off a culture.

    - ``microscopy_ascus_count``: asci and cells counted under an optical microscope.
    - ``flow_cytometry_scatter``: spores told from vegetative cells by forward and side
      scatter (cell size and granularity) on a flow cytometer, no stain.
    - ``flow_cytometry_viability_stain``: dead cells told from living ones by uptake of
      membrane-impermeable nucleic-acid dyes (propidium iodide, YO-PRO-1).
    """

    microscopy_ascus_count = "microscopy_ascus_count"
    flow_cytometry_scatter = "flow_cytometry_scatter"
    flow_cytometry_viability_stain = "flow_cytometry_viability_stain"


#: The measurement types each assay can produce; a record mixing others is refused.
LIFE_CYCLE_ASSAY_MEASUREMENTS: dict[
    LifeCycleAssay, frozenset[LifeCycleMeasurementType]
] = {
    LifeCycleAssay.microscopy_ascus_count: frozenset(
        {
            LifeCycleMeasurementType.sporulation_efficiency,
            LifeCycleMeasurementType.dyad_fraction,
        }
    ),
    LifeCycleAssay.flow_cytometry_scatter: frozenset(
        {LifeCycleMeasurementType.sporulation_efficiency}
    ),
    LifeCycleAssay.flow_cytometry_viability_stain: frozenset(
        {LifeCycleMeasurementType.stationary_phase_viability}
    ),
}


class LifeCycleTimeUnit(StrEnum):
    """The unit a life-cycle time point is stated in, as the source states it."""

    hour = "h"
    day = "d"


_LIFE_CYCLE_KEY_RE = re.compile(
    r"^(?P<measurement>[a-z_]+)@(?P<time>\d+(?:\.\d+)?)(?P<unit>h|d)$"
)


def life_cycle_key(
    measurement_type: LifeCycleMeasurementType, time: float, unit: LifeCycleTimeUnit
) -> str:
    """The ``life_cycle`` key of one time point: ``<measurement type>@<time><unit>``.

    ``sporulation_efficiency@24h``, ``stationary_phase_viability@35d``. A whole time is
    written without a decimal point, so one time point has one spelling.
    """
    if not math.isfinite(time) or time < 0:
        raise ValueError(f"a life-cycle time point is finite and >= 0, got {time}")
    written = str(int(time)) if float(time).is_integer() else repr(float(time))
    return f"{LifeCycleMeasurementType(measurement_type).value}@{written}{LifeCycleTimeUnit(unit).value}"


def parse_life_cycle_key(
    key: str,
) -> tuple[LifeCycleMeasurementType, float, LifeCycleTimeUnit]:
    """Split a ``life_cycle`` key into (measurement type, time, unit); refuse others.

    The key must also be the canonical spelling ``life_cycle_key`` writes, so
    ``spo@24.0h`` and ``spo@24h`` cannot both name one time point.
    """
    match = _LIFE_CYCLE_KEY_RE.match(key)
    if match is None:
        raise ValueError(
            f"life-cycle key {key!r} is not '<measurement type>@<time><h|d>'"
        )
    measurement = LifeCycleMeasurementType(match.group("measurement"))
    time = float(match.group("time"))
    unit = LifeCycleTimeUnit(match.group("unit"))
    if life_cycle_key(measurement, time, unit) != key:
        raise ValueError(
            f"life-cycle key {key!r} is not canonical; write "
            f"{life_cycle_key(measurement, time, unit)!r}"
        )
    return measurement, time, unit


class LifeCyclePhenotype(Phenotype, ModelStrict):
    """A culture's progression through the life cycle, as percentages over time.

    One record is one culture followed over time by one assay: an isolate in
    sporulation medium counted at 24 h and 72 h, or a stationary-phase culture stained
    at days 7, 21 and 35. ``life_cycle`` maps ``life_cycle_key(...)`` to the released
    percentage, so the time course stays in the record that measured it and each value
    says what it counts and when.

    Why a new leaf rather than an existing one: ``FitnessPhenotype`` is a positive
    growth ratio that clamps non-positive values, ``EnvironmentResponsePhenotype`` is
    one scalar response per record and has no time axis, and a sporulation efficiency
    is neither a growth rate nor a response to an added compound. The class is
    additive: no served record changes.

    Fields beside the values:

    - ``assay``: how the percentages were read; every key's measurement type must be
      one the assay produces (``LIFE_CYCLE_ASSAY_MEASUREMENTS``).
    - ``n_samples`` / ``sample_unit``: independent cultures behind each value, both set
      or both ``None`` (a ``None`` the source does not state is a typed gap).
    - ``cells_per_sample``: cells counted per culture and time point;
      ``cells_per_sample_is_minimum`` marks a source that states a floor (">100 cells")
      rather than a count.
    """

    graph_level: str = "global"
    label_name: str = "life_cycle"
    label_statistic_name: str | None = None

    life_cycle: dict[str, float] = Field(
        description=(
            "'<measurement type>@<time><h|d>' -> percentage of cells, in sorted key "
            "order; a time point the source did not release (or a ratio it left "
            "undefined) is not a key"
        )
    )
    assay: LifeCycleAssay = Field(description="how the percentages were read")
    n_samples: int | None = Field(
        default=None,
        description="independent cultures behind each value; None when not stated",
    )
    sample_unit: SampleUnit | None = Field(
        default=None, description="what one sample in n_samples is"
    )
    cells_per_sample: int | None = Field(
        default=None,
        description="cells counted per culture and time point; None when not stated",
    )
    cells_per_sample_is_minimum: bool = Field(
        default=False,
        description="True when the source states a floor ('>100 cells') rather than a "
        "count; requires cells_per_sample",
    )

    @field_validator("life_cycle", mode="before")
    @classmethod
    def validate_life_cycle(cls, v: Any) -> Any:  # raw pre-validation input
        """Sort the keys; every key canonical, every value a valid percentage."""
        if not isinstance(v, dict):
            raise ValueError(f"life_cycle must be a dict, got {type(v).__name__}")
        if not v:
            raise ValueError("life_cycle cannot be empty")
        for key, value in v.items():
            measurement, _, _ = parse_life_cycle_key(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"life_cycle[{key!r}] is not a number")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"life_cycle[{key!r}] must be a finite percentage >= 0, got {value}"
                )
            if measurement in LIFE_CYCLE_BOUNDED_PERCENT and value > 100.0:
                raise ValueError(
                    f"life_cycle[{key!r}] = {value}: a {measurement.value} is a part of "
                    "its whole and cannot exceed 100"
                )
        return v if isinstance(v, SortedDict) else SortedDict(v)

    @model_validator(mode="after")
    def _check_life_cycle(self) -> "LifeCyclePhenotype":
        """The assay produces every key's measurement; counts are positive and paired."""
        allowed = LIFE_CYCLE_ASSAY_MEASUREMENTS[self.assay]
        for key in self.life_cycle:
            measurement, _, _ = parse_life_cycle_key(key)
            if measurement not in allowed:
                raise ValueError(
                    f"{self.assay.value} does not measure {measurement.value} "
                    f"(key {key!r})"
                )
        if (self.n_samples is None) != (self.sample_unit is None):
            raise ValueError("n_samples and sample_unit are set together or not at all")
        if self.n_samples is not None and self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {self.n_samples}")
        if self.cells_per_sample is not None and self.cells_per_sample < 1:
            raise ValueError(
                f"cells_per_sample must be >= 1, got {self.cells_per_sample}"
            )
        if self.cells_per_sample_is_minimum and self.cells_per_sample is None:
            raise ValueError("cells_per_sample_is_minimum requires cells_per_sample")
        return self

    def __repr__(self) -> str:
        """Summary repr naming the assay and the time points."""
        return (
            f"LifeCyclePhenotype(assay={self.assay.value}, "
            f"keys={list(self.life_cycle)})"
        )


class LifeCycleExperimentReference(ExperimentReference, ModelStrict):
    """Reference for a life-cycle experiment; its environment states the protocol."""

    experiment_reference_type: str = "life_cycle"
    environment_reference: CultureEnvironment
    phenotype_reference: LifeCyclePhenotype


class LifeCycleExperiment(Experiment, ModelStrict):
    """One culture's life-cycle time course under one assay and one protocol.

    ``environment`` is a ``CultureEnvironment`` because a sporulation efficiency is
    meaningless without the protocol that induced it (pre-sporulation medium, vessel,
    shaking, inoculum density); declared, not inherited, so pydantic serializes it.
    """

    experiment_type: str = "life_cycle"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    environment: CultureEnvironment
    phenotype: LifeCyclePhenotype


# --------------------------------------------------------------------------- #
# LIFE CYCLE BLOCK END (De Chiara 2022)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# GENOME ARCHITECTURE BLOCK START (Puddu 2019, yeast expansion row 7)
# The REALIZED genome content of one sequenced sample: repeat-array and replicon copy
# numbers, per-chromosome copy number, a telomere-length proxy and a rearrangement
# count, each read off whole-genome sequencing coverage.
# --------------------------------------------------------------------------- #
#: How far a record's ``ANtot`` aggregate may differ from the sum of its own
#: per-chromosome deviations before the record is refused. The aggregate is the source's
#: own column and the per-chromosome values are the source's own columns, so the identity
#: is a property of the RELEASE, not of this schema: measured over all 8,843 released
#: Puddu 2019 colonies the largest disagreement is 3.1e-15, i.e. float noise.
PLOIDY_DEVIATION_ATOL = 1e-6


class GenomeArchitecturePhenotype(Phenotype, ModelStrict):
    """One sequenced sample's genome-architecture profile, keyed by its assay's symbols.

    The quantity is the genome content the sequencing FOUND, as distinct from the edit
    the strain was designed to carry: a kanMX deletion strain of the yeast knockout
    collection is designed as one absent ORF and is realized as that ORF plus whatever
    repeat expansion, plasmid or mitochondrial copy-number change, aneuploidy and
    chromosomal rearrangement the collection accumulated. Puddu et al. 2019 (Nature
    573:416) measured exactly that for 8,843 colonies of the homozygous diploid
    collection, so the designed perturbation lives in ``Genotype`` and this phenotype
    carries the realized architecture.

    Why its own class rather than a field on an existing one. ``CalMorphPhenotype`` and
    ``BacterialMorphologyPhenotype`` hold the right SHAPE (named per-sample values plus a
    dispersion map) and the wrong quantities: both are microscopy morphology vocabularies
    and neither can express a copy number per haploid genome. Nothing already in the
    schema carries a per-chromosome copy number at all, which is why the Hoepfner 2014
    background records a typed gap reading "The schema has no per-chromosome copy number,
    so the aneuploidy and the amplification cannot be recorded per strain". This class is
    additive: no served record changes.

    Key semantics, which is the whole point of the class:

    - ``assay`` names an :class:`~torchcell.datamodels.genome_architecture_features.ArchitectureAssay`
      in ``GENOME_ARCHITECTURE_ASSAYS``, and that assay's symbols are the only permitted
      keys. Every symbol DECLARES its unit basis there, because the numbers in one record
      are not in one unit: a repeat copy number is per haploid genome, mtDNA and a plasmid
      are per cell, a chromosome copy number is per cell with 2.0 as the diploid baseline,
      and a telomere proxy is reads per million. A consumer that averages across keys is
      averaging quantities the source never claimed were comparable.
    - A record carries the features its sample was MEASURED for, not necessarily the whole
      vocabulary: ``required`` coverage is the loader's statement, not this class's.
    - ``genome_architecture_sd`` is the dispersion of the same symbols when the stored
      value is a mean over samples (the wild-type reference profile), with
      ``uncertainty_type`` naming WHAT that number is and ``n_samples`` /``sample_unit``
      the design behind it. It is ``None`` for a single sequenced sample, which has no
      dispersion.
    - ``sample_id`` and ``sample_accession`` identify the sequenced sample the numbers
      came from (the colony id and its archive accession). They are part of the
      measurement: two colonies of one strain are two samples, never two replicates to be
      averaged, and the accession is what makes the stored number traceable to the reads.
    """

    graph_level: str = "global"
    label_name: str = "genome_architecture"
    label_statistic_name: str | None = "genome_architecture_sd"
    assay: str = Field(
        description="the registered genome-architecture assay whose vocabulary names "
        "the keys (a key of "
        "torchcell.datamodels.genome_architecture_features.GENOME_ARCHITECTURE_ASSAYS)"
    )
    genome_architecture: dict[str, float] = Field(
        description="per-sample values keyed by the assay's own symbol; each symbol's "
        "unit basis is declared by the assay, so keys are never pooled",
        repr=False,
    )
    genome_architecture_sd: dict[str, float] | None = Field(
        default=None,
        description="dispersion of the same symbols when the value is a mean over "
        "samples; None for a single sequenced sample",
        repr=False,
    )
    uncertainty_type: UncertaintyType | None = Field(
        default=None,
        description="what the numbers in genome_architecture_sd ARE; required with them",
    )
    n_samples: int | None = Field(
        default=None, description="samples the stored values are taken over"
    )
    sample_unit: SampleUnit | None = Field(
        default=None, description="what one sample in n_samples physically is"
    )
    sample_id: str | None = Field(
        default=None,
        description="the source's own identifier of the sequenced sample, verbatim "
        "(Puddu's colony id 'SD0863b'); None for a profile that is not one sample",
    )
    sample_accession: str | None = Field(
        default=None,
        description="the sequence-archive accession of that sample, verbatim "
        "('ERS838232'); None when the source releases none",
    )

    def __repr__(self) -> str:
        """Summary repr instead of dumping the per-feature dict."""
        return (
            f"GenomeArchitecturePhenotype(assay={self.assay!r}, "
            f"features={len(self.genome_architecture)}, sample_id={self.sample_id!r})"
        )

    @property
    def assay_vocabulary(self) -> ArchitectureAssay:
        """The assay this record names, which is the authority on its keys."""
        return architecture_assay(self.assay)

    @model_validator(mode="after")
    def validate_against_the_assay(self) -> "GenomeArchitecturePhenotype":
        """Keys belong to the named assay, values fit their basis, aggregates agree.

        The assay lookup raises on an unregistered name, so a record can never be
        written against a vocabulary this version of the schema does not hold.
        """
        assay = architecture_assay(self.assay)
        if not self.genome_architecture:
            raise ValueError(
                f"genome_architecture cannot be empty (assay {self.assay!r})"
            )
        self._check_keys("genome_architecture", self.genome_architecture, assay)
        for symbol, value in self.genome_architecture.items():
            assay.validate_value(symbol, float(value))
        if self.genome_architecture_sd is not None:
            self._check_keys(
                "genome_architecture_sd", self.genome_architecture_sd, assay
            )
            for symbol, value in self.genome_architecture_sd.items():
                if not math.isfinite(float(value)) or float(value) < 0.0:
                    raise ValueError(
                        f"genome_architecture_sd[{symbol}] must be a finite "
                        f"non-negative dispersion, got {value}"
                    )
            if self.uncertainty_type is None:
                raise ValueError(
                    "genome_architecture_sd needs uncertainty_type to say what the "
                    "number is (a sample SD and a standard error convert differently)"
                )
        elif self.uncertainty_type is not None:
            raise ValueError(
                "uncertainty_type names the kind of a dispersion this record does not "
                "carry; leave it None"
            )
        self._check_ploidy_deviation(assay)
        return self

    @staticmethod
    def _check_keys(
        field: str, values: dict[str, float], assay: ArchitectureAssay
    ) -> None:
        """Every key is a feature of the named assay."""
        permitted = frozenset(assay.by_symbol)
        outside = sorted(set(values) - permitted)
        if outside:
            raise ValueError(
                f"{field}: {outside} are not features of assay {assay.name}; it "
                f"declares {len(permitted)} symbols"
            )

    def _check_ploidy_deviation(self, assay: ArchitectureAssay) -> None:
        """A stored ploidy-deviation aggregate equals the deviations it aggregates.

        Only checked when the record carries BOTH the aggregate and every chromosome it
        sums over: a reference profile that states the aggregate alone is stating what
        the source published, not a derived sum.
        """
        aggregates = assay.symbols_with_basis(ArchitectureBasis.ploidy_deviation_units)
        chromosomes = assay.symbols_with_basis(
            ArchitectureBasis.chromosome_copies_per_cell
        )
        if not aggregates or not chromosomes <= set(self.genome_architecture):
            return
        expected = sum(
            abs(float(self.genome_architecture[symbol]) - 2.0) for symbol in chromosomes
        )
        for symbol in aggregates & set(self.genome_architecture):
            stored = float(self.genome_architecture[symbol])
            if abs(stored - expected) > PLOIDY_DEVIATION_ATOL:
                raise ValueError(
                    f"{symbol} is {stored} but the record's {len(chromosomes)} "
                    f"chromosome copy numbers deviate from 2.0 by {expected} in total"
                )


class GenomeArchitectureExperimentReference(ExperimentReference, ModelStrict):
    """Reference profile for a genome-architecture experiment.

    ``genome_reference`` is a ``StrainReferenceGenome`` so the diploid baseline every
    chromosome copy number is read against, and the background alleles the collection
    carries, are stated rather than implied.
    """

    experiment_reference_type: str = "genome_architecture"
    genome_reference: StrainReferenceGenome
    phenotype_reference: GenomeArchitecturePhenotype


class GenomeArchitectureExperiment(Experiment, ModelStrict):
    """One sequenced sample's realized genome architecture under a designed genotype."""

    experiment_type: str = "genome_architecture"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: GenomeArchitecturePhenotype


# --------------------------------------------------------------------------- #
# GENOME ARCHITECTURE BLOCK END (Puddu 2019, yeast expansion row 7)
# --------------------------------------------------------------------------- #


PhenotypeType = (
    Phenotype
    | FitnessPhenotype
    | GeneInteractionPhenotype
    | GeneEssentialityPhenotype
    | GradedGeneEssentialityPhenotype  # #909
    | SyntheticLethalityPhenotype
    | SyntheticRescuePhenotype
    | CalMorphPhenotype
    | MicroarrayExpressionPhenotype
    | MicroarrayExpressionScorePhenotype  # issue #1022
    | MicroarrayExpressionRatioPhenotype  # issue #989
    | InductionTimecourseExpressionPhenotype  # Hackett 2020 (additive)
    | RNASeqExpressionPhenotype
    | PseudobulkExpressionPhenotype
    | VisualScorePhenotype
    | MetabolitePhenotype
    | ProteinAbundancePhenotype
    # --- begin #770: the protein fold-change family ---
    | ProteinFoldChangePhenotype
    # --- end #770 ---
    | EnvironmentResponsePhenotype
    | ProductTiterPhenotype
    | ProteinTurnoverPhenotype
    | ProteinSynthesisRatePhenotype  # issue #857
    | PhosphositeIntensityPhenotype  # issue #921
    | FluxPhenotype
    | PromoterActivityPhenotype
    | PooledSortSeqExpressionPhenotype  # Vaishnav 2022 (#991)
    | BiosensorFluorescencePhenotype  # Zhang 2020 (#990)
    | BacterialMorphologyPhenotype  # issue #774
    | MrnaNumberFractionPhenotype  # issue #854
    | FitnessContrastPhenotype  # issue #913
    | SingleCellExpressionPhenotype  # boocock2025
    | LifeCyclePhenotype  # De Chiara 2022
    # ADDITIVE (Puddu 2019 genome architecture)
    | GenomeArchitecturePhenotype
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
    | MicroarrayExpressionScoreExperiment  # issue #1022
    | MicroarrayExpressionRatioExperiment  # issue #989
    | InductionTimecourseExpressionExperiment  # Hackett 2020 (additive)
    | RNASeqExpressionExperiment
    | PseudobulkExpressionExperiment
    | VisualScoreExperiment
    | MetaboliteExperiment
    | ProteinAbundanceExperiment
    # --- begin #770: the protein fold-change family ---
    | ProteinFoldChangeExperiment
    # --- end #770 ---
    | EnvironmentResponseExperiment
    | StrainEnvironmentResponseExperiment
    | SegregantGrowthExperiment
    | SegregantPerturbationExperiment  # Hale 2024
    | ProductTiterExperiment
    | ProteinTurnoverExperiment
    | ProteinSynthesisRateExperiment  # issue #857
    | FluxExperiment
    | PromoterActivityExperiment
    | PooledSortSeqExpressionExperiment  # Vaishnav 2022 (#991)
    | BiosensorFluorescenceExperiment  # Zhang 2020 (#990)
    | GradedGeneEssentialityExperiment  # #909
    | PhosphositeIntensityExperiment  # issue #921
    | MorphologyExperiment  # issue #774
    | MrnaNumberFractionExperiment  # issue #854
    | FitnessContrastExperiment  # issue #913
    | SegregantExpressionExperiment  # boocock2025
    | LifeCycleExperiment  # De Chiara 2022
    | PseudobulkUmiCountExperiment  # jackson2020
    # ADDITIVE (Puddu 2019 genome architecture)
    | GenomeArchitectureExperiment
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
    | MicroarrayExpressionScoreExperimentReference  # issue #1022
    | MicroarrayExpressionRatioExperimentReference  # issue #989
    | InductionTimecourseExpressionExperimentReference  # Hackett 2020 (additive)
    | RNASeqExpressionExperimentReference
    | PseudobulkExpressionExperimentReference
    | VisualScoreExperimentReference
    | MetaboliteExperimentReference
    | ProteinAbundanceExperimentReference
    # --- begin #770: the protein fold-change family ---
    | ProteinFoldChangeExperimentReference
    # --- end #770 ---
    | EnvironmentResponseExperimentReference
    | StrainEnvironmentResponseExperimentReference
    | SegregantGrowthExperimentReference
    | SegregantPerturbationExperimentReference  # Hale 2024
    | ProductTiterExperimentReference
    | ProteinTurnoverExperimentReference
    | ProteinSynthesisRateExperimentReference  # issue #857
    | FluxExperimentReference
    | PromoterActivityExperimentReference
    | PooledSortSeqExpressionExperimentReference  # Vaishnav 2022 (#991)
    | BiosensorFluorescenceExperimentReference  # Zhang 2020 (#990)
    | GradedGeneEssentialityExperimentReference  # #909
    | PhosphositeIntensityExperimentReference  # issue #921
    | MorphologyExperimentReference  # issue #774
    | MrnaNumberFractionExperimentReference  # issue #854
    | FitnessContrastExperimentReference  # issue #913
    | SegregantExpressionExperimentReference  # boocock2025
    | LifeCycleExperimentReference  # De Chiara 2022
    | PseudobulkUmiCountExperimentReference  # jackson2020
    # ADDITIVE (Puddu 2019 genome architecture)
    | GenomeArchitectureExperimentReference
)


EXPERIMENT_TYPE_MAP = {
    "fitness": FitnessExperiment,
    "gene interaction": GeneInteractionExperiment,
    "gene essentiality": GeneEssentialityExperiment,
    "synthetic lethality": SyntheticLethalityExperiment,
    "synthetic rescue": SyntheticRescueExperiment,
    "calmorph": CalMorphExperiment,
    "microarray_expression": MicroarrayExpressionExperiment,
    "microarray_expression_score": MicroarrayExpressionScoreExperiment,  # issue #1022
    "microarray_expression_ratio": MicroarrayExpressionRatioExperiment,  # issue #989
    "induction_timecourse_expression": InductionTimecourseExpressionExperiment,  # Hackett 2020
    "rnaseq_expression": RNASeqExpressionExperiment,
    "pseudobulk_expression": PseudobulkExpressionExperiment,
    "visual_score": VisualScoreExperiment,
    "metabolite": MetaboliteExperiment,
    "protein_abundance": ProteinAbundanceExperiment,
    # --- begin #770: the protein fold-change family ---
    "protein_fold_change": ProteinFoldChangeExperiment,
    # --- end #770 ---
    "environment_response": EnvironmentResponseExperiment,
    "strain_environment_response": StrainEnvironmentResponseExperiment,
    "segregant_growth": SegregantGrowthExperiment,
    "segregant_perturbation": SegregantPerturbationExperiment,  # Hale 2024
    "product_titer": ProductTiterExperiment,
    "protein_turnover": ProteinTurnoverExperiment,
    "protein_synthesis_rate": ProteinSynthesisRateExperiment,  # issue #857
    "flux": FluxExperiment,
    "promoter_activity": PromoterActivityExperiment,
    "pooled_sort_seq_expression": PooledSortSeqExpressionExperiment,  # #991
    "biosensor_fluorescence": BiosensorFluorescenceExperiment,  # #990
    "graded_gene_essentiality": GradedGeneEssentialityExperiment,
    # --- begin #770: the protein fold-change family ---
    # --- end #770 ---
    "phosphosite_intensity": (  # issue #921
        PhosphositeIntensityExperiment
    ),
    "morphology": MorphologyExperiment,  # issue #774
    "mrna_number_fraction": MrnaNumberFractionExperiment,  # issue #854
    "fitness_contrast": FitnessContrastExperiment,  # issue #913
    "segregant_expression": SegregantExpressionExperiment,  # boocock2025
    "life_cycle": LifeCycleExperiment,  # De Chiara 2022
    "pseudobulk_umi_count": PseudobulkUmiCountExperiment,  # jackson2020
    # ADDITIVE (Puddu 2019 genome architecture)
    "genome_architecture": GenomeArchitectureExperiment,
}

EXPERIMENT_REFERENCE_TYPE_MAP = {
    "fitness": FitnessExperimentReference,
    "gene interaction": GeneInteractionExperimentReference,
    "gene essentiality": GeneEssentialityExperimentReference,
    "synthetic lethality": SyntheticLethalityExperimentReference,
    "synthetic rescue": SyntheticRescueExperimentReference,
    "calmorph": CalMorphExperimentReference,
    "microarray_expression": MicroarrayExpressionExperimentReference,
    "microarray_expression_score": (  # issue #1022
        MicroarrayExpressionScoreExperimentReference
    ),
    "microarray_expression_ratio": (  # issue #989
        MicroarrayExpressionRatioExperimentReference
    ),
    "induction_timecourse_expression": (  # Hackett 2020
        InductionTimecourseExpressionExperimentReference
    ),
    "rnaseq_expression": RNASeqExpressionExperimentReference,
    "pseudobulk_expression": PseudobulkExpressionExperimentReference,
    "visual_score": VisualScoreExperimentReference,
    "metabolite": MetaboliteExperimentReference,
    "protein_abundance": ProteinAbundanceExperimentReference,
    # --- begin #770: the protein fold-change family ---
    "protein_fold_change": ProteinFoldChangeExperimentReference,
    # --- end #770 ---
    "environment_response": EnvironmentResponseExperimentReference,
    "strain_environment_response": StrainEnvironmentResponseExperimentReference,
    "segregant_growth": SegregantGrowthExperimentReference,
    "segregant_perturbation": SegregantPerturbationExperimentReference,  # Hale 2024
    "product_titer": ProductTiterExperimentReference,
    "protein_turnover": ProteinTurnoverExperimentReference,
    "protein_synthesis_rate": ProteinSynthesisRateExperimentReference,  # issue #857
    "flux": FluxExperimentReference,
    "promoter_activity": PromoterActivityExperimentReference,
    "pooled_sort_seq_expression": (  # #991
        PooledSortSeqExpressionExperimentReference
    ),
    "biosensor_fluorescence": BiosensorFluorescenceExperimentReference,  # #990
    "graded_gene_essentiality": (GradedGeneEssentialityExperimentReference),
    # --- begin #770: the protein fold-change family ---
    # --- end #770 ---
    "phosphosite_intensity": (  # issue #921
        PhosphositeIntensityExperimentReference
    ),
    "morphology": MorphologyExperimentReference,  # issue #774
    "mrna_number_fraction": MrnaNumberFractionExperimentReference,  # issue #854
    "fitness_contrast": (  # issue #913
        FitnessContrastExperimentReference
    ),
    "segregant_expression": SegregantExpressionExperimentReference,  # boocock2025
    "life_cycle": LifeCycleExperimentReference,  # De Chiara 2022
    "pseudobulk_umi_count": PseudobulkUmiCountExperimentReference,  # jackson2020
    # ADDITIVE (Puddu 2019 genome architecture)
    "genome_architecture": GenomeArchitectureExperimentReference,
}


#: The 14 experiment-type tags retired by the ontology unification (P2), each with the
#: tag its records now carry. A dev LMDB written under one of these keys is UNREADABLE by
#: ``torchcell.data.aggregate`` and ``torchcell.data.deduplicate`` until it is rebuilt:
#: the maps above no longer hold the key, and the readers name the retired tag and this
#: table instead of guessing a successor.
RETIRED_EXPERIMENT_TYPES: dict[str, str] = {
    "bacterial_fitness": "fitness",
    "bacterial_environment_response": "environment_response",
    "bacterial_gene_interaction": "gene interaction",
    "bacterial_gene_essentiality": "gene essentiality",
    "bacterial_graded_gene_essentiality": "graded_gene_essentiality",
    "bacterial_protein_abundance": "protein_abundance",
    "bacterial_protein_fold_change": "protein_fold_change",
    "bacterial_phosphosite_intensity": "phosphosite_intensity",
    "bacterial_metabolite": "metabolite",
    "bacterial_rnaseq_expression": "rnaseq_expression",
    "bacterial_visual_score": "visual_score",
    "bacterial_morphology": "morphology",
    "bacterial_fitness_contrast": "fitness_contrast",
    "yeast_protein_turnover": "protein_turnover",
}


if __name__ == "__main__":
    pass
