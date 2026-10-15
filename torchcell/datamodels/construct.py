# torchcell/datamodels/construct.py
# [[torchcell.datamodels.construct]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/construct.py
# Test file: tests/torchcell/datamodels/test_construct.py

"""Where added DNA sits: on which construct, at which host site, carrying which variant.

The material layer of a gain-of-function perturbation, as a composed pointer instead of
four loose nullable strings. Nothing here is referenced by a served class yet: these are
NEW symbols for the composition step, so no served closure's fingerprint moves and no
store rebuilds for this module alone. They are built to be composed, which is what
fixes the defect they exist for: a ``plasmid_contig_id: str | None`` and a
``locus_tag: str | None`` can be half-filled and were ``None`` in 26 of 26 served
gene-addition stores, while a ``placement`` is either a whole pointer or a typed gap.

Three facts, three objects:

- ``ConstructPlacement`` says WHICH DNA was added: a contig by its sha256 identity, a
  feature of that contig by its key, and the ``ArtifactRef`` that pins the bytes, so a
  stored record verifies itself without the construct index being present.
- ``HostPlacement`` says WHERE in the host it went: the assembly set the coordinates are
  written against, the site's label verbatim, and the three spans an integration has
  (the host interval it landed in, the host interval it replaced, and the interval of
  the CONTIG that was transferred). Absent for an episomal construct, which is a
  replicon of its own and sits at no host site.
- ``ConstructVariantCall`` says WHAT differs from the construct's own sequence: a
  coordinate on the contig, in the shape the resequencing releases already use. Cachera
  2023's ``ARO4-K229L`` and ``ARO7-G141S`` are free strings today
  (``variant="K229L"``), and a free string is not a sequence fact.

Hashability is load-bearing, not incidental. ``Genotype.__eq__`` compares
``set(self.perturbations)``, and a frozen pydantic model hashes the tuple of its field
values, so every object that can hang off a perturbation must hash. Each class here
therefore derives from ``HashableProvenanceGapMixin`` (canonical-JSON hash), which is
also what lets them carry ``provenance_gaps`` at all.

One design choice worth stating, because the obvious alternative is a defect. The
CONTIG-side insert span (the segment between the homology arms that is transferred) is
a single field on ``HostPlacement``, and ``ConstructPlacement.insert_span`` is a
read-only property that reads it. Two fields holding one fact in two classes is what
principle 5 forbids, and the fact belongs with the integration: an episomal construct
has no insert span, because the whole replicon is present.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from torchcell.datamodels.schema import (
    ArtifactRef,
    BacterialVariantType,
    GenomicSpan,
    HashableProvenanceGapMixin,
    VariantCallMode,
    VariantFrequencyBasis,
    _require_value_or_gap,
)
from torchcell.sequence.construct.registry import (
    PLACEMENT_TARGET_REFUSED_SO,
    ConstructSpan,
    FeatureKey,
)

_SHA256_HEX = set("0123456789abcdef")


class Localization(StrEnum):
    """Where added DNA resides in the cell: in a chromosome, or on its own replicon.

    Two members, because there are two physical situations and the replicon's identity
    (2 micron, CEN/ARS, a bacterial broad-host-range origin) is a property of the
    CONTIG, read from its origin-of-replication feature, not of the gene that rides on
    it. Folding ``episomal_2micron`` into ``episomal_replicon`` is lossless exactly when
    a contig exists to carry the distinction, so a record with no deposited construct
    keeps the source's own words in a verbatim statement on its leaf. Unused by any
    served class today.
    """

    chromosomal_integration = "chromosomal_integration"
    episomal_replicon = "episomal_replicon"


class ConstructVariantCall(HashableProvenanceGapMixin):
    """One variant on a construct contig, as a coordinate rather than a name.

    The field shape is ``BacterialVariantCall``'s, deliberately: the same physical
    statement (this base at this position became that base) gets the same fields and the
    same closed vocabularies, which is why the ``BacterialVariantType``,
    ``VariantCallMode`` and ``VariantFrequencyBasis`` enums are reused here rather than
    copied. One difference, and it is the point of the class:
    ``reference_contig_sha256`` is a sha256, not a free replicon string, because a
    construct's identity IS its hash and the coordinate is meaningless against anything
    else. ``variant_statement`` keeps what the source wrote (``K229L``, ``Y20F``,
    ``a single A inserted at -10``), so the verbatim claim survives beside the typed
    coordinate and a disagreement between them stays visible.

    ``position_start`` / ``position_end`` are 1-based end-inclusive on the contig, as
    ``ConstructSpan`` is; for an insertion ``position_end`` may be ``position_start - 1``,
    the zero-length reference interval an insertion sits in.
    """

    variant_type: BacterialVariantType
    variant_statement: str = Field(
        description="the source's own statement of the variant, verbatim (e.g. 'K229L')"
    )
    reference_contig_sha256: str = Field(
        description="sha256 of the construct contig the coordinates are on"
    )
    position_start: int = Field(description="1-based start of the change on the contig")
    position_end: int = Field(
        description="1-based, end-INCLUSIVE end; equals position_start for a point "
        "change and may be position_start - 1 for an insertion"
    )
    sequence_change: str | None = Field(
        default=None, description="the change as the source writes it (e.g. 'G->A')"
    )
    reference_allele: str | None = Field(
        default=None, description="the reference base(s), when stated separately"
    )
    alternate_allele: str | None = Field(
        default=None, description="the alternate base(s), on the same condition"
    )
    amino_acid_change: str | None = Field(
        default=None,
        description="the protein-level change, verbatim, when the source states it",
    )
    codon_number: int | None = Field(
        default=None, description="the 1-based codon of the CDS the change sits in"
    )
    call_mode: VariantCallMode | None = Field(
        default=None,
        description="set only for a CALLED variant (a resequenced construct); None for "
        "a DESIGNED allele, which has no frequency to be a fraction of",
    )
    frequency: float | None = Field(
        default=None, description="the frequency, when the source gives one value"
    )
    frequency_basis: VariantFrequencyBasis | None = Field(
        default=None,
        description="the scale `frequency` is on; required whenever frequency is set",
    )

    @model_validator(mode="after")
    def _check_call(self) -> ConstructVariantCall:
        """A hex contig identity, ordered 1-based coordinates, a stated frequency scale."""
        if (
            len(self.reference_contig_sha256) != 64
            or set(self.reference_contig_sha256) - _SHA256_HEX
        ):
            raise ValueError(
                "reference_contig_sha256 must be 64 lowercase hex, got "
                f"{self.reference_contig_sha256!r}"
            )
        if not self.variant_statement.strip():
            raise ValueError("ConstructVariantCall needs a variant_statement")
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
        if self.frequency is None and self.frequency_basis is not None:
            raise ValueError(
                "frequency_basis names the scale of a frequency, but frequency is None"
            )
        if self.frequency is not None and self.frequency_basis is None:
            raise ValueError(
                "frequency is set with no frequency_basis; a fraction is never read as "
                "a percent"
            )
        if self.frequency is not None and self.call_mode is None:
            raise ValueError(
                "a frequency without a call_mode cannot be read: a clone's within-"
                "isolate fraction and a population allele frequency are different facts"
            )
        return self


class HostPlacement(HashableProvenanceGapMixin):
    r"""Where an integrated construct landed in the host, as sequence rather than a label.

    ``assembly_set`` is a set id the genomes registry knows, so the coordinates are
    written against deposited bytes rather than a release name. ``site_label`` is the
    source's own name for the landing pad, verbatim (``"XII-5"``, ``"Delta"``,
    ``"PP_5322intergenic"``, ``"EasyClone XII-5"``), and it is required: a named pad is
    what the source actually states, and it is a different fact from a coordinate.

    ``span`` is the host interval the insert landed in and is set OR carries a typed
    ``ProvenanceGap``, never a silent ``None``. The gap is the common case and is
    honest: Cachera's ``XII-5`` and Lian's ``X3`` are named pads with no published
    coordinate, and the genome never gates a record. ``replaced_span`` is the interval
    the insert REPLACED, for a delins integration, and is optional because a clean
    insertion replaces nothing, which is a real ``None`` rather than an absent value.
    ``insert_span`` is the interval OF THE CONTIG that was transferred (Cachera's
    8,104 bp between ``XII-5\\UP`` at 1548-2002 and ``XII-5\\Down`` at 10106-10782) and
    is likewise set or gapped, because the arms are not always annotated.
    """

    assembly_set: str = Field(
        description="an assembly-set id the genomes registry holds, e.g. "
        "'sgd_S288C_R64-4-1_20230830'"
    )
    site_label: str = Field(
        description="the source's own name for the landing site, verbatim"
    )
    span: GenomicSpan | None = Field(
        default=None, description="the host interval the insert landed in, or gapped"
    )
    insert_span: ConstructSpan | None = Field(
        default=None,
        description="the interval OF THE CONTIG that was transferred, or gapped",
    )
    replaced_span: GenomicSpan | None = Field(
        default=None,
        description="the host interval the insert replaced; None for a clean insertion",
    )

    @model_validator(mode="after")
    def _check_placement(self) -> HostPlacement:
        """A named site and an assembly set; the two gap-able spans set or typed absent."""
        if not self.assembly_set.strip():
            raise ValueError("HostPlacement needs an assembly_set")
        if not self.site_label.strip():
            raise ValueError(
                "HostPlacement needs the source's own site_label; a coordinate does "
                "not replace what the source called the site"
            )
        _require_value_or_gap(self, ("span", "insert_span"))
        if self.span is not None and self.span.assembly != self.assembly_set:
            raise ValueError(
                f"span is on assembly {self.span.assembly!r} but the placement pins "
                f"{self.assembly_set!r}"
            )
        if self.replaced_span is not None and (
            self.replaced_span.assembly != self.assembly_set
        ):
            raise ValueError(
                f"replaced_span is on assembly {self.replaced_span.assembly!r} but the "
                f"placement pins {self.assembly_set!r}"
            )
        return self


class ConstructPlacement(HashableProvenanceGapMixin):
    """The composed pointer from an added gene to the construct sequence it came from.

    ``contig_sha256`` is the construct's identity and ``contig_ref`` pins the same bytes,
    so the two must agree; ``feature_key`` names the feature within that contig by
    label, SO id and 1-based span, which is the only key the released maps support
    (0 of 570 features carry a ``/locus_tag``). A primer binding site is refused as a
    target: it is an oligo's footprint on the map rather than a part of the construct,
    and 297 of those 570 features are primers, so without the refusal the modal feature
    of the store would be a legal target for an integrated gene.

    ``host_placement`` is set for a chromosomal integration and ``None`` for an episomal
    construct. ``variants`` are contig-relative calls and must every one name THIS
    contig: a coordinate against another sequence is not a variant of this construct.
    """

    contig_sha256: str = Field(description="sha256 of the construct contig")
    contig_ref: ArtifactRef = Field(
        description="the sha256-pinned bytes, so a stored record verifies itself"
    )
    feature_key: FeatureKey = Field(
        description="the feature within the contig: label, SO id, 1-based span"
    )
    host_placement: HostPlacement | None = Field(
        default=None,
        description="set for a chromosomal integration, None for an episomal replicon",
    )
    variants: list[ConstructVariantCall] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_placement(self) -> ConstructPlacement:
        """One contig identity throughout, and a feature a pointer may point at."""
        if len(self.contig_sha256) != 64 or set(self.contig_sha256) - _SHA256_HEX:
            raise ValueError(
                f"contig_sha256 must be 64 lowercase hex, got {self.contig_sha256!r}"
            )
        if self.contig_ref.sha256 != self.contig_sha256:
            raise ValueError(
                f"contig_ref pins {self.contig_ref.sha256} but the placement names "
                f"contig {self.contig_sha256}"
            )
        if self.feature_key.so_id in PLACEMENT_TARGET_REFUSED_SO:
            raise ValueError(
                f"feature {self.feature_key.token!r} may not be a placement target: "
                f"{PLACEMENT_TARGET_REFUSED_SO[self.feature_key.so_id]}"
            )
        wrong = sorted(
            {
                call.reference_contig_sha256
                for call in self.variants
                if call.reference_contig_sha256 != self.contig_sha256
            }
        )
        if wrong:
            raise ValueError(
                f"variants name contigs {wrong}, not this placement's "
                f"{self.contig_sha256}; a call is relative to the contig it is on"
            )
        return self

    @property
    def insert_span(self) -> ConstructSpan | None:
        """The interval of the contig that was transferred into the host, if integrated.

        A property, not a field: the fact is one fact, it belongs with the integration,
        and an episomal construct has none because the whole replicon is present.
        """
        return None if self.host_placement is None else self.host_placement.insert_span

    @property
    def localization(self) -> Localization:
        """Which of the two physical situations this placement describes."""
        return (
            Localization.episomal_replicon
            if self.host_placement is None
            else Localization.chromosomal_integration
        )
