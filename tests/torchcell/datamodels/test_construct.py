# tests/torchcell/datamodels/test_construct.py
# [[tests.torchcell.datamodels.test_construct]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodels/test_construct.py
"""``ConstructPlacement``, ``HostPlacement`` and the contig-relative variant call.

The models a gene addition will be composed with. Each test names the defect it defends
against, because every invariant here exists because its absence produced a measured
defect: a silent ``None`` pointer in 26 of 26 served stores, a landing pad with no
coordinate written as if it had one, an allele carried as the free string ``"K229L"``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from torchcell.datamodels.construct import (
    ConstructPlacement,
    ConstructVariantCall,
    HostPlacement,
    Localization,
)
from torchcell.datamodels.schema import (
    ArtifactRef,
    AssemblySet,
    GenomicSpan,
    VariantCallMode,
    VariantFrequencyBasis,
    VariantType,
)
from torchcell.sequence.construct import ConstructSpan, FeatureKey
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

CONTIG = "63f89c1eaa13f888610ed781c4de12c8ac34f4e20512b4eb2c4f32d31af096da"
OTHER = "bbaca60509e079222f7cdd1bb54b70d93445af3a699c65109d94b553fc0087f7"
S288C: AssemblySet = "sgd_S288C_R64-4-1_20230830"
CENPK: AssemblySet = "scerevisiae_CENPK113-7D_ASM257140v2"
REF = ArtifactRef(
    tier="library",
    key="cacheraCRISPAHighthroughputMethod2023",
    path="si/plasmids/pBTX002.gb",
    sha256=CONTIG,
    bytes=21901,
)
ARO4 = FeatureKey(
    label="ARO4-K229L",
    so_id="SO:0000316",
    span=ConstructSpan(start=3001, end=4113, orientation="inline"),
)
PRIMER_KEY = FeatureKey(
    label="HOP181",
    so_id="SO:0005850",
    span=ConstructSpan(start=10, end=30, orientation="inline"),
)


def _gap(field: str) -> ProvenanceGap:
    return ProvenanceGap(
        field=field, reason=ProvenanceGapReason.not_reported_by_primary
    )


def _host(**kwargs: object) -> HostPlacement:
    base: dict[str, object] = {
        "assembly_set": S288C,
        "site_label": "XII-5",
        "insert_span": ConstructSpan(start=2003, end=10105, orientation="inline"),
        "provenance_gaps": [_gap("span")],
    }
    base.update(kwargs)
    return HostPlacement(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# HostPlacement
# --------------------------------------------------------------------------- #
def test_a_named_landing_pad_with_no_coordinate_is_a_typed_gap() -> None:
    """``XII-5`` is a named pad, not a coordinate, and that is the common case.

    The site label is what the source states and is required; the coordinate is set or
    carries a ``ProvenanceGap``. Nothing defaults, and the genome never gates the
    record: an integration at an unmapped pad is storable and countable.
    """
    host = _host()
    assert host.site_label == "XII-5"
    assert host.span is None
    assert host.gapped_fields() == {"span"}
    assert host.insert_span is not None
    assert host.insert_span.length == 8103


def test_a_host_placement_with_neither_a_span_nor_a_gap_is_refused() -> None:
    """The silent ``None`` is the defect; an absence is typed or it is not admitted."""
    with pytest.raises(ValidationError, match="span is unset and carries no"):
        HostPlacement(
            assembly_set=S288C,
            site_label="XII-5",
            insert_span=ConstructSpan(start=1, end=9, orientation="inline"),
        )


def test_an_unannotated_insert_span_is_gapped_too() -> None:
    """The homology arms are not always annotated, so the insert span is gap-able."""
    host = _host(insert_span=None, provenance_gaps=[_gap("span"), _gap("insert_span")])
    assert host.insert_span is None
    assert host.gapped_fields() == {"span", "insert_span"}


def test_a_host_placement_needs_the_sources_own_site_label() -> None:
    """A coordinate does not replace what the source called the site."""
    with pytest.raises(ValidationError, match="site_label"):
        HostPlacement(
            assembly_set=S288C,
            site_label="  ",
            span=GenomicSpan(chromosome="XII", start=1, end=9, assembly=S288C),
            insert_span=ConstructSpan(start=1, end=9, orientation="inline"),
        )


def test_a_span_on_another_assembly_is_refused() -> None:
    """A coordinate belongs to the assembly the placement pins, not another one.

    ``GenomicSpan.assembly`` is the ``AssemblySet`` vocabulary (P2), so the other
    assembly is a real member, the CEN.PK113-7D set, not a free string.
    """
    with pytest.raises(ValidationError, match="but the placement pins"):
        HostPlacement(
            assembly_set=S288C,
            site_label="XII-5",
            span=GenomicSpan(chromosome="XII", start=1, end=9, assembly=CENPK),
            insert_span=ConstructSpan(start=1, end=9, orientation="inline"),
        )


def test_a_clean_insertion_replaces_nothing_and_needs_no_gap() -> None:
    """``replaced_span`` is a real ``None`` for a clean insertion, not an absent value.

    This is the distinction a typed gap exists to protect: "replaced nothing" is a fact
    about the integration, while "we do not know the coordinate" is a fact about the
    source. Only the second is a gap.
    """
    host = _host()
    assert host.replaced_span is None
    assert "replaced_span" not in host.gapped_fields()


# --------------------------------------------------------------------------- #
# ConstructPlacement
# --------------------------------------------------------------------------- #
def test_a_placement_pins_one_contig_through_both_its_fields() -> None:
    """The contig's sha256 and the ref's must agree: a stored record verifies itself."""
    placement = ConstructPlacement(
        contig_sha256=CONTIG, contig_ref=REF, feature_key=ARO4, host_placement=_host()
    )
    assert placement.localization is Localization.chromosomal_integration
    assert placement.insert_span == placement.host_placement.insert_span  # type: ignore[union-attr]
    with pytest.raises(ValidationError, match="but the placement names contig"):
        ConstructPlacement(contig_sha256=OTHER, contig_ref=REF, feature_key=ARO4)


def test_an_episomal_placement_has_no_host_site_and_no_insert_span() -> None:
    """A present plasmid is a replicon of its own: the whole contig is in the cell.

    There is nothing between homology arms to be the insert, which is why the insert
    span lives with the integration and reads ``None`` here rather than being a second
    nullable field a loader could half-fill.
    """
    placement = ConstructPlacement(
        contig_sha256=CONTIG, contig_ref=REF, feature_key=ARO4
    )
    assert placement.host_placement is None
    assert placement.insert_span is None
    assert placement.localization is Localization.episomal_replicon


def test_a_primer_cannot_be_a_placement_target() -> None:
    """297 of 570 features in the Cachera set are primers; none may be pointed at."""
    with pytest.raises(ValidationError, match="may not be a placement target"):
        ConstructPlacement(contig_sha256=CONTIG, contig_ref=REF, feature_key=PRIMER_KEY)


def test_a_placement_hashes_so_it_can_sit_in_a_genotype() -> None:
    """``Genotype.__eq__`` compares perturbations as a SET, so a placement must hash.

    A frozen pydantic model hashes the tuple of its field values, so an unhashable
    nested object would make every perturbation carrying it unhashable and break
    genotype equality. ``HashableProvenanceGapMixin`` hashes canonical JSON, which
    agrees with pydantic's field-wise equality.
    """
    first = ConstructPlacement(
        contig_sha256=CONTIG, contig_ref=REF, feature_key=ARO4, host_placement=_host()
    )
    second = ConstructPlacement(
        contig_sha256=CONTIG, contig_ref=REF, feature_key=ARO4, host_placement=_host()
    )
    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1


def test_a_placement_round_trips_through_json() -> None:
    """A placement reloaded from its dump equals the live model (the LMDB path)."""
    placement = ConstructPlacement(
        contig_sha256=CONTIG,
        contig_ref=REF,
        feature_key=ARO4,
        host_placement=_host(),
        variants=[
            ConstructVariantCall(
                variant_type=VariantType.snv,
                variant_statement="K229L",
                reference_contig_sha256=CONTIG,
                position_start=3685,
                position_end=3685,
                sequence_change="A->T",
                amino_acid_change="K -> L",
                codon_number=229,
            )
        ],
    )
    assert ConstructPlacement.model_validate(placement.model_dump()) == placement


# --------------------------------------------------------------------------- #
# ConstructVariantCall
# --------------------------------------------------------------------------- #
def test_a_variant_on_an_added_gene_is_a_coordinate_and_a_quote() -> None:
    """Cachera's ``ARO4-K229L`` is a coordinate on the contig, with the name kept.

    Today it is the free string ``variant="K229L"``, which is not a sequence fact. The
    typed call carries the position AND the source's own statement, so the verbatim
    claim survives beside the coordinate and a disagreement stays visible.
    """
    call = ConstructVariantCall(
        variant_type=VariantType.snv,
        variant_statement="K229L",
        reference_contig_sha256=CONTIG,
        position_start=3685,
        position_end=3685,
        reference_allele="A",
        alternate_allele="T",
        amino_acid_change="K -> L",
        codon_number=229,
    )
    assert call.variant_statement == "K229L"
    assert call.call_mode is None
    assert call.frequency is None


def test_a_call_against_another_contig_is_refused() -> None:
    """A coordinate against another sequence is not a variant of this construct."""
    stray = ConstructVariantCall(
        variant_type=VariantType.snv,
        variant_statement="Y20F",
        reference_contig_sha256=OTHER,
        position_start=59,
        position_end=59,
        sequence_change="A->T",
    )
    with pytest.raises(ValidationError, match="not this placement's"):
        ConstructPlacement(
            contig_sha256=CONTIG, contig_ref=REF, feature_key=ARO4, variants=[stray]
        )


def test_the_contig_reference_is_a_hash_not_a_free_string() -> None:
    """``GenomeVariantCall`` names a replicon by a released string; a construct has a hash."""
    with pytest.raises(ValidationError, match="64 lowercase hex"):
        ConstructVariantCall(
            variant_type=VariantType.snv,
            variant_statement="K229L",
            reference_contig_sha256="pBTX2",
            position_start=1,
            position_end=1,
            sequence_change="A->T",
        )


def test_an_insertion_may_end_one_before_it_starts() -> None:
    """An insertion sits in a zero-length reference interval, as the releases write it."""
    call = ConstructVariantCall(
        variant_type=VariantType.insertion,
        variant_statement="a single A inserted at -10",
        reference_contig_sha256=CONTIG,
        position_start=200,
        position_end=199,
        sequence_change="+A",
    )
    assert (call.position_start, call.position_end) == (200, 199)
    with pytest.raises(ValidationError, match="position_start <= position_end"):
        ConstructVariantCall(
            variant_type=VariantType.snv,
            variant_statement="x",
            reference_contig_sha256=CONTIG,
            position_start=200,
            position_end=199,
            sequence_change="A->T",
        )


def test_a_frequency_states_its_scale_and_its_mode() -> None:
    """A fraction is never read as a percent, and a clone fraction is not a population's."""
    called = ConstructVariantCall(
        variant_type=VariantType.snv,
        variant_statement="G->A at 0.9",
        reference_contig_sha256=CONTIG,
        position_start=10,
        position_end=10,
        sequence_change="G->A",
        call_mode=VariantCallMode.clone,
        frequency=0.9,
        frequency_basis=VariantFrequencyBasis.fraction,
    )
    assert called.frequency_basis == "fraction"
    with pytest.raises(ValidationError, match="no frequency_basis"):
        ConstructVariantCall.model_validate(
            called.model_dump() | {"frequency_basis": None}
        )
    with pytest.raises(ValidationError, match="cannot be read"):
        ConstructVariantCall.model_validate(called.model_dump() | {"call_mode": None})


def test_a_basis_without_a_frequency_is_refused() -> None:
    """A range or an absent cell has no scale to state."""
    with pytest.raises(ValidationError, match="frequency is None"):
        ConstructVariantCall(
            variant_type=VariantType.snv,
            variant_statement="x",
            reference_contig_sha256=CONTIG,
            position_start=1,
            position_end=1,
            sequence_change="A->T",
            frequency_basis=VariantFrequencyBasis.fraction,
        )


def test_localization_has_exactly_the_two_physical_situations() -> None:
    """Two members: the replicon class is a property of the contig, not of the gene."""
    assert [member.value for member in Localization] == [
        "chromosomal_integration",
        "episomal_replicon",
    ]
