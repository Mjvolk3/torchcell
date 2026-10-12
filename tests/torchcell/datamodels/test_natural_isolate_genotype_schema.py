# tests/torchcell/datamodels/test_natural_isolate_genotype_schema.py
# [[tests.torchcell.datamodels.test_natural_isolate_genotype_schema]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodels/test_natural_isolate_genotype_schema.py
"""The #987 / #988 schema additions: the loss-of-function leaf, the isolate descriptor
on ``Genotype``, and ``DeletionPerturbation`` as a stored union member.

Placement (the DAG): ``NaturalLossOfFunctionPerturbation`` is a child of
``SequencePerturbation`` (AXIS 3) and a SIBLING of ``SequenceVariantPerturbation``, not
its child, so ``isinstance(p, SequenceVariantPerturbation)`` never counts a call as a
variant. ``NaturalIsolateDescriptor`` is composed onto ``Genotype``, never a member of
the gene-perturbation union, since nothing in it is gene-keyed.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from torchcell.datamodels.schema import (
    ArtifactRef,
    ChromosomeCopyChange,
    DeletionPerturbation,
    GenePerturbation,
    GenePerturbationType,
    GenomeZygosity,
    GenomicSpan,
    Genotype,
    IsolateManipulation,
    KanMxDeletionPerturbation,
    LossOfFunctionAlleleStatus,
    NaturalIsolateDescriptor,
    NaturalLossOfFunctionCause,
    NaturalLossOfFunctionPerturbation,
    SequencePerturbation,
    SequenceVariantPerturbation,
)
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

_ADAPTER: TypeAdapter[Any] = TypeAdapter(GenePerturbationType)
_ALLELE = ArtifactRef(
    tier="genomes",
    key="peter2018_1011_assemblies",
    path="allReferenceGenesWithSNPsAndIndelsInferred.tar.gz",
    member="YAL001C.fasta#AAB_YAL001C_TFC3",
    sha256="a" * 64,
)
_MATRIX = ArtifactRef(
    tier="genomes",
    key="peter2018_1011_assemblies",
    path="1011LossOfFunction.xls.gz",
    sha256="b" * 64,
)
_SPAN = GenomicSpan(chromosome="chrI", start=147594, end=151166, assembly="R64-1-1")
_WORKBOOK = ArtifactRef(
    tier="genomes",
    key="peter2018_1011_assemblies",
    path="41586_2018_30_MOESM3_ESM.xls",
    member="Table S1",
    sha256="c" * 64,
)


def _call(**overrides: Any) -> NaturalLossOfFunctionPerturbation:
    fields: dict[str, Any] = dict(
        systematic_gene_name="YAL001C",
        perturbed_gene_name="TFC3",
        strain_id="AAB",
        cause=NaturalLossOfFunctionCause.nonsense_sift_or_frameshift,
        allele_status=LossOfFunctionAlleleStatus.sequence_variant,
        sequence_source="peterGenomeEvolution10112018",
        sequence_ref=_ALLELE,
        call_ref=_MATRIX,
        release_code=1768,
        release_gene_name="YAL001C",
        release_span=_SPAN,
        gene_present_in_isolate=True,
    )
    fields.update(overrides)
    return NaturalLossOfFunctionPerturbation(**fields)


def _gap(field: str) -> ProvenanceGap:
    return ProvenanceGap(
        field=field, reason=ProvenanceGapReason.not_reported_by_primary, note="n"
    )


def _descriptor(**overrides: Any) -> NaturalIsolateDescriptor:
    fields: dict[str, Any] = dict(
        isolate="AAB",
        annotation_ref=_WORKBOOK,
        genetic_manipulation=IsolateManipulation.ho_deletion,
        ploidy=1,
        zygosity=GenomeZygosity.homozygous,
        chromosome_copy_changes=[],
    )
    fields.update(overrides)
    return NaturalIsolateDescriptor(**fields)


# --------------------------------------------------------------------------- #
# DAG placement
# --------------------------------------------------------------------------- #
def test_the_call_is_a_sibling_of_the_variant_under_sequence_perturbation() -> None:
    assert NaturalLossOfFunctionPerturbation.__mro__[1] is SequencePerturbation
    assert SequenceVariantPerturbation.__mro__[1] is SequencePerturbation
    assert not issubclass(
        NaturalLossOfFunctionPerturbation, SequenceVariantPerturbation
    )
    assert issubclass(NaturalLossOfFunctionPerturbation, GenePerturbation)
    call = _call()
    assert (call.provenance, call.mechanism_so_id, call.mechanism_so_name) == (
        "natural",
        "SO:0002054",
        "loss_of_function_variant",
    )


def test_both_union_members_round_trip_as_themselves() -> None:
    call = _call()
    assert type(_ADAPTER.validate_python(call.model_dump())) is (
        NaturalLossOfFunctionPerturbation
    )
    ho = DeletionPerturbation(
        systematic_gene_name="YDL227C", perturbed_gene_name="HO", description="d"
    )
    back = _ADAPTER.validate_python(ho.model_dump())
    assert type(back) is DeletionPerturbation
    assert back.perturbation_type == "deletion"
    kanmx = KanMxDeletionPerturbation(
        systematic_gene_name="YAL001C", perturbed_gene_name="TFC3"
    )
    assert type(_ADAPTER.validate_python(kanmx.model_dump())) is (
        KanMxDeletionPerturbation
    )


def test_the_descriptor_is_not_a_gene_perturbation() -> None:
    assert not issubclass(NaturalIsolateDescriptor, GenePerturbation)
    assert Genotype.model_fields["natural_isolate"].default is None


# --------------------------------------------------------------------------- #
# The call's own rules
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"allele_status": LossOfFunctionAlleleStatus.allele_equals_reference},
            "sequence_ref is set exactly when allele_status is sequence_variant",
        ),
        (
            {"sequence_ref": None, "sequence_source": None},
            "sequence_ref is set exactly when allele_status is sequence_variant",
        ),
        ({"sequence_source": None}, "sequence_source is set exactly when"),
        (
            {"release_code": None, "gene_present_in_isolate": None},
            "only a frameshift row is released without a pangenome code",
        ),
        (
            {
                "release_code": None,
                "cause": NaturalLossOfFunctionCause.homozygous_frameshift,
            },
            "a row without a pangenome code has no presence-matrix cell",
        ),
        (
            {"gene_present_in_isolate": None},
            "a row with a pangenome code carries its presence-matrix cell",
        ),
        ({"provenance": "engineered"}, "has provenance 'natural'"),
        ({"systematic_gene_name": "1-EC1118_1F14_0012g"}, "Invalid systematic gene"),
    ],
)
def test_a_call_that_misstates_its_evidence_is_refused(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _call(**overrides)


def test_a_codeless_frameshift_call_with_no_allele_is_valid() -> None:
    call = _call(
        cause=NaturalLossOfFunctionCause.homozygous_frameshift,
        allele_status=LossOfFunctionAlleleStatus.no_inferred_allele,
        sequence_ref=None,
        sequence_source=None,
        release_code=None,
        release_gene_name="PAU8",
        gene_present_in_isolate=None,
    )
    assert (call.release_code, call.gene_present_in_isolate) == (None, None)


# --------------------------------------------------------------------------- #
# The descriptor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "field", ["genetic_manipulation", "ploidy", "chromosome_copy_changes"]
)
def test_a_descriptor_element_is_set_or_gapped(field: str) -> None:
    with pytest.raises(ValidationError, match=f"{field} is None and carries no"):
        _descriptor(**{field: None})
    gapped = _descriptor(**{field: None}, provenance_gaps=[_gap(field)])
    assert getattr(gapped, field) is None


def test_a_gapped_element_cannot_also_hold_a_value() -> None:
    with pytest.raises(ValidationError, match="ploidy"):
        _descriptor(provenance_gaps=[_gap("ploidy")])


def test_copy_changes_are_one_per_chromosome_and_nonzero() -> None:
    with pytest.raises(ValidationError, match="a chromosome appears twice"):
        _descriptor(
            chromosome_copy_changes=[
                ChromosomeCopyChange(chromosome=9, change=1),
                ChromosomeCopyChange(chromosome=9, change=-1),
            ]
        )
    with pytest.raises(ValidationError, match="cannot be 0"):
        ChromosomeCopyChange(chromosome=9, change=0)
    with pytest.raises(ValidationError, match="less than or equal to 16"):
        ChromosomeCopyChange(chromosome=17, change=1)
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        _descriptor(ploidy=0)


# --------------------------------------------------------------------------- #
# Genotype.natural_isolate
# --------------------------------------------------------------------------- #
def test_a_genotype_without_a_descriptor_dumps_exactly_as_before() -> None:
    """``natural_isolate`` is left out of the dump when None, so every genotype stored
    before #988 serializes to the same bytes and keeps its content id.
    """
    genotype = Genotype(
        perturbations=[
            KanMxDeletionPerturbation(
                systematic_gene_name="YAL001C", perturbed_gene_name="TFC3"
            )
        ]
    )
    assert list(genotype.model_dump()) == ["perturbations"]
    assert '"natural_isolate"' not in genotype.model_dump_json()
    assert Genotype.model_validate(genotype.model_dump()) == genotype


def test_a_descriptor_round_trips_and_takes_part_in_equality() -> None:
    genotype = Genotype(perturbations=[_call()], natural_isolate=_descriptor())
    dump = genotype.model_dump()
    assert dump["natural_isolate"]["isolate"] == "AAB"
    back = Genotype.model_validate(dump)
    assert back.model_dump() == dump
    assert back == genotype
    other = Genotype(
        perturbations=[_call()],
        natural_isolate=_descriptor(zygosity=GenomeZygosity.heterozygous),
    )
    assert other != genotype
    assert hash(genotype.natural_isolate) == hash(back.natural_isolate)


def test_a_descriptor_refuses_a_perturbation_of_another_isolate() -> None:
    ho = DeletionPerturbation(
        systematic_gene_name="YDL227C", perturbed_gene_name="HO", description="d"
    )
    assert Genotype(perturbations=[_call(), ho], natural_isolate=_descriptor())
    with pytest.raises(ValidationError, match=r"carries perturbations of \['ZZZ'\]"):
        Genotype(perturbations=[_call(strain_id="ZZZ")], natural_isolate=_descriptor())
