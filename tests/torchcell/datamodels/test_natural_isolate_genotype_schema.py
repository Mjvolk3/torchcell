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
    NaturalIsolateDescriptor,
    NaturalLossOfFunctionCause,
    NaturalLossOfFunctionPerturbation,
    SequencePerturbation,
    SequenceVariantPerturbation,
)
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

_ADAPTER: TypeAdapter[Any] = TypeAdapter(GenePerturbationType)
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
    """A loss-of-function matrix call, keyed as that matrix keys its row."""
    fields: dict[str, Any] = dict(
        systematic_gene_name="YAL001C",
        perturbed_gene_name="YAL001C",
        gene_namespace="scerevisiae_s288c_sgd_systematic",
        strain_id="AAB",
        cause=NaturalLossOfFunctionCause.nonsense_or_sift,
        call_ref=_MATRIX,
        release_code=1768,
        release_name="YAL001C",
        release_gene_name="TFC3",
        release_span=_SPAN,
    )
    fields.update(overrides)
    return NaturalLossOfFunctionPerturbation(**fields)


def _frameshift(**overrides: Any) -> NaturalLossOfFunctionPerturbation:
    """A frameshift call released without a code: keyed by its Gene, verbatim."""
    fields: dict[str, Any] = dict(
        systematic_gene_name="PAU8",
        perturbed_gene_name="PAU8",
        cause=NaturalLossOfFunctionCause.homozygous_frameshift,
        release_code=None,
        release_name=None,
        release_gene_name="PAU8",
    )
    fields.update(overrides)
    return _call(**fields)


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
# The call's own rules: keyed as its matrix keys the row, nothing else
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"release_code": None}, "a loss-of-function matrix row carries its CODE"),
        ({"release_name": None}, "a loss-of-function matrix row carries its CODE"),
        (
            {"systematic_gene_name": "TFC3", "perturbed_gene_name": "TFC3"},
            "a loss-of-function call is keyed by the row's NAME",
        ),
        ({"perturbed_gene_name": "TFC3"}, "perturbed_gene_name is the release's"),
        ({"provenance": "engineered"}, "has provenance 'natural'"),
        ({"systematic_gene_name": "", "perturbed_gene_name": ""}, "non-empty"),
    ],
)
def test_a_call_keyed_unlike_its_row_is_refused(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _call(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"release_name": "PAU8"}, "the frameshift matrix has no NAME column"),
        (
            {"systematic_gene_name": "YAL068C", "perturbed_gene_name": "YAL068C"},
            "a frameshift row without a code is keyed by its Gene, verbatim",
        ),
    ],
)
def test_a_frameshift_call_keyed_unlike_its_row_is_refused(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _frameshift(**overrides)


def test_a_codeless_frameshift_call_keeps_the_release_gene_name_as_its_key() -> None:
    """A row the release gives no code is stored under its ``Gene``, a standard name
    here, which the yeast systematic-name pattern would refuse elsewhere.
    """
    call = _frameshift()
    assert (call.systematic_gene_name, call.release_code, call.release_name) == (
        "PAU8",
        None,
        None,
    )
    coded = _frameshift(
        systematic_gene_name="YAL001C",
        perturbed_gene_name="YAL001C",
        release_code=1768,
        release_gene_name="TFC3",
    )
    assert (coded.systematic_gene_name, coded.release_gene_name) == ("YAL001C", "TFC3")
    with pytest.raises(ValidationError, match="Invalid systematic gene"):
        KanMxDeletionPerturbation(systematic_gene_name="PAU8", perturbed_gene_name="P")


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
    assert Genotype(
        perturbations=[_call(), _frameshift(), ho], natural_isolate=_descriptor()
    )
    with pytest.raises(ValidationError, match=r"carries perturbations of \['ZZZ'\]"):
        Genotype(perturbations=[_call(strain_id="ZZZ")], natural_isolate=_descriptor())
