# tests/torchcell/datamodels/test_segregant_genotype.py
"""The segregant (haplotype-mosaic) genotype: a sibling of Genotype, not a subclass.

The two facts the design rests on: a ``SegregantGrowthExperiment`` validates and round
trips with a ``SegregantGenotype``, and the base ``Experiment`` REJECTS one, so
``TypeAdapter(ExperimentType)`` resolves the segregant class unambiguously and no
gene-keyed consumer can read a segregant as a wild-type S288C by accident.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from torchcell.datamodels.media import YPD
from torchcell.datamodels.schema import (
    AssayType,
    Environment,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentType,
    Genotype,
    HaplotypeBlock,
    MeasurementType,
    SegregantGenotype,
    SegregantGrowthExperiment,
    SegregantParent,
    Temperature,
)


def _parent(name: str, peter: str | None) -> SegregantParent:
    return SegregantParent(
        name=name,
        peter_strain_id=peter,
        assembly_member="1011Assemblies.tar.gz::GENOMES_ASSEMBLED/AAA_6.re.fa"
        if peter
        else "S288C reference",
        assembly_sha256="53540d09" if peter else "S288C reference",
        engineered_background=f"{name} MatA ho::HphMX",
    )


def _blocks() -> list[HaplotypeBlock]:
    return [
        HaplotypeBlock(
            chromosome="chrI", start=693, end=50_000, parent=1, n_markers=40
        ),
        HaplotypeBlock(
            chromosome="chrI", start=51_000, end=229_000, parent=2, n_markers=60
        ),
        HaplotypeBlock(
            chromosome="chrII", start=100, end=800_000, parent=1, n_markers=90
        ),
    ]


def _genotype() -> SegregantGenotype:
    return SegregantGenotype(
        cross="A",
        segregant_id="A01_01",
        parent_1=_parent("BYa", None),
        parent_2=_parent("RMx", "AAA"),
        blocks=_blocks(),
        call_method="R/qtl argmax.geno hard call",
        marker_matrix_sha256="21c4bed2",
    )


def _phenotype() -> EnvironmentResponsePhenotype:
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.control_regression_residual,
        assay_type=AssayType.colony_size_array,
        environment_response=-1.25,
    )


def _experiment() -> SegregantGrowthExperiment:
    return SegregantGrowthExperiment(
        dataset_name="Bloom2019Dataset",
        genotype=_genotype(),
        environment=Environment(media=YPD, temperature=Temperature(value=30.0)),
        phenotype=_phenotype(),
    )


def test_segregant_genotype_is_not_a_genotype() -> None:
    assert not issubclass(SegregantGenotype, Genotype)
    assert _genotype().n_blocks == 3


def test_experiment_round_trips_through_the_union() -> None:
    exp = _experiment()
    dumped = exp.model_dump()
    parsed: Any = TypeAdapter(ExperimentType).validate_python(dumped)
    assert isinstance(parsed, SegregantGrowthExperiment)
    assert parsed.model_dump() == dumped
    assert parsed.genotype.blocks[1].parent == 2


def test_base_experiment_rejects_a_segregant_genotype() -> None:
    dumped = _experiment().model_dump()
    dumped["experiment_type"] = "base"
    with pytest.raises(ValidationError):
        Experiment(**dumped)


@pytest.mark.parametrize(
    "bad",
    [
        dict(chromosome="chrI", start=10, end=5, parent=1, n_markers=1),
        dict(chromosome="chrI", start=1, end=5, parent=1, n_markers=0),
        dict(chromosome="chrI", start=1, end=5, parent=1, n_markers=1, posterior=1.5),
        dict(chromosome="chrI", start=1, end=5, parent=3, n_markers=1),
    ],
)
def test_block_invariants(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        HaplotypeBlock(**bad)  # type: ignore[arg-type]


def test_blocks_must_alternate_and_not_overlap() -> None:
    same_parent = _blocks()
    same_parent[1] = HaplotypeBlock(
        chromosome="chrI", start=51_000, end=229_000, parent=1, n_markers=60
    )
    with pytest.raises(ValidationError, match="adjacent blocks share parent"):
        _genotype().model_copy(update={"blocks": same_parent}).model_validate(
            {
                **_genotype().model_dump(),
                "blocks": [b.model_dump() for b in same_parent],
            }
        )
    overlapping = _blocks()
    overlapping[1] = HaplotypeBlock(
        chromosome="chrI", start=40_000, end=229_000, parent=2, n_markers=60
    )
    with pytest.raises(ValidationError, match="overlap"):
        SegregantGenotype.model_validate(
            {
                **_genotype().model_dump(),
                "blocks": [b.model_dump() for b in overlapping],
            }
        )
    with pytest.raises(ValidationError, match="at least one"):
        SegregantGenotype.model_validate({**_genotype().model_dump(), "blocks": []})
