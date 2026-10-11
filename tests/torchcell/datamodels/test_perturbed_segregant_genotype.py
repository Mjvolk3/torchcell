# tests/torchcell/datamodels/test_perturbed_segregant_genotype.py
# [[tests.torchcell.datamodels.test_perturbed_segregant_genotype]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodels/test_perturbed_segregant_genotype.py
"""The perturbation-on-segregant genotype (Hale 2024): a mosaic plus designed edits.

What the class promises, each pinned by a test: at least one perturbation (an unperturbed
segregant is a ``SegregantGenotype``); perturbations sorted exactly as ``Genotype`` sorts
them; a parent role only over a one-parent mosaic of that parent; the gene-keyed
properties read the perturbations only; and ``TypeAdapter(ExperimentType)`` resolves a
``segregant_perturbation`` record to its own experiment class while the base
``Experiment`` and the segregant growth experiment both reject the composite genotype.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from torchcell.datamodels.media import SC_URA
from torchcell.datamodels.schema import (
    EXPERIMENT_REFERENCE_TYPE_MAP,
    EXPERIMENT_TYPE_MAP,
    ArtifactRef,
    AssayType,
    CrisprConstruct,
    CrisprInterferencePerturbation,
    Environment,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentType,
    GenePerturbationType,
    HaplotypeBlock,
    MeasurementType,
    PerturbedSegregantGenotype,
    SegregantGenotype,
    SegregantGrowthExperiment,
    SegregantParent,
    SegregantPerturbationExperiment,
    SegregantPerturbationExperimentReference,
    Temperature,
)


def _parent(name: str) -> SegregantParent:
    return SegregantParent(
        name=name,
        peter_strain_id=None,
        assembly_ref=ArtifactRef(
            tier="genomes",
            key="sgd_S288C_R64-4-1_20230830",
            path="x.fsa",
            sha256="d" * 64,
        ),
        engineered_background=f"{name} fcy1Δ",
    )


def _background(parents: list[int]) -> SegregantGenotype:
    blocks = [
        HaplotypeBlock(
            chromosome="chrI",
            start=100 * i + 1,
            end=100 * i + 50,
            parent=p,  # type: ignore[arg-type]
            n_markers=2,
        )
        for i, p in enumerate(parents)
    ]
    return SegregantGenotype(
        cross="BYx3S",
        segregant_id="seg",
        parent_1=_parent("BY"),
        parent_2=_parent("3S"),
        blocks=blocks,
        call_method="hard calls",
        marker_matrix_sha256="0" * 64,
    )


def _crispri(orf: str, name: str, guide: str) -> CrisprInterferencePerturbation:
    return CrisprInterferencePerturbation(
        systematic_gene_name=orf,
        perturbed_gene_name=name,
        crispr=CrisprConstruct(effector="dCas9-Mxi1", guide_sequence=guide, n_guides=1),
    )


def _phenotype() -> EnvironmentResponsePhenotype:
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.relative_fitness,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=-0.25,
        screen_id="ATC1",
        replicate_id="AAA_CCC",
    )


def test_perturbations_are_sorted_and_feed_the_gene_properties() -> None:
    genotype = PerturbedSegregantGenotype(
        background=_background([1, 2]),
        background_role="segregant",
        perturbations=[
            _crispri("YLR314C", "CDC3", "TTTCGTAAAAAGGAACCACC"),
            _crispri("YDR116C", "MRPL1", "TTTTCTCTTTTCGCTTAAAA"),
        ],
    )
    assert genotype.systematic_gene_names == ["YDR116C", "YLR314C"]
    assert genotype.perturbed_gene_names == ["MRPL1", "CDC3"]
    assert genotype.perturbation_types == ["crispr_interference"] * 2
    assert len(genotype) == 2


def test_no_perturbation_is_refused() -> None:
    with pytest.raises(
        ValidationError, match="unperturbed segregant is a SegregantGenotype"
    ):
        PerturbedSegregantGenotype(
            background=_background([1]), background_role="segregant", perturbations=[]
        )


def test_a_parent_role_needs_a_one_parent_mosaic_of_that_parent() -> None:
    pert: list[GenePerturbationType] = [
        _crispri("YDR116C", "MRPL1", "TTTTCTCTTTTCGCTTAAAA")
    ]
    ok = PerturbedSegregantGenotype(
        background=_background([1]), background_role="parent_1", perturbations=pert
    )
    assert ok.background_role == "parent_1"
    with pytest.raises(ValidationError, match="1 block\\(s\\) carry the other parent"):
        PerturbedSegregantGenotype(
            background=_background([1, 2]),
            background_role="parent_1",
            perturbations=pert,
        )
    with pytest.raises(ValidationError, match="carry the other parent"):
        PerturbedSegregantGenotype(
            background=_background([1]), background_role="parent_2", perturbations=pert
        )


def _record() -> dict[str, Any]:
    experiment = SegregantPerturbationExperiment(
        dataset_name="CrispriSegregantHale2024Dataset",
        genotype=PerturbedSegregantGenotype(
            background=_background([1, 2]),
            background_role="segregant",
            perturbations=[_crispri("YDR116C", "MRPL1", "TTTTCTCTTTTCGCTTAAAA")],
        ),
        environment=Environment(media=SC_URA, temperature=Temperature(value=21.0)),
        phenotype=_phenotype(),
    )
    return experiment.model_dump()


def test_the_experiment_union_resolves_the_segregant_perturbation_class() -> None:
    dump = _record()
    assert dump["experiment_type"] == "segregant_perturbation"
    model: object = TypeAdapter(ExperimentType).validate_python(dump)
    assert type(model) is SegregantPerturbationExperiment
    assert model.model_dump() == dump
    assert (
        EXPERIMENT_TYPE_MAP["segregant_perturbation"] is SegregantPerturbationExperiment
    )
    assert (
        EXPERIMENT_REFERENCE_TYPE_MAP["segregant_perturbation"]
        is SegregantPerturbationExperimentReference
    )


def test_neither_sibling_experiment_accepts_the_composite_genotype() -> None:
    dump = _record()
    with pytest.raises(ValidationError):
        Experiment.model_validate(dump)
    with pytest.raises(ValidationError):
        SegregantGrowthExperiment.model_validate(
            {**dump, "experiment_type": "segregant_growth"}
        )


def test_relative_fitness_is_a_measurement_type() -> None:
    assert MeasurementType.relative_fitness.value == "relative_fitness"
    assert _phenotype().measurement_type is MeasurementType.relative_fitness
