# tests/torchcell/datamodels/test_pseudobulk_umi_count.py
# [[tests.torchcell.datamodels.test_pseudobulk_umi_count]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodels/test_pseudobulk_umi_count.py
"""The pseudobulk UMI-count family (Jackson 2020): what the phenotype refuses, and the
narrowed reference and environment types of its experiment pair.

Every invariant the phenotype states is pinned by the message it raises: the counts are
positive integers (a zero is omitted, not stored), the markers non-negative, the per-cell
totals and the two dicts each sum to ``total_umis``, the barcode and total lists are
aligned and the barcodes distinct, the universe pin is a 64-hex digest at least as large
as the keyed genes, and no feature is both a gene and a marker.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from torchcell.datamodels.media import YPD_LIQUID
from torchcell.datamodels.schema import (
    EXPERIMENT_REFERENCE_TYPE_MAP,
    EXPERIMENT_TYPE_MAP,
    BarcodedKanMxDeletionPerturbation,
    CultureEnvironment,
    Environment,
    ExperimentReferenceType,
    ExperimentType,
    Genotype,
    MatingType,
    PseudobulkUmiCountExperiment,
    PseudobulkUmiCountExperimentReference,
    PseudobulkUmiCountPhenotype,
    ReferenceGenome,
    StrainBackground,
    StrainReferenceGenome,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

SHA = "0" * 64


def _phenotype(**overrides: Any) -> PseudobulkUmiCountPhenotype:
    kwargs: dict[str, Any] = {
        "umi_count": {"YAL002W": 5, "YAL001C": 2},
        "marker_umi_count": {"NATMX": 0, "KANMX": 3},
        "total_umis": 10,
        "n_features": 3,
        "feature_universe_sha256": SHA,
        "n_cells": 2,
        "cell_barcodes": ["AAAC", "GGGT"],
        "cell_total_umis": [6, 4],
    }
    kwargs.update(overrides)
    return PseudobulkUmiCountPhenotype(**kwargs)


def _reference_genome() -> StrainReferenceGenome:
    background = StrainBackground(
        name="FY4/FY5",
        parents=["FY4", "FY5"],
        mating_type=MatingType.a_alpha,
        ploidy="diploid",
        provenance=[
            SourcedValue(
                value="FY4/FY5",
                provenance=Provenance(
                    source_uri="paper.md", citation_key="test2020", sha256=SHA
                ),
                quote="FY4 (MATa) or FY5",
            )
        ],
    )
    return StrainReferenceGenome(
        species="Saccharomyces cerevisiae",
        strain="FY4/FY5",
        ploidy="diploid",
        background=background,
    )


def _genotype() -> Genotype:
    return Genotype(
        perturbations=[
            BarcodedKanMxDeletionPerturbation(
                systematic_gene_name="YFL021W",
                perturbed_gene_name="GAT1",
                barcode="TGGGAGTGCGCCGTCGGCG",
                cassette="KanMX/NatMX",
            )
        ]
    )


def test_a_valid_phenotype_sorts_both_dicts_and_summarizes_itself() -> None:
    phenotype = _phenotype()
    assert list(phenotype.umi_count) == ["YAL001C", "YAL002W"]
    assert list(phenotype.marker_umi_count) == ["KANMX", "NATMX"]
    assert phenotype.label_name == "umi_count"
    assert phenotype.measurement_type == "scrnaseq_pseudobulk_umi_count"
    assert repr(phenotype) == (
        "PseudobulkUmiCountPhenotype(genes=2, total_umis=10, n_cells=2)"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"umi_count": {}}, "umi_count cannot be empty"),
        (
            {"umi_count": [("YAL001C", 1)]},
            "umi_count must be a per-gene dict, got list",
        ),
        (
            {"umi_count": {"YAL001C": 0}, "total_umis": 3, "cell_total_umis": [2, 1]},
            "umi_count for YAL001C must be a positive integer (zeros are omitted, not "
            "stored), got 0",
        ),
        (
            {"umi_count": {"YAL001C": True}},
            "umi_count for YAL001C must be a positive integer",
        ),
        (
            {"marker_umi_count": {"KANMX": -1}},
            "marker_umi_count for KANMX must be a non-negative integer, got -1",
        ),
        (
            {"marker_umi_count": ["KANMX"]},
            "marker_umi_count must be a per-marker dict, got list",
        ),
        (
            {"n_cells": 0, "cell_barcodes": [], "cell_total_umis": []},
            "n_cells must be a positive integer, got 0",
        ),
        ({"cell_barcodes": ["AAAC"]}, "1 cell barcodes for n_cells 2"),
        ({"cell_barcodes": ["AAAC", "AAAC"]}, "cell_barcodes repeats a barcode"),
        ({"cell_total_umis": [10]}, "1 per-cell totals for n_cells 2"),
        (
            {"cell_total_umis": [11, -1]},
            "cell_total_umis holds a negative or boolean count",
        ),
        ({"cell_total_umis": [6, 5]}, "cell_total_umis sums to 11, not total_umis 10"),
        (
            {"total_umis": 11, "cell_total_umis": [6, 5]},
            "umi_count + marker_umi_count sum to 10, not total_umis 11",
        ),
        ({"n_features": 1}, "n_features 1 is smaller than the 2 keyed features"),
        (
            {"umi_count": {"KANMX": 2, "YAL002W": 5}, "marker_umi_count": {"KANMX": 3}},
            "a feature is keyed as both a gene and a marker",
        ),
        (
            {"feature_universe_sha256": "ABC"},
            "feature_universe_sha256 must be 64 lowercase hex digits",
        ),
        ({"measurement_type": "  "}, "measurement_type cannot be blank"),
    ],
)
def test_the_phenotype_refuses_inconsistent_counts(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError) as caught:
        _phenotype(**overrides)
    assert message in str(caught.value)


def test_the_experiment_states_a_culture_environment_and_round_trips() -> None:
    environment = CultureEnvironment(media=YPD_LIQUID, duration_hours=4.0)
    experiment = PseudobulkUmiCountExperiment(
        dataset_name="d",
        genotype=_genotype(),
        environment=environment,
        phenotype=_phenotype(),
    )
    dumped = experiment.model_dump()
    assert dumped["experiment_type"] == "pseudobulk_umi_count"
    assert "culture_format" in dumped["environment"]
    resolved: object = TypeAdapter(ExperimentType).validate_python(dumped)
    assert type(resolved) is PseudobulkUmiCountExperiment
    assert resolved == experiment
    with pytest.raises(ValidationError, match="instance of CultureEnvironment"):
        PseudobulkUmiCountExperiment.model_validate(
            {
                "dataset_name": "d",
                "genotype": _genotype(),
                "environment": Environment(media=YPD_LIQUID),
                "phenotype": _phenotype(),
            }
        )


def test_the_reference_keeps_its_typed_background() -> None:
    reference = PseudobulkUmiCountExperimentReference(
        dataset_name="d",
        genome_reference=_reference_genome(),
        environment_reference=CultureEnvironment(media=YPD_LIQUID),
        phenotype_reference=_phenotype(),
    )
    dumped = reference.model_dump()
    assert dumped["experiment_reference_type"] == "pseudobulk_umi_count"
    assert dumped["genome_reference"]["background"]["mating_type"] == "a/alpha"
    resolved: object = TypeAdapter(ExperimentReferenceType).validate_python(dumped)
    assert type(resolved) is PseudobulkUmiCountExperimentReference
    assert resolved == reference
    with pytest.raises(ValidationError, match="instance of StrainReferenceGenome"):
        PseudobulkUmiCountExperimentReference.model_validate(
            {
                "dataset_name": "d",
                "genome_reference": ReferenceGenome(
                    species="Saccharomyces cerevisiae", strain="S288C"
                ),
                "environment_reference": CultureEnvironment(media=YPD_LIQUID),
                "phenotype_reference": _phenotype(),
            }
        )


def test_the_family_is_registered_under_one_type_tag() -> None:
    assert EXPERIMENT_TYPE_MAP["pseudobulk_umi_count"] is PseudobulkUmiCountExperiment
    assert (
        EXPERIMENT_REFERENCE_TYPE_MAP["pseudobulk_umi_count"]
        is PseudobulkUmiCountExperimentReference
    )
