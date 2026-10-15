# tests/torchcell/datamodels/test_single_cell_expression_schema.py
# [[tests.torchcell.datamodels.test_single_cell_expression_schema]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodels/test_single_cell_expression_schema.py
"""The one single-cell UMI-count leaf (#1016) in its Jackson 2020 shape, and the
narrowed reference and environment types of Jackson's experiment pair.

``SingleCellExpressionPhenotype`` holds Boocock 2025's and Jackson 2020's records alike
(Boocock's shape is pinned in ``test_schema.py``). Every invariant is pinned by the
message it raises: the counts are positive integers (a zero is omitted, not stored), the
markers non-negative or a typed gap, the stored counts never exceed ``total_umis`` (equal
for Jackson, whose every released feature is stored; larger for Boocock, whose total
covers features outside the analysis set), the per-cell totals sum to ``total_umis``, the
three per-cell lists are aligned with ``n_cells``, a (library, barcode) pair names one
cell, the library partition is the count of the per-cell libraries, the universe pin is a
64-hex digest at least as large as the keyed genes, and no feature is both a gene and a
marker.
"""

from __future__ import annotations

from typing import Any, get_args

import pytest
from pydantic import TypeAdapter, ValidationError

from torchcell.datamodels.media import YPD_LIQUID
from torchcell.datamodels.schema import (
    EXPERIMENT_REFERENCE_TYPE_MAP,
    EXPERIMENT_TYPE_MAP,
    BarcodedKanMxDeletionPerturbation,
    CultureEnvironment,
    Environment,
    EukaryoticStrainBackground,
    ExperimentReferenceType,
    ExperimentType,
    Genotype,
    MatingType,
    PseudobulkUmiCountExperiment,
    PseudobulkUmiCountExperimentReference,
    ReferenceGenome,
    SegregantExpressionExperiment,
    SegregantExpressionExperimentReference,
    SingleCellExpressionPhenotype,
    StrainReferenceGenome,
)
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

SHA = "0" * 64


def _gap(field: str) -> ProvenanceGap:
    return ProvenanceGap(
        field=field, reason=ProvenanceGapReason.not_reported_by_primary
    )


def _phenotype(**overrides: Any) -> SingleCellExpressionPhenotype:
    kwargs: dict[str, Any] = {
        "umi_count": {"YAL002W": 5, "YAL001C": 2},
        "marker_umi_count": {"NATMX": 0, "KANMX": 3},
        "total_umis": 10,
        "n_features": 3,
        "feature_universe_sha256": SHA,
        "n_cells": 2,
        "cell_cycle_stage_counts": None,
        "library_cell_counts": {"YPD": 2},
        "cell_barcodes": ["AAAC", "GGGT"],
        "cell_libraries": ["YPD", "YPD"],
        "cell_total_umis": [6, 4],
        "measurement_type": "scrnaseq_umi_count",
        "provenance_gaps": [_gap("cell_cycle_stage_counts")],
    }
    kwargs.update(overrides)
    return SingleCellExpressionPhenotype(**kwargs)


def _reference_genome() -> StrainReferenceGenome:
    background = EukaryoticStrainBackground(
        name="FY4/FY5",
        reference_strain="S288C",
        assembly_set=SGD_S288C_R64,
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
        assembly_set=SGD_S288C_R64,
        assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
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
    assert phenotype.marker_umi_count is not None
    assert list(phenotype.marker_umi_count) == ["KANMX", "NATMX"]
    assert phenotype.label_name == "umi_count"
    assert phenotype.measurement_type == "scrnaseq_umi_count"
    assert phenotype.gapped_fields() == {"cell_cycle_stage_counts"}
    assert repr(phenotype) == (
        "SingleCellExpressionPhenotype(genes=2, total_umis=10, n_cells=2)"
    )


def test_the_total_may_exceed_the_stored_counts_but_never_fall_below_them() -> None:
    """Boocock's total covers features outside its analysis set; Jackson's equals."""
    wider = _phenotype(total_umis=12, cell_total_umis=[7, 5])
    assert wider.total_umis - 10 == 2
    with pytest.raises(
        ValidationError,
        match=r"total_umis 9 is smaller than the stored counts 10 \(umi_count plus "
        r"marker_umi_count\)",
    ):
        _phenotype(total_umis=9, cell_total_umis=[5, 4])


def test_a_barcode_may_repeat_across_libraries_but_not_within_one() -> None:
    """Boocock's panels reuse a 10x barcode across libraries: the pair names the cell."""
    two_libraries = _phenotype(
        cell_barcodes=["AAAC-1", "AAAC-1"],
        cell_libraries=["lib_1", "lib_2"],
        library_cell_counts={"lib_1": 1, "lib_2": 1},
    )
    assert two_libraries.cell_barcodes == ["AAAC-1", "AAAC-1"]
    with pytest.raises(ValidationError, match=r"a \(library, barcode\) cell repeats"):
        _phenotype(cell_barcodes=["AAAC-1", "AAAC-1"])


def test_the_markers_may_be_a_typed_gap_and_only_with_one() -> None:
    gapped = _phenotype(
        marker_umi_count=None,
        total_umis=10,
        provenance_gaps=[_gap("cell_cycle_stage_counts"), _gap("marker_umi_count")],
    )
    assert gapped.marker_umi_count is None
    assert gapped.gapped_fields() == {"cell_cycle_stage_counts", "marker_umi_count"}
    with pytest.raises(ValidationError, match="marker_umi_count is unset"):
        _phenotype(marker_umi_count=None)


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
        ({"cell_barcodes": ["AAAC"]}, "cell_barcodes lists 1 cells for n_cells 2"),
        ({"cell_libraries": ["YPD"]}, "cell_libraries lists 1 cells for n_cells 2"),
        ({"cell_total_umis": [10]}, "cell_total_umis lists 1 cells for n_cells 2"),
        (
            {"cell_libraries": ["YPD", "MMD"]},
            "library_cell_counts is not the count of cell_libraries",
        ),
        (
            {"cell_total_umis": [11, -1]},
            "cell_total_umis holds a negative or boolean count",
        ),
        ({"cell_total_umis": [6, 5]}, "cell_total_umis sums to 11, not total_umis 10"),
        (
            {"library_cell_counts": {"YPD": 1}},
            "library_cell_counts sums to 1, not n_cells 2",
        ),
        (
            {"cell_cycle_stage_counts": {"G1": 2, "S": 0}, "provenance_gaps": []},
            "cell_cycle_stage_counts holds a non-positive count",
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
        ({"measurement_type": "  "}, "Input should be 'scrnaseq_umi_count'"),
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
                    species="Saccharomyces cerevisiae",
                    strain="S288C",
                    assembly_set=SGD_S288C_R64,
                    assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
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


def test_one_leaf_one_tag_and_the_designs_diverge_above_it() -> None:
    """Jackson 2020 and Boocock 2025 are one physical measurement, so one leaf and tag.

    Both releases are 10x Chromium 3' UMI counts: the experiment pairs differ (pooled
    TF-deletion strains, haplotype-mosaic segregants) and both hold this leaf, which
    admits exactly one ``measurement_type``. Summing a strain's cells is analysis that
    ``n_cells`` records, so the tag Jackson wrote before the unification is refused.
    """
    field = SingleCellExpressionPhenotype.model_fields["measurement_type"]
    assert get_args(field.annotation) == ("scrnaseq_umi_count",)
    assert field.default == "scrnaseq_umi_count"
    for experiment in (PseudobulkUmiCountExperiment, SegregantExpressionExperiment):
        assert (
            experiment.model_fields["phenotype"].annotation
            is SingleCellExpressionPhenotype
        )
    for reference in (
        PseudobulkUmiCountExperimentReference,
        SegregantExpressionExperimentReference,
    ):
        assert (
            reference.model_fields["phenotype_reference"].annotation
            is SingleCellExpressionPhenotype
        )
    dumped = _phenotype().model_dump()
    assert dumped["measurement_type"] == "scrnaseq_umi_count"
    del dumped["measurement_type"]
    defaulted = SingleCellExpressionPhenotype.model_validate(dumped)
    assert defaulted.measurement_type == "scrnaseq_umi_count"
    assert defaulted == _phenotype()
    with pytest.raises(ValidationError) as caught:
        _phenotype(measurement_type="scrnaseq_pseudobulk_umi_count")
    assert "Input should be 'scrnaseq_umi_count'" in str(caught.value)
