# tests/torchcell/datasets/ecoli/test_campos2018.py
# [[tests.torchcell.datasets.ecoli.test_campos2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_campos2018.py
"""Campos 2018 loader: the 26-feature inventory, the medium, the rows, the records.

The synthetic tests run everywhere. The identifier tests read the REAL
``EcoliK12BW25113Genome`` over the synthetic BW25113 assembly of
``tests/torchcell/sequence/genome/_bacterial_fixtures.py`` through a stubbed ``resolve``
with the network refused. Derived expectations for that assembly: ``thrL`` and its
synonym ``ECK0001`` both reach ``BW25113_0001`` (a merged-locus collision with no direct
member), ``ECK0005`` sits on two loci (ambiguous), ``yaaP`` is a pseudogene locus, and
``nosuchgene`` resolves to nothing.

The data tests (``--data``) audit every module-level ``SourcedValue`` against the
sha256-pinned paper OCR and the pinned Dataset EV2 legend, read the real Dataset EV2
from the raw mirror, pin the measured row and identifier histograms, and re-run the
affine cross-check of the derived fitness against the released score.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import torchcell.datasets.ecoli.campos2018 as c
from tests.torchcell.datasets._genome_injection_fakes import (
    FakeBW25113Genome,
    install_bacterial_fakes,
)
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    BW25113_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.datamodels.bacterial_morphology_features import (
    CAMPOS2018_MORPHOLOGY_ASSAY as MORPHOLOGY_ASSAY,
)
from torchcell.datamodels.media import M9, MEDIA_LIBRARY
from torchcell.datamodels.schema import (
    ASSEMBLY_SET_ACCESSIONS,
    BACTERIAL_ASSEMBLY_SETS,
    AssemblyReferenceGenome,
    BacterialDeletionPerturbation,
    BacterialFitnessExperiment,
    BacterialFitnessExperimentReference,
    ComponentDefinition,
    ConcentrationUnit,
    MediaComponentRole,
    SampleUnit,
)
from torchcell.datasets.bacteria_common import (
    BacterialGenomeInjector,
    LocusTagResolutionError,
)
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.sequence.genome.ecoli.k12 import (
    BW25113_ASSEMBLY,
    EcoliK12BW25113Genome,
    EcoliK12StrainName,
)
from torchcell.verification.sourced import ProvenanceGapReason, audit_sourced_value

# --------------------------------------------------------------------------- #
# The 26-feature inventory
# --------------------------------------------------------------------------- #


def test_the_phenotype_is_twenty_six_features_split_nineteen_two_five() -> None:
    assert len(c.MORPHOLOGICAL_FEATURES) == 19
    assert len(c.GROWTH_FEATURES) == 2
    assert len(c.CELL_CYCLE_FEATURES) == 5
    assert len(c.PAPER_FEATURES) == 26
    assert len(set(c.PAPER_FEATURES)) == 26
    assert c.MORPHOLOGICAL_COUNT.value == 19
    assert c.CELL_CYCLE_COUNT.value == 5
    assert c.GROWTH_FEATURE_DEFINITION.value == c.GROWTH_FEATURES
    # the paper's own cross-sum of the morphological and cell cycle groups
    assert c.FEATURE_CROSS_SUM.value == 24


def test_the_nineteen_morphological_features_are_the_mean_and_cv_pairs_plus_cv_dr() -> (
    None
):
    means = [f for f in c.MORPHOLOGICAL_FEATURES if f.startswith("<")]
    cvs = [f for f in c.MORPHOLOGICAL_FEATURES if f.startswith("CV_")]
    assert len(means) == 9
    assert len(cvs) == 10
    # every mean has its CV; the division ratio has a CV and no mean, which is why the
    # count is 19 and not 20
    paired = {f.strip("<>") for f in means}
    assert {f.removeprefix("CV_") for f in cvs} - paired == {"DR"}
    assert "<DR>" not in c.MORPHOLOGICAL_FEATURES


def test_the_two_datasets_together_leave_exactly_one_feature_unserved() -> None:
    """Issue #774 closed 25 of the 26 gaps; only the saturating density is left."""
    assert c.SERVED_FEATURE == "alpha_max"
    assert c.SERVED_FEATURE in c.GROWTH_FEATURES
    assert set(c.UNSERVED_FEATURES) == {"ODmax"}
    served = {c.SERVED_FEATURE, *c.MORPHOLOGY_FEATURES}
    assert set(c.PAPER_FEATURES) - served == {"ODmax"}
    # the morphology dataset also serves the two nucleoid-area symbols the main text's
    # headline count of 19 leaves out, so it serves 2 more than PAPER_FEATURES names
    assert set(c.MORPHOLOGY_FEATURES) - set(c.PAPER_FEATURES) == {"<NA>", "CV_NA"}


def test_the_one_unserved_feature_states_its_exact_schema_mismatch() -> None:
    reason = c.UNSERVED_FEATURES["ODmax"]
    assert "MeasurementType has no member" in reason
    assert "not morphology" in reason


def test_the_morphology_vocabulary_is_table_s1s_twenty_six_symbols() -> None:
    assert len(c.MORPHOLOGY_FEATURES) == 26
    assert len(set(c.MORPHOLOGY_FEATURES)) == 26
    assert set(c.MORPHOLOGY_FEATURES) == set(MORPHOLOGY_ASSAY.by_symbol)
    assert set(c.GROWTH_FEATURES).isdisjoint(c.MORPHOLOGY_FEATURES)
    assert len(c.MORPHOLOGY_REQUIRED_FEATURES) == 19
    assert len(c.MORPHOLOGY_DAPI_FEATURES) == 7
    assert c.MORPHOLOGY_REQUIRED_FEATURES.isdisjoint(c.MORPHOLOGY_DAPI_FEATURES)
    assert c.MORPHOLOGY_REQUIRED_FEATURES | c.MORPHOLOGY_DAPI_FEATURES == set(
        c.MORPHOLOGY_FEATURES
    )


def test_the_two_released_columns_outside_the_vocabulary_are_recorded_as_redundant() -> (
    None
):
    """Not dropped for want of a class: each is an exact inverse of a stored feature."""
    assert set(c.RELEASED_COLUMNS_OUTSIDE_THE_VOCABULARY) == {"%non-div", "%1N"}
    assert set(c.RELEASED_COLUMNS_OUTSIDE_THE_VOCABULARY).isdisjoint(
        c.UNSERVED_FEATURES
    )
    assert "Rel.timing div" in c.RELEASED_COLUMNS_OUTSIDE_THE_VOCABULARY["%non-div"]
    assert "Rel.timing nuc" in c.RELEASED_COLUMNS_OUTSIDE_THE_VOCABULARY["%1N"]
    assert c.RELEASED_CLUSTER_COLUMNS == ("MorphoIsland", "CellCyleIsland")


def test_the_column_header_of_a_feature_is_its_symbol_plus_its_released_unit() -> None:
    """Built from the vocabulary, so a header that moved stops the build."""
    assert c.MORPHOLOGY_COLUMNS["<L>"] == "<L> (\u00b5m)"
    assert c.MORPHOLOGY_COLUMNS["<SA/V>"] == "<SA/V> (\u00b5m-1)"
    assert c.MORPHOLOGY_COLUMNS["CV_L"] == "CV_L"
    assert c.MORPHOLOGY_COLUMNS["%2N"] == "%2N"
    unitless = [f for f in MORPHOLOGY_ASSAY.features if f.unit is None]
    assert len(unitless) == 18
    for feature in unitless:
        assert c.MORPHOLOGY_COLUMNS[feature.symbol] == feature.symbol


def test_the_calmorph_vocabulary_rejects_every_campos_feature_name() -> None:
    """The exact mismatch the module records: the shape fits, the vocabulary does not."""
    from torchcell.datamodels.calmorph_labels import (
        CALMORPH_LABELS,
        CALMORPH_STATISTICS,
    )
    from torchcell.datamodels.schema import CalMorphPhenotype

    vocabulary = set(CALMORPH_LABELS) | set(CALMORPH_STATISTICS)
    assert vocabulary.isdisjoint(c.PAPER_FEATURES)
    with pytest.raises(ValueError, match="Invalid CalMorph base parameter"):
        CalMorphPhenotype(calmorph={c.MORPHOLOGICAL_FEATURES[0]: 3.2})


# --------------------------------------------------------------------------- #
# Media and environment
# --------------------------------------------------------------------------- #
def test_the_medium_is_the_m9_base_plus_the_two_stated_supplements() -> None:
    media = c.CAMPOS2018_M9_CASAMINO_GLUCOSE
    assert media.base_medium == "M9"
    assert media.state == "liquid"
    assert media.is_synthetic is False
    assert len(media.components) == len(M9.components) + 2
    added = media.components[len(M9.components) :]
    assert [component.compound.name for component in added] == [
        "casamino acids",
        "D-glucose",
    ]
    assert [component.role for component in added] == [
        MediaComponentRole.complex_ingredient,
        MediaComponentRole.carbon_source,
    ]
    doses = [component.concentration for component in added]
    assert [(d.value, d.unit) for d in doses if d is not None] == [
        (0.1, ConcentrationUnit.percent_w_v),
        (0.2, ConcentrationUnit.percent_w_v),
    ]


def test_casamino_acids_is_typed_undefined_rather_than_given_a_fake_structure() -> None:
    casamino = next(
        component
        for component in c.CAMPOS2018_M9_CASAMINO_GLUCOSE.components
        if component.compound.name == "casamino acids"
    )
    assert casamino.definition is ComponentDefinition.intrinsically_undefined
    assert casamino.compound.inchikey is None
    assert casamino.compound.chebi_id is None


def test_the_medium_derives_from_a_shared_library_base() -> None:
    assert c.CAMPOS2018_M9_CASAMINO_GLUCOSE.base_medium in MEDIA_LIBRARY
    assert c.CAMPOS2018_M9_CASAMINO_GLUCOSE.name not in MEDIA_LIBRARY


def test_the_environment_is_one_unperturbed_medium_at_thirty_degrees() -> None:
    env = c.environment()
    assert env.media is c.CAMPOS2018_M9_CASAMINO_GLUCOSE
    assert env.temperature is not None
    assert env.temperature.value == 30.0
    assert env.perturbations == []
    assert env.aerobicity == "aerobic"
    assert env.duration_hours is None
    gaps = {gap.field: gap.reason for gap in env.provenance_gaps}
    assert gaps == {"duration_hours": ProvenanceGapReason.not_reported_by_primary}


# --------------------------------------------------------------------------- #
# Phenotype and genotype
# --------------------------------------------------------------------------- #
def test_the_phenotype_is_the_growth_rate_ratio_with_the_release_gapped() -> None:
    phenotype = c.phenotype(0.0049, 0.0098)
    assert phenotype.fitness == pytest.approx(0.5)
    assert phenotype.n_samples == 1
    assert phenotype.sample_unit is SampleUnit.biological_replicate
    assert phenotype.fitness_uncertainty is None
    assert phenotype.fitness_uncertainty_type is None
    assert phenotype.fitness_se is None
    gaps = {gap.field: gap.reason for gap in phenotype.provenance_gaps}
    assert gaps == {
        "fitness_uncertainty": ProvenanceGapReason.not_reported_by_primary,
        "fitness_se": ProvenanceGapReason.not_reported_by_primary,
    }


def test_the_reference_is_the_parent_at_one_over_its_released_replicates() -> None:
    reference = c.reference_phenotype()
    assert reference.fitness == 1.0
    assert reference.n_samples == c.WILD_TYPE_ROWS == 240
    assert reference.sample_unit is SampleUnit.biological_replicate
    assert c.reference_phenotype(7).n_samples == 7


def test_the_genotype_is_one_keio_deletion_with_its_cassette_and_well() -> None:
    record = c.StrainRecord(
        row=c.StrainRow(label="thrA", plate="3", well="B012", alpha_max=0.0098),
        locus_tag="BW25113_0002",
    )
    genotype = c.genotype(record)
    (perturbation,) = genotype.perturbations
    assert isinstance(perturbation, BacterialDeletionPerturbation)
    assert perturbation.systematic_gene_name == "BW25113_0002"
    assert perturbation.perturbed_gene_name == "thrA"
    assert perturbation.gene_namespace == "ecoli_k12_bw25113_locus_tag"
    assert perturbation.collection == c.KEIO_COLLECTION == "Keio collection"
    assert perturbation.cassette == c.KEIO_CASSETTE == "kanamycin-resistance cassette"
    assert perturbation.identifier_mapping is not None
    assert perturbation.identifier_mapping.route == "gene_symbol"
    assert perturbation.identifier_mapping.source_identifier == "thrA"
    assert perturbation.construction is not None
    assert (perturbation.construction.plate, perturbation.construction.well) == (
        "3",
        "B012",
    )


def test_a_locus_tag_of_another_strains_namespace_is_refused() -> None:
    with pytest.raises(ValueError):
        BacterialDeletionPerturbation(
            systematic_gene_name="b0002",
            perturbed_gene_name="thrA",
            gene_namespace=c.BW25113_NAMESPACE,
        )


# --------------------------------------------------------------------------- #
# The reader
# --------------------------------------------------------------------------- #
#: The synthetic release: wild type, the four writable labels, then one row per rule.
_SYNTHETIC_ROWS: tuple[tuple[str | None, str | None, str | None, float | None], ...] = (
    ("WT0001", "1", "A001", 0.0100),
    ("WT0001", "1", "A002", 0.0098),
    ("WT0001", "1", "A003", 0.0102),
    ("thrA", "1", "A004", 0.0049),
    ("hokC", "1", "A005", 0.0098),
    ("yaaP", "1", "A006", 0.0147),
    ("yaaX", "1", "A007", 0.0196),
    ("thrL", "1", "A008", 0.0090),
    ("ECK0001", "1", "A009", 0.0091),
    ("ECK0005", "1", "A010", 0.0092),
    ("nosuchgene", "1", "A011", 0.0093),
    ("hfq*", "1", "A012", 0.0094),
    ("fabH°", "1", "A013", 0.0095),
    ("proB", "1", "A014", 0.0096),
    ("proB", "2", "A014", 0.0097),
    (None, None, "Mean", 0.0098),
)
_SYNTHETIC_RELEASED = len(_SYNTHETIC_ROWS)
_SYNTHETIC_WILD_TYPE = 3
_SYNTHETIC_KEPT = 4
#: The wild-type median of the three synthetic replicates.
_SYNTHETIC_DENOMINATOR = 0.0100


#: The row of the synthetic release that has no DAPI channel, so its 7 nucleoid-derived
#: features are NaN, which is the shape the real release has on 278 of its strains.
_SYNTHETIC_NO_DAPI_LABEL = "yaaX"
#: A value per declared statistic, inside the bound that statistic implies, so the
#: synthetic release exercises the per-feature L2 bound rather than one pooled range.
_SYNTHETIC_BY_STATISTIC: dict[str, float] = {
    "mean": 2.5,
    "coefficient_of_variation": 0.25,
    "pearson_correlation": 0.6,
    "regression_intercept": 0.4,
    "fraction_of_cells": 0.2,
    "inferred_relative_timing": 0.8,
}


def _morphology_columns(rows: Any) -> dict[str, list[float | None]]:
    """The 26 morphology columns of a synthetic "Normalized data" sheet."""
    columns: dict[str, list[float | None]] = {}
    for feature in MORPHOLOGY_ASSAY.features:
        base = _SYNTHETIC_BY_STATISTIC[feature.statistic.value]
        values: list[float | None] = []
        for i, row in enumerate(rows):
            if row[0] is None:  # a footer row carries no feature value
                values.append(None)
            elif (
                row[0] == _SYNTHETIC_NO_DAPI_LABEL
                and feature.symbol in c.MORPHOLOGY_DAPI_FEATURES
            ):
                values.append(None)
            else:
                values.append(round(base + i / 1000, 4))
        columns[c.MORPHOLOGY_COLUMNS[feature.symbol]] = values
    return columns


def _cell_counts(path: Path, rows: Any = _SYNTHETIC_ROWS) -> Path:
    """Write a synthetic Dataset EV1: the Keio position and its segmented-cell count.

    Dataset EV1 carries no footer summary, so the labelled rows of Dataset EV2 are its
    whole content, which is the row alignment the join checks.
    """
    labelled = [row for row in rows if row[0] is not None]
    frame = pd.DataFrame(
        {
            c.LABEL_COLUMN: [row[0] for row in labelled],
            c.CELL_COUNT_COLUMN: [200 + i for i, _ in enumerate(labelled)],
            c.PLATE_COLUMN: [row[1] for row in labelled],
            c.WELL_COLUMN: [row[2] for row in labelled],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name=c.RAW_SHEET, index=False)
    return path


def _release(path: Path, rows: Any = _SYNTHETIC_ROWS) -> Path:
    """Write a synthetic Dataset EV2 with a normalized sheet and a scores sheet."""
    frame = pd.DataFrame(
        {
            c.LABEL_COLUMN: [row[0] for row in rows],
            c.PLATE_COLUMN: [row[1] for row in rows],
            c.WELL_COLUMN: [row[2] for row in rows],
            c.ALPHA_COLUMN: [row[3] for row in rows],
            **_morphology_columns(rows),
        }
    )
    scores = frame.rename(columns={c.ALPHA_COLUMN: c.SERVED_FEATURE})
    scores[c.SERVED_FEATURE] = [
        None if row[3] is None else (row[3] / _SYNTHETIC_DENOMINATOR - 1) * 10.0
        for row in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name=c.NORMALIZED_SHEET, index=False)
        scores.to_excel(writer, sheet_name=c.SCORES_SHEET, index=False)
    return path


@pytest.fixture
def shaped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the reader's shape checks to the synthetic release's shape."""
    monkeypatch.setattr(c, "RELEASED_ROWS", _SYNTHETIC_RELEASED)
    monkeypatch.setattr(c, "WILD_TYPE_ROWS", _SYNTHETIC_WILD_TYPE)


def test_the_reader_splits_footer_wild_type_and_mutant_rows(
    tmp_path: Path, shaped: None
) -> None:
    table = c.read_normalized_table(_release(tmp_path / "ev2.xlsx"))
    assert table.released_rows == _SYNTHETIC_RELEASED
    assert table.footer_rows == 1
    assert len(table.wild_type) == _SYNTHETIC_WILD_TYPE
    assert len(table.mutants) == _SYNTHETIC_RELEASED - 1 - _SYNTHETIC_WILD_TYPE
    assert {row.label for row in table.wild_type} == {"WT0001"}
    assert table.wild_type_median_alpha_max == pytest.approx(_SYNTHETIC_DENOMINATOR)


def test_the_median_of_an_even_number_of_replicates_is_the_midpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = (*_SYNTHETIC_ROWS, ("WT0001", "2", "A001", 0.0104))
    monkeypatch.setattr(c, "RELEASED_ROWS", len(rows))
    monkeypatch.setattr(c, "WILD_TYPE_ROWS", 4)
    table = c.read_normalized_table(_release(tmp_path / "ev2.xlsx", rows))
    assert table.wild_type_median_alpha_max == pytest.approx((0.0100 + 0.0102) / 2)


def test_the_reader_refuses_a_release_whose_row_count_moved(tmp_path: Path) -> None:
    path = _release(tmp_path / "ev2.xlsx")
    with pytest.raises(ValueError, match=r"has 16 rows, expected 4471"):
        c.read_normalized_table(path)


def test_the_reader_refuses_a_missing_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame({c.LABEL_COLUMN: ["thrA"], c.PLATE_COLUMN: ["1"]})
    path = tmp_path / "ev2.xlsx"
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name=c.NORMALIZED_SHEET, index=False)
    with pytest.raises(ValueError, match="is missing columns"):
        c.read_normalized_table(path)


def test_the_reader_refuses_a_repeated_plate_well_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = (*_SYNTHETIC_ROWS, ("yqiG", "1", "A004", 0.0098))
    monkeypatch.setattr(c, "RELEASED_ROWS", len(rows))
    monkeypatch.setattr(c, "WILD_TYPE_ROWS", _SYNTHETIC_WILD_TYPE)
    with pytest.raises(ValueError, match="positions appear twice"):
        c.read_normalized_table(_release(tmp_path / "ev2.xlsx", rows))


def test_the_reader_refuses_a_wild_type_count_that_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(c, "RELEASED_ROWS", _SYNTHETIC_RELEASED)
    monkeypatch.setattr(c, "WILD_TYPE_ROWS", 240)
    with pytest.raises(ValueError, match=r"3 wild-type rows, expected 240"):
        c.read_normalized_table(_release(tmp_path / "ev2.xlsx"))


def test_a_table_with_no_wild_type_row_has_no_fitness_denominator() -> None:
    table = c.NormalizedTable(
        released_rows=1,
        footer_rows=0,
        wild_type=(),
        mutants=(c.StrainRow(label="thrA", plate="1", well="A001", alpha_max=0.0098),),
    )
    with pytest.raises(ValueError, match="no fitness denominator"):
        _ = table.wild_type_median_alpha_max


# --------------------------------------------------------------------------- #
# Row resolution against the synthetic BW25113 assembly
# --------------------------------------------------------------------------- #
@pytest.fixture
def bw25113(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12BW25113Genome:
    """The synthetic BW25113 genome; the network refuses."""
    files = write_assembly(tmp_path / "tier", BW25113_ASSEMBLY, BW25113_LOCI)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    return EcoliK12BW25113Genome(genome_root=str(tmp_path / "bw25113"), overwrite=True)


def test_each_row_rule_claims_its_own_labels(
    tmp_path: Path,
    shaped: None,
    bw25113: EcoliK12BW25113Genome,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(c, "MIN_RESOLVED_FRACTION", 0.7)
    table = c.read_normalized_table(_release(tmp_path / "ev2.xlsx"))
    resolution = c.resolve_rows(table, bw25113, label="synthetic")
    assert [(r.row.label, r.locus_tag) for r in resolution.kept] == [
        ("thrA", "BW25113_0002"),
        ("hokC", "BW25113_4412"),
        ("yaaP", "BW25113_0004"),
        ("yaaX", "BW25113_0008"),
    ]
    assert resolution.dropped_labels == {
        c.DROP_FOOTER: [],
        c.DROP_WILD_TYPE: ["WT0001"],
        c.DROP_ANNOTATED: ["fabH°", "hfq*"],
        c.DROP_NOT_IN_ANNOTATION: ["nosuchgene"],
        c.DROP_MERGED_LOCUS: ["ECK0001", "thrL"],
        c.DROP_AMBIGUOUS: ["ECK0005"],
        c.DROP_DUPLICATE_LABEL: ["proB"],
    }
    assert resolution.dropped_rows == {
        c.DROP_FOOTER: 1,
        c.DROP_WILD_TYPE: 3,
        c.DROP_ANNOTATED: 2,
        c.DROP_NOT_IN_ANNOTATION: 1,
        c.DROP_MERGED_LOCUS: 2,
        c.DROP_AMBIGUOUS: 1,
        c.DROP_DUPLICATE_LABEL: 2,
    }
    assert (
        table.released_rows - sum(resolution.dropped_rows.values()) == _SYNTHETIC_KEPT
    )


def test_a_pseudogene_locus_is_kept_rather_than_dropped(
    tmp_path: Path,
    shaped: None,
    bw25113: EcoliK12BW25113Genome,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(c, "MIN_RESOLVED_FRACTION", 0.7)
    table = c.read_normalized_table(_release(tmp_path / "ev2.xlsx"))
    resolution = c.resolve_rows(table, bw25113, label="synthetic")
    assert "BW25113_0004" in {record.locus_tag for record in resolution.kept}


def test_a_release_below_the_resolution_threshold_stops_instead_of_dropping(
    tmp_path: Path,
    shaped: None,
    bw25113: EcoliK12BW25113Genome,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(c, "MIN_RESOLVED_FRACTION", 0.99)
    table = c.read_normalized_table(_release(tmp_path / "ev2.xlsx"))
    with pytest.raises(LocusTagResolutionError, match=r"7 of 9 names \(0\.778\)"):
        c.resolve_rows(table, bw25113, label="synthetic")


class _NoDrops:
    """A reconciliation that drops nothing, so the post-rule guards are what act."""

    retired_kept: tuple[str, ...] = ()
    kept_on_collision: tuple[str, ...] = ()
    ambiguous_kept: dict[str, tuple[str, ...]] = {}

    def require_resolved(self, minimum: float) -> None:
        return None


def _stub_reconciler(tags: Any) -> Any:
    """A ``reconcile_locus_tags`` stub that returns ``tags`` for the rows it is given."""

    def _stub(genome: Any, names: Any, *, label: str) -> tuple[Any, Any]:
        return pd.Series(tags(len(names))), _NoDrops()

    return _stub


def test_two_kept_rows_claiming_one_locus_stop_the_build(
    tmp_path: Path,
    shaped: None,
    bw25113: EcoliK12BW25113Genome,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The last line of defence: two kept rows on one locus tag is never written."""
    table = c.read_normalized_table(_release(tmp_path / "ev2.xlsx"))
    monkeypatch.setattr(
        c, "reconcile_locus_tags", _stub_reconciler(lambda n: ["BW25113_0002"] * n)
    )
    with pytest.raises(RuntimeError, match="claimed by more than one kept row"):
        c.resolve_rows(table, bw25113, label="synthetic")


def test_a_kept_tag_outside_the_pinned_assembly_stops_the_build(
    tmp_path: Path,
    shaped: None,
    bw25113: EcoliK12BW25113Genome,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = c.read_normalized_table(_release(tmp_path / "ev2.xlsx"))
    monkeypatch.setattr(
        c,
        "reconcile_locus_tags",
        _stub_reconciler(lambda n: [f"BW25113_9{i:03d}" for i in range(n)]),
    )
    with pytest.raises(RuntimeError, match="not loci of the pinned assembly"):
        c.resolve_rows(table, bw25113, label="synthetic")


def test_the_build_refuses_a_drop_ledger_that_does_not_add_up(
    tmp_path: Path, mirrored: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = c.resolve_rows

    def _lying(table: Any, genome: Any, *, label: str) -> Any:
        resolution = real(table, genome, label=label)
        resolution.dropped_rows[c.DROP_FOOTER] += 1
        return resolution

    monkeypatch.setattr(c, "resolve_rows", _lying)
    with pytest.raises(RuntimeError, match="drop accounting mismatch"):
        c.GrowthRateCampos2018Dataset(root=str(tmp_path / "dataset"))


def test_the_library_dir_points_at_this_keys_literature_mirror() -> None:
    assert c.library_dir("/root") == Path(
        "/root/torchcell-library/camposGenomewidePhenotypicAnalysis2018"
    )


def test_the_rule_table_lists_every_rule_once_with_a_description() -> None:
    rules = [rule for rule, _ in c.ROW_RULES]
    assert rules == [
        c.DROP_FOOTER,
        c.DROP_WILD_TYPE,
        c.DROP_ANNOTATED,
        c.DROP_NOT_IN_ANNOTATION,
        c.DROP_MERGED_LOCUS,
        c.DROP_AMBIGUOUS,
        c.DROP_DUPLICATE_LABEL,
    ]
    assert len(set(rules)) == len(rules)
    assert all(len(description) > 40 for _, description in c.ROW_RULES)


def test_the_annotation_marker_matches_only_the_two_legend_symbols() -> None:
    assert c.ANNOTATION_MARKER.search("hfq*") is not None
    assert c.ANNOTATION_MARKER.search("fabH°") is not None
    assert c.ANNOTATION_MARKER.search("hfq") is None
    assert c.MARKER_REIMAGED.value == "*"
    assert c.MARKER_CHECKED.value == "°"


# --------------------------------------------------------------------------- #
# The raw mirror
# --------------------------------------------------------------------------- #
@pytest.fixture
def deposited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A tmp ``DATA_ROOT`` whose raw mirror holds both synthetic released tables."""
    source = _release(tmp_path / "source" / "ev2.xlsx")
    counts = _cell_counts(tmp_path / "source" / "ev1.xlsx")
    monkeypatch.setattr(
        c, "DATA_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        c, "DATASET_EV1_SHA256", hashlib.sha256(counts.read_bytes()).hexdigest()
    )
    data_root = tmp_path / "data_root"
    c.deposit_raw_mirror(
        data_path=source, cell_counts_path=counts, data_root=str(data_root)
    )
    return source, data_root


def test_the_deposit_is_idempotent_and_refuses_a_changed_mirror_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deposited: tuple[Path, Path]
) -> None:
    source, data_root = deposited
    counts = tmp_path / "source" / "ev1.xlsx"
    assert c.deposit_raw_mirror(
        data_path=source, cell_counts_path=counts, data_root=str(data_root)
    ) == (c.raw_mirror_dir(str(data_root)))
    mirrored = c.raw_mirror_dir(str(data_root)) / c.DATA_REL
    mirrored.write_bytes(b"not the released bytes")
    with pytest.raises(RuntimeError, match="different sha256; refusing"):
        c.deposit_raw_mirror(
            data_path=source, cell_counts_path=counts, data_root=str(data_root)
        )


def test_the_deposit_refuses_bytes_that_do_not_match_the_pin(tmp_path: Path) -> None:
    other = tmp_path / "other.xlsx"
    other.write_bytes(b"not the released bytes")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        c.deposit_raw_mirror(
            data_path=other, cell_counts_path=other, data_root=str(tmp_path / "root")
        )


def test_the_deposit_refuses_cell_counts_that_do_not_match_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dataset EV2 matching its pin does not excuse Dataset EV1 missing its own."""
    source = _release(tmp_path / "source" / "ev2.xlsx")
    monkeypatch.setattr(
        c, "DATA_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    other = tmp_path / "other.xlsx"
    other.write_bytes(b"not the released bytes")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        c.deposit_raw_mirror(
            data_path=source, cell_counts_path=other, data_root=str(tmp_path / "root")
        )


def test_the_manifest_records_the_rerunnable_pmc_retrieval(
    deposited: tuple[Path, Path],
) -> None:
    _, data_root = deposited
    manifest = c.load_manifest(str(data_root))
    # both released tables are build inputs, so both carry their own retrieval record
    assert [record.path for record in manifest.files] == [c.DATA_REL, c.CELL_COUNTS_REL]
    for record, key in (
        (manifest.files[0], c.PMC_CLOUD_KEY),
        (manifest.files[1], c.CELL_COUNTS_PMC_CLOUD_KEY),
    ):
        assert record.retrieval is not None
        assert record.retrieval.method == "pmc_cloud"
        assert (
            record.retrieval.retriever
            == "torchcell.literature.retrieve.pmc_cloud_object"
        )
        assert record.retrieval.params == {"key": key}
    assert c.manifest_sha256(manifest, c.DATA_REL) == c.DATA_SHA256
    assert c.manifest_sha256(manifest, c.CELL_COUNTS_REL) == c.DATASET_EV1_SHA256
    with pytest.raises(KeyError, match="not in the raw manifest"):
        c.manifest_sha256(manifest, "data/nope.xlsx")


def test_si_expected_names_the_artifacts_that_are_deliberately_not_duplicated(
    deposited: tuple[Path, Path],
) -> None:
    _, data_root = deposited
    expected = " ".join(c.load_manifest(str(data_root)).si_expected)
    # Dataset EV1 is no longer among them: the morphology loader consumes it, so it is
    # mirrored with its own record rather than named as deliberately absent
    assert c.RAW_CELL_COUNTS_FILENAME in expected
    assert c.APPENDIX_SHA256 in expected
    assert "NOT duplicated here" in expected
    assert "NOT mirrored" in expected


# --------------------------------------------------------------------------- #
# Registry and genome injection
# --------------------------------------------------------------------------- #
def test_the_dataset_is_registered_and_takes_the_bw25113_genome_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        dataset_registry["GrowthRateCampos2018Dataset"] is c.GrowthRateCampos2018Dataset
    )
    assert c.GrowthRateCampos2018Dataset.REFERENCE_STRAIN == "BW25113"
    install_bacterial_fakes(monkeypatch)
    injector = BacterialGenomeInjector(data_root="/nowhere")
    kwargs = injector.genome_kwargs(c.GrowthRateCampos2018Dataset)
    assert set(kwargs) == {"ecoli_genome"}
    assert isinstance(kwargs["ecoli_genome"], FakeBW25113Genome)


def test_the_loader_declares_the_bacterial_fitness_classes() -> None:
    dataset = c.GrowthRateCampos2018Dataset.__new__(c.GrowthRateCampos2018Dataset)
    assert dataset.experiment_class is BacterialFitnessExperiment
    assert dataset.reference_class is BacterialFitnessExperimentReference
    assert dataset.raw_file_names == [c.DATA_FILENAME]
    frame = object()
    assert dataset.preprocess_raw(frame) is frame
    with pytest.raises(NotImplementedError, match="builds records in process"):
        dataset.create_experiment()


# --------------------------------------------------------------------------- #
# The whole loader, hermetic
# --------------------------------------------------------------------------- #
def _pin(strain: EcoliK12StrainName) -> AssemblyReferenceGenome:
    assembly_set = BACTERIAL_ASSEMBLY_SETS[strain]
    return AssemblyReferenceGenome(
        species="Escherichia coli",
        strain=strain,
        assembly_set=cast(Any, assembly_set),
        assembly_accession=ASSEMBLY_SET_ACCESSIONS[assembly_set][0],
    )


@pytest.fixture
def mirrored(
    monkeypatch: pytest.MonkeyPatch,
    shaped: None,
    bw25113: EcoliK12BW25113Genome,
    deposited: tuple[Path, Path],
) -> Path:
    """The deposited synthetic mirror, with the build's genome and pin stubbed."""
    _, data_root = deposited
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setattr(c, "MIN_RESOLVED_FRACTION", 0.7)
    monkeypatch.setattr(
        c, "bacterial_genome", lambda host, strain, data_root=None: bw25113
    )
    monkeypatch.setattr(c, "assembly_reference", _pin)
    return data_root


def test_the_loader_builds_the_synthetic_release_end_to_end(
    tmp_path: Path, mirrored: Path
) -> None:
    root = tmp_path / "dataset"
    dataset = c.GrowthRateCampos2018Dataset(root=str(root))
    assert len(dataset) == _SYNTHETIC_KEPT
    assert sorted(dataset.gene_set) == [
        "BW25113_0002",
        "BW25113_0004",
        "BW25113_0008",
        "BW25113_4412",
    ]
    references = dataset.experiment_reference_index
    assert references is not None
    assert len(references) == 1
    first = dataset[0]
    experiment = first["experiment"]
    assert experiment["phenotype"]["fitness"] == pytest.approx(0.49)
    assert experiment["phenotype"]["n_samples"] == 1
    assert experiment["phenotype"]["sample_unit"] == "biological_replicate"
    (perturbation,) = experiment["genotype"]["perturbations"]
    assert perturbation["systematic_gene_name"] == "BW25113_0002"
    assert perturbation["cassette"] == c.KEIO_CASSETTE
    assert experiment["environment"]["temperature"]["value"] == 30.0
    assert experiment["environment"]["perturbations"] == []
    reference = first["reference"]["phenotype_reference"]
    assert reference["fitness"] == 1.0
    assert reference["n_samples"] == _SYNTHETIC_WILD_TYPE
    assert first["publication"]["doi"] == c.DOI
    assert (root / "raw" / c.DATA_FILENAME).is_file()
    assert (root / "preprocess" / "build_manifest.json").is_file()


def test_the_build_writes_the_three_ledgers_with_the_arithmetic_closed(
    tmp_path: Path, mirrored: Path
) -> None:
    root = tmp_path / "dataset"
    c.GrowthRateCampos2018Dataset(root=str(root))
    dropped = json.loads((root / "preprocess" / "dropped_records.json").read_text())
    assert dropped["released_rows"] == _SYNTHETIC_RELEASED
    assert dropped["kept_records"] == _SYNTHETIC_KEPT
    assert (
        dropped["released_rows"] - dropped["dropped_records"]
        == (dropped["kept_records"])
    )
    assert [rule["rule"] for rule in dropped["rules"]] == [
        rule for rule, _ in c.ROW_RULES
    ]
    identifiers = json.loads(
        (root / "preprocess" / "identifier_reconciliation.json").read_text()
    )
    assert identifiers["identifier_route"] == "gene_symbol"
    assert identifiers["reconciliation"]["gene_namespace"] == c.BW25113_NAMESPACE
    features = json.loads((root / "preprocess" / "served_features.json").read_text())
    assert features["served"] == ["alpha_max"]
    assert len(features["paper_features"]) == 26
    assert list(features["unserved"]) == ["ODmax"]
    assert features["wild_type_replicates"] == _SYNTHETIC_WILD_TYPE
    assert features["wild_type_median_alpha_max"] == pytest.approx(
        _SYNTHETIC_DENOMINATOR
    )


def test_the_released_score_is_an_affine_function_of_the_derived_fitness(
    tmp_path: Path, mirrored: Path
) -> None:
    """The docstring's cross-check, on the synthetic release built the same way."""
    path = c.raw_mirror_dir(str(mirrored)) / c.DATA_REL
    table = c.read_normalized_table(path)
    scores = c.released_alpha_max_scores(path)
    denominator = table.wild_type_median_alpha_max
    fitness = np.array([row.alpha_max / denominator for row in table.mutants])
    released = np.array([scores[(row.plate, row.well)] for row in table.mutants])
    slope, intercept = np.polyfit(fitness, released, 1)
    assert slope == pytest.approx(10.0, abs=1e-6)
    assert intercept == pytest.approx(-10.0, abs=1e-6)
    assert float(np.abs(released - (slope * fitness + intercept)).max()) < 1e-9


def test_a_direct_run_opens_the_bw25113_genome_itself(
    mirrored: Path, bw25113: EcoliK12BW25113Genome
) -> None:
    dataset = c.GrowthRateCampos2018Dataset.__new__(c.GrowthRateCampos2018Dataset)
    dataset.ecoli_genome = None
    dataset.name = "GrowthRateCampos2018Dataset"
    assert dataset._genome() is bw25113
    assert dataset.ecoli_genome is bw25113


def test_the_loader_refuses_a_genome_of_the_wrong_strain() -> None:
    from torchcell.sequence.genome.ecoli.k12 import EcoliK12MG1655Genome

    dataset = c.GrowthRateCampos2018Dataset.__new__(c.GrowthRateCampos2018Dataset)
    dataset.name = "GrowthRateCampos2018Dataset"
    dataset.ecoli_genome = cast(Any, object.__new__(EcoliK12MG1655Genome))
    with pytest.raises(TypeError, match="needs the BW25113 genome"):
        dataset._genome()


def test_download_refuses_a_mirror_whose_file_is_gone(mirrored: Path) -> None:
    (c.raw_mirror_dir(str(mirrored)) / c.DATA_REL).unlink()
    dataset = c.GrowthRateCampos2018Dataset.__new__(c.GrowthRateCampos2018Dataset)
    with pytest.raises(RuntimeError, match="required raw artifact missing"):
        dataset.download()


def test_verify_build_passes_on_the_synthetic_build(
    tmp_path: Path, mirrored: Path, bw25113: EcoliK12BW25113Genome
) -> None:
    root = tmp_path / "dataset"
    c.GrowthRateCampos2018Dataset(root=str(root))
    report = c.verify_build(str(root), genome=bw25113, expected_count=_SYNTHETIC_KEPT)
    assert report.passed, report.summary()
    rows = {result.name: result for result in report.results}
    assert rows["count"].details["observed"] == _SYNTHETIC_KEPT
    assert rows["pair_uniqueness"].details["n_duplicated"] == 0
    assert rows["reference_one"].passed
    assert (root / "preprocess" / "verification_report.json").exists()


# --------------------------------------------------------------------------- #
# Data-gated: the real mirror
# --------------------------------------------------------------------------- #
@pytest.mark.data
def test_every_paper_sourced_value_is_backed_by_its_verbatim_quote() -> None:
    import os
    import os.path as osp

    from torchcell.verification.sourced import SourcedValue

    root = osp.join(os.environ["DATA_ROOT"], "torchcell-library")
    declared = {
        id(v)
        for v in (
            *c.SOURCED_VALUES,
            *c.LEGEND_SOURCED_VALUES,
            *c.APPENDIX_SOURCED_VALUES,
        )
    }
    module_values = [v for v in vars(c).values() if isinstance(v, SourcedValue)]
    # every module-level SourcedValue is in one of the three audited tuples
    assert {id(v) for v in module_values} == declared
    assert len(c.SOURCED_VALUES) == 20
    for value in c.SOURCED_VALUES:
        result = audit_sourced_value(value, root)
        assert result.passed, f"{value.value!r}: {result.message}"


@pytest.mark.data
@pytest.mark.parametrize(
    "sourced",
    c.APPENDIX_SOURCED_VALUES,
    ids=[v.quote[:40] for v in c.APPENDIX_SOURCED_VALUES],
)
def test_every_appendix_sourced_value_quotes_the_pinned_docx_text_runs(
    sourced: Any,
) -> None:
    """``audit_sourced_value`` reads text, so a .docx needs its own reader.

    The quote is checked against the text runs of ``word/document.xml`` extracted the
    way the provenance ``method`` field describes: a subscript run written ``_{...}``, a
    Symbol-font glyph named, and an OMML equation contributing no run.
    """
    import os
    import xml.etree.ElementTree as ElementTree
    import zipfile

    docx = c.library_dir(os.environ["DATA_ROOT"]) / c.APPENDIX_REL
    assert c._sha256(docx) == sourced.provenance.sha256 == c.APPENDIX_SHA256
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(docx) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
    paragraphs = []
    for paragraph in document.iter(f"{namespace}p"):
        parts = []
        for run in paragraph.iter(f"{namespace}r"):
            properties = run.find(f"{namespace}rPr")
            subscript = False
            if properties is not None:
                alignment = properties.find(f"{namespace}vertAlign")
                subscript = (
                    alignment is not None
                    and alignment.get(f"{namespace}val") == "subscript"
                )
            text = "".join(node.text or "" for node in run.iter(f"{namespace}t"))
            for symbol in run.iter(f"{namespace}sym"):
                text += (
                    f"[SYM char={symbol.get(f'{namespace}char')} "
                    f"font={symbol.get(f'{namespace}font')}]"
                )
            if text:
                parts.append(f"_{{{text}}}" if subscript else text)
        if "".join(parts).strip():
            paragraphs.append("".join(parts))
    assert sourced.quote in "\n".join(paragraphs)


@pytest.mark.data
@pytest.mark.parametrize(
    "sourced",
    c.LEGEND_SOURCED_VALUES,
    ids=[v.quote[:40] for v in c.LEGEND_SOURCED_VALUES],
)
def test_every_legend_sourced_value_quotes_a_cell_of_the_pinned_dataset_ev2(
    sourced: Any,
) -> None:
    import os

    data_root = os.environ["DATA_ROOT"]
    path = c.raw_mirror_dir(data_root) / c.DATA_REL
    assert c._sha256(path) == sourced.provenance.sha256 == c.DATA_SHA256
    # the sheet the value's own page names, not always "Legend scores": the feature
    # units and the non-determined-field statement are on "Legend normalized data"
    sheet = sourced.provenance.page.split("sheet '")[1].rstrip("'")
    legend = pd.read_excel(path, sheet_name=sheet, header=None)
    cells = [
        str(value).strip()
        for row in legend.itertuples(index=False)
        for value in row
        if str(value) != "nan"
    ]
    # a substring of a cell, which is what SourcedValue.quote is defined to be: the
    # legend's header cell packs three sentences into one cell
    assert any(sourced.quote in cell for cell in cells)


@pytest.mark.data
def test_the_real_release_resolves_to_the_measured_row_counts() -> None:
    import os

    data_root = os.environ["DATA_ROOT"]
    path = c.raw_mirror_dir(data_root) / c.DATA_REL
    table = c.read_normalized_table(path)
    assert table.released_rows == 4471
    assert table.footer_rows == 4
    assert len(table.wild_type) == 240
    assert len(table.mutants) == c.IMAGED_STRAINS == 4227
    assert {row.label for row in table.wild_type} == {"WT0511", "WT0813", "WT0815"}
    assert table.wild_type_median_alpha_max == pytest.approx(0.0097366516, abs=1e-10)
    genome = c._bw25113(data_root)
    resolution = c.resolve_rows(table, genome, label="campos2018")
    assert len(resolution.kept) == c.EXPECTED_RECORDS == 3664
    assert resolution.dropped_rows == {
        c.DROP_FOOTER: 4,
        c.DROP_WILD_TYPE: 240,
        c.DROP_ANNOTATED: 5,
        c.DROP_NOT_IN_ANNOTATION: 438,
        c.DROP_MERGED_LOCUS: 0,
        c.DROP_AMBIGUOUS: 0,
        c.DROP_DUPLICATE_LABEL: 120,
    }
    reconciliation = resolution.reconciliation
    assert reconciliation.unique_names == 4158
    assert reconciliation.resolved == 3720
    assert reconciliation.resolved_fraction == pytest.approx(0.8947, abs=5e-5)
    assert reconciliation.layer_histogram == {
        "locus tag": 0,
        "old locus tag": 0,
        "RefSeq locus tag": 0,
        "gene symbol": 3718,
        "gene synonym": 2,
        "not found": 438,
    }
    retired = set(reconciliation.retired_kept)
    jw = {name for name in retired if name.startswith("JW")}
    assert len(jw) == 412
    assert len(retired - jw) == 26


@pytest.mark.data
def test_none_of_the_released_jw_labels_is_a_synonym_of_the_pinned_annotation() -> None:
    """The measured reason the 412 JW rows are dropped: a real absence, not a miss."""
    import os

    data_root = os.environ["DATA_ROOT"]
    genome = c._bw25113(data_root)
    annotation_jw = {
        synonym
        for locus in genome.genbank.loci.values()
        for synonym in locus.synonyms
        if synonym.startswith("JW")
    }
    assert len(annotation_jw) == 4334
    table = c.read_normalized_table(c.raw_mirror_dir(data_root) / c.DATA_REL)
    released_jw = {row.label for row in table.mutants if row.label.startswith("JW")}
    assert len(released_jw) == 412
    assert released_jw & annotation_jw == set()


@pytest.mark.data
def test_the_real_derived_fitness_is_an_exact_affine_map_of_the_released_score() -> (
    None
):
    import os

    data_root = os.environ["DATA_ROOT"]
    path = c.raw_mirror_dir(data_root) / c.DATA_REL
    table = c.read_normalized_table(path)
    genome = c._bw25113(data_root)
    resolution = c.resolve_rows(table, genome, label="campos2018")
    scores = c.released_alpha_max_scores(path)
    denominator = table.wild_type_median_alpha_max
    fitness = np.array(
        [record.row.alpha_max / denominator for record in resolution.kept]
    )
    released = np.array(
        [scores[(record.row.plate, record.row.well)] for record in resolution.kept]
    )
    slope, intercept = np.polyfit(fitness, released, 1)
    assert slope == pytest.approx(23.777071, abs=1e-5)
    assert intercept == pytest.approx(-slope, abs=1e-4)
    assert float(np.abs(released - (slope * fitness + intercept)).max()) < 1e-9
    assert float(fitness.min()) > 0.0
    assert float(fitness.min()) == pytest.approx(0.121789, abs=1e-6)
    assert float(fitness.max()) == pytest.approx(1.351417, abs=1e-6)


# --------------------------------------------------------------------------- #
# Morphology: the reader, the phenotypes, the build, the gate
# --------------------------------------------------------------------------- #
def _no_dapi_position() -> tuple[str, str]:
    """The Keio position of the synthetic row that has no DAPI channel."""
    (row,) = [r for r in _SYNTHETIC_ROWS if r[0] == _SYNTHETIC_NO_DAPI_LABEL]
    assert row[1] is not None and row[2] is not None
    return row[1], row[2]


def _morphology_rows(tmp_path: Path) -> dict[tuple[str, str], c.MorphologyRow]:
    return c.read_morphology_rows(
        _release(tmp_path / "ev2.xlsx"), _cell_counts(tmp_path / "ev1.xlsx")
    )


def test_the_morphology_reader_joins_the_two_sheets_on_the_keio_position(
    tmp_path: Path,
) -> None:
    rows = _morphology_rows(tmp_path)
    positions = [
        (str(row[1]), str(row[2])) for row in _SYNTHETIC_ROWS if row[0] is not None
    ]
    assert len(rows) == len(positions)
    assert set(rows) == set(positions)
    # the cell count rides across from Dataset EV1, in that sheet's own row order
    assert [rows[position].n_cells for position in positions] == [
        200 + i for i, _ in enumerate(positions)
    ]


def test_the_morphology_reader_files_each_value_under_its_declared_statistic(
    tmp_path: Path,
) -> None:
    row = _morphology_rows(tmp_path)[("1", "A004")]
    assert set(row.values) == set(MORPHOLOGY_ASSAY.value_symbols)
    assert set(row.coefficients_of_variation) == set(
        MORPHOLOGY_ASSAY.coefficient_of_variation_symbols
    )
    assert len(row.values) == 15
    assert len(row.coefficients_of_variation) == 11
    assert row.values.keys().isdisjoint(row.coefficients_of_variation)


def test_a_non_determined_field_is_absent_rather_than_imputed(tmp_path: Path) -> None:
    """The release writes a non-determined field as NaN, which is not a measurement."""
    rows = _morphology_rows(tmp_path)
    position = _no_dapi_position()
    stripped = rows[position]
    stored = set(stripped.values) | set(stripped.coefficients_of_variation)
    assert stored == set(c.MORPHOLOGY_REQUIRED_FEATURES)
    assert stored.isdisjoint(c.MORPHOLOGY_DAPI_FEATURES)
    assert len(stored) == 19


def test_the_morphology_reader_refuses_a_missing_feature_column(tmp_path: Path) -> None:
    path = _release(tmp_path / "ev2.xlsx")
    frame = pd.read_excel(path, sheet_name=c.NORMALIZED_SHEET)
    frame = frame.drop(columns=[c.MORPHOLOGY_COLUMNS["<L>"]])
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name=c.NORMALIZED_SHEET, index=False)
    with pytest.raises(ValueError, match="missing morphology columns"):
        c.read_morphology_rows(path, _cell_counts(tmp_path / "ev1.xlsx"))


def test_the_morphology_reader_refuses_a_cell_count_sheet_missing_its_column(
    tmp_path: Path,
) -> None:
    counts = _cell_counts(tmp_path / "ev1.xlsx")
    frame = pd.read_excel(counts, sheet_name=c.RAW_SHEET).drop(
        columns=[c.CELL_COUNT_COLUMN]
    )
    with pd.ExcelWriter(counts) as writer:
        frame.to_excel(writer, sheet_name=c.RAW_SHEET, index=False)
    with pytest.raises(ValueError, match=f"missing column '{c.CELL_COUNT_COLUMN}'"):
        c.read_morphology_rows(_release(tmp_path / "ev2.xlsx"), counts)


def test_the_morphology_reader_refuses_a_position_with_no_cell_count(
    tmp_path: Path,
) -> None:
    """A row the join cannot serve stops the build; no record gets a guessed n_samples."""
    counts = _cell_counts(tmp_path / "ev1.xlsx")
    frame = pd.read_excel(counts, sheet_name=c.RAW_SHEET)
    frame.loc[frame[c.WELL_COLUMN] == "A004", c.WELL_COLUMN] = "Z999"
    with pd.ExcelWriter(counts) as writer:
        frame.to_excel(writer, sheet_name=c.RAW_SHEET, index=False)
    with pytest.raises(ValueError, match=r"no cell count for position"):
        c.read_morphology_rows(_release(tmp_path / "ev2.xlsx"), counts)


def test_the_morphology_reader_refuses_a_repeated_keio_position(tmp_path: Path) -> None:
    counts = _cell_counts(tmp_path / "ev1.xlsx")
    frame = pd.read_excel(counts, sheet_name=c.RAW_SHEET)
    frame.loc[frame.index[1], [c.PLATE_COLUMN, c.WELL_COLUMN]] = frame.loc[
        frame.index[0], [c.PLATE_COLUMN, c.WELL_COLUMN]
    ].to_numpy()
    with pd.ExcelWriter(counts) as writer:
        frame.to_excel(writer, sheet_name=c.RAW_SHEET, index=False)
    with pytest.raises(ValueError, match="repeats a .plate, well. position"):
        c.read_morphology_rows(_release(tmp_path / "ev2.xlsx"), counts)


def test_a_morphology_phenotype_names_the_assay_and_counts_cells(
    tmp_path: Path,
) -> None:
    row = _morphology_rows(tmp_path)[("1", "A004")]
    phenotype = c.morphology_phenotype(row)
    assert phenotype.assay == "campos2018"
    assert phenotype.assay_vocabulary is MORPHOLOGY_ASSAY
    assert phenotype.morphology == row.values
    assert phenotype.morphology_coefficient_of_variation == (
        row.coefficients_of_variation
    )
    assert phenotype.n_samples == row.n_cells
    assert phenotype.sample_unit == SampleUnit.cell
    assert phenotype.provenance_gaps == []


def test_the_morphology_reference_is_the_median_over_the_parental_wells(
    tmp_path: Path,
) -> None:
    rows = _morphology_rows(tmp_path)
    parental = [rows[("1", well)] for well in ("A001", "A002", "A003")]
    reference = c.morphology_reference_phenotype(parental)
    assert reference.n_samples == 3
    assert reference.sample_unit == SampleUnit.biological_replicate
    # the three synthetic wells are rows 0, 1 and 2, so the median is row 1's value
    assert reference.morphology["<L>"] == pytest.approx(parental[1].values["<L>"])
    assert set(reference.morphology) | set(
        reference.morphology_coefficient_of_variation or {}
    ) == set(c.MORPHOLOGY_FEATURES)


def test_an_even_number_of_parental_wells_gives_the_midpoint(tmp_path: Path) -> None:
    rows = _morphology_rows(tmp_path)
    parental = [rows[("1", well)] for well in ("A001", "A002")]
    reference = c.morphology_reference_phenotype(parental)
    assert reference.n_samples == 2
    assert reference.morphology["<L>"] == pytest.approx(
        (parental[0].values["<L>"] + parental[1].values["<L>"]) / 2
    )


def test_a_morphology_reference_needs_at_least_one_parental_well() -> None:
    with pytest.raises(ValueError, match="no morphology reference"):
        c.morphology_reference_phenotype([])


def test_a_reference_feature_no_parental_well_determined_is_absent(
    tmp_path: Path,
) -> None:
    """The median is per feature over the wells that determined it, never imputed."""
    rows = _morphology_rows(tmp_path)
    position = _no_dapi_position()
    reference = c.morphology_reference_phenotype([rows[position]])
    stored = set(reference.morphology) | set(
        reference.morphology_coefficient_of_variation or {}
    )
    assert stored == set(c.MORPHOLOGY_REQUIRED_FEATURES)


def test_the_morphology_loader_builds_the_synthetic_release_end_to_end(
    tmp_path: Path, mirrored: Path
) -> None:
    root = tmp_path / "morphology"
    dataset = c.MorphologyCampos2018Dataset(root=str(root))
    assert len(dataset) == _SYNTHETIC_KEPT
    assert sorted(dataset.gene_set) == [
        "BW25113_0002",
        "BW25113_0004",
        "BW25113_0008",
        "BW25113_4412",
    ]
    references = dataset.experiment_reference_index
    assert references is not None
    assert len(references) == 1
    experiment = dataset[0]["experiment"]
    phenotype = experiment["phenotype"]
    assert experiment["experiment_type"] == "bacterial_morphology"
    assert phenotype["assay"] == "campos2018"
    assert phenotype["label_name"] == "morphology"
    assert phenotype["label_statistic_name"] == "morphology_coefficient_of_variation"
    assert phenotype["sample_unit"] == "cell"
    assert phenotype["n_samples"] >= 200
    stored = set(phenotype["morphology"]) | set(
        phenotype["morphology_coefficient_of_variation"]
    )
    assert stored <= set(c.MORPHOLOGY_FEATURES)
    assert set(c.MORPHOLOGY_REQUIRED_FEATURES) <= stored
    (perturbation,) = experiment["genotype"]["perturbations"]
    assert perturbation["gene_namespace"] == c.BW25113_NAMESPACE
    assert perturbation["collection"] == c.KEIO_COLLECTION


def test_the_morphology_build_writes_the_per_feature_coverage_ledger(
    tmp_path: Path, mirrored: Path
) -> None:
    root = tmp_path / "morphology"
    c.MorphologyCampos2018Dataset(root=str(root))
    features = json.loads((root / "preprocess" / "served_features.json").read_text())
    assert features["assay"] == "campos2018"
    assert features["served"] == list(c.MORPHOLOGY_FEATURES)
    assert list(features["unserved"]) == ["ODmax"]
    assert set(features["released_columns_not_in_the_vocabulary"]) == {
        "%non-div",
        "%1N",
    }
    by_feature = features["determined_values_by_feature"]
    assert set(by_feature) == set(c.MORPHOLOGY_FEATURES)
    # 4 kept rows, 1 of them without the DAPI channel
    for symbol in c.MORPHOLOGY_REQUIRED_FEATURES:
        assert by_feature[symbol] == _SYNTHETIC_KEPT
    for symbol in c.MORPHOLOGY_DAPI_FEATURES:
        assert by_feature[symbol] == _SYNTHETIC_KEPT - 1
    assert features["determined_values"] == sum(by_feature.values())
    assert features["determined_values"] == 19 * 4 + 7 * 3
    assert set(features["feature_units"]["<L>"]) == set("µm")
    assert features["feature_units"]["CV_L"] is None
    assert set(features["features_by_statistic"]) == {
        f.statistic.value for f in MORPHOLOGY_ASSAY.features
    }


def test_the_morphology_build_refuses_a_drop_ledger_that_does_not_add_up(
    tmp_path: Path, mirrored: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = c.resolve_rows

    def _lying(table: Any, genome: Any, *, label: str) -> Any:
        resolution = real(table, genome, label=label)
        resolution.dropped_rows[c.DROP_FOOTER] += 1
        return resolution

    monkeypatch.setattr(c, "resolve_rows", _lying)
    with pytest.raises(RuntimeError, match="drop accounting mismatch"):
        c.MorphologyCampos2018Dataset(root=str(tmp_path / "morphology"))


def test_the_morphology_download_refuses_a_missing_mirror_file(
    tmp_path: Path, mirrored: Path
) -> None:
    (c.raw_mirror_dir(str(mirrored)) / c.CELL_COUNTS_REL).unlink()
    dataset = c.MorphologyCampos2018Dataset.__new__(c.MorphologyCampos2018Dataset)
    with pytest.raises(RuntimeError, match="required raw artifact missing"):
        dataset.download()


def test_the_morphology_verifier_passes_on_the_synthetic_build(
    tmp_path: Path, mirrored: Path, bw25113: EcoliK12BW25113Genome
) -> None:
    root = tmp_path / "morphology"
    c.MorphologyCampos2018Dataset(root=str(root))
    report = c.verify_morphology_build(
        str(root), genome=bw25113, expected_count=_SYNTHETIC_KEPT
    )
    assert report.passed, report.summary()
    rows = {result.name: result for result in report.results}
    assert rows["count"].details["observed"] == _SYNTHETIC_KEPT
    assert rows["assay_coverage"].details["assay"] == "campos2018"
    assert rows["assay_coverage"].details["coverage_strata"] == {"19": 1, "26": 3}
    assert rows["value_fidelity"].details["n_values"] == 19 * 4 + 7 * 3
    assert rows["value_fidelity"].details["n_features"] == 26
    # 11 CVs on the three full rows and 10 on the row without the DAPI channel, whose
    # missing CV_NA is one of the seven nucleoid-derived features
    assert rows["cv_nonnegative"].details["n_values"] == 11 * 3 + 10
    assert rows["reference_populated"].passed
    assert (root / "preprocess" / "verification_report.json").exists()


def test_the_two_campos_datasets_keep_exactly_the_same_rows(
    tmp_path: Path, mirrored: Path
) -> None:
    """One release, one set of row rules: a strain is writable for both or for neither."""
    fitness = c.GrowthRateCampos2018Dataset(root=str(tmp_path / "fitness"))
    morphology = c.MorphologyCampos2018Dataset(root=str(tmp_path / "morphology"))
    assert set(fitness.gene_set) == set(morphology.gene_set)
    assert len(fitness) == len(morphology)


@pytest.mark.data
def test_the_real_release_leaves_the_same_seven_features_non_determined() -> None:
    """The DAPI split is measured, not assumed from what a feature is derived from."""
    import os

    data_root = os.environ["DATA_ROOT"]
    mirror = c.raw_mirror_dir(data_root)
    rows = c.read_morphology_rows(mirror / c.DATA_REL, mirror / c.CELL_COUNTS_REL)
    assert len(rows) == 4467
    stored = [set(r.values) | set(r.coefficients_of_variation) for r in rows.values()]
    assert all(set(c.MORPHOLOGY_REQUIRED_FEATURES) <= s for s in stored)
    short = [s for s in stored if len(s) != 26]
    assert len(short) == c.STRAINS_WITHOUT_A_NUCLEOID_CHANNEL == 278
    assert {frozenset(s) for s in short} == {frozenset(c.MORPHOLOGY_REQUIRED_FEATURES)}
    counts = [r.n_cells for r in rows.values()]
    assert sum(counts) == 1301055
    assert min(counts) == 43


@pytest.mark.data
def test_the_two_released_proportions_are_exact_inverses_of_the_stored_timings() -> (
    None
):
    """The measurement that justifies leaving %non-div and %1N out of the vocabulary."""
    import math
    import os

    data_root = os.environ["DATA_ROOT"]
    frame = pd.read_excel(
        c.raw_mirror_dir(data_root) / c.DATA_REL, sheet_name=c.NORMALIZED_SHEET
    )
    frame = frame[frame[c.LABEL_COLUMN].notna()]
    for proportion, timing in (
        ("%non-div", "Rel.timing div"),
        ("%1N", "Rel.timing nuc"),
    ):
        pairs = frame[[proportion, timing]].dropna()
        assert len(pairs) == 4189
        worst = max(
            abs(t - (-math.log2(1.0 - f / 2.0)))
            for f, t in zip(pairs[proportion], pairs[timing], strict=True)
        )
        assert worst < 2e-15
