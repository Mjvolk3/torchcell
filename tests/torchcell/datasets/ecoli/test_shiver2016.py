# tests/torchcell/datasets/ecoli/test_shiver2016.py
# [[tests.torchcell.datasets.ecoli.test_shiver2016]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_shiver2016.py
"""Shiver 2016 loader: the condition table, unit conversion, columns, records.

The synthetic tests run everywhere. The identifier tests read the REAL
``EcoliK12BW25113Genome`` over the synthetic BW25113 assembly of
``tests/torchcell/sequence/genome/_bacterial_fixtures.py`` through a stubbed ``resolve``
with the network refused. Derived expectations for that assembly: ``thrL`` and its
synonym ``ECK0001`` both reach ``BW25113_0001`` (a merged-locus collision with no direct
member), ``ECK0005`` sits on two loci (ambiguous), ``yaaP`` is a pseudogene locus, and
``thrL-SPA`` and ``nosuchgene`` resolve to nothing.

The data tests (``--data``) audit every module-level ``SourcedValue`` against the
sha256-pinned paper OCR, read the real S1 Dataset from the raw mirror, pin the measured
column histograms, and re-run the S1 Table cross-check of the loader's docstring.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import os.path as osp
import re
import zipfile
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

import torchcell.datasets.ecoli.shiver2016 as s
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
from torchcell.datamodels.media import LB_LENNOX, M9, MEDIA_LIBRARY
from torchcell.datamodels.schema import (
    ASSEMBLY_SET_ACCESSIONS,
    BACTERIAL_ASSEMBLY_SETS,
    AssayType,
    AssemblyReferenceGenome,
    BacterialDeletionPerturbation,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    Concentration,
    ConcentrationUnit,
    DoseBasis,
    EnvironmentPhysicalPerturbation,
    FitnessPhenotype,
    MeasurementType,
    MediaComponentRole,
    PhysicalFactor,
    SmallMoleculePerturbation,
)
from torchcell.datasets.bacteria_common import (
    BacterialGenomeInjector,
    LocusTagResolutionError,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.sequence.genome.base import GeneNameStatus
from torchcell.sequence.genome.ecoli.k12 import (
    BW25113_ASSEMBLY,
    EcoliK12BW25113Genome,
    EcoliK12MG1655Genome,
    EcoliK12StrainName,
)
from torchcell.verification.environment_response import _condition_signature
from torchcell.verification.sourced import (
    ProvenanceGapReason,
    SourcedValue,
    audit_sourced_value,
)

# --------------------------------------------------------------------------- #
# The condition table
# --------------------------------------------------------------------------- #
M9_PREFIX = "M9min "


def test_the_table_holds_the_fifty_seven_conditions_of_the_two_study_batches() -> None:
    assert len(s.CONDITIONS) == 57
    assert len(set(s.CONDITION_LABELS)) == 57
    batches = [spec.batch for spec in s.CONDITIONS]
    assert (batches.count(1), batches.count(4)) == (30, 27)
    assert set(batches) == set(s.STUDY_BATCHES) == {1, 4}


def test_the_m9_prefix_is_exactly_what_selects_the_m9_plate() -> None:
    for spec in s.CONDITIONS:
        on_m9 = spec.label.startswith(M9_PREFIX)
        assert (spec.base == "m9_minimal_agar") is on_m9, spec.label
        assert (spec.carbon_source is not None) is on_m9, spec.label


def test_every_m9_condition_carries_the_recipe_glucose_unless_it_names_a_carbon_source() -> (
    None
):
    carbon = {
        spec.label: (spec.carbon_source.label, spec.carbon_source.value)
        for spec in s.CONDITIONS
        if spec.carbon_source is not None
    }
    assert carbon["M9min acetate [0.6% (w/v)] {4}"] == ("acetate", 0.6)
    others = {label: dose for label, dose in carbon.items() if "acetate" not in label}
    assert set(others.values()) == {("glucose", 0.2)}
    assert len(others) == 10


def test_only_the_cold_conditions_leave_thirty_seven_degrees() -> None:
    temperatures = {
        spec.label: spec.temperature_c
        for spec in s.CONDITIONS
        if spec.temperature_c != s.TEMPERATURE_C.value
    }
    assert temperatures == {
        "10C [-] {4}": 10.0,
        "25C [-] {4}": 25.0,
        "UV+10C [12 sec] {4}": 10.0,
        "4C survival [5 wk] {4}": 4.0,
    }


def test_only_the_four_degree_survival_condition_releases_its_exposure() -> None:
    durations = {
        spec.label: spec.duration_hours
        for spec in s.CONDITIONS
        if spec.duration_hours is not None
    }
    assert durations == {"4C survival [5 wk] {4}": 840.0}
    assert s.SURVIVAL_4C.value == (4.0, 840.0)


def test_only_the_three_labeled_uv_conditions_are_irradiated() -> None:
    assert [spec.label for spec in s.CONDITIONS if spec.irradiated] == [
        "M9min glucose+UV [0.2% (w/v); 12 sec] {4}",
        "UV+10C [12 sec] {4}",
        "UV [12 sec] {4}",
    ]


def test_the_same_compound_at_the_same_dose_appears_in_two_distinct_screens() -> None:
    doses = {
        spec.label: tuple((d.label, d.value, d.unit) for d in spec.small_molecules)
        for spec in s.CONDITIONS
    }
    assert doses["gliotoxin-A [10 ug/mL] {1}"] == doses["gliotoxin-B [10 ug/mL] {1}"]
    assert doses["EDTA [1 mM] {1}"] == doses["EDTA [1 mM] {4}"]
    assert doses["SDS [1% (w/v)] {1}"] == doses["SDS [1% (w/v)] {4}"]
    assert doses["ampicillin [4 ug/mL] {1}"] == doses["ampicillin [4 ug/mL] {4}"]


def test_the_only_two_compound_condition_is_sds_plus_edta() -> None:
    assert [spec.label for spec in s.CONDITIONS if len(spec.small_molecules) > 1] == [
        "SDS+EDTA [0.5% (w/v); 500 uM] {1}"
    ]


def test_a_condition_spec_refuses_a_label_that_contradicts_its_batch() -> None:
    with pytest.raises(ValueError, match="does not end with batch 4"):
        s.ConditionSpec(label="urea [320 mM] {1}", batch=4, base="lb_lennox_agar")


def test_a_condition_spec_refuses_a_batch_outside_this_study() -> None:
    with pytest.raises(ValueError, match="batch 0 is not this study's"):
        s.ConditionSpec(label="SDS [1% (w/v)] {0}", batch=0, base="lb_lennox_agar")


def test_a_condition_spec_pairs_the_m9_plate_with_a_carbon_source_both_ways() -> None:
    with pytest.raises(ValueError, match="names its carbon source"):
        s.ConditionSpec(label="x [1 mM] {1}", batch=1, base="m9_minimal_agar")
    with pytest.raises(ValueError, match="names its carbon source"):
        s.ConditionSpec(
            label="x [1 mM] {1}",
            batch=1,
            base="lb_lennox_agar",
            carbon_source=s.M9_GLUCOSE_DOSE,
        )


# --------------------------------------------------------------------------- #
# Dose units
# --------------------------------------------------------------------------- #
def test_the_mass_per_volume_units_convert_onto_the_one_typed_microgram_per_ml() -> (
    None
):
    assert s.Dose(label="x", value=250.0, unit="ng/mL").concentration == (
        Concentration(
            value=0.25, unit=ConcentrationUnit.ug_per_ml, basis=DoseBasis.fixed
        )
    )
    assert s.Dose(label="x", value=1.0, unit="mg/mL").concentration.value == 1000.0
    assert s.Dose(label="x", value=36.0, unit="ug/mL").concentration.value == 36.0
    for unit in ("ug/mL", "ng/mL", "mg/mL"):
        dose = s.Dose(label="x", value=1.0, unit=cast(Any, unit))
        assert dose.concentration.unit is ConcentrationUnit.ug_per_ml


def test_the_other_units_pass_through_as_released() -> None:
    cases = {
        "mM": ConcentrationUnit.millimolar,
        "uM": ConcentrationUnit.micromolar,
        "% (w/v)": ConcentrationUnit.percent_w_v,
        "% (v/v)": ConcentrationUnit.percent_v_v,
    }
    for unit, expected in cases.items():
        dose = s.Dose(label="x", value=2.5, unit=cast(Any, unit))
        assert (dose.concentration.value, dose.concentration.unit) == (2.5, expected)


def test_every_dose_of_the_table_uses_a_convertible_unit() -> None:
    used = {
        dose.unit
        for spec in s.CONDITIONS
        for dose in (
            *spec.small_molecules,
            *((spec.carbon_source,) if spec.carbon_source is not None else ()),
        )
    }
    assert used <= set(s.UNIT_CONVERSIONS)
    assert used == {"ug/mL", "ng/mL", "mg/mL", "mM", "uM", "% (w/v)", "% (v/v)"}


def test_no_milligram_per_liter_unit_is_needed_because_ug_per_ml_is_the_same_number() -> (
    None
):
    assert "mg_per_l" not in ConcentrationUnit.__members__
    assert "mg/L" not in {unit.value for unit in ConcentrationUnit}
    assert s.UNIT_CONVERSIONS["ug/mL"] == (ConcentrationUnit.ug_per_ml, 1.0)


def test_every_dose_is_a_fixed_dose_never_a_target_basis() -> None:
    bases = {
        dose.concentration.basis
        for spec in s.CONDITIONS
        for dose in spec.small_molecules
    }
    assert bases == {DoseBasis.fixed}


# --------------------------------------------------------------------------- #
# Media
# --------------------------------------------------------------------------- #
def _component(medium: Any, name: str) -> Any:
    (found,) = [c for c in medium.components if c.compound.name == name]
    return found


def test_the_default_plate_is_solid_lennox_with_the_stated_ninety_millimolar_salt() -> (
    None
):
    medium = s.SHIVER2016_LB_LENNOX_AGAR
    assert medium.state == "solid"
    assert medium.is_synthetic is False
    assert medium.base_medium == "LB_LENNOX"
    assert MEDIA_LIBRARY[medium.base_medium] is LB_LENNOX
    for name in ("tryptone", "yeast extract"):
        assert _component(medium, name) == _component(LB_LENNOX, name)
    salt = _component(medium, "sodium chloride")
    assert salt.concentration.value == 90.0
    assert salt.concentration.unit is ConcentrationUnit.millimolar
    assert _component(LB_LENNOX, "sodium chloride").concentration.unit is (
        ConcentrationUnit.g_per_l
    )
    agar = _component(medium, "agar")
    assert agar.role is MediaComponentRole.gelling_agent
    assert (agar.concentration.value, agar.concentration.unit) == (
        2.0,
        ConcentrationUnit.percent_w_v,
    )


def test_the_m9_plate_is_the_shared_salts_plus_agar_with_no_carbon_component() -> None:
    medium = s.SHIVER2016_M9_MINIMAL_AGAR
    assert medium.state == "solid"
    assert medium.is_synthetic is True
    assert medium.base_medium == "M9"
    assert MEDIA_LIBRARY[medium.base_medium] is M9
    assert medium.components[:-1] == M9.components
    assert medium.components[-1].compound.name == "agar"
    assert [
        c.compound.name
        for c in medium.components
        if c.role is MediaComponentRole.carbon_source
    ] == []


def test_both_screen_media_name_a_media_library_base() -> None:
    for medium in s.SCREEN_MEDIA.values():
        assert medium.base_medium in MEDIA_LIBRARY


# --------------------------------------------------------------------------- #
# Environments
# --------------------------------------------------------------------------- #
def test_every_condition_is_one_distinct_environment_signature() -> None:
    signatures = {
        spec.label: _condition_signature(
            {
                "environment": s.environment(spec).model_dump(),
                "phenotype": s.phenotype(1.0, spec).model_dump(),
            }
        )
        for spec in s.CONDITIONS
    }
    assert len(set(signatures.values())) == 57


def test_dropping_the_screen_id_would_collapse_the_repeated_conditions() -> None:
    """The verbatim label is load-bearing: five condition groups are otherwise identical."""
    keyed = [
        (
            _condition_signature(
                {
                    "environment": s.environment(spec).model_dump(),
                    "phenotype": {"screen_id": None},
                }
            ),
            spec.label,
        )
        for spec in s.CONDITIONS
    ]
    assert len({key for key, _ in keyed}) == 51
    grouped: dict[Any, list[str]] = {}
    for key, label in keyed:
        grouped.setdefault(key, []).append(label)
    collapsed = sorted(tuple(labels) for labels in grouped.values() if len(labels) > 1)
    assert collapsed == [
        ("EDTA [1 mM] {1}", "EDTA [1 mM] {4}"),
        (
            "M9min glucose-A [0.2% (w/v)] {1}",
            "M9min glucose-B [0.2% (w/v)] {1}",
            "M9min glucose [0.2% (w/v)] {4}",
        ),
        ("SDS [1% (w/v)] {1}", "SDS [1% (w/v)] {4}"),
        ("ampicillin [4 ug/mL] {1}", "ampicillin [4 ug/mL] {4}"),
        ("gliotoxin-A [10 ug/mL] {1}", "gliotoxin-B [10 ug/mL] {1}"),
    ]


def test_a_dosed_chemical_is_a_small_molecule_at_its_converted_dose() -> None:
    (spec,) = [c for c in s.CONDITIONS if c.label.startswith("tetracycline")]
    environment = s.environment(spec)
    (perturbation,) = environment.perturbations
    assert isinstance(perturbation, SmallMoleculePerturbation)
    assert perturbation.compound.name == "tetracycline"
    assert perturbation.compound.inchikey is not None
    assert perturbation.concentration.value == 0.5
    assert perturbation.concentration.unit is ConcentrationUnit.ug_per_ml


def test_a_compound_with_no_curated_row_carries_a_typed_gap_and_no_invented_key() -> (
    None
):
    (spec,) = [c for c in s.CONDITIONS if c.label.startswith("kasugamycin")]
    (perturbation,) = s.environment(spec).perturbations
    assert isinstance(perturbation, SmallMoleculePerturbation)
    compound = perturbation.compound
    assert compound.name == "kasugamycin"
    assert compound.inchikey is None
    assert [g.field for g in compound.provenance_gaps] == ["inchikey"]
    assert compound.provenance_gaps[0].reason is (
        ProvenanceGapReason.deferred_pending_source_review
    )


def test_the_m9_carbon_source_is_a_physical_factor_at_the_recipe_amount() -> None:
    (spec,) = [c for c in s.CONDITIONS if c.label.startswith("M9min acetate")]
    perturbations = s.environment(spec).perturbations
    assert len(perturbations) == 1
    factor = perturbations[0]
    assert isinstance(factor, EnvironmentPhysicalPerturbation)
    assert factor.factor is PhysicalFactor.carbon_source
    assert factor.agent is not None and factor.agent.name == "acetic acid"
    assert factor.magnitude is not None
    assert (factor.magnitude.value, factor.magnitude.unit) == (
        0.6,
        ConcentrationUnit.percent_w_v,
    )


def test_uv_is_a_radiation_factor_whose_dose_is_a_typed_absence() -> None:
    (spec,) = [c for c in s.CONDITIONS if c.label == "UV [12 sec] {4}"]
    environment = s.environment(spec)
    (factor,) = environment.perturbations
    assert isinstance(factor, EnvironmentPhysicalPerturbation)
    assert factor.factor is PhysicalFactor.radiation
    assert factor.magnitude is None
    assert [g.field for g in factor.provenance_gaps] == ["magnitude"]
    assert factor.provenance_gaps[0].reason is (
        ProvenanceGapReason.not_reported_by_primary
    )
    assert "exposure TIME" in str(factor.provenance_gaps[0].note)


def test_a_temperature_only_condition_carries_no_perturbation() -> None:
    for label, temperature in (("10C [-] {4}", 10.0), ("25C [-] {4}", 25.0)):
        (spec,) = [c for c in s.CONDITIONS if c.label == label]
        environment = s.environment(spec)
        assert environment.perturbations == []
        assert environment.temperature is not None
        assert environment.temperature.value == temperature


def test_the_duration_is_a_typed_gap_everywhere_but_the_four_degree_condition() -> None:
    gapped = []
    for spec in s.CONDITIONS:
        environment = s.environment(spec)
        fields = [g.field for g in environment.provenance_gaps]
        if environment.duration_hours is None:
            assert fields == ["duration_hours"], spec.label
            gapped.append(spec.label)
        else:
            assert fields == []
            assert (spec.label, environment.duration_hours) == (
                "4C survival [5 wk] {4}",
                840.0,
            )
    assert len(gapped) == 56


def test_every_environment_is_an_aerobic_plate() -> None:
    assert {s.environment(spec).aerobicity for spec in s.CONDITIONS} == {"aerobic"}


# --------------------------------------------------------------------------- #
# Phenotype and genotype
# --------------------------------------------------------------------------- #
def test_a_negative_fitness_score_is_stored_verbatim_where_a_fitness_would_clamp() -> (
    None
):
    spec = s.CONDITIONS[0]
    assert s.phenotype(-30.935, spec).environment_response == -30.935
    assert FitnessPhenotype(fitness=-30.935).fitness == 0.0


def test_the_phenotype_types_the_score_and_gaps_what_the_release_omits() -> None:
    spec = s.CONDITIONS[0]
    phenotype = s.phenotype(2.5, spec)
    assert phenotype.measurement_type is MeasurementType.z_score
    assert phenotype.assay_type is AssayType.colony_size_array
    assert phenotype.screen_id == spec.label
    assert phenotype.units == s.UNITS
    assert phenotype.environment_response_uncertainty is None
    assert phenotype.environment_response_uncertainty_type is None
    assert phenotype.n_samples is None
    assert phenotype.sample_unit is None
    assert [g.field for g in phenotype.provenance_gaps] == [
        "n_samples",
        "sample_unit",
        "environment_response_uncertainty",
        "environment_response_se",
    ]


def test_the_replicate_gaps_defer_to_the_cited_method_paper() -> None:
    gaps = {g.field: g for g in s.phenotype(1.0, s.CONDITIONS[0]).provenance_gaps}
    for field in ("n_samples", "sample_unit"):
        gap = gaps[field]
        assert gap.reason is ProvenanceGapReason.deferred_pending_source_review
        assert gap.resolve_with is not None
        assert gap.resolve_with.citation_key == (
            "nicholsPhenotypicLandscapeBacterial2011"
        )
    for field in ("environment_response_uncertainty", "environment_response_se"):
        assert gaps[field].reason is ProvenanceGapReason.not_reported_by_primary


def test_the_reference_is_the_unaffected_strain_at_a_score_of_zero() -> None:
    spec = s.CONDITIONS[0]
    reference = s.reference_phenotype(spec)
    assert reference.environment_response == 0.0
    assert reference.measurement_type is MeasurementType.z_score
    assert reference.screen_id == spec.label
    assert reference.units == s.UNITS_REFERENCE


def test_the_genotype_names_the_collection_and_the_symbol_route_and_no_cassette() -> (
    None
):
    genotype = s.genotype("thrA", "BW25113_0002")
    (perturbation,) = genotype.perturbations
    assert isinstance(perturbation, BacterialDeletionPerturbation)
    assert perturbation.systematic_gene_name == "BW25113_0002"
    assert perturbation.perturbed_gene_name == "thrA"
    assert perturbation.gene_namespace == "ecoli_k12_bw25113_locus_tag"
    assert perturbation.collection == s.KEIO_COLLECTION == "KEIO deletion library"
    assert perturbation.cassette is None
    assert perturbation.identifier_mapping is not None
    assert perturbation.identifier_mapping.source_identifier == "thrA"
    assert perturbation.identifier_mapping.route == "gene_symbol"


def test_the_genotype_refuses_a_tag_of_another_strains_namespace() -> None:
    with pytest.raises(ValueError):
        s.genotype("thrA", "b0002")


# --------------------------------------------------------------------------- #
# Reading S1 Dataset
# --------------------------------------------------------------------------- #
_SYNTHETIC_GENES: tuple[str, ...] = (
    "thrA",
    "hokC",
    "yaaP",
    "yaaX",
    "proB",
    "proB",
    "thrL",
    "ECK0001",
    "ECK0005",
    "thrL-SPA",
    "nosuchgene",
)


def _release(
    path: Path,
    genes: tuple[str, ...] = _SYNTHETIC_GENES,
    *,
    labels: tuple[str, ...] | None = None,
    extra_rows: tuple[tuple[str, ...], ...] = (),
    blanks: frozenset[tuple[str, int]] = frozenset(),
) -> Path:
    """Write a synthetic S1 Dataset: the real condition labels over ``genes``."""
    rows: list[list[str]] = [[s.CONDITION_COLUMN, *genes]]
    for row_index, label in enumerate(labels or s.CONDITION_LABELS):
        cells = [
            "" if (label, column) in blanks else f"{row_index + column / 100:.3f}"
            for column in range(len(genes))
        ]
        rows.append([label, *cells])
    rows.extend(list(row) for row in extra_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        csv.writer(handle, delimiter="\t").writerows(rows)
    return path


def test_read_keeps_this_studys_batches_and_skips_the_nichols_rows(
    tmp_path: Path,
) -> None:
    nichols = ("SDS [1% (w/v)] {0}", *("0.0" for _ in _SYNTHETIC_GENES))
    path = _release(tmp_path / "s1.txt", extra_rows=(nichols,))
    matrix = s.read_fitness_matrix(path)
    assert matrix.gene_labels == _SYNTHETIC_GENES
    assert set(matrix.rows) == set(s.CONDITION_LABELS)
    assert matrix.n_condition_rows == 58
    assert len(matrix.rows["A22 [5 ug/mL] {1}"]) == len(_SYNTHETIC_GENES)


def test_read_refuses_a_renamed_first_column(tmp_path: Path) -> None:
    path = _release(tmp_path / "s1.txt")
    text = path.read_text().replace(s.CONDITION_COLUMN, "Stress", 1)
    path.write_text(text)
    with pytest.raises(ValueError, match="expected the first column"):
        s.read_fitness_matrix(path)


def test_read_refuses_a_row_that_is_not_the_header_width(tmp_path: Path) -> None:
    short = ("urea [1 mM] {1}", "0.0")
    path = _release(tmp_path / "s1.txt", extra_rows=(short,))
    with pytest.raises(ValueError, match="has 2 fields, expected 12"):
        s.read_fitness_matrix(path)


def test_read_refuses_a_study_batch_condition_the_loader_does_not_type(
    tmp_path: Path,
) -> None:
    stranger = ("brand new stress [1 mM] {1}", *("0.0" for _ in _SYNTHETIC_GENES))
    path = _release(tmp_path / "s1.txt", extra_rows=(stranger,))
    with pytest.raises(ValueError, match="is in batch 1 but is not one of the 57"):
        s.read_fitness_matrix(path)


def test_read_refuses_a_repeated_condition_row(tmp_path: Path) -> None:
    repeat = ("SDS [1% (w/v)] {1}", *("0.0" for _ in _SYNTHETIC_GENES))
    path = _release(tmp_path / "s1.txt", extra_rows=(repeat,))
    with pytest.raises(ValueError, match="appears twice"):
        s.read_fitness_matrix(path)


def test_read_refuses_a_release_missing_a_typed_condition(tmp_path: Path) -> None:
    path = _release(tmp_path / "s1.txt", labels=s.CONDITION_LABELS[:-1])
    with pytest.raises(ValueError, match="typed conditions absent from the release"):
        s.read_fitness_matrix(path)


def test_read_refuses_a_condition_label_with_no_batch(tmp_path: Path) -> None:
    unlabeled = ("SDS [1% (w/v)]", *("0.0" for _ in _SYNTHETIC_GENES))
    path = _release(tmp_path / "s1.txt", extra_rows=(unlabeled,))
    with pytest.raises(ValueError, match="names no batch in curly brackets"):
        s.read_fitness_matrix(path)


# --------------------------------------------------------------------------- #
# Columns against the synthetic BW25113 annotation
# --------------------------------------------------------------------------- #
@pytest.fixture
def bw25113(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12BW25113Genome:
    """The synthetic BW25113 genome; the network refuses."""
    files = write_assembly(tmp_path / "tier", BW25113_ASSEMBLY, BW25113_LOCI)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    return EcoliK12BW25113Genome(genome_root=str(tmp_path / "bw25113"), overwrite=True)


def test_each_column_rule_claims_its_own_labels(
    bw25113: EcoliK12BW25113Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(s, "MIN_RESOLVED_FRACTION", 0.7)
    resolution = s.resolve_columns(_SYNTHETIC_GENES, bw25113, label="synthetic")
    assert [(c.index, c.source_label, c.locus_tag) for c in resolution.kept] == [
        (0, "thrA", "BW25113_0002"),
        (1, "hokC", "BW25113_4412"),
        (2, "yaaP", "BW25113_0004"),
        (3, "yaaX", "BW25113_0008"),
    ]
    assert resolution.dropped_labels == {
        s.DROP_NOT_A_DELETION: ["thrL-SPA"],
        s.DROP_NOT_IN_ANNOTATION: ["nosuchgene"],
        s.DROP_MERGED_LOCUS: ["ECK0001", "thrL"],
        s.DROP_AMBIGUOUS: ["ECK0005"],
        s.DROP_DUPLICATE_COLUMN: ["proB"],
    }
    assert resolution.dropped_columns == {
        s.DROP_NOT_A_DELETION: 1,
        s.DROP_NOT_IN_ANNOTATION: 1,
        s.DROP_MERGED_LOCUS: 2,
        s.DROP_AMBIGUOUS: 1,
        s.DROP_DUPLICATE_COLUMN: 2,
    }
    assert sum(resolution.dropped_columns.values()) + len(resolution.kept) == len(
        _SYNTHETIC_GENES
    )


def test_a_pseudogene_locus_is_kept_as_a_perturbation_target(
    bw25113: EcoliK12BW25113Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(s, "MIN_RESOLVED_FRACTION", 0.7)
    resolution = s.resolve_columns(_SYNTHETIC_GENES, bw25113, label="synthetic")
    (pseudo,) = [c for c in resolution.kept if c.source_label == "yaaP"]
    status = bw25113.resolve_gene_name(pseudo.locus_tag).status
    assert status is GeneNameStatus.NON_GENE_FEATURE
    assert resolution.reconciliation.status_histogram[status] == 1


def test_a_release_below_the_resolution_threshold_stops_the_build(
    bw25113: EcoliK12BW25113Genome,
) -> None:
    labels = ("thrA", "nosuchgene", "alsoabsent", "stillabsent")
    with pytest.raises(LocusTagResolutionError, match=r"1 of 4 names \(0\.250\)"):
        s.resolve_columns(labels, bw25113, label="threshold")


def test_every_column_rule_has_a_description_in_the_drop_log() -> None:
    rules = dict(s.COLUMN_RULES)
    assert list(rules) == [
        s.DROP_NOT_A_DELETION,
        s.DROP_NOT_IN_ANNOTATION,
        s.DROP_MERGED_LOCUS,
        s.DROP_AMBIGUOUS,
        s.DROP_DUPLICATE_COLUMN,
    ]
    assert all(len(description) > 40 for description in rules.values())


def test_the_non_deletion_suffixes_match_the_releases_own_allele_markers() -> None:
    for label in (
        "fusA-SPA",
        "lolA-DAS",
        "imp-DAS+4",
        "fabZ-kan",
        "lpxc-kan",
        "bamA{del(64)}",
        "fabZ{F101Y}",
        "yfiO*",
    ):
        assert s.NON_DELETION_ALLELE.search(label) is not None, label
    for label in ("thrA", "ygaQ_2", "istR-1", "murE-A", "ECK0503", "rdlABC"):
        assert s.NON_DELETION_ALLELE.search(label) is None, label


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def test_deposit_is_idempotent_and_refuses_a_changed_mirror_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "s1.txt"
    source.write_bytes(b"Condition\tthrA\n")
    monkeypatch.setattr(
        s, "DATA_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    data_root = tmp_path / "root"
    first = s.deposit_raw_mirror(data_path=source, data_root=str(data_root))
    second = s.deposit_raw_mirror(data_path=source, data_root=str(data_root))
    assert first == second == s.raw_mirror_dir(str(data_root))
    manifest = s.load_manifest(str(data_root))
    assert s.manifest_sha256(manifest, s.DATA_REL) == s.DATA_SHA256
    (record,) = manifest.files
    assert record.retrieval is not None
    assert record.retrieval.method == "pmc_cloud"
    assert record.retrieval.params == {"key": s.PMC_CLOUD_KEY}
    (data_root / s.RAW_DIR_REL / s.DATA_REL).write_bytes(b"tampered\n")
    with pytest.raises(RuntimeError, match="different sha256; refusing"):
        s.deposit_raw_mirror(data_path=source, data_root=str(data_root))


def test_deposit_refuses_bytes_that_are_not_the_pinned_release(tmp_path: Path) -> None:
    other = tmp_path / "other.txt"
    other.write_bytes(b"not the release\n")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        s.deposit_raw_mirror(data_path=other, data_root=str(tmp_path / "root"))


def test_manifest_sha256_refuses_a_path_the_manifest_does_not_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "s1.txt"
    source.write_bytes(b"Condition\n")
    monkeypatch.setattr(
        s, "DATA_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    s.deposit_raw_mirror(data_path=source, data_root=str(tmp_path / "root"))
    manifest = s.load_manifest(str(tmp_path / "root"))
    with pytest.raises(KeyError, match="is not in the raw-mirror manifest"):
        s.manifest_sha256(manifest, "data/absent.txt")


def test_the_mirror_names_the_unmirrored_sources_it_deliberately_leaves_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "s1.txt"
    source.write_bytes(b"Condition\n")
    monkeypatch.setattr(
        s, "DATA_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    s.deposit_raw_mirror(data_path=source, data_root=str(tmp_path / "root"))
    expected = " ".join(s.load_manifest(str(tmp_path / "root")).si_expected)
    assert s.S1_TABLE_SHA256 in expected
    assert s.DRYAD_DOI in expected
    assert "NOT mirrored" in expected


# --------------------------------------------------------------------------- #
# Registry and genome injection
# --------------------------------------------------------------------------- #
def test_the_loader_is_registered_and_receives_the_bw25113_genome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        dataset_registry["EnvChemgenShiver2016Dataset"] is s.EnvChemgenShiver2016Dataset
    )
    assert s.EnvChemgenShiver2016Dataset.REFERENCE_STRAIN == "BW25113"
    install_bacterial_fakes(monkeypatch)
    injector = BacterialGenomeInjector(data_root="/nowhere")
    kwargs = injector.genome_kwargs(s.EnvChemgenShiver2016Dataset)
    assert set(kwargs) == {"ecoli_genome"}
    assert isinstance(kwargs["ecoli_genome"], FakeBW25113Genome)


# --------------------------------------------------------------------------- #
# The whole loader, hermetic. Derived expectations for the synthetic frame:
# 4 of 11 columns are kept (thrA, hokC, yaaP, yaaX), 57 conditions, one blank cell,
# so 4 x 57 - 1 = 227 records and 7 x 57 = 399 dropped by the column rules.
# --------------------------------------------------------------------------- #
def _pin(strain: EcoliK12StrainName) -> AssemblyReferenceGenome:
    assembly_set = BACTERIAL_ASSEMBLY_SETS[strain]
    return AssemblyReferenceGenome(
        species="Escherichia coli",
        strain=strain,
        assembly_set=cast(Any, assembly_set),
        assembly_accession=ASSEMBLY_SET_ACCESSIONS[assembly_set][0],
    )


_BLANK = frozenset({("A22 [5 ug/mL] {1}", 0)})


@pytest.fixture
def mirrored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bw25113: EcoliK12BW25113Genome
) -> Path:
    """A tmp ``DATA_ROOT`` whose raw mirror holds a synthetic S1 Dataset; returns it."""
    source = _release(tmp_path / "source" / "s1.txt", blanks=_BLANK)
    monkeypatch.setattr(
        s, "DATA_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    data_root = tmp_path / "data_root"
    s.deposit_raw_mirror(data_path=source, data_root=str(data_root))
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setattr(s, "MIN_RESOLVED_FRACTION", 0.7)
    monkeypatch.setattr(
        s, "bacterial_genome", lambda host, strain, data_root=None: bw25113
    )
    monkeypatch.setattr(s, "assembly_reference", _pin)
    return data_root


def test_the_loader_builds_the_synthetic_release_end_to_end(
    tmp_path: Path, mirrored: Path
) -> None:
    root = tmp_path / "dataset"
    dataset = s.EnvChemgenShiver2016Dataset(root=str(root))
    assert len(dataset) == 227
    assert sorted(dataset.gene_set) == [
        "BW25113_0002",
        "BW25113_0004",
        "BW25113_0008",
        "BW25113_4412",
    ]
    references = dataset.experiment_reference_index
    assert references is not None
    assert len(references) == 57

    first = dataset[0]["experiment"]
    assert first["phenotype"]["measurement_type"] == "z_score"
    assert first["phenotype"]["screen_id"] == "A22 [5 ug/mL] {1}"
    assert first["genotype"]["perturbations"][0]["systematic_gene_name"] == (
        "BW25113_4412"
    )
    assert dataset[0]["reference"]["phenotype_reference"]["environment_response"] == 0.0
    assert dataset[0]["publication"]["doi"] == s.DOI

    drops = json.loads((root / "preprocess" / "dropped_records.json").read_text())
    assert (drops["source_records"], drops["kept_records"]) == (627, 227)
    assert {r["rule"]: (r["n_columns"], r["n_records"]) for r in drops["rules"]} == {
        s.DROP_NOT_A_DELETION: (1, 57),
        s.DROP_NOT_IN_ANNOTATION: (1, 57),
        s.DROP_MERGED_LOCUS: (2, 114),
        s.DROP_AMBIGUOUS: (1, 57),
        s.DROP_DUPLICATE_COLUMN: (2, 114),
        s.DROP_BLANK_CELL: (0, 1),
    }

    report = json.loads(
        (root / "preprocess" / "identifier_reconciliation.json").read_text()
    )
    assert report["released_gene_columns"] == 11
    assert report["released_condition_rows"] == 57
    assert (report["kept_columns"], report["distinct_locus_tags"]) == (4, 4)
    assert report["identifier_route"] == "gene_symbol"
    assert report["reconciliation"]["gene_namespace"] == ("ecoli_k12_bw25113_locus_tag")
    assert (root / "preprocess" / "build_manifest.json").is_file()
    assert (root / "raw" / s.DATA_FILENAME).is_file()


def test_a_direct_run_opens_the_bw25113_genome_itself(
    tmp_path: Path, mirrored: Path
) -> None:
    dataset = s.EnvChemgenShiver2016Dataset.__new__(s.EnvChemgenShiver2016Dataset)
    dataset.ecoli_genome = None
    dataset.name = "EnvChemgenShiver2016Dataset"
    genome = dataset._genome()
    assert isinstance(genome, EcoliK12BW25113Genome)
    assert dataset.ecoli_genome is genome


def test_the_loader_refuses_a_genome_of_the_wrong_strain(tmp_path: Path) -> None:
    dataset = s.EnvChemgenShiver2016Dataset.__new__(s.EnvChemgenShiver2016Dataset)
    dataset.name = "EnvChemgenShiver2016Dataset"
    dataset.ecoli_genome = cast(Any, object.__new__(EcoliK12MG1655Genome))
    with pytest.raises(TypeError, match="needs the BW25113 genome"):
        dataset._genome()


def test_download_refuses_a_mirror_whose_file_is_gone(
    tmp_path: Path, mirrored: Path
) -> None:
    (mirrored / s.RAW_DIR_REL / s.DATA_REL).unlink()
    dataset = s.EnvChemgenShiver2016Dataset.__new__(s.EnvChemgenShiver2016Dataset)
    with pytest.raises(RuntimeError, match="required raw artifact missing"):
        dataset.download()


# --------------------------------------------------------------------------- #
# Data tests: the mirrors and the real release
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    return os.environ["DATA_ROOT"]


@pytest.mark.data
def test_every_sourced_value_is_backed_by_its_verbatim_quote() -> None:
    root = osp.join(_data_root(), "torchcell-library")
    values = [v for v in vars(s).values() if isinstance(v, SourcedValue)]
    assert len(values) == len(s.SOURCED_VALUES) == 19
    for value in values:
        result = audit_sourced_value(value, root)
        assert result.passed, f"{value.value!r}: {result.message}"


@pytest.mark.data
def test_the_release_resolves_to_the_measured_column_counts() -> None:
    from torchcell.datasets.bacteria_common import bacterial_genome

    matrix = s.read_fitness_matrix(osp.join(_data_root(), s.RAW_DIR_REL, s.DATA_REL))
    assert len(matrix.gene_labels) + 1 == s.RELEASED_COLUMNS == 3976
    assert matrix.n_condition_rows == s.RELEASED_CONDITION_ROWS == 292
    assert len(matrix.rows) == 57
    assert s.SCREEN_SIZE.value == (len(matrix.gene_labels), len(matrix.rows))

    genome = bacterial_genome("ecoli", "BW25113")
    assert isinstance(genome, EcoliK12BW25113Genome)
    resolution = s.resolve_columns(matrix.gene_labels, genome, label="release")
    assert len(resolution.kept) == 3720
    assert len({c.locus_tag for c in resolution.kept}) == 3720
    assert resolution.dropped_columns == {
        s.DROP_NOT_A_DELETION: 134,
        s.DROP_NOT_IN_ANNOTATION: 49,
        s.DROP_MERGED_LOCUS: 48,
        s.DROP_AMBIGUOUS: 2,
        s.DROP_DUPLICATE_COLUMN: 22,
    }
    assert resolution.dropped_labels[s.DROP_AMBIGUOUS] == ["rffT", "spr"]
    histogram = {
        status.value: n
        for status, n in resolution.reconciliation.status_histogram.items()
    }
    assert histogram == {
        "current": 0,
        "renamed": 3649,
        "non_gene_feature": 130,
        "retired": 182,
        "ambiguous": 2,
    }
    assert resolution.reconciliation.layer_histogram == {
        "locus tag": 0,
        "old locus tag": 0,
        "RefSeq locus tag": 0,
        "gene symbol": 3635,
        "gene synonym": 146,
        "not found": 182,
    }

    blanks = sum(
        1
        for row in matrix.rows.values()
        for column in resolution.kept
        if row[column.index] == ""
    )
    assert blanks == 8007
    assert len(resolution.kept) * 57 - blanks == s.EXPECTED_RECORDS == 204033


_S1_TABLE_SCORES: dict[str, float] = {
    "typA": -10.5,
    "ihfB": -8.7,
    "ihfA": -8.7,
    "dinJ": -8.3,
    "rbfA": -6.9,
    "deaD": -6.7,
    "hfq": -5.7,
    "crr": -4.8,
    "ycbK": -4.1,
    "nfuA": -4.0,
}


def _s1_table_rows(path: Path) -> list[list[str]]:
    """The visible cells of each S1 Table row, Zotero field codes stripped."""
    document = zipfile.ZipFile(path).read("word/document.xml").decode("utf8")
    document = re.sub(r"<w:instrText[^>]*>.*?</w:instrText>", "", document, flags=re.S)
    document = re.sub(r"</w:tr>", "\n@ROW@\n", document)
    document = re.sub(r"</w:tc>", "\t", document)
    text = html.unescape(re.sub(r"<[^>]+>", "", document))
    rows = []
    for raw in text.split("@ROW@"):
        cells = [cell.strip() for cell in raw.strip().split("\t")]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(cells)
    return rows


@pytest.mark.data
def test_the_ten_degree_row_agrees_with_the_cold_sensitive_table(
    tmp_path: Path,
) -> None:
    """The parse cross-check of the module docstring, re-run against both mirrors."""
    table = Path(_data_root(), "torchcell-library", s.CITATION_KEY, s.S1_TABLE_REL)
    assert hashlib.sha256(table.read_bytes()).hexdigest() == s.S1_TABLE_SHA256
    printed = {
        cells[0]: float(cells[-3])
        for cells in _s1_table_rows(table)
        if cells[0] in _S1_TABLE_SCORES
    }
    assert printed == _S1_TABLE_SCORES

    matrix = s.read_fitness_matrix(osp.join(_data_root(), s.RAW_DIR_REL, s.DATA_REL))
    index = {label: i for i, label in enumerate(matrix.gene_labels)}
    row = matrix.rows["10C [-] {4}"]
    for gene, expected in _S1_TABLE_SCORES.items():
        assert round(float(row[index[gene]]), 1) == expected, gene


# --------------------------------------------------------------------------- #
# Guards and the schema classes
# --------------------------------------------------------------------------- #
def test_two_kept_columns_on_one_locus_tag_stop_the_build(
    bw25113: EcoliK12BW25113Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard behind the drop rules: nothing may silently merge two strains."""
    reconcile = reconcile_locus_tags

    def collide(genome: Any, names: pd.Series, *, label: str) -> tuple[pd.Series, Any]:
        stored, report = reconcile(genome, names, label=label)
        return pd.Series(["BW25113_0002"] * len(stored)), report

    monkeypatch.setattr(s, "MIN_RESOLVED_FRACTION", 0.7)
    monkeypatch.setattr(s, "reconcile_locus_tags", collide)
    with pytest.raises(RuntimeError, match="claimed by more than one kept column"):
        s.resolve_columns(_SYNTHETIC_GENES, bw25113, label="collide")


def test_a_kept_tag_that_is_no_locus_of_the_assembly_stops_the_build(
    bw25113: EcoliK12BW25113Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    reconcile = reconcile_locus_tags

    def stray(genome: Any, names: pd.Series, *, label: str) -> tuple[pd.Series, Any]:
        stored, report = reconcile(genome, names, label=label)
        return pd.Series([f"BW25113_9{i:03d}" for i in range(len(stored))]), report

    monkeypatch.setattr(s, "MIN_RESOLVED_FRACTION", 0.7)
    monkeypatch.setattr(s, "reconcile_locus_tags", stray)
    with pytest.raises(RuntimeError, match="not loci of the pinned assembly"):
        s.resolve_columns(_SYNTHETIC_GENES, bw25113, label="stray")


def test_the_loader_declares_the_bacterial_environment_response_classes() -> None:
    dataset = s.EnvChemgenShiver2016Dataset.__new__(s.EnvChemgenShiver2016Dataset)
    assert dataset.experiment_class is BacterialEnvironmentResponseExperiment
    assert dataset.reference_class is BacterialEnvironmentResponseExperimentReference
    assert dataset.raw_file_names == [s.DATA_FILENAME]
    frame = object()
    assert dataset.preprocess_raw(frame) is frame
    with pytest.raises(NotImplementedError, match="builds records in process"):
        dataset.create_experiment()


def test_verify_build_passes_l0_to_l4_on_the_synthetic_build(
    tmp_path: Path, mirrored: Path, bw25113: EcoliK12BW25113Genome
) -> None:
    root = tmp_path / "dataset"
    s.EnvChemgenShiver2016Dataset(root=str(root))
    report = s.verify_build(str(root), genome=bw25113, expected_count=227)
    assert report.passed, report.summary()
    rows = {result.name: result for result in report.results}
    assert rows["count"].details["observed"] == 227
    assert rows["pair_uniqueness"].details["n_duplicated"] == 0
    assert rows["measurement_type_consistent"].details["measurement_types"] == [
        "z_score"
    ]
    assert rows["reference_zero"].details["worst_abs"] == 0.0
    assert rows["environment_perturbed"].details["n_missing"] == 0
    assert rows["environment_perturbed"].details["baseline_temperature"] == 37.0
    assert (
        json.loads((root / "preprocess" / "verification_report.json").read_text())[
            "dataset_name"
        ]
        == "dataset"
    )
