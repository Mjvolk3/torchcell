# tests/torchcell/datasets/ecoli/test_lamoureux2023_growth.py
# [[tests.torchcell.datasets.ecoli.test_lamoureux2023_growth]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_lamoureux2023_growth.py
"""Lamoureux 2023 growth-rate loader
(``torchcell.datasets.ecoli.lamoureux2023_growth``).

The synthetic tests write a ``metadata_qc.csv`` carrying the release's own columns and
one row per behaviour the loader has to get right: two base-condition rows in two
projects (which the reference aggregates and which repeat ``rep_id``), a base-condition
row on another carbon concentration (which the environment comparison must exclude), a
deletion row, a zero-rate row, a blank-rate row, an evolved isolate and a non-MG1655
strain. The genome is the real MG1655 class over the synthetic assembly of
``tests/torchcell/sequence/genome/_bacterial_fixtures.py``, so nothing reads
``$DATA_ROOT`` and no network call is permitted.

The ``@pytest.mark.data`` tests read the real mirror and pin the numbers the dendron note
and issue #826 state: 354 released rate cells all of them ``p1k_*`` ids, 103 surviving
the expression loader's own genotype and environment rules, 14 of those exactly zero, 89
records spanning 0.07 to 1.42 1/hr, and the base condition's 8 rows averaging 0.63875.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from pathlib import Path

import pandas as pd
import pytest

import torchcell.datasets.ecoli.lamoureux2023 as lm
import torchcell.datasets.ecoli.lamoureux2023_growth as lg
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    MG1655_GAF,
    MG1655_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.datamodels.schema import AssemblyReferenceGenome, MeasurementType
from torchcell.sequence.genome.ecoli.k12 import MG1655_ASSEMBLY, EcoliK12MG1655Genome
from torchcell.verification.report import Level

MG1655_PIN = AssemblyReferenceGenome(
    species="Escherichia coli",
    strain="MG1655",
    assembly_set="ecoli_K12_MG1655_ASM584v2",
    assembly_accession="GCA_000005845.2",
)

#: The synthetic rows, and what each one exercises. The two ``ica``/``ytf`` base rows
#: repeat ``rep_id`` 1, which is what makes ``screen_id`` load-bearing for L1.
SYNTHETIC: tuple[tuple[str, str, str, str, str, str], ...] = (
    # (sample, project, condition, rate, description, carbon)
    ("p1k_00001", "ica", "wt_glc", "0.58", "", "glucose(2)"),
    ("p1k_00002", "ytf", "wt_glc", "0.68", "", "glucose(2)"),
    ("p1k_00003", "ssw", "wt_glc", "0.73", "", "glucose(4)"),
    ("p1k_00004", "ytf", "delthrA", "0.41", " del_thrA", "glucose(2)"),
    ("p1k_00005", "oxidative", "wt_pq", "0.0", "", "glucose(2)"),
    ("p1k_00006", "misc2", "wt_fru", "", "", "fructose(2)"),
    ("p1k_00007", "glu", "ale", "0.90", "", "glucose(2)"),
    ("p1k_00008", "omics", "bw", "0.80", "", "glucose(2)"),
)
#: The synthetic rows that become records, in sample order.
SYNTHETIC_RECORDS = ("p1k_00001", "p1k_00002", "p1k_00003", "p1k_00004")
SYNTHETIC_RATE_CELLS = 7
SYNTHETIC_ZEROS = 1
SYNTHETIC_BASE = ("p1k_00001", "p1k_00002")
SYNTHETIC_BASE_MEAN = 0.63


def _metadata_frame() -> pd.DataFrame:
    """A metadata table with the release's columns and the synthetic rows."""
    rows: dict[str, dict[str, str]] = {}
    for sample, project, condition, rate, suffix, carbon in SYNTHETIC:
        row = {
            lm.COL_SAMPLE: f"{project}__{condition}__1",
            lm.COL_STUDY: "pColi" if project == "pcoli" else project,
            lm.COL_PROJECT: project,
            lm.COL_CONDITION: condition,
            lm.COL_DESCRIPTION: f"Escherichia coli K-12 MG1655{suffix}",
            lm.COL_STRAIN: "BW25113" if project == "omics" else "MG1655",
            lm.COL_CULTURE: "Batch",
            lm.COL_EVOLVED: "Endpoint" if condition == "ale" else "No",
            lm.COL_MEDIA: "M9",
            lm.COL_TEMPERATURE: "37",
            lm.COL_PH: "7",
            lm.COL_CARBON: carbon,
            lm.COL_NITROGEN: "NH4Cl(1)",
            lm.COL_ACCEPTOR: "O2",
            lm.COL_TRACE: "sauer trace element mixture",
            lm.COL_SUPPLEMENT: "",
            lm.COL_ANTIBIOTIC: "Kanamycin (50 ug/mL)" if suffix else "",
            lm.COL_FULL_NAME: f"{project}:{condition}",
            lm.COL_REP: "1",
            lm.COL_REPLICATES: "2.0",
            lm.COL_PROJECT_REFERENCE: "p1k_00001;p1k_00002",
            lg.COL_GROWTH_RATE: rate,
        }
        rows[sample] = row
    return pd.DataFrame.from_dict(rows, orient="index")


@pytest.fixture
def metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The synthetic metadata CSV, with the module's pin repointed at its bytes."""
    path = tmp_path / lm.METADATA.name
    _metadata_frame().to_csv(path)
    monkeypatch.setattr(
        lm,
        "METADATA",
        lm.METADATA.model_copy(
            update={"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        ),
    )
    monkeypatch.setattr(lg, "EXPECTED_RATE_CELLS", SYNTHETIC_RATE_CELLS)
    monkeypatch.setattr(lg, "EXPECTED_ZERO_RATES", SYNTHETIC_ZEROS)
    monkeypatch.setattr(lg, "EXPECTED_AFTER_LOADER_RULES", len(SYNTHETIC_RECORDS) + 1)
    monkeypatch.setattr(lg, "EXPECTED_RECORDS", len(SYNTHETIC_RECORDS))
    monkeypatch.setattr(lg, "EXPECTED_BASE_RECORDS", len(SYNTHETIC_BASE))
    monkeypatch.setattr(lg, "MIN_RATE", 0.41)
    monkeypatch.setattr(lg, "MAX_RATE", 0.73)
    monkeypatch.setattr(lg, "BASE_RATE_MEAN", SYNTHETIC_BASE_MEAN)
    return path


@pytest.fixture
def mg1655(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12MG1655Genome:
    """The real MG1655 class over the synthetic assembly; network refused."""
    tier = write_assembly(tmp_path / "tier", MG1655_ASSEMBLY, MG1655_LOCI, MG1655_GAF)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, tier)
    (tmp_path / "mg1655").mkdir()
    return EcoliK12MG1655Genome(genome_root=str(tmp_path / "mg1655"), overwrite=False)


# --------------------------------------------------------------------------- #
# The declared module constants
# --------------------------------------------------------------------------- #
def test_declared_counts_are_the_measured_ones() -> None:
    assert lg.EXPECTED_RATE_CELLS == 354
    assert lg.EXPECTED_AFTER_LOADER_RULES == 103
    assert lg.EXPECTED_ZERO_RATES == 14
    assert lg.EXPECTED_RECORDS == 89
    assert lg.EXPECTED_RECORDS + lg.EXPECTED_ZERO_RATES == (
        lg.EXPECTED_AFTER_LOADER_RULES
    )
    assert lg.EXPECTED_UNPERTURBED == 0
    assert lg.EXPECTED_BASE_RECORDS == 8
    assert (lg.MIN_RATE, lg.MAX_RATE) == (0.07, 1.42)
    assert lg.BASE_RATE_MEAN == 0.63875
    assert lg.UNITS == "Growth Rate (1/hr)"


def test_the_sourcing_is_the_released_header_and_lamoureuxs_own_quotes() -> None:
    column = lg.SOURCED_VALUES["growth_rate_column"]
    assert column.value == lg.COL_GROWTH_RATE
    assert column.quote == lg.COL_GROWTH_RATE
    assert column.provenance.sha256 == lm.METADATA.sha256
    assert column.note is not None and "defines nothing" in column.note
    assert lg.SOURCED_VALUES["strain"] is lm.STRAIN
    assert lg.SOURCED_VALUES["harvest"] is lm.HARVEST


def test_every_unstated_field_is_a_typed_gap_and_none_is_a_reading() -> None:
    assert [gap.field for gap in lg.RECORD_GAPS] == [
        "assay_type",
        "environment_response_uncertainty",
        "environment_response_se",
        "n_samples",
        "sample_unit",
    ]
    assert [gap.field for gap in lg.REFERENCE_GAPS[-2:]] == [
        "screen_id",
        "replicate_id",
    ]
    # the n_samples gap carries the measurement that a released row is not a culture
    note = dict((gap.field, gap.note) for gap in lg.RECORD_GAPS)["n_samples"]
    assert note is not None and "three values on three pairs" in note


# --------------------------------------------------------------------------- #
# Reading the released column
# --------------------------------------------------------------------------- #
def test_read_rate_rows_keeps_only_what_the_expression_rules_keep(
    metadata: Path,
) -> None:
    rows, rules = lg.read_rate_rows(str(metadata))
    assert [row.sample for row in rows] == list(SYNTHETIC_RECORDS)
    assert {row.project for row in rows} == {"ica", "ytf", "ssw"}
    by_rule = {rule.rule: rule for rule in rules}
    assert by_rule[lg.DROP_ZERO_RATE].n_records == SYNTHETIC_ZEROS
    assert by_rule[lg.DROP_ZERO_RATE].stage == "readout"
    assert by_rule[lg.DROP_ZERO_RATE].items == ["p1k_00005 (oxidative:wt_pq, 0.0)"]
    assert by_rule["evolved_isolate"].n_records == 1
    assert by_rule["strain_bw25113"].n_records == 1
    assert sum(rule.n_records for rule in rules) + len(rows) == SYNTHETIC_RATE_CELLS
    # the blank-rate row is not a drop: it never enters the pool
    assert all("p1k_00006" not in rule.items for rule in rules)
    deletions = [row for row in rows if row.deleted_symbols]
    assert [(row.sample, row.deleted_symbols) for row in deletions] == [
        ("p1k_00004", ("thrA",))
    ]


def test_read_rate_rows_refuses_a_changed_number_of_released_cells(
    metadata: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lg, "EXPECTED_RATE_CELLS", SYNTHETIC_RATE_CELLS + 1)
    with pytest.raises(RuntimeError, match="the module declares"):
        lg.read_rate_rows(str(metadata))


def test_the_zero_rate_rule_states_why_a_zero_is_not_a_measurement() -> None:
    assert "indistinguishable from an unrecorded one" in lg.ZERO_RATE_DESCRIPTION
    assert "reversible" in lg.ZERO_RATE_DESCRIPTION


# --------------------------------------------------------------------------- #
# The base condition and the reference
# --------------------------------------------------------------------------- #
def test_the_base_condition_builds_one_environment_from_its_released_rows(
    metadata: Path,
) -> None:
    base = lg.base_condition_environment(str(metadata))
    assert base.media.name is not None
    rows, _ = lg.read_rate_rows(str(metadata))
    environments = lg.environments_of(str(metadata), rows)
    aggregate = lg.base_aggregate(rows, environments, base)
    assert aggregate.samples == list(SYNTHETIC_BASE)
    assert aggregate.rates == [0.58, 0.68]
    assert aggregate.mean == SYNTHETIC_BASE_MEAN
    assert aggregate.spread == pytest.approx(0.10)


def test_the_base_aggregate_excludes_a_row_grown_on_another_carbon_dose(
    metadata: Path,
) -> None:
    """``p1k_00003`` is a ``*:wt_glc`` wild-type row with a rate, on ``glucose(4)``."""
    rows, _ = lg.read_rate_rows(str(metadata))
    environments = lg.environments_of(str(metadata), rows)
    base = lg.base_condition_environment(str(metadata))
    aggregate = lg.base_aggregate(rows, environments, base)
    assert "p1k_00003" in {row.sample for row in rows}
    assert "p1k_00003" not in aggregate.samples
    assert environments["p1k_00003"].model_dump_json() != base.model_dump_json()


def test_the_base_aggregate_refuses_a_changed_member_count(
    metadata: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, _ = lg.read_rate_rows(str(metadata))
    environments = lg.environments_of(str(metadata), rows)
    base = lg.base_condition_environment(str(metadata))
    monkeypatch.setattr(lg, "EXPECTED_BASE_RECORDS", 3)
    with pytest.raises(RuntimeError, match="carry the base environment"):
        lg.base_aggregate(rows, environments, base)


def test_the_base_aggregate_refuses_a_drifted_mean(
    metadata: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, _ = lg.read_rate_rows(str(metadata))
    environments = lg.environments_of(str(metadata), rows)
    base = lg.base_condition_environment(str(metadata))
    monkeypatch.setattr(lg, "BASE_RATE_MEAN", 0.5)
    with pytest.raises(RuntimeError, match="the module declares"):
        lg.base_aggregate(rows, environments, base)


def test_base_condition_environment_refuses_a_declaration_nothing_matches(
    metadata: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(lg.BASE_CONDITION, lm.COL_CARBON, "sorbitol(2)")
    with pytest.raises(RuntimeError, match="no released row matches"):
        lg.base_condition_environment(str(metadata))


# --------------------------------------------------------------------------- #
# The phenotype
# --------------------------------------------------------------------------- #
def test_the_phenotype_stores_the_released_rate_with_its_screen_and_replicate(
    metadata: Path,
) -> None:
    rows, _ = lg.read_rate_rows(str(metadata))
    phenotype = lg.phenotype(rows[0])
    assert phenotype.measurement_type is MeasurementType.growth_rate
    assert phenotype.environment_response == 0.58
    assert phenotype.units == "Growth Rate (1/hr)"
    assert phenotype.screen_id == "ica"
    assert phenotype.replicate_id == "1"
    assert phenotype.assay_type is None
    assert phenotype.n_samples is None
    assert phenotype.sample_unit is None
    assert phenotype.environment_response_uncertainty is None
    assert phenotype.environment_response_se is None
    assert [gap.field for gap in phenotype.provenance_gaps] == [
        gap.field for gap in lg.RECORD_GAPS
    ]


def test_the_reference_phenotype_carries_the_aggregate_and_two_more_gaps(
    metadata: Path,
) -> None:
    rows, _ = lg.read_rate_rows(str(metadata))
    environments = lg.environments_of(str(metadata), rows)
    aggregate = lg.base_aggregate(
        rows, environments, lg.base_condition_environment(str(metadata))
    )
    reference = lg.reference_phenotype(aggregate)
    assert reference.environment_response == SYNTHETIC_BASE_MEAN
    assert reference.screen_id is None
    assert reference.replicate_id is None
    assert len(reference.provenance_gaps) == len(lg.RECORD_GAPS) + 2


# --------------------------------------------------------------------------- #
# End-to-end build on the synthetic metadata
# --------------------------------------------------------------------------- #
@pytest.fixture
def built(
    tmp_path: Path,
    metadata: Path,
    mg1655: EcoliK12MG1655Genome,
    monkeypatch: pytest.MonkeyPatch,
) -> lg.GrowthRateLamoureux2023Dataset:
    """The synthetic store, built into ``tmp_path``."""
    root = tmp_path / "growth_rate_lamoureux2023"
    (root / "raw").mkdir(parents=True)
    (root / "raw" / lm.METADATA.name).write_bytes(metadata.read_bytes())
    monkeypatch.setattr(lg, "assembly_reference", lambda strain, **_: MG1655_PIN)
    return lg.GrowthRateLamoureux2023Dataset(root=str(root), ecoli_genome=mg1655)


def test_process_writes_one_record_per_stored_rate_with_the_aggregate_reference(
    built: lg.GrowthRateLamoureux2023Dataset,
) -> None:
    assert len(built) == len(SYNTHETIC_RECORDS)
    items = [built.transform_item(built[i]) for i in range(len(built))]
    assert sorted(i["experiment"].phenotype.environment_response for i in items) == [
        0.41,
        0.58,
        0.68,
        0.73,
    ]
    assert {i["reference"].phenotype_reference.environment_response for i in items} == {
        SYNTHETIC_BASE_MEAN
    }
    assert {i["publication"].doi for i in items} == {lm.PAPER_DOI}
    assert sorted(len(i["experiment"].genotype.perturbations) for i in items) == [
        0,
        0,
        0,
        1,
    ]
    assert built.gene_set == {"b0002"}
    # screen_id separates the two base rows, which share an environment and a rep_id
    base = [
        i
        for i in items
        if i["experiment"].phenotype.environment_response in (0.58, 0.68)
    ]
    assert {i["experiment"].phenotype.screen_id for i in base} == {"ica", "ytf"}
    assert len({i["experiment"].environment.model_dump_json() for i in base}) == 1
    assert {i["experiment"].phenotype.replicate_id for i in base} == {"1"}
    built.close_lmdb()


def test_process_writes_the_four_declared_ledgers(
    built: lg.GrowthRateLamoureux2023Dataset,
) -> None:
    preprocess = Path(built.preprocess_dir)
    built.close_lmdb()
    drops = json.loads((preprocess / "dropped_records.json").read_text())
    assert drops["source_rows"] == SYNTHETIC_RATE_CELLS
    assert drops["kept_records"] == len(SYNTHETIC_RECORDS)
    assert drops["dropped_records"] == SYNTHETIC_RATE_CELLS - len(SYNTHETIC_RECORDS)
    assert lg.DROP_ZERO_RATE in {rule["rule"] for rule in drops["rules"]}
    assert any("oxyR, soxR and soxS" in note for note in drops["notes"])
    base = json.loads((preprocess / "base_condition.json").read_text())
    assert base["declared_cells"] == lg.BASE_CONDITION
    assert base["aggregate"]["samples"] == list(SYNTHETIC_BASE)
    assert "EMPTY rate cell" in base["why_not_the_expression_loaders_reference"]
    sourced = json.loads((preprocess / "sourced_values.json").read_text())
    assert sourced["growth_rate_column"]["value"] == lg.COL_GROWTH_RATE
    samples = json.loads((preprocess / "record_samples.json").read_text())
    assert samples == list(SYNTHETIC_RECORDS)
    assert (preprocess / "locus_tag_reconciliation.json").exists()
    frame = pd.read_csv(preprocess / "growth_rates.csv")
    assert len(frame) == len(SYNTHETIC_RECORDS)
    assert frame["is_base_condition"].sum() == len(SYNTHETIC_BASE)
    assert min(frame["n_environment_perturbations"]) >= 1  # L3 environment_perturbed
    assert (preprocess / "build_manifest.json").exists()


def test_verify_build_passes_every_level_on_the_absolute_branch(
    built: lg.GrowthRateLamoureux2023Dataset, mg1655: EcoliK12MG1655Genome
) -> None:
    root = built.root
    built.close_lmdb()
    report = lg.verify_build(root, genome=mg1655, expected_count=len(SYNTHETIC_RECORDS))
    assert [(r.level, r.name) for r in report.results if not r.passed] == []
    assert report.passed
    assert (Level.L4, "gene_containment_mg1655_b_numbers") in [
        (r.level, r.name) for r in report.results
    ]
    assert any(
        "absolute rule" in r.message
        for r in report.results
        if r.name == "reference_zero"
    )


def test_l4_refuses_a_deletion_that_is_not_an_mg1655_locus(
    mg1655: EcoliK12MG1655Genome,
) -> None:
    records = [
        {
            "experiment": {
                "genotype": {"perturbations": [{"systematic_gene_name": "b9999"}]}
            }
        }
    ]
    result = lg.l4_deleted_genes_are_mg1655_loci(records, mg1655)
    assert result.passed is False
    assert result.details["missing"] == ["b9999"]


# --------------------------------------------------------------------------- #
# Data-gated: the real mirror
# --------------------------------------------------------------------------- #
def _real_metadata() -> str:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT is not set")
    path = osp.join(data_root, lm.RAW_DIR_REL, lm.METADATA.relpath)
    if not osp.exists(path):
        pytest.skip(f"{path} is not in the raw mirror")
    return path


@pytest.mark.data
def test_real_release_carries_354_rate_cells_all_of_them_precise1k_ids() -> None:
    frame = lm.read_metadata(_real_metadata())
    cells = [
        str(sample)
        for sample in frame.index
        if str(frame.loc[sample, lg.COL_GROWTH_RATE]).strip()
    ]
    assert len(cells) == lg.EXPECTED_RATE_CELLS
    assert all(sample.startswith("p1k_") for sample in cells)


@pytest.mark.data
def test_real_rules_keep_103_of_which_14_are_zero_and_89_are_records() -> None:
    rows, rules = lg.read_rate_rows(_real_metadata())
    by_rule = {rule.rule: rule.n_records for rule in rules}
    assert by_rule[lg.DROP_ZERO_RATE] == lg.EXPECTED_ZERO_RATES
    assert len(rows) == lg.EXPECTED_RECORDS
    assert len(rows) + by_rule[lg.DROP_ZERO_RATE] == lg.EXPECTED_AFTER_LOADER_RULES
    assert min(row.rate for row in rows) == lg.MIN_RATE
    assert max(row.rate for row in rows) == lg.MAX_RATE
    assert by_rule["evolved_isolate"] == 138
    assert by_rule["heterologous_expression_construct"] == 94
    assert by_rule["strain_bw25113"] == 12
    assert by_rule["point_mutation_allele"] == 7


@pytest.mark.data
def test_real_base_condition_is_eight_rows_averaging_the_declared_mean() -> None:
    path = _real_metadata()
    rows, _ = lg.read_rate_rows(path)
    environments = lg.environments_of(path, rows)
    aggregate = lg.base_aggregate(
        rows, environments, lg.base_condition_environment(path)
    )
    assert len(aggregate.samples) == lg.EXPECTED_BASE_RECORDS
    assert sorted(aggregate.rates) == [0.58, 0.58, 0.63, 0.63, 0.66, 0.66, 0.68, 0.69]
    assert aggregate.mean == lg.BASE_RATE_MEAN


@pytest.mark.data
def test_real_public_k12_arm_contributes_no_rate() -> None:
    """Every released rate cell is a PRECISE-1K id, so the sibling arm has none."""
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT is not set")
    path = Path(
        data_root,
        "data/torchcell/rnaseq_public_k12_lamoureux2023/preprocess/record_samples.json",
    )
    if not path.exists():
        pytest.skip(f"{path} is not built")
    frame = lm.read_metadata(_real_metadata())
    with_rate = {
        str(sample)
        for sample in frame.index
        if str(frame.loc[sample, lg.COL_GROWTH_RATE]).strip()
    }
    assert set(json.loads(path.read_text())) & with_rate == set()


@pytest.mark.data
def test_real_dev_store_holds_89_records_and_its_declared_ledger() -> None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT is not set")
    root = osp.join(data_root, lg.DATASET_ROOT_REL)
    drops_path = Path(root, "preprocess", "dropped_records.json")
    if not drops_path.exists():
        pytest.skip(f"{root} is not built")
    drops = json.loads(drops_path.read_text())
    assert drops["kept_records"] == lg.EXPECTED_RECORDS
    base = json.loads(Path(root, "preprocess", "base_condition.json").read_text())
    assert base["aggregate"]["mean"] == lg.BASE_RATE_MEAN
    assert len(base["aggregate"]["samples"]) == lg.EXPECTED_BASE_RECORDS
