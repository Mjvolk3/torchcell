# tests/torchcell/datasets/ecoli/test_lamoureux2023_growth.py
# [[tests.torchcell.datasets.ecoli.test_lamoureux2023_growth]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_lamoureux2023_growth.py
"""Lamoureux 2023 growth-rate loader
(``torchcell.datasets.ecoli.lamoureux2023_growth``).

The synthetic tests write a ``metadata_qc.csv`` carrying the release's own columns and
one row per behaviour the loader has to get right: two base-condition rows in two
projects (which the reference aggregates and which repeat ``rep_id``), a base-condition
row on another carbon concentration (which the environment comparison must exclude), a
deletion row, a zero-rate row, a blank-rate row, an evolved isolate, a non-MG1655
strain, an rpoB substitution and a ``pColi`` heterologous construct; since #1045 part 2
every rate row is a record. The genome is the real MG1655 class over the synthetic assembly of
``tests/torchcell/sequence/genome/_bacterial_fixtures.py``, so nothing reads
``$DATA_ROOT`` and no network call is permitted.

The ``@pytest.mark.data`` tests read the real mirror and pin the numbers the dendron note
and issue #826 state: 354 released rate cells all of them ``p1k_*`` ids, all 354 of them
records under the expression loader's own genotype and environment rules (#1045 part 2), 16 of those
exactly zero, spanning 0 to 1.42 1/hr, and the base condition's 8 rows averaging 0.63875.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from pathlib import Path
from typing import Any

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
from torchcell.datamodels.schema import (
    EnvironmentResponseExperiment,
    MeasurementType,
    ReferenceGenome,
    StrainReferenceGenome,
)
from torchcell.sequence.genome.ecoli.k12 import MG1655_ASSEMBLY, EcoliK12MG1655Genome

#: The shared verifier's genome-reference rules. Under the owner rule (#1057) a gene
#: symbol stored as released need not be a key of the annotation, so these may read it
#: as a failure until the verifier changes (a separate branch); nothing else may fail.
GENOME_REFERENCE_RULES = frozenset(
    {"canonical_gene_names", "gene_containment_sgd", "current_genome_genes"}
)

MG1655_PIN = ReferenceGenome(
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
    ("p1k_00005", "oxidative", "wt_pq", "0.0", "", "fructose(2)"),
    ("p1k_00006", "misc2", "wt_fru", "", "", "fructose(2)"),
    ("p1k_00007", "glu", "ale", "0.90", "", "glucose(2)"),
    ("p1k_00008", "omics", "bw", "0.80", "", "glucose(2)"),
    ("p1k_00009", "rpob", "e546v", "0.70", " rpoBE546V", "glucose(2)"),
    ("p1k_00010", "pcoli", "ec_gene", "0.50", "", "glucose(2)"),
)
#: The synthetic rows that become records, in sample order (#1057: the zero rate, the
#: evolved isolate, the BW25113 sample and the rpoB substitution included; #1045 part 2:
#: the pColi construct too).
SYNTHETIC_RECORDS = (
    "p1k_00001",
    "p1k_00002",
    "p1k_00003",
    "p1k_00004",
    "p1k_00005",
    "p1k_00007",
    "p1k_00008",
    "p1k_00009",
    "p1k_00010",
)
SYNTHETIC_RATE_CELLS = 9
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
            lm.COL_DESCRIPTION: (
                "Escherichia coli K-12 BW25113"
                if project == "omics"
                else f"Escherichia coli K-12 MG1655{suffix}"
            ),
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
    monkeypatch.setattr(lg, "EXPECTED_AFTER_LOADER_RULES", len(SYNTHETIC_RECORDS))
    monkeypatch.setattr(lg, "EXPECTED_RECORDS", len(SYNTHETIC_RECORDS))
    monkeypatch.setattr(lg, "EXPECTED_BASE_RECORDS", len(SYNTHETIC_BASE))
    monkeypatch.setattr(lg, "MIN_RATE", 0.0)
    monkeypatch.setattr(lg, "MAX_RATE", 0.90)
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
    assert lg.EXPECTED_AFTER_LOADER_RULES == 354
    assert lg.EXPECTED_ZERO_RATES == 16
    assert lg.EXPECTED_RECORDS == 354
    # #1057: a released 0.0 is stored, so the zero rates are records
    assert lg.EXPECTED_RECORDS == lg.EXPECTED_AFTER_LOADER_RULES
    assert lg.EXPECTED_UNPERTURBED == 0
    assert lg.EXPECTED_BASE_RECORDS == 8
    assert (lg.MIN_RATE, lg.MAX_RATE) == (0.0, 1.42)
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
    assert rules == []
    assert len(rows) == SYNTHETIC_RATE_CELLS
    by_sample = {row.sample: row for row in rows}
    assert by_sample["p1k_00005"].rate == 0.0
    evolved = by_sample["p1k_00007"].background
    assert evolved is not None and evolved.evolved
    strain = by_sample["p1k_00008"].background
    assert strain is not None and strain.name == "BW25113"
    assert [a.gene for a in by_sample["p1k_00009"].released_alleles] == ["rpoB"]
    construct = by_sample["p1k_00010"]
    assert construct.untyped_edits and construct.background is not None
    assert construct.background.name == "Escherichia coli K-12 MG1655"
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


def test_genotype_of_stores_a_released_allele_under_its_strain_description(
    metadata: Path,
) -> None:
    rows, _ = lg.read_rate_rows(str(metadata))
    row = next(r for r in rows if r.sample == "p1k_00009")
    (perturbation,) = lg.genotype_of(row).perturbations
    assert perturbation.perturbation_type == "released_identifier"
    assert perturbation.systematic_gene_name == "rpoB"
    assert getattr(perturbation, "strain_label") == (
        "Escherichia coli K-12 MG1655 rpoBE546V"
    )


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
    assert phenotype.replicate_id == "ica__wt_glc__1"
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

    def pin(
        strain: str, *, background: Any = None, data_root: Any = None
    ) -> ReferenceGenome:
        if background is None:
            return MG1655_PIN
        return StrainReferenceGenome.model_validate(
            MG1655_PIN.model_dump()
            | {"strain": background.name, "background": background}
        )

    monkeypatch.setattr(lg, "assembly_reference", pin)
    return lg.GrowthRateLamoureux2023Dataset(root=str(root))


def test_process_writes_one_record_per_stored_rate_with_the_aggregate_reference(
    built: lg.GrowthRateLamoureux2023Dataset,
) -> None:
    assert len(built) == len(SYNTHETIC_RECORDS)
    items = [built.transform_item(built[i]) for i in range(len(built))]
    assert sorted(i["experiment"].phenotype.environment_response for i in items) == [
        0.0,
        0.41,
        0.50,
        0.58,
        0.68,
        0.70,
        0.73,
        0.80,
        0.90,
    ]
    assert {i["reference"].phenotype_reference.environment_response for i in items} == {
        SYNTHETIC_BASE_MEAN
    }
    assert {i["publication"].doi for i in items} == {lm.PAPER_DOI}
    assert sorted(len(i["experiment"].genotype.perturbations) for i in items) == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
    ]
    # #1045 part 2: the deletion keys the symbol its del_ token releases
    assert built.gene_set == {"thrA", "rpoB"}
    # the evolved isolate, the BW25113 sample and the pColi construct sit on their
    # released strain; the construct and the isolate share a Strain Description and
    # are still two backgrounds
    backgrounds = [
        i["reference"].genome_reference.background
        for i in items
        if isinstance(i["reference"].genome_reference, StrainReferenceGenome)
    ]
    assert sorted(b.name for b in backgrounds) == [
        "BW25113",
        "Escherichia coli K-12 MG1655",
        "Escherichia coli K-12 MG1655",
    ]
    shared = [b for b in backgrounds if b.name == "Escherichia coli K-12 MG1655"]
    assert sorted(str(b.parents) for b in shared) == ["None", "['MG1655']"]
    # screen_id separates the two base rows, which share an environment and a rep_id
    base = [
        i
        for i in items
        if i["experiment"].phenotype.environment_response in (0.58, 0.68)
    ]
    assert {i["experiment"].phenotype.screen_id for i in base} == {"ica", "ytf"}
    assert len({i["experiment"].environment.model_dump_json() for i in base}) == 1
    assert {i["experiment"].phenotype.replicate_id for i in base} == {
        "ica__wt_glc__1",
        "ytf__wt_glc__1",
    }
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
    assert drops["rules"] == []
    assert any("p1k_00005 (oxidative:wt_pq)" in note for note in drops["notes"])
    base = json.loads((preprocess / "base_condition.json").read_text())
    assert base["declared_cells"] == lg.BASE_CONDITION
    assert base["aggregate"]["samples"] == list(SYNTHETIC_BASE)
    assert "EMPTY rate cell" in base["why_not_the_expression_loaders_reference"]
    sourced = json.loads((preprocess / "sourced_values.json").read_text())
    assert sourced["growth_rate_column"]["value"] == lg.COL_GROWTH_RATE
    samples = json.loads((preprocess / "record_samples.json").read_text())
    assert samples == list(SYNTHETIC_RECORDS)
    assert not (preprocess / "locus_tag_reconciliation.json").exists()
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
    # the released rpoB symbol is not an annotation key; the shared genome-reference
    # rules may say so and never gate (#1057), and nothing else may fail
    failed = {r.name for r in report.results if not r.passed}
    assert failed <= GENOME_REFERENCE_RULES, failed
    assert report.passed
    assert any(
        "absolute rule" in r.message
        for r in report.results
        if r.name == "reference_zero"
    )


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
def test_real_rules_keep_all_354_of_which_16_are_zero() -> None:
    rows, rules = lg.read_rate_rows(_real_metadata())
    assert rules == []
    assert len(rows) == lg.EXPECTED_RECORDS
    assert sum(row.rate == 0.0 for row in rows) == lg.EXPECTED_ZERO_RATES
    assert min(row.rate for row in rows) == lg.MIN_RATE
    assert max(row.rate for row in rows) == lg.MAX_RATE
    # #1057: these were drops; they are records on their released strain or allele
    assert sum(
        row.background is not None and row.background.evolved for row in rows
    ) == (138)
    assert (
        sum(
            row.background is not None and row.background.name == "BW25113"
            for row in rows
        )
        == 12
    )
    assert sum(bool(row.released_alleles) for row in rows) == 7
    # #1045 part 2: the pColi constructs, on a background named as released
    assert sum(row.untyped_edits for row in rows) == 94


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
def test_real_dev_store_holds_260_records_and_its_declared_ledger() -> None:
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


# --------------------------------------------------------------------------- #
# 2026.10.15 (lane V): two evolved isolates with one rate are two records
# --------------------------------------------------------------------------- #
def test_two_evolved_isolates_with_one_released_rate_are_two_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two adaptive-laboratory-evolution isolates of one project, released with the same
    growth rate in the same condition family, are two measurements and must dump to two
    content ids.

    This is the release's own shape: ``glu:glu_ale3``, ``glu:glu_ale5`` and
    ``glu:glu_ale9`` all release 1.01 1/hr, and an isolate's strain rides on the
    ``StrainReferenceGenome`` background rather than on the experiment, so all three
    dumped as the same empty MG1655 genotype in the same environment with the same
    ``screen_id`` (``glu``) and the same ``rep_id`` (``1``). 13 groups and 14 records of
    the real store were in that state, and the graph held one ``Experiment`` node per
    group. ``replicate_id`` now carries the released ``sample_id``, so the two differ in
    exactly that field: before lane V the two dumps were byte-identical and this test
    failed on ``assert digests[0] != digests[1]``.
    """
    frame = _metadata_frame()
    for sample, condition, description in (
        ("p1k_09001", "glu_ale3", "Escherichia coli GLU.3.412.1"),
        ("p1k_09002", "glu_ale9", "Escherichia coli GLU.9.433.1"),
    ):
        row = dict(frame.loc["p1k_00007"])
        row[lm.COL_SAMPLE] = f"glu__{condition}__1"
        row[lm.COL_CONDITION] = condition
        row[lm.COL_FULL_NAME] = f"glu:{condition}"
        row[lm.COL_DESCRIPTION] = description
        row[lg.COL_GROWTH_RATE] = "1.01"
        frame.loc[sample] = row
    path = tmp_path / lm.METADATA.name
    frame.to_csv(path)
    monkeypatch.setattr(lg, "EXPECTED_RATE_CELLS", SYNTHETIC_RATE_CELLS + 2)
    rows, _ = lg.read_rate_rows(str(path))
    environments = lg.environments_of(str(path), rows)
    pair = [row for row in rows if row.sample in ("p1k_09001", "p1k_09002")]
    assert [row.rate for row in pair] == [1.01, 1.01]
    dumps = [
        EnvironmentResponseExperiment(
            dataset_name="growth_rate_lamoureux2023",
            genotype=lg.genotype_of(row),
            environment=environments[row.sample],
            phenotype=lg.phenotype(row),
        ).model_dump()
        for row in pair
    ]
    digests = [
        hashlib.sha256(json.dumps(dump).encode("utf-8")).hexdigest() for dump in dumps
    ]
    assert digests[0] != digests[1]
    assert [dump["phenotype"]["replicate_id"] for dump in dumps] == [
        "glu__glu_ale3__1",
        "glu__glu_ale9__1",
    ]
    assert {dump["phenotype"]["screen_id"] for dump in dumps} == {"glu"}
    assert [len(dump["genotype"]["perturbations"]) for dump in dumps] == [0, 0]
    # the released sample id is the whole difference between the two dumps, which is
    # what made them one node before it was carried
    for dump in dumps:
        dump["phenotype"]["replicate_id"] = None
    assert dumps[0] == dumps[1]


def test_a_release_whose_sample_id_repeats_refuses_the_build(
    tmp_path: Path, mg1655: EcoliK12MG1655Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``sample_id`` is what each record's ``replicate_id`` carries and what tells two
    samples of one condition apart, so a release that repeats it stops the build naming
    the repeated id rather than storing two records that dump to one content id.
    """
    frame = _metadata_frame()
    row = dict(frame.loc["p1k_00007"])
    row[lg.COL_GROWTH_RATE] = "0.91"
    frame.loc["p1k_09003"] = row
    path = tmp_path / lm.METADATA.name
    frame.to_csv(path)
    root = tmp_path / "growth_rate_lamoureux2023"
    (root / "raw").mkdir(parents=True)
    (root / "raw" / lm.METADATA.name).write_bytes(path.read_bytes())
    monkeypatch.setattr(
        lm,
        "METADATA",
        lm.METADATA.model_copy(
            update={"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        ),
    )
    monkeypatch.setattr(lg, "EXPECTED_RATE_CELLS", SYNTHETIC_RATE_CELLS + 1)
    monkeypatch.setattr(lg, "EXPECTED_ZERO_RATES", SYNTHETIC_ZEROS)
    monkeypatch.setattr(lg, "EXPECTED_AFTER_LOADER_RULES", len(SYNTHETIC_RECORDS) + 1)
    monkeypatch.setattr(lg, "EXPECTED_RECORDS", len(SYNTHETIC_RECORDS) + 1)
    monkeypatch.setattr(lg, "EXPECTED_BASE_RECORDS", len(SYNTHETIC_BASE))
    monkeypatch.setattr(lg, "MIN_RATE", 0.0)
    monkeypatch.setattr(lg, "MAX_RATE", 0.91)
    monkeypatch.setattr(lg, "BASE_RATE_MEAN", SYNTHETIC_BASE_MEAN)
    with pytest.raises(RuntimeError, match="glu__ale__1"):
        lg.GrowthRateLamoureux2023Dataset(root=str(root))
    assert not (root / "processed" / "lmdb").exists()
