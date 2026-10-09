# tests/torchcell/datasets/ecoli/test_wang2015_growth.py
# [[tests.torchcell.datasets.ecoli.test_wang2015_growth]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_wang2015_growth.py
"""Wang 2015 no-isoprenol loader (``torchcell.datasets.ecoli.wang2015_growth``).

The synthetic tests build a four-row Table S3 and a matching Table S2 in the SI PDF's
own layout, then drive the environment builder, both phenotype builders, the
fitness-range oracle and a full ``process()`` into ``tmp_path``. The genome is the real
BW25113 class over the synthetic assembly of
``tests/torchcell/sequence/genome/_bacterial_fixtures.py``, so nothing reads
``$DATA_ROOT`` and no network call is permitted.

The ``@pytest.mark.data`` tests read the real mirror and pin the numbers the dendron note
and issue #826 state: 47 released rows, the 46 ratios spanning 0.877108 to 1.030120 with
a median of 0.956024, none non-positive and none with a zero SD, and the parent row at
8.30 +/- 0.16.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import os.path as osp
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest

import torchcell.datasets.ecoli.wang2015 as wg
import torchcell.datasets.ecoli.wang2015_growth as gw
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    BW25113_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.data import file_sha256
from torchcell.datamodels.media import YT_2X
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    BacterialFitnessExperiment,
    BacterialFitnessExperimentReference,
    EnvironmentPhysicalPerturbation,
    PhysicalFactor,
    SampleUnit,
    UncertaintyType,
)
from torchcell.sequence.genome.ecoli.k12 import BW25113_ASSEMBLY, EcoliK12BW25113Genome

D = "Δ"
PM = "±"
TABLE_S2 = f"""Table S2. Mutant strains used in this study.

  Names        Keio No.      MDT               Names     Keio No.      MDT
                            families                                  families
BW{D}thrA        JW0001         ABC          BW{D}hokC       JW4367        OMP
BW{D}thrL        JJW4367        RND
"""
#: Parent 8.00 +/- 0.10 and three mutants, so every stored ratio is exact in binary:
#: 8.00 -> 1.0, 4.00 -> 0.5, 6.00 -> 0.75.
TABLE_S3 = f"""Table S3. Cell growth of the MDT null mutants in the absence and presence of isoprenol.

    Strains          No         0.5% (v/v)           Strains        No          0.5% (v/v)
                  isoprenol      isoprenol                       isoprenol       isoprenol
  BW25113        8.00 {PM} 0.10     4.00 {PM} 0.01         BW{D}hokC      6.00 {PM} 0.20   6.00 {PM} 0.10

  BW{D}thrA        8.00 {PM} 0.05     2.00 {PM} 0.04
  BW{D}thrL        4.00 {PM} 0.30     2.50 {PM} 0.01
Note: The results are presented as means {PM} standard divisions.
"""
TABLE_S4 = f"""Table S4. Transcript profiles of targeted transporters in BW{D}acrA.
        acrA             3.16 {PM} 0.87                           1.59 {PM} 0.08
"""
TEXT = "header page\n" + TABLE_S2 + TABLE_S3 + TABLE_S4
SECTION_S3 = wg.table_section(TEXT, wg.TABLE_S3_MARKER, wg.TABLE_S4_MARKER)
SYNTHETIC_DIGEST = wg.table_s3_digest(wg.parse_table_s3(SECTION_S3))
SYNTHETIC_ROWS = 4
SYNTHETIC_RECORDS = 3
#: 4.00/8.00 and 8.00/8.00 over the synthetic parent.
SYNTHETIC_MIN = 0.5
SYNTHETIC_MAX = 1.0

REFERENCE = AssemblyReferenceGenome(
    species="Escherichia coli",
    strain="BW25113",
    assembly_set="ecoli_K12_BW25113_ASM75055v1",
    assembly_accession="GCA_000750555.1",
)


def _row(strain: str, mean: float, sd: float) -> wg.TableS3Row:
    """One Table S3 row; only the no-isoprenol pair is read by this module."""
    return wg.TableS3Row(
        strain=strain,
        od600_without=mean,
        sd_without=sd,
        od600_with=mean / 2.0,
        sd_with=0.01,
    )


@pytest.fixture
def bw25113(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12BW25113Genome:
    """The real BW25113 class over the synthetic assembly; network refused."""
    files = write_assembly(tmp_path / "tier", BW25113_ASSEMBLY, BW25113_LOCI)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    root = tmp_path / "bw25113"
    root.mkdir()
    return EcoliK12BW25113Genome(genome_root=str(root), overwrite=False)


@pytest.fixture
def synthetic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bw25113: EcoliK12BW25113Genome
) -> Path:
    """A dataset root whose ``raw/`` holds a stand-in PDF with the synthetic text."""

    def presence_only(raw_dir: str, pins: Mapping[str, str]) -> None:
        missing = [name for name in pins if not osp.exists(osp.join(raw_dir, name))]
        if missing:
            raise FileNotFoundError(f"pinned raw files absent: {missing}")

    root = tmp_path / gw.DATASET_ROOT_REL
    (root / "raw").mkdir(parents=True)
    for raw in wg.RAW_FILES:
        (root / "raw" / raw.name).write_bytes(b"%PDF-1.4 synthetic")
    monkeypatch.setattr(gw, "verify_raw_files", presence_only)
    monkeypatch.setattr(wg, "layout_text", lambda path: TEXT)
    monkeypatch.setattr(wg, "pdftotext_version", lambda: "pdftotext version test")
    monkeypatch.setattr(wg, "TABLE_S3_SHA256", SYNTHETIC_DIGEST)
    monkeypatch.setattr(wg, "TABLE_S3_MUTANTS", SYNTHETIC_RECORDS)
    monkeypatch.setattr(wg, "MIN_RESOLVED_FRACTION", 0.5)
    monkeypatch.setattr(gw, "assembly_reference", lambda strain, **_: REFERENCE)
    monkeypatch.setattr(gw, "EXPECTED_SOURCE_ROWS", SYNTHETIC_ROWS)
    monkeypatch.setattr(gw, "EXPECTED_RECORDS", SYNTHETIC_RECORDS)
    monkeypatch.setattr(gw, "MIN_FITNESS", SYNTHETIC_MIN)
    monkeypatch.setattr(gw, "MAX_FITNESS", SYNTHETIC_MAX)
    return root


# --------------------------------------------------------------------------- #
# The declared module constants
# --------------------------------------------------------------------------- #
def test_declared_shape_is_47_released_rows_and_46_records() -> None:
    assert (gw.EXPECTED_SOURCE_ROWS, gw.EXPECTED_RECORDS) == (47, 46)
    assert gw.EXPECTED_REFERENCES == 1
    assert (gw.MIN_FITNESS, gw.MAX_FITNESS) == (0.877108, 1.030120)
    assert gw.N_SAMPLES == 2
    assert gw.PARENT_STRAIN == "BW25113"
    assert gw.DATASET_ROOT_REL == "data/torchcell/growth_wang2015"


def test_the_sourcing_is_wang2015s_own_and_adds_only_the_column_statement() -> None:
    shared = {
        name: value
        for name, value in gw.SOURCED_VALUES.items()
        if name != "no_isoprenol_column"
    }
    assert all(value in wg.SOURCED_VALUES.values() for value in shared.values())
    assert gw.SOURCED_VALUES["n_samples"].value == 2
    assert "two biological replicates" in gw.SOURCED_VALUES["n_samples"].quote
    column = gw.SOURCED_VALUES["no_isoprenol_column"]
    assert column.value == gw.FITNESS_DEFINITION
    assert column.quote == wg.SOURCED_VALUES["growth_inhibition_definition"].quote
    assert column.note is not None and "divides BY it" in column.note


# --------------------------------------------------------------------------- #
# The environment and the phenotypes
# --------------------------------------------------------------------------- #
def test_environment_is_the_stored_datasets_minus_the_isoprenol() -> None:
    plain = gw.environment()
    assert plain.media == YT_2X
    assert plain.temperature is not None and plain.temperature.value == 30.0
    assert plain.duration_hours == 12.0
    assert plain.aerobicity == "aerobic"
    only = plain.perturbations[0]
    assert isinstance(only, EnvironmentPhysicalPerturbation)
    assert (len(plain.perturbations), only.factor) == (1, PhysicalFactor.ph)
    # the isoprenol arm's environment carries one more edit, and only one more
    with_isoprenol = wg.environment()
    assert len(with_isoprenol.perturbations) == len(plain.perturbations) + 1
    assert plain.model_dump_json() != with_isoprenol.model_dump_json()


def test_fitness_is_the_ratio_to_the_parent_of_the_same_column() -> None:
    parent = _row("BW25113", 8.0, 0.10)
    phenotype = gw.fitness_phenotype(_row(f"BW{D}thrA", 4.0, 0.20), parent)
    assert phenotype.fitness == 0.5
    assert phenotype.fitness_uncertainty == 0.20 / 8.0
    assert phenotype.fitness_uncertainty_type is UncertaintyType.sample_sd
    assert phenotype.n_samples == 2
    assert phenotype.sample_unit is SampleUnit.biological_replicate
    expected_se = 0.5 * math.sqrt(
        (0.20 / math.sqrt(2) / 4.0) ** 2 + (0.10 / math.sqrt(2) / 8.0) ** 2
    )
    assert phenotype.fitness_se == pytest.approx(expected_se)
    # the propagated SE carries the parent's spread too, so it is never the optimistic
    # parent-conditioned derivation
    assert phenotype.fitness_uncertainty is not None
    assert phenotype.fitness_se is not None
    assert phenotype.fitness_se > phenotype.fitness_uncertainty / math.sqrt(2)


def test_fitness_refuses_a_propagated_se_below_the_conditioned_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gw, "standard_error", lambda sd: 0.0)
    with pytest.raises(RuntimeError, match="smaller than the parent-conditioned"):
        gw.fitness_phenotype(_row(f"BW{D}thrA", 4.0, 0.20), _row("BW25113", 8.0, 0.10))


def test_reference_fitness_is_one_with_the_parents_own_relative_spread() -> None:
    reference = gw.reference_phenotype(_row("BW25113", 8.0, 0.16))
    assert reference.fitness == 1.0
    assert reference.fitness_uncertainty == 0.02
    assert reference.n_samples == 2
    assert reference.fitness_uncertainty_type is UncertaintyType.sample_sd


def test_check_fitness_range_refuses_a_clamp_a_zero_sd_and_a_drifted_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _row("BW25113", 8.0, 0.10)
    mutants = [_row(f"BW{D}thrA", 4.0, 0.20), _row(f"BW{D}thrL", 8.0, 0.30)]
    monkeypatch.setattr(gw, "MIN_FITNESS", 0.5)
    monkeypatch.setattr(gw, "MAX_FITNESS", 1.0)
    span = gw.check_fitness_range(mutants, parent)
    assert (span.n_records, span.minimum, span.maximum) == (2, 0.5, 1.0)
    assert (span.n_non_positive, span.n_zero_sd) == (0, 0)
    assert span.median == 0.75

    with pytest.raises(RuntimeError, match="non-positive"):
        gw.check_fitness_range([_row(f"BW{D}thrA", 0.0, 0.20)], parent)
    with pytest.raises(RuntimeError, match="exactly 0"):
        gw.check_fitness_range([_row(f"BW{D}thrA", 4.0, 0.0)], parent)
    monkeypatch.setattr(gw, "MAX_FITNESS", 2.0)
    with pytest.raises(RuntimeError, match="the module declares"):
        gw.check_fitness_range(mutants, parent)


def test_build_experiment_and_reference_round_trip() -> None:
    parent = _row("BW25113", 8.0, 0.10)
    identity = wg.StrainIdentity(
        row=_row(f"BW{D}thrA", 4.0, 0.20),
        locus_tag="BW25113_0002",
        symbol="thrA",
        keio_token="JW0001",
        mdt_family="ABC",
        keio_accession="JW0001",
    )
    env = gw.environment()
    experiment = gw.build_experiment("growth_wang2015", identity, parent, env)
    assert isinstance(experiment, BacterialFitnessExperiment)
    genotype = experiment.genotype
    assert not isinstance(genotype, list)
    assert [p.systematic_gene_name for p in genotype.perturbations] == ["BW25113_0002"]
    assert experiment.phenotype.fitness == 0.5
    reference = gw.build_reference("growth_wang2015", REFERENCE, env, parent)
    assert isinstance(reference, BacterialFitnessExperimentReference)
    assert reference.phenotype_reference.fitness == 1.0
    assert reference.genome_reference.assembly_accession == REFERENCE.assembly_accession


# --------------------------------------------------------------------------- #
# End-to-end build on the synthetic tables
# --------------------------------------------------------------------------- #
def test_process_builds_one_record_per_mutant_with_the_parent_reference(
    synthetic: Path, bw25113: EcoliK12BW25113Genome
) -> None:
    dataset = gw.GrowthWang2015Dataset(root=str(synthetic), ecoli_genome=bw25113)
    assert len(dataset) == SYNTHETIC_RECORDS
    items = [dataset.transform_item(dataset[i]) for i in range(len(dataset))]
    assert {len(i["experiment"].genotype.perturbations) for i in items} == {1}
    assert {i["reference"].phenotype_reference.fitness for i in items} == {1.0}
    assert sorted(i["experiment"].phenotype.fitness for i in items) == [0.5, 0.75, 1.0]
    assert {i["publication"].doi for i in items} == {wg.PAPER_DOI}
    # the environment is one object shared by every record and the reference
    assert len({i["experiment"].environment.model_dump_json() for i in items}) == 1
    dataset.close_lmdb()

    preprocess = Path(dataset.preprocess_dir)
    drops = json.loads((preprocess / "dropped_records.json").read_text())
    assert (drops["table_rows"], drops["source_records"]) == (
        SYNTHETIC_ROWS,
        SYNTHETIC_RECORDS,
    )
    assert drops["kept_records"] == SYNTHETIC_RECORDS
    assert drops["dropped_records"] == 0
    assert drops["reference_rows"] == ["BW25113"]
    extraction = json.loads((preprocess / "extraction.json").read_text())
    assert extraction["column"] == gw.COLUMN
    assert extraction["n_samples"] == 2
    assert extraction["parent_od600_without"] == 8.0
    assert extraction["fitness_range"]["n_non_positive"] == 0
    not_stored = json.loads((preprocess / "not_stored.json").read_text())
    assert not_stored["issue"] == 826
    assert not_stored["items"][0]["n_values"] == SYNTHETIC_RECORDS + 1
    sourced = json.loads((preprocess / "sourced_values.json").read_text())
    assert sourced["no_isoprenol_column"]["value"] == gw.FITNESS_DEFINITION
    frame = pd.read_csv(preprocess / "no_isoprenol.csv")
    assert len(frame) == SYNTHETIC_RECORDS + 1
    assert frame["fitness"].tolist()[0] == 1.0
    assert (preprocess / "identifier_reconciliation.json").exists()


def test_process_refuses_a_table_whose_row_count_moved(
    synthetic: Path, bw25113: EcoliK12BW25113Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gw, "EXPECTED_SOURCE_ROWS", SYNTHETIC_ROWS + 1)
    with pytest.raises(wg.TableExtractionError, match="the module declares"):
        gw.GrowthWang2015Dataset(root=str(synthetic), ecoli_genome=bw25113)


def test_process_refuses_a_digest_other_than_the_pin(
    synthetic: Path, bw25113: EcoliK12BW25113Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wg, "TABLE_S3_SHA256", "0" * 64)
    with pytest.raises(wg.TableExtractionError, match="parsed Table S3 sha256"):
        gw.GrowthWang2015Dataset(root=str(synthetic), ecoli_genome=bw25113)


def test_verify_build_passes_every_level_on_the_fitness_gate(
    synthetic: Path, bw25113: EcoliK12BW25113Genome
) -> None:
    dataset = gw.GrowthWang2015Dataset(root=str(synthetic), ecoli_genome=bw25113)
    dataset.close_lmdb()
    report = gw.verify_build(
        str(synthetic), genome=bw25113, expected_count=SYNTHETIC_RECORDS
    )
    assert [(r.level, r.name) for r in report.results if not r.passed] == []
    assert report.passed
    names = [r.name for r in report.results]
    assert "se_is_not_the_conditioned_derivation" in names
    assert "reference_one" in names


def test_l2_rule_catches_a_stored_se_below_the_conditioned_derivation() -> None:
    def record(se: float) -> dict[str, object]:
        return {
            "experiment": {
                "phenotype": {
                    "fitness_se": se,
                    "fitness_uncertainty": 0.08,
                    "n_samples": 2,
                }
            }
        }

    honest = gw.l2_se_is_never_the_conditioned_one([record(0.08)])
    assert honest.passed is True
    optimistic = gw.l2_se_is_never_the_conditioned_one([record(0.001)])
    assert optimistic.passed is False
    assert "below the conditioned" in optimistic.message


# --------------------------------------------------------------------------- #
# Data-gated: the real mirror
# --------------------------------------------------------------------------- #
def _real_rows() -> list[wg.TableS3Row]:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT is not set")
    path = osp.join(data_root, wg.RAW_DIR_REL, "data", wg.SI_PDF)
    if not osp.exists(path):
        pytest.skip(f"{path} is not in the raw mirror")
    text = wg.layout_text(path)
    return wg.parse_table_s3(
        wg.table_section(text, wg.TABLE_S3_MARKER, wg.TABLE_S4_MARKER)
    )


@pytest.mark.data
def test_real_table_s3_releases_47_rows_and_the_declared_fitness_span() -> None:
    rows = _real_rows()
    assert len(rows) == gw.EXPECTED_SOURCE_ROWS
    parent, mutants = wg.split_parent(rows)
    assert (parent.strain, parent.od600_without, parent.sd_without) == (
        "BW25113",
        8.30,
        0.16,
    )
    span = gw.check_fitness_range(mutants, parent)
    assert span.n_records == gw.EXPECTED_RECORDS
    assert span.minimum == pytest.approx(gw.MIN_FITNESS, abs=gw.FITNESS_RANGE_ATOL)
    assert span.maximum == pytest.approx(gw.MAX_FITNESS, abs=gw.FITNESS_RANGE_ATOL)
    assert span.median == pytest.approx(0.956024, abs=1e-6)
    assert (span.n_non_positive, span.n_zero_sd) == (0, 0)


@pytest.mark.data
def test_real_no_isoprenol_column_is_not_the_stored_log2_of_the_isoprenol_arm() -> None:
    """The audit's rank-12 measurement: neither column is recoverable from the stored
    log2, so this dataset is not a second copy of ``env_chemgen_wang2015``.
    """
    rows = _real_rows()
    parent, mutants = wg.split_parent(rows)
    stored = [wg.log2_relative_tolerance(row, parent) for row in mutants]
    plain = [math.log2(row.od600_without / parent.od600_without) for row in mutants]
    assert all(value not in {row.od600_without for row in mutants} for value in stored)
    assert max(abs(a - b) for a, b in zip(plain, stored, strict=True)) > 0.6


@pytest.mark.data
def test_real_dev_store_holds_46_records_and_its_declared_ledger() -> None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT is not set")
    root = osp.join(data_root, gw.DATASET_ROOT_REL)
    drops_path = Path(root, "preprocess", "dropped_records.json")
    if not drops_path.exists():
        pytest.skip(f"{root} is not built")
    drops = json.loads(drops_path.read_text())
    assert drops["kept_records"] == gw.EXPECTED_RECORDS
    assert drops["dropped_records"] == 0
    extraction = json.loads(Path(root, "preprocess", "extraction.json").read_text())
    assert extraction["table_s3_sha256"] == wg.TABLE_S3_SHA256
    assert extraction["fitness_range"]["median"] == pytest.approx(0.956024, abs=1e-6)


# --------------------------------------------------------------------------- #
# download() and the genome guard
# --------------------------------------------------------------------------- #
def test_download_links_every_pinned_mirror_file_and_refuses_an_absent_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pins are ``wang2015``'s, so the mirror is deposited through its depositor."""
    staging = tmp_path / "staging"
    staging.mkdir()
    sources: dict[str, Path] = {}
    for index, raw in enumerate(wg.RAW_FILES):
        path = staging / raw.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic {index}".encode())
        sources[raw.name] = path
    pinned = tuple(
        raw.model_copy(
            update={
                "sha256": hashlib.sha256(sources[raw.name].read_bytes()).hexdigest()
            }
        )
        for raw in wg.RAW_FILES
    )
    monkeypatch.setattr(wg, "RAW_FILES", pinned)
    data_root = tmp_path / "root"
    wg.deposit_raw_mirror(
        sources={name: path for name, path in sources.items()}, data_root=str(data_root)
    )
    monkeypatch.setenv("DATA_ROOT", str(data_root))

    root = tmp_path / "build"
    (root / "raw").mkdir(parents=True)
    dataset = gw.GrowthWang2015Dataset.__new__(gw.GrowthWang2015Dataset)
    monkeypatch.setattr(
        type(dataset), "raw_dir", property(lambda self: str(root / "raw"))
    )
    dataset.download()
    assert sorted(p.name for p in (root / "raw").iterdir()) == sorted(
        raw.name for raw in pinned
    )
    for raw in pinned:
        assert file_sha256(root / "raw" / raw.name) == raw.sha256

    (data_root / wg.RAW_DIR_REL / pinned[0].mirror_relpath).unlink()
    with pytest.raises(RuntimeError, match="missing from mirror"):
        dataset.download()


def test_the_genome_guard_refuses_an_assembly_other_than_bw25113(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = gw.GrowthWang2015Dataset.__new__(gw.GrowthWang2015Dataset)
    wrong = type("WrongGenome", (), {"ASSEMBLY_SET": "ecoli_K12_MG1655_ASM584v2"})()
    dataset.ecoli_genome = wrong
    with pytest.raises(ValueError, match="needs the ecoli_K12_BW25113_ASM75055v1"):
        dataset._genome()
