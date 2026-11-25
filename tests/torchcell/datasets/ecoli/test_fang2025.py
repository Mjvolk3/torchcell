# tests/torchcell/datasets/ecoli/test_fang2025.py
# [[tests.torchcell.datasets.ecoli.test_fang2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_fang2025.py
"""The Fang 2025 CRISPRi-FACS loader (``torchcell.datasets.ecoli.fang2025``).

Synthetic tests (run everywhere) build every input in ``tmp_path``. The build uses the
real ``EcoliK12MG1655Genome`` over the synthetic MG1655 assembly of
``tests/torchcell/sequence/genome/_bacterial_fixtures.py`` (thrL b0001, thrA b0002,
thrW b0003 tRNA, pseudogene yaaP b0004, proB b0005, proC b0006), served through a
stubbed ``resolve`` with the network refused, and replaces the loader's
``verify_raw_files`` with a presence check (the synthetic workbooks cannot carry the
real pins; those are asserted by the refusal test and the data-gated tests).

The synthetic release is read-count driven, because the ONE retention rule is a read
floor: every expected count below falls out of the counts in ``ROUND_1_READS`` and
``ROUND_2_READS`` rather than being asserted separately. The loader reads no genome:
``b4590``, which the synthetic MG1655 assembly does not carry, is stored as released
and the verifier's genome-reference rows report it.

    guide            cluster members   round 1 (tra/bs/as)   round 2 (bs/as)
    thrLb0001_10     b0001             100/100/50  keep      100/100  keep
    thrLb0001_25     b0001             100/100/200 keep      100/10   floor
    thrAb0002_12     b0002             100/100/5   floor     100/100  keep
    proBb0005_60     b0005, b0006      100/100/100 keep      100/100  keep
    yaaPb0004_45     b0004             100/100/40  keep      100/5    floor
    ybfKb4590_5      b4590 (no locus)  100/100/60  keep      100/100  keep
    thrWb0003_31     b0003             10/100/100  floor     10/100   floor

Round 1: 5 rows clear the floor, so 5 records over 6 perturbations (``proBb0005_60``
carries two, one per cluster member). Round 2: 4 rows clear the floor, so 4 records,
each carrying ONE EXTRA knockdown for the synthetic pcnBi background (``yaaPb0004_45``
on b0004), giving 9 perturbations. Nine records, 15 perturbations, 6 distinct genes.
``yaaPb0004_45`` is the background guide AND is below the round-2 floor on purpose, so
no record repeats a gene.

Both Methods equations are written into the synthetic workbook by computing them from
the synthetic read counts, which is what lets the loader's own equation check and its
floor cross-tabulation be exercised rather than bypassed.

Data-gated tests (``@pytest.mark.data``) read the real raw mirror and the built dev-tree
LMDB under ``$DATA_ROOT`` (they never build it): the manifest pins, the provenance audit
of every sourced value, the per-round arithmetic, and the L0-L4 verifier.
"""

from __future__ import annotations

import json
import math
import os
import os.path as osp
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import openpyxl
import pytest

import torchcell.datasets.ecoli.fang2025 as m
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    MG1655_GAF,
    MG1655_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.datamodels.media import M9_MODIFIED_FANG2025
from torchcell.datamodels.schema import (
    AssayType,
    ConcentrationUnit,
    CrisprInterferencePerturbation,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    MeasurementType,
    ReferenceGenome,
    SampleUnit,
    SmallMoleculePerturbation,
    UncertaintyType,
)
from torchcell.datasets.ecoli.wang2018 import CLUSTER_COLUMNS, LIBRARY_COLUMNS
from torchcell.literature.manifest import (
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
)
from torchcell.sequence.genome.ecoli.k12 import MG1655_ASSEMBLY, EcoliK12MG1655Genome
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

# --------------------------------------------------------------------------- #
# The synthetic release
# --------------------------------------------------------------------------- #
PROTEIN_CLUSTERS: list[tuple[str, str]] = [
    ("thrLb0001", "thrLb0001"),
    ("thrAb0002", "thrAb0002"),
    ("proBb0005", "proBb0005,proCb0006"),
    ("yaaPb0004", "yaaPb0004"),
    ("ybfKb4590", "ybfKb4590"),
]
NCRNA_CLUSTERS: list[tuple[str, str]] = [("thrWb0003", "thrWb0003")]
N_SYNTHETIC_CLUSTERS = len(PROTEIN_CLUSTERS) + len(NCRNA_CLUSTERS)
N_SYNTHETIC_MEMBERS = 7

GUIDES: dict[str, str] = {
    "thrLb0001_10": "AAAACGCGCGTCACGCGTCC",
    "thrLb0001_25": "AAAGCGCGCGTCACGCGTCC",
    "thrAb0002_12": "AACGCGCGCGTCACGCGTCC",
    "proBb0005_60": "AAGGCGCGCGTCACGCGTCC",
    "yaaPb0004_45": "AATGCGCGCGTCACGCGTCC",
    "ybfKb4590_5": "ACAGCGCGCGTCACGCGTCC",
    "thrWb0003_31": "ACCGCGCGCGTCACGCGTCC",
    "NC_1": "ACGGCGCGCGTCACGCGTCC",
    "NC_2": "ACTGCGCGCGTCACGCGTCC",
}
N_SYNTHETIC_TARGETING = 7
N_SYNTHETIC_CONTROLS = 2

#: ``guide -> (transformation, before sorting, after sorting)``.
ROUND_1_READS: dict[str, tuple[int, int, int]] = {
    "thrLb0001_10": (100, 100, 50),
    "thrLb0001_25": (100, 100, 200),
    "thrAb0002_12": (100, 100, 5),
    "proBb0005_60": (100, 100, 100),
    "yaaPb0004_45": (100, 100, 40),
    "ybfKb4590_5": (100, 100, 60),
    "thrWb0003_31": (10, 100, 100),
}
#: ``guide -> (before sorting, after sorting)``.
ROUND_2_READS: dict[str, tuple[int, int]] = {
    "thrLb0001_10": (100, 100),
    "thrLb0001_25": (100, 10),
    "thrAb0002_12": (100, 100),
    "proBb0005_60": (100, 100),
    "yaaPb0004_45": (100, 5),
    "ybfKb4590_5": (100, 100),
    "thrWb0003_31": (10, 100),
}
#: The synthetic pcnBi background: a single-gene guide below the round-2 floor.
SYNTHETIC_BACKGROUND_GUIDE = "yaaPb0004_45"

SCREEN_RAW = m.RawFile(
    moesm=8, data_number=6, description="synthetic source data", sha256="0" * 64
)
CLUSTERS_RAW = m.RawFile(
    moesm=5, data_number=2, description="synthetic clusters", sha256="1" * 64
)
LIBRARY_RAW = m.RawFile(
    moesm=6, data_number=3, description="synthetic library", sha256="2" * 64
)

ROUND_1 = m.Round(
    round_id="round_1_cf",
    sheet="Figure 1",
    panel="Figure 1d",
    host_strain="CF",
    libraries=("transformation", "before_sorting", "after_sorting"),
    id_column=10,
    read_columns=(11, 12, 13),
    fitness_column=16,
    excluded_columns=(18, 19, 20),
    fitness_all_column=21,
    carries_background_knockdown=False,
    floor_kept_rows=5,
    kept_records=5,
)
ROUND_2 = m.Round(
    round_id="round_2_pcnbi",
    sheet="Figure 5",
    panel="Figure 5c",
    host_strain="pcnBi",
    libraries=("before_sorting", "after_sorting"),
    id_column=10,
    read_columns=(11, 12),
    fitness_column=15,
    excluded_columns=(17, 18),
    fitness_all_column=19,
    carries_background_knockdown=True,
    floor_kept_rows=4,
    kept_records=4,
)
SYNTHETIC_ROUNDS = (ROUND_1, ROUND_2)
N_SYNTHETIC_RECORDS = ROUND_1.kept_records + ROUND_2.kept_records
N_SYNTHETIC_PERTURBATIONS = 15
N_SYNTHETIC_GENES = 6

REFERENCE = ReferenceGenome(
    species="Escherichia coli",
    strain="MG1655",
    assembly_set="ecoli_K12_MG1655_ASM584v2",
    assembly_accession="GCA_000005845.2",
)


def _sha256_of(path: Path) -> str:
    """Hex sha256 of a synthetic workbook, so its RawFile pin is its real bytes."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sheets(
    path: Path, sheets: Sequence[tuple[str, Sequence[Sequence[Any]]]]
) -> None:
    """Write one workbook, one sheet per ``(name, rows)`` pair."""
    workbook = openpyxl.Workbook()
    default = workbook.active
    assert default is not None
    workbook.remove(default)
    for name, rows in sheets:
        sheet = workbook.create_sheet(name)
        for row in rows:
            sheet.append(list(row))
    workbook.save(path)


def write_clusters(path: Path) -> None:
    """A synthetic Supplementary Data 2 (Wang 2018's) with its two sheets."""
    _write_sheets(
        path,
        [
            ("protein-coding genes", [CLUSTER_COLUMNS, *PROTEIN_CLUSTERS]),
            ("ncRNA-coding genes", [CLUSTER_COLUMNS, *NCRNA_CLUSTERS]),
        ],
    )


def write_library(path: Path) -> None:
    """A synthetic Supplementary Data 3 (Wang 2018's)."""
    _write_sheets(path, [("sheet1", [LIBRARY_COLUMNS, *GUIDES.items()])])


def _round_rows(
    reads: Mapping[str, tuple[int, ...]], round_: m.Round, floor: int = 20
) -> list[list[Any]]:
    """One round's sheet body, with both Methods equations computed from the reads.

    The ``Fitness`` column is populated exactly where every library clears ``floor``,
    and the value otherwise goes into the per-library excluded column, which is the
    split the released workbook carries and the loader cross-tabulates against.
    """
    width = round_.fitness_all_column + 1
    totals = [
        sum(values[position] for values in reads.values())
        for position in range(len(round_.libraries))
    ]
    before = round_.libraries.index("before_sorting")
    after = round_.libraries.index("after_sorting")
    body: list[list[Any]] = []
    for guide, values in reads.items():
        row: list[Any] = [None] * width
        row[round_.id_column] = guide
        for position, column in enumerate(round_.read_columns):
            row[column] = values[position]
        normalized_before = (values[before] + 1) / totals[before]
        normalized_after = (values[after] + 1) / totals[after]
        row[round_.fitness_column - 2] = normalized_before
        row[round_.fitness_column - 1] = normalized_after
        fitness = math.log2(normalized_after / normalized_before)
        row[round_.fitness_all_column] = fitness
        if all(count >= floor for count in values):
            row[round_.fitness_column] = fitness
        else:
            low = next(
                position for position, count in enumerate(values) if count < floor
            )
            row[round_.excluded_columns[low]] = fitness
        body.append(row)
    return body


def write_screen(path: Path) -> None:
    """A synthetic Supplementary Data 6: two panel rows, a header row, then the guides."""
    sheets: list[tuple[str, Sequence[Sequence[Any]]]] = []
    for round_, reads in ((ROUND_1, ROUND_1_READS), (ROUND_2, ROUND_2_READS)):
        width = round_.fitness_all_column + 1
        panels: list[Any] = [None] * width
        panels[round_.id_column - 9] = round_.panel
        header: list[Any] = [None] * width
        header[round_.id_column] = "sgRNA-ID"
        sheets.append((round_.sheet, [panels, header, *_round_rows(reads, round_)]))
    _write_sheets(path, sheets)


def write_raw(raw: Path) -> None:
    """Every synthetic file the loader reads, under ``raw``."""
    raw.mkdir(parents=True, exist_ok=True)
    write_clusters(raw / CLUSTERS_RAW.filename)
    write_library(raw / LIBRARY_RAW.filename)
    write_screen(raw / SCREEN_RAW.filename)


@pytest.fixture
def synthetic_library_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the released-size check at the synthetic library's seven guides."""
    monkeypatch.setattr(m, "N_TARGETING_GUIDES", N_SYNTHETIC_TARGETING)


@pytest.fixture
def mg1655(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12MG1655Genome:
    """The real MG1655 class over the synthetic assembly; network refused."""
    files = write_assembly(tmp_path / "tier", MG1655_ASSEMBLY, MG1655_LOCI, MG1655_GAF)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    root = tmp_path / "mg1655"
    root.mkdir()
    return EcoliK12MG1655Genome(genome_root=str(root), overwrite=False)


@pytest.fixture
def presence_only_pins(monkeypatch: pytest.MonkeyPatch) -> list[Mapping[str, str]]:
    """Replace the build-time byte check with a presence check that records the pins."""
    calls: list[Mapping[str, str]] = []

    def record(raw_dir: str, pins: Mapping[str, str]) -> None:
        missing = [f for f in pins if not osp.exists(osp.join(raw_dir, f))]
        if missing:
            raise FileNotFoundError(f"pinned raw files absent: {missing}")
        calls.append(dict(pins))

    monkeypatch.setattr(m, "verify_raw_files", record)
    return calls


@pytest.fixture
def synthetic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mg1655: EcoliK12MG1655Genome,
    presence_only_pins: list[Mapping[str, str]],
) -> Path:
    """A dataset root whose ``raw/`` holds the synthetic release."""
    root = tmp_path / "data/torchcell/crispri_guide_ffa_enrichment_fang2025"
    write_raw(root / "raw")
    monkeypatch.setattr(m, "ROUNDS", SYNTHETIC_ROUNDS)
    monkeypatch.setattr(m, "RAW_FILES", (SCREEN_RAW,))
    monkeypatch.setattr(m, "SCREEN_FILE", SCREEN_RAW)
    monkeypatch.setattr(m, "WANG_RAW_FILES", (CLUSTERS_RAW, LIBRARY_RAW))
    monkeypatch.setattr(m, "WANG_CLUSTERS_FILE", CLUSTERS_RAW)
    monkeypatch.setattr(m, "WANG_LIBRARY_FILE", LIBRARY_RAW)
    monkeypatch.setattr(m, "N_CLUSTERS", N_SYNTHETIC_CLUSTERS)
    monkeypatch.setattr(m, "N_CLUSTER_MEMBERS", N_SYNTHETIC_MEMBERS)
    monkeypatch.setattr(m, "N_TARGETING_GUIDES", N_SYNTHETIC_TARGETING)
    monkeypatch.setattr(m, "N_CONTROL_GUIDES", N_SYNTHETIC_CONTROLS)
    monkeypatch.setattr(m, "EXPECTED_RECORDS", N_SYNTHETIC_RECORDS)
    monkeypatch.setattr(m, "EXPECTED_PERTURBATIONS", N_SYNTHETIC_PERTURBATIONS)
    monkeypatch.setattr(m, "EXPECTED_GENES", N_SYNTHETIC_GENES)
    monkeypatch.setattr(m, "host_reference", lambda *a, **k: REFERENCE)
    monkeypatch.setattr(
        m,
        "ROUND_TWO_BACKGROUND_GUIDE",
        SourcedValue(
            value=SYNTHETIC_BACKGROUND_GUIDE,
            quote=m.ROUND_TWO_BACKGROUND_GUIDE.quote,
            provenance=m.ROUND_TWO_BACKGROUND_GUIDE.provenance,
        ),
    )
    return root


@pytest.fixture
def built(synthetic: Path) -> m.CrispriGuideFfaEnrichmentFang2025Dataset:
    """The synthetic build; the loader takes no genome."""
    return m.CrispriGuideFfaEnrichmentFang2025Dataset(root=str(synthetic))


def _records(
    dataset: m.CrispriGuideFfaEnrichmentFang2025Dataset,
) -> list[dict[str, Any]]:
    """Every stored record, with the LMDB handle closed before anything re-reads it."""
    out = [dataset[index] for index in range(len(dataset))]
    dataset.close_lmdb()
    return out


# --------------------------------------------------------------------------- #
# Reading one round
# --------------------------------------------------------------------------- #
def test_read_round_reproduces_both_equations_and_splits_on_the_read_floor(
    tmp_path: Path, synthetic_library_size: None
) -> None:
    """Round 1: 5 of 7 rows clear the floor, and both equations reproduce exactly."""
    path = tmp_path / SCREEN_RAW.filename
    write_screen(path)
    rows = m.read_round(path, ROUND_1)
    assert rows.round_id == "round_1_cf"
    assert rows.source_rows == N_SYNTHETIC_TARGETING
    assert list(rows.kept["guide_id"]) == [
        "thrLb0001_10",
        "thrLb0001_25",
        "proBb0005_60",
        "yaaPb0004_45",
        "ybfKb4590_5",
    ]
    assert rows.excluded_rows == 2
    assert rows.excluded_by_library == {
        "transformation": 1,
        "before_sorting": 0,
        "after_sorting": 1,
    }
    assert rows.total_reads == {
        "transformation": 610,
        "before_sorting": 700,
        "after_sorting": 555,
    }
    assert rows.max_equation_1_error < 1e-12
    assert rows.max_equation_2_error < 1e-12


def test_read_round_two_reads_two_libraries_and_keeps_four(
    tmp_path: Path, synthetic_library_size: None
) -> None:
    """Round 2 has no transformation library, so its floor is a two-way test."""
    path = tmp_path / SCREEN_RAW.filename
    write_screen(path)
    rows = m.read_round(path, ROUND_2)
    assert list(rows.kept["guide_id"]) == [
        "thrLb0001_10",
        "thrAb0002_12",
        "proBb0005_60",
        "ybfKb4590_5",
    ]
    assert rows.excluded_by_library == {"before_sorting": 1, "after_sorting": 2}
    assert set(rows.total_reads) == {"before_sorting", "after_sorting"}


def test_read_round_refuses_a_fitness_that_is_not_the_equation(
    tmp_path: Path, synthetic_library_size: None
) -> None:
    """A released value that is not log2(normalized AS / normalized BS) stops the read."""
    path = tmp_path / SCREEN_RAW.filename
    write_screen(path)
    workbook = openpyxl.load_workbook(path)
    sheet = workbook[ROUND_1.sheet]
    sheet.cell(row=3, column=ROUND_1.fitness_all_column + 1).value = 99.0
    workbook.save(path)
    with pytest.raises(ValueError, match="Methods equations do not reproduce"):
        m.read_round(path, ROUND_1)


def test_read_round_refuses_a_floor_that_disagrees_with_the_fitness_column(
    tmp_path: Path, synthetic_library_size: None
) -> None:
    """A qualified value on a row below the floor means the released split moved."""
    path = tmp_path / SCREEN_RAW.filename
    write_screen(path)
    workbook = openpyxl.load_workbook(path)
    sheet = workbook[ROUND_1.sheet]
    # thrAb0002_12 is row 5 and is below the floor on 'after_sorting'.
    sheet.cell(row=5, column=ROUND_1.fitness_column + 1).value = -1.0
    workbook.save(path)
    with pytest.raises(ValueError, match="floor disagrees with the released Fitness"):
        m.read_round(path, ROUND_1)


def test_read_round_refuses_a_library_that_is_not_the_released_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sheet whose guide count is not the library's is a different release."""
    path = tmp_path / SCREEN_RAW.filename
    write_screen(path)
    monkeypatch.setattr(m, "N_TARGETING_GUIDES", N_SYNTHETIC_TARGETING + 1)
    with pytest.raises(ValueError, match="guide rows, the library is"):
        m.read_round(path, ROUND_1)


def test_read_round_refuses_a_repeated_guide_id(
    tmp_path: Path, synthetic_library_size: None
) -> None:
    """One sheet row per guide; a repeat would make two records of one measurement."""
    path = tmp_path / SCREEN_RAW.filename
    write_screen(path)
    workbook = openpyxl.load_workbook(path)
    sheet = workbook[ROUND_1.sheet]
    sheet.cell(row=4, column=ROUND_1.id_column + 1).value = "thrLb0001_10"
    workbook.save(path)
    with pytest.raises(ValueError, match="repeats a guide id"):
        m.read_round(path, ROUND_1)


# --------------------------------------------------------------------------- #
# The objects a record is made of
# --------------------------------------------------------------------------- #
def test_screen_environment_is_the_one_culture_both_arms_share() -> None:
    """One environment: BS and AS differ by the sort gate, not by the medium."""
    environment = m.screen_environment()
    assert environment.media == M9_MODIFIED_FANG2025
    assert environment.temperature is not None and environment.temperature.value == 30.0
    assert environment.duration_hours == 40.0
    assert environment.aerobicity == "aerobic"
    (inducer,) = environment.perturbations
    assert isinstance(inducer, SmallMoleculePerturbation)
    # IPTG is not in the shared compound table, so it carries a typed InChIKey gap. It
    # is deliberately NOT added: the resolver returns the TABLE's canonical name, so a
    # row for it would rename the IPTG compound node of the already-served Foo 2014
    # records, which is the full-rebuild case rather than an additive change.
    assert inducer.compound.name == "IPTG"
    assert inducer.compound.inchikey is None
    assert [g.field for g in inducer.compound.provenance_gaps] == ["inchikey"]
    assert inducer.concentration.value == 1.0
    assert inducer.concentration.unit is ConcentrationUnit.millimolar


def test_guide_phenotype_is_a_signed_log2_sort_enrichment() -> None:
    """The typed axes: a log2 ratio read off a biosensor sort, not a growth selection."""
    phenotype = m.guide_phenotype(ROUND_1, -6.25)
    assert phenotype.measurement_type is MeasurementType.log2_ratio
    assert phenotype.assay_type is AssayType.biosensor_readout
    assert phenotype.environment_response == -6.25
    assert phenotype.n_samples == 1
    assert phenotype.sample_unit is SampleUnit.pooled
    assert phenotype.screen_id == "round_1_cf"
    assert {gap.field for gap in phenotype.provenance_gaps} == {
        "environment_response_se",
        "environment_response_uncertainty",
        "environment_response_uncertainty_type",
    }


def test_reference_is_the_unsorted_library_at_exactly_zero() -> None:
    """A guide the sort does not move has fitness log2(1) = 0 by equation (2)."""
    reference = m.build_reference("fang", ROUND_2, REFERENCE)
    assert reference.phenotype_reference.environment_response == 0.0
    assert reference.phenotype_reference.screen_id == "round_2_pcnbi"
    assert reference.environment_reference == m.screen_environment()
    assert reference.genome_reference == REFERENCE


def test_host_background_keeps_the_lesion_as_a_verbatim_statement() -> None:
    """``alleles`` is empty because the paper prints no locus tag for ``fadE``."""
    background = m.host_background()
    assert background.name == "CF"
    assert background.alleles == []
    assert background.genotype_statement is not None
    assert "fadE deletion" in background.genotype_statement
    assert background.parents == ["E. coli MG1655(DE3)"]
    assert background.provenance is not None
    assert len(background.provenance) == 2


def test_build_genotype_adds_the_background_knockdown_only_when_asked() -> None:
    """Round 1 genotypes are single, round 2 genotypes carry the pcnBi repression too."""
    targets = (("b0005", "proB"), ("b0006", "proC"))
    plain = m.build_genotype(targets, GUIDES["proBb0005_60"])
    assert [p.systematic_gene_name for p in plain.perturbations] == ["b0005", "b0006"]
    doubled = m.build_genotype(
        targets, GUIDES["proBb0005_60"], (("b0143", "pcnB", "CAGTGGGTACCAGAACATGG"),)
    )
    assert [p.systematic_gene_name for p in doubled.perturbations] == [
        "b0005",
        "b0006",
        "b0143",
    ]
    knockdowns = [
        p
        for p in doubled.perturbations
        if isinstance(p, CrisprInterferencePerturbation)
    ]
    assert len(knockdowns) == len(doubled.perturbations)
    assert knockdowns[-1].crispr is not None
    assert knockdowns[-1].crispr.guide_sequence == "CAGTGGGTACCAGAACATGG"
    assert all(
        p.crispr is not None and p.crispr.effector == "dCas9" for p in knockdowns
    )


# --------------------------------------------------------------------------- #
# The synthetic build
# --------------------------------------------------------------------------- #
def test_build_writes_one_record_per_kept_guide_round_pair(
    built: m.CrispriGuideFfaEnrichmentFang2025Dataset,
) -> None:
    assert len(built) == N_SYNTHETIC_RECORDS == 9


def test_build_validates_every_record_against_the_schema_classes(
    built: m.CrispriGuideFfaEnrichmentFang2025Dataset,
) -> None:
    for record in _records(built):
        experiment = EnvironmentResponseExperiment.model_validate(record["experiment"])
        reference = EnvironmentResponseExperimentReference.model_validate(
            record["reference"]
        )
        assert experiment.phenotype.measurement_type is MeasurementType.log2_ratio
        assert reference.phenotype_reference.environment_response == 0.0


def test_build_keeps_the_two_rounds_apart_by_screen_id(
    built: m.CrispriGuideFfaEnrichmentFang2025Dataset,
) -> None:
    screens = [r["experiment"]["phenotype"]["screen_id"] for r in _records(built)]
    assert screens == ["round_1_cf"] * 5 + ["round_2_pcnbi"] * 4


def test_build_stores_the_singleton_on_no_locus_as_released(
    built: m.CrispriGuideFfaEnrichmentFang2025Dataset,
) -> None:
    """``ybfKb4590_5`` clears the floor in both rounds and is stored in both, under the
    b-number and symbol Wang 2018 released; no annotation decides it.
    """
    records = _records(built)
    stored = {
        perturbation["systematic_gene_name"]: perturbation["perturbed_gene_name"]
        for record in records
        for perturbation in record["experiment"]["genotype"]["perturbations"]
    }
    assert stored["b4590"] == "ybfK"
    assert set(stored) == {"b0001", "b0002", "b0004", "b0005", "b0006", "b4590"}
    assert len(stored) == N_SYNTHETIC_GENES
    on_b4590 = [
        r["experiment"]["phenotype"]["screen_id"]
        for r in records
        if any(
            p["systematic_gene_name"] == "b4590"
            for p in r["experiment"]["genotype"]["perturbations"]
        )
    ]
    assert on_b4590 == ["round_1_cf", "round_2_pcnbi"]


def test_build_gives_every_round_two_record_the_background_knockdown(
    built: m.CrispriGuideFfaEnrichmentFang2025Dataset,
) -> None:
    """Round 2 is a double knockdown; the extra leaf is the same gene on every record."""
    records = _records(built)
    round_two = [
        r
        for r in records
        if r["experiment"]["phenotype"]["screen_id"] == "round_2_pcnbi"
    ]
    assert len(round_two) == 4
    for record in round_two:
        perturbations = record["experiment"]["genotype"]["perturbations"]
        names = [p["systematic_gene_name"] for p in perturbations]
        assert "b0004" in names
        assert len(names) == len(set(names))
        (background,) = [
            p for p in perturbations if p["systematic_gene_name"] == "b0004"
        ]
        assert (
            background["crispr"]["guide_sequence"] == GUIDES[SYNTHETIC_BACKGROUND_GUIDE]
        )
    round_one = [
        r for r in records if r["experiment"]["phenotype"]["screen_id"] == "round_1_cf"
    ]
    carries_b0004 = [
        r
        for r in round_one
        if any(
            p["systematic_gene_name"] == "b0004"
            for p in r["experiment"]["genotype"]["perturbations"]
        )
    ]
    # Round 1 keeps yaaPb0004_45 as a LIBRARY guide, and that single record is the only
    # round-1 record on b0004; no round-1 record gains the background leaf.
    assert len(carries_b0004) == 1
    assert len(carries_b0004[0]["experiment"]["genotype"]["perturbations"]) == 1


def test_build_carries_one_knockdown_per_cluster_member(
    built: m.CrispriGuideFfaEnrichmentFang2025Dataset,
) -> None:
    """``proBb0005_60`` targets a two-member cluster, so its record carries two leaves."""
    records = _records(built)
    counts = [
        len(r["experiment"]["genotype"]["perturbations"])
        for r in records
        if r["experiment"]["phenotype"]["screen_id"] == "round_1_cf"
    ]
    assert sorted(counts) == [1, 1, 1, 1, 2]
    total = sum(len(r["experiment"]["genotype"]["perturbations"]) for r in records)
    assert total == N_SYNTHETIC_PERTURBATIONS == 15


def test_build_stores_the_released_fitness_verbatim(
    built: m.CrispriGuideFfaEnrichmentFang2025Dataset,
) -> None:
    """The stored value is the workbook's own number, recomputed here from the reads."""
    totals_before = sum(v[1] for v in ROUND_1_READS.values())
    totals_after = sum(v[2] for v in ROUND_1_READS.values())
    reads = ROUND_1_READS["thrLb0001_10"]
    expected = math.log2(
        ((reads[2] + 1) / totals_after) / ((reads[1] + 1) / totals_before)
    )
    first = _records(built)[0]["experiment"]["phenotype"]["environment_response"]
    assert first == pytest.approx(expected, abs=1e-15)


def test_build_writes_the_round_retention_accounting(
    built: m.CrispriGuideFfaEnrichmentFang2025Dataset, synthetic: Path
) -> None:
    built.close_lmdb()
    accounting = json.loads(
        (synthetic / "preprocess" / "build_accounting.json").read_text()
    )
    assert accounting["kept_records"] == N_SYNTHETIC_RECORDS
    assert accounting["perturbations"] == N_SYNTHETIC_PERTURBATIONS
    assert accounting["distinct_genes"] == N_SYNTHETIC_GENES
    assert accounting["source_rows"] == 2 * N_SYNTHETIC_TARGETING
    assert accounting["read_floor"] == 20
    assert accounting["library_citation_key"] == "wangPooledCRISPRInterference2018"
    assert accounting["strain_background"] == "CF"
    assert "retired_targets_dropped" not in accounting
    assert "reconciliation" not in accounting
    by_round = {r["round_id"]: r for r in accounting["rounds"]}
    assert by_round["round_1_cf"]["dropped_below_read_floor"] == 2
    assert "dropped_retired_target" not in by_round["round_1_cf"]
    assert by_round["round_1_cf"]["kept_records"] == 5
    assert by_round["round_2_pcnbi"]["dropped_below_read_floor"] == 3
    assert by_round["round_2_pcnbi"]["background_knockdowns"] == 4
    assert (synthetic / "preprocess" / "round_retention.csv").exists()


def test_build_verifies_the_pins_of_both_mirrors(
    built: m.CrispriGuideFfaEnrichmentFang2025Dataset,
    presence_only_pins: list[Mapping[str, str]],
) -> None:
    """One ``verify_raw_files`` call covering this paper's file AND Wang 2018's two."""
    built.close_lmdb()
    (pins,) = presence_only_pins
    assert set(pins) == {
        SCREEN_RAW.filename,
        CLUSTERS_RAW.filename,
        LIBRARY_RAW.filename,
    }


def test_round_accounting_refuses_arithmetic_that_does_not_close() -> None:
    """The one drop count must account for every released row."""
    accounting = m.RoundAccounting(
        round_id="round_1_cf",
        sheet="Figure 1",
        panel="Figure 1d",
        host_strain="CF",
        libraries=["before_sorting", "after_sorting"],
        total_reads={"before_sorting": 10, "after_sorting": 10},
        source_rows=7,
        dropped_below_read_floor=2,
        dropped_by_library={"before_sorting": 1, "after_sorting": 1},
        kept_records=6,
        perturbations=6,
        background_knockdowns=0,
        max_equation_1_error=0.0,
        max_equation_2_error=0.0,
    )
    with pytest.raises(ValueError, match="not the 6 records written"):
        accounting.check()


def test_background_knockdown_refuses_a_guide_outside_the_library(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the short-form resolution were wrong, the build stops rather than deriving."""
    monkeypatch.setattr(
        m,
        "ROUND_TWO_BACKGROUND_GUIDE",
        SourcedValue(
            value="pcnBb0143_956",
            quote=m.ROUND_TWO_BACKGROUND_GUIDE.quote,
            provenance=m.ROUND_TWO_BACKGROUND_GUIDE.provenance,
        ),
    )
    with pytest.raises(RuntimeError, match="does not resolve"):
        m.CrispriGuideFfaEnrichmentFang2025Dataset(root=str(synthetic))


def test_the_loader_reads_no_genome() -> None:
    import inspect

    assert (
        "ecoli_genome"
        not in inspect.signature(
            m.CrispriGuideFfaEnrichmentFang2025Dataset.__init__
        ).parameters
    )
    assert not hasattr(m, "reconcile_locus_tags")
    assert not hasattr(m, "stored_gene_names")
    assert not hasattr(
        m.CrispriGuideFfaEnrichmentFang2025Dataset, "MIN_RESOLVED_FRACTION"
    )


def test_dataset_exposes_its_schema_classes() -> None:
    dataset = m.CrispriGuideFfaEnrichmentFang2025Dataset.__new__(
        m.CrispriGuideFfaEnrichmentFang2025Dataset
    )
    assert dataset.experiment_class is EnvironmentResponseExperiment
    assert dataset.reference_class is EnvironmentResponseExperimentReference
    with pytest.raises(NotImplementedError, match="builds records in process"):
        dataset.create_experiment()
    assert dataset.preprocess_raw("frame") == "frame"


def test_manifest_sha256_refuses_a_path_it_does_not_carry() -> None:
    from torchcell.literature.manifest import Manifest

    manifest = Manifest(citation_key=m.CITATION_KEY, doi=m.DOI, title=m.TITLE, files=[])
    with pytest.raises(KeyError, match="is not in the Fang 2025 raw-mirror manifest"):
        m.manifest_sha256(manifest, "si/si_data/nope.xlsx")


def test_raw_file_names_the_publisher_object_and_its_cloud_key() -> None:
    assert m.SCREEN_FILE.filename == "41467_2025_58368_MOESM8_ESM.xlsx"
    assert m.SCREEN_FILE.relpath == "si/si_data/41467_2025_58368_MOESM8_ESM.xlsx"
    assert m.SCREEN_FILE.cloud_key == ("PMC11954867.1/41467_2025_58368_MOESM8_ESM.xlsx")
    assert m.SCREEN_FILE.url.startswith("https://pmc-oa-opendata.s3.amazonaws.com/")


def test_deposit_refuses_a_source_whose_bytes_are_not_the_pin(tmp_path: Path) -> None:
    """A mismatching file is refused BEFORE anything is written."""
    wrong = tmp_path / "wrong.xlsx"
    wrong.write_bytes(b"not the workbook")
    with pytest.raises(RuntimeError, match="hashes to"):
        m.deposit_raw_mirror(
            {m.SCREEN_FILE.relpath: wrong}, data_root=str(tmp_path / "root")
        )
    assert not (tmp_path / "root").exists()


def test_deposit_writes_the_mirror_and_its_manifest_from_a_staged_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``sources=`` path copies the file, records its retrieval and is idempotent."""
    workbook = tmp_path / "staged.xlsx"
    write_screen(workbook)
    staged = m.RawFile(
        moesm=8,
        data_number=6,
        description="synthetic source data",
        sha256=_sha256_of(workbook),
    )
    monkeypatch.setattr(m, "RAW_FILES", (staged,))
    monkeypatch.setattr(m, "MIRROR_FILES", (staged,))
    root = m.deposit_raw_mirror(
        {staged.relpath: workbook}, data_root=str(tmp_path / "root")
    )
    assert (root / staged.relpath).exists()
    manifest = m.load_manifest(str(tmp_path / "root"))
    assert manifest.citation_key == m.CITATION_KEY
    (record,) = manifest.files
    assert record.path == staged.relpath
    assert record.sha256 == staged.sha256
    assert record.retrieval is not None
    assert record.retrieval.method is RetrievalMethod.pmc_cloud
    assert record.retrieval.params == {"key": staged.cloud_key}
    assert manifest.provenance_complete is True
    assert m.manifest_sha256(manifest, staged.relpath) == staged.sha256
    # Idempotent: a second deposit of identical bytes leaves the mirror alone.
    assert (
        m.deposit_raw_mirror(
            {staged.relpath: workbook}, data_root=str(tmp_path / "root")
        )
        == root
    )


def test_deposit_refuses_to_overwrite_a_mirror_file_that_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deposited file whose bytes changed is a NEW version, never an overwrite."""
    workbook = tmp_path / "staged.xlsx"
    write_screen(workbook)
    staged = m.RawFile(
        moesm=8,
        data_number=6,
        description="synthetic source data",
        sha256=_sha256_of(workbook),
    )
    monkeypatch.setattr(m, "RAW_FILES", (staged,))
    monkeypatch.setattr(m, "MIRROR_FILES", (staged,))
    root = m.deposit_raw_mirror(
        {staged.relpath: workbook}, data_root=str(tmp_path / "root")
    )
    (root / staged.relpath).write_bytes(b"different bytes")
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        m.deposit_raw_mirror(
            {staged.relpath: workbook}, data_root=str(tmp_path / "root")
        )


def test_download_links_both_mirrors_after_verifying_each_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mg1655: EcoliK12MG1655Genome
) -> None:
    """One ``raw/`` fed by two mirrors, each read under its own citation key."""
    data_root = tmp_path / "root"
    screen = tmp_path / "screen.xlsx"
    write_screen(screen)
    clusters = tmp_path / "clusters.xlsx"
    write_clusters(clusters)
    library = tmp_path / "library.xlsx"
    write_library(library)
    screen_raw = m.RawFile(
        moesm=8, data_number=6, description="screen", sha256=_sha256_of(screen)
    )
    clusters_raw = m.RawFile(
        moesm=5, data_number=2, description="clusters", sha256=_sha256_of(clusters)
    )
    library_raw = m.RawFile(
        moesm=6, data_number=3, description="library", sha256=_sha256_of(library)
    )
    monkeypatch.setattr(m, "RAW_FILES", (screen_raw,))
    monkeypatch.setattr(m, "MIRROR_FILES", (screen_raw,))
    monkeypatch.setattr(m, "WANG_RAW_FILES", (clusters_raw, library_raw))
    m.deposit_raw_mirror({screen_raw.relpath: screen}, data_root=str(data_root))
    monkeypatch.setattr(m, "_data_root", lambda: str(data_root))

    wang_manifest = Manifest(
        citation_key="wangPooledCRISPRInterference2018",
        doi="10.1038/s41467-018-04899-x",
        title="Wang 2018",
        files=[
            ArtifactRecord(
                path=raw.relpath,
                role=ROLE_SI_DATA,
                bytes=source.stat().st_size,
                sha256=raw.sha256,
                source=raw.url,
                original_filename=raw.filename,
            )
            for raw, source in ((clusters_raw, clusters), (library_raw, library))
        ],
    )
    wang_root = tmp_path / "wang"
    for raw, source in ((clusters_raw, clusters), (library_raw, library)):
        destination = wang_root / raw.relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    (wang_root / "manifest.json").write_text(wang_manifest.model_dump_json(indent=2))
    monkeypatch.setattr(m, "wang_raw_mirror_dir", lambda *a, **k: wang_root)
    monkeypatch.setattr(m, "wang_load_manifest", lambda *a, **k: wang_manifest)
    monkeypatch.setattr(
        m,
        "wang_manifest_sha256",
        lambda manifest, relpath: next(
            f.sha256 for f in manifest.files if f.path == relpath
        ),
    )

    dataset = m.CrispriGuideFfaEnrichmentFang2025Dataset.__new__(
        m.CrispriGuideFfaEnrichmentFang2025Dataset
    )
    raw_dir = tmp_path / "dataset" / "raw"
    monkeypatch.setattr(
        type(dataset), "raw_dir", property(lambda self: str(raw_dir)), raising=False
    )
    dataset.download()
    assert sorted(f.name for f in raw_dir.iterdir()) == sorted(
        [screen_raw.filename, clusters_raw.filename, library_raw.filename]
    )
    assert dataset.raw_file_names == [
        screen_raw.filename,
        clusters_raw.filename,
        library_raw.filename,
    ]


def test_download_refuses_a_mirror_missing_the_pinned_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest that pins a file the mirror does not hold stops the build."""
    data_root = tmp_path / "root"
    screen = tmp_path / "screen.xlsx"
    write_screen(screen)
    screen_raw = m.RawFile(
        moesm=8, data_number=6, description="screen", sha256=_sha256_of(screen)
    )
    monkeypatch.setattr(m, "RAW_FILES", (screen_raw,))
    monkeypatch.setattr(m, "MIRROR_FILES", (screen_raw,))
    root = m.deposit_raw_mirror({screen_raw.relpath: screen}, data_root=str(data_root))
    (root / screen_raw.relpath).unlink()
    monkeypatch.setattr(m, "_data_root", lambda: str(data_root))
    dataset = m.CrispriGuideFfaEnrichmentFang2025Dataset.__new__(
        m.CrispriGuideFfaEnrichmentFang2025Dataset
    )
    raw_dir = tmp_path / "dataset" / "raw"
    monkeypatch.setattr(
        type(dataset), "raw_dir", property(lambda self: str(raw_dir)), raising=False
    )
    with pytest.raises(RuntimeError, match="missing from mirror"):
        dataset.download()


def test_verify_build_runs_every_level_on_the_synthetic_store(
    built: m.CrispriGuideFfaEnrichmentFang2025Dataset,
    synthetic: Path,
    mg1655: EcoliK12MG1655Genome,
) -> None:
    """The L0-L4 verifier over the synthetic build, with its report written beside it."""
    built.close_lmdb()
    report = m.verify_build(
        str(synthetic), genome=mg1655, expected_count=N_SYNTHETIC_RECORDS
    )
    failed = [r for r in report.results if not r.passed]
    # b4590 is on no synthetic locus: the genome-reference rows report it under the
    # verdict and never decide it
    assert [r.name for r in failed if r.finding_class.value != "report"] == []
    assert report.passed
    assert failed, "b4590 must be reported by the genome-reference rows"
    assert {int(r.level) for r in report.results} >= {0, 1, 2, 3, 4}
    written = json.loads(
        (synthetic / "preprocess" / "verification_report.json").read_text()
    )
    assert written["results"]
    assert report.summary()


def test_publication_carries_the_doi_and_asserts_no_pubmed_id() -> None:
    publication = m.publication()
    assert publication.doi == m.DOI
    assert publication.doi_url == f"https://doi.org/{m.DOI}"
    assert publication.pubmed_id is None


# --------------------------------------------------------------------------- #
# Data-gated: the real mirror and the built dev-tree store
# --------------------------------------------------------------------------- #
def _data_root() -> str | None:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ.get("DATA_ROOT")


DATA_ROOT = _data_root()
MIRROR = (
    Path(DATA_ROOT) / m.RAW_DIR_REL / "manifest.json" if DATA_ROOT else Path("absent")
)
STORE = (
    Path(DATA_ROOT) / "data/torchcell/crispri_guide_ffa_enrichment_fang2025"
    if DATA_ROOT
    else Path("absent")
)
LIBRARY = Path(DATA_ROOT) / "torchcell-library" if DATA_ROOT else Path("absent")
needs_mirror = pytest.mark.skipif(
    not MIRROR.exists(), reason="the Fang 2025 raw mirror is not deposited here"
)
needs_store = pytest.mark.skipif(
    not (STORE / "processed/lmdb").exists(),
    reason="the Fang 2025 dev-tree LMDB is not built here",
)
needs_library = pytest.mark.skipif(
    not (LIBRARY / m.CITATION_KEY / m.PAPER_MD).exists(),
    reason="the Fang 2025 paper OCR is not mirrored here",
)

#: Every module-level sourced value, which is what the quote audit walks.
SOURCED_VALUES: dict[str, SourcedValue] = {
    name: value for name, value in vars(m).items() if isinstance(value, SourcedValue)
}


@pytest.mark.data
@needs_mirror
def test_the_mirror_pins_the_screen_workbook_to_a_rerunnable_retrieval() -> None:
    """Four deposited files, each with a scriptable ``pmc_cloud`` retrieval and its sha256.

    The screen workbook, and since 2026-11-25 the strain, plasmid and primer tables the
    titer dataset parses (``MIRROR_FILES``), which were quoted from the literature
    mirror before.
    """
    manifest = m.load_manifest(DATA_ROOT)
    assert {f.path for f in manifest.files} == {r.relpath for r in m.MIRROR_FILES}
    assert {r.relpath for r in m.RAW_FILES} <= {r.relpath for r in m.MIRROR_FILES}
    for raw in m.MIRROR_FILES:
        assert m.manifest_sha256(manifest, raw.relpath) == raw.sha256
        record = next(f for f in manifest.files if f.path == raw.relpath)
        assert record.retrieval is not None
        assert record.retrieval.method is RetrievalMethod.pmc_cloud
        assert record.retrieval.params == {"key": raw.cloud_key}
        assert record.retrieval.source_url == raw.url
    assert manifest.provenance_complete is True
    assert any("wangPooledCRISPRInterference2018" in e for e in manifest.si_expected)


@pytest.mark.data
@needs_library
@pytest.mark.parametrize("name", sorted(SOURCED_VALUES))
def test_every_quote_is_still_verbatim_in_the_pinned_ocr(name: str) -> None:
    """The sha256 still matches and the quote is still a substring of those bytes."""
    result = audit_sourced_value(SOURCED_VALUES[name], LIBRARY)
    assert result.passed, result.message


@pytest.mark.data
@needs_store
def test_the_built_store_holds_the_measured_record_count() -> None:
    """15,708 records over 2,628 genes, with the per-round arithmetic balancing."""
    payload = json.loads((STORE / "preprocess/build_accounting.json").read_text())
    accounting = m.BuildAccounting.model_validate(payload)
    accounting.check()
    assert accounting.source_rows == 111342
    assert accounting.kept_records == m.EXPECTED_RECORDS == 15711
    assert accounting.perturbations == m.EXPECTED_PERTURBATIONS == 16191
    assert accounting.distinct_genes == m.EXPECTED_GENES == 2630
    assert accounting.read_floor == 20
    assert accounting.library_citation_key == "wangPooledCRISPRInterference2018"
    by_round = {r.round_id: r for r in accounting.rounds}
    assert by_round["round_1_cf"].kept_records == 15380
    assert by_round["round_2_pcnbi"].kept_records == 331
    assert by_round["round_1_cf"].dropped_by_library["after_sorting"] == 38379
    assert by_round["round_2_pcnbi"].dropped_by_library["after_sorting"] == 55334
    for round_ in accounting.rounds:
        assert round_.max_equation_1_error < 1e-12
        assert round_.max_equation_2_error < 1e-12


@pytest.mark.data
@needs_store
def test_the_built_store_carries_the_round_two_background_knockdown() -> None:
    """Every round-2 record is a double knockdown whose extra leaf is pcnB on b0143."""
    dataset = m.CrispriGuideFfaEnrichmentFang2025Dataset(root=str(STORE))
    try:
        last = dataset[len(dataset) - 1]["experiment"]
        assert last["phenotype"]["screen_id"] == "round_2_pcnbi"
        names = {p["systematic_gene_name"] for p in last["genotype"]["perturbations"]}
        assert "b0143" in names
        assert len(names) == 2
        first = dataset[0]["experiment"]
        assert first["phenotype"]["screen_id"] == "round_1_cf"
        assert len(first["genotype"]["perturbations"]) == 1
    finally:
        dataset.close_lmdb()


@pytest.mark.data
@needs_store
def test_the_built_store_passes_every_verification_level() -> None:
    report = m.verify_build(str(STORE), data_root=DATA_ROOT)
    failed = [r for r in report.results if not r.passed]
    assert [r.name for r in failed if r.finding_class.value != "report"] == []
    assert report.passed


# --------------------------------------------------------------------------- #
# The per-strain FFA titers (#1178): one record per biological replicate culture
# --------------------------------------------------------------------------- #
STRAIN_RAW = m.RawFile(
    moesm=30, data_number=1, description="synthetic strains", sha256="3" * 64
)
PLASMID_RAW = m.RawFile(
    moesm=31, data_number=2, description="synthetic plasmids", sha256="4" * 64
)
PRIMER_RAW = m.RawFile(
    moesm=32, data_number=3, description="synthetic primers", sha256="5" * 64
)

#: The synthetic strain table: one unmodulated control per background, one single
#: knockdown, one double, and one carrying a P_BAD gene addition.
SYNTHETIC_STRAINS: dict[str, str] = {
    "MG": "E. coli MG1655(DE3) ΔfadE",
    "CF": "MG with plasmid pCF",
    "thrL_10": "MG with plasmids pCF and pSg-thrL_10",
    "pcnBi": "MG with plasmid pCF-pcnBi",
    "pcnBi-proB_60": "MG with plasmids pCF-pcnBi and pSg-proB_60",
    "pcnBi-thrA+": "MG with plasmids pCF-pcnBi and Sg-thrA+",
    "CF-": "MG with plasmid pC",
}
SYNTHETIC_PLASMIDS: dict[str, str] = {
    "pCF": "PTrc: dCas9, PT7: tesA’, p15A, CmR",
    "pC": "PTrc: dCas9, p15A, CmR",
    "pCF-pcnBi": "PTrc: dCas9, PT7: tesA’, PLlacO-1: sgRNA thrA_12, p15A, CmR",
    "pSg-thrL_10": "PLlacO-1: sgRNA thrL_10, pBR322, AmpR",
    "pSg-proB_60": "PLlacO-1: sgRNA proB_60, pBR322, AmpR",
    "Sg-thrA+": "PBAD: thrA, ColE1, AmpR",
}
SYNTHETIC_PRIMERS: dict[str, str] = {
    f"{short}-for": f"gcac{GUIDES[long]}"
    for short, long in (
        ("thrL_10", "thrLb0001_10"),
        ("thrA_12", "thrAb0002_12"),
        ("proB_60", "proBb0005_60"),
    )
}

#: The synthetic panels: a four-row block and a two-row block that re-releases CF.
TITER_BLOCK_A = m.TiterBlock(
    sheet="Figure 1",
    panel="Figure 1e",
    header_row=2,
    strain_column=1,
    value_column=2,
    rows=4,
)
TITER_BLOCK_B = m.TiterBlock(
    sheet="Figure S9",
    panel="Figure S9b",
    header_row=2,
    strain_column=1,
    value_column=2,
    rows=3,
)
SYNTHETIC_TITER_BLOCKS = (TITER_BLOCK_A, TITER_BLOCK_B)
#: ``panel -> strain -> the three released cultures``. CF repeats in both panels with
#: the same three numbers, which is what the real release does in six panels.
SYNTHETIC_TITERS: dict[str, dict[str, tuple[float, float, float]]] = {
    "Figure 1e": {
        "CF": (756.0229, 761.0764, 723.361),
        "thrL_10": (223.4, 259.6, 369.3),
        "pcnBi": (2876.923, 2939.514, 3160.207),
        "pcnBi-proB_60": (3511.231, 3763.546, 3649.475),
    },
    "Figure S9b": {
        "CF": (756.0229, 761.0764, 723.361),
        "CF-": (67.13326, 82.81343, 109.0665),
        "pcnBi-thrA+": (4951.984, 5193.743, 5170.569),
    },
}
N_SYNTHETIC_TITER_ROWS = 7
N_SYNTHETIC_TITER_CULTURES = 6
N_SYNTHETIC_TITER_RECORDS = 18


def _titer_body(block: m.TiterBlock) -> list[list[Any]]:
    """One synthetic block: the panel row, the header rows and its strain rows."""
    import statistics

    width = block.value_column + 5
    rows: list[list[Any]] = []
    for index in range(block.header_row):
        lead: list[Any] = [None] * width
        lead[block.strain_column] = block.panel if index == 0 else "FFAs (mg L-1)"
        rows.append(lead)
    header: list[Any] = [None] * width
    header[block.strain_column] = "Strain"
    for offset, label in enumerate(("1", "2", "3", "Mean", "SD")):
        header[block.value_column + offset] = label
    rows.append(header)
    for strain, values in SYNTHETIC_TITERS[block.panel].items():
        row: list[Any] = [None] * width
        row[block.strain_column] = strain
        for offset, value in enumerate(values):
            row[block.value_column + offset] = value
        row[block.value_column + 3] = sum(values) / 3.0
        row[block.value_column + 4] = statistics.stdev(values)
        rows.append(row)
    return rows


def write_titer_workbook(path: Path) -> None:
    """A synthetic Supplementary Data 6 holding only the two titer blocks."""
    _write_sheets(
        path, [(block.sheet, _titer_body(block)) for block in SYNTHETIC_TITER_BLOCKS]
    )


def write_table(path: Path, title: str, table: Mapping[str, str]) -> None:
    """A synthetic ``Supplementary Data`` table in the layout all three share."""
    rows: list[list[Any]] = [
        [None, title, None, None],
        [None, "Name", "Description", "Reference"],
    ]
    rows += [
        [None, name, description, "This study"] for name, description in table.items()
    ]
    _write_sheets(path, [("Supplementary Data", rows)])


@pytest.fixture
def synthetic_titer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    presence_only_pins: list[Mapping[str, str]],
) -> Path:
    """A titer dataset root whose ``raw/`` holds the synthetic release."""
    root = tmp_path / "data/torchcell/ffa_titer_fang2025"
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    write_clusters(raw / CLUSTERS_RAW.filename)
    write_library(raw / LIBRARY_RAW.filename)
    write_titer_workbook(raw / SCREEN_RAW.filename)
    write_table(raw / STRAIN_RAW.filename, "Supplementary Data 1.", SYNTHETIC_STRAINS)
    write_table(raw / PLASMID_RAW.filename, "Supplementary Data 2.", SYNTHETIC_PLASMIDS)
    write_table(raw / PRIMER_RAW.filename, "Supplementary Data 3.", SYNTHETIC_PRIMERS)
    monkeypatch.setattr(m, "SCREEN_FILE", SCREEN_RAW)
    monkeypatch.setattr(m, "STRAIN_FILE", STRAIN_RAW)
    monkeypatch.setattr(m, "PLASMID_FILE", PLASMID_RAW)
    monkeypatch.setattr(m, "PRIMER_FILE", PRIMER_RAW)
    monkeypatch.setattr(
        m, "TITER_RAW_FILES", (SCREEN_RAW, STRAIN_RAW, PLASMID_RAW, PRIMER_RAW)
    )
    monkeypatch.setattr(m, "WANG_RAW_FILES", (CLUSTERS_RAW, LIBRARY_RAW))
    monkeypatch.setattr(m, "WANG_CLUSTERS_FILE", CLUSTERS_RAW)
    monkeypatch.setattr(m, "WANG_LIBRARY_FILE", LIBRARY_RAW)
    monkeypatch.setattr(m, "N_CLUSTERS", N_SYNTHETIC_CLUSTERS)
    monkeypatch.setattr(m, "N_CLUSTER_MEMBERS", N_SYNTHETIC_MEMBERS)
    monkeypatch.setattr(m, "N_TARGETING_GUIDES", N_SYNTHETIC_TARGETING)
    monkeypatch.setattr(m, "N_CONTROL_GUIDES", N_SYNTHETIC_CONTROLS)
    monkeypatch.setattr(m, "TITER_BLOCKS", SYNTHETIC_TITER_BLOCKS)
    monkeypatch.setattr(m, "TITER_SOURCE_ROWS", N_SYNTHETIC_TITER_ROWS)
    monkeypatch.setattr(m, "EXPECTED_TITER_RECORDS", N_SYNTHETIC_TITER_RECORDS)
    monkeypatch.setattr(m, "EXPECTED_TITER_CULTURES", N_SYNTHETIC_TITER_CULTURES)
    monkeypatch.setattr(m, "titer_reference_genome", lambda *a, **k: REFERENCE)
    return root


@pytest.fixture
def built_titer(synthetic_titer: Path) -> m.FfaTiterFang2025Dataset:
    """The synthetic titer build."""
    return m.FfaTiterFang2025Dataset(root=str(synthetic_titer))


def test_titer_blocks_are_the_measured_thirteen_panels() -> None:
    """13 blocks over 144 strain rows, which is 432 released culture values."""
    assert len(m.TITER_BLOCKS) == 13
    assert sum(block.rows for block in m.TITER_BLOCKS) == m.TITER_SOURCE_ROWS == 144
    assert m.TITER_SOURCE_VALUES == 3 * 144 == 432
    assert m.EXPECTED_TITER_RECORDS == 3 * m.EXPECTED_TITER_CULTURES == 306


def test_read_titer_blocks_reads_every_row_and_checks_both_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every strain row comes back with its three cultures, mean and SD."""
    monkeypatch.setattr(m, "TITER_BLOCKS", SYNTHETIC_TITER_BLOCKS)
    monkeypatch.setattr(m, "TITER_SOURCE_ROWS", N_SYNTHETIC_TITER_ROWS)
    path = tmp_path / "titers.xlsx"
    write_titer_workbook(path)
    rows = m.read_titer_blocks(path)
    assert [row.strain for row in rows] == [
        "CF",
        "thrL_10",
        "pcnBi",
        "pcnBi-proB_60",
        "CF",
        "CF-",
        "pcnBi-thrA+",
    ]
    assert {row.medium for row in rows} == {"glycerol"}
    control = rows[0]
    assert control.replicates == SYNTHETIC_TITERS["Figure 1e"]["CF"]
    assert control.released_mean == pytest.approx(746.8201, abs=1e-6)
    assert control.panel == "Figure 1e"


def test_read_titer_blocks_refuses_a_row_whose_released_mean_disagrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Mean the three columns do not reproduce means the columns are not replicates."""
    monkeypatch.setattr(m, "TITER_BLOCKS", (TITER_BLOCK_A,))
    monkeypatch.setattr(m, "TITER_SOURCE_ROWS", TITER_BLOCK_A.rows)
    path = tmp_path / "broken.xlsx"
    body = _titer_body(TITER_BLOCK_A)
    body[TITER_BLOCK_A.header_row + 1][TITER_BLOCK_A.value_column + 3] = 1.0
    _write_sheets(path, [(TITER_BLOCK_A.sheet, body)])
    with pytest.raises(ValueError, match="the released Mean is"):
        m.read_titer_blocks(path)


def test_read_titer_blocks_refuses_a_block_whose_header_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The geometry is pinned, so a shifted panel stops the read."""
    monkeypatch.setattr(
        m, "TITER_BLOCKS", (TITER_BLOCK_A.model_copy(update={"strain_column": 4}),)
    )
    path = tmp_path / "shifted.xlsx"
    write_titer_workbook(path)
    with pytest.raises(ValueError, match="not 'Strain'"):
        m.read_titer_blocks(path)


def test_strain_spec_reads_the_plasmids_the_two_tables_state() -> None:
    """A strain's sentence plus its plasmids' parts give its guides and additions."""
    spec = m.strain_spec("pcnBi-proB_60", SYNTHETIC_STRAINS, SYNTHETIC_PLASMIDS)
    assert spec.background_plasmid == "pCF-pcnBi"
    assert spec.guides == ("thrA_12", "proB_60")
    assert spec.additions == ()
    assert spec.deletions == ()
    assert spec.carries_thioesterase is True
    added = m.strain_spec("pcnBi-thrA+", SYNTHETIC_STRAINS, SYNTHETIC_PLASMIDS)
    assert [(a.gene, a.promoter) for a in added.additions] == [("thrA", "PBAD")]
    control = m.strain_spec("CF-", SYNTHETIC_STRAINS, SYNTHETIC_PLASMIDS)
    assert control.background_plasmid == "pC"
    assert control.carries_thioesterase is False
    assert (control.guides, control.additions, control.deletions) == ((), (), ())


def test_strain_spec_folds_the_releases_own_spellings() -> None:
    """The panels write ``folK_38`` where Supplementary Data 1 writes ``folk_38``."""
    strains = {"folk_38": "MG with plasmids pCF and pSg-folk_38"}
    plasmids = {
        "pCF": SYNTHETIC_PLASMIDS["pCF"],
        "pSg-folk_38": "PLlacO-1: sgRNA folk_38, pBR322, AmpR",
    }
    assert m.strain_spec("folK_38", strains, plasmids).guides == ("folk_38",)


def test_strain_spec_completes_the_releases_one_plasmid_typo() -> None:
    """Supplementary Data 1's ``paaG_218`` row names ``pSg-aaG_218``, a dropped ``p``."""
    strains = {"paaG_218": "MG with plasmids pCF and pSg-aaG_218"}
    plasmids = {
        "pCF": SYNTHETIC_PLASMIDS["pCF"],
        "pSg-paaG_218": "PLlacO-1: sgRNA paaG_218, pBR322, AmpR",
    }
    assert m.strain_spec("paaG_218", strains, plasmids).guides == ("paaG_218",)


def test_strain_spec_refuses_a_strain_or_plasmid_the_tables_do_not_state() -> None:
    """Nothing is inferred: an unknown strain or plasmid stops the build."""
    with pytest.raises(ValueError, match="states no strain"):
        m.strain_spec("nope", SYNTHETIC_STRAINS, SYNTHETIC_PLASMIDS)
    with pytest.raises(ValueError, match="states no plasmid"):
        m.strain_spec("thrL_10", SYNTHETIC_STRAINS, {"pCF": SYNTHETIC_PLASMIDS["pCF"]})


def test_resolve_short_guide_is_the_wang_id_and_refuses_an_unknown_one() -> None:
    """``thrL_10`` is Wang's ``thrLb0001_10``; an absent short form raises."""
    assert m.resolve_short_guide("thrL_10", GUIDES) == "thrLb0001_10"
    with pytest.raises(ValueError, match="resolves to"):
        m.resolve_short_guide("nope_1", GUIDES)


def test_check_spacer_against_primer_accepts_either_orientation() -> None:
    """The primer carries the 4-base tag plus the spacer or its reverse complement."""
    spacer = GUIDES["thrLb0001_10"]
    assert (
        m.check_spacer_against_primer("thrL_10", spacer, SYNTHETIC_PRIMERS)
        == "thrL_10-for"
    )
    reverse = spacer.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    assert (
        m.check_spacer_against_primer("x_1", spacer, {"x_1-rev": f"gcac{reverse}"})
        == "x_1-rev"
    )
    with pytest.raises(ValueError, match="carries no primer whose 20-mer"):
        m.check_spacer_against_primer("thrL_10", "A" * 20, SYNTHETIC_PRIMERS)


def test_titer_phenotype_is_one_culture_with_typed_absences() -> None:
    """n_samples 1, the label, and the two uncertainty fields as not-applicable gaps."""
    phenotype = m.titer_phenotype(746.8201, "CF 1")
    assert phenotype.titer == 746.8201
    assert phenotype.titer_unit is ConcentrationUnit.ug_per_ml
    assert phenotype.n_samples == 1
    assert phenotype.sample_unit is SampleUnit.biological_replicate
    assert phenotype.replicate_id == "CF 1"
    assert phenotype.titer_uncertainty is None
    assert phenotype.titer_se is None
    gaps = {gap.field: gap.reason.value for gap in phenotype.provenance_gaps}
    assert gaps["titer_uncertainty"] == "not_applicable_single_measurement"
    assert gaps["titer_uncertainty_type"] == "not_applicable_single_measurement"
    assert gaps["product_yield"] == "not_reported_by_primary"


def test_titer_reference_phenotype_is_the_released_aggregate() -> None:
    """The reference states the control's released mean, its SD and n, with no label."""
    reference = m.titer_reference_phenotype(746.8201, 20.472701218207654, 3)
    assert reference.replicate_id is None
    assert reference.n_samples == 3
    assert reference.titer_uncertainty == pytest.approx(20.4727012, abs=1e-6)
    assert reference.titer_uncertainty_type is UncertaintyType.sample_sd
    assert reference.titer_se == pytest.approx(20.4727012 / math.sqrt(3), abs=1e-6)


def _compound_name(perturbation: Any) -> str:
    """The compound name of one small-molecule environment perturbation."""
    assert isinstance(perturbation, SmallMoleculePerturbation)
    return str(perturbation.compound.name)


def test_titer_environment_gaps_the_vessel_and_adds_arabinose_only_for_pbad() -> None:
    """One environment, the culture format typed as absent, arabinose where P_BAD is."""
    plain = m.titer_environment(arabinose=False)
    assert plain.media is M9_MODIFIED_FANG2025
    assert plain.temperature is not None and plain.temperature.value == 30.0
    assert plain.duration_hours == 40.0
    assert plain.culture_format is None
    assert [gap.field for gap in plain.provenance_gaps] == ["culture_format"]
    assert [_compound_name(p) for p in plain.perturbations] == ["IPTG"]
    induced = m.titer_environment(arabinose=True)
    assert [_compound_name(p) for p in induced.perturbations] == ["IPTG", "arabinose"]
    arabinose = induced.perturbations[1]
    assert isinstance(arabinose, SmallMoleculePerturbation)
    assert arabinose.concentration is not None
    assert arabinose.concentration.value == 10.0


def test_declined_rows_name_their_reason() -> None:
    """Each declined row carries the reason it is not stored, and nothing else is."""
    shell = m.FfaTiterFang2025Dataset.__new__(m.FfaTiterFang2025Dataset)

    def row(strain: str, medium: str = "glycerol") -> m.TiterRow:
        return m.TiterRow(
            panel="Figure 4",
            sheet="Figure 4",
            row=3,
            strain=strain,
            medium=medium,
            replicates=(1.0, 2.0, 3.0),
            released_mean=2.0,
            released_sd=1.0,
        )

    assert shell._declined(row("JM-CF")) == "alternate_host_no_deposited_assembly"
    assert (
        shell._declined(row("CF", "glucose")) == "glucose_medium_not_in_the_media_layer"
    )
    assert shell._declined(row("pcnBmE")) == "named_pcnB_point_allele_leaf_undecided"
    assert shell._declined(row("CF")) is None


def test_titer_accounting_closes_on_every_released_value() -> None:
    """Stored + declined + repeated must equal the released cells, three per strain."""
    accounting = m.TiterAccounting(
        dataset="FfaTiterFang2025Dataset",
        source_blocks=13,
        source_rows=144,
        source_values=432,
        distinct_cultures=102,
        kept_records=306,
        declined_values=42,
        repeated_release_values=84,
        declined={},
        max_repeat_disagreement=0.0,
        repeated_keys=11,
        references={},
        notes=[],
    )
    accounting.check()
    with pytest.raises(ValueError, match="is not the 432 released culture values"):
        accounting.model_copy(update={"repeated_release_values": 0}).check()
    with pytest.raises(ValueError, match="is not three replicates each"):
        accounting.model_copy(
            update={
                "kept_records": 300,
                "repeated_release_values": 90,
                "distinct_cultures": 99,
            }
        ).check()


def test_synthetic_titer_build_stores_one_record_per_culture(
    built_titer: m.FfaTiterFang2025Dataset,
) -> None:
    """Five strains over six released rows, three records each, CF stored once."""
    try:
        assert len(built_titer) == N_SYNTHETIC_TITER_RECORDS
        labels = [
            built_titer[index]["experiment"]["phenotype"]["replicate_id"]
            for index in range(len(built_titer))
        ]
        assert len(set(labels)) == N_SYNTHETIC_TITER_RECORDS
        assert sorted({label.rsplit(" ", 1)[0] for label in labels}) == [
            "CF",
            "CF-",
            "pcnBi",
            "pcnBi-proB_60",
            "pcnBi-thrA+",
            "thrL_10",
        ]
        assert sorted(label.rsplit(" ", 1)[1] for label in labels) == sorted(
            ["1", "2", "3"] * N_SYNTHETIC_TITER_CULTURES
        )
    finally:
        built_titer.close_lmdb()


def test_synthetic_titer_build_types_every_leaf_the_tables_state(
    built_titer: m.FfaTiterFang2025Dataset,
) -> None:
    """A guide is a knockdown with its spacer, an addition is a gene addition."""
    try:
        by_strain: dict[str, list[dict[str, Any]]] = {}
        for index in range(len(built_titer)):
            experiment = built_titer[index]["experiment"]
            strain = str(experiment["phenotype"]["replicate_id"]).rsplit(" ", 1)[0]
            by_strain[strain] = experiment["genotype"]["perturbations"]
        assert by_strain["CF"] == []
        assert by_strain["CF-"] == []
        (single,) = by_strain["thrL_10"]
        assert single["perturbation_type"] == "crispr_interference"
        assert single["systematic_gene_name"] == "b0001"
        assert single["crispr"]["guide_sequence"] == GUIDES["thrLb0001_10"]
        # the pcnBi background's guide plus the library guide's two cluster members
        assert len(by_strain["pcnBi-proB_60"]) == 3
        added = [
            p
            for p in by_strain["pcnBi-thrA+"]
            if p["perturbation_type"] == "gene_addition"
        ]
        assert len(added) == 1
        assert added[0]["systematic_gene_name"] == "b0002"
        assert added[0]["is_heterologous"] is False
        assert added[0]["localization"] == "episomal_replicon"
        assert added[0]["construct_name"] == "Sg-thrA+"
        assert added[0]["replicon_statement"] == "PBAD: thrA, ColE1, AmpR"
        assert [gap["field"] for gap in added[0]["provenance_gaps"]] == ["placement"]
    finally:
        built_titer.close_lmdb()


def test_synthetic_titer_build_points_each_record_at_its_backgrounds_reference(
    built_titer: m.FfaTiterFang2025Dataset,
) -> None:
    """The pC family's reference is the released CF- mean, the pCF family's the CF one."""
    try:
        references: dict[str, float] = {}
        for index in range(len(built_titer)):
            record = built_titer[index]
            strain = str(record["experiment"]["phenotype"]["replicate_id"]).rsplit(
                " ", 1
            )[0]
            references[strain] = record["reference"]["phenotype_reference"]["titer"]
        assert references["CF"] == pytest.approx(746.8201, abs=1e-6)
        assert references["pcnBi"] == pytest.approx(746.8201, abs=1e-6)
        assert references["CF-"] == pytest.approx(86.33773, abs=1e-6)
    finally:
        built_titer.close_lmdb()


def test_synthetic_titer_build_accounts_for_the_repeated_release(
    built_titer: m.FfaTiterFang2025Dataset, synthetic_titer: Path
) -> None:
    """The CF row of the second panel is a re-release, not a second set of cultures."""
    try:
        payload = json.loads(
            (synthetic_titer / "preprocess/titer_build_accounting.json").read_text()
        )
    finally:
        built_titer.close_lmdb()
    accounting = m.TiterAccounting.model_validate(payload)
    accounting.check()
    assert accounting.kept_records == N_SYNTHETIC_TITER_RECORDS
    assert accounting.repeated_release_values == 3
    assert accounting.declined_values == 0
    assert accounting.repeated_keys == 1
    assert accounting.max_repeat_disagreement == 0.0
    ledger = (synthetic_titer / "preprocess/titer_rows.csv").read_text()
    assert "Figure 1e" in ledger and "Figure S9b" in ledger


def test_synthetic_titer_build_passes_the_family_gate(
    built_titer: m.FfaTiterFang2025Dataset, synthetic_titer: Path
) -> None:
    """The L0-L3 titer gate and the panel ledger pass on the synthetic store."""
    from torchcell.verification.product_titer import verify_product_titer_dataset
    from torchcell.verification.runners import load_records

    try:
        records = [dict(record) for record in load_records(str(synthetic_titer))]
    finally:
        built_titer.close_lmdb()
    report = verify_product_titer_dataset(
        records,
        dataset_name="ffa_titer_fang2025",
        provenance=m._titer_provenance(),
        expected_count=N_SYNTHETIC_TITER_RECORDS,
        titer_unit=ConcentrationUnit.ug_per_ml.value,
        titer_unit_detail="synthetic",
        se_tol=m.TITER_SE_TOL,
        pathway_gene_counts=(0,),
        product_names=(m.TITER_PRODUCT_NAME,),
    )
    report.add(m._l3_three_cultures_per_strain(records))
    failed = [result.name for result in report.results if not result.passed]
    assert failed == []
    assert report.results[-1].message.startswith(
        f"{N_SYNTHETIC_TITER_CULTURES} stored strains"
    )


# --------------------------------------------------------------------------- #
# The titer store, built from the real mirror (``--data``)
# --------------------------------------------------------------------------- #
TITER_STORE = (
    Path(DATA_ROOT) / "data/torchcell/ffa_titer_fang2025"
    if DATA_ROOT
    else Path("absent")
)
needs_titer_store = pytest.mark.skipif(
    not (TITER_STORE / "processed/lmdb").exists(),
    reason="the Fang 2025 titer dev-tree LMDB is not built here",
)


@pytest.mark.data
@needs_titer_store
def test_the_built_titer_store_holds_one_record_per_released_culture() -> None:
    """306 records over 102 strains, with every released value accounted for."""
    payload = json.loads(
        (TITER_STORE / "preprocess/titer_build_accounting.json").read_text()
    )
    accounting = m.TiterAccounting.model_validate(payload)
    accounting.check()
    assert accounting.kept_records == m.EXPECTED_TITER_RECORDS == 306
    assert accounting.distinct_cultures == m.EXPECTED_TITER_CULTURES == 102
    assert accounting.source_values == 432
    assert accounting.declined_values == 42
    assert accounting.repeated_release_values == 84
    assert accounting.max_repeat_disagreement < m.TITER_REPEAT_TOLERANCE
    assert set(accounting.declined) == {
        "alternate_host_no_deposited_assembly",
        "glucose_medium_not_in_the_media_layer",
        "named_pcnB_point_allele_leaf_undecided",
    }
    assert accounting.references["pCF"]["mean"] == pytest.approx(746.8201, abs=1e-6)
    assert accounting.references["pC"]["mean"] == pytest.approx(86.33773, abs=1e-6)


@pytest.mark.data
@needs_titer_store
def test_the_built_titer_store_reproduces_every_released_mean() -> None:
    """Grouping the stored cultures by strain gives back the released Mean and SD."""
    import statistics

    dataset = m.FfaTiterFang2025Dataset(root=str(TITER_STORE))
    try:
        groups: dict[str, list[float]] = {}
        for index in range(len(dataset)):
            phenotype = dataset[index]["experiment"]["phenotype"]
            strain = str(phenotype["replicate_id"]).rsplit(" ", 1)[0]
            groups.setdefault(strain, []).append(float(phenotype["titer"]))
    finally:
        dataset.close_lmdb()
    rows = {
        row.strain.lower(): row
        for row in m.read_titer_blocks(
            m.raw_mirror_dir(DATA_ROOT) / m.SCREEN_FILE.relpath
        )
    }
    assert len(groups) == 102
    for strain, values in groups.items():
        row = rows[strain.lower()]
        assert sum(values) / len(values) == pytest.approx(row.released_mean, abs=1e-9)
        assert statistics.stdev(values) == pytest.approx(row.released_sd, abs=1e-9)


@pytest.mark.data
@needs_titer_store
def test_the_built_titer_store_passes_every_verification_level() -> None:
    report = m.verify_titer_build(str(TITER_STORE), data_root=DATA_ROOT)
    failed = [r for r in report.results if not r.passed]
    assert [r.name for r in failed if r.finding_class.value != "report"] == []
    assert report.passed
