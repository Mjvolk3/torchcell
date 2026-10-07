# tests/torchcell/datasets/ecoli/test_wang2018.py
# [[tests.torchcell.datasets.ecoli.test_wang2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_wang2018.py
"""The Wang 2018 pooled-CRISPRi loader (``torchcell.datasets.ecoli.wang2018``).

Synthetic tests (run everywhere) build every input in ``tmp_path``. The hermetic build
uses the real ``EcoliK12MG1655Genome`` over the synthetic MG1655 assembly of
``tests/torchcell/sequence/genome/_bacterial_fixtures.py`` (thrL b0001, thrA b0002,
thrW b0003 tRNA, pseudogene yaaP b0004, proB b0005, proC b0006), served through a
stubbed ``resolve`` with the network refused, and replaces the loader's
``verify_raw_files`` with a presence check (the synthetic workbooks cannot carry the real
pins; those are asserted by the refusal test and the data-gated tests).

The synthetic library, which exercises every branch of the three drop rules:

    cluster        members                  guides                      note
    thrLb0001      b0001                    _10, _25                    two guides, one gene
    thrAb0002      b0002                    _12                         Bad in screen A
    proBb0005      b0005, b0006             _60                         multi-member cluster
    yaaPb0004      b0004                    _45                         pseudogene, KEPT
    ybfKb4590      b4590                    _5                          retired, DROPPED
    thrWb0003      b0003                    _31                         the ncRNA sheet
    (none)         (none)                   NC_1, NC_2                  non-targeting

Screen A (``essentiality``, sigma 0.5) holds all nine rows with ``thrAb0002_12`` and
``NC_2`` flagged Bad: 9 - 2 controls - 2 Bad + 1 both - 1 retired = 5 records over 6
perturbations. Screen B (``furfural_tolerance``, sigma 2.0) holds all nine rows Good:
9 - 2 - 0 + 0 - 1 = 6 records over 7 perturbations. Eleven records, 13 perturbations,
5 distinct genes. Six of seven names resolve (0.857), so the success path lowers
``MIN_RESOLVED_FRACTION`` and a separate test shows the default stops the build.

Data-gated tests (``@pytest.mark.data``) read the real raw mirror and the built dev-tree
LMDB under ``$DATA_ROOT`` (they never build it): the manifest pins, the provenance audit
of every sourced value, the per-screen arithmetic, and the L0-L4 verifier.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
import pytest

import torchcell.datasets.ecoli.wang2018 as m
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    BW25113_LOCI,
    MG1655_GAF,
    MG1655_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.datamodels.media import LB, MOPS_CASAMINO_WANG2018, MOPS_MINIMAL
from torchcell.datamodels.schema import (
    AssayType,
    AssemblyReferenceGenome,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    ConcentrationUnit,
    MeasurementType,
    PhysicalFactor,
    SampleUnit,
)
from torchcell.datasets.bacteria_common import (
    LocusTagReconciliation,
    LocusTagResolutionError,
)
from torchcell.literature.manifest import RetrievalMethod
from torchcell.sequence.genome.base import GeneNameStatus
from torchcell.sequence.genome.ecoli.k12 import (
    MG1655_ASSEMBLY,
    EcoliK12BW25113Genome,
    EcoliK12MG1655Genome,
)
from torchcell.verification.report import Provenance, VerificationReport
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

#: ``guide id -> 20-mer spacer``; every spacer distinct, as the release's are.
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

#: ``(guide id, fitness, quality)`` of the two synthetic screens.
SCREEN_A_ROWS: list[tuple[str, float, str]] = [
    ("thrLb0001_10", -4.0, "Good"),
    ("thrLb0001_25", -3.5, "Good"),
    ("thrAb0002_12", -2.0, "Bad"),
    ("proBb0005_60", 1.25, "Good"),
    ("yaaPb0004_45", 0.5, "Good"),
    ("ybfKb4590_5", -1.0, "Good"),
    ("thrWb0003_31", -6.5, "Good"),
    ("NC_1", 0.125, "Good"),
    ("NC_2", -0.25, "Bad"),
]
SCREEN_B_ROWS: list[tuple[str, float, str]] = [
    (guide, fitness * 2.0, "Good") for guide, fitness, _ in SCREEN_A_ROWS
]
SIGMA_A = 0.5
SIGMA_B = 2.0

SCREEN_A = m.Screen(
    screen_id="essentiality",
    raw=m.RawFile(
        moesm=9, data_number=6, description="synthetic screen A", sha256="0" * 64
    ),
    sheet="essential genes",
    phenotype_label="Essentiality",
    selective="dCas9, LB",
    control="Empty plasmid, LB",
    nc_sigma=SIGMA_A,
    generations=15.0,
    kept_records=5,
)
SCREEN_B = m.Screen(
    screen_id="furfural_tolerance",
    raw=m.RawFile(
        moesm=12, data_number=9, description="synthetic screen B", sha256="1" * 64
    ),
    sheet="furfural tolerance",
    phenotype_label="Furfural tolerance",
    selective="0.4 g/L furfural, MOPS",
    control="initial, see Supplementary Fig. 7",
    nc_sigma=SIGMA_B,
    generations=5.0,
    kept_records=6,
)
SYNTHETIC_SCREENS = (SCREEN_A, SCREEN_B)
SYNTHETIC_RAW_FILES = (m.CLUSTERS_FILE, m.LIBRARY_FILE, SCREEN_A.raw, SCREEN_B.raw)
N_SYNTHETIC_RECORDS = SCREEN_A.kept_records + SCREEN_B.kept_records
N_SYNTHETIC_PERTURBATIONS = 13
N_SYNTHETIC_GENES = 6

REFERENCE = AssemblyReferenceGenome(
    species="Escherichia coli",
    strain="MG1655",
    assembly_set="ecoli_K12_MG1655_ASM584v2",
    assembly_accession="GCA_000005845.2",
)


def _write_sheets(
    path: Path, sheets: Sequence[tuple[str, Sequence[Sequence[Any]]]]
) -> None:
    """Write one workbook, one sheet per ``(name, rows)`` pair, rows including a header."""
    workbook = openpyxl.Workbook()
    default = workbook.active
    assert default is not None
    workbook.remove(default)
    for name, rows in sheets:
        sheet = workbook.create_sheet(name)
        for row in rows:
            sheet.append(list(row))
    workbook.save(path)


def write_clusters(
    path: Path,
    protein: list[tuple[str, str]] | None = None,
    ncrna: list[tuple[str, str]] | None = None,
    *,
    header: tuple[str, ...] = m.CLUSTER_COLUMNS,
) -> None:
    """A synthetic Supplementary Data 2 with its two sheets."""
    _write_sheets(
        path,
        [
            ("protein-coding genes", [header, *(protein or PROTEIN_CLUSTERS)]),
            ("ncRNA-coding genes", [header, *(ncrna or NCRNA_CLUSTERS)]),
        ],
    )


def write_library(path: Path, guides: dict[str, str] | None = None) -> None:
    """A synthetic Supplementary Data 3."""
    rows = list((guides if guides is not None else GUIDES).items())
    _write_sheets(path, [("sheet1", [m.LIBRARY_COLUMNS, *rows])])


def write_screen(
    path: Path,
    screen: m.Screen,
    rows: list[tuple[str, float, str]],
    sigma: float,
    *,
    gene_cell: dict[str, str] | None = None,
) -> None:
    """A synthetic Supplementary Data 6-10 sheet: id, gene, fitness, Z, Quality."""
    overrides = gene_cell or {}
    body: list[Sequence[Any]] = [m.FITNESS_COLUMNS]
    for guide, fitness, quality in rows:
        match = m.GUIDE_ID_RE.match(guide)
        default = (
            m.CONTROL_GENE_CELL
            if match is None
            else m._member(str(match.group("token"))).symbol
        )
        body.append(
            [
                guide,
                overrides.get(guide, default),
                repr(fitness),
                repr(fitness / sigma),
                quality,
            ]
        )
    _write_sheets(path, [(screen.sheet, body)])


def write_raw(raw: Path) -> None:
    """Every synthetic file the loader reads, under ``raw``."""
    raw.mkdir(parents=True, exist_ok=True)
    write_clusters(raw / m.CLUSTERS_FILE.filename)
    write_library(raw / m.LIBRARY_FILE.filename)
    write_screen(raw / SCREEN_A.raw.filename, SCREEN_A, SCREEN_A_ROWS, SIGMA_A)
    write_screen(raw / SCREEN_B.raw.filename, SCREEN_B, SCREEN_B_ROWS, SIGMA_B)


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
    root = tmp_path / "data/torchcell/crispri_guide_fitness_wang2018"
    write_raw(root / "raw")
    monkeypatch.setattr(m, "SCREENS", SYNTHETIC_SCREENS)
    monkeypatch.setattr(m, "RAW_FILES", SYNTHETIC_RAW_FILES)
    monkeypatch.setattr(m, "N_CLUSTERS", N_SYNTHETIC_CLUSTERS)
    monkeypatch.setattr(m, "N_CLUSTER_MEMBERS", N_SYNTHETIC_MEMBERS)
    monkeypatch.setattr(m, "N_TARGETING_GUIDES", N_SYNTHETIC_TARGETING)
    monkeypatch.setattr(m, "N_CONTROL_GUIDES", N_SYNTHETIC_CONTROLS)
    monkeypatch.setattr(m, "EXPECTED_RECORDS", N_SYNTHETIC_RECORDS)
    monkeypatch.setattr(m, "EXPECTED_PERTURBATIONS", N_SYNTHETIC_PERTURBATIONS)
    monkeypatch.setattr(m, "EXPECTED_GENES", N_SYNTHETIC_GENES)
    monkeypatch.setattr(m, "host_reference", lambda *a, **k: REFERENCE)
    return root


@pytest.fixture
def built(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch, mg1655: EcoliK12MG1655Genome
) -> m.CrispriGuideFitnessWang2018Dataset:
    """The dataset built over the synthetic release."""
    monkeypatch.setattr(
        m.CrispriGuideFitnessWang2018Dataset, "MIN_RESOLVED_FRACTION", 0.8
    )
    return m.CrispriGuideFitnessWang2018Dataset(
        root=str(synthetic), ecoli_genome=mg1655
    )


def _records(dataset: m.CrispriGuideFitnessWang2018Dataset) -> list[dict[str, Any]]:
    return [dataset[i] for i in range(len(dataset))]


# --------------------------------------------------------------------------- #
# Identifier tokens
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("token", "symbol", "bnumber"),
    [
        ("thrLb0001", "thrL", "b0001"),
        ("ybfKb4590", "ybfK", "b4590"),
        ("phaAZb1234", "phaAZ", "b1234"),
        ("tRNA-Arg1b0001", "tRNA-Arg1", "b0001"),
    ],
)
def test_a_cluster_member_token_splits_into_symbol_and_bnumber(
    token: str, symbol: str, bnumber: str
) -> None:
    """The source's own token is where every stored b-number comes from."""
    member = m._member(token)
    assert (member.symbol, member.bnumber) == (symbol, bnumber)


def test_a_token_with_no_bnumber_is_refused() -> None:
    """A member the loader cannot key to a locus stops the build."""
    with pytest.raises(ValueError, match="cluster member token"):
        m._member("thrL")


@pytest.mark.parametrize(
    ("guide_id", "token", "position"),
    [("gspKb3332_817", "gspKb3332", "817"), ("rsmEb3167_9", "rsmEb3167", "9")],
)
def test_a_guide_id_splits_into_its_cluster_token_and_pam_position(
    guide_id: str, token: str, position: str
) -> None:
    """``<symbol><bNNNN>_<position>`` is the release's naming rule."""
    match = m.GUIDE_ID_RE.match(guide_id)
    assert match is not None
    assert (match.group("token"), match.group("position")) == (token, position)


def test_a_non_targeting_id_matches_only_the_control_pattern() -> None:
    """``NC_983`` is a control id and not a guide id, which is how rows are split."""
    assert m.GUIDE_ID_RE.match("NC_983") is None
    assert m.CONTROL_ID_RE.match("NC_983") is not None


# --------------------------------------------------------------------------- #
# Supplementary Data 2: the clusters
# --------------------------------------------------------------------------- #
def test_read_clusters_reads_both_sheets_and_expands_every_member(
    tmp_path: Path,
) -> None:
    """A cluster's members are the guide's whole target set, with its gene class."""
    path = tmp_path / "clusters.xlsx"
    write_clusters(path)
    clusters = m.read_clusters(
        path, n_clusters=N_SYNTHETIC_CLUSTERS, n_members=N_SYNTHETIC_MEMBERS
    )
    assert len(clusters) == N_SYNTHETIC_CLUSTERS
    assert [mem.bnumber for mem in clusters["proBb0005"].members] == ["b0005", "b0006"]
    assert clusters["proBb0005"].gene_class == "protein_coding"
    assert clusters["thrWb0003"].gene_class == "ncRNA_coding"


def test_a_cluster_whose_token_is_not_its_first_member_is_refused(
    tmp_path: Path,
) -> None:
    """The representative must open the member list; otherwise the join is ambiguous."""
    path = tmp_path / "clusters.xlsx"
    write_clusters(path, protein=[("thrLb0001", "thrAb0002,thrLb0001")])
    with pytest.raises(ValueError, match="does not open with itself"):
        m.read_clusters(path, n_clusters=1, n_members=2)


def test_a_member_in_two_clusters_is_refused(tmp_path: Path) -> None:
    """One gene in two clusters would make a guide's target set unknowable."""
    path = tmp_path / "clusters.xlsx"
    write_clusters(
        path,
        protein=[("thrLb0001", "thrLb0001,thrAb0002")],
        ncrna=[("thrWb0003", "thrWb0003,thrAb0002")],
    )
    with pytest.raises(ValueError, match="is in clusters"):
        m.read_clusters(path, n_clusters=2, n_members=3)


def test_a_cluster_count_that_is_not_the_librarys_is_refused(tmp_path: Path) -> None:
    """The library shape is pinned, so a changed export stops the build."""
    path = tmp_path / "clusters.xlsx"
    write_clusters(path)
    with pytest.raises(ValueError, match="the paper's library is"):
        m.read_clusters(path, n_clusters=4205, n_members=4317)


def test_a_renamed_cluster_column_is_refused(tmp_path: Path) -> None:
    """A renamed column means the export moved; parsing on would mis-key the data."""
    path = tmp_path / "clusters.xlsx"
    write_clusters(path, header=("Cluster", "Members"))
    with pytest.raises(ValueError, match="expected"):
        m.read_clusters(
            path, n_clusters=N_SYNTHETIC_CLUSTERS, n_members=N_SYNTHETIC_MEMBERS
        )


# --------------------------------------------------------------------------- #
# Supplementary Data 3: the library
# --------------------------------------------------------------------------- #
def test_read_library_returns_every_spacer_and_counts_the_two_id_forms(
    tmp_path: Path,
) -> None:
    """The spacer is what makes two guides of one gene two strains, so it is stored."""
    write_clusters(tmp_path / "clusters.xlsx")
    write_library(tmp_path / "library.xlsx")
    clusters = m.read_clusters(
        tmp_path / "clusters.xlsx",
        n_clusters=N_SYNTHETIC_CLUSTERS,
        n_members=N_SYNTHETIC_MEMBERS,
    )
    spacers = m.read_library(
        tmp_path / "library.xlsx",
        clusters,
        n_targeting=N_SYNTHETIC_TARGETING,
        n_non_targeting=N_SYNTHETIC_CONTROLS,
    )
    assert spacers == GUIDES
    assert spacers["thrLb0001_10"] != spacers["thrLb0001_25"]


def test_a_spacer_that_is_not_a_20mer_is_refused(tmp_path: Path) -> None:
    """Every designed guide is a 20-mer; a shorter one is a broken export."""
    write_clusters(tmp_path / "clusters.xlsx")
    guides = dict(GUIDES)
    guides["thrLb0001_10"] = "AAAACGCGCGTCACGCGTC"
    write_library(tmp_path / "library.xlsx", guides)
    clusters = m.read_clusters(
        tmp_path / "clusters.xlsx",
        n_clusters=N_SYNTHETIC_CLUSTERS,
        n_members=N_SYNTHETIC_MEMBERS,
    )
    with pytest.raises(ValueError, match="are not 20-mers"):
        m.read_library(
            tmp_path / "library.xlsx",
            clusters,
            n_targeting=N_SYNTHETIC_TARGETING,
            n_non_targeting=N_SYNTHETIC_CONTROLS,
        )


def test_a_guide_naming_no_cluster_is_refused(tmp_path: Path) -> None:
    """A guide whose token is in no cluster has no target set to store."""
    write_clusters(tmp_path / "clusters.xlsx")
    guides = dict(GUIDES)
    guides["mutTb0099_7"] = "AGAGCGCGCGTCACGCGTCC"
    write_library(tmp_path / "library.xlsx", guides)
    clusters = m.read_clusters(
        tmp_path / "clusters.xlsx",
        n_clusters=N_SYNTHETIC_CLUSTERS,
        n_members=N_SYNTHETIC_MEMBERS,
    )
    with pytest.raises(ValueError, match="name no cluster"):
        m.read_library(
            tmp_path / "library.xlsx",
            clusters,
            n_targeting=N_SYNTHETIC_TARGETING + 1,
            n_non_targeting=N_SYNTHETIC_CONTROLS,
        )


def test_a_library_split_that_is_not_the_stated_one_is_refused(tmp_path: Path) -> None:
    """55,671 targeting and 400 control guides is a sourced number, so it is checked."""
    write_clusters(tmp_path / "clusters.xlsx")
    write_library(tmp_path / "library.xlsx")
    clusters = m.read_clusters(
        tmp_path / "clusters.xlsx",
        n_clusters=N_SYNTHETIC_CLUSTERS,
        n_members=N_SYNTHETIC_MEMBERS,
    )
    with pytest.raises(ValueError, match="the paper states"):
        m.read_library(tmp_path / "library.xlsx", clusters)


def test_an_id_of_neither_form_is_refused(tmp_path: Path) -> None:
    """A library id that is neither a guide nor a control cannot be classified."""
    write_clusters(tmp_path / "clusters.xlsx")
    guides = dict(GUIDES)
    guides["weird-id"] = "AGCGCGCGCGTCACGCGTCC"
    write_library(tmp_path / "library.xlsx", guides)
    clusters = m.read_clusters(
        tmp_path / "clusters.xlsx",
        n_clusters=N_SYNTHETIC_CLUSTERS,
        n_members=N_SYNTHETIC_MEMBERS,
    )
    with pytest.raises(ValueError, match="neither form"):
        m.read_library(
            tmp_path / "library.xlsx",
            clusters,
            n_targeting=N_SYNTHETIC_TARGETING,
            n_non_targeting=N_SYNTHETIC_CONTROLS,
        )


# --------------------------------------------------------------------------- #
# Supplementary Data 6-10: the screens and the Z back-solve
# --------------------------------------------------------------------------- #
def test_read_screen_counts_every_drop_rule_and_keeps_the_rest(tmp_path: Path) -> None:
    """The three counts plus the candidates account for every released row."""
    path = tmp_path / "screen.xlsx"
    write_screen(path, SCREEN_A, SCREEN_A_ROWS, SIGMA_A)
    rows = m.read_screen(path, SCREEN_A)
    assert rows.source_rows == len(SCREEN_A_ROWS)
    assert (rows.control_rows, rows.bad_rows, rows.control_and_bad_rows) == (2, 2, 1)
    assert len(rows.candidates) == 6
    assert "ybfKb4590_5" in set(rows.candidates["guide_id"])


def test_read_screen_back_solves_the_screens_sigma(tmp_path: Path) -> None:
    """Equation (4) divides by one constant, so the ratio recovers it exactly."""
    path = tmp_path / "screen.xlsx"
    write_screen(path, SCREEN_B, SCREEN_B_ROWS, SIGMA_B)
    assert m.read_screen(path, SCREEN_B).sigma == pytest.approx(SIGMA_B)


def test_back_solved_sigma_refuses_a_ratio_that_is_not_one_constant() -> None:
    """A Z column that is not fitness/sigma means the release is not equation (4)."""
    with pytest.raises(ValueError, match="not one constant"):
        m.back_solved_sigma(pd.Series([1.0, 2.0]), pd.Series([2.0, 1.0]))


def test_back_solved_sigma_skips_rows_with_a_zero_z_score() -> None:
    """A zero Z carries no information about sigma, so it is skipped, not divided by."""
    sigma = m.back_solved_sigma(pd.Series([0.0, 3.0]), pd.Series([0.0, 1.5]))
    assert sigma == pytest.approx(2.0)


def test_a_screen_whose_sigma_is_not_the_recorded_one_is_refused(
    tmp_path: Path,
) -> None:
    """The recorded sigma is a measured constant, so a drift stops the build."""
    path = tmp_path / "screen.xlsx"
    write_screen(path, SCREEN_A, SCREEN_A_ROWS, SIGMA_A * 2)
    with pytest.raises(ValueError, match="is not the recorded"):
        m.read_screen(path, SCREEN_A)


def test_a_third_quality_value_is_refused(tmp_path: Path) -> None:
    """Only Good and Bad exist; a third value would be silently kept as good."""
    path = tmp_path / "screen.xlsx"
    rows = [(guide, fitness, "Unclear") for guide, fitness, _ in SCREEN_A_ROWS]
    write_screen(path, SCREEN_A, rows, SIGMA_A)
    with pytest.raises(ValueError, match="are outside"):
        m.read_screen(path, SCREEN_A)


def test_a_control_row_naming_a_gene_is_refused(tmp_path: Path) -> None:
    """A non-targeting row must carry the source's own ``0`` sentinel."""
    path = tmp_path / "screen.xlsx"
    write_screen(path, SCREEN_A, SCREEN_A_ROWS, SIGMA_A, gene_cell={"NC_1": "thrL"})
    with pytest.raises(ValueError, match="sentinel"):
        m.read_screen(path, SCREEN_A)


def test_a_row_whose_gene_cell_is_not_its_ids_symbol_is_refused(tmp_path: Path) -> None:
    """The gene cell agrees with the id on every released row, so disagreement stops."""
    path = tmp_path / "screen.xlsx"
    write_screen(
        path, SCREEN_A, SCREEN_A_ROWS, SIGMA_A, gene_cell={"thrLb0001_10": "thrA"}
    )
    with pytest.raises(ValueError, match="not their id's symbol"):
        m.read_screen(path, SCREEN_A)


# --------------------------------------------------------------------------- #
# Environments
# --------------------------------------------------------------------------- #
def test_the_essentiality_screen_is_lb_with_both_stated_antibiotics() -> None:
    """The essentiality arm IS the LB outgrowth, which states kanamycin and ampicillin."""
    environment = m.selective_environment(SCREEN_A)
    assert environment.media == LB
    assert [p.compound.name for p in environment.perturbations] == [  # type: ignore[union-attr]  # both are SmallMoleculePerturbation
        "kanamycin",
        "ampicillin",
    ]
    doses = [p.concentration for p in environment.perturbations]  # type: ignore[union-attr]  # both carry a dose
    assert [(d.value, d.unit) for d in doses] == [
        (50.0, ConcentrationUnit.ug_per_ml),
        (100.0, ConcentrationUnit.ug_per_ml),
    ]
    assert (environment.duration_hours, environment.duration_generations) == (9.0, 15.0)


@pytest.mark.parametrize(
    ("screen_id", "media_name", "agents"),
    [
        ("auxotrophy", MOPS_MINIMAL.name, ["D-glucose"]),
        ("trp_biosynthesis", MOPS_CASAMINO_WANG2018.name, ["D-glucose"]),
        ("furfural_tolerance", MOPS_MINIMAL.name, ["D-glucose", "furfural"]),
        ("isobutanol_tolerance", MOPS_MINIMAL.name, ["D-glucose", "isobutanol"]),
    ],
)
def test_each_mops_screen_carries_its_base_and_its_stated_additions(
    screen_id: str, media_name: str, agents: list[str]
) -> None:
    """The MOPS base is shared and the glucose is a typed carbon-source factor."""
    screen = next(s for s in m.SCREENS if s.screen_id == screen_id)
    environment = m.selective_environment(screen)
    assert environment.media.name == media_name
    names = [
        getattr(p, "agent", None).name  # type: ignore[union-attr]  # the factor carries an agent
        if p.perturbation_type == "environment_physical"
        else p.compound.name  # type: ignore[union-attr]  # the rest are small molecules
        for p in environment.perturbations
    ]
    assert names == agents
    carbon = environment.perturbations[0]
    assert carbon.factor is PhysicalFactor.carbon_source  # type: ignore[union-attr]  # checked above
    assert carbon.magnitude is not None  # type: ignore[union-attr]  # a stated 10 g/L
    assert (carbon.magnitude.value, carbon.magnitude.unit) == (  # type: ignore[union-attr]  # checked above
        10.0,
        ConcentrationUnit.g_per_l,
    )
    assert environment.duration_hours is None
    assert [g.field for g in environment.provenance_gaps] == ["duration_hours"]


def test_a_stressor_dose_matches_table_2() -> None:
    """Table 2's selective cells print 0.4 g/L furfural and 4 g/L isobutanol."""
    assert m._stressor("furfural").concentration.value == 0.4
    assert m._stressor("isobutanol").concentration.value == 4.0
    assert [g.field for g in m._stressor("furfural").provenance_gaps] == ["solvent"]


@pytest.mark.parametrize(
    ("screen_id", "media_name", "n_perturbations"),
    [
        ("essentiality", LB.name, 2),
        ("auxotrophy", LB.name, 0),
        ("trp_biosynthesis", LB.name, 0),
        ("furfural_tolerance", LB.name, 2),
        ("isobutanol_tolerance", LB.name, 2),
    ],
)
def test_each_control_condition_is_the_culture_table_2_names(
    screen_id: str, media_name: str, n_perturbations: int
) -> None:
    """The outgrowth for the essentiality and tolerance screens, re-seeded LB otherwise."""
    screen = next(s for s in m.SCREENS if s.screen_id == screen_id)
    environment = m.control_environment(screen)
    assert environment.media.name == media_name
    assert len(environment.perturbations) == n_perturbations


def test_an_unknown_screen_has_no_environment() -> None:
    """A screen this paper did not run is a refusal, never a defaulted environment."""
    other = SCREEN_A.model_copy(update={"screen_id": "not_a_screen"})
    with pytest.raises(ValueError, match="five screens"):
        m.selective_environment(other)
    with pytest.raises(ValueError, match="five screens"):
        m.control_environment(other)


# --------------------------------------------------------------------------- #
# Record parts
# --------------------------------------------------------------------------- #
def test_the_phenotype_is_a_signed_log2_ratio_over_two_biological_replicates() -> None:
    """A negative fitness is the point: this is not a clamped FitnessPhenotype."""
    phenotype = m.guide_phenotype(SCREEN_A, -4.0)
    assert phenotype.measurement_type is MeasurementType.log2_ratio
    assert phenotype.assay_type is AssayType.pooled_competitive_growth_barcode
    assert phenotype.environment_response == -4.0
    assert (phenotype.n_samples, phenotype.sample_unit) == (
        2,
        SampleUnit.biological_replicate,
    )
    assert phenotype.screen_id == "essentiality"
    assert phenotype.environment_response_se is None
    assert {g.field for g in phenotype.provenance_gaps} == {
        "environment_response_uncertainty",
        "environment_response_uncertainty_type",
        "environment_response_se",
    }


def test_the_reference_response_is_zero_by_the_normalization() -> None:
    """Equation (3) subtracts the control median, so the baseline is exactly 0.0."""
    reference = m.build_reference("ds", SCREEN_B, REFERENCE)
    assert reference.phenotype_reference.environment_response == 0.0
    assert reference.phenotype_reference.screen_id == "furfural_tolerance"
    assert reference.genome_reference == REFERENCE
    assert reference.environment_reference.media == LB


def test_a_multi_member_cluster_becomes_one_perturbation_per_gene() -> None:
    """One guide repressing a BLASTN cluster is one record over several genes."""
    genotype = m.build_genotype(
        [("b0005", "proB"), ("b0006", "proC")], GUIDES["proBb0005_60"]
    )
    assert [p.systematic_gene_name for p in genotype.perturbations] == [
        "b0005",
        "b0006",
    ]
    assert [p.perturbed_gene_name for p in genotype.perturbations] == ["proB", "proC"]
    for perturbation in genotype.perturbations:
        assert perturbation.gene_namespace == "ecoli_k12_mg1655_bnumber"  # type: ignore[union-attr]  # a bacterial leaf
        assert perturbation.identifier_mapping is None  # type: ignore[union-attr]  # released as stored
        assert perturbation.crispr is not None  # type: ignore[union-attr]  # a guide-directed leaf
        assert perturbation.crispr.effector == "dCas9"  # type: ignore[union-attr]  # checked above
        assert perturbation.crispr.guide_sequence == GUIDES["proBb0005_60"]  # type: ignore[union-attr]  # checked above
        assert perturbation.crispr.n_guides == 1  # type: ignore[union-attr]  # one guide per record
        assert perturbation.crispr.library_pool is None  # type: ignore[union-attr]  # one screened pool


def test_a_yeast_systematic_name_is_refused_by_the_bacterial_leaf() -> None:
    """The leaf's validator keeps a yeast ORF out of the b-number namespace."""
    with pytest.raises(ValueError):
        m.build_genotype([("YAL001C", "TFC3")], GUIDES["thrLb0001_10"])


def test_stored_gene_names_prefer_the_annotation_and_fall_back_to_the_source(
    tmp_path: Path, mg1655: EcoliK12MG1655Genome
) -> None:
    """One gene, one spelling; a locus the annotation does not name keeps the source's."""
    write_clusters(tmp_path / "clusters.xlsx")
    clusters = m.read_clusters(
        tmp_path / "clusters.xlsx",
        n_clusters=N_SYNTHETIC_CLUSTERS,
        n_members=N_SYNTHETIC_MEMBERS,
    )
    names = m.stored_gene_names(mg1655, clusters.values())
    assert names["b0001"] == "thrL"
    assert names["b0005"] == "proB"
    assert names["b4590"] == "ybfK"


def test_the_publication_is_doi_only() -> None:
    """The PubMed id is in none of the mirrored bytes, so it is not asserted."""
    publication = m.publication()
    assert publication.pubmed_id is None
    assert publication.doi == m.DOI


# --------------------------------------------------------------------------- #
# The accounting
# --------------------------------------------------------------------------- #
_RECONCILIATION = LocusTagReconciliation(
    label="ds",
    assembly_set="ecoli_K12_MG1655_ASM584v2",
    gene_namespace="ecoli_k12_mg1655_bnumber",
    unique_names=N_SYNTHETIC_MEMBERS,
    status_histogram={status: 0 for status in GeneNameStatus},
    layer_histogram={"locus tag": N_SYNTHETIC_MEMBERS},
    remapped=0,
    kept_on_collision=(),
    retired_kept=("b4590",),
    ambiguous_kept={},
    case_insensitive=(),
    outside_namespace=(),
)


def _screen_accounting(**overrides: Any) -> m.ScreenAccounting:
    fields: dict[str, Any] = {
        "screen_id": "essentiality",
        "phenotype_label": "Essentiality",
        "selective_condition": "dCas9, LB",
        "control_condition": "Empty plasmid, LB",
        "supplementary_data": 6,
        "source_rows": 9,
        "dropped_non_targeting": 2,
        "dropped_bad_quality": 2,
        "dropped_non_targeting_and_bad": 1,
        "dropped_retired_target": 1,
        "kept_records": 5,
        "perturbations": 6,
        "nc_sigma": SIGMA_A,
    }
    fields.update(overrides)
    return m.ScreenAccounting(**fields)


def test_screen_accounting_accepts_arithmetic_that_balances() -> None:
    """9 rows minus 2 controls minus 2 Bad plus 1 both minus 1 retired is 5 records."""
    accounting = _screen_accounting()
    accounting.check()
    assert (
        (
            accounting.source_rows
            - accounting.dropped_non_targeting
            - accounting.dropped_bad_quality
            + accounting.dropped_non_targeting_and_bad
            - accounting.dropped_retired_target
        )
        == accounting.kept_records
        == 5
    )


def test_screen_accounting_refuses_arithmetic_that_does_not_balance() -> None:
    """A drop count that does not add up is a bookkeeping error, never rounded away."""
    with pytest.raises(ValueError, match="not the 4 records written"):
        _screen_accounting(kept_records=4).check()


def test_build_accounting_refuses_totals_that_do_not_sum_over_the_screens() -> None:
    """The totals a reader quotes must be the per-screen numbers added up."""
    accounting = m.BuildAccounting(
        dataset="ds",
        reference_strain="MG1655",
        assembly_set="ecoli_K12_MG1655_ASM584v2",
        library_guides=len(GUIDES),
        library_targeting_guides=N_SYNTHETIC_TARGETING,
        library_non_targeting_guides=N_SYNTHETIC_CONTROLS,
        clusters=N_SYNTHETIC_CLUSTERS,
        cluster_members=N_SYNTHETIC_MEMBERS,
        multi_member_clusters=1,
        largest_cluster=2,
        screens=[_screen_accounting()],
        source_rows=9,
        kept_records=4,
        perturbations=6,
        distinct_genes=5,
        retired_targets_dropped=["b4590"],
        pseudogene_targets_kept=["b0004"],
        source_symbol_disagreements=0,
        reconciliation=_RECONCILIATION,
        drop_rules=m.DROP_RULES,
        notes=[],
    )
    with pytest.raises(ValueError, match="do not sum to"):
        accounting.check()


# --------------------------------------------------------------------------- #
# The hermetic build
# --------------------------------------------------------------------------- #
def test_the_build_writes_one_record_per_kept_guide_and_screen(
    built: m.CrispriGuideFitnessWang2018Dataset,
) -> None:
    """Eleven records over the two synthetic screens, at their stated counts."""
    assert len(built) == N_SYNTHETIC_RECORDS
    records = _records(built)
    screens = [r["experiment"]["phenotype"]["screen_id"] for r in records]
    assert screens.count("essentiality") == SCREEN_A.kept_records
    assert screens.count("furfural_tolerance") == SCREEN_B.kept_records


def test_the_build_validates_against_the_bacterial_experiment_classes(
    built: m.CrispriGuideFitnessWang2018Dataset,
) -> None:
    """Every stored record round-trips through the schema classes it declares."""
    record = _records(built)[0]
    experiment = BacterialEnvironmentResponseExperiment.model_validate(
        record["experiment"]
    )
    reference = BacterialEnvironmentResponseExperimentReference.model_validate(
        record["reference"]
    )
    assert experiment.experiment_type == "bacterial_environment_response"
    assert experiment.phenotype.measurement_type is MeasurementType.log2_ratio
    assert reference.experiment_reference_type == "bacterial_environment_response"
    assert reference.phenotype_reference.environment_response == 0.0


def test_a_retired_target_is_dropped_and_a_pseudogene_target_is_kept(
    built: m.CrispriGuideFitnessWang2018Dataset,
) -> None:
    """``b4590`` has no locus in the assembly; the pseudogene ``b0004`` does."""
    genes = {
        p["systematic_gene_name"]
        for record in _records(built)
        for p in record["experiment"]["genotype"]["perturbations"]
    }
    assert genes == {"b0001", "b0002", "b0003", "b0004", "b0005", "b0006"}
    assert "b4590" not in genes


def test_a_bad_quality_row_is_dropped_only_in_the_screen_that_flags_it(
    built: m.CrispriGuideFitnessWang2018Dataset,
) -> None:
    """``thrAb0002_12`` is Bad in screen A and Good in screen B, so it has one record."""
    rows = [
        (
            r["experiment"]["phenotype"]["screen_id"],
            r["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"],
        )
        for r in _records(built)
    ]
    assert ("furfural_tolerance", "b0002") in rows
    assert ("essentiality", "b0002") not in rows


def test_two_guides_of_one_gene_are_two_records_with_different_spacers(
    built: m.CrispriGuideFitnessWang2018Dataset,
) -> None:
    """The spacer is the strain discriminator the L1 uniqueness key needs."""
    spacers = [
        r["experiment"]["genotype"]["perturbations"][0]["crispr"]["guide_sequence"]
        for r in _records(built)
        if r["experiment"]["phenotype"]["screen_id"] == "essentiality"
        and r["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        == "b0001"
    ]
    assert sorted(spacers) == sorted([GUIDES["thrLb0001_10"], GUIDES["thrLb0001_25"]])


def test_the_stored_fitness_is_the_released_number(
    built: m.CrispriGuideFitnessWang2018Dataset,
) -> None:
    """Screen B's rows are twice screen A's, verbatim, with no rescaling."""
    by_key = {
        (
            r["experiment"]["phenotype"]["screen_id"],
            r["experiment"]["genotype"]["perturbations"][0]["crispr"]["guide_sequence"],
        ): r["experiment"]["phenotype"]["environment_response"]
        for r in _records(built)
    }
    assert by_key[("essentiality", GUIDES["thrLb0001_10"])] == -4.0
    assert by_key[("furfural_tolerance", GUIDES["thrLb0001_10"])] == -8.0


def test_the_build_accounting_balances_and_names_the_drop_rules(
    built: m.CrispriGuideFitnessWang2018Dataset,
) -> None:
    """The report a reader audits: per-screen arithmetic plus the totals."""
    payload = json.loads(
        Path(osp.join(built.preprocess_dir, "build_accounting.json")).read_text()
    )
    accounting = m.BuildAccounting.model_validate(payload)
    accounting.check()
    assert accounting.kept_records == N_SYNTHETIC_RECORDS
    assert accounting.perturbations == N_SYNTHETIC_PERTURBATIONS
    assert accounting.distinct_genes == N_SYNTHETIC_GENES
    assert accounting.retired_targets_dropped == ["b4590"]
    assert accounting.pseudogene_targets_kept == ["b0004"]
    assert accounting.multi_member_clusters == 1
    assert accounting.largest_cluster == 2
    assert len(accounting.drop_rules) == 3
    assert [s.nc_sigma for s in accounting.screens] == [SIGMA_A, SIGMA_B]


def test_the_guide_library_table_keeps_the_sources_own_symbols(
    built: m.CrispriGuideFitnessWang2018Dataset,
) -> None:
    """The release's spelling is preserved beside the stored one, never overwritten."""
    frame = pd.read_csv(osp.join(built.preprocess_dir, "guide_library.csv"))
    assert len(frame) == len(GUIDES)
    row = frame[frame["guide_id"] == "proBb0005_60"].iloc[0]
    assert row["source_symbols"] == "proB;proC"
    assert row["cluster_members"] == "b0005;b0006"
    assert bool(frame[frame["guide_id"] == "NC_1"].iloc[0]["is_non_targeting"])


def test_the_build_refuses_a_resolved_fraction_below_its_floor(
    synthetic: Path, mg1655: EcoliK12MG1655Genome
) -> None:
    """Six of seven names resolve, which is below the default floor, so the build stops."""
    with pytest.raises(LocusTagResolutionError, match="resolve to"):
        m.CrispriGuideFitnessWang2018Dataset(root=str(synthetic), ecoli_genome=mg1655)


def test_the_build_refuses_a_genome_of_another_k12_strain(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ``BW25113_`` tag is not a b-number, so the wrong K-12 genome is refused."""
    from torchcell.sequence.genome.ecoli.k12 import BW25113_ASSEMBLY

    monkeypatch.setattr(
        m.CrispriGuideFitnessWang2018Dataset, "MIN_RESOLVED_FRACTION", 0.8
    )
    files = write_assembly(tmp_path / "bw-tier", BW25113_ASSEMBLY, BW25113_LOCI)
    serve_tier(monkeypatch, files)
    root = tmp_path / "bw25113"
    root.mkdir()
    bw25113 = EcoliK12BW25113Genome(genome_root=str(root), overwrite=False)
    with pytest.raises(ValueError, match="needs the ecoli_K12_MG1655_ASM584v2 genome"):
        m.CrispriGuideFitnessWang2018Dataset(root=str(synthetic), ecoli_genome=bw25113)


def test_the_build_refuses_a_record_count_that_is_not_the_pinned_one(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch, mg1655: EcoliK12MG1655Genome
) -> None:
    """A screen that writes a count other than its measured one stops the build."""
    monkeypatch.setattr(
        m.CrispriGuideFitnessWang2018Dataset, "MIN_RESOLVED_FRACTION", 0.8
    )
    monkeypatch.setattr(
        m, "SCREENS", (SCREEN_A.model_copy(update={"kept_records": 4}), SCREEN_B)
    )
    with pytest.raises(RuntimeError, match="the pinned bytes give"):
        m.CrispriGuideFitnessWang2018Dataset(root=str(synthetic), ecoli_genome=mg1655)


# --------------------------------------------------------------------------- #
# The supplementary verifier rows
# --------------------------------------------------------------------------- #
def _rows(records: list[dict[str, Any]]) -> m.SupplementaryRows:
    return m.SupplementaryRows().add_all(records)


def test_screen_coverage_passes_on_the_measured_counts(
    built: m.CrispriGuideFitnessWang2018Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every screen is present at exactly its stated record count."""
    monkeypatch.setattr(m, "SCREENS", SYNTHETIC_SCREENS)
    result = _rows(_records(built)).screen_coverage()
    assert result.passed
    assert result.details["observed"] == {
        "essentiality": SCREEN_A.kept_records,
        "furfural_tolerance": SCREEN_B.kept_records,
    }


def test_screen_coverage_fails_when_a_screen_is_short(
    built: m.CrispriGuideFitnessWang2018Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing record is a failure, not a rounded-down count."""
    monkeypatch.setattr(m, "SCREENS", SYNTHETIC_SCREENS)
    assert not _rows(_records(built)[:-1]).screen_coverage().passed


def test_every_knockdown_carries_its_spacer_and_the_dcas9_effector(
    built: m.CrispriGuideFitnessWang2018Dataset,
) -> None:
    """Without the spacer the L1 strain key would collapse two guides into one."""
    result = _rows(_records(built)).guide_spacers()
    assert result.passed
    assert result.details["n_perturbations"] == N_SYNTHETIC_PERTURBATIONS
    assert result.details["effectors"] == {"dCas9": N_SYNTHETIC_PERTURBATIONS}


def test_guide_spacers_fails_on_a_missing_spacer(
    built: m.CrispriGuideFitnessWang2018Dataset,
) -> None:
    """A record whose guide was not joined is a defect the row catches."""
    records = _records(built)
    records[0]["experiment"]["genotype"]["perturbations"][0]["crispr"][
        "guide_sequence"
    ] = None
    assert not _rows(records).guide_spacers().passed


def test_guide_spacers_fails_on_an_unexpected_effector(
    built: m.CrispriGuideFitnessWang2018Dataset,
) -> None:
    """A dead-Cas effector other than this paper's dCas9 is not this dataset."""
    records = _records(built)
    records[0]["experiment"]["genotype"]["perturbations"][0]["crispr"]["effector"] = (
        "dCpf1"
    )
    assert not _rows(records).guide_spacers().passed


def test_the_assembly_pin_is_mg1655_genbank_with_no_background(
    built: m.CrispriGuideFitnessWang2018Dataset,
) -> None:
    """Every reference pins GCA_000005845.2 and asserts no strain background."""
    result = _rows(_records(built)).assembly_pin()
    assert result.passed
    assert result.details["pins"] == [m.EXPECTED_ASSEMBLY_PIN]


def test_the_assembly_pin_fails_on_a_second_pin(
    built: m.CrispriGuideFitnessWang2018Dataset,
) -> None:
    """Two assembly pins in one dataset means a record is keyed to the wrong genome."""
    records = _records(built)
    records[0]["reference"]["genome_reference"]["assembly_set"] = (
        "ecoli_K12_BW25113_ASM75055v1"
    )
    assert not _rows(records).assembly_pin().passed


def test_the_three_supplementary_rows_come_back_in_level_order(
    built: m.CrispriGuideFitnessWang2018Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``results`` is what ``verify_build`` appends, so its shape is pinned here."""
    monkeypatch.setattr(m, "SCREENS", SYNTHETIC_SCREENS)
    results = _rows(_records(built)).results()
    assert [int(r.level) for r in results] == [1, 1, 3]
    assert all(r.passed for r in results)


# --------------------------------------------------------------------------- #
# Sourced values
# --------------------------------------------------------------------------- #
SOURCED_VALUES: dict[str, SourcedValue] = {
    name: value for name, value in vars(m).items() if isinstance(value, SourcedValue)
}


def test_every_sourced_value_in_the_module_is_audited() -> None:
    """The module's sourcing layer is not empty, and this test enumerates it."""
    assert len(SOURCED_VALUES) >= 20


@pytest.mark.parametrize("name", sorted(SOURCED_VALUES))
def test_a_sourced_value_carries_a_quote_and_a_pinned_provenance(name: str) -> None:
    """Every value the schema needs names the pinned bytes and quotes them."""
    value = SOURCED_VALUES[name]
    assert value.provenance.citation_key == m.CITATION_KEY
    assert value.provenance.sha256 == m.PAPER_MD_SHA256
    assert value.provenance.source_uri == m.PAPER_MD
    assert value.provenance.page
    assert value.quote


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
    Path(DATA_ROOT) / "data/torchcell/crispri_guide_fitness_wang2018"
    if DATA_ROOT
    else Path("absent")
)
needs_mirror = pytest.mark.skipif(
    not MIRROR.exists(), reason="the Wang 2018 raw mirror is not deposited here"
)
needs_store = pytest.mark.skipif(
    not (STORE / "processed/lmdb").exists(),
    reason="the Wang 2018 dev-tree LMDB is not built here",
)


LIBRARY = Path(DATA_ROOT) / "torchcell-library" if DATA_ROOT else Path("absent")
needs_library = pytest.mark.skipif(
    not (LIBRARY / m.CITATION_KEY / m.PAPER_MD).exists(),
    reason="the Wang 2018 paper OCR is not mirrored here",
)


@pytest.mark.data
@needs_library
@pytest.mark.parametrize("name", sorted(SOURCED_VALUES))
def test_every_quote_is_still_verbatim_in_the_pinned_ocr(name: str) -> None:
    """The sha256 still matches and the quote is still a substring of those bytes."""
    result = audit_sourced_value(SOURCED_VALUES[name], LIBRARY)
    assert result.passed, result.message


@pytest.mark.data
@needs_mirror
def test_the_mirror_pins_every_consumed_file_to_its_recorded_retrieval() -> None:
    """Seven files, each with a re-runnable ``pmc_cloud`` retrieval and its sha256."""
    manifest = m.load_manifest(DATA_ROOT)
    assert {f.path for f in manifest.files} == {r.relpath for r in m.RAW_FILES}
    for raw in m.RAW_FILES:
        assert m.manifest_sha256(manifest, raw.relpath) == raw.sha256
        record = next(f for f in manifest.files if f.path == raw.relpath)
        assert record.retrieval is not None
        assert record.retrieval.method is RetrievalMethod.pmc_cloud
        assert record.retrieval.params == {"key": raw.cloud_key}


@pytest.mark.data
@needs_store
def test_the_built_store_holds_the_measured_record_count() -> None:
    """240,481 records over 4,218 genes, with the per-screen arithmetic balancing."""
    payload = json.loads((STORE / "preprocess/build_accounting.json").read_text())
    accounting = m.BuildAccounting.model_validate(payload)
    accounting.check()
    assert accounting.source_rows == m.EXPECTED_SOURCE_ROWS
    assert accounting.kept_records == m.EXPECTED_RECORDS
    assert accounting.perturbations == m.EXPECTED_PERTURBATIONS
    assert accounting.distinct_genes == m.EXPECTED_GENES
    assert accounting.retired_targets_dropped == ["b4590", "b4629", "b4635", "b4700"]
    assert [s.nc_sigma for s in accounting.screens] == [s.nc_sigma for s in m.SCREENS]


@pytest.mark.data
@needs_store
def test_the_three_drop_rules_account_for_every_released_row() -> None:
    """1,942 controls, 4,880 Bad rows (10 of them both) and 55 retired targets."""
    payload = json.loads((STORE / "preprocess/build_accounting.json").read_text())
    accounting = m.BuildAccounting.model_validate(payload)
    assert sum(s.dropped_non_targeting for s in accounting.screens) == 1942
    assert sum(s.dropped_bad_quality for s in accounting.screens) == 4880
    assert sum(s.dropped_non_targeting_and_bad for s in accounting.screens) == 10
    assert sum(s.dropped_retired_target for s in accounting.screens) == 55


@pytest.mark.data
@needs_store
def test_every_member_bnumber_resolves_against_the_pinned_assembly() -> None:
    """4,310 current, 3 pseudogene loci and 4 retired over 4,317 names."""
    payload = json.loads((STORE / "preprocess/build_accounting.json").read_text())
    accounting = m.BuildAccounting.model_validate(payload)
    histogram = {
        GeneNameStatus(status).value: count
        for status, count in accounting.reconciliation.status_histogram.items()
    }
    assert histogram["current"] == 4310
    assert histogram["non_gene_feature"] == 3
    assert histogram["retired"] == 4
    assert accounting.reconciliation.remapped == 0
    assert accounting.reconciliation.outside_namespace == ()


@pytest.mark.data
@needs_store
def test_the_verification_report_passes_l0_to_l4() -> None:
    """The verifier's own report, as the build wrote it."""
    report = VerificationReport.model_validate_json(
        (STORE / "preprocess/verification_report.json").read_text()
    )
    assert isinstance(report.provenance, Provenance)
    failures = [r.name for r in report.results if not r.passed]
    assert failures == []
