# tests/torchcell/datasets/ecoli/test_rapp2026_platforms.py
# [[tests.torchcell.datasets.ecoli.test_rapp2026_platforms]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_rapp2026_platforms.py
"""Rapp 2026's three other quantity families
(``torchcell.datasets.ecoli.rapp2026_platforms``).

Synthetic tests (run everywhere) write every consumed workbook into ``tmp_path`` and
build all three families against the real ``EcoliK12MG1655Genome`` over the synthetic
MG1655 assembly of ``tests/torchcell/sequence/genome/_bacterial_fixtures.py``, with two
synonyms added: ``b0099`` on ``b0005`` ``proB`` alone, which the ``locus_tag_synonym``
route describes, and ``b0098`` on BOTH ``b0006`` ``proC`` and the pseudogene ``b0004``
``yaaP``, which no route describes. ``verify_raw_files`` is replaced by a presence check
(synthetic bytes cannot carry the real pins, which the data-gated tests assert instead).
The synthetic screen:

    strain   b-number   outcome
    thrL     b0001      kept
    thrA     b0002      kept
    proC     b0006      kept (growth only; it has no accumulating feature)
    ghostG   b0099      kept: one carrier, so the record stores b0005 with a
                        locus_tag_synonym DerivedIdentifierMapping
    ctrl1    b0000      the growth denominator / a released control token
    ctrl2    b0000      the growth denominator

``b0098`` is released by no strain of the screen; the ``resolve_genes`` test builds it
directly, so the remap drop rule keeps a case.

Growth curves are flat, so a culture's trapezoid AUC is its OD600 times the 30 h axis
and the growth-defect split is exactly one strain (``proB`` at OD 0.3, AUC 9 < 18).
The two accumulation tables are built from one batch median per (feature, batch), so
``Mean_Int / Mean_FC`` is that median by construction, which is what the build checks.

Data-gated tests (``@pytest.mark.data``) read the real mirror and the three built
dev-tree LMDBs under ``$DATA_ROOT`` (they never build them): the two new mirror pins,
the provenance audit of every sourced value, the record counts, the measured platform
agreement, the back-solved batch medians, two hand-checked records, and the L0-L4
verifiers.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pytest

import torchcell.datasets.ecoli.rapp2026 as metabolome
import torchcell.datasets.ecoli.rapp2026_platforms as m
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    MG1655_GAF,
    MG1655_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.datamodels.schema import (
    CrisprInterferencePerturbation,
    SampleUnit,
    StrainReferenceGenome,
    UncertaintyType,
)
from torchcell.datasets.ecoli.rapp2026 import MIN_RESOLVED_FRACTION, DropRule
from torchcell.sequence.genome.ecoli.k12 import MG1655_ASSEMBLY, EcoliK12MG1655Genome
from torchcell.verification.sourced import audit_sourced_value

# --------------------------------------------------------------------------- #
# The synthetic screen
# --------------------------------------------------------------------------- #
#: ``b0099`` on ``b0005`` alone is the recoverable ``locus_tag_synonym`` case; ``b0098``
#: on both ``b0006`` and the pseudogene ``b0004`` is the case no route describes.
_EXTRA_SYNONYMS = {
    "b0004": ("ECK0004", "b0098"),
    "b0005": ("ECK0005", "b0099"),
    "b0006": ("ECK0006", "Pro2", "b0098"),
}
SYNTHETIC_LOCI = [
    locus.model_copy(update={"synonyms": _EXTRA_SYNONYMS[locus.tag]})
    if locus.tag in _EXTRA_SYNONYMS
    else locus
    for locus in MG1655_LOCI
]

#: ``(gene, sgRNA id, b-number, spacer)``.
GUIDES: list[tuple[str, str, str, str]] = [
    ("thrL", "thrL #1", "b0001", "ACGTACGTACGTACGTACGT"),
    ("thrA", "thrA #2", "b0002", "TTTTCCCCGGGGAAAATTTT"),
    ("proC", "proC #1", "b0006", "GGGGAAAATTTTCCCCGGGG"),
    ("ghostG", "ghostG #1", "b0099", "CCCCTTTTAAAAGGGGCCCC"),
]
#: ``(gene, b-number, plate, well, batch)``, one row per strain of Table S3.
STRAINS: list[tuple[str, str, str, str, int]] = [
    ("thrL", "b0001", "1", "A2", 1),
    ("thrA", "b0002", "1", "A3", 1),
    ("ghostG", "b0099", "2", "B4", 2),
    ("ctrl1", "b0000", "1", "A1", 1),
    ("ctrl2", "b0000", "2", "A1", 2),
]
#: ``(abbreviation, BiGG, names, KEGG, monoisotopic mass, formula)``.
METABOLITES: list[tuple[str, str, str, str, float, str]] = [
    ("ppal", "ppal", "Propanal", "C00479", 58.0419, "C3H6O"),
    (
        "ac-gcald",
        "ac-gcald",
        "Acetate; Glycolaldehyde",
        "C00033-NaN",
        60.0211,
        "C2H4O2",
    ),
    (
        "didp",
        "didp",
        "DIDP; 2'-deoxyinosine-5'-diphosphate(3-)",
        "C01344",
        412.0291,
        "C10H14N4O10P2",
    ),
]
#: Table S9 writes a missing KEGG id as ``XXX``; the accumulation tables write ``NaN``.
IDENTITY_KEGG = {
    abbreviation: kegg.replace("NaN", "XXX")
    for abbreviation, _, _, kegg, _, _ in METABOLITES
}
#: Per-strain OD600, constant over the axis, so AUC = OD x 30 h.
GROWTH_OD: dict[str, tuple[float, float, float]] = {
    "thrL": (0.9, 0.9, 0.9),
    "thrA": (0.8, 0.85, 0.9),
    "proC": (0.3, 0.3, 0.3),
    "ghostG": (0.7, 0.7, 0.7),
    "ctrl1": (1.0, 1.0, 1.0),
    "ctrl2": (0.8, 0.8, 0.8),
}
#: ``(feature key, batch) -> the batch median intensity the fold changes divide by``.
BATCH_MEDIAN: dict[tuple[str, int], float] = {
    ("ppal[M+H]+", 1): 10_000.0,
    ("ppal[M+H]+", 2): 20_000.0,
    ("didp[M-H]-", 1): 5_000.0,
    ("ac-gcald[M+H]+", 2): 8_000.0,
}
#: ``(gene, key, R1 fold change, R2 fold change)``, the released accumulating pairs.
PAIRS: list[tuple[str, str, float, float]] = [
    ("thrL", "ppal[M+H]+", 4.0, 6.0),
    ("thrL", "didp[M-H]-", 8.0, 12.0),
    ("thrA", "ppal[M+H]+", 10.0, 14.0),
    ("ghostG", "ppal[M+H]+", 5.0, 7.0),
    ("ghostG", "ac-gcald[M+H]+", 20.0, 30.0),
    ("ctrl1", "didp[M-H]-", 4.5, 5.5),
]
#: The targeted fold change of each pair, deliberately NOT the FI-MS one.
TARGETED_FOLD: dict[tuple[str, str], float] = {
    ("thrL", "ppal[M+H]+"): 3.0,
    ("thrL", "didp[M-H]-"): 15.0,
    ("thrA", "ppal[M+H]+"): 9.0,
    ("ghostG", "ppal[M+H]+"): 11.0,
    ("ghostG", "ac-gcald[M+H]+"): 18.0,
    ("ctrl1", "didp[M-H]-"): 6.0,
}
QC_PASSED = {("thrL", "ppal[M+H]+"), ("thrA", "ppal[M+H]+"), ("ctrl1", "didp[M-H]-")}
#: Which extract each targeted pair was injected from (default replicate 1): thrL's
#: didp comes from its second plate, so thrL carries two targeted records.
TARGETED_REPLICATE: dict[tuple[str, str], int] = {("thrL", "didp[M-H]-"): 2}

REFERENCE = StrainReferenceGenome(
    species="Escherichia coli",
    strain=metabolome.HOST_STRAIN,
    assembly_set=metabolome.MG1655_ASSEMBLY_SET,
    assembly_accession="GCA_000005845.2",
    background=metabolome.host_background(),
)


def _batch_of(gene: str) -> int:
    return next(row[4] for row in STRAINS if row[0] == gene)


def _well_of(gene: str) -> str:
    return next(row[3] for row in STRAINS if row[0] == gene)


def wells_map() -> dict[str, str]:
    """``{strain token: its Table S3 well}``, which every DataFile name carries."""
    return {gene: well for gene, _, _, well, _ in STRAINS}


def run_code(gene: str, replicate: int) -> str:
    """The synthetic FI-MS run code of ``gene``'s plate ``replicate`` (``msAV9NN``)."""
    index = [row[0] for row in STRAINS].index(gene)
    return f"msAV{900 + 2 * index + replicate - 1}"


def sample_id(gene: str, replicate: int) -> str:
    """The synthetic Table S3 ``Sample ID`` of ``gene``'s plate ``replicate``."""
    return f"{gene}_R{replicate}_{run_code(gene, replicate)}_B{_batch_of(gene)}"


def _plate_of(gene: str) -> str:
    return next(row[2] for row in STRAINS if row[0] == gene)


def _split_key(key: str) -> tuple[str, str]:
    index = key.index("[")
    return key[:index], key[index:]


def _mass_of(abbreviation: str) -> float:
    return next(row[4] for row in METABOLITES if row[0] == abbreviation)


def _released_kegg_of(abbreviation: str) -> str:
    return next(row[3] for row in METABOLITES if row[0] == abbreviation)


def _mz_of(abbreviation: str, adduct: str) -> float:
    mono = _mass_of(abbreviation)
    return mono + (
        metabolome.PROTON_MASS if adduct == "[M+H]+" else -metabolome.PROTON_MASS
    )


def sample_columns() -> list[str]:
    """The ten synthetic Table S3 sample ids, one FI-MS run code per plate replicate."""
    return [sample_id(gene, replicate) for gene, *_ in STRAINS for replicate in (1, 2)]


def growth_hours() -> list[float]:
    """The released 181-point axis: 0 to 30 h at 10 min spacing."""
    return [index / 6.0 for index in range(m.GROWTH_TIME_POINTS)]


def expected_auc(gene: str, replicate: int) -> float:
    """A flat curve's trapezoid AUC: its OD600 over the 30 h axis."""
    return GROWTH_OD[gene][replicate - 1] * m.GROWTH_LAST_HOUR


def control_baseline() -> float:
    """The grand mean AUC of the six synthetic control cultures."""
    return float(
        np.mean(
            [
                expected_auc(gene, replicate)
                for gene in ("ctrl1", "ctrl2")
                for replicate in (1, 2, 3)
            ]
        )
    )


# --------------------------------------------------------------------------- #
# Synthetic workbooks
# --------------------------------------------------------------------------- #
def _sheet(path: Path, title: str) -> tuple[Any, Any]:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = title
    return workbook, sheet


def _write_table_s1(path: Path) -> None:
    workbook, sheet = _sheet(path, metabolome.TABLE_S1_SHEET)
    sheet.append(
        ["Gene", "sgRNA Nr.", "b-Nr.", "base pairing region", "Oligo sequence"]
    )
    for gene, sgrna, b_number, spacer in GUIDES:
        sheet.append([gene, sgrna, b_number, spacer, f"gatc{spacer.lower()}gttt"])
    workbook.save(path)


def _write_table_s2(path: Path, *, blank_cell: bool = False) -> None:
    workbook, sheet = _sheet(path, metabolome.TABLE_S2_SHEET)
    hours = growth_hours()
    sheet.append([None] * 6 + ["Time (hours)"])
    sheet.append([*m.TABLE_S2_LABEL_COLUMNS, *hours])
    for index, (gene, _, b_number, _) in enumerate(GUIDES, start=1):
        for replicate in (1, 2, 3):
            values: list[float | None] = [GROWTH_OD[gene][replicate - 1]] * len(hours)
            if blank_cell and gene == "thrA" and replicate == 1:
                values[7] = None
            sheet.append(
                [
                    replicate,
                    index,
                    gene,
                    f"{gene} #1",
                    b_number,
                    f"P{index}_A1",
                    *values,
                ]
            )
    for well, gene in enumerate(("ctrl1", "ctrl2"), start=1):
        for replicate in (1, 2, 3):
            sheet.append(
                [
                    replicate,
                    100 + well,
                    None,
                    gene,
                    None,
                    f"P{well}_A1",
                    *([GROWTH_OD[gene][replicate - 1]] * len(hours)),
                ]
            )
    workbook.save(path)


def _write_table_s3(path: Path) -> None:
    workbook, sheet = _sheet(path, metabolome.TABLE_S3_SHEET)
    sheet.append(
        ["Target gene", "b number", "OD", "Plate ID", "Well", "Replicate", "Sample ID"]
    )
    for column in sample_columns():
        parsed = metabolome.parse_sample_id(column)
        gene, b_number, plate, well, _ = next(
            row for row in STRAINS if row[0] == parsed.gene
        )
        sheet.append([gene, b_number, 0.7, plate, well, f"R{parsed.replicate}", column])
    workbook.save(path)


def _write_table_s9(path: Path) -> None:
    workbook, sheet = _sheet(path, metabolome.TABLE_S9_SHEET)
    sheet.append(
        [
            "Abbreviation",
            "BIGG",
            "Metabolite",
            "KEGG",
            "Monoisotopic mass",
            "Neutral Formula",
        ]
    )
    for abbreviation, bigg, names, kegg, mass, formula in METABOLITES:
        sheet.append(
            [abbreviation, bigg, names, kegg.replace("NaN", "XXX"), mass, formula]
        )
    workbook.save(path)


def _legend_rows(columns: Sequence[m.SourcedColumn]) -> list[tuple[str, str]]:
    return [(column.column, column.quote) for column in columns]


def _write_table_s5(
    path: Path,
    *,
    pairs: Sequence[tuple[str, str, float, float]] | None = None,
    mean_int: float | None = None,
    legend: Sequence[tuple[str, str]] | None = None,
) -> None:
    workbook, sheet = _sheet(path, metabolome.TABLE_S5_SHEET)
    sheet.append(
        [
            "Gene",
            "Metabolite",
            "Metabolite Abbreviation",
            "Kegg ID",
            "Polarity",
            "Mode",
            "Mass",
            "MonoMass",
            "Mean_FC",
            "R1_FC",
            "R2_FC",
            "Mean_Int",
            "R1_Int",
            "R2_Int",
            "LC-MS/MS",
        ]
    )
    for gene, key, r1, r2 in pairs if pairs is not None else PAIRS:
        abbreviation, adduct = _split_key(key)
        median = BATCH_MEDIAN[(key, _batch_of(gene))]
        mean_fc = (r1 + r2) / 2
        sheet.append(
            [
                gene,
                next(row[2] for row in METABOLITES if row[0] == abbreviation),
                abbreviation,
                _released_kegg_of(abbreviation),
                "pos" if "+" in adduct else "neg",
                adduct,
                _mz_of(abbreviation, adduct),
                _mass_of(abbreviation),
                mean_fc,
                r1,
                r2,
                mean_int if mean_int is not None else median * mean_fc,
                median * r1,
                median * r2,
                1,
            ]
        )
    sheet = workbook.create_sheet(metabolome.LEGEND_SHEET)
    for column, quote in (
        legend if legend is not None else _legend_rows(m.INTENSITY_COLUMNS)
    ):
        sheet.append([column, quote])
    workbook.save(path)


def _write_table_s6(
    path: Path,
    *,
    datafile_well: str | None = None,
    datafile_code: str | None = None,
    legend: Sequence[tuple[str, str]] | None = None,
) -> None:
    workbook, sheet = _sheet(path, metabolome.TABLE_S6_SHEET)
    sheet.append(
        [
            "QC passed",
            "Gene",
            "Abbreviation",
            "Kegg",
            "Polarity",
            "Mode",
            "PrecMz",
            "Monoisotopic Mass",
            "fold-change",
            "Intensity PrecMz",
            "DataFile",
        ]
    )
    for gene, key, _, _ in PAIRS:
        abbreviation, adduct = _split_key(key)
        polarity = "pos" if "+" in adduct else "neg"
        well = datafile_well or _well_of(gene)
        code = datafile_code or run_code(gene, TARGETED_REPLICATE.get((gene, key), 1))
        sheet.append(
            [
                int((gene, key) in QC_PASSED),
                gene,
                abbreviation,
                _released_kegg_of(abbreviation),
                polarity,
                adduct,
                _mz_of(abbreviation, adduct),
                _mass_of(abbreviation),
                TARGETED_FOLD[(gene, key)],
                1_234.0,
                f"{abbreviation}_{gene}_P{_plate_of(gene)}{well}{code}_{polarity}.mzML",
            ]
        )
    sheet = workbook.create_sheet(metabolome.LEGEND_SHEET)
    for column, quote in (
        legend if legend is not None else _legend_rows(m.TARGETED_COLUMNS)
    ):
        sheet.append([column, quote])
    workbook.save(path)


def synthetic_agreement() -> dict[str, float]:
    """The platform agreement the synthetic tables imply, by the module's own formula."""
    fi_ms = np.asarray([(r1 + r2) / 2 for _, _, r1, r2 in PAIRS], dtype=np.float64)
    lc_ms = np.asarray(
        [TARGETED_FOLD[(gene, key)] for gene, key, _, _ in PAIRS], dtype=np.float64
    )
    return {
        "n_pairs": float(len(PAIRS)),
        "pearson_r_linear": m.pearson_r(fi_ms, lc_ms),
        "pearson_r_log2": m.pearson_r(np.log2(fi_ms), np.log2(lc_ms)),
        "median_abs_log2_difference": float(
            np.median(np.abs(np.log2(lc_ms) - np.log2(fi_ms)))
        ),
    }


def _samples(raw_dir: Path) -> dict[str, metabolome.SampleRow]:
    """The synthetic Table S3, parsed."""
    return metabolome.read_sample_rows(raw_dir / metabolome.TABLE_S3)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def mg1655(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12MG1655Genome:
    """The real MG1655 class over the synthetic assembly; network refused."""
    files = write_assembly(
        tmp_path / "tier", MG1655_ASSEMBLY, SYNTHETIC_LOCI, gaf_rows=MG1655_GAF
    )
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
        missing = [name for name in pins if not osp.exists(osp.join(raw_dir, name))]
        if missing:
            raise FileNotFoundError(f"pinned raw files absent: {missing}")
        calls.append(dict(pins))

    monkeypatch.setattr(m, "verify_raw_files", record)
    return calls


@pytest.fixture
def small_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The released counts, scaled to the synthetic screen."""
    monkeypatch.setattr(metabolome, "LIBRARY_GENES", len(GUIDES))
    monkeypatch.setattr(metabolome, "N_EXTRACTS", len(sample_columns()))
    monkeypatch.setattr(metabolome, "ISOBARIC_METABOLITES", len(METABOLITES))
    monkeypatch.setattr(m, "LIBRARY_GENES", len(GUIDES))
    monkeypatch.setattr(m, "TABLE_S2_ROWS", 3 * (len(GUIDES) + 2))
    monkeypatch.setattr(m, "GROWTH_CONTROL_TOKENS", 2)
    monkeypatch.setattr(m, "MEASURED_GROWTH_DEFECT", 1)
    monkeypatch.setattr(m, "PAPER_GROWTH_DEFECT", 1)
    monkeypatch.setattr(m, "PAPER_NO_GROWTH_DEFECT", len(GUIDES) - 1)
    monkeypatch.setattr(m, "INTENSITY_ROWS", len(PAIRS))
    monkeypatch.setattr(m, "TARGETED_PAIRS", len(PAIRS))
    monkeypatch.setattr(m, "TARGETED_QC_PASSED", len(QC_PASSED))
    monkeypatch.setattr(m, "ACCUMULATION_TOKENS", 4)
    monkeypatch.setattr(m, "ACCUMULATION_CONTROL_TOKENS", 1)
    monkeypatch.setattr(m, "EXPECTED_PLATFORM_AGREEMENT", synthetic_agreement())


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Every synthetic workbook, in one directory the dataset roots link to."""
    raw = tmp_path / "workbooks"
    raw.mkdir()
    _write_table_s1(raw / metabolome.TABLE_S1)
    _write_table_s2(raw / metabolome.TABLE_S2)
    _write_table_s3(raw / metabolome.TABLE_S3)
    _write_table_s5(raw / metabolome.TABLE_S5)
    _write_table_s6(raw / metabolome.TABLE_S6)
    _write_table_s9(raw / metabolome.TABLE_S9)
    return raw


@pytest.fixture
def synthetic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_dir: Path,
    small_counts: None,
    mg1655: EcoliK12MG1655Genome,
    presence_only_pins: list[Mapping[str, str]],
) -> Any:
    """A factory for one family's dataset over the synthetic workbooks."""
    monkeypatch.setattr(m, "assembly_reference", lambda strain, **_: REFERENCE)

    def build(family: str) -> Any:
        dataset_cls, _, _ = m.FAMILIES[family]
        root = tmp_path / family
        (root / "raw").mkdir(parents=True)
        for name in dataset_cls.CONSUMED_FILES:
            os.symlink(raw_dir / name, root / "raw" / name)
        return dataset_cls(root=str(root), ecoli_genome=mg1655)

    return build


# --------------------------------------------------------------------------- #
# Table S2 and the growth statistic
# --------------------------------------------------------------------------- #
def test_read_growth_table_reads_the_axis_the_cultures_and_the_controls(
    raw_dir: Path, small_counts: None
) -> None:
    table = m.read_growth_table(raw_dir / metabolome.TABLE_S2)
    assert len(table.hours) == m.GROWTH_TIME_POINTS
    assert (table.hours[0], table.hours[-1]) == (0.0, m.GROWTH_LAST_HOUR)
    assert len(table.rows) == m.TABLE_S2_ROWS
    assert table.matrix.shape == (m.TABLE_S2_ROWS, m.GROWTH_TIME_POINTS)
    controls = [row for row in table.rows if row.is_control]
    assert len(controls) == 6
    assert {row.token for row in controls} == {"ctrl1", "ctrl2"}
    assert [row.token for row in table.rows if row.replicate == 1][:4] == [
        "thrL",
        "thrA",
        "proC",
        "ghostG",
    ]


def test_read_growth_table_refuses_a_missing_od_cell(
    tmp_path: Path, small_counts: None
) -> None:
    path = tmp_path / "blank.xlsx"
    _write_table_s2(path, blank_cell=True)
    with pytest.raises(ValueError, match="1 non-finite OD600 cells"):
        m.read_growth_table(path)


def test_read_growth_table_refuses_a_wrong_row_count(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch, small_counts: None
) -> None:
    monkeypatch.setattr(m, "TABLE_S2_ROWS", 7)
    with pytest.raises(ValueError, match="holds 18 rows, expected 7"):
        m.read_growth_table(raw_dir / metabolome.TABLE_S2)


def test_strain_growth_is_the_trapezoid_auc_of_each_released_culture(
    raw_dir: Path, small_counts: None
) -> None:
    strains = m.strain_growth(m.read_growth_table(raw_dir / metabolome.TABLE_S2))
    by_token = {strain.token: strain for strain in strains}
    assert sorted(by_token) == ["ctrl1", "ctrl2", "ghostG", "proC", "thrA", "thrL"]
    assert by_token["thrL"].areas == pytest.approx((27.0, 27.0, 27.0))
    assert by_token["thrA"].mean_area == pytest.approx(25.5)
    assert by_token["ctrl1"].is_control and not by_token["thrL"].is_control
    assert by_token["proC"].guide_id == "proC #1"


def test_check_growth_defects_reproduces_the_released_split(
    raw_dir: Path, small_counts: None
) -> None:
    table = m.read_growth_table(raw_dir / metabolome.TABLE_S2)
    check = m.check_growth_defects(table, m.strain_growth(table))
    assert (check.measured_growth_defect, check.measured_no_growth_defect) == (1, 3)
    assert check.measured_from_mean_curve == 1
    assert check.strain_difference == 0
    assert (check.auc_minimum, check.auc_maximum) == pytest.approx((9.0, 27.0))


def test_check_growth_defects_refuses_a_different_measured_count(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch, small_counts: None
) -> None:
    monkeypatch.setattr(m, "MEASURED_GROWTH_DEFECT", 2)
    table = m.read_growth_table(raw_dir / metabolome.TABLE_S2)
    with pytest.raises(ValueError, match="1 strains fall below the AUC cutoff"):
        m.check_growth_defects(table, m.strain_growth(table))


def test_check_growth_defects_refuses_a_library_of_another_size(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch, small_counts: None
) -> None:
    monkeypatch.setattr(m, "LIBRARY_GENES", 9)
    table = m.read_growth_table(raw_dir / metabolome.TABLE_S2)
    with pytest.raises(ValueError, match="holds 4 library genes, the paper states 9"):
        m.check_growth_defects(table, m.strain_growth(table))


def test_fitness_phenotype_reports_the_sample_sd_of_the_replicate_ratios() -> None:
    phenotype = m.fitness_phenotype([0.9, 1.0, 1.1], 3)
    assert phenotype.fitness == pytest.approx(1.0)
    assert phenotype.fitness_uncertainty == pytest.approx(0.1)
    assert phenotype.fitness_uncertainty_type == UncertaintyType.sample_sd
    assert phenotype.fitness_se == pytest.approx(0.1 / np.sqrt(3))
    assert phenotype.n_samples == 3
    assert phenotype.sample_unit == SampleUnit.biological_replicate
    with pytest.raises(ValueError, match="2 ratios for n_samples=3"):
        m.fitness_phenotype([1.0, 1.0], 3)


# --------------------------------------------------------------------------- #
# The column legends
# --------------------------------------------------------------------------- #
def test_read_column_legends_reads_the_released_descriptions(raw_dir: Path) -> None:
    legends = m.read_column_legends(raw_dir / metabolome.TABLE_S6)
    assert legends["fold-change"] == (
        "Intensity of highest peak in EIC of the strain compared to median peak "
        "intensity of all other strains."
    )


def test_check_column_legends_passes_on_the_released_text(raw_dir: Path) -> None:
    checked = m.check_column_legends(raw_dir / metabolome.TABLE_S5, m.INTENSITY_COLUMNS)
    assert [column.column for column in checked] == [
        "Mean_Int",
        "R1_Int+R2_Int",
        "R1_FC + R2_FC",
        "LC-MS/MS",
    ]
    assert {column.sha256 for column in checked} == {
        metabolome.DATA_SHA256[metabolome.TABLE_S5]
    }


def test_check_column_legends_refuses_an_absent_column(tmp_path: Path) -> None:
    path = tmp_path / "s6.xlsx"
    _write_table_s6(path, legend=_legend_rows(m.TARGETED_COLUMNS)[1:])
    with pytest.raises(ValueError, match="has no row for column 'fold-change'"):
        m.check_column_legends(path, m.TARGETED_COLUMNS)


def test_check_column_legends_refuses_a_reworded_legend(tmp_path: Path) -> None:
    path = tmp_path / "s5.xlsx"
    rows = _legend_rows(m.INTENSITY_COLUMNS)
    _write_table_s5(path, legend=[(rows[0][0], "Mean intensity."), *rows[1:]])
    with pytest.raises(ValueError, match="row 'Mean_Int' reads 'Mean intensity.'"):
        m.check_column_legends(path, m.INTENSITY_COLUMNS)


def test_read_column_legends_refuses_a_sheet_of_another_shape(tmp_path: Path) -> None:
    path = tmp_path / "wide.xlsx"
    workbook, sheet = _sheet(path, metabolome.LEGEND_SHEET)
    sheet.append(["a", "b", "c"])
    workbook.save(path)
    with pytest.raises(ValueError, match="has 3 columns, expected 2"):
        m.read_column_legends(path)


# --------------------------------------------------------------------------- #
# The accumulation tables
# --------------------------------------------------------------------------- #
def test_released_kegg_normalizes_both_spellings_of_a_missing_id() -> None:
    assert m._released_kegg("C00033-NaN") == "C00033-XXX"
    assert m._released_kegg("nan") == "XXX"
    assert m._released_kegg("C00479") == "C00479"


def test_read_intensities_stores_the_mean_and_back_solves_the_batch_median(
    raw_dir: Path, small_counts: None
) -> None:
    metabolites = metabolome.read_metabolites(raw_dir / metabolome.TABLE_S9)
    samples = metabolome.read_sample_rows(raw_dir / metabolome.TABLE_S3)
    sizes = m.batch_sizes(samples)
    reference_n = {
        gene: sizes[batch] for gene, batch in m.batch_of_gene(samples).items()
    }
    items = m.read_intensities(
        raw_dir / metabolome.TABLE_S5, metabolites, reference_n, samples
    )
    assert len(items) == len(PAIRS)
    first = next(item for item in items if (item.gene, item.key) == PAIRS[0][:2])
    assert first.value == pytest.approx(10_000.0 * 5.0)
    assert first.replicates == pytest.approx((40_000.0, 60_000.0))
    assert first.sample_ids == (sample_id("thrL", 1), sample_id("thrL", 2))
    assert first.reference_level == pytest.approx(10_000.0)
    assert first.reference_n == 6
    assert first.monoisotopic_mass == pytest.approx(58.0419)
    assert first.kegg == "C00479"
    assert not first.is_control
    assert next(item for item in items if item.gene == "ctrl1").is_control


def test_read_intensities_refuses_a_mean_that_is_not_the_mean_of_the_plates(
    tmp_path: Path, raw_dir: Path, monkeypatch: pytest.MonkeyPatch, small_counts: None
) -> None:
    metabolites = metabolome.read_metabolites(raw_dir / metabolome.TABLE_S9)
    monkeypatch.setattr(m, "INTENSITY_ROWS", 1)
    path = tmp_path / "s5.xlsx"
    _write_table_s5(path, pairs=PAIRS[:1], mean_int=1.0)
    with pytest.raises(ValueError, match="Mean_Int 1.0 is not the mean of"):
        m.read_intensities(path, metabolites, {"thrL": 2}, _samples(raw_dir))


def test_read_intensities_refuses_a_row_count_the_release_does_not_have(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch, small_counts: None
) -> None:
    metabolites = metabolome.read_metabolites(raw_dir / metabolome.TABLE_S9)
    monkeypatch.setattr(m, "INTENSITY_ROWS", 2)
    with pytest.raises(ValueError, match="holds 6 rows, expected 2"):
        m.read_intensities(
            raw_dir / metabolome.TABLE_S5,
            metabolites,
            dict.fromkeys(wells_map(), 2),
            _samples(raw_dir),
        )


def test_read_intensities_refuses_a_flag_count_table_s6_does_not_match(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch, small_counts: None
) -> None:
    metabolites = metabolome.read_metabolites(raw_dir / metabolome.TABLE_S9)
    monkeypatch.setattr(m, "TARGETED_PAIRS", 3)
    with pytest.raises(ValueError, match="flags 6 rows for targeted LC-MS/MS"):
        m.read_intensities(
            raw_dir / metabolome.TABLE_S5,
            metabolites,
            dict.fromkeys(wells_map(), 2),
            _samples(raw_dir),
        )


def test_check_batch_medians_requires_one_median_per_feature_and_batch(
    raw_dir: Path, small_counts: None
) -> None:
    metabolites = metabolome.read_metabolites(raw_dir / metabolome.TABLE_S9)
    samples = metabolome.read_sample_rows(raw_dir / metabolome.TABLE_S3)
    sizes = m.batch_sizes(samples)
    batch_of = m.batch_of_gene(samples)
    items = m.read_intensities(
        raw_dir / metabolome.TABLE_S5,
        metabolites,
        {gene: sizes[batch] for gene, batch in batch_of.items()},
        samples,
    )
    check = m.check_batch_medians(items, batch_of, sizes)
    assert check.batch_sizes == {1: 6, 2: 4}
    assert check.n_groups == 4
    assert check.n_multi_strain_groups == 2
    assert check.max_relative_spread == pytest.approx(0.0, abs=1e-12)

    broken = [
        items[0].model_copy(update={"reference_level": items[0].reference_level * 2})
    ] + list(items[1:])
    with pytest.raises(ValueError, match="varies within a \\(feature, batch\\) group"):
        m.check_batch_medians(broken, batch_of, sizes)


def test_read_targeted_reads_the_fold_change_its_qc_flag_and_its_datafile(
    raw_dir: Path, small_counts: None
) -> None:
    metabolites = metabolome.read_metabolites(raw_dir / metabolome.TABLE_S9)
    items = m.read_targeted(
        raw_dir / metabolome.TABLE_S6, metabolites, _samples(raw_dir)
    )
    assert len(items) == len(PAIRS)
    first = items[0]
    assert (first.gene, first.key) == ("thrL", "ppal[M+H]+")
    assert first.value == pytest.approx(3.0)
    assert first.replicates == (3.0,)
    assert first.sample_ids == (sample_id("thrL", 1),)
    second = next(item for item in items if item.key == "didp[M-H]-")
    assert second.sample_ids == (sample_id("thrL", 2),)
    assert first.reference_level == 1.0
    assert first.reference_n == 3
    assert first.qc_passed is True
    assert first.datafile is not None and first.datafile.endswith("_pos.mzML")
    assert sum(1 for item in items if item.qc_passed) == len(QC_PASSED)


def test_read_targeted_refuses_a_datafile_naming_another_well(
    tmp_path: Path, raw_dir: Path, small_counts: None
) -> None:
    metabolites = metabolome.read_metabolites(raw_dir / metabolome.TABLE_S9)
    path = tmp_path / "s6.xlsx"
    _write_table_s6(path, datafile_well="H9")
    with pytest.raises(ValueError, match="is well 'H9', but Table S3 puts"):
        m.read_targeted(path, metabolites, _samples(raw_dir))


def test_read_targeted_refuses_a_run_code_no_sample_of_the_strain_carries(
    tmp_path: Path, raw_dir: Path, small_counts: None
) -> None:
    """The DataFile's run code names the measured extract; a code no Table S3 sample
    of that strain carries leaves the extract unknown, which is refused.
    """
    metabolites = metabolome.read_metabolites(raw_dir / metabolome.TABLE_S9)
    path = tmp_path / "s6.xlsx"
    _write_table_s6(path, datafile_code="msAV999")
    with pytest.raises(ValueError, match="carries run code 'msAV999', which no"):
        m.read_targeted(path, metabolites, _samples(raw_dir))


def test_sample_of_run_code_refuses_a_code_two_samples_of_a_strain_share(
    raw_dir: Path, small_counts: None
) -> None:
    samples = dict(_samples(raw_dir))
    assert m.sample_of_run_code(samples)[("thrL", "msAV901")] == sample_id("thrL", 2)
    key, row = next(iter(samples.items()))
    twin = row.model_copy(
        update={
            "sample": row.sample.model_copy(update={"column": "twin", "replicate": 2})
        }
    )
    samples["twin"] = twin
    with pytest.raises(ValueError, match="gives \\('thrL', 'msAV900'\\) two samples"):
        m.sample_of_run_code(samples)


def test_read_targeted_refuses_more_metabolites_per_strain_than_the_method_states(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch, small_counts: None
) -> None:
    metabolites = metabolome.read_metabolites(raw_dir / metabolome.TABLE_S9)
    monkeypatch.setattr(m, "TARGETED_MAX_PER_STRAIN", 1)
    with pytest.raises(ValueError, match="above the stated maximum of 1"):
        m.read_targeted(raw_dir / metabolome.TABLE_S6, metabolites, _samples(raw_dir))


def test_read_targeted_refuses_a_qc_count_the_paper_does_not_state(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch, small_counts: None
) -> None:
    metabolites = metabolome.read_metabolites(raw_dir / metabolome.TABLE_S9)
    monkeypatch.setattr(m, "TARGETED_QC_PASSED", 1)
    with pytest.raises(ValueError, match="marks 3 rows QC passed"):
        m.read_targeted(raw_dir / metabolome.TABLE_S6, metabolites, _samples(raw_dir))


def test_identity_join_refuses_a_mass_a_polarity_and_a_kegg_that_disagree(
    raw_dir: Path, small_counts: None
) -> None:
    metabolites = metabolome.read_metabolites(raw_dir / metabolome.TABLE_S9)

    def join(
        *,
        abbreviation: str = "ppal",
        adduct: str = "[M+H]+",
        polarity: str = "pos",
        mz: float = _mz_of("ppal", "[M+H]+"),
        monoisotopic_mass: float = 58.0419,
        kegg: str = "C00479",
    ) -> Any:
        return m._check_identity(
            1,
            "ppal[M+H]+",
            abbreviation,
            adduct,
            polarity,
            mz,
            monoisotopic_mass,
            kegg,
            metabolites,
        )

    assert join().bigg_ids == ("ppal",)
    with pytest.raises(ValueError, match="which Table S9 does not carry"):
        join(abbreviation="nope")
    with pytest.raises(ValueError, match="releases monoisotopic mass 1.0"):
        join(monoisotopic_mass=1.0)
    with pytest.raises(ValueError, match="is polarity 'neg', but adduct"):
        join(polarity="neg")
    with pytest.raises(ValueError, match="m/z minus the monoisotopic mass is"):
        join(mz=100.0)
    with pytest.raises(ValueError, match="releases KEGG 'C99999'"):
        join(kegg="C99999")


def test_pearson_r_is_one_on_a_perfect_line() -> None:
    x = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert m.pearson_r(x, 3.0 * x + 1.0) == pytest.approx(1.0)
    assert m.pearson_r(x, -x) == pytest.approx(-1.0)


def test_check_platform_agreement_refuses_a_number_other_than_the_pin(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch, small_counts: None
) -> None:
    metabolites = metabolome.read_metabolites(raw_dir / metabolome.TABLE_S9)
    samples = metabolome.read_sample_rows(raw_dir / metabolome.TABLE_S3)
    sizes = m.batch_sizes(samples)
    reference_n = {
        gene: sizes[batch] for gene, batch in m.batch_of_gene(samples).items()
    }
    intensities = m.read_intensities(
        raw_dir / metabolome.TABLE_S5, metabolites, reference_n, samples
    )
    targeted = m.read_targeted(
        raw_dir / metabolome.TABLE_S6, metabolites, _samples(raw_dir)
    )
    agreement = m.check_platform_agreement(targeted, intensities)
    assert agreement.n_pairs == len(PAIRS)
    assert agreement.pearson_r_linear == pytest.approx(
        synthetic_agreement()["pearson_r_linear"]
    )

    monkeypatch.setattr(
        m,
        "EXPECTED_PLATFORM_AGREEMENT",
        {**synthetic_agreement(), "pearson_r_linear": 0.1},
    )
    with pytest.raises(ValueError, match="pearson_r_linear is"):
        m.check_platform_agreement(targeted, intensities)

    with pytest.raises(ValueError, match="Table S6 pairs are not in Table S5"):
        m.check_platform_agreement(targeted, intensities[1:])


def test_extract_values_split_a_strain_by_the_extract_each_pair_was_measured_in(
    raw_dir: Path, small_counts: None
) -> None:
    """#1178: thrL's ppal was injected from plate 1 and its didp from plate 2, so thrL
    carries two targeted records; every record is one extract, n = 1, SE gapped.
    """
    metabolites = metabolome.read_metabolites(raw_dir / metabolome.TABLE_S9)
    items = m.read_targeted(
        raw_dir / metabolome.TABLE_S6, metabolites, _samples(raw_dir)
    )
    groups = m.extract_values(items)
    assert [(group.token, group.sample_id) for group in groups] == [
        ("ctrl1", sample_id("ctrl1", 1)),
        ("ghostG", sample_id("ghostG", 1)),
        ("thrA", sample_id("thrA", 1)),
        ("thrL", sample_id("thrL", 1)),
        ("thrL", sample_id("thrL", 2)),
    ]
    assert [group.is_control for group in groups] == [True, False, False, False, False]
    target_ids = m.target_metabolite_ids(items, metabolites)
    first, second = groups[3], groups[4]
    phenotype = m.metabolite_phenotype(first, m.TARGETED_MEASUREMENT_TYPE, target_ids)
    assert phenotype.metabolite_level == {"ppal[M+H]+": 3.0}
    assert phenotype.metabolite_level_se is None
    assert phenotype.n_replicates == {"ppal[M+H]+": 1}
    assert phenotype.replicate_id == sample_id("thrL", 1)
    assert phenotype.target_metabolite_ids == {"ppal[M+H]+": "ppal"}
    (gap,) = phenotype.provenance_gaps
    assert (gap.field, gap.reason) == (
        "metabolite_level_se",
        "not_applicable_single_measurement",
    )
    assert m.metabolite_phenotype(
        second, m.TARGETED_MEASUREMENT_TYPE, target_ids
    ).metabolite_level == {"didp[M-H]-": 15.0}
    reference = m.reference_phenotype(
        [*first.items, *second.items], m.TARGETED_MEASUREMENT_TYPE, target_ids
    )
    assert set(reference.metabolite_level.values()) == {1.0}
    assert reference.replicate_id is None
    assert reference.n_replicates == {"ppal[M+H]+": 3, "didp[M-H]-": 2}

    intensities = m.read_intensities(
        raw_dir / metabolome.TABLE_S5,
        metabolites,
        dict.fromkeys(wells_map(), 2),
        _samples(raw_dir),
    )
    plates = [group for group in m.extract_values(intensities) if group.token == "thrL"]
    assert [group.sample_id for group in plates] == [
        sample_id("thrL", 1),
        sample_id("thrL", 2),
    ]
    assert [group.values for group in plates] == [
        (40_000.0, 40_000.0),
        (60_000.0, 60_000.0),
    ]
    broken = intensities[0].model_copy(update={"sample_ids": ("only-one",)})
    with pytest.raises(ValueError, match="2 values for 1 sample ids"):
        m.extract_values([broken])


def test_target_metabolite_ids_skips_a_merged_isobaric_group(
    raw_dir: Path, small_counts: None
) -> None:
    metabolites = metabolome.read_metabolites(raw_dir / metabolome.TABLE_S9)
    items = m.read_targeted(
        raw_dir / metabolome.TABLE_S6, metabolites, _samples(raw_dir)
    )
    target_ids = m.target_metabolite_ids(items, metabolites)
    assert "ac-gcald[M+H]+" not in target_ids
    assert target_ids["ppal[M+H]+"] == "ppal"


# --------------------------------------------------------------------------- #
# Strain resolution
# --------------------------------------------------------------------------- #
def test_resolve_genes_keeps_a_one_carrier_synonym_on_a_typed_route(
    raw_dir: Path, mg1655: EcoliK12MG1655Genome, small_counts: None
) -> None:
    guides = metabolome.read_guides(raw_dir / metabolome.TABLE_S1)
    kept, rule, ledger = m.resolve_genes(
        mg1655, ["thrL", "thrA", "proC", "ghostG"], guides, label="synthetic"
    )
    assert [item.gene for item in kept] == ["thrL", "thrA", "proC", "ghostG"]
    assert [item.locus_tag for item in kept] == ["b0001", "b0002", "b0006", "b0005"]
    assert [item.identifier_mapping for item in kept[:3]] == [None, None, None]
    ghost = kept[3].identifier_mapping
    assert ghost is not None
    assert ghost.model_dump() == {
        "source_identifier": "b0099",
        "route": "locus_tag_synonym",
    }
    assert rule.rule == "b_number_remapped_by_the_annotation"
    assert rule.n_records == 0
    assert rule.items == []
    assert ledger.locus_tag_synonyms == [
        "ghostG (b0099): the annotation carries it as a gene synonym of current gene "
        "b0005, so the record stores b0005 with route locus_tag_synonym"
    ]
    assert ledger.reconciliation.unique_names == 4
    assert ledger.min_resolved_fraction == MIN_RESOLVED_FRACTION


def test_resolve_genes_still_drops_a_synonym_two_loci_list(
    raw_dir: Path, mg1655: EcoliK12MG1655Genome, small_counts: None
) -> None:
    """``b0098`` is a synonym of two loci, so no DerivedIdentifierRoute describes it."""
    guide = metabolome.Guide(
        gene="twinG",
        sgrna_id="twinG #1",
        b_number="b0098",
        spacer="AAAACCCCGGGGTTTTAAAA",
    )
    kept, rule, ledger = m.resolve_genes(
        mg1655, ["twinG"], {"twinG": guide}, label="ambiguous synonym"
    )
    assert kept == []
    assert ledger.locus_tag_synonyms == []
    assert rule.n_records == 1
    assert rule.items == [
        "twinG (b0098): the annotation carries it as a gene synonym of current gene "
        "b0006, so the record would store b0006"
    ]


def test_resolve_genes_refuses_a_gene_with_no_released_sgrna(
    raw_dir: Path, mg1655: EcoliK12MG1655Genome, small_counts: None
) -> None:
    guides = metabolome.read_guides(raw_dir / metabolome.TABLE_S1)
    with pytest.raises(ValueError, match="1 genes have no Table S1 sgRNA"):
        m.resolve_genes(mg1655, ["thrL", "argR"], guides, label="synthetic")


def test_crispri_genotype_carries_the_released_spacer_and_the_b_number(
    raw_dir: Path, mg1655: EcoliK12MG1655Genome, small_counts: None
) -> None:
    guides = metabolome.read_guides(raw_dir / metabolome.TABLE_S1)
    kept, _, _ = m.resolve_genes(mg1655, ["thrA"], guides, label="synthetic")
    (perturbation,) = m.crispri_genotype(kept[0]).perturbations
    assert isinstance(perturbation, CrisprInterferencePerturbation)
    assert perturbation.systematic_gene_name == "b0002"
    assert perturbation.gene_namespace == "ecoli_k12_mg1655_bnumber"
    assert perturbation.crispr is not None
    assert perturbation.crispr.guide_sequence == "TTTTCCCCGGGGAAAATTTT"
    assert perturbation.crispr.effector == "dCas9"
    assert perturbation.crispr.n_guides == 1


def test_drop_log_refuses_rules_that_do_not_account_for_the_drops() -> None:
    rule = DropRule(rule="r", description="d", n_records=0, items=[])
    with pytest.raises(RuntimeError, match="do not account for every dropped record"):
        m._drop_log(
            "x",
            strain_tokens=4,
            reference_tokens=["c"],
            source_records=6,
            kept=4,
            control_records=1,
            rule=rule,
        )


def test_batch_of_gene_refuses_a_strain_sampled_in_two_batches(
    raw_dir: Path, small_counts: None
) -> None:
    samples = dict(metabolome.read_sample_rows(raw_dir / metabolome.TABLE_S3).items())
    key, row = next(iter(samples.items()))
    samples[key] = row.model_copy(
        update={"sample": row.sample.model_copy(update={"batch": row.sample.batch + 5})}
    )
    with pytest.raises(ValueError, match="has no single batch median"):
        m.batch_of_gene(samples)


# --------------------------------------------------------------------------- #
# Hermetic builds
# --------------------------------------------------------------------------- #
def test_growth_build_stores_the_auc_ratio_against_the_control_cultures(
    synthetic: Any, presence_only_pins: list[Mapping[str, str]]
) -> None:
    dataset = synthetic("growth")
    assert len(dataset) == 4
    assert presence_only_pins == [
        {name: metabolome.DATA_SHA256[name] for name in dataset.CONSUMED_FILES}
    ]
    baseline = control_baseline()

    index_of = {
        dataset[index]["experiment"]["genotype"]["perturbations"][0][
            "perturbed_gene_name"
        ]: index
        for index in range(len(dataset))
    }
    record = dataset[index_of["proC"]]["experiment"]
    (perturbation,) = record["genotype"]["perturbations"]
    assert perturbation["systematic_gene_name"] == "b0006"
    assert perturbation["perturbed_gene_name"] == "proC"
    assert record["phenotype"]["fitness"] == pytest.approx(9.0 / baseline)
    assert record["phenotype"]["fitness_uncertainty"] == pytest.approx(0.0)
    assert record["phenotype"]["n_samples"] == 3
    assert record["phenotype"]["sample_unit"] == "biological_replicate"

    reference = dataset[index_of["proC"]]["reference"]["phenotype_reference"]
    assert reference["fitness"] == pytest.approx(1.0)
    assert reference["n_samples"] == 6
    assert reference["fitness_uncertainty"] == pytest.approx(
        float(np.std([30.0, 30.0, 30.0, 24.0, 24.0, 24.0], ddof=1) / baseline)
    )
    assert {
        dataset[index]["experiment"]["genotype"]["perturbations"][0][
            "perturbed_gene_name"
        ]
        for index in range(len(dataset))
    } == {"proB", "proC", "thrA", "thrL"}
    mapped = {
        dataset[index]["experiment"]["genotype"]["perturbations"][0][
            "perturbed_gene_name"
        ]: dataset[index]["experiment"]["genotype"]["perturbations"][0][
            "identifier_mapping"
        ]
        for index in range(len(dataset))
    }
    assert mapped["proB"] == {
        "source_identifier": "b0099",
        "route": "locus_tag_synonym",
    }
    assert [mapped[gene] for gene in ("proC", "thrA", "thrL")] == [None, None, None]


def test_growth_build_writes_its_ledgers(synthetic: Any) -> None:
    dataset = synthetic("growth")
    out = Path(dataset.preprocess_dir)
    drops = json.loads((out / "dropped_records.json").read_text())
    assert (drops["strain_tokens"], drops["source_records"]) == (6, 4)
    assert (drops["kept_records"], drops["dropped_records"]) == (4, 0)
    assert drops["control_records"] == 0
    assert drops["reference_tokens"] == ["ctrl1", "ctrl2"]
    baseline = json.loads((out / "control_baseline.json").read_text())
    assert baseline["n_control_cultures"] == 6
    assert baseline["mean_auc"] == pytest.approx(control_baseline())
    assert baseline["statistic"] == m.GROWTH_STATISTIC
    strains = (out / "strains.csv").read_text().splitlines()
    assert strains[0].startswith(
        "record,gene,b_number,locus_tag,symbol,identifier_route"
    )
    assert len(strains) == 5
    assert [line.split(",")[5] for line in strains[1:]].count("locus_tag_synonym") == 1
    assert "True" in [line.split(",")[-1] for line in strains[1:]]
    defects = json.loads((out / "growth_defect_check.json").read_text())
    assert defects["cutoff"] == m.AUC_DEFECT_CUTOFF


def _by_replicate(dataset: Any) -> dict[str, dict[str, Any]]:
    """``{replicate_id: record}`` over a built accumulation dataset."""
    return {
        record["experiment"]["phenotype"]["replicate_id"]: record
        for record in (dataset[index] for index in range(len(dataset)))
    }


def test_targeted_build_stores_one_record_per_measured_extract(synthetic: Any) -> None:
    dataset = synthetic("targeted")
    assert len(dataset) == 5
    records = _by_replicate(dataset)
    assert sorted(records) == sorted(
        [
            sample_id("ctrl1", 1),
            sample_id("ghostG", 1),
            sample_id("thrA", 1),
            sample_id("thrL", 1),
            sample_id("thrL", 2),
        ]
    )
    control = records[sample_id("ctrl1", 1)]["experiment"]
    assert control["genotype"]["perturbations"] == []
    assert control["phenotype"]["metabolite_level"] == {"didp[M-H]-": 6.0}
    first = records[sample_id("thrL", 1)]["experiment"]
    assert first["genotype"]["perturbations"][0]["perturbed_gene_name"] == "thrL"
    assert first["phenotype"]["metabolite_level"] == {"ppal[M+H]+": 3.0}
    assert records[sample_id("thrL", 2)]["experiment"]["phenotype"][
        "metabolite_level"
    ] == {"didp[M-H]-": 15.0}
    assert first["phenotype"]["metabolite_level_se"] is None
    assert first["phenotype"]["provenance_gaps"][-1]["reason"] == (
        "not_applicable_single_measurement"
    )
    assert (
        first["phenotype"]["measurement_type"] == m.TARGETED_MEASUREMENT_TYPE
    ) and m.TARGETED_MEASUREMENT_TYPE != m.INTENSITY_MEASUREMENT_TYPE
    reference = records[sample_id("thrL", 1)]["reference"]["phenotype_reference"]
    assert reference["metabolite_level"] == {"ppal[M+H]+": 1.0}
    assert reference["replicate_id"] is None

    out = Path(dataset.preprocess_dir)
    ledger = json.loads((out / "accumulation_ledger.json").read_text())
    assert (
        ledger["n_rows"],
        ledger["n_records"],
        ledger["n_control_records"],
        ledger["n_values"],
    ) == (6, 5, 1, 6)
    assert ledger["control_tokens"] == ["ctrl1"]
    assert ledger["measurement_type"] == m.TARGETED_MEASUREMENT_TYPE
    drops = json.loads((out / "dropped_records.json").read_text())
    assert (
        drops["source_records"],
        drops["kept_records"],
        drops["control_records"],
        drops["dropped_records"],
    ) == (5, 5, 1, 0)
    rows = (out / "records.csv").read_text().splitlines()
    assert rows[0] == (
        "record,replicate_id,gene,control,b_number,locus_tag,identifier_route,"
        "n_values,keys"
    )
    assert rows[1] == f"0,{sample_id('ctrl1', 1)},ctrl1,True,,,,1,didp[M-H]-"
    agreement = json.loads((out / "platform_agreement.json").read_text())
    assert agreement["n_pairs"] == len(PAIRS)


def test_intensity_build_stores_each_plate_s_absolute_intensity(synthetic: Any) -> None:
    dataset = synthetic("intensity")
    assert len(dataset) == 8
    records = _by_replicate(dataset)
    assert sorted(records) == sorted(
        sample_id(gene, replicate)
        for gene in ("ctrl1", "ghostG", "thrA", "thrL")
        for replicate in (1, 2)
    )
    for replicate, level in ((1, 40_000.0), (2, 60_000.0)):
        record = records[sample_id("thrL", replicate)]
        phenotype = record["experiment"]["phenotype"]
        assert phenotype["metabolite_level"]["ppal[M+H]+"] == pytest.approx(level)
        assert phenotype["metabolite_level_se"] is None
        assert phenotype["n_replicates"] == {"ppal[M+H]+": 1, "didp[M-H]-": 1}
        assert phenotype["measurement_type"] == m.INTENSITY_MEASUREMENT_TYPE
        reference = record["reference"]["phenotype_reference"]
        assert reference["metabolite_level"]["ppal[M+H]+"] == pytest.approx(10_000.0)
        assert reference["n_replicates"]["ppal[M+H]+"] == 6
    # The mean of a strain's two plate records is the released Mean_Int.
    assert float(
        np.mean(
            [
                records[sample_id("thrL", replicate)]["experiment"]["phenotype"][
                    "metabolite_level"
                ]["ppal[M+H]+"]
                for replicate in (1, 2)
            ]
        )
    ) == pytest.approx(50_000.0)
    controls = [
        records[sample_id("ctrl1", replicate)]["experiment"] for replicate in (1, 2)
    ]
    assert [c["genotype"]["perturbations"] for c in controls] == [[], []]
    assert [c["phenotype"]["metabolite_level"] for c in controls] == [
        {"didp[M-H]-": pytest.approx(5_000.0 * 4.5)},
        {"didp[M-H]-": pytest.approx(5_000.0 * 5.5)},
    ]
    medians = json.loads(
        (Path(dataset.preprocess_dir) / "batch_median_check.json").read_text()
    )
    assert medians["batch_sizes"] == {"1": 6, "2": 4}


def test_accumulation_build_refuses_a_token_count_the_release_does_not_have(
    synthetic: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "ACCUMULATION_TOKENS", 9)
    with pytest.raises(ValueError, match="holds 4 strain tokens, expected 9"):
        synthetic("intensity")


def test_accumulation_build_refuses_a_control_count_the_release_does_not_have(
    synthetic: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "ACCUMULATION_CONTROL_TOKENS", 2)
    with pytest.raises(ValueError, match="1 control tokens, expected 2"):
        synthetic("intensity")


def test_growth_build_refuses_a_control_well_count_the_release_does_not_have(
    synthetic: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "GROWTH_CONTROL_TOKENS", 5)
    with pytest.raises(ValueError, match="holds 2 control wells, expected 5"):
        synthetic("growth")


def test_the_three_families_declare_their_own_files_classes_and_statistics() -> None:
    growth = object.__new__(m.GrowthAucRapp2026Dataset)
    assert growth.experiment_class.__name__ == "FitnessExperiment"
    assert growth.reference_class.__name__ == "FitnessExperimentReference"
    assert growth.raw_file_names == [metabolome.TABLE_S1, metabolome.TABLE_S2]
    targeted = object.__new__(m.TargetedMetabolomeRapp2026Dataset)
    assert targeted.experiment_class.__name__ == "MetaboliteExperiment"
    assert targeted.reference_class.__name__ == "MetaboliteExperimentReference"
    assert metabolome.TABLE_S6 in targeted.raw_file_names
    intensity = object.__new__(m.MetaboliteIntensityRapp2026Dataset)
    assert metabolome.TABLE_S6 not in intensity.raw_file_names
    assert metabolome.TABLE_S5 in intensity.raw_file_names
    assert len({cls.MEASUREMENT_TYPE for cls in (targeted, intensity)}) == 2
    frame = __import__("pandas").DataFrame({"a": [1]})
    assert growth.preprocess_raw(frame) is frame
    with pytest.raises(NotImplementedError):
        growth.create_experiment()


def test_a_genome_of_another_assembly_set_is_refused(
    synthetic: Any, monkeypatch: pytest.MonkeyPatch, mg1655: EcoliK12MG1655Genome
) -> None:
    monkeypatch.setattr(type(mg1655), "ASSEMBLY_SET", "ecoli_K12_BW25113_ASM75055v1")
    with pytest.raises(ValueError, match="needs the ecoli_K12_MG1655_ASM584v2 genome"):
        synthetic("growth")


def test_main_refuses_an_unknown_family() -> None:
    with pytest.raises(SystemExit):
        m.main(["build", "nope"])
    assert sorted(m.FAMILIES) == ["growth", "intensity", "targeted"]


# --------------------------------------------------------------------------- #
# Real data (``--data``): the mirror and the three built dev-tree LMDBs
# --------------------------------------------------------------------------- #
def _real_data_root() -> str:
    return os.environ["DATA_ROOT"]


@pytest.mark.data
def test_real_mirror_pins_the_two_new_workbooks() -> None:
    from torchcell.data.experiment_dataset import file_sha256

    manifest = metabolome.load_manifest(_real_data_root())
    for name in (metabolome.TABLE_S2, metabolome.TABLE_S6):
        raw = metabolome.RAW_FILES_BY_NAME[name]
        assert metabolome.manifest_sha256(manifest, raw.mirror_relpath) == raw.sha256
        path = metabolome.raw_mirror_dir(_real_data_root()) / raw.mirror_relpath
        assert path.stat().st_size == raw.bytes
        assert file_sha256(path) == raw.sha256


@pytest.mark.data
@pytest.mark.parametrize("key", sorted(m.SOURCED_VALUES))
def test_real_sourced_values_are_verbatim(key: str) -> None:
    result = audit_sourced_value(
        m.SOURCED_VALUES[key], Path(_real_data_root()) / "torchcell-library"
    )
    assert result.passed, result.message


@pytest.mark.data
def test_real_column_legends_are_verbatim_in_the_pinned_workbooks() -> None:
    mirror = metabolome.raw_mirror_dir(_real_data_root())
    for columns in (m.INTENSITY_COLUMNS, m.TARGETED_COLUMNS):
        path = mirror / "data" / columns[0].file
        assert m.check_column_legends(path, columns) == list(columns)


@pytest.mark.data
@pytest.mark.parametrize(
    ("family", "expected", "controls"),
    [("growth", 1515, 0), ("targeted", 482, 5), ("intensity", 822, 8)],
)
def test_real_builds_hold_the_measured_record_counts(
    family: str, expected: int, controls: int
) -> None:
    """``phnE`` is kept in every family, on a ``locus_tag_synonym`` mapping; the two
    accumulation families store one record per measured extract, control wells
    included (#1178).
    """
    _, root_rel, _ = m.FAMILIES[family]
    preprocess = Path(_real_data_root(), root_rel, "preprocess")
    drops = json.loads((preprocess / "dropped_records.json").read_text())
    assert (drops["kept_records"], drops["control_records"]) == (expected, controls)
    assert drops["dropped_records"] == 0
    assert drops["rules"][0]["rule"] == "b_number_remapped_by_the_annotation"
    assert drops["rules"][0]["n_records"] == 0
    identifiers = json.loads(
        (preprocess / "identifier_reconciliation.json").read_text()
    )
    assert identifiers["locus_tag_synonyms"] == [
        "phnE (b4104): the annotation carries it as a gene synonym of pseudogene b4583 "
        "(not a gene feature), so the record stores b4583 with route locus_tag_synonym"
    ]
    table = "strains.csv" if family == "growth" else "records.csv"
    rows = (preprocess / table).read_text().splitlines()
    route = rows[0].split(",").index("identifier_route")
    synonyms = [line.split(",")[route] for line in rows[1:]].count("locus_tag_synonym")
    assert synonyms == {"growth": 1, "targeted": 1, "intensity": 2}[family]
    assert len(rows) - 1 == expected


@pytest.mark.data
def test_real_accumulation_tables_name_the_extract_of_every_value() -> None:
    """#1178, measured on the pinned bytes: Table S5's R1_FC / R2_FC are Table S4's
    cells of the strain's ``_R1_`` / ``_R2_`` samples, and every Table S6 DataFile
    carries the run code of exactly one of the strain's two samples (1,036 plate 1,
    220 plate 2), so 71 strains were measured from both extracts.
    """
    mirror = metabolome.raw_mirror_dir(_real_data_root()) / "data"
    metabolites = metabolome.read_metabolites(mirror / metabolome.TABLE_S9)
    samples = metabolome.read_sample_rows(mirror / metabolome.TABLE_S3)
    targeted = m.read_targeted(mirror / metabolome.TABLE_S6, metabolites, samples)
    replicate = [samples[item.sample_ids[0]].sample.replicate for item in targeted]
    assert (replicate.count(1), replicate.count(2)) == (1036, 220)
    groups = m.extract_values(targeted)
    assert len(groups) == 482
    per_token: dict[str, int] = {}
    for group in groups:
        per_token[group.token] = per_token.get(group.token, 0) + 1
    assert sorted(per_token.values()).count(2) == 71
    assert sum(1 for group in groups if group.is_control) == 5

    intensities = m.read_intensities(
        mirror / metabolome.TABLE_S5,
        metabolites,
        dict.fromkeys({row.sample.gene for row in samples.values()}, 1),
        samples,
    )
    table = metabolome.read_feature_table(mirror / metabolome.TABLE_S4)
    column_of = {sample.column: index for index, sample in enumerate(table.samples)}
    row_of = {feature.key: index for index, feature in enumerate(table.features)}
    worst = 0.0
    for item in intensities:
        for intensity, sample in zip(item.replicates, item.sample_ids, strict=True):
            cell = float(table.matrix[row_of[item.key], column_of[sample]])
            worst = max(worst, abs(intensity / item.reference_level - cell) / cell)
    assert worst < 1e-12
    plates = m.extract_values(intensities)
    assert (len(plates), sum(1 for group in plates if group.is_control)) == (822, 8)
    assert sum(len(group.items) for group in plates) == 2770


@pytest.mark.data
def test_real_growth_release_holds_three_cultures_per_strain() -> None:
    """#1178 evidence: Table S2 releases 4,593 culture curves, 1,515 genes x 3 plus 16
    control wells x 3, which a per-culture store would hold as 4,545 + 48 records.
    The store still holds one AUC ratio per gene (owner decision pending, see the
    2026.11.24 note section), so this pins the release, not the store.
    """
    table = m.read_growth_table(
        metabolome.raw_mirror_dir(_real_data_root()) / "data" / metabolome.TABLE_S2
    )
    assert len(table.rows) == 4593
    controls = [row for row in table.rows if row.is_control]
    assert len(controls) == 48
    assert len({row.token for row in controls}) == 16
    by_replicate = [row.replicate for row in table.rows]
    assert [by_replicate.count(r) for r in (1, 2, 3)] == [1531, 1531, 1531]


@pytest.mark.data
def test_real_platform_agreement_is_the_measured_one() -> None:
    agreement = json.loads(
        Path(
            _real_data_root(),
            m.TARGETED_ROOT_REL,
            "preprocess",
            "platform_agreement.json",
        ).read_text()
    )
    assert agreement["n_pairs"] == 1256
    assert agreement["pearson_r_linear"] == pytest.approx(0.6722, abs=5e-5)
    assert agreement["pearson_r_log2"] == pytest.approx(0.6701, abs=5e-5)
    assert agreement["median_abs_log2_difference"] == pytest.approx(1.1649, abs=5e-5)


@pytest.mark.data
def test_real_batch_medians_are_one_number_per_feature_and_batch() -> None:
    check = json.loads(
        Path(
            _real_data_root(),
            m.INTENSITY_ROOT_REL,
            "preprocess",
            "batch_median_check.json",
        ).read_text()
    )
    assert check["n_groups"] == 717
    assert check["n_multi_strain_groups"] == 246
    assert check["max_relative_spread"] < 1e-12
    assert sum(check["batch_sizes"].values()) == 3026


@pytest.mark.data
def test_real_growth_split_reproduces_the_paper_to_one_strain() -> None:
    check = json.loads(
        Path(
            _real_data_root(),
            m.GROWTH_ROOT_REL,
            "preprocess",
            "growth_defect_check.json",
        ).read_text()
    )
    assert (check["measured_growth_defect"], check["paper_growth_defect"]) == (490, 489)
    assert check["measured_from_mean_curve"] == 490
    assert check["n_strains"] == 1515
    assert check["auc_maximum"] == pytest.approx(26.2601, abs=1e-4)


@pytest.mark.data
@pytest.mark.parametrize("family", ["growth", "targeted", "intensity"])
def test_real_verifiers_pass(family: str) -> None:
    _, _, verifier = m.FAMILIES[family]
    report = verifier(_real_data_root())
    assert report.passed, report.summary()
