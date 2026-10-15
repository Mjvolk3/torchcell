# tests/torchcell/datasets/scerevisiae/test_muenzner2024.py
# [[tests.torchcell.datasets.scerevisiae.test_muenzner2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_muenzner2024.py
"""Muenzner 2024 proteome and turnover loaders, built end to end on synthetic raw files.

The genotype half reuses the Caudal 2024 synthetic Peter tier
(``test_caudal2024_synthetic``): Peter isolates AAA, SACE_YAU, ZZZ, XTRA_ABC and FY4-6,
the four-member reference-gene tarball and the two matrices. The Muenzner workbook is
written here with the release's own captions and header rows. Release isolates: AAA,
SACE.YAU (the release's dotted spelling of SACE_YAU), ZZZ, QQQ (no Peter genome) and
Reference (FY4, the S288C-derived column), so four build (FY4 last, on its own
reference genome with no perturbation, #1045) and QQQ is refused.

SI Table 3 (linear MaxLFQ, ``NA`` = missing)::

    protein  AAA   SACE.YAU  ZZZ   QQQ  Reference
    YAL001C  10.0  30.0      20.0  5.0  7.0
    YBR001W  NA    4.0       6.0   6.0  8.0
    Q0010    2.0   NA        NA    3.0  1.0

SI Table 7 (the integrated set) is AAA and ZZZ, both euploid, so the reference is the
median over those two: YAL001C 15.0 (n=2), YBR001W 6.0 (n=1), Q0010 2.0 (n=1).
SACE.YAU is in SI Table 5 (a karyotype mismatch) and QQQ in SI Table 6.

SI Table 17 (turnover): AAA YAL001C 0.5 / YBR001W 0.25; ZZZ YAL001C 0.3 / Q0010 0.2;
SACE.YAU YAL001C 0.4 / R0040C 0.1, each HL = ln 2 / kdp. AAA and ZZZ are euploid, so the
turnover reference is YAL001C 0.4 (n=2), YBR001W 0.25, Q0010 0.2; R0040C (only in the
aneuploid SACE_YAU) has no reference key.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import os.path as osp
import pickle
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import openpyxl
import pandas as pd
import pytest

from tests.torchcell.datasets.scerevisiae.test_caudal2024_synthetic import (
    _WORKBOOK,
    _absent,
    _descriptor,
    _present,
    _raw_bytes,
    _variant,
    _write_peter_tier,
    _write_sgd_tier,
)
from torchcell.data import RawSha256MismatchError
from torchcell.datamodels.schema import (
    ChromosomeCopyChange,
    ConcentrationUnit,
    Environment,
    GenePerturbationType,
    Genotype,
    ProteinAbundanceExperiment,
    ProteinAbundanceExperimentReference,
    ProteinAbundancePhenotype,
    ProteinTurnoverExperiment,
    ProteinTurnoverExperimentReference,
    ProteinTurnoverPhenotype,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.scerevisiae import muenzner2024 as m
from torchcell.sequence import natural_isolate as ni
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.natural_isolate_genotype import IsolateGenotypeAudit
from torchcell.verification.protein import (
    verify_natural_isolate_proteome,
    verify_natural_isolate_turnover,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import audit_sourced_value

LN2 = math.log(2)


def _hl(rate: float) -> float:
    """The release's half-life for ``rate``, at the 12 decimals a workbook keeps."""
    return round(LN2 / rate, 12)


PROV = Provenance(source_uri="test://muenzner", citation_key="muenznerTest2026")

# --------------------------------------------------------------------------- #
# synthetic workbook
# --------------------------------------------------------------------------- #
_T1_ROWS = [
    [
        1,
        "AAA",
        "RM11",
        1,
        "euploid",
        "1. Wine/European",
        "Dom",
        "Dom",
        "Wine",
        "x",
        "y",
    ],
    [2, "SACE.YAU", "YJM1", 2, "aneu;+1*1;", "M1", "Wild", "Wild", "Nature", "x", "y"],
    [3, "ZZZ", "Z1", 2, "euploid", "M2", "Wild", "Wild", "Nature", "x", "y"],
    [4, "QQQ", "Q1", 2, "euploid", "M2", "Wild", "Wild", "Nature", "x", "y"],
    [5, "Reference", "FY4", 1, "NA", "M3", None, None, None, None, None],
]
_T3_ROWS = [
    ["YAL001C", 10.0, 30.0, 20.0, 5.0, 7.0],
    ["YBR001W", "NA", 4.0, 6.0, 6.0, 8.0],
    ["Q0010", 2.0, "NA", "NA", 3.0, 1.0],
]
_T3_ISOLATES = ["AAA", "SACE.YAU", "ZZZ", "QQQ", "Reference"]
_T17_ROWS = [
    ["AAA", 1, "euploid", "P1", "YAL001C", "TFC3", 0.5, 1e9, _hl(0.5)],
    ["AAA", 1, "euploid", "P2", "YBR001W", "X", 0.25, 1e9, _hl(0.25)],
    ["ZZZ", 2, "euploid", "P1", "YAL001C", "TFC3", 0.3, 1e9, _hl(0.3)],
    ["ZZZ", 2, "euploid", "P3", "Q0010", "COX1", 0.2, 1e9, _hl(0.2)],
    ["SACE.YAU", 2, "aneu;+1*1;", "P1", "YAL001C", "TFC3", 0.4, 1e9, _hl(0.4)],
    ["SACE.YAU", 2, "aneu;+1*1;", "P4", "R0040C", "R0040C", 0.1, 1e9, _hl(0.1)],
]


def _sheet_rows(spec: m.SheetSpec, data: list[list[Any]]) -> list[list[Any]]:
    """Caption rows, blank padding, the header, then ``data`` (the release layout)."""
    rows: list[list[Any]] = [[text] for text in spec.caption]
    while len(rows) < spec.header_row:
        rows.append([None])
    header = list(spec.header) if spec.header is not None else []
    if spec is m.T3:
        header = ["Protein", *_T3_ISOLATES]
    rows.append(header)
    rows.extend(data)
    return rows


def _workbook(sheets: dict[str, list[list[Any]]]) -> bytes:
    wb = openpyxl.Workbook()
    assert wb.active is not None
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for row in rows:
            ws.append(row)
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def _release_sheets() -> dict[str, list[list[Any]]]:
    return {
        m.T1.name: _sheet_rows(m.T1, _T1_ROWS),
        m.T3.name: _sheet_rows(m.T3, _T3_ROWS),
        m.T5.name: _sheet_rows(m.T5, [["SACE.YAU"]]),
        m.T6.name: _sheet_rows(m.T6, [["QQQ"]]),
        m.T7.name: _sheet_rows(
            m.T7, [["AAA", 1, "euploid", "FALSE"], ["ZZZ", 2, "euploid", "FALSE"]]
        ),
        m.T16.name: _sheet_rows(
            m.T16,
            [["AAA", 120.0, 0.01, 0.9], ["SACE.YAU", 130.0, 0.01, 0.9]]
            + [["ZZZ", 150.0, 0.01, 0.9]],
        ),
        m.T17.name: _sheet_rows(m.T17, _T17_ROWS),
    }


def _write_raw(raw: Path, sheets: dict[str, list[list[Any]]] | None = None) -> None:
    raw.mkdir(parents=True)
    (raw / m.SI_TABLES_FILENAME).write_bytes(_workbook(sheets or _release_sheets()))
    peter = _raw_bytes()
    for name in ni.GENOTYPE_FILES:
        (raw / name).write_bytes(peter[name])


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(root))
    _write_sgd_tier(root)
    _write_peter_tier(root)
    return root


@pytest.fixture
def proteome(data_root: Path, tmp_path: Path) -> m.ProteomeMuenzner2024Dataset:
    root = tmp_path / "proteome"
    _write_raw(root / "raw")
    return m.ProteomeMuenzner2024Dataset(root=str(root))


@pytest.fixture
def turnover(data_root: Path, tmp_path: Path) -> m.ProteinTurnoverMuenzner2024Dataset:
    root = tmp_path / "turnover"
    _write_raw(root / "raw")
    return m.ProteinTurnoverMuenzner2024Dataset(root=str(root))


# Genotypes: the Caudal construction on the shared synthetic Peter tier.
_GENOTYPES: dict[str, list[GenePerturbationType]] = {
    "AAA": [
        _variant("AAA", "Q0010", "COX1", "AAA_Q0010_COX1"),
        _present("AAA", "2-EC1118_1F14_0012g", 2.5),
        _absent("AAA", "YAL005C", "3-YAL005C_NumOfGenes_3"),
    ],
    "SACE_YAU": [
        _variant("SACE_YAU", "YAL001C", "TFC3", "SACE_YAU_YAL001C_TFC3"),
        _variant("SACE_YAU", "YBR001W", "YBR001W", "SACE_YAU_YBR001W_"),
        _present("SACE_YAU", "5-contig_7", 1.0),
        _absent("SACE_YAU", "YCR001W", "6-YCR001W"),
    ],
    "ZZZ": [
        _variant("ZZZ", "YAL001C", "TFC3", "ZZZ_YAL001C_TFC3"),
        _absent("ZZZ", "YBR001W", "4-YBR001W"),
    ],
}


def _genotype(strain: str) -> Genotype:
    """The isolate's genotype as ``natural_isolate`` builds it on the Caudal fixture."""
    return Genotype(
        perturbations=_GENOTYPES[strain], natural_isolate=_descriptor(strain)
    )


_PROTEOME_ENV = Environment(media=m.MUENZNER_SM, temperature=Temperature(value=30))
_TURNOVER_ENV = Environment(media=m.MUENZNER_SM_LYS8, temperature=Temperature(value=30))
_S288C = ReferenceGenome(
    species="Saccharomyces cerevisiae",
    strain="S288C",
    assembly_set=SGD_S288C_R64,
    assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
)


def _abundance(strain: str, values: dict[str, float]) -> ProteinAbundanceExperiment:
    return ProteinAbundanceExperiment(
        dataset_name="ProteomeMuenzner2024Dataset",
        genotype=_genotype(strain),
        environment=_PROTEOME_ENV,
        phenotype=ProteinAbundancePhenotype(
            protein_abundance=values,
            protein_abundance_se=None,
            n_replicates={k: 1 for k in values},
            measurement_type=m.PROTEOME_MEASUREMENT_TYPE,
        ),
    )


def _turnover(strain: str, rates: dict[str, float]) -> ProteinTurnoverExperiment:
    return ProteinTurnoverExperiment(
        dataset_name="ProteinTurnoverMuenzner2024Dataset",
        genotype=_genotype(strain),
        environment=_TURNOVER_ENV,
        phenotype=ProteinTurnoverPhenotype(
            degradation_rate=rates,
            half_life={k: _hl(v) for k, v in rates.items()},
            n_replicates={k: 1 for k in rates},
            measurement_type=m.TURNOVER_MEASUREMENT_TYPE,
        ),
    )


# --------------------------------------------------------------------------- #
# end-to-end builds
# --------------------------------------------------------------------------- #
def test_proteome_builds_three_isolates_with_caudal_genotypes(
    proteome: m.ProteomeMuenzner2024Dataset,
) -> None:
    """AAA, SACE_YAU and ZZZ build in Peter-code order; blanks are key-absent."""
    assert len(proteome) == 4
    expected = [
        _abundance("AAA", {"YAL001C": 10.0, "Q0010": 2.0}),
        _abundance("SACE_YAU", {"YAL001C": 30.0, "YBR001W": 4.0}),
        _abundance("ZZZ", {"YAL001C": 20.0, "YBR001W": 6.0}),
    ]
    reference = ProteinAbundanceExperimentReference(
        dataset_name="ProteomeMuenzner2024Dataset",
        genome_reference=_S288C,
        environment_reference=_PROTEOME_ENV,
        phenotype_reference=ProteinAbundancePhenotype(
            protein_abundance={"YAL001C": 15.0, "YBR001W": 6.0, "Q0010": 2.0},
            protein_abundance_se=None,
            n_replicates={"YAL001C": 2, "YBR001W": 1, "Q0010": 1},
            measurement_type=m.PROTEOME_MEASUREMENT_TYPE,
        ),
    )
    for i, experiment in enumerate(expected):
        assert proteome[i]["experiment"] == experiment.model_dump()
        assert proteome[i]["reference"] == reference.model_dump()
    assert proteome[0]["publication"] == m.PUBLICATION.model_dump()
    assert m.PUBLICATION.pubmed_url == "https://pubmed.ncbi.nlm.nih.gov/38778096/"


def test_fy4_is_built_on_s288c_with_no_perturbation_and_its_own_genome(
    proteome: m.ProteomeMuenzner2024Dataset,
) -> None:
    """The release's Reference column (SI Table 1: FY4) is record 3 (#1045 row 18): no
    perturbation off S288C, its SI Table 3 column, and the shared euploid reference
    whose genome names FY4 (haploid) instead of S288C.
    """
    fy4 = ProteinAbundanceExperiment(
        dataset_name="ProteomeMuenzner2024Dataset",
        genotype=Genotype(perturbations=[]),
        environment=_PROTEOME_ENV,
        phenotype=ProteinAbundancePhenotype(
            protein_abundance={"YAL001C": 7.0, "YBR001W": 8.0, "Q0010": 1.0},
            protein_abundance_se=None,
            n_replicates={"YAL001C": 1, "YBR001W": 1, "Q0010": 1},
            measurement_type=m.PROTEOME_MEASUREMENT_TYPE,
        ),
    )
    assert proteome[3]["experiment"] == fy4.model_dump()
    reference = proteome[3]["reference"]
    assert reference["genome_reference"] == {
        "provenance_gaps": [],
        "species": "Saccharomyces cerevisiae",
        "strain": "FY4",
        "ploidy": "haploid",
        "assembly_set": "sgd_S288C_R64-4-1_20230830",
        "assembly_accession": "GCA_000146045.2",
    }
    assert (
        reference["phenotype_reference"]
        == proteome[0]["reference"]["phenotype_reference"]
    )
    assert (
        reference["environment_reference"]
        == proteome[0]["reference"]["environment_reference"]
    )


def test_an_s288c_derived_column_must_be_named_fy4_and_haploid() -> None:
    """SI Table 1 must name the Reference column FY4 at ploidy 1, else the build stops."""
    rows = [list(row) for row in _T1_ROWS]
    frame = pd.DataFrame([row[1:] for row in rows], columns=list(m.T1.header or ())[1:])
    assert m.s288c_derived_isolate("Reference", frame).strain == "FY4"
    renamed = frame.copy()
    renamed.loc[
        renamed["Standardized_name"] == "Reference", "alternative_strain_name"
    ] = "BY4741"
    with pytest.raises(ValueError, match="names 'Reference' 'BY4741', not 'FY4'"):
        m.s288c_derived_isolate("Reference", renamed)
    diploid = frame.copy()
    diploid.loc[diploid["Standardized_name"] == "Reference", "Ploidy"] = 2
    with pytest.raises(ValueError, match="gives FY4 ploidy 2, not 1"):
        m.s288c_derived_isolate("Reference", diploid)


def test_proteome_ledger_accounts_for_every_column_and_cell(
    proteome: m.ProteomeMuenzner2024Dataset,
) -> None:
    """One refusal with its reason, FY4 built on S288C, 3 of 12 built cells missing,
    karyotype flags.
    """
    preprocess = Path(proteome.root) / "preprocess"
    ledger = m.ProteomeLedger.model_validate_json(
        (preprocess / "muenzner2024_ledger.json").read_text()
    )
    assert [(r.release_code, r.alternative_strain_name) for r in ledger.refused] == [
        ("QQQ", "Q1")
    ]
    assert ledger.refused[0].reason == (
        "no Peter 2018 genome: 'QQQ' is not an isolate of "
        "genesMatrix_PresenceAbsence.tab.gz, so no genotype can be written"
    )
    assert [
        (i.release_code, i.strain, i.ploidy, i.aneuploidies)
        for i in ledger.s288c_derived
    ] == [("Reference", "FY4", "1", "NA")]
    assert (ledger.n_release_isolates, ledger.n_release_proteins) == (5, 3)
    assert (ledger.n_built, ledger.n_cells_built) == (4, 12)
    assert (ledger.n_measured_built, ledger.n_missing_built) == (9, 3)
    assert ledger.missing_by_isolate == {"AAA": 1, "SACE_YAU": 1, "ZZZ": 1}
    assert ledger.reference_isolates == ["AAA", "ZZZ"]
    assert [
        (
            k.isolate,
            k.release_code,
            k.ploidy,
            k.aneuploidies,
            k.integrated,
            k.excluded_by,
        )
        for k in ledger.karyotypes
    ] == [
        ("AAA", "AAA", "1", "euploid", True, None),
        ("SACE_YAU", "SACE.YAU", "2", "aneu;+1*1;", False, "SI Table 5"),
        ("ZZZ", "ZZZ", "2", "euploid", True, None),
    ]
    assert ledger.karyotypes[1].copy_changes == [
        ChromosomeCopyChange(chromosome=1, change=1)
    ]
    assert (preprocess / "data.csv").read_text() == (
        "strain_id\nAAA\nSACE_YAU\nZZZ\nFY4\n"
    )


def test_turnover_builds_three_isolates_and_a_partial_euploid_reference(
    turnover: m.ProteinTurnoverMuenzner2024Dataset,
) -> None:
    """R0040C, measured only in the aneuploid SACE_YAU, has no reference key."""
    assert len(turnover) == 3
    expected = [
        _turnover("AAA", {"YAL001C": 0.5, "YBR001W": 0.25}),
        _turnover("SACE_YAU", {"YAL001C": 0.4, "R0040C": 0.1}),
        _turnover("ZZZ", {"YAL001C": 0.3, "Q0010": 0.2}),
    ]
    reference = ProteinTurnoverExperimentReference(
        dataset_name="ProteinTurnoverMuenzner2024Dataset",
        genome_reference=_S288C,
        environment_reference=_TURNOVER_ENV,
        phenotype_reference=ProteinTurnoverPhenotype(
            degradation_rate={"YAL001C": 0.4, "YBR001W": 0.25, "Q0010": 0.2},
            n_replicates={"YAL001C": 2, "YBR001W": 1, "Q0010": 1},
            measurement_type=m.TURNOVER_MEASUREMENT_TYPE,
        ),
    )
    for i, experiment in enumerate(expected):
        assert turnover[i]["experiment"] == experiment.model_dump()
        assert turnover[i]["reference"] == reference.model_dump()

    ledger = m.TurnoverLedger.model_validate_json(
        (Path(turnover.root) / "preprocess" / "muenzner2024_ledger.json").read_text()
    )
    assert (ledger.n_release_rows, ledger.n_release_isolates) == (6, 3)
    assert (ledger.n_built_isolates, ledger.n_built_rows, ledger.refused) == (3, 6, [])
    assert ledger.reference_isolates == ["AAA", "ZZZ"]
    assert (ledger.n_proteins, ledger.n_proteins_with_reference) == (4, 3)
    assert ledger.proteins_without_reference == ["R0040C"]
    assert ledger.n_rows_without_reference == 1
    assert 0 < ledger.half_life_identity_max_relative_error < 1e-11
    assert ledger.doubling_time_minutes == {
        "AAA": 120.0,
        "SACE_YAU": 130.0,
        "ZZZ": 150.0,
    }


def test_turnover_stops_on_an_s288c_derived_column_it_has_no_path_for(
    data_root: Path, tmp_path: Path
) -> None:
    """The released Table 17 has no FY4 rows; one appearing stops the build by name."""
    sheets = _release_sheets()
    sheets[m.T17.name] = _sheet_rows(
        m.T17,
        [
            *_T17_ROWS,
            ["Reference", 1, "NA", "P1", "YAL001C", "TFC3", 0.5, 1e9, _hl(0.5)],
        ],
    )
    root = tmp_path / "turnover_fy4"
    _write_raw(root / "raw", sheets)
    with pytest.raises(
        RuntimeError, match=r"S288C-derived column\(s\) \['Reference'\]"
    ):
        m.ProteinTurnoverMuenzner2024Dataset(root=str(root))


def test_both_builds_verify_their_four_raw_files_before_reading(
    proteome: m.ProteomeMuenzner2024Dataset, data_root: Path, raw_pin_calls: list[Any]
) -> None:
    """The recorder saw the workbook pin and the three Peter-tier manifest pins.

    The Peter pins are read back from the tier manifest the fixture wrote (the synthetic
    matrices are gzip-compressed with a timestamp, so their digest is per write).
    """
    from torchcell.sequence.genome.registry import PETER2018_1011, load_genome_manifest

    calls = [c for c in raw_pin_calls if c[1] == proteome.raw_dir]
    assert len(calls) == 1
    pins = calls[0][2]
    assert sorted(pins) == sorted(m.RAW_FILES)
    assert pins[m.SI_TABLES_FILENAME] == m.SI_TABLES_SHA256
    tier = load_genome_manifest(PETER2018_1011, str(data_root))
    for name in ni.GENOTYPE_FILES:
        assert pins[name] == tier.record(name).sha256


def test_an_off_pin_workbook_is_refused_before_any_record(
    data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from torchcell.data.experiment_dataset import verify_raw_files

    monkeypatch.setattr(m, "verify_raw_files", verify_raw_files)
    root = tmp_path / "offpin"
    _write_raw(root / "raw")
    with pytest.raises(RawSha256MismatchError, match=m.SI_TABLES_FILENAME):
        m.ProteomeMuenzner2024Dataset(root=str(root))
    assert not (root / "processed" / "lmdb").exists()


# --------------------------------------------------------------------------- #
# workbook parsing
# --------------------------------------------------------------------------- #
def test_sheet_frame_reads_the_release_layout_and_names_the_index_column() -> None:
    frame = m.sheet_frame(
        [tuple(r) for r in _sheet_rows(m.T1, _T1_ROWS[:1] + [[None] * 11])], m.T1
    )
    assert list(frame.columns) == ["row_number", *[c for c in m.T1.header or ()][1:]]
    assert frame.to_dict(orient="records") == [
        {
            "row_number": 1,
            "Standardized_name": "AAA",
            "alternative_strain_name": "RM11",
            "Ploidy": 1,
            "Aneuploidies": "euploid",
            "Clades": "1. Wine/European",
            "Clade_class": "Dom",
            "Single_strain_class": "Dom",
            "Simple_Ecology": "Wine",
            "Isolation": "x",
            "Geographical.origins": "y",
        }
    ]


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda r: r.__setitem__(0, ("SI Table 1: renamed",)), "row 1 column A is"),
        (
            lambda r: r.__setitem__(4, (None, "Standardized_name")),
            "SI Table 1 header is",
        ),
        (lambda r: r.insert(5, ()), "empty row 6 inside the data"),
        (lambda r: r.append((1,) * 12), "has cells right of the header"),
    ],
)
def test_sheet_frame_refuses_a_layout_it_was_not_written_for(
    mutate: Any, message: str
) -> None:
    rows = [tuple(r) for r in _sheet_rows(m.T1, _T1_ROWS[:2])]
    mutate(rows)
    with pytest.raises(m.SheetLayoutError, match=message):
        m.sheet_frame(rows, m.T1)


def test_sheet_frame_checks_a_wide_sheet_by_its_first_header() -> None:
    rows = [tuple(r) for r in _sheet_rows(m.T3, _T3_ROWS)]
    assert list(m.sheet_frame(rows, m.T3).columns) == ["Protein", *_T3_ISOLATES]
    rows[3] = ("Gene", *_T3_ISOLATES)
    with pytest.raises(m.SheetLayoutError, match="header starts \\('Gene',\\)"):
        m.sheet_frame(rows, m.T3)


def test_proteome_matrix_reads_na_as_missing_and_refuses_anything_else() -> None:
    frame = m.sheet_frame([tuple(r) for r in _sheet_rows(m.T3, _T3_ROWS)], m.T3)
    values = m.proteome_matrix(frame)
    assert bool(values.isna().to_numpy()[1, 0])  # YBR001W, AAA
    assert int(values.isna().to_numpy().sum()) == 3
    assert list(values.to_numpy()[0]) == [10.0, 30.0, 20.0, 5.0, 7.0]
    for cell, message in [
        (None, "empty cell"),
        ("n.d.", "1 non-numeric cells other than 'NA'"),
        (0.0, "non-positive"),
    ]:
        bad = frame.copy()
        bad.loc[0, "AAA"] = cell
        with pytest.raises(ValueError, match=message):
            m.proteome_matrix(bad)
    doubled = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="repeats a protein"):
        m.proteome_matrix(doubled)


def test_median_reference_skips_unmeasured_cells_and_rows() -> None:
    values = pd.DataFrame(
        {"A": [1.0, np.nan, np.nan], "B": [3.0, 4.0, np.nan], "C": [9.0, 9.0, 9.0]},
        index=["p1", "p2", "p3"],
    )
    assert m.median_reference(values, ["A", "B"]) == (
        {"p1": 2.0, "p2": 4.0},
        {"p1": 2, "p2": 1},
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("euploid", []),
        ("aneu;+1*3;+1*12;", [(3, 1), (12, 1)]),
        ("aneu;-1*1;", [(1, -1)]),
        ("aneu;+2*9;", [(9, 2)]),
    ],
)
def test_parse_aneuploidies_reads_the_legend(
    text: str, expected: list[tuple[int, int]]
) -> None:
    assert [(c.chromosome, c.change) for c in ni.parse_aneuploidy(text)] == expected


@pytest.mark.parametrize("text", ["NA", "aneu;", "aneu;+1*3", "trisomy 3"])
def test_parse_aneuploidies_refuses_other_text(text: str) -> None:
    with pytest.raises(ValueError, match="unparseable aneuploidy annotation"):
        ni.parse_aneuploidy(text)


def test_parse_aneuploidies_refuses_a_seventeenth_chromosome() -> None:
    with pytest.raises(ValueError, match="less than or equal to 16"):
        ni.parse_aneuploidy("aneu;+1*17;")


def test_peter_code_restores_the_underscore_prefixes() -> None:
    assert [m.peter_code(c) for c in ("AAA", "SACE.YAB", "XTRA.DGU")] == [
        "AAA",
        "SACE_YAB",
        "XTRA_DGU",
    ]


def test_match_isolates_refuses_a_code_absent_from_si_table_1() -> None:
    t1 = m.sheet_frame([tuple(r) for r in _sheet_rows(m.T1, _T1_ROWS)], m.T1)
    with pytest.raises(ValueError, match=r"absent from SI Table 1: \['NEW'\]"):
        m.match_isolates(["AAA", "NEW"], {"AAA"}, t1)


def test_turnover_frame_refuses_a_repeated_pair_and_a_blank_rate() -> None:
    frame = m.sheet_frame([tuple(r) for r in _sheet_rows(m.T17, _T17_ROWS)], m.T17)
    assert 0 < m.half_life_identity_error(m.turnover_frame(frame)) < 1e-11
    doubled = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="repeats a \\(strain, systematic_name\\)"):
        m.turnover_frame(doubled)
    blank = frame.copy()
    blank.loc[0, "HL"] = None
    with pytest.raises(ValueError, match="'HL' holds a blank or non-numeric cell"):
        m.turnover_frame(blank)


def test_turnover_annotations_must_match_si_table_1() -> None:
    t1 = m.sheet_frame([tuple(r) for r in _sheet_rows(m.T1, _T1_ROWS)], m.T1)
    rows = [list(r) for r in _T17_ROWS]
    rows[0][2] = "aneu;+1*3;"
    frame = m.turnover_frame(
        m.sheet_frame([tuple(r) for r in _sheet_rows(m.T17, rows)], m.T17)
    )
    with pytest.raises(ValueError, match="Aneuploidies for AAA is"):
        m.check_annotation_agreement(frame, t1)


# --------------------------------------------------------------------------- #
# media
# --------------------------------------------------------------------------- #
def test_proteome_medium_carries_the_printed_6_8_g_per_l() -> None:
    ynb, ammonium, glucose = m.MUENZNER_SM.components
    assert ynb.compound.name == "yeast nitrogen base (w/o amino acids)"
    assert ynb.concentration is not None
    assert (ynb.concentration.value, ynb.concentration.unit) == (
        6.8,
        ConcentrationUnit.g_per_l,
    )
    assert ammonium.compound.name == "ammonium sulfate"
    assert glucose.concentration is not None and glucose.concentration.value == 2.0
    assert m.MUENZNER_SM.base_medium == "SM"


def test_silac_medium_is_library_sm_plus_80_mg_per_l_heavy_lysine() -> None:
    from torchcell.datamodels.media import SM

    *base, lysine = m.MUENZNER_SM_LYS8.components
    assert base == list(SM.components)
    assert lysine.compound.name == m.HEAVY_LYSINE
    assert lysine.compound.inchikey is None
    assert [g.field for g in lysine.compound.provenance_gaps] == ["inchikey"]
    assert lysine.concentration is not None
    assert (lysine.concentration.value, lysine.concentration.unit) == (
        80.0,
        ConcentrationUnit.ug_per_ml,
    )


# --------------------------------------------------------------------------- #
# raw mirror
# --------------------------------------------------------------------------- #
def test_deposit_writes_the_workbook_and_a_pmc_cloud_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "si3.xlsx"
    source.write_bytes(b"workbook bytes")
    digest = hashlib.sha256(b"workbook bytes").hexdigest()
    monkeypatch.setattr(m, "SI_TABLES_SHA256", digest)
    root = m.deposit_raw_mirror(source, str(tmp_path / "dr"))
    assert root == tmp_path / "dr" / "torchcell-raw" / m.CITATION_KEY
    assert (root / m.SI_TABLES_REL).read_bytes() == b"workbook bytes"
    manifest = m.load_manifest(str(tmp_path / "dr"))
    assert (manifest.citation_key, manifest.doi) == (m.CITATION_KEY, m.DOI)
    [record] = manifest.files
    assert (record.path, record.sha256, record.bytes) == (m.SI_TABLES_REL, digest, 14)
    assert record.retrieval is not None
    assert record.retrieval.params == {
        "key": "PMC11153158.1/41586_2024_7442_MOESM3_ESM.xlsx"
    }
    assert record.retrieval.source_url == (
        "https://pmc-oa-opendata.s3.amazonaws.com/PMC11153158.1/"
        "41586_2024_7442_MOESM3_ESM.xlsx"
    )
    assert m.manifest_sha256(manifest, m.SI_TABLES_REL) == digest
    with pytest.raises(KeyError, match="not in the raw-mirror manifest"):
        m.manifest_sha256(manifest, "si/other.xlsx")


def test_deposit_refuses_bytes_that_are_not_the_release(tmp_path: Path) -> None:
    source = tmp_path / "si3.xlsx"
    source.write_bytes(b"not the release")
    with pytest.raises(RawSha256MismatchError):
        m.deposit_raw_mirror(source, str(tmp_path / "dr"))
    assert not (
        tmp_path / "dr" / "torchcell-raw" / m.CITATION_KEY / "manifest.json"
    ).exists()


def test_download_links_the_mirror_workbook_and_the_peter_files(
    data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workbook = _workbook(_release_sheets())
    source = tmp_path / "si3.xlsx"
    source.write_bytes(workbook)
    monkeypatch.setattr(m, "SI_TABLES_SHA256", hashlib.sha256(workbook).hexdigest())
    m.deposit_raw_mirror(source, str(data_root))
    dataset = m.ProteomeMuenzner2024Dataset.__new__(m.ProteomeMuenzner2024Dataset)
    dataset.root = str(tmp_path / "dl")
    dataset.download()
    raw = Path(dataset.raw_dir)
    assert sorted(p.name for p in raw.iterdir()) == sorted(m.RAW_FILES)
    assert (raw / m.SI_TABLES_FILENAME).read_bytes() == workbook
    tier = data_root / "torchcell-genomes" / "peter2018_1011_assemblies"
    assert os.readlink(raw / ni.PRESENCE_NAME) == str(tier / ni.PRESENCE_NAME)


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def _names(report: Any) -> list[str]:
    return [r.name for r in report.results]


def _audit(
    dataset: Any, data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> IsolateGenotypeAudit:
    """The shared genotype audit over the store's isolates, with Table S1 mirrored."""
    library = data_root / "torchcell-library"
    _library_tables(library, monkeypatch)
    return IsolateGenotypeAudit(
        ni.NaturalIsolateGenotypes(
            dataset.raw_dir,
            dataset.preprocess_dir,
            ["AAA", "SACE_YAU", "ZZZ"],
            str(data_root),
        ),
        library,
    )


def _library_tables(library: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The library mirror's copy of the Peter workbook, pinned at its own digest."""
    path = library / ni.CITATION_KEY / ni.LIBRARY_TABLES_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_WORKBOOK)
    monkeypatch.setattr(ni, "TABLES_SHA256", hashlib.sha256(_WORKBOOK).hexdigest())


def test_proteome_gate_passes_the_synthetic_store(
    proteome: m.ProteomeMuenzner2024Dataset,
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [proteome[i] for i in range(len(proteome))]
    report = verify_natural_isolate_proteome(
        records,
        dataset_name="proteome_muenzner2024",
        provenance=PROV,
        expected_count=4,
        sgd_genes={"YAL001C", "YBR001W", "Q0010"},
        caudal_isolates={"AAA", "ZZZ", "OTHER"},
        expected_caudal_overlap=2,
        caudal_genotypes={"AAA": _genotype("AAA").model_dump()},
        genotype_audit=_audit(proteome, data_root, monkeypatch),
    )
    assert _names(report) == [
        "structural",
        "count",
        "strain_uniqueness",
        "genotype_strain_consistency",
        "value_fidelity",
        "single_acquisition",
        "shared_reference",
        "reference_covers_every_key",
        "measurement_type_consistent",
        "gene_containment_sgd",
        "caudal_isolate_overlap",
        "caudal_genotype_agreement",
        "isolate_genotype_identity",
        "isolate_genotype_round_trip",
        "isolate_descriptor_matches_table_s1",
        "loss_of_function_calls_match_release",
    ]
    assert [r.name for r in report.results if not r.passed] == []
    consistency = report.results[3]
    assert consistency.details["n_s288c_derived"] == 1
    assert report.results[2].details["n_strains"] == 4
    agreement = report.results[11]
    assert agreement.message == (
        "1 isolates compared with the Caudal 2024 store: 1 identical genotypes"
    )


def test_proteome_gate_fails_a_changed_genotype_and_a_wrong_overlap(
    proteome: m.ProteomeMuenzner2024Dataset,
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [proteome[i] for i in range(len(proteome))]
    report = verify_natural_isolate_proteome(
        records,
        dataset_name="proteome_muenzner2024",
        provenance=PROV,
        expected_count=4,
        sgd_genes={"YAL001C"},
        caudal_isolates={"AAA"},
        expected_caudal_overlap=2,
        caudal_genotypes={"AAA": {"perturbations": []}, "NOPE": {"perturbations": []}},
        genotype_audit=_audit(proteome, data_root, monkeypatch),
    )
    failed = {r.name: r for r in report.results if not r.passed}
    assert sorted(failed) == [
        "caudal_genotype_agreement",
        "caudal_isolate_overlap",
        "gene_containment_sgd",
    ]
    assert failed["caudal_genotype_agreement"].details["differ"] == ["AAA"]
    assert failed["caudal_genotype_agreement"].details["absent"] == ["NOPE"]
    assert failed["gene_containment_sgd"].details["outside"] == ["Q0010", "YBR001W"]


def test_turnover_gate_counts_uncovered_cells_against_the_ledger(
    turnover: m.ProteinTurnoverMuenzner2024Dataset,
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [turnover[i] for i in range(len(turnover))]
    kwargs: dict[str, Any] = {
        "dataset_name": "protein_turnover_muenzner2024",
        "provenance": PROV,
        "expected_count": 3,
        "sgd_genes": {"YAL001C", "YBR001W", "Q0010", "R0040C"},
    }
    passing = verify_natural_isolate_turnover(
        records,
        expected_rows_without_reference=1,
        genotype_audit=_audit(turnover, data_root, monkeypatch),
        **kwargs,
    )
    assert [r.name for r in passing.results if not r.passed] == []
    assert "half_life_is_ln2_over_rate" in _names(passing)
    failing = verify_natural_isolate_turnover(
        records,
        expected_rows_without_reference=0,
        genotype_audit=_audit(turnover, data_root, monkeypatch),
        **kwargs,
    )
    [bad] = [r for r in failing.results if not r.passed]
    assert bad.name == "uncovered_cells_match_ledger"
    assert (
        bad.message == "1 measured cells without a euploid reference value (ledger: 0)"
    )


def _caudal_store(data_root: Path, records: list[dict[str, Any]]) -> None:
    """A Caudal 2024 store holding ``records`` (sorted strain order) and its data.csv."""
    root = data_root / m.CAUDAL_ROOT
    (root / "preprocess").mkdir(parents=True)
    strains = [ni.genotype_isolate(r["experiment"]["genotype"]) for r in records]
    pd.DataFrame({"strain_id": strains}).to_csv(
        root / "preprocess" / "data.csv", index=False
    )
    (root / "processed").mkdir()
    env = lmdb.open(str(root / "processed" / "lmdb"), map_size=int(1e8))
    with env.begin(write=True) as txn:
        for i, rec in enumerate(records):
            txn.put(str(i).encode(), pickle.dumps(rec))
    env.close()


def test_verify_build_dispatches_by_store_and_reads_the_caudal_store(
    proteome: m.ProteomeMuenzner2024Dataset,
    turnover: m.ProteinTurnoverMuenzner2024Dataset,
    data_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``verify_build`` picks the gate from the root's name; the Caudal sample comes
    from that store's LMDB in ``data.csv`` order (first, middle, last, first SACE_).
    """
    _caudal_store(data_root, [proteome[0], proteome[1]])
    _library_tables(data_root / "torchcell-library", monkeypatch)
    # verify_build reopens each store's LMDB; an env this process already holds
    # (the fixtures read records) is refused by lmdb, so release both first.
    proteome.close_lmdb()
    turnover.close_lmdb()
    monkeypatch.setattr(
        m, "_sgd_genes", lambda root: {"YAL001C", "YBR001W", "Q0010", "R0040C"}
    )
    monkeypatch.setattr(m, "EXPECTED_PROTEOME_RECORDS", 4)
    monkeypatch.setattr(m, "EXPECTED_CAUDAL_OVERLAP", 2)
    monkeypatch.setattr(m, "EXPECTED_TURNOVER_RECORDS", 3)
    for dataset, name in (
        (proteome, "proteome_muenzner2024"),
        (turnover, "protein_turnover_muenzner2024"),
    ):
        link = tmp_path / name
        link.symlink_to(dataset.root)
        report = m.verify_build(str(link), data_root=str(data_root))
        assert report.dataset_name == name
        assert [r.name for r in report.results if not r.passed] == []
    assert m.caudal_strain_order(str(data_root)) == ["AAA", "SACE_YAU"]
    sample = m.caudal_genotype_sample(
        str(data_root), ["AAA", "SACE_YAU"], {"AAA", "SACE_YAU"}
    )
    assert sorted(sample) == ["AAA", "SACE_YAU"]
    with pytest.raises(ValueError, match="is not a Muenzner 2024 store"):
        m.verify_build(str(tmp_path / "elsewhere"), data_root=str(data_root))


def test_caudal_sample_refuses_a_store_out_of_data_csv_order(
    proteome: m.ProteomeMuenzner2024Dataset, data_root: Path
) -> None:
    _caudal_store(data_root, [proteome[1], proteome[0]])
    with pytest.raises(ValueError, match="Caudal record 0 is SACE_YAU, not AAA"):
        m.caudal_genotype_sample(str(data_root), ["AAA", "SACE_YAU"], {"AAA"})


# --------------------------------------------------------------------------- #
# the real release (mirror-backed; run with --data)
# --------------------------------------------------------------------------- #
_LIBRARY = osp.join(os.environ.get("DATA_ROOT", ""), "torchcell-library")
_needs_library = pytest.mark.skipif(
    not osp.isdir(osp.join(_LIBRARY, m.CITATION_KEY)),
    reason="requires the Muenzner 2024 library mirror at $DATA_ROOT",
)


@pytest.mark.data
@_needs_library
@pytest.mark.parametrize("key", sorted(m.SOURCED_VALUES))
def test_every_sourced_value_audits_against_the_mirror(key: str) -> None:
    """Each quote is a verbatim substring of its mirrored file at the pinned sha256."""
    result = audit_sourced_value(m.SOURCED_VALUES[key], _LIBRARY)
    assert result.passed, result.message


@pytest.mark.data
def test_the_deposited_workbook_parses_to_the_measured_counts() -> None:
    """SI Table 3: 1,576 proteins x 796 isolates, 37,667 'NA' cells; SI Table 17:
    58,061 rows over 55 isolates with HL = ln 2 / kdp to 6e-15.
    """
    path = m.raw_mirror_dir() / m.SI_TABLES_REL
    if not path.exists():
        pytest.skip("raw mirror not deposited on this machine")
    tables = m.read_sheets(str(path), [m.T3, m.T17])
    values = m.proteome_matrix(tables[m.T3.name])
    assert values.shape == (1576, 796)
    assert int(values.isna().to_numpy().sum()) == 37667
    frame = m.turnover_frame(tables[m.T17.name])
    assert (len(frame), frame["strain"].nunique()) == (58061, 55)
    assert m.half_life_identity_error(frame) < 1e-14


def test_the_loader_names_the_shared_genotype_files_and_classes() -> None:
    """The genotype is ``natural_isolate``'s, and the loader's own imports name every
    class it holds (the schema-impact closure reads them).
    """
    assert m.RAW_FILES[1:] == list(ni.GENOTYPE_FILES)
    assert m.GENOTYPE_CLASSES == ni.ISOLATE_GENOTYPE_CLASSES
