# tests/torchcell/datasets/ecoli/test_schmidt2016_replicate_intensity.py
# [[tests.torchcell.datasets.ecoli.test_schmidt2016_replicate_intensity]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_schmidt2016_replicate_intensity.py
"""Schmidt 2016 Table S8 per-culture loader
(``torchcell.datasets.ecoli.schmidt2016_replicate_intensity``).

The synthetic tests reuse the proteome test's Table S6 and Table S25 writers and add a
Table S8 sheet with the release's per-culture ``normInt`` block, the ``Glucose.2``
re-analysis columns and the glucose median, then drive the readers, the key join, the
phenotype builders and a full ``process()`` into ``tmp_path``.

The ``@pytest.mark.data`` tests read the real mirrors and pin what the dendron note
states: 2,058 Table S8 rows, 66 stored culture columns and 3 unstored ``Glucose.2``
columns, the released glucose median equal to the median of its three cultures, the two
``clpB`` rows keyed by their Table S8 row, every quote verbatim in its pinned artifact,
and that within each condition the per-culture records reproduce the released median and
the copies/cell store's per-protein ratios.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import statistics
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
import pytest

import torchcell.datasets.ecoli.schmidt2016 as sm
import torchcell.datasets.ecoli.schmidt2016_replicate_intensity as ri
from tests.torchcell.datasets.ecoli import test_schmidt2016 as ts
from torchcell.data import file_sha256
from torchcell.verification.report import FindingClass, finding_class_of
from torchcell.verification.sourced import ProvenanceGapReason

#: The released Table S8 symbol of each synthetic dataset-2 row: one differs from its
#: Table S6 symbol and one is empty, as five real rows do.
S8_GENES = {"P00001": "ace", "P00003": "dupA", "P00004": "dupA", "P00005": None}
#: The synthetic Table S8 rows: every dataset-2 row once, and P00005 a second time, as
#: the release files P63284 twice.
S8_ACCESSIONS = ("P00001", "P00003", "P00004", "P00005", "P00006", "P00005")
GLUCOSE_2 = ("A14-07088", "A14-07089", "A14-07090")


def _intensity(accession: str, file_name: str, copy: int = 0) -> float:
    """A deterministic per-culture intensity, distinct per row, file and repeat row."""
    return 1.0e5 * int(accession[1:]) + 10.0 * int(file_name[-3:]) + 1.0 + copy


def _gene(accession: str) -> str | None:
    if accession in S8_GENES:
        return S8_GENES[accession]
    return next(row[0] for row in ts.ROWS if row[1] == accession)


def _write_s8(sheet: Any) -> None:
    files = [name for spec in sm.CONDITIONS for name in spec.s25_files]
    header = [
        sm.COL_UNIPROT,
        sm.COL_GENE,
        *(sm._s8_column(name) for name in [*files, *GLUCOSE_2]),
        ri.REFERENCE_MEDIAN_COLUMN,
    ]
    sheet.append([sm._Q_TABLE_S8])
    sheet.append([None] * len(header))
    sheet.append(header)
    glucose = sm.CONDITIONS_BY_COLUMN[sm.REFERENCE_CONDITION].s25_files
    seen: dict[str, int] = {}
    for accession in S8_ACCESSIONS:
        copy = seen.get(accession, 0)
        seen[accession] = copy + 1
        values = [_intensity(accession, name, copy) for name in [*files, *GLUCOSE_2]]
        median = statistics.median(_intensity(accession, n, copy) for n in glucose)
        sheet.append([accession, _gene(accession), *values, median])


def write_workbook(path: Path) -> Path:
    """Table S6 and Table S25 as the proteome test writes them, plus this Table S8."""
    book = openpyxl.Workbook()
    ts._write_s6(book.active)
    _write_s8(book.create_sheet(sm.SHEET_S8))
    ts._write_s25(book.create_sheet(sm.SHEET_S25))
    book.save(path)
    return path


@pytest.fixture
def workbook(tmp_path: Path) -> Path:
    """The synthetic workbook, written once per test."""
    return write_workbook(tmp_path / sm.SI2)


# --------------------------------------------------------------------------- #
# Declared constants and sourcing
# --------------------------------------------------------------------------- #
def test_declared_shape_is_22_conditions_by_3_cultures() -> None:
    files = ri.condition_files()
    assert len(files) == 22
    assert {len(names) for names in files.values()} == {ri.N_CULTURES_PER_CONDITION}
    assert ri.EXPECTED_RECORDS == 66 == sum(len(n) for n in files.values())
    assert ri.MEASUREMENT_TYPE != sm.MEASUREMENT_TYPE
    assert ri.CITATION_KEY == sm.CITATION_KEY
    # the second glucose triplicate is not a stored culture
    stored = {name for names in files.values() for name in names}
    assert stored.isdisjoint(sm.GLUCOSE_REPRODUCIBILITY_FILES)


def test_sourced_values_name_their_pinned_artifacts() -> None:
    assert ri.SOURCED_VALUES["replicate_kind"].value == "biological"
    assert ri.SOURCED_VALUES["replicate_kind"].provenance.sha256 == sm.PAPER_MD_SHA256
    assert ri.SOURCED_VALUES["glucose_2_is_a_reanalysis"].provenance.sha256 == (
        ri.SI1_MD_SHA256
    )
    assert ri.SOURCED_VALUES["replicate_block"].provenance.sha256 == sm.SI2_SHA256


# --------------------------------------------------------------------------- #
# Reading Table S8
# --------------------------------------------------------------------------- #
def test_read_table_s8_intensities_keeps_every_row_and_stored_file(
    workbook: Path,
) -> None:
    rows = ri.read_table_s8_intensities(str(workbook))
    assert [row.uniprot for row in rows] == list(S8_ACCESSIONS)
    assert rows[0].released_gene == "ace"
    assert rows[3].released_gene is None
    stored = {name for names in ri.condition_files().values() for name in names}
    for row in rows:
        assert set(row.intensity) == stored
    assert rows[0].intensity["A14-07036"] == _intensity("P00001", "A14-07036")
    assert ri.unstored_columns(str(workbook)) == [
        sm._s8_column(name) for name in GLUCOSE_2
    ]


def test_read_table_s8_refuses_a_non_numeric_stored_cell(tmp_path: Path) -> None:
    path = write_workbook(tmp_path / sm.SI2)
    book = openpyxl.load_workbook(path)
    book[sm.SHEET_S8].cell(row=4, column=3, value="NA")
    book.save(path)
    with pytest.raises(RuntimeError, match="not a number"):
        ri.read_table_s8_intensities(str(path))


def test_reference_median_is_checked_against_the_three_glucose_cultures(
    workbook: Path,
) -> None:
    rows = ri.read_table_s8_intensities(str(workbook))
    summary = ri.check_reference_median(rows)
    assert summary["worst_rel_dev"] == 0.0
    assert summary["files"] == ["A14-07036", "A14-07037", "A14-07038"]
    drifted = rows[0].model_copy(update={"reference_median": 1.0})
    with pytest.raises(RuntimeError, match="is not the median"):
        ri.check_reference_median([drifted])


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #
def test_keys_join_through_the_accession_to_the_copies_store(workbook: Path) -> None:
    rows = ri.read_table_s8_intensities(str(workbook))
    s6_rows, _ = sm.read_table_s6(str(workbook))
    keys = ri.protein_keys(rows, s6_rows)
    by_row = {row.row_number: row for row in rows}
    stored = {
        by_row[number].uniprot + f"@{number}": key for number, key in keys.items()
    }
    # the Table S6 key, not Table S8's own symbol
    assert stored["P00001@4"] == "aceA"
    assert stored["P00003@5"] == "dupA | Table S6 row 6"
    assert stored["P00006@8"] == "ghost"
    # an accession Table S8 files twice is keyed by its own Table S8 rows
    assert stored["P00005@7"] == "None | Table S8 row 7"
    assert stored["P00005@9"] == "None | Table S8 row 9"


def test_keys_refuse_an_accession_table_s6_does_not_carry(workbook: Path) -> None:
    rows = ri.read_table_s8_intensities(str(workbook))
    s6_rows, _ = sm.read_table_s6(str(workbook))
    stray = rows[0].model_copy(update={"uniprot": "Q99999"})
    with pytest.raises(RuntimeError, match="is not in Table S6"):
        ri.protein_keys([stray, *rows[1:]], s6_rows)


# --------------------------------------------------------------------------- #
# Phenotypes
# --------------------------------------------------------------------------- #
def test_a_culture_record_is_one_culture_verbatim_and_the_reference_the_median(
    workbook: Path,
) -> None:
    rows = ri.read_table_s8_intensities(str(workbook))
    s6_rows, _ = sm.read_table_s6(str(workbook))
    keys = ri.protein_keys(rows, s6_rows)
    culture = ri.culture_phenotype(rows, keys, "A14-07044")
    assert culture.replicate_id == "A14-07044"
    assert set(culture.n_replicates.values()) == {1}
    assert culture.protein_abundance_se is None
    assert culture.measurement_type == ri.MEASUREMENT_TYPE
    assert culture.protein_abundance["aceA"] == _intensity("P00001", "A14-07044")
    (gap,) = culture.provenance_gaps
    assert (gap.field, gap.reason) == (
        "protein_abundance_se",
        ProvenanceGapReason.not_applicable_single_measurement,
    )
    reference = ri.reference_phenotype(rows, keys)
    assert reference.replicate_id is None
    assert set(reference.n_replicates.values()) == {3}
    assert set(reference.protein_abundance) == set(culture.protein_abundance)
    assert reference.protein_abundance["aceA"] == rows[0].reference_median
    (gap,) = reference.provenance_gaps
    assert gap.reason is ProvenanceGapReason.not_reported_by_primary


# --------------------------------------------------------------------------- #
# End-to-end build on the synthetic workbook
# --------------------------------------------------------------------------- #
def test_process_builds_one_record_per_culture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "proteome_replicate_intensity_schmidt2016"
    raw = root / "raw"
    raw.mkdir(parents=True)
    path = write_workbook(raw / sm.SI2)
    monkeypatch.setattr(sm, "DATA_SHA256", {sm.SI2: file_sha256(path)})
    monkeypatch.setattr(ri, "assembly_reference", lambda strain: ts._reference_genome())
    monkeypatch.setattr(ri, "EXPECTED_PROTEIN_KEYS", len(S8_ACCESSIONS))

    dataset = ri.ProteomeReplicateIntensitySchmidt2016Dataset(root=str(root))
    assert len(dataset) == ri.EXPECTED_RECORDS
    items = [dataset.transform_item(dataset[i]) for i in range(len(dataset))]
    files = [name for names in ri.condition_files().values() for name in names]
    assert [i["experiment"].phenotype.replicate_id for i in items] == files
    assert {len(i["experiment"].genotype.perturbations) for i in items} == {0}
    assert {len(i["experiment"].phenotype.protein_abundance) for i in items} == {
        len(S8_ACCESSIONS)
    }
    assert {i["reference"].phenotype_reference.replicate_id for i in items} == {None}
    environments = {i["experiment"].environment.model_dump_json() for i in items}
    assert len(environments) == 22
    dataset.close_lmdb()

    preprocess = Path(dataset.preprocess_dir)
    ledger = json.loads((preprocess / "dropped_records.json").read_text())
    assert ledger["kept_records"] == ri.EXPECTED_RECORDS
    assert ledger["source_normint_columns"] == 69
    assert ledger["not_stored"]["columns"] == [
        sm._s8_column(name) for name in GLUCOSE_2
    ]
    assert ledger["dropped_protein_rows"] == 0
    identifiers = pd.read_csv(preprocess / "protein_identifiers.csv")
    assert len(identifiers) == len(S8_ACCESSIONS)
    records = pd.read_csv(preprocess / "records.csv")
    assert int(records["is_reference_condition"].sum()) == 3

    report = ri.verify_build(str(root), genome=ts.FakeGenome())  # type: ignore[arg-type]
    assert report.passed
    assert all(
        finding_class_of(r.name) is FindingClass.report
        for r in report.results
        if not r.passed
    )


def test_culture_uniqueness_rule_refuses_a_repeated_culture() -> None:
    environment = {"media": "m"}
    record = {
        "experiment": {
            "environment": environment,
            "phenotype": {"replicate_id": "A14-07036"},
        }
    }
    assert ri.culture_uniqueness_rule([record] * 2).passed is False
    three = [
        {
            "experiment": {
                "environment": environment,
                "phenotype": {"replicate_id": f"A14-0703{k}"},
            }
        }
        for k in (6, 7, 8)
    ]
    assert ri.culture_uniqueness_rule(three).passed is True


# --------------------------------------------------------------------------- #
# Data-gated: the real mirrors
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT is not set")
    return data_root


def _real_workbook() -> str:
    path = osp.join(_data_root(), sm.RAW_DIR_REL, sm.SI2_MIRROR_RELPATH)
    if not osp.exists(path):
        pytest.skip(f"raw mirror not deposited: {path}")
    return path


@pytest.mark.data
def test_real_quotes_are_verbatim_in_their_pinned_artifacts() -> None:
    library = Path(_data_root(), sm.LIBRARY_DIR_REL)
    si1 = library / ri.SI1_MD
    paper = library / sm.PAPER_MD
    assert file_sha256(si1) == ri.SI1_MD_SHA256
    assert file_sha256(paper) == sm.PAPER_MD_SHA256
    for quote in ri.SI1_QUOTES.values():
        assert quote in si1.read_text(encoding="utf-8")
    for quote in ri.PAPER_QUOTES.values():
        assert quote in paper.read_text(encoding="utf-8")
    book = openpyxl.load_workbook(_real_workbook(), read_only=True, data_only=True)
    try:
        s8 = list(book[sm.SHEET_S8].iter_rows(values_only=True, max_row=2))
        s25 = list(book[sm.SHEET_S25].iter_rows(values_only=True, max_row=3))
    finally:
        book.close()
    assert ri._Q_TABLE_S8_BLOCK in {str(c).strip() for c in s8[1] if c is not None}
    assert ri._Q_TABLE_S25_HEADER in {str(c) for c in s25[2] if c is not None}


@pytest.mark.data
def test_real_release_is_2058_rows_66_cultures_and_3_unstored_columns() -> None:
    path = _real_workbook()
    rows = ri.read_table_s8_intensities(path)
    assert len(rows) == ri.EXPECTED_PROTEIN_KEYS == 2058
    assert {len(row.intensity) for row in rows} == {ri.EXPECTED_RECORDS}
    assert ri.unstored_columns(path) == [
        "normInt_A14.07088.1",
        "normInt_A14.07089.1",
        "normInt_A14.07090.1",
    ]
    assert ri.check_reference_median(rows)["worst_rel_dev"] == 0.0
    s6_rows, _ = sm.read_table_s6(path)
    keys = ri.protein_keys(rows, s6_rows)
    assert sorted(k for k in keys.values() if ri.S8_ROW_SEPARATOR in k) == [
        "clpB | Table S8 row 1040",
        "clpB | Table S8 row 1973",
    ]
    by_accession = {row.uniprot: keys[row.row_number] for row in rows}
    assert by_accession["P36680"] == "zapD"  # Table S8 prints yacF
    assert by_accession["E1MTY0"] == "MettuDRAFT_4149"  # Table S8 prints Mett


@pytest.mark.data
def test_real_cultures_reproduce_the_released_median_and_the_copies_ratios() -> None:
    """Grouped by condition, the culture records give back what the stores held.

    The median of a condition's three cultures, over the glucose median, is the stored
    copies/cell over the glucose copies/cell times ONE constant per condition, the same
    for every protein (the authors' per-condition protein-mass adjustment; LB 1.5353).
    """
    path = _real_workbook()
    rows = ri.read_table_s8_intensities(path)
    s6_rows, _ = sm.read_table_s6(path)
    copies = {row.uniprot: row.copies for row in s6_rows}
    glucose = sm.CONDITIONS_BY_COLUMN[sm.REFERENCE_CONDITION].s25_files
    ratios: dict[str, list[float]] = {}
    for row in rows:
        if sum(1 for r in rows if r.uniprot == row.uniprot) > 1:
            continue
        g_median = statistics.median(row.intensity[n] for n in glucose)
        assert g_median == row.reference_median
        for column, files in ri.condition_files().items():
            median = statistics.median(row.intensity[n] for n in files)
            stored = copies[row.uniprot][column]
            reference = copies[row.uniprot][sm.REFERENCE_CONDITION]
            assert stored is not None and reference is not None
            ratios.setdefault(column, []).append(
                (stored / reference) / (median / g_median)
            )
    assert len(ratios) == 22
    assert {len(values) for values in ratios.values()} == {2056}
    spread = max((max(v) - min(v)) / min(v) for v in ratios.values())
    assert spread < 1e-8
    assert ratios[sm.REFERENCE_CONDITION][0] == 1.0
    assert ratios["LB"][0] == pytest.approx(1.5353048, abs=1e-6)
