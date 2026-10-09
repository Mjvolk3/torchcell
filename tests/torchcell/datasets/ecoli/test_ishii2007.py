# tests/torchcell/datasets/ecoli/test_ishii2007.py
# [[tests.torchcell.datasets.ecoli.test_ishii2007]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_ishii2007.py
"""The Ishii 2007 paired-modality loader (``torchcell.datasets.ecoli.ishii2007``).

Synthetic tests (run everywhere) build the workbook in memory. The release is a legacy
BIFF ``.xls`` and nothing in the environment WRITES that format, so the fixture is a
fake that duck-types the narrow ``xlrd`` surface the loader reads -- ``sheet_by_name``,
``sheet_names``, ``nrows``, ``ncols``, ``cell_value``, ``cell_xf_index`` and the book's
``xf_list[...].background.pattern_colour_index``. That last one is not incidental: the
``Metabolite`` sheet encodes the CE-TOFMS protocol as a FILL COLOUR and the
``Information`` sheet prints the legend as three coloured swatches, so the fake carries
colours and the protocol split is exercised rather than stubbed.

The synthetic panel is two disruptants over the real ``EcoliK12BW25113Genome`` on the
synthetic BW25113 assembly of ``_bacterial_fixtures`` (``thrA`` on BW25113_0002,
``hokC`` on BW25113_4412), with ``thrA`` grown twice so the duplicate-culture rule and
its named preference are both live, a GR column with data so the dilution-rate RECORD is
live, a second GR column that is empty so the no-data drop is live, and a ``gpmG``
protein row that carries nothing so the retired-symbol check is live:

    sample   name        kind           series   fate
    KO01     hokC        disruptant     1        record at 0.2 h-1, reference RF02
    KO05x    thrA_1      disruptant     1        dropped, duplicate_culture (KO05 kept)
    KO05     thrA_2      disruptant     2        record at 0.2 h-1, reference RF03
    GR01     WT, 0.1h-1  dilution_rate  2        record at 0.1 h-1, reference RF03
    GR04x    WT, 0.7h-1  dilution_rate  -        dropped, no_data_in_this_layer
    RF02     WT(Mar)     reference      1        reference of series 1, 0.2 h-1
    RF03     WT(Jun)     reference      2        reference of series 2, 0.2 h-1

The fake also carries the ``IDs`` sheet, because the dilution rate of a ``GR`` column is
read off its label and cross-checked against that roster, and the synthetic arm is
narrowed to ``(0.1, 0.2, 0.7)`` since the fixture holds two of the release's five rates.

Data-gated tests (``@pytest.mark.data``) read the real raw mirror and the three built
dev-tree LMDBs under ``$DATA_ROOT`` (they never build them): the manifest pins, the
provenance audit of every sourced value, the record counts, one hand-checked record per
arm, and the L0-L4 verifier. Every one closes its LMDB handle before the next reads the
same store.
"""

from __future__ import annotations

import csv
import json
import math
import os
import os.path as osp
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import xlrd

import torchcell.datasets.ecoli.ishii2007 as m
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    BW25113_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.datamodels.schema import AssemblyReferenceGenome, ComponentDefinition
from torchcell.sequence.genome.ecoli.k12 import BW25113_ASSEMBLY, EcoliK12BW25113Genome
from torchcell.verification.sourced import audit_sourced_value

# --------------------------------------------------------------------------- #
# A fake xlrd workbook: the loader's whole reading surface, colours included
# --------------------------------------------------------------------------- #
#: Fill-colour indices, chosen to match the real workbook's so the fixture reads like it.
ANION, CATION, NUCLEOTIDE, NO_FILL = 47, 41, 42, 64


class FakeBackground:
    """The ``background`` of an xlrd extended-format record."""

    def __init__(self, colour: int) -> None:
        """Store the fill colour this background reports."""
        self.pattern_colour_index = colour


class FakeXf:
    """One xlrd extended-format record: all this loader reads is its background."""

    def __init__(self, colour: int) -> None:
        """Wrap one fill colour as an extended-format record."""
        self.background = FakeBackground(colour)


class FakeSheet:
    """A rectangular sheet of cell values, plus a per-cell fill colour."""

    def __init__(
        self,
        name: str,
        rows: Sequence[Sequence[Any]],
        colours: Mapping[tuple[int, int], int] | None = None,
    ) -> None:
        """Pad every row to the widest and index the per-cell fill colours."""
        self.name = name
        width = max((len(row) for row in rows), default=0)
        self._rows = [list(row) + [""] * (width - len(row)) for row in rows]
        self.nrows = len(self._rows)
        self.ncols = width
        self._colours = dict(colours or {})

    def cell_value(self, row: int, col: int) -> Any:
        """The cell's stored value."""
        return self._rows[row][col]

    def cell_xf_index(self, row: int, col: int) -> int:
        """The cell's extended-format index, which is its fill colour here."""
        return self._colours.get((row, col), NO_FILL)


class FakeBook:
    """A workbook of :class:`FakeSheet`, indexing ``xf_list`` by colour."""

    def __init__(self, sheets: Sequence[FakeSheet]) -> None:
        """Index the sheets by name."""
        self._sheets = {sheet.name: sheet for sheet in sheets}
        self.xf_list = _ColourIndexedXfList()

    def sheet_names(self) -> list[str]:
        """Every sheet name, in order."""
        return list(self._sheets)

    def sheet_by_name(self, name: str) -> FakeSheet:
        """One sheet by name."""
        return self._sheets[name]


class _ColourIndexedXfList:
    """``xf_list[i]`` is the format whose fill colour IS ``i``."""

    def __getitem__(self, index: int) -> FakeXf:
        return FakeXf(index)


# --------------------------------------------------------------------------- #
# The synthetic release
# --------------------------------------------------------------------------- #
PROTOCOL_ROWS = {"anion": 2, "nucleotide": 1}
SYNTHETIC_DISRUPTANTS = ("thrA", "hokC")
#: The release ran five dilution rates; the fixture holds the 0.2 h-1 reference, one
#: served rate (GR01 at 0.1) and one empty column (GR04x at 0.7).
SYNTHETIC_DILUTION_RATE_ARM = (0.1, 0.2, 0.7)
REFERENCE = AssemblyReferenceGenome(
    species="Escherichia coli",
    strain="BW25113",
    assembly_set="ecoli_K12_BW25113_ASM75055v1",
    assembly_accession="GCA_000750555.1",
)


def _information_sheet() -> FakeSheet:
    rows: list[list[Any]] = [["", "", ""] for _ in range(81)]
    rows[32] = ["Measurements", "Unit", ""]
    rows[33] = ["Metabolites (intracellular)", "mM", ""]
    rows[34] = ["Proteins", "mg-protein/g-dry cell weight", ""]
    rows[77] = [
        "In the metabolite sheet, the colors of column A (metabolite names) denote "
        "measurement protocols.",
        "",
        "",
    ]
    rows[78] = ["", "", "Method :Anion"]
    rows[79] = ["", "", "Method: Cation"]
    rows[80] = ["", "", "Method: Nucleotide"]
    colours = {(78, 1): ANION, (79, 1): CATION, (80, 1): NUCLEOTIDE}
    return FakeSheet(m.SHEET_INFORMATION, rows, colours)


def _ids_sheet() -> FakeSheet:
    """The release's sample roster: Sample ID in column B, Sample Name in column C.

    The roster calls both pfkA cultures by the bare gene symbol while the data sheets
    suffix them, exactly as the real release does, so the roster cross-check is the
    dilution-rate arm's and nothing else's.
    """
    rows: list[list[Any]] = [
        ["", "", "", ""],
        ["", "", "Series ID", ""],
        ["", "Sample ID", "Sample Name", "Culture Date"],
        ["", "KO01", "hokC", "38512"],
        ["", "KO05x", "thrA", "38624"],
        ["", "KO05", "thrA", "38652"],
        ["", "", "", ""],
        ["", "GR01", "WT, 0.1h-1", "38645"],
        ["", "GR04x", "WT, 0.7h-1", "38693"],
        ["", "", "", ""],
        ["", "RF02", "WT, 0.2h-1", "38442"],
        ["", "RF03", "WT, 0.2h-1", "38526"],
    ]
    return FakeSheet(m.SHEET_IDS, rows)


def _metabolite_sheet() -> FakeSheet:
    rows: list[list[Any]] = [
        [
            "Sample ID",
            "KO01",
            "KO05x",
            "KO05",
            "GR01",
            "GR04x",
            "RF02",
            "RF03",
            "RFs",
            "",
            "",
        ],
        ["Series ID", 1.0, 1.0, 2.0, 2.0, "", 1.0, 2.0, "RFs", "", ""],
        [
            "",
            "hokC",
            "thrA_1",
            "thrA_2",
            "WT, 0.1h-1",
            "",
            "WT(Mar)",
            "WT(Jun)",
            "Ave",
            "SD",
            "CV",
        ],
        ["Pyruvate", 1.0, 2.0, 3.0, 4.0, "", 5.0, 6.0, 5.5, 0.5, 9.1],
        ["Citrate", "", "", "", "", "", "", "", "", "", ""],
        ["Citrate", 7.0, 8.0, 9.0, 10.0, "", 11.0, "", 11.0, 0.0, 0.0],
    ]
    colours = {(3, 0): ANION, (4, 0): ANION, (5, 0): NUCLEOTIDE}
    return FakeSheet(m.SHEET_METABOLITE, rows, colours)


def _protein_sheet() -> FakeSheet:
    rows: list[list[Any]] = [
        [
            "Sample ID",
            "KO01",
            "",
            "KO05",
            "",
            "GR01",
            "",
            "GR04x",
            "",
            "RF02",
            "",
            "RF03",
            "",
        ],
        ["Series ID", 1.0, "", 2.0, "", 2.0, "", "", "", 1.0, "", 2.0, ""],
        [
            "",
            "hokC",
            "",
            "thrA_2",
            "",
            "WT, 0.1h-1",
            "",
            "",
            "",
            "WT(Mar)",
            "",
            "WT(Jun)",
            "",
        ],
        [
            "",
            "Conc",
            "CV",
            "Conc",
            "CV",
            "Conc",
            "CV",
            "",
            "",
            "Conc",
            "CV",
            "Conc",
            "CV",
        ],
        ["thrA", 10.0, 20.0, 11.0, "", 12.0, 5.0, "", "", 13.0, 10.0, 14.0, 10.0],
        ["hokC", 1.0, 50.0, 2.0, 10.0, 3.0, 5.0, "", "", 4.0, 25.0, 5.0, 20.0],
        ["gpmG", "", "", "", "", "", "", "", "", "", "", "", ""],
    ]
    return FakeSheet(m.SHEET_PROTEIN, rows)


def _flux_sheet() -> FakeSheet:
    rows: list[list[Any]] = [
        ["Sample ID", "KO01", "KO05x", "KO05", "GR01", "GR04x", "RF03", "RF04"],
        ["", "hokC", "thrA_1", "thrA_2", "WT, 0.1h-1", "", "WT(Jun)", "WT(Jul)"],
        ["Glucose + PEP -> G6P + PYR", 100.0, "", 100.0, 100.0, "", 100.0, 100.0],
        ["G6P <-> F6P", 60.0, "", "-", 70.0, "", 80.0, 81.0],
        ["Ru5P -> X5P", -5.0, "", -4.0, -3.0, "", -2.0, -1.0],
        ["", "", "", "", "", "", "", ""],
        ["Exch. (G6P <-> F6P)", 0.5, "", 0.4, 0.3, "", 0.2, 0.1],
    ]
    return FakeSheet(m.SHEET_FLUX, rows)


def _rates_sheet() -> FakeSheet:
    rows: list[list[Any]] = [
        ["Sample ID", "KO01", "KO05x", "KO05", "GR01", "GR04x", "RF03"],
        ["", "hokC", "thrA_1", "thrA_2", "WT, 0.1h-1", "", "WT(Jun)"],
        [m.GLUCOSE_UPTAKE_ROW, 3.0, "", 3.1, 1.3, "", 3.2],
        [m.OXYGEN_UPTAKE_ROW, 5.0, "", 5.1, 2.2, "", 5.2],
    ]
    return FakeSheet(m.SHEET_RATES, rows)


def quantitative_book() -> FakeBook:
    """The synthetic ``Quantitative_data.xls``."""
    return FakeBook(
        [
            _information_sheet(),
            _ids_sheet(),
            _metabolite_sheet(),
            _protein_sheet(),
            _flux_sheet(),
            _rates_sheet(),
        ]
    )


def gc_ms_book() -> FakeBook:
    """The synthetic ``Flux_GC-MS_data.xls``: one sheet per culture that was fitted."""
    return FakeBook(
        [
            FakeSheet(m.SHEET_INFORMATION, [["Mass distributions"]]),
            FakeSheet("hokC", [["Ala", "M-57", 0.8, 0.2]]),
            FakeSheet("thrA_1", [["Ala", "M-57", 0.7, 0.3]]),
            FakeSheet("thrA_2", [["Ala", "M-57", 0.6, 0.4]]),
            # Named by SAMPLE ID, as the real workbook names its GR and RF sheets.
            FakeSheet("GR01", [["Ala", "M-57", 0.5, 0.5]]),
        ]
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
    """A dataset root whose raw/ is present and whose workbooks are the fakes."""
    pins: list[Mapping[str, str]] = []

    def record(raw_dir: str, expected: Mapping[str, str]) -> None:
        missing = [f for f in expected if not osp.exists(osp.join(raw_dir, f))]
        if missing:
            raise FileNotFoundError(f"pinned raw files absent: {missing}")
        pins.append(dict(expected))

    def open_workbook(path: str, **_: Any) -> FakeBook:
        return (
            gc_ms_book() if osp.basename(path) == m.GC_MS_FILE else quantitative_book()
        )

    monkeypatch.setattr(m, "verify_raw_files", record)
    monkeypatch.setattr(xlrd, "open_workbook", open_workbook)
    monkeypatch.setattr(m, "EXPECTED_PROTOCOL_ROWS", PROTOCOL_ROWS)
    monkeypatch.setattr(m, "DISRUPTANT_SYMBOLS", SYNTHETIC_DISRUPTANTS)
    monkeypatch.setattr(m, "SUFFIXED_SAMPLE_NAMES", frozenset({"thrA_1", "thrA_2"}))
    monkeypatch.setattr(m, "DILUTION_RATE_ARM", SYNTHETIC_DILUTION_RATE_ARM)
    monkeypatch.setattr(m, "EXPECTED_RECORDS", dict.fromkeys(m.EXPECTED_RECORDS, 3))
    monkeypatch.setattr(m, "EXPECTED_PROTEIN_KEYS_WITHOUT_REFERENCE", 0)
    monkeypatch.setattr(m, "FLUX_REFERENCE_SAMPLE", "RF03")
    monkeypatch.setattr(m, "assembly_reference", lambda strain, **_: REFERENCE)
    root = tmp_path / "ishii2007"
    for slug in m.DATASET_ROOTS:
        raw = root / slug / "raw"
        raw.mkdir(parents=True)
        for raw_file in m.RAW_FILES:
            (raw / raw_file.name).write_bytes(b"")
    return root


@pytest.fixture
def synthetic_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two panel constants the synthetic workbook is smaller than.

    The pinned values describe the real release (579 metabolite rows over three
    protocols, pfkA as the one twice-grown disruptant, five dilution rates); the fixture
    is three rows, thrA and two rates, so the tests that read it narrow exactly those
    three and nothing else.
    """
    monkeypatch.setattr(m, "EXPECTED_PROTOCOL_ROWS", PROTOCOL_ROWS)
    monkeypatch.setattr(m, "SUFFIXED_SAMPLE_NAMES", frozenset({"thrA_1", "thrA_2"}))
    monkeypatch.setattr(m, "DILUTION_RATE_ARM", SYNTHETIC_DILUTION_RATE_ARM)


def _build(cls: type[Any], root: Path, genome: EcoliK12BW25113Genome) -> Any:
    return cls(root=str(root / cls.SLUG), ecoli_genome=genome)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_disruptant_symbol_strips_only_the_known_culture_replicates() -> None:
    assert m.disruptant_symbol("pfkA_1") == "pfkA"
    assert m.disruptant_symbol("pfkA_2") == "pfkA"
    assert m.disruptant_symbol("galM") == "galM"
    assert m.disruptant_symbol("WT, 0.1h-1") == "WT, 0.1h-1"
    with pytest.raises(RuntimeError, match="carries a culture-replicate suffix"):
        m.disruptant_symbol("zwf_3")


def test_standard_error_from_cv_divides_by_the_root_of_the_replicate_count() -> None:
    assert m.standard_error_from_cv(10.0, 20.0, 2) == pytest.approx(
        10.0 * 0.2 / math.sqrt(2)
    )
    assert m.standard_error_from_cv(10.0, 0.0, 2) == 0.0
    assert math.isnan(m.standard_error_from_cv(10.0, None, 2))


def test_dilution_rate_from_name_reads_the_rate_off_the_released_label() -> None:
    """The rate comes from the label, not from the GR numbering."""
    assert m.dilution_rate_from_name("WT, 0.1h-1") == 0.1
    assert m.dilution_rate_from_name("WT, 0.4h-1") == 0.4
    assert m.dilution_rate_from_name("WT, 0.5h-1") == 0.5
    assert m.dilution_rate_from_name("WT, 0.7h-1") == 0.7
    with pytest.raises(RuntimeError, match="states no dilution rate"):
        m.dilution_rate_from_name("WT(Jun)")
    with pytest.raises(RuntimeError, match="states no dilution rate"):
        m.dilution_rate_from_name("galM")


def test_read_ids_roster_reads_the_releases_own_sample_names() -> None:
    assert m.read_ids_roster(quantitative_book()) == {
        "KO01": "hokC",
        "KO05x": "thrA",
        "KO05": "thrA",
        "GR01": "WT, 0.1h-1",
        "GR04x": "WT, 0.7h-1",
        "RF02": "WT, 0.2h-1",
        "RF03": "WT, 0.2h-1",
    }


def test_read_ids_roster_refuses_a_sheet_without_exactly_one_header() -> None:
    book = quantitative_book()
    book.sheet_by_name(m.SHEET_IDS)._rows[3][1] = "Sample ID"
    with pytest.raises(RuntimeError, match="2 'Sample ID' cells in column 1"):
        m.read_ids_roster(book)


def test_check_dilution_rate_arm_parses_every_gr_column(
    synthetic_constants: None,
) -> None:
    """Both GR columns get a rate from the roster, the unmeasured one included.

    ``GR04x`` carries a blank name in every served sheet, as it does in the release, so
    the roster is the only label it has.
    """
    book = quantitative_book()
    sheet = book.sheet_by_name(m.SHEET_METABOLITE)
    columns = m.read_sample_columns(sheet, paired=False, series_row=1, name_row=2)
    assert [c.name for c in columns if c.sample_id == "GR04x"] == [""]
    assert m.check_dilution_rate_arm(book, columns) == {"GR01": 0.1, "GR04x": 0.7}


def test_check_dilution_rate_arm_refuses_a_gr_column_absent_from_the_roster(
    synthetic_constants: None,
) -> None:
    book = quantitative_book()
    book.sheet_by_name(m.SHEET_IDS)._rows[7][1] = "GR09"
    sheet = book.sheet_by_name(m.SHEET_METABOLITE)
    columns = m.read_sample_columns(sheet, paired=False, series_row=1, name_row=2)
    with pytest.raises(
        RuntimeError, match="GR01 is a sample column of .* not in the IDs sheet"
    ):
        m.check_dilution_rate_arm(book, columns)


def test_check_dilution_rate_arm_refuses_a_label_row_the_roster_contradicts(
    synthetic_constants: None,
) -> None:
    """Two label rows state the rate; a build never picks one of two disagreeing ones."""
    book = quantitative_book()
    book.sheet_by_name(m.SHEET_IDS)._rows[7][2] = "WT, 0.4h-1"
    sheet = book.sheet_by_name(m.SHEET_METABOLITE)
    columns = m.read_sample_columns(sheet, paired=False, series_row=1, name_row=2)
    with pytest.raises(RuntimeError, match="the IDs sheet calls GR01 'WT, 0.4h-1'"):
        m.check_dilution_rate_arm(book, columns)


def test_check_dilution_rate_arm_refuses_rates_that_are_not_the_papers_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m, "EXPECTED_PROTOCOL_ROWS", PROTOCOL_ROWS)
    monkeypatch.setattr(m, "DILUTION_RATE_ARM", (0.2, 0.4, 0.5))
    book = quantitative_book()
    sheet = book.sheet_by_name(m.SHEET_METABOLITE)
    columns = m.read_sample_columns(sheet, paired=False, series_row=1, name_row=2)
    with pytest.raises(
        RuntimeError, match=r"state rates \[0.1, 0.7\], expected the paper's arm"
    ):
        m.check_dilution_rate_arm(book, columns)


def test_culture_identity_separates_a_genotype_from_its_dilution_rate(
    synthetic_constants: None,
) -> None:
    """A disruptant is (symbol, 0.2); a wild-type culture is ('', its own rate)."""
    sheet = _metabolite_sheet()
    columns = m.read_sample_columns(sheet, paired=False, series_row=1, name_row=2)
    by_id = {column.sample_id: column for column in columns}
    rates = {"GR01": 0.1, "GR04x": 0.7}
    assert m.culture_identity(by_id["KO01"], rates) == ("hokC", 0.2)
    assert m.culture_identity(by_id["KO05x"], rates) == ("thrA", 0.2)
    assert m.culture_identity(by_id["KO05"], rates) == ("thrA", 0.2)
    assert m.culture_identity(by_id["GR01"], rates) == ("", 0.1)
    assert m.culture_identity(by_id["RF03"], rates) == ("WT(Jun)", 0.2)
    assert m.culture_dilution_rate(by_id["GR04x"], rates) == 0.7


def test_wild_type_genotype_carries_no_perturbation() -> None:
    """Nothing is invented for a culture whose only difference is its dilution rate."""
    assert m.wild_type_genotype().perturbations == []
    assert len(m.wild_type_genotype()) == 0


def test_retired_drop_rules_records_that_culture_not_batch_is_gone() -> None:
    """The rule is retired, not narrowed: the build still says what it used to drop."""
    (retired,) = m.RETIRED_DROP_RULES
    assert retired["rule"] == "culture_not_batch"
    assert retired["retired_by"] == "Environment.dilution_rate_per_hour (#753)"
    assert "GR01-GR04" in retired["was_dropping"]
    assert not hasattr(m, "DROP_CULTURE_NOT_BATCH")
    assert [rule.rule for rule in (m.DROP_NO_DATA, m.DROP_REFERENCE)] == [
        "no_data_in_this_layer",
        "reference_sample",
    ]


def test_normalize_series_strips_the_unexplained_marker() -> None:
    assert m.normalize_series("1*") == "1"
    assert m.normalize_series("3") == "3"
    with pytest.raises(RuntimeError, match="carries no series row"):
        m.normalize_series(None)


def test_protocol_legend_and_duplicate_keys_come_off_the_fill_colours(
    synthetic_constants: None,
) -> None:
    """The protocol is read from the colour, and the one repeated name is disambiguated.

    ``Citrate`` is released twice: once on the anion protocol with no values and once on
    the nucleotide protocol with values. Keying on the bare name would let the second
    row silently overwrite the first.
    """
    book = quantitative_book()
    assert m.read_protocol_legend(book) == {
        ANION: "anion",
        CATION: "cation",
        NUCLEOTIDE: "nucleotide",
    }
    rows = m.read_metabolite_rows(book)
    assert rows == [
        (3, "Pyruvate", "anion"),
        (4, "Citrate", "anion"),
        (5, "Citrate", "nucleotide"),
    ]
    assert m.metabolite_keys(rows) == {
        3: "Pyruvate",
        4: "Citrate (anion)",
        5: "Citrate (nucleotide)",
    }


def test_read_metabolite_rows_refuses_changed_protocol_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m, "EXPECTED_PROTOCOL_ROWS", {"anion": 1, "nucleotide": 1})
    with pytest.raises(RuntimeError, match="metabolite protocol row counts"):
        m.read_metabolite_rows(quantitative_book())


def test_metabolite_keys_refuses_a_second_repeated_name() -> None:
    rows = [(3, "Pyruvate", "anion"), (4, "Pyruvate", "cation")]
    with pytest.raises(RuntimeError, match=r"repeats \['Pyruvate'\]"):
        m.metabolite_keys(rows)


def test_classify_columns_gives_every_column_exactly_one_reason(
    synthetic_constants: None,
) -> None:
    """The order is emptiness, reference, duplicate culture; a GR column is a record."""
    sheet = _metabolite_sheet()
    columns = m.read_sample_columns(sheet, paired=False, series_row=1, name_row=2)
    kept, ledger = m.classify_columns(
        sheet, columns, [3, 4, 5], dataset="d", dilution_rates={"GR01": 0.1}
    )
    assert [column.sample_id for column in kept] == ["KO01", "KO05", "GR01"]
    assert {rule.rule: rule.sample_ids for rule in ledger.rules} == {
        "no_data_in_this_layer": ("GR04x",),
        "reference_sample": ("RF02", "RF03"),
        "duplicate_culture_same_genotype_and_environment": ("KO05x",),
    }
    assert (ledger.sample_columns, ledger.kept_records) == (7, 3)
    ledger.check()


def test_classify_columns_keeps_gr04_of_the_two_cultures_at_0_7(
    synthetic_constants: None,
) -> None:
    """``GR04`` and ``GR04x`` are two wild-type cultures at ONE rate, so one is a record.

    The release grew the 0.7 h-1 wild type twice ("Used for 2nd measurement of mRNAs.",
    Culture Date 38693 against 38631). In the three served sheets ``GR04x`` is empty and
    never reaches this rule; in the unserved mRNA layer both carry data, and ``GR04`` is
    the named keeper.
    """
    sheet = FakeSheet(
        "mRNA",
        [
            ["Sample ID", "GR04", "GR04x"],
            ["Series ID", 5.0, 6.0],
            ["", "WT, 0.7h-1", ""],
            ["thrA", 1.0, 2.0],
        ],
    )
    columns = m.read_sample_columns(sheet, paired=False, series_row=1, name_row=2)
    kept, ledger = m.classify_columns(
        sheet, columns, [3], dataset="d", dilution_rates={"GR04": 0.7, "GR04x": 0.7}
    )
    assert [column.sample_id for column in kept] == ["GR04"]
    assert {rule.rule: rule.sample_ids for rule in ledger.rules}[
        "duplicate_culture_same_genotype_and_environment"
    ] == ("GR04x",)
    assert m.PREFERRED_DUPLICATE_SAMPLES == frozenset({"KO05", "GR04"})


def test_classify_columns_refuses_a_duplicate_with_no_named_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m, "SUFFIXED_SAMPLE_NAMES", frozenset({"thrA_1", "thrA_2"}))
    monkeypatch.setattr(m, "EXPECTED_PROTOCOL_ROWS", PROTOCOL_ROWS)
    monkeypatch.setattr(m, "PREFERRED_DUPLICATE_SAMPLES", frozenset())
    sheet = _metabolite_sheet()
    columns = m.read_sample_columns(sheet, paired=False, series_row=1, name_row=2)
    with pytest.raises(RuntimeError, match="PREFERRED_DUPLICATE_SAMPLES"):
        m.classify_columns(
            sheet, columns, [3, 4, 5], dataset="d", dilution_rates={"GR01": 0.1}
        )


def test_sample_columns_label_each_repeated_sample_id_by_its_column() -> None:
    """The Protein sheet repeats RF03 across series, so the ledger needs column labels."""
    sheet = FakeSheet(
        "Protein",
        [
            ["Sample ID", "RF03", "", "RF03", "", "KO01", ""],
            ["Series ID", 2.0, "", 3.0, "", 1.0, ""],
            ["", "WT(Jun)", "", "WT(Jun)", "", "hokC", ""],
            ["", "Conc", "CV", "Conc", "CV", "Conc", "CV"],
        ],
    )
    columns = m.read_sample_columns(sheet, paired=True, series_row=1, name_row=2)
    assert [(c.sample_id, c.label, c.series) for c in columns] == [
        ("RF03", "RF03@B", "2"),
        ("RF03", "RF03@D", "3"),
        ("KO01", "KO01", "1"),
    ]


def test_sample_columns_refuse_a_header_pair_that_is_neither_conc_cv_nor_blank() -> (
    None
):
    sheet = FakeSheet(
        "Protein",
        [
            ["Sample ID", "KO01", ""],
            ["Series ID", 1.0, ""],
            ["", "hokC", ""],
            ["", "Level", "SD"],
        ],
    )
    with pytest.raises(RuntimeError, match="expected 'Conc'/'CV'"):
        m.read_sample_columns(sheet, paired=True, series_row=1, name_row=2)


def test_reference_by_series_requires_one_reference_per_record_series(
    synthetic_constants: None,
) -> None:
    sheet = _metabolite_sheet()
    columns = m.read_sample_columns(sheet, paired=False, series_row=1, name_row=2)
    kept, _ = m.classify_columns(
        sheet, columns, [3, 4, 5], dataset="d", dilution_rates={"GR01": 0.1}
    )
    references = m.reference_by_series(columns, kept)
    assert {series: column.sample_id for series, column in references.items()} == {
        "1": "RF02",
        "2": "RF03",
    }
    orphan = [c for c in kept if c.series == "2"]
    with pytest.raises(RuntimeError, match=r"no reference column for series \['2'\]"):
        m.reference_by_series([c for c in columns if c.sample_id != "RF03"], orphan)


def test_flux_rows_split_net_flux_from_exchange_coefficients() -> None:
    """``Exch.`` rows are reversibility parameters and must never reach ``net_flux``."""
    net, exchange = m.FluxIshii2007Dataset._split_rows(_flux_sheet())
    assert [name for _, name in net] == [
        "Glucose + PEP -> G6P + PYR",
        "G6P <-> F6P",
        "Ru5P -> X5P",
    ]
    assert [name for _, name in exchange] == ["Exch. (G6P <-> F6P)"]


def test_net_flux_omits_a_dash_and_keeps_the_sign() -> None:
    """A '-' means the reaction was excluded from that strain's model: key absence."""
    sheet = _flux_sheet()
    net, _ = m.FluxIshii2007Dataset._split_rows(sheet)
    assert m.FluxIshii2007Dataset._net_flux(sheet, net, 3) == {
        "Glucose + PEP -> G6P + PYR": 100.0,
        "Ru5P -> X5P": -4.0,
    }


def test_check_aerobic_refuses_a_missing_or_non_positive_oxygen_uptake() -> None:
    rates = m.read_specific_rates(quantitative_book())
    m.check_aerobic(rates, ["KO01", "KO05"])
    with pytest.raises(RuntimeError, match="no oxygen uptake rate for"):
        m.check_aerobic(rates, ["KO05x"])
    with pytest.raises(RuntimeError, match="non-positive oxygen uptake rate"):
        m.check_aerobic({"KO01": {m.OXYGEN_UPTAKE_ROW: 0.0}}, ["KO01"])


def test_check_units_refuses_a_changed_measurements_table() -> None:
    m.check_units(quantitative_book())
    book = quantitative_book()
    book.sheet_by_name(m.SHEET_INFORMATION)._rows[33][1] = "uM"
    with pytest.raises(
        RuntimeError, match="the stored measurement_type names the unit"
    ):
        m.check_units(book)


def test_check_gpmg_unused_passes_on_an_empty_row_and_stops_on_a_filled_one() -> None:
    book = quantitative_book()
    assert m.check_gpmg_unused(book) == 0
    book.sheet_by_name(m.SHEET_PROTEIN)._rows[6][1] = 1.0
    with pytest.raises(RuntimeError, match="but is retired on"):
        m.check_gpmg_unused(book)


def test_environment_is_aerobic_with_a_typed_temperature_gap_and_deferred_medium() -> (
    None
):
    """The medium's absence lives on the component, because the mixin forbids a gap on
    a field that holds a value.
    """
    env = m.environment(0.2)
    assert env.aerobicity == "aerobic"
    assert env.temperature is None
    assert env.dilution_rate_per_hour == 0.2
    assert [gap.field for gap in env.provenance_gaps] == ["temperature"]
    assert env.media.is_synthetic is True
    assert env.media.base_medium is None
    (component,) = env.media.components
    assert component.definition is ComponentDefinition.composition_deferred
    assert component.concentration is None
    assert component.defers_to == [
        "Ishii 2007 Supporting Online Material, Materials and Methods "
        f"({m.PUBLISHER_SUPPLEMENT_URL}); not mirrored"
    ]


def test_environment_is_one_environment_per_dilution_rate() -> None:
    """Two rates are two environments, and a rate is required, finite and positive."""
    slow, fast = m.environment(0.1), m.environment(0.7)
    assert (slow.dilution_rate_per_hour, fast.dilution_rate_per_hour) == (0.1, 0.7)
    assert slow.model_dump() != fast.model_dump()
    assert slow.model_dump() == m.environment(0.1).model_dump()
    with pytest.raises(ValueError, match="must be finite and positive, got 0.0"):
        m.environment(0.0)
    with pytest.raises(ValueError, match="must be finite and positive, got -0.2"):
        m.environment(-0.2)
    with pytest.raises(TypeError):
        m.environment()  # type: ignore[call-arg]


def test_flux_phenotype_states_no_interval_and_one_labeling_experiment() -> None:
    phenotype = m.flux_phenotype({"G6P <-> F6P": -3.5})
    assert phenotype.net_flux == {"G6P <-> F6P": -3.5}
    assert (phenotype.net_flux_lower, phenotype.net_flux_upper) == (None, None)
    assert phenotype.confidence_level is None
    assert phenotype.label_statistic_name is None
    assert phenotype.n_samples == 1
    assert phenotype.sample_unit is not None
    assert phenotype.sample_unit.value == "biological_replicate"
    assert [gap.field for gap in phenotype.provenance_gaps] == ["confidence_level"]


def test_not_mirrored_names_the_publisher_block_and_its_manual_recipe() -> None:
    """The row is retrieval-gated on the supplement TEXT, and the recipe says how."""
    supplement = next(
        item for item in m.NOT_MIRRORED if m.PUBLISHER_SUPPLEMENT_URL in item
    )
    assert "MANUAL RECIPE" in supplement
    assert "RetrievalMethod.manual_browser" in supplement
    assert "Cloudflare JavaScript challenge" in supplement
    for raw in m.RAW_FILES:
        assert raw.retrieval.method.value == "direct_url"
        assert raw.retrieval.source_url == f"{m.PROJECT_SITE}{raw.name}"


# --------------------------------------------------------------------------- #
# The synthetic build, end to end
# --------------------------------------------------------------------------- #
def test_metabolome_build_pairs_each_record_with_its_own_series_reference(
    synthetic: Path, bw25113: EcoliK12BW25113Genome
) -> None:
    """Two records, keyed by BW25113 locus tag, each referenced to its own series.

    The reference is restricted to the metabolites the record also detected: KO05's
    series reference RF03 has no ``Citrate (nucleotide)``, so that key is on the record
    and not on the reference, which is what makes the reference a proper subset.
    """
    dataset = _build(m.MetabolomeIshii2007Dataset, synthetic, bw25113)
    try:
        assert len(dataset) == 3
        records = [dataset[i] for i in range(3)]
        loci = [
            r["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
            for r in records[:2]
        ]
        assert loci == ["BW25113_4412", "BW25113_0002"]
        assert [
            r["experiment"]["genotype"]["perturbations"][0]["perturbed_gene_name"]
            for r in records[:2]
        ] == ["hokC", "thrA"]
        assert records[2]["experiment"]["genotype"]["perturbations"] == []
        hok, thr, _wild_type = records
        assert hok["experiment"]["phenotype"]["metabolite_level"] == {
            "Pyruvate": 1.0,
            "Citrate (nucleotide)": 7.0,
        }
        assert hok["reference"]["phenotype_reference"]["metabolite_level"] == {
            "Pyruvate": 5.0,
            "Citrate (nucleotide)": 11.0,
        }
        assert thr["experiment"]["phenotype"]["metabolite_level"] == {
            "Pyruvate": 3.0,
            "Citrate (nucleotide)": 9.0,
        }
        assert thr["reference"]["phenotype_reference"]["metabolite_level"] == {
            "Pyruvate": 6.0
        }
        assert thr["experiment"]["phenotype"]["n_replicates"] == {
            "Pyruvate": 1,
            "Citrate (nucleotide)": 1,
        }
        assert thr["experiment"]["phenotype"]["metabolite_level_se"] is None
        assert (
            thr["experiment"]["phenotype"]["measurement_type"]
            == "ce_tofms_intracellular_concentration_mm"
        )
    finally:
        dataset.close_lmdb()


def test_metabolome_build_serves_the_dilution_rate_culture_as_its_own_environment(
    synthetic: Path, bw25113: EcoliK12BW25113Genome
) -> None:
    """GR01 is a record: no perturbation, 0.1 h-1, referenced to the 0.2 h-1 control.

    The reference is a CROSS-ENVIRONMENT one and says so in the stored bytes: the
    record's environment carries 0.1 and its ``environment_reference`` carries 0.2.
    """
    dataset = _build(m.MetabolomeIshii2007Dataset, synthetic, bw25113)
    try:
        wild_type = dataset[2]
        assert wild_type["experiment"]["genotype"]["perturbations"] == []
        assert wild_type["experiment"]["environment"]["dilution_rate_per_hour"] == 0.1
        assert (
            wild_type["reference"]["environment_reference"]["dilution_rate_per_hour"]
            == 0.2
        )
        assert wild_type["experiment"]["phenotype"]["metabolite_level"] == {
            "Pyruvate": 4.0,
            "Citrate (nucleotide)": 10.0,
        }
        assert wild_type["reference"]["phenotype_reference"]["metabolite_level"] == {
            "Pyruvate": 6.0
        }
        disruptant = dataset[0]
        assert disruptant["experiment"]["environment"]["dilution_rate_per_hour"] == 0.2
        assert (
            disruptant["reference"]["environment_reference"]["dilution_rate_per_hour"]
            == 0.2
        )
    finally:
        dataset.close_lmdb()


def test_metabolome_build_writes_the_protocol_and_record_ledgers(
    synthetic: Path, bw25113: EcoliK12BW25113Genome
) -> None:
    dataset = _build(m.MetabolomeIshii2007Dataset, synthetic, bw25113)
    try:
        out = Path(dataset.preprocess_dir)
        metabolites = (out / "metabolites.csv").read_text().splitlines()
        assert metabolites[0].startswith("key,released_name,protocol,sheet_row")
        assert metabolites[1] == "Pyruvate,Pyruvate,anion,4,5.5,0.5,9.1"
        records = (out / "records.csv").read_text().splitlines()
        assert records[0] == (
            "sample_id,perturbed_gene_symbol,culture_name_verbatim,"
            "dilution_rate_per_hour,series_verbatim,reference_sample_id"
        )
        assert records[1] == "KO01,hokC,hokC,0.2,1,RF02"
        assert records[2] == "KO05,thrA,thrA_2,0.2,2,RF03"
        assert records[3] == 'GR01,,"WT, 0.1h-1",0.1,2,RF03'
        assert len(records) == 4
        drops = json.loads((out / "dropped_records.json").read_text())
        assert drops["kept_records"] == 3
        assert [rule["rule"] for rule in drops["rules"]] == [
            "no_data_in_this_layer",
            "reference_sample",
            "duplicate_culture_same_genotype_and_environment",
        ]
        accounting = json.loads((out / "build_accounting.json").read_text())
        assert [v["field"] for v in accounting["unpinned_environment_values"]] == [
            "media",
            "temperature",
        ]
        assert accounting["reference_dilution_rate_per_hour"] == 0.2
        assert accounting["dilution_rates_loaded"] == [0.1, 0.2, 0.7]
        assert accounting["dilution_rate_by_sample_column"] == {
            "GR01": 0.1,
            "GR04x": 0.7,
        }
        assert [r["rule"] for r in accounting["retired_drop_rules"]] == [
            "culture_not_batch"
        ]
    finally:
        dataset.close_lmdb()


def test_proteome_build_derives_the_se_from_the_cv_over_duplicate_measurement(
    synthetic: Path, bw25113: EcoliK12BW25113Genome
) -> None:
    """Abundance keyed by locus tag; SE = level * CV / 100 / sqrt(2); ``gpmG`` unused."""
    dataset = _build(m.ProteomeIshii2007Dataset, synthetic, bw25113)
    try:
        assert len(dataset) == 3
        hok = dataset[0]
        phenotype = hok["experiment"]["phenotype"]
        assert phenotype["protein_abundance"] == {
            "BW25113_0002": 10.0,
            "BW25113_4412": 1.0,
        }
        assert phenotype["n_replicates"] == {"BW25113_0002": 2, "BW25113_4412": 2}
        assert phenotype["protein_abundance_se"]["BW25113_0002"] == pytest.approx(
            10.0 * 0.20 / math.sqrt(2)
        )
        assert phenotype["protein_abundance_se"]["BW25113_4412"] == pytest.approx(
            1.0 * 0.50 / math.sqrt(2)
        )
        reference = hok["reference"]["phenotype_reference"]
        assert reference["protein_abundance"] == {
            "BW25113_0002": 13.0,
            "BW25113_4412": 4.0,
        }
        assert set(reference["protein_abundance"]) == set(
            phenotype["protein_abundance"]
        )
        thr = dataset[1]
        assert math.isnan(
            thr["experiment"]["phenotype"]["protein_abundance_se"]["BW25113_0002"]
        )
        wild_type = dataset[2]
        assert wild_type["experiment"]["genotype"]["perturbations"] == []
        assert wild_type["experiment"]["environment"]["dilution_rate_per_hour"] == 0.1
        assert (
            wild_type["reference"]["environment_reference"]["dilution_rate_per_hour"]
            == 0.2
        )
        assert wild_type["experiment"]["phenotype"]["protein_abundance"] == {
            "BW25113_0002": 12.0,
            "BW25113_4412": 3.0,
        }
        assert wild_type["reference"]["phenotype_reference"]["protein_abundance"] == {
            "BW25113_0002": 14.0,
            "BW25113_4412": 5.0,
        }
    finally:
        dataset.close_lmdb()


def test_flux_build_stores_one_fit_per_culture_against_one_reference_fit(
    synthetic: Path, bw25113: EcoliK12BW25113Genome
) -> None:
    """A dash is key absence, the sign is kept, and every record shares the RF03 fit."""
    dataset = _build(m.FluxIshii2007Dataset, synthetic, bw25113)
    try:
        assert len(dataset) == 3
        hok, thr, wild_type = dataset[0], dataset[1], dataset[2]
        assert wild_type["experiment"]["genotype"]["perturbations"] == []
        assert wild_type["experiment"]["environment"]["dilution_rate_per_hour"] == 0.1
        assert wild_type["experiment"]["phenotype"]["net_flux"] == {
            "Glucose + PEP -> G6P + PYR": 100.0,
            "G6P <-> F6P": 70.0,
            "Ru5P -> X5P": -3.0,
        }
        assert hok["experiment"]["phenotype"]["net_flux"] == {
            "Glucose + PEP -> G6P + PYR": 100.0,
            "G6P <-> F6P": 60.0,
            "Ru5P -> X5P": -5.0,
        }
        assert thr["experiment"]["phenotype"]["net_flux"] == {
            "Glucose + PEP -> G6P + PYR": 100.0,
            "Ru5P -> X5P": -4.0,
        }
        assert (
            hok["reference"]["phenotype_reference"]["net_flux"]
            == thr["reference"]["phenotype_reference"]["net_flux"]
            == {
                "Glucose + PEP -> G6P + PYR": 100.0,
                "G6P <-> F6P": 80.0,
                "Ru5P -> X5P": -2.0,
            }
        )
        out = Path(dataset.preprocess_dir)
        exchange = (out / "exchange_coefficients.csv").read_text().splitlines()
        assert exchange[0] == "reaction,KO01,KO05,GR01,RF03,RF04"
        assert exchange[1] == "Exch. (G6P <-> F6P),0.5,0.4,0.3,0.2,0.1"
        assert len(exchange) == 2
        flux_records = (out / "records.csv").read_text().splitlines()
        assert flux_records[0] == (
            "sample_id,perturbed_gene_symbol,culture_name_verbatim,"
            "dilution_rate_per_hour,reference_sample_id"
        )
        assert flux_records[3] == 'GR01,,"WT, 0.1h-1",0.1,RF03'

        fits = (out / "flux_reference_fits.csv").read_text().splitlines()
        assert fits[0] == "reaction,RF03,RF04"
        assert len(fits) == 4
    finally:
        dataset.close_lmdb()


def test_flux_build_refuses_a_record_whose_labeling_data_is_not_mirrored(
    synthetic: Path, bw25113: EcoliK12BW25113Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fitted flux is only honest with its input recorded, so a missing sheet stops."""

    def open_workbook(path: str, **_: Any) -> FakeBook:
        if osp.basename(path) == m.GC_MS_FILE:
            return FakeBook([FakeSheet(m.SHEET_INFORMATION, [["Mass distributions"]])])
        return quantitative_book()

    monkeypatch.setattr(xlrd, "open_workbook", open_workbook)
    with pytest.raises(
        RuntimeError,
        match=r"no mass-distribution sheet for \['GR01', 'hokC', 'thrA_2'\]",
    ):
        _build(m.FluxIshii2007Dataset, synthetic, bw25113)


def test_flux_fit_input_sheet_is_the_sample_id_for_a_dilution_rate_culture(
    synthetic_constants: None,
) -> None:
    """The GC-MS workbook names GR/RF sheets by Sample ID and disruptants by name."""
    sheet = _flux_sheet()
    columns = m.read_sample_columns(sheet, paired=False, series_row=None, name_row=1)
    by_id = {column.sample_id: column for column in columns}
    pick = m.FluxIshii2007Dataset._fit_input_sheet
    assert pick(by_id["KO01"]) == "hokC"
    assert pick(by_id["KO05"]) == "thrA_2"
    assert pick(by_id["GR01"]) == "GR01"


def test_flux_build_refuses_a_release_that_dropped_the_stored_reference(
    synthetic: Path, bw25113: EcoliK12BW25113Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "FLUX_REFERENCE_SAMPLE", "RF99")
    with pytest.raises(RuntimeError, match="is not a reference column of the Flux"):
        _build(m.FluxIshii2007Dataset, synthetic, bw25113)


def test_build_refuses_a_panel_that_is_not_the_papers_disruptants(
    synthetic: Path, bw25113: EcoliK12BW25113Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "DISRUPTANT_SYMBOLS", ("thrA", "hokC", "zwf"))
    with pytest.raises(RuntimeError, match="are not the paper's 24"):
        _build(m.MetabolomeIshii2007Dataset, synthetic, bw25113)


def test_build_refuses_a_record_count_other_than_the_pinned_one(
    synthetic: Path, bw25113: EcoliK12BW25113Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "EXPECTED_RECORDS", dict.fromkeys(m.EXPECTED_RECORDS, 4))
    with pytest.raises(RuntimeError, match="wrote 3 records, the pinned workbook"):
        _build(m.MetabolomeIshii2007Dataset, synthetic, bw25113)


def test_build_refuses_a_protein_symbol_off_the_namespace_other_than_gpmg(
    synthetic: Path, bw25113: EcoliK12BW25113Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A renamed protein would be stored under a name that is not a locus of the pin."""

    def open_workbook(path: str, **_: Any) -> FakeBook:
        if osp.basename(path) == m.GC_MS_FILE:
            return gc_ms_book()
        book = quantitative_book()
        book.sheet_by_name(m.SHEET_PROTEIN)._rows[5][0] = "notAGene"
        return book

    monkeypatch.setattr(xlrd, "open_workbook", open_workbook)
    with pytest.raises(RuntimeError, match="off the BW25113 namespace"):
        _build(m.ProteomeIshii2007Dataset, synthetic, bw25113)


def test_build_refuses_an_abundance_with_no_baseline_beyond_the_pinned_count(
    synthetic: Path, bw25113: EcoliK12BW25113Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An extra record key with no reference value would silently leave the record."""

    def open_workbook(path: str, **_: Any) -> FakeBook:
        if osp.basename(path) == m.GC_MS_FILE:
            return gc_ms_book()
        book = quantitative_book()
        book.sheet_by_name(m.SHEET_PROTEIN)._rows[4][9] = ""
        return book

    monkeypatch.setattr(xlrd, "open_workbook", open_workbook)
    with pytest.raises(RuntimeError, match="have no value in their own series"):
        _build(m.ProteomeIshii2007Dataset, synthetic, bw25113)


def test_build_refuses_a_genome_of_another_assembly_set(
    synthetic: Path, bw25113: EcoliK12BW25113Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bw25113, "ASSEMBLY_SET", "ecoli_K12_MG1655_ASM584v2")
    with pytest.raises(ValueError, match="needs the ecoli_K12_BW25113_ASM75055v1"):
        _build(m.FluxIshii2007Dataset, synthetic, bw25113)


# --------------------------------------------------------------------------- #
# Data-gated: the real raw mirror and the three built dev-tree LMDBs
# --------------------------------------------------------------------------- #
def _records(slug: str) -> list[dict[str, Any]]:
    from torchcell.verification.runners import load_records

    return load_records(osp.join(os.environ["DATA_ROOT"], m.DATASET_ROOTS[slug]))


def _disrupted_gene(record: Mapping[str, Any]) -> str | None:
    """The record's disrupted gene symbol, or None for a wild-type culture."""
    perturbations = record["experiment"]["genotype"]["perturbations"]
    if not perturbations:
        return None
    (perturbation,) = perturbations
    return str(perturbation["perturbed_gene_name"])


@pytest.mark.data
def test_raw_mirror_pins_both_workbooks_as_scriptable_direct_urls() -> None:
    """The mirror's manifest matches the module's pins, retrieval method included."""
    manifest = m.load_manifest()
    assert manifest.citation_key == m.CITATION_KEY
    recorded = {record.path: record for record in manifest.files}
    assert sorted(recorded) == [
        "data/Flux_GC-MS_data.xls",
        "data/Quantitative_data.xls",
    ]
    for raw in m.RAW_FILES:
        record = recorded[raw.mirror_relpath]
        assert record.sha256 == raw.sha256
        assert record.bytes == raw.bytes
        assert record.retrieval is not None
        assert record.retrieval.method.value == "direct_url"
        assert record.retrieval.source_url == f"{m.PROJECT_SITE}{raw.name}"
        assert osp.exists(m.raw_mirror_dir() / raw.mirror_relpath)
        assert m._sha256(m.raw_mirror_dir() / raw.mirror_relpath) == raw.sha256
    assert any("MANUAL RECIPE" in item for item in manifest.si_expected)


@pytest.mark.data
def test_every_sourced_value_is_a_verbatim_quote_of_its_pinned_artifact() -> None:
    """Each quote is found byte for byte in the file its provenance names.

    Eight are from the Ishii ``paper.md``, two from Baba 2006 (the Keio deferral), one
    from the ``IDs`` sheet and the rest from the ``Information`` sheet of the mirrored
    workbook.
    """
    for key, value in m.SOURCED_VALUES.items():
        result = audit_sourced_value(value, m.sourced_value_root(value))
        assert result.passed, (key, result.message)
    keys = {value.provenance.citation_key for value in m.SOURCED_VALUES.values()}
    assert keys == {m.CITATION_KEY, m.BABA_KEY}
    assert len(m.SOURCED_VALUES) == 24
    pages = sorted(
        (value.provenance.page or "paper") for value in m.SOURCED_VALUES.values()
    )
    assert pages.count("sheet 'IDs'") == 1
    assert pages.count("sheet 'Information'") == 13


@pytest.mark.data
@pytest.mark.parametrize(
    ("slug", "label"),
    [
        ("metabolome_ishii2007", "metabolite_level"),
        ("proteome_ishii2007", "protein_abundance"),
        ("flux_ishii2007", "net_flux"),
    ],
)
def test_each_arm_holds_the_same_28_cultures(slug: str, label: str) -> None:
    """The three arms are paired: 24 disruptants at 0.2 h-1 plus four wild-type rates."""
    records = _records(slug)
    assert len(records) == m.EXPECTED_RECORDS[slug] == 28
    disruptants = [r for r in records if r["experiment"]["genotype"]["perturbations"]]
    wild_type = [r for r in records if not r["experiment"]["genotype"]["perturbations"]]
    assert (len(disruptants), len(wild_type)) == (24, 4)
    symbols = [
        r["experiment"]["genotype"]["perturbations"][0]["perturbed_gene_name"]
        for r in disruptants
    ]
    assert sorted(symbols) == sorted(m.DISRUPTANT_SYMBOLS)
    loci = {
        r["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        for r in disruptants
    }
    assert len(loci) == 24
    assert all(locus.startswith("BW25113_") for locus in loci)
    assert all(r["experiment"]["phenotype"][label] for r in records)
    assert {
        r["experiment"]["genotype"]["perturbations"][0]["gene_namespace"]
        for r in disruptants
    } == {"ecoli_k12_bw25113_locus_tag"}
    assert {
        r["experiment"]["environment"]["dilution_rate_per_hour"] for r in disruptants
    } == {0.2}
    assert sorted(
        r["experiment"]["environment"]["dilution_rate_per_hour"] for r in wild_type
    ) == [0.1, 0.4, 0.5, 0.7]


@pytest.mark.data
@pytest.mark.parametrize(
    "slug", ["metabolome_ishii2007", "proteome_ishii2007", "flux_ishii2007"]
)
def test_every_record_has_its_own_content_addressed_id(slug: str) -> None:
    """Four wild-type records share one empty genotype and still never collide.

    The adapter's experiment id is the sha256 of the whole serialized experiment, the
    environment included, so ``Environment.dilution_rate_per_hour`` is what separates
    them. The 0.2 h-1 wild type is only ever a reference, never a record, so there is no
    fifth wild-type experiment for them to collide with, and every record's
    ``environment_reference`` carries that 0.2 h-1.
    """
    import hashlib

    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], Any] = TypeAdapter(ExperimentType).validate_python
    records = _records(slug)
    ids = {
        hashlib.sha256(
            json.dumps(validate(r["experiment"]).model_dump()).encode("utf-8")
        ).hexdigest()
        for r in records
    }
    assert len(ids) == len(records) == 28
    assert {
        r["reference"]["environment_reference"]["dilution_rate_per_hour"]
        for r in records
    } == {0.2}


def _record_ledger(slug: str) -> list[dict[str, str]]:
    """One arm's ``preprocess/records.csv`` as rows."""
    path = (
        Path(os.environ["DATA_ROOT"], m.DATASET_ROOTS[slug], "preprocess")
        / "records.csv"
    )
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.data
def test_the_three_arms_agree_on_the_pfka_culture_they_kept() -> None:
    """``KO05`` is the pfkA record in all three arms, which is what keeps them paired."""
    for slug in m.DATASET_ROOTS:
        rows = _record_ledger(slug)
        assert len(rows) == 28, slug
        (pfka,) = [row for row in rows if row["sample_id"] == "KO05"]
        assert pfka["perturbed_gene_symbol"] == "pfkA", slug
        assert pfka["culture_name_verbatim"] == "pfkA_2", slug
        assert pfka["dilution_rate_per_hour"] == "0.2", slug


@pytest.mark.data
def test_the_record_ledger_names_the_dilution_rate_arm_and_its_reference() -> None:
    """The four wild-type cultures, their rates and the control each is scored against.

    The Metabolite sheet puts ``GR01``-``GR03`` in series 5 and ``GR04`` in series 4, so
    their references are ``RF06`` and ``RF05``; the Protein sheet puts all four in series
    4, whose reference is ``RF03``; the Flux sheet carries no series row, so every record
    there is scored against the stored ``RF03`` fit.
    """
    expected_references = {
        "metabolome_ishii2007": ["RF06", "RF06", "RF06", "RF05"],
        "proteome_ishii2007": ["RF03", "RF03", "RF03", "RF03"],
        "flux_ishii2007": ["RF03", "RF03", "RF03", "RF03"],
    }
    for slug, references in expected_references.items():
        rows = [
            row for row in _record_ledger(slug) if row["sample_id"].startswith("GR")
        ]
        assert [row["sample_id"] for row in rows] == ["GR01", "GR02", "GR03", "GR04"], (
            slug
        )
        assert [row["perturbed_gene_symbol"] for row in rows] == [""] * 4, slug
        assert [row["culture_name_verbatim"] for row in rows] == [
            "WT, 0.1h-1",
            "WT, 0.4h-1",
            "WT, 0.5h-1",
            "WT, 0.7h-1",
        ], slug
        assert [row["dilution_rate_per_hour"] for row in rows] == [
            "0.1",
            "0.4",
            "0.5",
            "0.7",
        ], slug
        assert [row["reference_sample_id"] for row in rows] == references, slug


@pytest.mark.data
def test_the_drop_ledger_accounts_for_every_sample_column_of_every_arm() -> None:
    """28 kept plus one reason each for the rest, and no ``culture_not_batch`` rule.

    35 - 1 - 5 - 1 = 28, 36 - 2 - 6 = 28, 34 - 2 - 4 = 28.
    """
    expected = {
        "metabolome_ishii2007": (35, {"GR04x"}, 5, 1),
        "proteome_ishii2007": (36, {"GR04x", "KO05x"}, 6, 0),
        "flux_ishii2007": (34, {"GR04x", "KO05x"}, 4, 0),
    }
    for slug, (columns, empty, references, duplicates) in expected.items():
        drops = json.loads(
            (
                Path(os.environ["DATA_ROOT"], m.DATASET_ROOTS[slug], "preprocess")
                / "dropped_records.json"
            ).read_text()
        )
        rules = {rule["rule"]: rule["sample_ids"] for rule in drops["rules"]}
        assert (drops["sample_columns"], drops["kept_records"]) == (columns, 28), slug
        assert set(rules["no_data_in_this_layer"]) == empty, slug
        assert len(rules["reference_sample"]) == references, slug
        assert len(rules["duplicate_culture_same_genotype_and_environment"]) == (
            duplicates
        ), slug
        assert sorted(rules) == [
            "duplicate_culture_same_genotype_and_environment",
            "no_data_in_this_layer",
            "reference_sample",
        ], slug
        accounting = json.loads(
            (
                Path(os.environ["DATA_ROOT"], m.DATASET_ROOTS[slug], "preprocess")
                / "build_accounting.json"
            ).read_text()
        )
        assert accounting["dilution_rates_loaded"] == [0.1, 0.2, 0.4, 0.5, 0.7], slug
        assert accounting["dilution_rate_by_sample_column"] == {
            "GR01": 0.1,
            "GR02": 0.4,
            "GR03": 0.5,
            "GR04": 0.7,
            "GR04x": 0.7,
        }, slug
        assert [r["rule"] for r in accounting["retired_drop_rules"]] == [
            "culture_not_batch"
        ], slug


@pytest.mark.data
def test_the_zwf_flux_record_matches_the_released_column_by_hand() -> None:
    """One hand-checked record: zwf, whose reversed pentose-phosphate flux the paper
    singles out ("in the zwf disruptant, the overall flux within the pentose phosphate
    pathway was reversed"). Values read off the Flux sheet's column Q.
    """
    records = _records("flux_ishii2007")
    (zwf,) = [r for r in records if _disrupted_gene(r) == "zwf"]
    flux = zwf["experiment"]["phenotype"]["net_flux"]
    assert len(flux) == 41
    assert flux["Glucose + PEP -> G6P + PYR"] == 100.0
    assert flux["G6P <-> F6P"] == pytest.approx(98.6)
    assert flux["6PG -> Ru5P + CO2"] == 0.0
    # The reversal the paper names: X5P is made FROM Ru5P in the wild type and this
    # fit runs it backwards.
    assert flux["Ru5P -> X5P"] == pytest.approx(-5.067)
    # The deleted enzyme IS glucose-6-phosphate dehydrogenase, and the release excludes
    # its reaction from this strain's model with a dash, which is stored as key absence
    # rather than as a zero flux.
    assert "G6P -> 6PG" not in flux
    assert "6-PG -> G3P + PYR" not in flux
    assert not any(key.startswith("Exch.") for key in flux)
    phenotype = zwf["experiment"]["phenotype"]
    assert phenotype["net_flux_lower"] is None
    assert phenotype["net_flux_upper"] is None
    assert phenotype["confidence_level"] is None
    assert phenotype["n_samples"] == 1


@pytest.mark.data
def test_the_rpe_metabolite_record_matches_the_released_column_by_hand() -> None:
    """One hand-checked metabolome record: rpe, which the paper singles out for "a
    particularly high AEI for metabolites". Values read off the Metabolite sheet's
    column T.
    """
    records = _records("metabolome_ishii2007")
    (rpe,) = [r for r in records if _disrupted_gene(r) == "rpe"]
    levels = rpe["experiment"]["phenotype"]["metabolite_level"]
    assert len(levels) == 134
    assert levels["Ribulose 5-phosphate"] == pytest.approx(6.4644, rel=1e-4)
    assert levels["Ribose 5-phosphate"] == pytest.approx(0.69224, rel=1e-4)
    assert levels["Citrate (nucleotide)"] == pytest.approx(0.092218, rel=1e-4)
    assert set(rpe["reference"]["phenotype_reference"]["metabolite_level"]) <= set(
        levels
    )
    assert rpe["experiment"]["phenotype"]["metabolite_level_se"] is None


@pytest.mark.data
def test_the_gpmg_protein_never_becomes_a_key_in_any_record() -> None:
    """It is retired on the pinned annotation and carries no value in the release."""
    keys: set[str] = set()
    for record in _records("proteome_ishii2007"):
        keys |= set(record["experiment"]["phenotype"]["protein_abundance"])
    assert m.RETIRED_PROTEIN_SYMBOL not in keys
    assert all(key.startswith("BW25113_") for key in keys)
    assert len(keys) == 59


@pytest.mark.data
@pytest.mark.parametrize(
    "slug", ["metabolome_ishii2007", "proteome_ishii2007", "flux_ishii2007"]
)
def test_verification_passes_l0_to_l4_on_the_built_store(slug: str) -> None:
    report = m.run_verification(slug)
    assert report.passed, report.summary()
    levels = {result.level.name for result in report.results}
    assert {"L0", "L1", "L2", "L3", "L4"} <= levels


# --------------------------------------------------------------------------- #
# The raw mirror and the CLI, over a temporary DATA_ROOT
# --------------------------------------------------------------------------- #
FAKE_BYTES = {
    m.QUANTITATIVE_FILE: b"quantitative workbook bytes",
    m.GC_MS_FILE: b"gc-ms workbook bytes",
}


@pytest.fixture
def tiny_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A temporary ``DATA_ROOT`` and a download dir holding two tiny pinned files.

    ``RAW_FILES`` is re-pinned to those bytes so the deposit's sha256 refusal and its
    idempotence are exercised on real files instead of on the 824 kB release.
    """
    data_root = tmp_path / "data_root"
    download = tmp_path / "download"
    download.mkdir()
    files: list[m.RawFile] = []
    for name, payload in FAKE_BYTES.items():
        (download / name).write_bytes(payload)
        files.append(
            m.RawFile(
                name=name,
                sha256=hashlib_sha256(payload),
                bytes=len(payload),
                description=f"tiny stand-in for {name}",
            )
        )
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setattr(m, "RAW_FILES", tuple(files))
    monkeypatch.setattr(m, "DATA_SHA256", {f.name: f.sha256 for f in files})
    monkeypatch.setattr(m, "RAW_FILES_BY_NAME", {f.name: f for f in files})
    return data_root, download


def hashlib_sha256(payload: bytes) -> str:
    """The sha256 of some bytes, as the loader records it."""
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def test_deposit_raw_mirror_writes_a_manifest_naming_the_direct_url_retrieval(
    tiny_release: tuple[Path, Path],
) -> None:
    data_root, download = tiny_release
    root = m.deposit_raw_mirror(
        sources={name: download / name for name in m.DATA_SHA256}
    )
    assert root == data_root / m.RAW_DIR_REL
    manifest = m.load_manifest()
    assert manifest.doi == m.PAPER_DOI
    assert [record.path for record in manifest.files] == [
        f"data/{m.QUANTITATIVE_FILE}",
        f"data/{m.GC_MS_FILE}",
    ]
    for record in manifest.files:
        assert record.retrieval is not None
        assert record.retrieval.method.value == "direct_url"
        assert m.manifest_sha256(manifest, record.path) == record.sha256
    assert manifest.si_data_sources == [m.PROJECT_SITE, m.PUBLISHER_SUPPLEMENT_URL]
    with pytest.raises(KeyError, match="is not in the raw-mirror manifest"):
        m.manifest_sha256(manifest, "data/nope.xls")


def test_deposit_raw_mirror_is_idempotent_and_refuses_different_bytes(
    tiny_release: tuple[Path, Path],
) -> None:
    data_root, download = tiny_release
    sources = {name: download / name for name in m.DATA_SHA256}
    first = m.deposit_raw_mirror(sources=sources)
    mirrored = first / "data" / m.GC_MS_FILE
    stamp = mirrored.stat().st_mtime_ns
    m.deposit_raw_mirror(sources=sources)
    assert mirrored.stat().st_mtime_ns == stamp
    mirrored.write_bytes(b"something else entirely")
    with pytest.raises(RuntimeError, match="exists with a different sha256; refusing"):
        m.deposit_raw_mirror(sources=sources)
    (download / m.GC_MS_FILE).write_bytes(b"a corrected re-export")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        m.deposit_raw_mirror(sources=sources)


def test_deposit_raw_mirror_refuses_a_missing_source(
    tiny_release: tuple[Path, Path],
) -> None:
    _, download = tiny_release
    with pytest.raises(KeyError, match="no source given for"):
        m.deposit_raw_mirror(
            sources={m.QUANTITATIVE_FILE: download / m.QUANTITATIVE_FILE}
        )


def test_retrieve_raw_files_runs_the_recorded_retriever_and_checks_the_bytes(
    tiny_release: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``RetrievalRecord`` is what runs, so this IS the re-runnable retrieval."""
    _, download = tiny_release
    seen: list[str] = []

    def run_retriever(record: Any) -> bytes:
        seen.append(record.source_url)
        return FAKE_BYTES[osp.basename(record.source_url)]

    monkeypatch.setattr(m, "run_retriever", run_retriever)
    out = m.retrieve_raw_files(download / "fresh")
    assert seen == [f"{m.PROJECT_SITE}{name}" for name in FAKE_BYTES]
    assert {name: path.read_bytes() for name, path in out.items()} == FAKE_BYTES


def test_sourced_value_root_separates_the_two_mirrors(
    tiny_release: tuple[Path, Path],
) -> None:
    data_root, _ = tiny_release
    assert m.sourced_value_root(m.SOURCED_VALUES["disruptants"]) == (
        data_root / "torchcell-library"
    )
    assert m.sourced_value_root(m.SOURCED_VALUES["metabolite_unit"]) == (
        data_root / "torchcell-raw"
    )


def test_cli_deposit_retrieves_and_writes_the_mirror(
    tiny_release: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_root, download = tiny_release
    monkeypatch.setattr(
        m, "run_retriever", lambda record: FAKE_BYTES[osp.basename(record.source_url)]
    )
    assert (
        m.main(["deposit", "--download-dir", str(download / "fresh"), "--retrieve"])
        == 0
    )
    assert capsys.readouterr().out.strip() == str(data_root / m.RAW_DIR_REL)
    assert m.load_manifest().citation_key == m.CITATION_KEY


def test_cli_build_opens_and_closes_the_requested_arm(
    synthetic: Path,
    bw25113: EcoliK12BW25113Genome,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(synthetic.parent))
    monkeypatch.setattr(
        m,
        "DATASET_ROOTS",
        {slug: osp.join(synthetic.name, slug) for slug in m.DATASET_ROOTS},
    )
    monkeypatch.setattr(m, "bacterial_genome", lambda host, strain: bw25113)
    assert m.main(["build", "--dataset", "flux_ishii2007"]) == 0
    assert capsys.readouterr().out.strip().endswith("len = 3")


# --------------------------------------------------------------------------- #
# The flux arm's own verification rows
# --------------------------------------------------------------------------- #
def _flux_record(**phenotype: Any) -> dict[str, Any]:
    base = {
        "measurement_type": m.MEASUREMENT_TYPE_FLUX,
        "net_flux": {"G6P <-> F6P": 1.0},
        "net_flux_lower": None,
        "net_flux_upper": None,
        "confidence_level": None,
    }
    return {"experiment": {"phenotype": base | phenotype}}


def test_l3_flux_interval_passes_when_no_interval_is_claimed() -> None:
    result = m._l3_flux_interval([_flux_record(), _flux_record()])
    assert result.passed
    assert result.name == "flux_measurement_type_and_interval"
    assert result.details["measurement_types"] == [m.MEASUREMENT_TYPE_FLUX]
    assert result.details["half_stated"] == []


def test_l3_flux_interval_fails_a_half_stated_interval_or_a_mixed_type() -> None:
    """One bound without the other, or a bound with no level, overstates precision."""
    one_bound = m._l3_flux_interval([_flux_record(net_flux_lower={"G6P <-> F6P": 0.5})])
    assert not one_bound.passed
    assert one_bound.details["half_stated"] == [0]
    no_level = m._l3_flux_interval(
        [
            _flux_record(
                net_flux_lower={"G6P <-> F6P": 0.5}, net_flux_upper={"G6P <-> F6P": 1.5}
            )
        ]
    )
    assert not no_level.passed
    mixed = m._l3_flux_interval(
        [_flux_record(), _flux_record(measurement_type="other")]
    )
    assert not mixed.passed
    assert mixed.details["measurement_types"] == [m.MEASUREMENT_TYPE_FLUX, "other"]


def test_containment_result_reports_the_identifiers_outside_the_assembly() -> None:
    passing = m._containment_result("l4", {"BW25113_0002"}, {"BW25113_0002"}, [])
    assert passing.passed
    assert passing.details == {"n_measured": 1, "n_universe": 1, "missing_examples": []}
    failing = m._containment_result(
        "l4", {"BW25113_0002", "gpmG"}, {"BW25113_0002"}, ["gpmG"]
    )
    assert not failing.passed
    assert failing.message.startswith("1 of 2 stored identifiers")
    assert failing.details["missing_examples"] == ["gpmG"]
