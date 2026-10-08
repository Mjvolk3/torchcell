# torchcell/datasets/private_torchcell/bioscreen.py
# [[torchcell.datasets.private_torchcell.bioscreen]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/private_torchcell/bioscreen.py
# Test file: tests/torchcell/datasets/private_torchcell/test_bioscreen.py
"""Bioscreen C inhibitor runs of 2021 on strain bAID: mirrors, plate layouts, traits.

The in-house runs behind the 2021 preliminary exam (library key
``volkPreliminaryExamReport2021``), read from the thesis archive on ``/bulk``
(``/bulk/thesis/thesis_archive``, every file sha256-listed in its ``MANIFEST.tsv``):

ex21
    single-inhibitor titrations: six inhibitors, nine doses each plus an uninhibited
    well, in blocks of ten wells per biological replicate (three replicates).
ex23
    the 63 non-empty combinations of the six inhibitors at one dose each, biological
    triplicate, with uninhibited (``WT``) and uninoculated (``blank``) wells.
ex26, ex27, ex28
    two-inhibitor isoboles (furfural, formic acid or 5-HMF against acetic acid): a
    10 x 10 grid of doses on each of two 100-well plates.

Every constant and rule that the analysis script
``experiments/039-inhibitor-combinations-wetlab/scripts/plot_inhibitor_runs.py``
established is reused here verbatim (``EX21_BLOCK_START``, ``ISOBOLES``, ``AA_STEP``,
``GROWTH_RISE``, ``FIT_WINDOW_H``, :func:`read_raw`, :func:`generation_time`, the isobole
grid rule). Two departures, both sourced: the ex21 doses are computed from the dispensed
volumes (``inhibitors_titration_array.xlsx``) and the stock concentrations
(``inhibitors.xlsx`` C12:C17) instead of the notebook's axis labels, two of which are
typos (FF ``28`` for 18, FA ``1.25`` for 0.125); and the ex23 doses are the ones the
well map dispensed (FF 1.5, AA 2, HMF 2.522, FA 1, LVA 6, LA 20 g/L), not the
``C2`` column of ``inhibitors.xlsx`` (FF 6, AA 4, HMF 2.522, FA 1, LVA 2, LA 40) that the
039 script carries, which is the literature column the titrations started from.

The relative growth rate is the thesis's fitness (``process_bsc.py`` of the analysis
repo): the mean wild-type generation time of the run divided by the well's generation
time, so 1 is wild-type growth. A well whose curve never rose by ``GROWTH_RISE`` did
not grow; it has no generation time and is reported as such, never dropped.

Generation times come from one of two sources (:class:`TraitSource`): this module's
raw-curve derivation, the only one available for ex27 and ex28, or the Bioscreen
software's trait call that the thesis used for ex21, ex23 and ex26. Measured
2026-10-08: on ex26 the two agree on 98.5% of the 200 grew calls (Spearman 0.80 on the
45 both call grown); on ex23 they disagree in scale (mean WT generation time 1.345 h
raw vs 1.815 h software), so a loader picks one source per run and records it.
"""

from __future__ import annotations

import csv
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import openpyxl
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    build_manifest,
    sha256_file,
    write_manifest,
)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "volkPreliminaryExamReport2021"
REPORT_TITLE = "Machine Learning for Engineering Improved Yeast Fitness"
ARCHIVE_ROOT = Path("/bulk/thesis/thesis_archive")
ARCHIVE_MANIFEST = "MANIFEST.tsv"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"
#: ISO date the archive copies were deposited into the mirrors.
RETRIEVED_AT = "2026-10-08"

#: The submitted report, ``reports/`` of the archive. ``MANIFEST.tsv`` lists ``data/``
#: only, so the report's original location is not recorded there; its sha256 is pinned
#: here from the archive copy.
REPORT_ARCHIVE_PATH = "reports/Prelim_Report_2021_Michael_Volk.pdf"
REPORT_PDF_SHA256 = "4c2fcf11e16468b20fbbd589f651821bff5904884a7584ccae27ee2b70ec60b9"

_RAW = "data/01_bioscreenc_raw"
_RECORDS = "data/02_wet_lab_records"
_MK = "data/03_analysis_code/multi_knockout"
_IT = "data/03_analysis_code/inhibitor_tolerance"

#: Raw Bioscreen C exports (OD600 every 15 min, UTF-16) and their native settings files.
RAW_CSV: dict[str, str] = {
    "ex21": f"{_RAW}/MV_ex21_inhibitor_titration.csv",
    "ex23": f"{_RAW}/MV_ex23_Magic_inhibitor_combinations.csv",
    "ex26": f"{_RAW}/MV_ex26_inhibitor_isobole_FF_AA.csv",
    "ex27": f"{_RAW}/MV_ex27_inhibitor_isobole_FA_AA.csv",
    "ex28": f"{_RAW}/MV_ex28_inhibitor_isobole_HMF_AA.csv",
}
RAW_BSM: dict[str, str] = {
    run: path.removesuffix(".csv") + ".bsm" for run, path in RAW_CSV.items()
}
EX21_FINAL_VOLUMES = f"{_RAW}/MV_ex21_final_volumes.xlsx"
#: ex1's temperature log, the one export that labels the ``.bsm`` ``T`` record fields
#: (``Time Tray Cover``), and ex1's own ``.bsm`` for the field-by-field match.
EX1_TEMPERATURE_TXT = f"{_RAW}/ex1.txt"
EX1_BSM = f"{_RAW}/MV_ex1.bsm"
INHIBITORS_XLSX = f"{_RECORDS}/experiments/inhibitors.xlsx"
TITRATION_ARRAY_XLSX = f"{_RECORDS}/experiments/inhibitors_titration_array.xlsx"
ISOBOLE_SETUP_PPTX = f"{_RECORDS}/experiments/Isobole_BioscreenC_plate_setup.pptx"
ISOBOLE_SETUP_XLSX = f"{_RECORDS}/Isobole_BioscreenC_plate_setup.xlsx"
ISOBOLE_DESIGN_XLSX = (
    f"{_RECORDS}/experiments/Yeast_transformation/BioscreenC_FF_AA_Isobole_exp.xlsx"
)
EX23_PREPROCESSED = f"{_MK}/experiments/bsc/ex23/MV_ex23_preprocessed.csv"
EX23_WELL_MAP = f"{_MK}/inhibitor_tolerance/1_inhibitor-screen_2021-04-14_205953.xlsx"
EX23_CELL_VOLUME = (
    f"{_MK}/inhibitor_tolerance/1_inhibitor-screen_cell_volume_2021-04-16_204038.xlsx"
)
EX23_OD_CALCULATION = (
    f"{_MK}/inhibitor_tolerance/1_inhibitor-screen_OD_calculation_2021_04_14.xlsx"
)
EX23_DESIGN_NOTEBOOK = f"{_MK}/inhibitor_tolerance/1_inhibitor-screen.ipynb"
EX21_SOFTWARE_TRAITS = f"{_MK}/inhibitor_tolerance/MV_ex21_inhibitor_titration.tsv"
EX26_SOFTWARE_TRAITS = f"{_IT}/MV_ex26_inhibitor_isobole_FF_AA_Traits.txt"
ISOBOLE_WORKLIST_CSV = f"{_IT}/notebooks/Bioscreen_plate_Fluent_Worklist.csv"

#: Every archive file the raw mirror holds, in deposit order.
RAW_FILES: tuple[str, ...] = (
    *RAW_CSV.values(),
    *RAW_BSM.values(),
    EX21_FINAL_VOLUMES,
    EX1_TEMPERATURE_TXT,
    EX1_BSM,
    INHIBITORS_XLSX,
    TITRATION_ARRAY_XLSX,
    ISOBOLE_SETUP_PPTX,
    ISOBOLE_SETUP_XLSX,
    ISOBOLE_DESIGN_XLSX,
    EX23_PREPROCESSED,
    EX23_WELL_MAP,
    EX23_CELL_VOLUME,
    EX23_OD_CALCULATION,
    EX23_DESIGN_NOTEBOOK,
    EX21_SOFTWARE_TRAITS,
    EX26_SOFTWARE_TRAITS,
    ISOBOLE_WORKLIST_CSV,
)

# --------------------------------------------------------------------------- #
# Constants of the 039 analysis (verbatim) and the sourced doses
# --------------------------------------------------------------------------- #


class Inhibitor(StrEnum):
    """The six sorghum-hydrolysate inhibitors, by the archive's abbreviations."""

    FF = "FF"
    AA = "AA"
    HMF = "HMF"
    FA = "FA"
    LVA = "LVA"
    LA = "LA"


INHIBITORS: tuple[Inhibitor, ...] = tuple(Inhibitor)
INHIBITOR_NAMES: dict[Inhibitor, str] = {
    Inhibitor.FF: "furfural",
    Inhibitor.AA: "acetic acid",
    Inhibitor.HMF: "5-hydroxymethylfurfural",
    Inhibitor.FA: "formic acid",
    Inhibitor.LVA: "levulinic acid",
    Inhibitor.LA: "lactic acid",
}


class Run(StrEnum):
    """The Bioscreen C runs this module covers (ex22, the calibration, is excluded)."""

    ex21 = "ex21"
    ex23 = "ex23"
    ex26 = "ex26"
    ex27 = "ex27"
    ex28 = "ex28"


ISOBOLE_RUNS: tuple[Run, ...] = (Run.ex26, Run.ex27, Run.ex28)

#: ex21: the first well of each inhibitor's first replicate block (039, verbatim). A
#: block is ten wells: the uninhibited control, then the nine doses, highest first.
EX21_BLOCK_START: dict[Inhibitor, int] = {
    Inhibitor.FF: 1,
    Inhibitor.AA: 31,
    Inhibitor.HMF: 61,
    Inhibitor.FA: 101,
    Inhibitor.LVA: 131,
    Inhibitor.LA: 161,
}
EX21_REPLICATES = 3
EX21_STEPS = 9
#: ex21: the processing notebook's dose labels (g/L), verbatim. Two are typos the
#: dispensed volumes correct: FF "28" is 18 and FA "1.25" is 0.125.
EX21_LABELS: dict[Inhibitor, list[str]] = {
    Inhibitor.FF: ["30", "24", "28", "12", "6", "3", "1.5", "0.75", "0.375"],
    Inhibitor.AA: ["20", "16", "12", "8", "4", "2", "1", "0.5", "0.25"],
    Inhibitor.HMF: ["12.5", "10", "7.5", "5", "2.52", "1.26", "0.63", "0.32", "0.16"],
    Inhibitor.FA: ["5", "4", "3", "2", "1", "0.5", "0.25", "1.25", "0.063"],
    Inhibitor.LVA: ["10", "8", "6", "4", "2", "1", "0.5", "0.25", "0.125"],
    Inhibitor.LA: ["200", "160", "120", "80", "40", "20", "10", "5", "2.5"],
}
#: ex21: each titration tube was made up to 4 mL (inhibitors.xlsx rows 33-41:
#: ``=ROUND(B21/$C$12*4,5)`` mL of stock), so dose = uL of stock x stock g/L / 4000 uL.
EX21_TUBE_UL = 4000.0
#: ex23: every culture tube held 1700 uL (``ypd_v = 1700`` in 1_inhibitor-screen.ipynb;
#: the cell-volume table takes the 68 uL of inoculum out of the YPD), so dose = uL of
#: stock x stock g/L / 1700 uL.
EX23_TUBE_UL = 1700.0
#: ex23: the one dose per inhibitor the well map dispensed, g/L (FF 42.5 uL of 60 g/L,
#: AA 8.5 of 400, HMF 17 of 252.2, FA 17 of 100, LVA 51 of 200, LA 85 of 400, in 1700 uL).
EX23_G_PER_L: dict[Inhibitor, float] = {
    Inhibitor.FF: 1.5,
    Inhibitor.AA: 2.0,
    Inhibitor.HMF: 2.522,
    Inhibitor.FA: 1.0,
    Inhibitor.LVA: 6.0,
    Inhibitor.LA: 20.0,
}
#: The isoboles from their raw curves: run -> (first inhibitor, its well dose per 10 uL
#: step in g/L, raw export); the design sheet's "well conc" column (039, verbatim).
ISOBOLES: dict[Run, tuple[Inhibitor, float]] = {
    Run.ex26: (Inhibitor.FF, 0.3333),
    Run.ex27: (Inhibitor.FA, 0.2),
    Run.ex28: (Inhibitor.HMF, 0.504),
}
AA_STEP = 0.4
#: The isobole well volume: 90 uL of inhibitor at most per inhibitor, YPD to 180 uL,
#: plus 20 uL of inoculated YPD (the design sheet's ``total volume (uL)``).
ISOBOLE_WELL_UL = 200.0
#: A curve counts as growth when its baseline-subtracted OD rises by at least this much.
GROWTH_RISE = 0.3
FIT_WINDOW_H = 3.0
WILD_TYPE = "WT"
BLANK = "blank"
WELLS_PER_PLATE = 100

# --------------------------------------------------------------------------- #
# Archive manifest and the two mirrors
# --------------------------------------------------------------------------- #


class ArchiveFile(BaseModel):
    """One row of the thesis archive's ``MANIFEST.tsv``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    archive_path: str = Field(description="Path relative to the archive root.")
    bytes: int
    sha256: str
    source_path: str = Field(description="Where the archive copied the file from.")
    source_sha256_match: bool


def read_archive_manifest(archive_root: Path = ARCHIVE_ROOT) -> dict[str, ArchiveFile]:
    """``MANIFEST.tsv`` keyed by archive path (tab-separated, one header row)."""
    with (archive_root / ARCHIVE_MANIFEST).open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {
        row["archive_path"]: ArchiveFile(
            archive_path=row["archive_path"],
            bytes=int(row["bytes"]),
            sha256=row["sha256"],
            source_path=row["source_path"],
            source_sha256_match=row["source_sha256_match"] == "True",
        )
        for row in rows
    }


def raw_mirror_dir(data_root: str | Path) -> Path:
    """``$DATA_ROOT/torchcell-raw/volkPreliminaryExamReport2021``."""
    return Path(data_root) / RAW_DIR_REL


def library_dir(data_root: str | Path) -> Path:
    """``$DATA_ROOT/torchcell-library/volkPreliminaryExamReport2021``."""
    return Path(data_root) / LIBRARY_DIR_REL


def _archive_retrieval(path: Path, sha256: str, source: str | None) -> RetrievalRecord:
    return RetrievalRecord(
        method=RetrievalMethod.local_archive,
        source_url=source,
        retriever="torchcell.literature.retrieve.local_archive",
        params={"path": str(path), "sha256": sha256},
        sha256=sha256,
        retrieved_at=RETRIEVED_AT,
    )


def _copy_verified(src: Path, dest: Path, sha256: str) -> None:
    """Copy ``src`` to ``dest`` after checking its sha256; an existing ``dest`` must match.

    Idempotent: a mirror file already holding the pinned bytes is left alone, and one
    holding other bytes raises instead of being overwritten.
    """
    got = sha256_file(src)
    if got != sha256:
        raise RuntimeError(f"{src} sha256 mismatch: got {got}, expected {sha256}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha256_file(dest) != sha256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        return
    shutil.copy2(src, dest)


def deposit_raw_mirror(
    data_root: str | Path, *, archive_root: Path = ARCHIVE_ROOT
) -> Path:
    """Copy :data:`RAW_FILES` from the archive into the raw mirror and write its manifest.

    Each file is verified against the sha256 ``MANIFEST.tsv`` lists for it and stored at
    the same archive-relative path (``data/...``) under :func:`raw_mirror_dir`. Its
    record names the original path from ``MANIFEST.tsv`` as ``source_url`` and the
    ``local_archive`` retriever with the archive copy's path, so a rebuild re-runs this
    function and every byte is re-checked.
    """
    listed = read_archive_manifest(archive_root)
    root = raw_mirror_dir(data_root)
    files: list[ArtifactRecord] = []
    for rel in RAW_FILES:
        entry = listed[rel]
        src = archive_root / rel
        dest = root / rel
        _copy_verified(src, dest, entry.sha256)
        files.append(
            ArtifactRecord(
                path=rel,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=entry.sha256,
                source=str(src),
                retrieval=_archive_retrieval(src, entry.sha256, entry.source_path),
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=None,
        title=REPORT_TITLE,
        files=files,
        si_data_sources=[str(archive_root / ARCHIVE_MANIFEST)],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def deposit_report(
    data_root: str | Path,
    *,
    archive_root: Path = ARCHIVE_ROOT,
    ocr: bool = False,
    device_mode: str = "cpu",
) -> Path:
    """Write the library key: the report as ``paper.pdf``, its OCR, and ``manifest.json``.

    The PDF is verified against :data:`REPORT_PDF_SHA256`. ``ocr=True`` runs MinerU on
    it (``torchcell.literature.ocr.ocr_pdf``), which writes ``paper.md`` and its
    processing record; with ``ocr=False`` an existing ``paper.md`` is kept. The key has
    no Zotero item: ``zotero_item_key`` and ``doi`` stay None, and the PDF's record
    carries a ``local_archive`` retrieval instead of a Zotero attachment.
    """
    key_dir = library_dir(data_root)
    src = archive_root / REPORT_ARCHIVE_PATH
    _copy_verified(src, key_dir / "paper.pdf", REPORT_PDF_SHA256)
    if ocr:
        from torchcell.literature.ocr import ocr_pdf

        ocr_pdf(key_dir / "paper.pdf", device_mode=device_mode)
    manifest = build_manifest(
        key_dir,
        citation_key=CITATION_KEY,
        doi=None,
        title=REPORT_TITLE,
        zotero_item_key=None,
        sources={"paper.pdf": str(src)},
    )
    files = [
        record.model_copy(
            update={
                "retrieval": _archive_retrieval(
                    src,
                    REPORT_PDF_SHA256,
                    f"{archive_root}/{REPORT_ARCHIVE_PATH} (reports/ is not listed "
                    "in the archive's MANIFEST.tsv, which covers data/ only)",
                )
            }
        )
        if record.path == "paper.pdf"
        else record
        for record in manifest.files
    ]
    write_manifest(key_dir, manifest.model_copy(update={"files": files}))
    return key_dir


# --------------------------------------------------------------------------- #
# Raw curves and the generation time (039, verbatim but for None on no growth)
# --------------------------------------------------------------------------- #
FloatArray = npt.NDArray[np.float64]


def read_raw(path: str | Path) -> pd.DataFrame:
    """A Bioscreen C export as hours x wells 1..200, OD600 as exported (UTF-16)."""
    raw = pd.read_csv(path, skiprows=2, index_col=0, encoding="UTF-16LE")
    hours = [
        int(h) + int(m) / 60 + int(sec) / 3600
        for h, m, sec in (str(t).split(":") for t in raw.index)
    ]
    out = raw.drop(columns=["Blank"]).astype(float)
    out.columns = out.columns.astype(int)
    out.index = pd.Index(hours, name="hours")
    return out


def generation_time(hours: FloatArray, od: FloatArray) -> float | None:
    """Hours per doubling on the steepest ``FIT_WINDOW_H`` stretch; None without growth.

    The curve is baseline-subtracted by the median of its first three readings; a well
    whose subtracted OD never rises by ``GROWTH_RISE`` did not grow. The rate is the
    largest slope of log2(OD) over any window of ``FIT_WINDOW_H`` whose readings are all
    at least 0.05 above baseline, by least squares.
    """
    y = od - np.median(od[:3])
    if y.max() < GROWTH_RISE:
        return None
    ok = y > 0.05
    log_y = np.where(ok, np.log2(np.where(ok, y, 1.0)), np.nan)
    best = 0.0
    n = len(hours)
    j = 0
    for i in range(n):
        while j < n and hours[j] - hours[i] < FIT_WINDOW_H:
            j += 1
        if j - i < 4 or j > n:
            continue
        window = slice(i, j)
        if not ok[window].all():
            continue
        slope = float(np.polyfit(hours[window], log_y[window], 1)[0])
        best = max(best, slope)
    return 1.0 / best if best > 0 else None


# --------------------------------------------------------------------------- #
# Plate layouts
# --------------------------------------------------------------------------- #


def plate_of(well: int) -> int:
    """Bioscreen C plate of a well: 1 for wells 1-100, 2 for 101-200."""
    if not 1 <= well <= 2 * WELLS_PER_PLATE:
        raise ValueError(f"well {well} is not on a two-plate Bioscreen C tray")
    return 1 if well <= WELLS_PER_PLATE else 2


def no_doses() -> dict[Inhibitor, float]:
    """Every inhibitor at 0 g/L."""
    return {inhibitor: 0.0 for inhibitor in INHIBITORS}


class WellLayout(BaseModel):
    """One inoculated well of a run as designed: its condition and its doses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run: Run
    plate: int = Field(ge=1, le=2)
    well: int = Field(ge=1, le=200)
    condition: str = Field(
        description="WT for an uninhibited well, else the inhibitors and doses."
    )
    doses_g_per_l: dict[Inhibitor, float] = Field(
        description="Every inhibitor's dose in the culture, g/L (0 when absent)."
    )
    biological_replicate_id: int = Field(ge=1)

    @model_validator(mode="after")
    def _check(self) -> WellLayout:
        if set(self.doses_g_per_l) != set(INHIBITORS):
            raise ValueError("doses_g_per_l must name every inhibitor exactly once")
        if any(dose < 0 for dose in self.doses_g_per_l.values()):
            raise ValueError("a dose cannot be negative")
        if plate_of(self.well) != self.plate:
            raise ValueError(f"well {self.well} is not on plate {self.plate}")
        uninhibited = all(dose == 0 for dose in self.doses_g_per_l.values())
        if uninhibited != (self.condition == WILD_TYPE):
            raise ValueError(
                f"condition {self.condition!r} disagrees with its doses: only the "
                "uninhibited well is named WT"
            )
        return self

    @property
    def is_wild_type(self) -> bool:
        """An uninhibited well (strain bAID in plain YPD)."""
        return self.condition == WILD_TYPE

    def present(self) -> list[Inhibitor]:
        """The inhibitors at a non-zero dose, in :data:`INHIBITORS` order."""
        return [i for i in INHIBITORS if self.doses_g_per_l[i] > 0]


class PlateLayout(BaseModel):
    """All inoculated wells of one run; blanks and unused wells are not listed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run: Run
    wells: list[WellLayout]

    @model_validator(mode="after")
    def _check(self) -> PlateLayout:
        numbers = [w.well for w in self.wells]
        if len(set(numbers)) != len(numbers):
            raise ValueError(f"{self.run}: a well is listed twice")
        if any(w.run != self.run for w in self.wells):
            raise ValueError(f"{self.run}: a well of another run is listed")
        return self

    def wild_type(self) -> list[WellLayout]:
        """The uninhibited wells."""
        return [w for w in self.wells if w.is_wild_type]


class TraitSource(StrEnum):
    """Where a well's generation time came from.

    ``raw_curve`` is :func:`generation_time` on the exported OD600, available for every
    run. ``bioscreen_software`` is the Bioscreen/PRECOG trait call the thesis used,
    which exists only for ex21, ex23 and ex26; the report says PRECOG worked from
    "estimated cell counts from calibration curves", so the two need not agree, and on
    ex23 they do not (see the module's note).
    """

    raw_curve = "raw_curve"
    bioscreen_software = "bioscreen_software"


class Well(WellLayout):
    """One well's derived traits: its generation time, if it grew, and its source."""

    generation_time_h: float | None = Field(
        description="Hours per doubling; None if the culture did not grow."
    )
    grew: bool
    trait_source: TraitSource

    @model_validator(mode="after")
    def _grew_iff_generation_time(self) -> Well:
        if self.grew != (self.generation_time_h is not None):
            raise ValueError("grew must be True exactly when a generation time exists")
        return self


def _read_cells(path: Path, sheet: str | None, cells: Sequence[str]) -> dict[str, Any]:
    """Cached values of the named cells (formulas read as Excel last computed them)."""
    book = openpyxl.load_workbook(path, data_only=True, read_only=False)
    ws = book[sheet] if sheet is not None else book.active
    if ws is None:
        raise ValueError(f"{path} has no active sheet")
    return {cell: ws[cell].value for cell in cells}


#: inhibitors.xlsx: the column of each inhibitor in the titration array tables, and
#: the row of its stock in "Finding effective concentration" (C12:C17).
_ARRAY_COLUMN: dict[Inhibitor, str] = dict(zip(INHIBITORS, "BCDEFG", strict=True))
_STOCK_ROW: dict[Inhibitor, int] = dict(zip(INHIBITORS, range(12, 18), strict=True))


def stock_g_per_l(data_dir: Path) -> dict[Inhibitor, float]:
    """Each inhibitor's stock (``inhibitors.xlsx`` C12:C17, "concentrated solution (g/L)").

    The sheet's own formula: 10 x the C2 target for FF and LA (``=B12*10``), 100 x for
    the others; the abbreviation in column A is checked against the row.
    """
    path = data_dir / INHIBITORS_XLSX
    cells = [f"{col}{row}" for row in _STOCK_ROW.values() for col in "AC"]
    values = _read_cells(path, None, cells)
    out: dict[Inhibitor, float] = {}
    for inhibitor, row in _STOCK_ROW.items():
        if values[f"A{row}"] != inhibitor.value:
            raise ValueError(f"{path} A{row} is {values[f'A{row}']!r}, not {inhibitor}")
        out[inhibitor] = float(values[f"C{row}"])
    return out


def titration_volumes_ul(data_dir: Path) -> dict[Inhibitor, list[float]]:
    """``inhibitors_titration_array.xlsx``: uL of stock per 4 mL tube, steps 1-9.

    Row 2 is the uninhibited tube (``WT``); rows 3-11 are the nine doses, highest first.
    """
    path = data_dir / TITRATION_ARRAY_XLSX
    cells = [f"{col}{row}" for col in "ABCDEFG" for row in range(1, 12)]
    values = _read_cells(path, None, cells)
    out: dict[Inhibitor, list[float]] = {}
    for inhibitor, col in _ARRAY_COLUMN.items():
        if values[f"{col}1"] != f"{inhibitor.value} (uL)":
            raise ValueError(f"{path} {col}1 is {values[f'{col}1']!r}")
        if values[f"{col}2"] != WILD_TYPE:
            raise ValueError(f"{path} {col}2 is not the WT tube")
        out[inhibitor] = [float(values[f"{col}{row}"]) for row in range(3, 12)]
    return out


def ex21_doses(
    volumes_ul: dict[Inhibitor, list[float]], stocks: dict[Inhibitor, float]
) -> dict[Inhibitor, list[float]]:
    """Dose of each titration step, g/L: uL of stock x stock / 4000 uL, to 4 decimals."""
    return {
        inhibitor: [round(v * stocks[inhibitor] / EX21_TUBE_UL, 4) for v in volumes]
        for inhibitor, volumes in volumes_ul.items()
    }


def _dose_name(doses: dict[Inhibitor, float]) -> str:
    present = [f"{i.value}{doses[i]:g}" for i in INHIBITORS if doses[i] > 0]
    return "_".join(present) if present else WILD_TYPE


def ex21_layout_from(doses: dict[Inhibitor, list[float]]) -> PlateLayout:
    """The ex21 wells: per inhibitor and replicate, a WT well then the nine doses.

    Blocks start at :data:`EX21_BLOCK_START` + 10 x replicate. Wells 91-100 and
    191-200 belong to no block and are not listed.
    """
    wells: list[WellLayout] = []
    for inhibitor, start in EX21_BLOCK_START.items():
        if len(doses[inhibitor]) != EX21_STEPS:
            raise ValueError(f"{inhibitor}: expected {EX21_STEPS} doses")
        for replicate in range(EX21_REPLICATES):
            block = start + 10 * replicate
            wells.append(
                WellLayout(
                    run=Run.ex21,
                    plate=plate_of(block),
                    well=block,
                    condition=WILD_TYPE,
                    doses_g_per_l=no_doses(),
                    biological_replicate_id=replicate + 1,
                )
            )
            for step, dose in enumerate(doses[inhibitor], start=1):
                d = no_doses() | {inhibitor: dose}
                wells.append(
                    WellLayout(
                        run=Run.ex21,
                        plate=plate_of(block + step),
                        well=block + step,
                        condition=_dose_name(d),
                        doses_g_per_l=d,
                        biological_replicate_id=replicate + 1,
                    )
                )
    return PlateLayout(run=Run.ex21, wells=sorted(wells, key=lambda w: w.well))


def ex21_layout(data_dir: Path) -> PlateLayout:
    """ex21 from the titration array and the stocks under ``data_dir``."""
    return ex21_layout_from(
        ex21_doses(titration_volumes_ul(data_dir), stock_g_per_l(data_dir))
    )


def ex23_doses(name: str) -> dict[Inhibitor, float]:
    """``FF_AA_HMF`` -> those inhibitors at :data:`EX23_G_PER_L`, the rest at 0."""
    members = [Inhibitor(m) for m in name.split("_")]
    if len(set(members)) != len(members):
        raise ValueError(f"{name}: an inhibitor is named twice")
    return no_doses() | {m: EX23_G_PER_L[m] for m in members}


class Ex23Row(BaseModel):
    """One row of ``MV_ex23_preprocessed.csv``: the well-to-condition map."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    well: int
    biological_replicate_id: int


def read_ex23_map(data_dir: Path) -> list[Ex23Row]:
    """``name``, ``well`` and ``biological_replicate_id`` of every ex23 well."""
    frame = pd.read_csv(data_dir / EX23_PREPROCESSED)
    return [
        Ex23Row(name=str(name), well=int(well), biological_replicate_id=int(replicate))
        for name, well, replicate in zip(
            frame["name"], frame["well"], frame["biological_replicate_id"], strict=True
        )
    ]


def ex23_layout_from(rows: Sequence[Ex23Row]) -> PlateLayout:
    """The ex23 wells; ``blank`` (uninoculated YPD) wells are excluded, ``WT`` kept."""
    wells = [
        WellLayout(
            run=Run.ex23,
            plate=plate_of(row.well),
            well=row.well,
            condition=WILD_TYPE if row.name == WILD_TYPE else row.name,
            doses_g_per_l=no_doses() if row.name == WILD_TYPE else ex23_doses(row.name),
            biological_replicate_id=row.biological_replicate_id,
        )
        for row in rows
        if row.name != BLANK
    ]
    return PlateLayout(run=Run.ex23, wells=sorted(wells, key=lambda w: w.well))


def ex23_layout(data_dir: Path) -> PlateLayout:
    """ex23 from ``MV_ex23_preprocessed.csv`` under ``data_dir``."""
    return ex23_layout_from(read_ex23_map(data_dir))


def isobole_layout(run: Run) -> PlateLayout:
    """The 039 grid rule: well 1..100 of each plate is ``divmod(well - 1, 10)``.

    The slow digit (column) is the acetic-acid step of :data:`AA_STEP`, the fast digit
    (row) the other inhibitor's step from :data:`ISOBOLES`; doses rounded to 3 decimals
    as 039 rounds them. Each plate's well 1 is the uninhibited well. The plate number is
    the replicate id.
    """
    inhibitor, step = ISOBOLES[run]
    wells: list[WellLayout] = []
    for plate, offset in ((1, 0), (2, WELLS_PER_PLATE)):
        for well in range(1, WELLS_PER_PLATE + 1):
            col, row = divmod(well - 1, 10)
            d = no_doses() | {
                inhibitor: round(step * row, 3),
                Inhibitor.AA: round(AA_STEP * col, 3),
            }
            wells.append(
                WellLayout(
                    run=run,
                    plate=plate,
                    well=offset + well,
                    condition=_dose_name(d),
                    doses_g_per_l=d,
                    biological_replicate_id=plate,
                )
            )
    return PlateLayout(run=run, wells=wells)


def layout(run: Run, data_dir: Path) -> PlateLayout:
    """The plate layout of any covered run."""
    if run == Run.ex21:
        return ex21_layout(data_dir)
    if run == Run.ex23:
        return ex23_layout(data_dir)
    return isobole_layout(run)


# --------------------------------------------------------------------------- #
# Traits
# --------------------------------------------------------------------------- #


def raw_curve_generation_times(run: Run, data_dir: Path) -> dict[int, float | None]:
    """:func:`generation_time` of every well of the run's raw export."""
    raw = read_raw(data_dir / RAW_CSV[run.value])
    hours = raw.index.to_numpy(dtype=np.float64)
    return {
        int(well): generation_time(hours, raw[well].to_numpy(dtype=np.float64))
        for well in raw.columns
    }


def software_generation_times(run: Run, data_dir: Path) -> dict[int, float | None]:
    """The Bioscreen software's ``GT`` per well, for the three runs it processed.

    ex21 from ``MV_ex21_inhibitor_titration.tsv``, ex23 from the ``gt`` column of
    ``MV_ex23_preprocessed.csv`` (the trait file merged onto the well map), ex26 from
    ``MV_ex26_..._Traits.txt``. ex27 and ex28 were never processed and raise.
    """
    if run == Run.ex21:
        return read_software_traits(data_dir / EX21_SOFTWARE_TRAITS)
    if run == Run.ex26:
        return read_software_traits(data_dir / EX26_SOFTWARE_TRAITS)
    if run == Run.ex23:
        frame = pd.read_csv(data_dir / EX23_PREPROCESSED)
        return {
            int(well): None if np.isnan(gt) else float(gt)
            for well, gt in zip(frame["well"], frame["gt"].astype(float), strict=True)
        }
    raise ValueError(f"{run}: the Bioscreen software never processed this run")


def derive_wells(
    run: Run, data_dir: Path, source: TraitSource = TraitSource.raw_curve
) -> list[Well]:
    """Every inoculated well of ``run`` with its generation time from ``source``."""
    plate = layout(run, data_dir)
    gts = (
        raw_curve_generation_times(run, data_dir)
        if source == TraitSource.raw_curve
        else software_generation_times(run, data_dir)
    )
    return [
        Well(
            **designed.model_dump(),
            generation_time_h=gts[designed.well],
            grew=gts[designed.well] is not None,
            trait_source=source,
        )
        for designed in plate.wells
    ]


def wild_type_generation_time(wells: Sequence[Well]) -> float:
    """Mean generation time over the run's uninhibited wells that grew.

    ``wells`` must all belong to one run; a run with no grown WT well raises, since
    every relative growth rate of that run would be undefined.
    """
    runs = {w.run for w in wells}
    if len(runs) != 1:
        raise ValueError(f"wells span {len(runs)} runs; pass one run's wells")
    if len({w.trait_source for w in wells}) != 1:
        raise ValueError("wells mix trait sources; pass one source's wells")
    gts = [
        w.generation_time_h
        for w in wells
        if w.is_wild_type and w.generation_time_h is not None
    ]
    if not gts:
        raise ValueError(f"{runs.pop()}: no uninhibited well grew")
    return float(np.mean(gts))


def relative_growth_rate(well: Well, wt_generation_time_h: float) -> float | None:
    """Wild-type generation time / the well's; None for a well that did not grow."""
    if well.generation_time_h is None:
        return None
    return wt_generation_time_h / well.generation_time_h


# --------------------------------------------------------------------------- #
# The Bioscreen software's trait files and the run's .bsm log
# --------------------------------------------------------------------------- #


def read_software_traits(path: str | Path) -> dict[int, float | None]:
    """A Bioscreen ``*_Traits.txt`` / ``*.tsv``: well -> ``GT``, None where it is NaN."""
    frame = pd.read_csv(path, sep="\t")
    wells = frame["Container Name"].str.split(" ", expand=True)[1].astype(int)
    out: dict[int, float | None] = {}
    for well, gt in zip(wells, frame["GT"].astype(float), strict=True):
        out[int(well)] = None if np.isnan(gt) else float(gt)
    return out


class TemperatureLog(BaseModel):
    """The ``T`` records of a ``.bsm``: one reading per minute of tray and cover.

    The field order (Excel serial time, tray, cover) is anchored by ex1, whose exported
    ``ex1.txt`` labels the columns ``Time Tray Cover`` and whose ``.bsm`` ``T`` records
    round to the same values row for row.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    time_serial: list[float]
    tray_c: list[float]
    cover_c: list[float]


def read_bsm_records(path: str | Path) -> list[list[str]]:
    """Every line of a ``.bsm`` split on commas (the file is CRLF text)."""
    text = Path(path).read_text(encoding="latin-1")
    return [line.split(",") for line in text.splitlines() if line]


def read_temperature_log(path: str | Path) -> TemperatureLog:
    """The run's tray and cover temperatures from its ``.bsm`` ``T`` records."""
    rows = [r for r in read_bsm_records(path) if r[0] == "T"]
    return TemperatureLog(
        time_serial=[float(r[1]) for r in rows],
        tray_c=[float(r[2]) for r in rows],
        cover_c=[float(r[3]) for r in rows],
    )


class TemperatureSummary(BaseModel):
    """Tray temperature over a run, after the instrument reached its set point."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run: Run
    n_readings: int
    tray_median_c: float
    tray_p01_c: float
    tray_p99_c: float
    warmup_readings_dropped: int


def summarize_tray_temperature(run: Run, log: TemperatureLog) -> TemperatureSummary:
    """Median and 1st/99th percentiles of the tray, from the first reading at >= 29.5 C.

    The leading readings below 29.5 C are the warm-up from room temperature (ex1's log
    climbs 27.1 -> 30.0 C over its first five minutes); they are dropped and counted.
    """
    tray = np.asarray(log.tray_c, dtype=np.float64)
    reached = np.flatnonzero(tray >= 29.5)
    if reached.size == 0:
        raise ValueError(f"{run}: the tray never reached 29.5 C")
    steady = tray[int(reached[0]) :]
    return TemperatureSummary(
        run=run,
        n_readings=int(steady.size),
        tray_median_c=float(np.median(steady)),
        tray_p01_c=float(np.percentile(steady, 1)),
        tray_p99_c=float(np.percentile(steady, 99)),
        warmup_readings_dropped=int(reached[0]),
    )


def read_run_settings(path: str | Path) -> dict[str, str]:
    """The ``S`` (settings) records of a ``.bsm`` as key -> value, last value wins.

    The keys are the instrument's own codes; no mirrored source documents them, so
    they are reported raw (``S,4`` = 900 matches the 15 min spacing of the export, and
    ``S,Y`` the elapsed days; the others are not decoded).
    """
    return {r[1]: ",".join(r[2:]) for r in read_bsm_records(path) if r[0] == "S"}


def read_run_events(path: str | Path) -> list[str]:
    """The ``E`` (event) records of a ``.bsm``: start, abort and completion lines."""
    return [",".join(r[1:]) for r in read_bsm_records(path) if r[0] == "E"]


def run_duration_h(data_dir: Path, run: Run) -> float:
    """Hours from the first to the last reading of the run's export."""
    hours = read_raw(data_dir / RAW_CSV[run.value]).index.to_numpy(dtype=np.float64)
    return float(hours[-1] - hours[0])
