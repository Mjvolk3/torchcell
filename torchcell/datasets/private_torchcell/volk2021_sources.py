# torchcell/datasets/private_torchcell/volk2021_sources.py
# [[torchcell.datasets.private_torchcell.volk2021_sources]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/private_torchcell/volk2021_sources.py
# Test file: tests/torchcell/datasets/private_torchcell/test_volk2021_sources.py
"""Sourced constants and typed gaps for the 2021 Bioscreen C inhibitor runs.

Three kinds of source, one :class:`~torchcell.verification.report.Provenance` per file:

* the OCR of the 2021 preliminary exam report (library key
  ``volkPreliminaryExamReport2021``, ``paper.md``), bound with :func:`_report` to a
  verbatim substring, exactly as ``vanacloig2022._paper`` binds a paper;
* the Lian 2019 MAGIC paper and its Supplementary Information, for the identity and
  genotype of the strain (``bAID``);
* the thesis-archive files copied into the raw mirror
  (``$DATA_ROOT/torchcell-raw/volkPreliminaryExamReport2021/<archive path>``), bound with
  :func:`_archive`. A text file (notebook, CSV, ``.bsm``, trait table) is quoted
  verbatim (``ex1.txt`` is UTF-16, quoted from its decoded text); a slide deck is
  quoted as its text runs joined by ``|``. A spreadsheet has no text to quote, so its quote is :func:`cell_quote` of
  the cited cells: each cell's stored content and, for a formula, the value Excel last
  computed (``Sheet1!C12==B12*10 -> 60``); the test suite re-reads those cells.

What no source states is a :class:`~torchcell.verification.sourced.ProvenanceGap` here,
for the loader to attach, never a guessed value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openpyxl

from torchcell.datasets.private_torchcell.bioscreen import (
    CITATION_KEY,
    EX1_TEMPERATURE_TXT,
    EX21_FINAL_VOLUMES,
    EX23_DESIGN_NOTEBOOK,
    EX23_OD_CALCULATION,
    EX23_WELL_MAP,
    INHIBITORS_XLSX,
    ISOBOLE_DESIGN_XLSX,
    ISOBOLE_SETUP_PPTX,
    RAW_BSM,
    TITRATION_ARRAY_XLSX,
    Inhibitor,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

# --------------------------------------------------------------------------- #
# Source anchors
# --------------------------------------------------------------------------- #
REPORT_MD = "paper.md"
REPORT_MD_SHA256 = "61f2b85096f5dce77de8e97061eb44daa1e42e3663c14cc0b8d8f09d4563cf33"
REPORT_METHOD = (
    "MinerU 2.7.6 OCR (cpu, 200 dpi) of the submitted report PDF "
    "(torchcell-library mirror)"
)

LIAN_KEY = "lianMultifunctionalGenomewideCRISPR2019"
LIAN_SI_MD = "si/si1.md"
LIAN_SI_SHA256 = "b2bcfe2e672674438216472e3e06903c93d4ee54cd8b6fd9b5f964ad2a3d32db"
LIAN_PAPER_MD = "paper.md"
LIAN_PAPER_SHA256 = "63fe2b7101fc48feb297f9e34b83d108b74f03f28bbc280e08c7219bc975086c"
LIAN_METHOD = "MinerU OCR of the publisher PDF (torchcell-library mirror)"

ARCHIVE_METHOD = (
    "thesis-archive copy in the raw mirror "
    "(torchcell-raw/volkPreliminaryExamReport2021), sha256 from the archive MANIFEST.tsv"
)

#: archive path -> sha256 listed for it in the archive's MANIFEST.tsv.
ARCHIVE_SHA256: dict[str, str] = {
    INHIBITORS_XLSX: "df3a32b98fb2fdbbefdc212517fe34953a51216f16990745e624126515f6b3c5",
    TITRATION_ARRAY_XLSX: (
        "13847d3b252807686c4713bc9b2f70bd6df4cf33e241291a1f37e1812260a530"
    ),
    EX23_WELL_MAP: "07c35fdc4a402a86f6712528e0624425a4de9eb278504a16d8fe08a803aacd59",
    EX23_OD_CALCULATION: (
        "a6ee6fecebfcee61bafd8e39076f0c65754e87f1dd7915ead615fd484130d3cf"
    ),
    EX23_DESIGN_NOTEBOOK: (
        "8940ca5d8f745f6a720ec707b656082d73c2333d93a13b9f58e6f9a6f1fc9182"
    ),
    ISOBOLE_DESIGN_XLSX: (
        "9e162703ef320a28998594759e57abc54a9a4569fff24d00449a402a1a3bc426"
    ),
    ISOBOLE_SETUP_PPTX: (
        "5fffc0c9615e8added846de854a55713beb9331518ada14978b65ee840543a1a"
    ),
    EX21_FINAL_VOLUMES: (
        "77fb170c06ea08717b0538e91a2b1b7ef0d2d409292d14caa47ba40e8390617b"
    ),
    EX1_TEMPERATURE_TXT: (
        "835cc7ae545445e2b725d01f499e2665b6b5936551975f66ff4a340368604691"
    ),
    RAW_BSM["ex21"]: "0afb8d68a542e771a38201063c058d78fedfc7fadf26fbdfba5659a8364c3fa4",
    RAW_BSM["ex23"]: "ff998c2b73ff129ab994fa83c6aac4a9b7472aaee3fae398e0ea30de3ebf95f5",
    RAW_BSM["ex26"]: "8505b0deb9dc87a0a2819221f13f318ab8905da583bb3d324542eadb186344de",
    RAW_BSM["ex27"]: "d7229771aeb7d1b93bff75c71494a6e44e39aebf07448442fea7e8404b6de812",
    RAW_BSM["ex28"]: "2f0862d18c9eed6a67fa0bd38000ff9caa73d260b73314231e8d97ea505128c7",
}


def archive_provenance(archive_path: str, page: str | None = None) -> Provenance:
    """The one :class:`Provenance` of an archive file in the raw mirror."""
    return Provenance(
        source_uri=archive_path,
        citation_key=CITATION_KEY,
        sha256=ARCHIVE_SHA256[archive_path],
        method=ARCHIVE_METHOD,
        page=page,
    )


REPORT = Provenance(
    source_uri=REPORT_MD,
    citation_key=CITATION_KEY,
    sha256=REPORT_MD_SHA256,
    method=REPORT_METHOD,
)
LIAN_SI = Provenance(
    source_uri=LIAN_SI_MD,
    citation_key=LIAN_KEY,
    sha256=LIAN_SI_SHA256,
    method=LIAN_METHOD,
    page="Supplementary Table 9 (strains)",
)


def _report(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote in the sha256-pinned report OCR."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=REPORT.model_copy(update={"page": page}),
    )


def _lian(
    value: Any, quote: str, *, md: str, sha256: str, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to Lian 2019, which built and named the strain."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=md,
            citation_key=LIAN_KEY,
            sha256=sha256,
            method=LIAN_METHOD,
            page=page,
        ),
    )


def _archive(
    value: Any, archive_path: str, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to an archive file in the raw mirror (verbatim text or cell quote)."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=archive_provenance(archive_path, page),
    )


def _render(value: Any) -> str:
    return (
        ""
        if value is None
        else f"{value:g}"
        if isinstance(value, float)
        else str(value)
    )


def cell_quote(path: Path, cells: list[str], sheet: str | None = None) -> str:
    """The cited cells as ``Sheet!A1=content`` joined by ``; ``.

    ``content`` is what the cell stores (a number, text, or a formula beginning with
    ``=``); a formula is followed by `` -> `` and the value Excel last computed for it.
    """
    stored_book = openpyxl.load_workbook(path, data_only=False)
    cached_book = openpyxl.load_workbook(path, data_only=True)
    stored = stored_book[sheet] if sheet is not None else stored_book.active
    cached = cached_book[sheet] if sheet is not None else cached_book.active
    if stored is None or cached is None:
        raise ValueError(f"{path} has no active sheet")
    parts = []
    for cell in cells:
        content = stored[cell].value
        text = f"{stored.title}!{cell}={_render(content)}"
        if isinstance(content, str) and content.startswith("="):
            text += f" -> {_render(cached[cell].value)}"
        parts.append(text)
    return "; ".join(parts)


# --------------------------------------------------------------------------- #
# Strain identity
# --------------------------------------------------------------------------- #
STRAIN_NAME = _report(
    "BY4742-iAID6",
    "MAGIC strain BY4742-iAID6 was grown over a gradient of inhibition concentrations "
    "to determine minimum concentrations where doubling time is observed to increase",
    page="2.2.1 Bioenergy Sorghum Inhibitor Screen on BY4742-iAID6",
    note="the report's name for the strain of every inhibitor run; the MAGIC strain of "
    "Lian 2019 (BAID_CONSTRUCTION, BAID_GENOTYPE)",
)
BAID_CONSTRUCTION = _lian(
    "bAID = BY4742 + integrated pAID6",
    "The CRISPR-AID strain (bAID) was constructed by integrating PmeI-digested "
    "$\\mathrm { \\ p A I D } 6 ^ { 8 }$ into the genome of BY4742 and selection for "
    "G418 resistance.",
    md=LIAN_PAPER_MD,
    sha256=LIAN_PAPER_SHA256,
    page="Methods, 'Plasmid and strain construction'",
    note="'BY4742-iAID6' reads as BY4742 with pAID6 integrated, which is how Lian "
    "constructs bAID; the archive's ex23 well table also carries strain "
    "'BY4742-iAID6' on every row",
)
BAID_GENOTYPE = _lian(
    "BY4742-Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]",
    "<td rowspan=1 colspan=1>bAID</td><td rowspan=1 colspan=1>"
    "BY4742-Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]</td>",
    md=LIAN_SI_MD,
    sha256=LIAN_SI_SHA256,
    page="Supplementary Table 9 (strains)",
)

# --------------------------------------------------------------------------- #
# What the report states
# --------------------------------------------------------------------------- #
DOSE_SELECTION_RULE = _report(
    "first dose with an increased doubling time and low doubling-time variance",
    "Concentrations for studying all 63 inhibitor combinations were selected by "
    "choosing the first concentration that showed an increase in doubling time while "
    "also showing low doubling time variance.",
    page="2.2.1, paragraph after Fig 11",
)
FITNESS_DEFINITION = _report(
    "wild-type doubling time / strain doubling time",
    "we elected to measure fitness with information rich liquid growth assays by "
    "computing fitness as the ratio of wild type doubling time to mutant doubling "
    "time (Fig 6)",
    page="1.2.3 Liquid Culture Fitness and Literature Comparisons",
    note="stated for the mutant screens; 2.2.1 applies it to the inhibitor media "
    "(EX23_FITNESS_AS_PROJECT_1), with the uninhibited strain as the wild type",
)
EX23_FITNESS_AS_PROJECT_1 = _report(
    3,
    "Biological triplicates in all media combinations were screened and a fitness "
    "value was computed as described in project 1.",
    page="2.2.1, paragraph after Fig 11",
    note="the value is the number of biological replicates of each ex23 combination",
)
EX21_REPLICATES = _report(
    3,
    "biological triplicates of BY4742- iAID6.",
    page="Fig 11 caption, panel (a)",
    note="ex21, the single-inhibitor titrations of Fig 11a",
)
FIG_11B_CAPTION = _report(
    "Fig 11b: 63 inhibitor combinations, one dose per inhibitor (dose table in figure)",
    "Fitness scores of 63 inhibitor combinations with only a subset of data displayed.",
    page="Fig 11 caption, panel (b)",
    note="the dose table sits inside the figure image, which MinerU did not capture "
    "(the page's Fig 11b region came out as an empty text block), so the doses are "
    "sourced from the ex23 well map and the stock table (EX23_VOLUMES_UL, STOCKS)",
)
INSTRUMENT = _report(
    "Bioscreen C, 200 wells per run, optical density over time",
    "We use the BioscreenC which can process 200 wells per batch to measure optical "
    "density over time to generate growth profiles.",
    page="1.2.3",
)
SOFTWARE_TRAITS = _report(
    "PRECOG doubling times from calibrated cell counts",
    "Growth curves are then processed with PRECOG software using estimated cell counts "
    "from calibration curves to compute doubling times26.",
    page="1.2.3",
    note="how the thesis's own generation times (TraitSource.bioscreen_software) "
    "were made; the raw-curve derivation uses OD600 directly",
)
WELLS_RANDOMIZED = _report(
    True,
    "Well locations are randomized, and wild type mutants are used as controls across "
    "experiments.",
    page="1.2.3",
    note="ex23's well map is a random permutation (1_inhibitor-screen.ipynb, "
    "np.random.shuffle); ex21 and the isoboles are laid out in blocks, not randomized",
)

# --------------------------------------------------------------------------- #
# What the archive records state
# --------------------------------------------------------------------------- #
_STOCK_CELLS = {
    Inhibitor.FF: ["A12", "B12", "C12"],
    Inhibitor.AA: ["A13", "B13", "C13"],
    Inhibitor.HMF: ["A14", "B14", "C14"],
    Inhibitor.FA: ["A15", "B15", "C15"],
    Inhibitor.LVA: ["A16", "B16", "C16"],
    Inhibitor.LA: ["A17", "B17", "C17"],
}
_STOCK_QUOTES = {
    Inhibitor.FF: "Sheet1!A12=FF; Sheet1!B12=6; Sheet1!C12==B12*10 -> 60",
    Inhibitor.AA: "Sheet1!A13=AA; Sheet1!B13=4; Sheet1!C13==B13*100 -> 400",
    Inhibitor.HMF: "Sheet1!A14=HMF; Sheet1!B14=2.522; Sheet1!C14==B14*100 -> 252.2",
    Inhibitor.FA: "Sheet1!A15=FA; Sheet1!B15=1; Sheet1!C15==B15*100 -> 100",
    Inhibitor.LVA: "Sheet1!A16=LVA; Sheet1!B16=2; Sheet1!C16==B16*100 -> 200",
    Inhibitor.LA: "Sheet1!A17=LA; Sheet1!B17=40; Sheet1!C17==B17*10 -> 400",
}
_STOCK_VALUES = {
    Inhibitor.FF: 60.0,
    Inhibitor.AA: 400.0,
    Inhibitor.HMF: 252.2,
    Inhibitor.FA: 100.0,
    Inhibitor.LVA: 200.0,
    Inhibitor.LA: 400.0,
}
#: Stock concentration of each inhibitor, g/L ("concentrated solution (g/L)" C11).
STOCKS: dict[Inhibitor, SourcedValue] = {
    inhibitor: _archive(
        _STOCK_VALUES[inhibitor],
        INHIBITORS_XLSX,
        _STOCK_QUOTES[inhibitor],
        page="Sheet1 'Finding effective concentration' table, "
        f"{_STOCK_CELLS[inhibitor][0]}:{_STOCK_CELLS[inhibitor][-1]}",
    )
    for inhibitor in _STOCK_VALUES
}

_C2_QUOTES = {
    Inhibitor.FF: "Sheet1!A3=FF; Sheet1!C3=6g/L; Sheet1!F3=6",
    Inhibitor.AA: "Sheet1!A4=AA; Sheet1!C4=4g/L; Sheet1!F4=4",
    Inhibitor.HMF: "Sheet1!A5=HMF; Sheet1!D5=20 mM (2.522 g/L); Sheet1!F5=2.522",
    Inhibitor.FA: "Sheet1!A6=FA; Sheet1!E6=1g/L; Sheet1!F6=1",
    Inhibitor.LVA: "Sheet1!A7=LVA; Sheet1!E7=2g/L; Sheet1!F7=2",
    Inhibitor.LA: "Sheet1!A8=LA; Sheet1!C8=4%w/v (40 g/L); Sheet1!F8=40",
}
_C2_LITERATURE = {
    Inhibitor.FF: "C2 'Singh (LIT 185)'",
    Inhibitor.AA: "C2 'Singh (LIT 185)'",
    Inhibitor.HMF: "D2 'Pathway based Sig (Lit 192)'",
    Inhibitor.FA: "E2 'Victor E. Balderas-Hernandez'",
    Inhibitor.LVA: "E2 'Victor E. Balderas-Hernandez'",
    Inhibitor.LA: "C2 'Singh (LIT 185)'",
}
_C2_VALUES = {
    Inhibitor.FF: 6.0,
    Inhibitor.AA: 4.0,
    Inhibitor.HMF: 2.522,
    Inhibitor.FA: 1.0,
    Inhibitor.LVA: 2.0,
    Inhibitor.LA: 40.0,
}
#: The literature "common concentration" (C2) of each inhibitor, g/L, and the source
#: column it came from. These are the titration's fifth step (the red line of Fig 11a),
#: not the ex23 doses.
LITERATURE_C2: dict[Inhibitor, SourcedValue] = {
    inhibitor: _archive(
        _C2_VALUES[inhibitor],
        INHIBITORS_XLSX,
        _C2_QUOTES[inhibitor],
        page="Sheet1 'Combinatoric Study' table, row of the inhibitor",
        note=f"literature column {_C2_LITERATURE[inhibitor]}; the references are named "
        "only by the sheet's own labels",
    )
    for inhibitor in _C2_VALUES
}

_WELL_MAP_COLUMN = dict(zip(Inhibitor, "CDEFGH", strict=True))
_EX23_VOLUME_QUOTES = {
    Inhibitor.FF: "Sheet1!C1=FF_v_ul; Sheet1!A3=1_FF; Sheet1!C3=42.5",
    Inhibitor.AA: "Sheet1!D1=AA_v_ul; Sheet1!A4=2_AA; Sheet1!D4=8.5",
    Inhibitor.HMF: "Sheet1!E1=HMF_v_ul; Sheet1!A5=3_HMF; Sheet1!E5=17",
    Inhibitor.FA: "Sheet1!F1=FA_v_ul; Sheet1!A6=4_FA; Sheet1!F6=17",
    Inhibitor.LVA: "Sheet1!G1=LVA_v_ul; Sheet1!A7=5_LVA; Sheet1!G7=51",
    Inhibitor.LA: "Sheet1!H1=LA_v_ul; Sheet1!A8=6_LA; Sheet1!H8=85",
}
_EX23_VOLUMES = {
    Inhibitor.FF: 42.5,
    Inhibitor.AA: 8.5,
    Inhibitor.HMF: 17.0,
    Inhibitor.FA: 17.0,
    Inhibitor.LVA: 51.0,
    Inhibitor.LA: 85.0,
}
#: uL of stock of each inhibitor per 1700 uL ex23 culture tube (the well map; every
#: well holding the inhibitor carries the same volume).
EX23_VOLUMES_UL: dict[Inhibitor, SourcedValue] = {
    inhibitor: _archive(
        _EX23_VOLUMES[inhibitor],
        EX23_WELL_MAP,
        _EX23_VOLUME_QUOTES[inhibitor],
        page=f"Sheet1 column {_WELL_MAP_COLUMN[inhibitor]}, the single-inhibitor row",
    )
    for inhibitor in _EX23_VOLUMES
}
EX23_TUBE_UL = _archive(
    1700.0,
    EX23_DESIGN_NOTEBOOK,
    "ypd_v =  1700",
    page="cell after 'Volumes in microliters'",
)
EX23_DOSES_FROM_EX21 = _archive(
    "ex23 volumes chosen from the ex21 doubling times",
    EX23_DESIGN_NOTEBOOK,
    "Volumes in microliters. These values are selected based on the doubling times "
    "from 'MV_ex21_data_data_processing_v1.ipynb.'",
    page="markdown before the volume cell",
)
EX23_INOCULUM_IN_TUBE = _archive(
    "16 uL of OD 5 culture per 400 uL, taken out of the 1700 uL tube's YPD",
    EX23_DESIGN_NOTEBOOK,
    "df['ypd_v_ul'] = df['ypd_v_ul'] - (1700/400)*cell_vol",
    page="'Adjusting Table for Cell Volume'",
    note="with 'cell_vol = 16', so the 68 uL of inoculum is inside the 1700 uL and the "
    "doses are culture doses; whether the cell-volume table (2021-04-16) or the "
    "unadjusted one drove the pipetting is not recorded, and the notebook names a "
    "'1_inhibitor-screen_2021-04-14_205643.xlsx' that the archive does not hold",
)
EX23_MEDIUM = _archive(
    "YPD",
    EX23_DESIGN_NOTEBOOK,
    "represent the wild type BY4742 in only YPD",
    page="'Descrition of Experiment'",
    note="the uninhibited wells are the strain in YPD, and every tube is made up with "
    "YPD (ypd_v_ul); the YPD recipe itself is not recorded (MEDIUM_RECIPE_GAP)",
)
EX23_INITIAL_OD = _archive(
    0.2,
    EX23_OD_CALCULATION,
    "Sheet1!A10=all; Sheet1!B10=5; Sheet1!C10=400; Sheet1!D10=0.2; "
    "Sheet1!E10==ROUND(D10*C10/B10,3) -> 16",
    page="'into 1.7 ml tubes' table",
    note="OD600 targeted at inoculation: 16 uL of an OD 5 culture per 400 uL",
)
EX23_PRECULTURE = _archive(
    "YPD + G418, 6 mL, per biological replicate",
    EX23_OD_CALCULATION,
    "Sheet1!B1=Combination Inhibitors OD's. Grown up in proper 1x G418 6 mL each",
    page="Sheet1 B1",
)
EX21_INITIAL_OD = _archive(
    0.2,
    INHIBITORS_XLSX,
    "Sheet1!A70=BR 1; Sheet1!D70=0.2; Sheet1!E70==D70*10 -> 2",
    page="Sheet1 rows 69-72 (OD Final, OD culture tube)",
    note="the targeted final OD600; the culture tube is made at 10x that OD, which "
    "implies a 1:10 addition to the medium (EX21_INOCULUM_GAP)",
)
EX21_TUBE_UL = _archive(
    4000.0,
    INHIBITORS_XLSX,
    "Sheet1!A33=2; Sheet1!B33==ROUND(B21/$C$12*4,5) -> 2",
    page="Sheet1 'BSC array' mL table, rows 33-41",
    note="mL of stock = target g/L / stock g/L x 4, so each titration tube is 4 mL",
)
ISOBOLE_WELL_UL = _archive(
    200.0,
    ISOBOLE_DESIGN_XLSX,
    "FF_AA_new !A12=total volume (uL); FF_AA_new !B12=200",
    page="sheet 'FF_AA_new ' A12:B12 (the same cells on FA_AA and HMF_AA)",
)
ISOBOLE_INOCULUM_UL = _archive(
    20.0,
    ISOBOLE_SETUP_PPTX,
    "we can use a standard inoculation of 20 |uL| for preprepared inoculated YPD",
    page="slide 1",
    note="a .pptx has no plain text, so the quote is the slide's text runs joined by "
    "'|' (ppt/slides/slide1.xml a:t elements); the Fluent worklist dispenses 20 uL "
    "from the inoculated-YPD trough to every well",
)
ISOBOLE_INITIAL_OD = _archive(
    0.2,
    ISOBOLE_DESIGN_XLSX,
    "FF_AA_new !C25=final od2; FF_AA_new !C26=0.2",
    page="sheet 'FF_AA_new ' OD table",
)
EX21_FINAL_VOLUME_RANGE_UL = _archive(
    (102.4, 194.8),
    EX21_FINAL_VOLUMES,
    "Sheet1!A1=Well; Sheet1!B1=Final Volume (uL)",
    page="Sheet1 B2:B201",
    note="volume left in each ex21 well at the end of the run, min and max over the "
    "200 wells; the starting volume is not recorded (WELL_VOLUME_GAP)",
)

# --------------------------------------------------------------------------- #
# Measured from the instrument's own log
# --------------------------------------------------------------------------- #
TRAY_FIELDS = _archive(
    ("time", "tray_c", "cover_c"),
    EX1_TEMPERATURE_TXT,
    "Time        Tray Cover",
    page="header line (UTF-16 text)",
    note="ex1's exported temperature log labels the columns; ex1's .bsm T records "
    "round to the same tray and cover values row for row (27.0535 -> 27.1, 26.9904 "
    "-> 27.0, 29.3899 -> 29.4, 29.9858 -> 30.0), which fixes the T-record field order",
)
#: Median tray temperature after warm-up, C, per run (bioscreen.summarize_tray_
#: temperature on the run's .bsm, measured 2026-10-08); 1st to 99th percentile
#: 29.9858 to 30.01 C on every run.
_STEADY_T_RECORD = {
    "ex21": "T,44290.5835967593,29.9858,31.0763",
    "ex23": "T,44303.5038695255,30.01,31.1017",
    "ex26": "T,44446.9827069792,30.01,30.9749",
    "ex27": "T,44451.2020734606,30.01,31.1272",
    "ex28": "T,44470.0565416319,30.01,31.1017",
}
_N_STEADY = {"ex21": 4316, "ex23": 5097, "ex26": 5756, "ex27": 5756, "ex28": 5756}
TRAY_TEMPERATURE_C: dict[str, SourcedValue] = {
    run: _archive(
        30.01,
        RAW_BSM[run],
        record,
        page="T records; the quote is the 600th (time, tray C, cover C)",
        note="measured, not a set point: the median logged tray temperature over the "
        f"{_N_STEADY[run]} T records from the first reading >= 29.5 C "
        "(bioscreen.summarize_tray_temperature)",
    )
    for run, record in _STEADY_T_RECORD.items()
}
RUN_EVENTS: dict[str, SourcedValue] = {
    "ex21": _archive(
        71.97,
        RAW_BSM["ex21"],
        "E,4/7/2021 4:01:36 AM (03 00:01:56) : Measurement Completed",
        page="E records",
        note="hours from first to last reading of the export",
    ),
    "ex23": _archive(
        84.97,
        RAW_BSM["ex23"],
        "E,4/20/2021 3:07:50 PM (03 13:03:07) : Measurement Aborted",
        page="E records",
        note="programmed for 345600 s (S,3) and stopped by hand after 3 d 13 h; hours "
        "from first to last reading of the export",
    ),
    "ex26": _archive(
        95.99,
        RAW_BSM["ex26"],
        "E,9/11/2021 1:37:05 PM (04 00:01:55) : Measurement Completed",
        page="E records",
        note="hours from first to last reading of the export",
    ),
    "ex27": _archive(
        95.98,
        RAW_BSM["ex27"],
        "E,9/15/2021 6:52:43 PM (04 00:01:55) : Measurement Completed",
        page="E records",
        note="hours from first to last reading of the export",
    ),
    "ex28": _archive(
        95.98,
        RAW_BSM["ex28"],
        "E,10/4/2021 3:23:10 PM (04 00:01:56) : Measurement Completed",
        page="E records",
        note="hours from first to last reading of the export",
    ),
}
READ_INTERVAL_S = _archive(
    900,
    RAW_BSM["ex23"],
    "S,4,900",
    page="S records",
    note="the instrument's code S,4 is not documented in any mirrored source; 900 s is "
    "read as the read interval because every export is spaced 15 min",
)

# --------------------------------------------------------------------------- #
# Typed gaps
# --------------------------------------------------------------------------- #
_BSM_LOOKED_IN = archive_provenance(RAW_BSM["ex23"], "S and P records")

SHAKING_GAP = ProvenanceGap(
    field="shaking",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_BSM_LOOKED_IN,
    note="neither the report nor any wet-lab record states the shaking amplitude, "
    "speed or duty. The .bsm settings are instrument codes no mirrored source "
    "decodes. Hypothesis (unverified): S,U encodes it, since S,U changes from 1 to 2 "
    "at ex13, the first run whose file name reads 'amp_med_speed_fast', and stays 2 "
    "for ex21-ex28",
)
TEMPERATURE_SET_POINT_GAP = ProvenanceGap(
    field="temperature_set_point",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_BSM_LOOKED_IN,
    note="the set point is not stated anywhere decoded; the measured tray temperature "
    "is TRAY_TEMPERATURE_C (median 30.01 C on every run). S,2 = 30 in every .bsm is a "
    "plausible set point, unverified",
)
WELL_VOLUME_GAP = ProvenanceGap(
    field="well_volume",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=archive_provenance(EX23_OD_CALCULATION),
    note="the volume dispensed per well is not recorded for ex21 or ex23 (ex23's "
    "'V2 (uL) 400' is the culture volume per 16 uL of inoculum, not shown to be the "
    "well volume). The isoboles are 200 uL (ISOBOLE_WELL_UL)",
)
EX21_INOCULUM_GAP = ProvenanceGap(
    field="ex21_inoculum_dilution",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=archive_provenance(INHIBITORS_XLSX, "Sheet1 rows 69-72"),
    note="ex21 doses are the 4 mL tube concentrations. How the 10x culture was added "
    "(into the tube or the well, and how much) is not recorded; if it was a 1:10 "
    "addition to medium already at the tube dose, the culture dose is 0.9x the "
    "recorded one. Hypothesis, unverified",
)
MEDIUM_RECIPE_GAP = ProvenanceGap(
    field="medium_recipe",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=REPORT.model_copy(update={"page": "whole report"}),
    note="the runs are in YPD (EX23_MEDIUM); neither the report nor the wet-lab "
    "records give its recipe or supplier",
)
ISOBOLE_ORIENTATION_GAP = ProvenanceGap(
    field="isobole_grid_orientation",
    reason=ProvenanceGapReason.deferred_pending_source_review,
    looked_in=archive_provenance(ISOBOLE_SETUP_PPTX, "slide 1"),
    resolve_with=Provenance(
        source_uri="data/03_analysis_code/inhibitor_tolerance/notebooks/"
        "n2--plotting_isoboles.ipynb",
        citation_key=CITATION_KEY,
        method="thesis-archive notebook, not in the raw mirror",
    ),
    note="the layout follows the 039 grid rule (well 1 uninhibited, the fast well "
    "digit the first inhibitor). The design slide says 'All the first row contains "
    "90 uL of inhibitor 1', and the Fluent worklist dispenses 90 uL from trough 1 to "
    "well 1, which disagree with it. Measured 2026-10-08: of the four axis flips, "
    "only the 039 orientation keeps the grew calls monotone in dose (0 to 2 "
    "violations per plate against 4 to 17); a transpose of the two inhibitors is not "
    "distinguishable that way",
)

GAPS: tuple[ProvenanceGap, ...] = (
    SHAKING_GAP,
    TEMPERATURE_SET_POINT_GAP,
    WELL_VOLUME_GAP,
    EX21_INOCULUM_GAP,
    MEDIUM_RECIPE_GAP,
    ISOBOLE_ORIENTATION_GAP,
)
