# tests/torchcell/datasets/scerevisiae/_jackson2020_release.py
"""A miniature Jackson 2020 release, written to disk, for the hermetic tests.

Four strains (two HO-deletion controls, ``gat1_A5``, and ``stp2_A12`` whose diploid
Table S2 cell names stp1, as seven real rows do), nine count columns (seven genes in
the R-mangled spellings the real header uses, plus the two cassette markers) and sixteen
cells over the eleven conditions. ``install`` writes the five raw files the loader
consumes and pins the module constants the real release would otherwise impose (strain
count, cell and count totals, the disagreement list, the file sha256s).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
from pathlib import Path
from typing import Any, cast

import openpyxl
import pandas as pd
import pytest

from torchcell.datasets.scerevisiae import jackson2020 as j
from torchcell.sequence.genome.base import GeneNameResolution, GeneNameStatus
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

RELEASED = [
    "YAL001C",
    "YAL002W.A",
    "snR10",
    "X15S_rRNA",
    "YIL168W",
    "YIL167W",
    "YIL080W",
    "KANMX",
    "NATMX",
]
BULK_HEADER = [
    "YAL001C",
    "YAL002W-A",
    "snR10",
    "15S_rRNA",
    "YIL168W",
    "YIL167W",
    "YIL080W",
    "KANMX",
    "NATMX",
]
#: The record key each released column gets (markers keep their names).
KEYS = [
    "YAL001C",
    "YAL002W-A",
    "YNCG0013W",
    "YNCQ0002W",
    "YIL168W",
    "YIL167W",
    "YIL080W",
    "KANMX",
    "NATMX",
]

STRAINS: dict[str, dict[str, Any]] = {
    "WT(ho)_A1": {
        "token": "ho",
        "group": "WT(ho)",
        "barcode": "AAAAGTTTGAAAGTTTGAA",
        "dgy": 1997.0,
    },
    "WT(ho)_B1": {
        "token": "ho",
        "group": "WT(ho)",
        "barcode": "CCCCGTTTGCCCGTTTGCC",
        "dgy": 2062.0,
    },
    "gat1_A5": {
        "token": "gat1",
        "group": "gat1",
        "barcode": "GGGGGTTTGGGGGTTTGGG",
        "dgy": 1919.0,
    },
    "stp2_A12": {
        "token": "stp2",
        "group": "stp2",
        "barcode": "TTTTGTTTGTTTTGTTTGT",
        "dgy": 1912.0,
    },
}
NATMX_PARENTS = {"ho": 2111.0, "gat1": 2103.0, "stp2": 2110.0}

#: (strain, release condition label, droplet barcode, counts per RELEASED column).
CELLS: list[tuple[str, str, str, list[int]]] = [
    *(
        ("WT(ho)_A1", label, f"AC{i:02d}", [3, 0, 1, 2, 0, 1, 0, 4, 2])
        for i, label in enumerate(
            [
                "CStarve",
                "Urea",
                "AmmoniumSulfate",
                "Proline",
                "Glutamine",
                "MinimalEtOH",
                "MinimalGlucose",
                "YPEtOH",
                "YPDRapa",
                "YPDDiauxic",
                "YPD",
            ]
        )
    ),
    ("WT(ho)_B1", "YPD", "BC01", [1, 1, 0, 0, 0, 0, 0, 1, 0]),
    ("WT(ho)_B1", "YPD", "BC02", [0, 2, 0, 0, 1, 0, 0, 0, 1]),
    ("gat1_A5", "YPD", "GC01", [5, 0, 0, 0, 0, 0, 2, 3, 3]),
    ("stp2_A12", "YPD", "SC01", [0, 0, 0, 1, 0, 0, 0, 1, 1]),
    ("gat1_A5", "CStarve", "GC02", [2, 0, 0, 0, 0, 0, 0, 0, 0]),
]
N_CELLS = len(CELLS)
N_UMIS = sum(sum(c[3]) for c in CELLS)
#: Strain-condition pairs holding a cell: 11 control pairs + 4.
N_RECORDS = 15

SETTINGS_TEXT = """# figure_common_settings.R (miniature)
CONDITION.LEVELS <<- c("CStarve", "Urea", "AmmoniumSulfate", "Proline", "Glutamine", "MinimalEtOH", "MinimalGlucose", "YPEtOH", "YPDRapa", "YPDDiauxic", "YPD")
CONDITION.LABELS <<- c("CSTARVE", "NLIM-UREA", "NLIM-NH4", "NLIM-PRO", "NLIM-GLN", "MMEtOH", "MMD", "YPEtOH", "RAPA", "DIAUXY", "YPD")
if (!exists('nitrogen.tfs')) {
  nitrogen.tfs <<- c("YFL021W", # gat1
                     "YHR006W"  # stp2
  )
}
"""

_RESOLUTIONS: dict[str, tuple[GeneNameStatus, str | None]] = {
    "YAL001C": (GeneNameStatus.CURRENT, "YAL001C"),
    "YAL002W-A": (GeneNameStatus.CURRENT, "YAL002W-A"),
    "SNR10": (GeneNameStatus.NON_GENE_FEATURE, "YNCG0013W"),
    "15S_RRNA": (GeneNameStatus.NON_GENE_FEATURE, "YNCQ0002W"),
    "YIL168W": (GeneNameStatus.NON_GENE_FEATURE, "YIL167W"),
    "YIL167W": (GeneNameStatus.NON_GENE_FEATURE, "YIL167W"),
    "YIL080W": (GeneNameStatus.RETIRED, "YIL080W"),
    "HO": (GeneNameStatus.RENAMED, "YDL227C"),
    "YDL227C": (GeneNameStatus.CURRENT, "YDL227C"),
    "GAT1": (GeneNameStatus.RENAMED, "YFL021W"),
    "YFL021W": (GeneNameStatus.CURRENT, "YFL021W"),
    "STP2": (GeneNameStatus.RENAMED, "YHR006W"),
    "YHR006W": (GeneNameStatus.CURRENT, "YHR006W"),
}


class FakeGenome:
    """The two genome surfaces the loader reads: the resolver and the gene table."""

    def __init__(
        self, extra: dict[str, tuple[GeneNameStatus, str | None]] | None = None
    ):
        self.resolutions = {**_RESOLUTIONS, **(extra or {})}
        self.gene_attribute_table = pd.DataFrame(
            {
                "ID": ["YDL227C", "YFL021W", "YHR006W", "YAL001C"],
                "gene": ["HO", "GAT1", "STP2", "TFC3"],
            }
        )

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        status, systematic = self.resolutions[name.strip().upper()]
        return GeneNameResolution(
            input_name=name, status=status, systematic_name=systematic, candidates=[]
        )


def fake_genome(
    extra: dict[str, tuple[GeneNameStatus, str | None]] | None = None,
) -> SCerevisiaeGenome:
    """A ``FakeGenome`` typed as the genome the loader takes (it reads two surfaces)."""
    return cast(SCerevisiaeGenome, FakeGenome(extra))


def _diploid_genotype(strain: str, token: str) -> str:
    gene = "stp1" if strain == "stp2_A12" else token
    return f"MATa/MATα Δ{gene}::KanMX/Δ{gene}::NatMX"


def write_workbook(path: Path, table4_rows: list[str] | None = None) -> None:
    """Table S2 as the real sheet lays it out, and Table 4 as one quote per row."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = j.TABLE_S2_SHEET
    sheet.append(
        [
            "Strain ID (DGY)",
            "Strain",
            "Genotype",
            "Parent",
            "Barcode",
            "Plate Location",
            "Reference",
        ]
    )
    sheet.append([1.0, "FY4", "MATa"])
    sheet.append([2.0, "FY5", "MATα"])
    for strain, info in STRAINS.items():
        token = info["token"]
        sheet.append(
            [
                info["dgy"],
                strain,
                f"MATa Δ{token}::KanMX",
                "FY4",
                info["barcode"],
                None,
                "This work",
            ]
        )
    for token, dgy in NATMX_PARENTS.items():
        sheet.append(
            [dgy, token, f"MATα Δ{token}::NatMX", "FY5", None, None, "This work"]
        )
    for strain, info in STRAINS.items():
        well = strain.rsplit("_", 1)[1]
        sheet.append(
            [
                "N/A",
                strain,
                _diploid_genotype(strain, info["token"]),
                None,
                info["barcode"],
                f"BC_DEL_1_{well}",
                "This work",
            ]
        )
    media = workbook.create_sheet("Supplemental Table 4 - Media")
    for row in table4_rows or []:
        media.append([row])
    workbook.save(path)


def write_matrix(path: Path, cells: list[tuple[str, str, str, list[int]]]) -> None:
    """The count matrix as R's ``write.table`` writes it: quoted strings, row names."""
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow([*RELEASED, *j.META_COLUMNS])
        for i, (strain, label, barcode, counts) in enumerate(cells, start=1):
            info = STRAINS[strain]
            writer.writerow(
                [
                    f"{i}_{label}",
                    *counts,
                    strain,
                    info["group"],
                    strain.rsplit("_", 1)[1],
                    label,
                    barcode,
                ]
            )


def write_raw(
    raw: Path, cells: list[tuple[str, str, str, list[int]]] = CELLS
) -> dict[str, str]:
    """The five raw files under ``raw``; returns their sha256 pins."""
    raw.mkdir(parents=True, exist_ok=True)
    write_matrix(raw / j.MATRIX_NAME, cells)
    write_workbook(raw / j.SUPP1_NAME)
    (raw / j.SETTINGS_NAME).write_text(SETTINGS_TEXT)
    (raw / j.BULK_NAME).write_text("\t".join(BULK_HEADER) + "\n")
    (raw / j.BARCODES_NAME).write_text(
        "".join(f">{s}\n{i['barcode']}\n" for s, i in STRAINS.items())
    )
    return {
        name: hashlib.sha256((raw / name).read_bytes()).hexdigest()
        for name in j.RAW_FILES
    }


def pin_constants(monkeypatch: pytest.MonkeyPatch, pins: dict[str, str]) -> None:
    """Point the module's release-shape constants at the miniature release."""
    monkeypatch.setattr(j, "RAW_FILES", pins)
    monkeypatch.setattr(j, "N_STRAINS", len(STRAINS))
    monkeypatch.setattr(j, "N_RELEASED_FEATURES", len(RELEASED))
    monkeypatch.setattr(j, "PAPER_CELLS", N_CELLS)
    monkeypatch.setattr(j, "PAPER_TRANSCRIPT_COUNTS", N_UMIS)
    monkeypatch.setattr(j, "TABLE_S2_GENOTYPE_DISAGREEMENTS", ("stp2_A12",))


def install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the miniature raw dir under ``<tmp>/store/raw`` and pin the constants."""
    root = tmp_path / "store"
    pins = write_raw(root / "raw")
    pin_constants(monkeypatch, pins)
    return root
