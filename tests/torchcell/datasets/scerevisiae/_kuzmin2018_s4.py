# tests/torchcell/datasets/scerevisiae/_kuzmin2018_s4.py
# [[tests.torchcell.datasets.scerevisiae._kuzmin2018_s4]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/_kuzmin2018_s4.py
"""A synthetic Additional Data S4 for the hermetic Kuzmin 2018 builds.

Smf and Dmf read the query strains' fitness SD from Additional Data S4, linked into
``raw/`` from the raw mirror and verified against ``kuzmin2018.S4_SHA256``.
``install(monkeypatch, tmp_path, rows)`` writes one S4 workbook (the released column
names) into a throwaway mirror, points ``kuzmin2018.raw_mirror_dir`` at it and pins
``kuzmin2018.S4_SHA256`` to its bytes, so ``download()`` links it and ``process()``
verifies it like the real file.
"""

import hashlib
from pathlib import Path

import openpyxl
import pytest

from torchcell.datasets.scerevisiae import kuzmin2018 as k

S4_COLUMNS = [
    "Query Strain ID",
    "ORF1",
    "ORF2",
    "Gene1",
    "Gene2",
    "Allele1",
    "Allele2",
    "Fitness",
    "St.dev.",
]

#: (bare tm number, fitness, SD); None for a blank S4 cell.
S4Row = tuple[str, float | None, float | None]


def install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rows: list[S4Row]) -> Path:
    """Write the S4 workbook into a mirror under ``tmp_path`` and pin it."""
    mirror = tmp_path / "s4_raw_mirror"
    (mirror / "data").mkdir(parents=True)
    path = mirror / "data" / k.S4_NAME
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Fitness standard"
    sheet.append(S4_COLUMNS)
    for tm, fitness, sd in rows:
        sheet.append(
            [tm, "YAA000W", "YDL227C", "AAA1", "HO", "aaa1Δ", "hoΔ", fitness, sd]
        )
    book.save(path)
    monkeypatch.setattr(k, "raw_mirror_dir", lambda data_root=None: str(mirror))
    monkeypatch.setattr(k, "S4_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return path
