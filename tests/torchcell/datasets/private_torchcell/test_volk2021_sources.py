# tests/torchcell/datasets/private_torchcell/test_volk2021_sources.py
# [[tests.torchcell.datasets.private_torchcell.test_volk2021_sources]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/private_torchcell/test_volk2021_sources.py
"""Every sourced constant of the 2021 inhibitor runs is re-read from its source.

Library-bound values (the report OCR, Lian 2019) are audited with
``audit_sourced_value`` against ``$DATA_ROOT/torchcell-library`` and skip when it is not
mounted. Archive-bound values are re-read from the thesis archive and skip when it is
not mounted: the file's sha256 must equal the pin, and the quote must be found (verbatim
text, decoded UTF-16 for ``ex1.txt``, slide text runs for the ``.pptx``, or the
regenerated :func:`cell_quote` for a spreadsheet).
"""

from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path

import pytest

from torchcell.datasets.private_torchcell import bioscreen as b
from torchcell.datasets.private_torchcell import volk2021_sources as s
from torchcell.literature.manifest import sha256_file
from torchcell.verification.sourced import (
    ProvenanceGap,
    SourcedValue,
    audit_sourced_value,
)

LIBRARY = Path(os.environ.get("DATA_ROOT", "/scratch/projects/torchcell-scratch")) / (
    "torchcell-library"
)


def _sourced_values() -> list[tuple[str, SourcedValue]]:
    out: list[tuple[str, SourcedValue]] = []
    for name, value in vars(s).items():
        if isinstance(value, SourcedValue):
            out.append((name, value))
        elif isinstance(value, dict):
            out += [
                (f"{name}[{k}]", v)
                for k, v in value.items()
                if isinstance(v, SourcedValue)
            ]
    return out


SOURCED = _sourced_values()
LIBRARY_BOUND = [
    (n, v) for n, v in SOURCED if not v.provenance.source_uri.startswith("data/")
]
ARCHIVE_BOUND = [
    (n, v) for n, v in SOURCED if v.provenance.source_uri.startswith("data/")
]


def test_the_module_binds_report_lian_and_archive_values() -> None:
    keys = {v.provenance.citation_key for _, v in SOURCED}
    assert keys == {b.CITATION_KEY, s.LIAN_KEY}
    assert len(LIBRARY_BOUND) == 12
    assert len(ARCHIVE_BOUND) >= 30


def test_every_archive_binding_pins_the_manifest_sha256() -> None:
    for name, value in ARCHIVE_BOUND:
        assert (
            value.provenance.sha256 == s.ARCHIVE_SHA256[value.provenance.source_uri]
        ), name


def test_ex23_doses_are_stock_times_volume_over_the_tube() -> None:
    for inhibitor in b.INHIBITORS:
        dose = (
            s.STOCKS[inhibitor].value
            * s.EX23_VOLUMES_UL[inhibitor].value
            / s.EX23_TUBE_UL.value
        )
        assert dose == pytest.approx(b.EX23_G_PER_L[inhibitor], abs=1e-12)
    assert s.EX23_TUBE_UL.value == b.EX23_TUBE_UL


def test_every_gap_is_typed_and_explained() -> None:
    assert all(isinstance(g, ProvenanceGap) for g in s.GAPS)
    assert len({g.field for g in s.GAPS}) == len(s.GAPS)
    assert all(g.note for g in s.GAPS)


@pytest.mark.skipif(not LIBRARY.is_dir(), reason="the library mirror is not mounted")
@pytest.mark.parametrize(
    ("name", "value"), LIBRARY_BOUND, ids=[n for n, _ in LIBRARY_BOUND]
)
def test_library_quote_is_in_the_pinned_ocr(name: str, value: SourcedValue) -> None:
    if not value.source_path(LIBRARY).is_file():
        pytest.skip(f"{value.source_path(LIBRARY)} is not mirrored here")
    result = audit_sourced_value(value, LIBRARY)
    assert result.passed, (name, result.message)


def _slide_text(path: Path) -> str:
    with zipfile.ZipFile(path) as deck:
        xml = deck.read("ppt/slides/slide1.xml").decode()
    return "|".join(re.findall(r"<a:t>([^<]*)</a:t>", xml))


def _cells(quote: str) -> tuple[str, list[str]]:
    sheet = ""
    cells: list[str] = []
    for part in quote.split("; "):
        location = part.split("=", 1)[0]
        sheet, cell = location.rsplit("!", 1)
        cells.append(cell)
    return sheet, cells


@pytest.mark.skipif(
    not (b.ARCHIVE_ROOT / b.ARCHIVE_MANIFEST).is_file(),
    reason="the thesis archive on /bulk is not mounted",
)
@pytest.mark.parametrize(
    ("name", "value"), ARCHIVE_BOUND, ids=[n for n, _ in ARCHIVE_BOUND]
)
def test_archive_quote_is_in_the_archive_file(name: str, value: SourcedValue) -> None:
    path = b.ARCHIVE_ROOT / value.provenance.source_uri
    assert sha256_file(path) == value.provenance.sha256, name
    if path.suffix == ".xlsx":
        sheet, cells = _cells(value.quote)
        stored_sheet = None if sheet == "Sheet1" else sheet
        assert s.cell_quote(path, cells, stored_sheet) == value.quote, name
    elif path.suffix == ".pptx":
        assert value.quote in _slide_text(path), name
    elif path.name == "ex1.txt":
        assert value.quote in path.read_text(encoding="utf-16"), name
    else:
        assert value.quote in path.read_text(encoding="latin-1"), name
