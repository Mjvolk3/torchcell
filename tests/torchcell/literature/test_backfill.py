# tests/torchcell/literature/test_backfill.py
# [[tests.torchcell.literature.test_backfill]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_backfill.py
"""Tests for torchcell.literature.backfill (offline manifest regularization).

2026.09.30 (Phase 13): every path runs on a synthetic mirror under ``tmp_path``; Zotero
is the read-only ``FakeZot`` of ``_fake_zotero.py`` behind ``ZoteroLibrary`` (its
pyzotero client stubbed at the import site), so nothing reaches the network and nothing
can write to a library. ``manifest.datetime`` is frozen at 2026-09-30T12:00:00+00:00 so
each written ``manifest.json`` is compared whole.

Fixture bytes and their digests (``hashlib.sha256(b).hexdigest()``, checked in Python):

- ``paper.pdf`` = ``b"%PDF-1.5 fake pdf bytes"`` (23 bytes, ``8fa52265...64cb5``)
- ``paper.md`` = the ``_MD`` constant, three newline-ended lines (33, ``2b52d5ad...7296b``)
- ``si/si1.pdf`` = ``b"%PDF-1.5 fake si"`` (16, ``c925eeea...aa693``)
- ``images/fig1.jpg`` = the ``_JPG`` constant, the JPEG magic bytes plus text (13,
  ``9a44b46c...e6eb9``)
- ``thesis.pdf`` = ``b"%PDF-1.5 thesis"`` (15), ``thesis.txt`` = one line of text (12)

``build_manifest`` walks ``sorted(rglob("*"))``, so the files of the paper key are
listed images/fig1.jpg, paper.md, paper.pdf, si/si1.pdf. Offline, only the markdown
MinerU writes (``paper.md`` and ``si/*.md``, by location) gets a default source
(``"mineru-ocr"``). Enriched, the Zotero children are listed SI
first; ``pdf_attachments`` sorts the main article first, so ``paper.pdf`` gets ATT1 (md5
``"md5main"``) and ``si/si1.pdf`` gets ATT2 (no md5 in Zotero, so ``zotero_md5`` None);
the item's collection keys ``["C1", "C9"]`` resolve to ``["yeast", "C9"]`` (C9 is not in
the library, so its key is kept).

2026.09.30 (issue #525): the five Phase 13 Findings are retired. An enriched manifest
is ``provenance_complete`` only when Zotero gives both a DOI and a title; a top-level
``thesis.txt`` is ``paper_ocr`` with no source (only ``paper.md`` and ``si/*.md`` are
MinerU output); only citation-key directories (no leading ``_``, at least one file) get
manifests; an existing manifest is read on skip and a corrupt one raises
``CorruptManifestError`` naming its path; two Zotero items sharing a citation key raise
``DuplicateCitationKeyError`` listing both item keys.

2026.09.30 (issue #564): MinerU sidecars under ``si/`` are OCR byproducts, not SI data.
``_run_mineru.py`` copies ``<stem>_content_list.json``, ``<stem>_middle.json`` and
``images/`` next to every PDF it OCRs, so ``si/si1_middle.json`` and
``si/si1_content_list.json`` are ``ocr_layout`` and ``si/images/*.jpg`` is ``ocr_image``,
exactly as their ``paper_*`` and ``images/`` counterparts. The rules are anchored to the
runner's output, not substrings: ``si/Figure_S1_images/a.png`` and
``si/Table_middle.json`` stay ``si_data``. ``si/si_data/`` keeps
``si_data`` even for a released file whose name looks like a sidecar, and a loose
``si/`` table is still ``si_data``. The full role table of a captured key is pinned on
the live layout of ``avsecEffectiveGeneExpression2021`` (one SI PDF, one image per
side here).

2026.10.01 (issue #579, #546): each SI PDF's figures live in their own
``si/images/<si stem>/`` (``ocr.images_dir_for``), and those are ``ocr_image``; the
flat ``si/images/<file>`` of keys OCR'd before the fix stays ``ocr_image`` so they are
valid without a re-OCR, while ``si/images/Figure/a.png`` (not an ``si*`` stem) is not
an OCR figure. A non-paper PDF at the key root (Costanzo 2016's ``SOM.pdf``) writes
``images/<stem>/<file>``, which is ``ocr_image`` (PR #585 review). ``<stem>_ocr_provenance.json``
is ``ocr_provenance`` and ``build_manifest`` attaches it, parsed as a
``ProcessingRecord``, to the markdown beside it; markdown with no such file keeps
``processing`` None.

2026.10.02 (issue #607): the citation index is read from Zotero's top-level items
(``top()``, which ``FakeZot`` answers with every item that has no ``parentItem``), and
an item whose ``itemType`` is ``attachment``, ``note`` or ``annotation`` is skipped even
when standalone. ``citation_index_with_duplicates`` returns the duplicates instead of
raising; ``build_citation_index`` and ``backfill_mirror`` still refuse any duplicate.
"""

import hashlib
import json
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tests.torchcell.literature._fake_zotero import FakeZot, collection, make_library
from torchcell.literature import backfill as bf
from torchcell.literature import manifest as manifest_module
from torchcell.literature.backfill import (
    BackfillReport,
    KeyBackfillResult,
    backfill_key,
    backfill_mirror,
    build_citation_index,
    library_root,
)
from torchcell.literature.manifest import (
    MANIFEST_FILENAME,
    Manifest,
    ProcessingRecord,
    _role_for,
    build_manifest,
    sha256_file,
)
from torchcell.literature.zotero import ZoteroLibrary

_FROZEN = "2026-09-30T12:00:00+00:00"
_PDF = b"%PDF-1.5 fake pdf bytes"
_MD = b"# Fake paper\n\nOCR markdown body.\n"
_SI = b"%PDF-1.5 fake si"
_JPG = b"\xff\xd8\xff fake jpeg"
_SHA_PDF = "8fa522657b5f1000e2644ccf23602da05bdff01076a17085ada49af62c564cb5"
_SHA_MD = "2b52d5ad3e114ff0d6e838ca872aa424e9ceb18987daa17144bbc33d7127296b"
_SHA_SI = "c925eeead58c6264fb8ce5da2dfea07ed9b7cd0f06d321cbd80dcc3c43eaa693"
_SHA_JPG = "9a44b46ca3566ef3e0573cfc90389f28653f7b2a8b8852b7c6ff623451ae6eb9"
_SHA_THESIS_PDF = "9624ba4f93744996e7c52f1be9dbc9bc1cafe109a7f39f91aed6d90b4af92765"
_SHA_THESIS_TXT = "4215a95e513cb1e675fa655be781b7d52bbd1fcd7f77e9842a91965fb2bd4bec"


class _FrozenDatetime:
    """Stands in for ``manifest.datetime``: ``now`` is a fixed instant."""

    @staticmethod
    def now(tz: Any = None) -> datetime:
        return datetime(2026, 9, 30, 12, 0, tzinfo=UTC)


@pytest.fixture
def frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze ``created_at`` so a written manifest can be compared whole."""
    monkeypatch.setattr(manifest_module, "datetime", _FrozenDatetime)


def _make_paper_key(root: Path, name: str = "fakePaperKey2020") -> Path:
    d = root / name
    (d / "images").mkdir(parents=True)
    (d / "paper.pdf").write_bytes(b"%PDF-1.5 fake pdf bytes")
    (d / "paper.md").write_text("# Fake paper\n\nOCR markdown body about kinases.")
    (d / "paper_content_list.json").write_text('[{"type": "text", "text": "x"}]')
    (d / "images" / "fig1.jpg").write_bytes(b"\xff\xd8\xff fake jpeg")
    return d


def _make_data_key(root: Path, name: str = "fakeDataKey2021") -> Path:
    d = root / name
    (d / "data").mkdir(parents=True)
    (d / "data" / "titers.xlsx").write_bytes(b"PK fake xlsx")
    return d


def _make_exact_key(root: Path, name: str = "fakePaperKey2020") -> Path:
    """A paper key whose four files have the digests in the module docstring."""
    d = root / name
    (d / "images").mkdir(parents=True)
    (d / "si").mkdir()
    (d / "paper.pdf").write_bytes(_PDF)
    (d / "paper.md").write_bytes(_MD)
    (d / "si" / "si1.pdf").write_bytes(_SI)
    (d / "images" / "fig1.jpg").write_bytes(_JPG)
    return d


def _record(
    path: str,
    role: str,
    size: int,
    sha: str,
    source: str | None = None,
    md5: str | None = None,
) -> dict[str, Any]:
    """One ``ArtifactRecord`` as ``model_dump_json`` writes it."""
    return {
        "path": path,
        "role": role,
        "bytes": size,
        "sha256": sha,
        "source": source,
        "zotero_md5": md5,
        "retrieval": None,
        "processing": None,
    }


def _manifest(
    citation_key: str, files: list[dict[str, Any]], **meta: Any
) -> dict[str, Any]:
    """A whole ``manifest.json`` with the backfill's defaults."""
    out: dict[str, Any] = {
        "version": 1,
        "citation_key": citation_key,
        "doi": None,
        "title": None,
        "library_id": None,
        "zotero_item_key": None,
        "collections": [],
        "files": files,
        "si_data_sources": [],
        "si_expected": [],
        "provenance_complete": False,
        "created_at": _FROZEN,
    }
    out.update(meta)
    return out


def _item(
    key: str, data: dict[str, Any], item_type: str = "journalArticle"
) -> dict[str, Any]:  # a Zotero item as pyzotero returns it
    return {"key": key, "data": {"itemType": item_type, **data}}


def _pdf_child(key: str, title: str, md5: str | None) -> dict[str, Any]:
    data: dict[str, Any] = {"contentType": "application/pdf", "title": title}
    if md5 is not None:
        data["md5"] = md5
    return {"key": key, "data": data}


def _zot(items: list[dict[str, Any]]) -> FakeZot:
    """A library with one collection and the paper item's three children."""
    return FakeZot(
        items=items,
        collections=[collection("C1", "yeast")],
        children={
            "ITEM1": [
                _pdf_child("ATT2", "Supplementary Information", None),
                {"key": "SNAP", "data": {"contentType": "text/html", "title": "Snap"}},
                _pdf_child("ATT1", "Full Text PDF", "md5main"),
            ]
        },
    )


def _read(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


def test_role_for_new_branches() -> None:
    assert _role_for("data/titers.xlsx") == "raw_data"
    assert _role_for("si/Table_S5.xls") == "si_data"
    assert _role_for("thesis.pdf") == "paper_pdf"
    assert _role_for("thesis.txt") == "paper_ocr"
    # Unchanged legacy behavior.
    assert _role_for("paper.pdf") == "paper_pdf"
    assert _role_for("si/si_data/a.xlsx") == "si_data"
    assert _role_for("images/a.jpg") == "ocr_image"


def test_role_for_mineru_sidecars_under_si_are_ocr_roles() -> None:
    assert _role_for("si/si1_middle.json") == "ocr_layout"
    assert _role_for("si/si1_content_list.json") == "ocr_layout"
    assert _role_for("si/si12_middle.json") == "ocr_layout"
    assert _role_for("si/images/ab12.jpg") == "ocr_image"
    assert _role_for("si/images/ab12.png") == "ocr_image"
    # Released files keep their data role: si/si_data/ wins over every OCR rule, and
    # a loose SI table that is not a sidecar is still si_data.
    assert _role_for("si/si_data/images/plate.png") == "si_data"
    assert _role_for("si/si_data/run_middle.json") == "si_data"
    assert _role_for("si/Table_S2.json") == "si_data"
    # Anchored to what _run_mineru.py writes: images/ directly beside the PDF and
    # <stem>_middle.json for a PDF stem (paper, si*), never a substring match.
    assert _role_for("si/Figure_S1_images/a.png") == "si_data"
    assert _role_for("si/Table_middle.json") == "si_data"
    assert _role_for("si/Table_content_list.json") == "si_data"
    assert _role_for("si/si1.pdf") == "si_pdf"
    assert _role_for("si/si1.md") == "si_ocr"


def test_role_for_per_pdf_si_figures_and_ocr_provenance() -> None:
    """Issue #579 layout: ``si/images/<si stem>/<file>`` is an OCR figure, the flat
    pre-fix ``si/images/<file>`` still is, and ``<stem>_ocr_provenance.json`` (paper or
    SI) is ``ocr_provenance``. A root PDF's ``images/<stem>/<file>`` is a figure; one
    directory deeper or a non-``si*`` subdirectory under ``si/images/`` is not.
    """
    assert _role_for("si/images/si1/ab12.jpg") == "ocr_image"
    assert _role_for("si/images/si12/ab12.png") == "ocr_image"
    assert _role_for("si/images/ab12.jpeg") == "ocr_image"
    assert _role_for("paper_ocr_provenance.json") == "ocr_provenance"
    assert _role_for("si/si10_ocr_provenance.json") == "ocr_provenance"
    assert _role_for("si/images/Figure/a.png") == "si_data"
    assert _role_for("si/images/si1/deeper/a.png") == "si_data"
    assert _role_for("si/images/si1/a.gif") == "si_data"
    assert _role_for("si/Table_ocr_provenance.json") == "si_data"
    assert _role_for("images/SOM/a.jpg") == "ocr_image"
    assert _role_for("images/SOM/deeper/a.jpg") == "other"


def test_captured_key_full_role_table(tmp_path: Path) -> None:
    """Every file MinerU and capture leave in an avsec-shaped key, with its role and
    default source, in ``build_manifest``'s sorted walk order.
    """
    key = tmp_path / "avsecEffectiveGeneExpression2021"
    names = [
        "images/3a34.jpg",
        "paper.md",
        "paper.pdf",
        "paper_content_list.json",
        "paper_middle.json",
        "si/images/77aa.jpg",
        "si/si1.md",
        "si/si1.pdf",
        "si/si1_content_list.json",
        "si/si1_middle.json",
    ]
    for name in names:
        (key / name).parent.mkdir(parents=True, exist_ok=True)
        (key / name).write_bytes(name.encode())

    manifest = build_manifest(
        key, citation_key=key.name, created_at=_FROZEN, provenance_complete=False
    )

    assert [(r.path, r.role, r.source) for r in manifest.files] == [
        ("images/3a34.jpg", "ocr_image", None),
        ("paper.md", "paper_ocr", "mineru-ocr"),
        ("paper.pdf", "paper_pdf", None),
        ("paper_content_list.json", "ocr_layout", None),
        ("paper_middle.json", "ocr_layout", None),
        ("si/images/77aa.jpg", "ocr_image", None),
        ("si/si1.md", "si_ocr", "mineru-ocr"),
        ("si/si1.pdf", "si_pdf", None),
        ("si/si1_content_list.json", "ocr_layout", None),
        ("si/si1_middle.json", "ocr_layout", None),
    ]


def test_per_pdf_si_figures_and_attached_ocr_provenance(tmp_path: Path) -> None:
    """A key OCR'd after issue #579 with two SI PDFs: both PDFs' figures are listed
    as ``ocr_image`` under their own directories, each ``_ocr_provenance.json`` is
    ``ocr_provenance``, and its ``ProcessingRecord`` is attached to exactly the
    markdown beside it (``paper.md`` and ``si/si2.md``); ``si/si1.md`` has no record
    file and keeps ``processing`` None.
    """
    key = tmp_path / "leeMappingCellularResponse2014"
    paper_record = ProcessingRecord(
        processor="torchcell.literature.ocr.ocr_pdf",
        tool="mineru",
        version="2.7.6",
        params={"dpi": 350, "images_dir": "images"},
        input_sha256=["aa" * 32],
    )
    si2_record = ProcessingRecord(
        processor="torchcell.literature.ocr.ocr_pdf",
        tool="mineru",
        version="2.7.6",
        params={"dpi": 200, "images_dir": "images/si2"},
        input_sha256=["bb" * 32],
    )
    files = {
        "images/3a34.jpg": "x",
        "paper.md": "x",
        "paper.pdf": "x",
        "paper_ocr_provenance.json": paper_record.model_dump_json(),
        "si/images/si1/11aa.jpg": "x",
        "si/images/si1/11bb.png": "x",
        "si/images/si2/22aa.jpg": "x",
        "si/si1.md": "x",
        "si/si1.pdf": "x",
        "si/si2.md": "x",
        "si/si2.pdf": "x",
        "si/si2_ocr_provenance.json": si2_record.model_dump_json(),
    }
    for name, text in files.items():
        (key / name).parent.mkdir(parents=True, exist_ok=True)
        (key / name).write_text(text)

    manifest = build_manifest(
        key, citation_key=key.name, created_at=_FROZEN, provenance_complete=False
    )

    assert [(r.path, r.role, r.source) for r in manifest.files] == [
        ("images/3a34.jpg", "ocr_image", None),
        ("paper.md", "paper_ocr", "mineru-ocr"),
        ("paper.pdf", "paper_pdf", None),
        ("paper_ocr_provenance.json", "ocr_provenance", None),
        ("si/images/si1/11aa.jpg", "ocr_image", None),
        ("si/images/si1/11bb.png", "ocr_image", None),
        ("si/images/si2/22aa.jpg", "ocr_image", None),
        ("si/si1.md", "si_ocr", "mineru-ocr"),
        ("si/si1.pdf", "si_pdf", None),
        ("si/si2.md", "si_ocr", "mineru-ocr"),
        ("si/si2.pdf", "si_pdf", None),
        ("si/si2_ocr_provenance.json", "ocr_provenance", None),
    ]
    processing = {r.path: r.processing for r in manifest.files if r.processing}
    assert processing == {"paper.md": paper_record, "si/si2.md": si2_record}
    reloaded = Manifest.model_validate_json(manifest.model_dump_json())
    assert reloaded.files[1].processing == paper_record


def test_backfill_offline_hashes_and_roundtrips(tmp_path: Path) -> None:
    paper = _make_paper_key(tmp_path)
    _make_data_key(tmp_path)

    report = backfill_mirror(tmp_path, use_zotero=False)

    assert report.used_zotero is False
    assert report.offline == 2 and report.enriched == 0 and report.skipped == 0

    # Every directory now has a manifest that round-trips.
    manifest_path = paper / MANIFEST_FILENAME
    assert manifest_path.is_file()
    manifest = Manifest.model_validate_json(manifest_path.read_text())
    assert manifest.citation_key == "fakePaperKey2020"
    assert manifest.provenance_complete is False
    assert manifest.doi is None and manifest.zotero_item_key is None

    # Every stored sha256 matches the bytes on disk.
    for record in manifest.files:
        on_disk = sha256_file(paper / record.path)
        assert record.sha256 == on_disk

    roles = {r.path: r.role for r in manifest.files}
    assert roles["paper.pdf"] == "paper_pdf"
    assert roles["paper.md"] == "paper_ocr"
    assert roles["images/fig1.jpg"] == "ocr_image"


def test_backfill_data_only_key_uses_raw_data(tmp_path: Path) -> None:
    data = _make_data_key(tmp_path)
    backfill_mirror(tmp_path, use_zotero=False)
    manifest = Manifest.model_validate_json((data / MANIFEST_FILENAME).read_text())
    assert manifest.provenance_complete is False
    assert [r.role for r in manifest.files] == ["raw_data"]
    assert manifest.doi is None and manifest.title is None


def test_backfill_is_idempotent_and_force_rewrites(tmp_path: Path) -> None:
    _make_paper_key(tmp_path)

    first = backfill_mirror(tmp_path, use_zotero=False)
    assert first.offline == 1

    # Second run skips the already-backfilled directory.
    second = backfill_mirror(tmp_path, use_zotero=False)
    assert second.skipped == 1 and second.offline == 0

    # force rewrites it.
    forced = backfill_mirror(tmp_path, use_zotero=False, force=True)
    assert forced.offline == 1 and forced.skipped == 0


def test_backfill_dry_run_writes_nothing(tmp_path: Path) -> None:
    paper = _make_paper_key(tmp_path)
    report = backfill_key(paper, dry_run=True)
    assert report.mode == "offline"
    assert not (paper / MANIFEST_FILENAME).exists()


# The literal digests are the bytes the fixtures write, not copied output; a wrong literal
# fails at import, before any test that compares a written manifest against them.
assert hashlib.sha256(_PDF).hexdigest() == _SHA_PDF
assert hashlib.sha256(_MD).hexdigest() == _SHA_MD
assert hashlib.sha256(_SI).hexdigest() == _SHA_SI
assert hashlib.sha256(_JPG).hexdigest() == _SHA_JPG


def test_library_root_is_the_torchcell_library_subdirectory() -> None:
    assert library_root("/x/y") == Path("/x/y/torchcell-library")
    assert library_root(Path("rel")) == Path("rel/torchcell-library")


def test_offline_manifest_is_written_exactly(tmp_path: Path, frozen: None) -> None:
    """The whole offline ``manifest.json``: files in sorted order, OCR tagged mineru."""
    paper = _make_exact_key(tmp_path)
    result = backfill_key(paper)
    assert result.model_dump() == {
        "citation_key": "fakePaperKey2020",
        "mode": "offline",
        "n_files": 4,
        "provenance_complete": False,
        "null_metadata": ["doi", "title", "zotero_item_key"],
        "doi": None,
    }
    assert _read(paper / MANIFEST_FILENAME) == _manifest(
        "fakePaperKey2020",
        [
            _record("images/fig1.jpg", "ocr_image", 13, _SHA_JPG),
            _record("paper.md", "paper_ocr", 33, _SHA_MD, "mineru-ocr"),
            _record("paper.pdf", "paper_pdf", 23, _SHA_PDF),
            _record("si/si1.pdf", "si_pdf", 16, _SHA_SI),
        ],
    )


def test_mineru_source_is_given_by_location_not_by_role(
    tmp_path: Path, frozen: None
) -> None:
    """``mineru-ocr`` marks only what ``ocr_artifact`` writes: ``paper.md`` beside
    ``paper.pdf`` and ``si/<stem>.md`` beside each SI PDF. A top-level born-digital
    ``thesis.txt`` keeps the ``paper_ocr`` role but no source, and a nested
    ``si/extra/notes.md`` (``si_ocr`` by role, but not where MinerU writes) has none.
    """
    key = tmp_path / "lopezThesis2024"
    (key / "si" / "extra").mkdir(parents=True)
    (key / "thesis.pdf").write_bytes(b"%PDF-1.5 thesis")
    (key / "thesis.txt").write_bytes(b"thesis text\n")
    (key / "si" / "si1.md").write_bytes(_MD)
    (key / "si" / "extra" / "notes.md").write_bytes(_MD)
    backfill_key(key)
    assert _read(key / MANIFEST_FILENAME)["files"] == [
        _record("si/extra/notes.md", "si_ocr", 33, _SHA_MD),
        _record("si/si1.md", "si_ocr", 33, _SHA_MD, "mineru-ocr"),
        _record("thesis.pdf", "paper_pdf", 15, _SHA_THESIS_PDF),
        _record("thesis.txt", "paper_ocr", 12, _SHA_THESIS_TXT),
    ]


def test_enriched_manifest_carries_zotero_metadata_and_attachment_sources(
    tmp_path: Path, frozen: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zotero mode end to end: the item matched by native ``citationKey``."""
    paper = _make_exact_key(tmp_path)
    item = _item(
        "ITEM1",
        {
            "citationKey": "fakePaperKey2020",
            "DOI": "10.1000/fake",
            "title": "A fake paper",
            "collections": ["C1", "C9"],
        },
    )
    zot = _zot([item])
    library = make_library(monkeypatch, zot)
    monkeypatch.setattr(ZoteroLibrary, "from_env", classmethod(lambda cls: library))

    report = backfill_mirror(tmp_path)

    assert report.model_dump() == {
        "root": str(tmp_path),
        "used_zotero": True,
        "results": [
            {
                "citation_key": "fakePaperKey2020",
                "mode": "enriched",
                "n_files": 4,
                "provenance_complete": True,
                "null_metadata": [],
                "doi": "10.1000/fake",
            }
        ],
    }
    assert (report.enriched, report.offline, report.skipped) == (1, 0, 0)
    assert _read(paper / MANIFEST_FILENAME) == _manifest(
        "fakePaperKey2020",
        [
            _record("images/fig1.jpg", "ocr_image", 13, _SHA_JPG),
            _record("paper.md", "paper_ocr", 33, _SHA_MD, "mineru-ocr"),
            _record(
                "paper.pdf",
                "paper_pdf",
                23,
                _SHA_PDF,
                "zotero:attachment:ATT1",
                "md5main",
            ),
            _record("si/si1.pdf", "si_pdf", 16, _SHA_SI, "zotero:attachment:ATT2"),
        ],
        doi="10.1000/fake",
        title="A fake paper",
        library_id="1",
        zotero_item_key="ITEM1",
        collections=["yeast", "C9"],
        provenance_complete=True,
    )
    # the paginated top-level index scan (issue #607), then the attachments, then
    # the collection names: three paged reads (each through ``everything``, issue
    # #563), no write method exists to be called
    assert zot.calls == [
        ("top",),
        ("everything",),
        ("children", "ITEM1"),
        ("everything",),
        ("collections",),
        ("everything",),
    ]


def test_enriched_manifest_without_doi_or_title_is_incomplete(
    tmp_path: Path, frozen: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Zotero match with an empty DOI and no title is enriched (item key, sources)
    but ``provenance_complete=False``, in the result and in the written manifest, as
    backfill.py's ``_METADATA_FIELDS`` comment says a missing doi or title marks.
    """
    key = tmp_path / "noDoiKey2019"
    key.mkdir()
    (key / "paper.pdf").write_bytes(_PDF)
    item = _item("ITEM1", {"extra": "Citation Key: noDoiKey2019", "DOI": ""})
    library = make_library(monkeypatch, _zot([item]))
    index = build_citation_index(library)
    result = backfill_key(key, citation_index=index, lib=library)
    assert result.model_dump() == {
        "citation_key": "noDoiKey2019",
        "mode": "enriched",
        "n_files": 1,
        "provenance_complete": False,
        "null_metadata": ["doi", "title"],
        "doi": None,
    }
    written = _read(key / MANIFEST_FILENAME)
    assert (written["doi"], written["title"], written["collections"]) == (
        None,
        None,
        [],
    )
    assert (written["zotero_item_key"], written["provenance_complete"]) == (
        "ITEM1",
        False,
    )


@pytest.mark.parametrize(
    ("data", "complete", "null_metadata"),
    [
        ({"DOI": "10.1000/fake", "title": ""}, False, ["title"]),
        ({"DOI": "", "title": "A fake paper"}, False, ["doi"]),
        ({"DOI": "10.1000/fake", "title": "A fake paper"}, True, []),
    ],
)
def test_enriched_completeness_needs_both_doi_and_title(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data: dict[str, str],
    complete: bool,
    null_metadata: list[str],
) -> None:
    """Either field missing alone is enough to mark the manifest incomplete."""
    key = tmp_path / "halfKey2019"
    key.mkdir()
    (key / "paper.pdf").write_bytes(_PDF)
    item = _item("ITEM1", {"citationKey": "halfKey2019", **data})
    library = make_library(monkeypatch, _zot([item]))
    result = backfill_key(
        key, citation_index=build_citation_index(library), lib=library
    )
    assert (result.mode, result.provenance_complete, result.null_metadata) == (
        "enriched",
        complete,
        null_metadata,
    )
    assert _read(key / MANIFEST_FILENAME)["provenance_complete"] is complete


def test_a_matched_item_without_a_library_is_written_offline(
    tmp_path: Path, frozen: None
) -> None:
    """Enrichment needs both the index hit AND a connected library (line 143)."""
    paper = _make_exact_key(tmp_path)
    index = {"fakePaperKey2020": _item("ITEM1", {"DOI": "10.1000/fake"})}
    result = backfill_key(paper, citation_index=index, lib=None)
    assert (result.mode, result.provenance_complete, result.doi) == (
        "offline",
        False,
        None,
    )
    assert _read(paper / MANIFEST_FILENAME)["zotero_item_key"] is None


def test_citation_index_keys_by_native_then_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native ``citationKey`` and an ``extra`` ``Citation Key:`` line both index."""
    native = _item("A", {"citationKey": "nativeKey2020", "title": "first"})
    extra = _item("C", {"extra": "tex.x: 1\nCitation Key: extraKey2021"})
    library = make_library(monkeypatch, FakeZot(items=[native, extra]))
    index = build_citation_index(library)
    assert {key: item["key"] for key, item in index.items()} == {
        "nativeKey2020": "A",
        "extraKey2021": "C",
    }


def test_citation_index_refuses_items_sharing_a_citation_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two items resolving to one key (here one native, one via ``extra``) raise
    ``DuplicateCitationKeyError`` naming the key and both item keys in scan order;
    picking either would attach one paper's metadata to the other's directory.
    """
    first = _item("A", {"citationKey": "dupKey2020", "title": "first"})
    other = _item("C", {"citationKey": "soloKey2021"})
    second = _item("B", {"extra": "Citation Key: dupKey2020", "title": "second"})
    library = make_library(monkeypatch, FakeZot(items=[first, other, second]))
    with pytest.raises(
        bf.DuplicateCitationKeyError,
        match=r"^Zotero items share a citation key \(dupKey2020: A, B\)$",
    ):
        build_citation_index(library)


def _child(key: str, item_type: str, parent: str, title: str) -> dict[str, Any]:
    """A Zotero child item: no citation key, a ``parentItem`` unless standalone."""
    data: dict[str, Any] = {"title": title}
    if parent:
        data["parentItem"] = parent
    return _item(key, data, item_type)


def test_citation_index_reads_top_level_items_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #607: the index is built from ``top()`` (paginated by ``everything``),
    so a paper's attachment, note and annotation children never reach it. Before the
    fix every item was scanned and children collided under generated keys such as
    ``unknownFullTextPDFXXXX``.
    """
    paper = _item("P1", {"citationKey": "paperKey2020", "title": "A paper"})
    book = _item("P2", {"extra": "Citation Key: bookKey2021"}, "book")
    zot = FakeZot(
        items=[
            paper,
            _child("ATT1", "attachment", "P1", "Full Text PDF"),
            _child("N1", "note", "P1", ""),
            _child("AN1", "annotation", "ATT1", ""),
            book,
        ]
    )
    library = make_library(monkeypatch, zot)
    index, duplicates = bf.citation_index_with_duplicates(library)
    assert {key: item["key"] for key, item in index.items()} == {
        "paperKey2020": "P1",
        "bookKey2021": "P2",
    }
    assert duplicates == {}
    assert zot.calls == [("top",), ("everything",)]


@pytest.mark.parametrize("item_type", ["attachment", "note", "annotation"])
def test_citation_index_skips_a_standalone_child_type(
    monkeypatch: pytest.MonkeyPatch, item_type: str
) -> None:
    """A standalone attachment or note is top-level in Zotero, so ``top()`` returns
    it; each of the three child types is skipped by ``itemType``. Two of them with
    one title would otherwise share a generated key.
    """
    paper = _item("P1", {"citationKey": "paperKey2020"})
    first = _child("S1", item_type, "", "Full Text PDF")
    second = _child("S2", item_type, "", "Full Text PDF")
    library = make_library(monkeypatch, FakeZot(items=[first, paper, second]))
    index, duplicates = bf.citation_index_with_duplicates(library)
    assert {key: item["key"] for key, item in index.items()} == {"paperKey2020": "P1"}
    assert duplicates == {}


def test_citation_index_with_duplicates_leaves_a_shared_key_out_of_the_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The companion of ``build_citation_index`` returns the duplicates instead of
    raising: a shared key maps to its item keys in scan order and is absent from the
    index, so no caller can pick one of the two papers.
    """
    items = [
        _item("A", {"citationKey": "dupKey2020"}),
        _item("C", {"citationKey": "soloKey2021"}),
        _item("B", {"extra": "Citation Key: dupKey2020"}),
        _item("E", {"citationKey": "aDup2019"}),
        _item("D", {"citationKey": "aDup2019"}),
    ]
    library = make_library(monkeypatch, FakeZot(items=items))
    index, duplicates = bf.citation_index_with_duplicates(library)
    assert {key: item["key"] for key, item in index.items()} == {"soloKey2021": "C"}
    assert duplicates == {"dupKey2020": ["A", "B"], "aDup2019": ["E", "D"]}
    assert bf.describe_duplicates(duplicates) == "aDup2019: E, D; dupKey2020: A, B"


def test_backfill_mirror_refuses_any_duplicate_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The backfill keeps refusing on every duplicate, even one no directory under
    the root is named after: a mirror directory can be any citation key.
    """
    paper = _make_exact_key(tmp_path)
    items = [
        _item("ITEM1", {"citationKey": "fakePaperKey2020"}),
        _item("X1", {"citationKey": "otherKey2022"}),
        _item("X2", {"citationKey": "otherKey2022"}),
    ]
    library = make_library(monkeypatch, _zot(items))
    monkeypatch.setattr(ZoteroLibrary, "from_env", classmethod(lambda cls: library))
    with pytest.raises(bf.DuplicateCitationKeyError) as refused:
        backfill_mirror(tmp_path)
    assert str(refused.value) == (
        "Zotero items share a citation key (otherKey2022: X1, X2)"
    )
    assert not (paper / MANIFEST_FILENAME).exists()


def test_existing_corrupt_manifest_raises_with_its_path(tmp_path: Path) -> None:
    """Without ``force`` an existing manifest is read; one that does not parse raises
    ``CorruptManifestError`` naming its path and is left in place (``force`` is the
    way to rebuild it).
    """
    paper = _make_exact_key(tmp_path)
    (paper / MANIFEST_FILENAME).write_text("not json")
    with pytest.raises(
        bf.CorruptManifestError,
        match=rf"^Existing manifest {re.escape(str(paper / MANIFEST_FILENAME))} is not a valid "
        r"Manifest: 1 validation error for Manifest\n",
    ):
        backfill_key(paper)
    assert (paper / MANIFEST_FILENAME).read_text() == "not json"


def test_skipped_key_reports_what_its_existing_manifest_records(
    tmp_path: Path, frozen: None
) -> None:
    """A skipped key's result carries the existing manifest's file count, completeness
    and null metadata, not the model defaults.
    """
    paper = _make_exact_key(tmp_path)
    backfill_key(paper)
    result = backfill_key(paper)
    assert result.model_dump() == {
        "citation_key": "fakePaperKey2020",
        "mode": "skipped",
        "n_files": 4,
        "provenance_complete": False,
        "null_metadata": ["doi", "title", "zotero_item_key"],
        "doi": None,
    }


def test_force_rehashes_changed_bytes_and_dry_run_leaves_the_old_manifest(
    tmp_path: Path, frozen: None
) -> None:
    """Force rebuilds from the bytes on disk now; force + dry_run writes nothing."""
    paper = _make_exact_key(tmp_path)
    backfill_key(paper)
    (paper / "paper.pdf").write_bytes(b"%PDF-1.5 edited")
    edited = "8c8c6cc6d27df58ad754e3cbc54f4321fa15c9016fa51edd58eb32e7293d6631"

    dry = backfill_key(paper, force=True, dry_run=True)
    assert dry.mode == "offline"
    stale = _read(paper / MANIFEST_FILENAME)["files"][2]
    assert (stale["path"], stale["bytes"], stale["sha256"]) == (
        "paper.pdf",
        23,
        _SHA_PDF,
    )

    backfill_key(paper, force=True)
    fresh = _read(paper / MANIFEST_FILENAME)["files"][2]
    assert (fresh["path"], fresh["bytes"], fresh["sha256"]) == ("paper.pdf", 15, edited)


def test_mirror_scan_visits_only_citation_key_directories(
    tmp_path: Path, frozen: None
) -> None:
    """Only citation-key directories get manifests: ``_bib`` (leading underscore), an
    empty directory and a directory holding only empty subdirectories are not keys,
    get no manifest and are absent from the report. Loose root files are ignored.
    """
    _make_exact_key(tmp_path)
    (tmp_path / "_bib").mkdir()
    (tmp_path / "_bib" / "paper.bib").write_bytes(b"@article{a,}\n")
    (tmp_path / "emptyKey2000").mkdir()
    (tmp_path / "hollowKey2001" / "si").mkdir(parents=True)
    (tmp_path / "README.txt").write_text("not a key")

    report = backfill_mirror(tmp_path, use_zotero=False)

    assert [(r.citation_key, r.mode, r.n_files) for r in report.results] == [
        ("fakePaperKey2020", "offline", 4)
    ]
    assert sorted(p.name for p in tmp_path.rglob(MANIFEST_FILENAME)) == [
        MANIFEST_FILENAME
    ]
    assert (tmp_path / "fakePaperKey2020" / MANIFEST_FILENAME).is_file()
    assert sorted(p.name for p in (tmp_path / "_bib").iterdir()) == ["paper.bib"]
    assert list((tmp_path / "emptyKey2000").iterdir()) == []


def test_zotero_mode_without_credentials_refuses_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``use_zotero=True`` needs ZOTERO_LIBRARY_ID; missing, it raises and writes none."""
    paper = _make_exact_key(tmp_path)
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    monkeypatch.setenv("ZOTERO_API_KEY", "k")
    with pytest.raises(KeyError, match="^'ZOTERO_LIBRARY_ID'$"):
        backfill_mirror(tmp_path)
    assert not (paper / MANIFEST_FILENAME).exists()


def test_report_counts_each_mode() -> None:
    report = BackfillReport(
        root="r",
        used_zotero=True,
        results=[
            KeyBackfillResult(citation_key="a", mode="enriched"),
            KeyBackfillResult(citation_key="b", mode="offline"),
            KeyBackfillResult(citation_key="c", mode="offline"),
            KeyBackfillResult(citation_key="d", mode="skipped"),
        ],
    )
    assert (report.enriched, report.offline, report.skipped) == (1, 2, 1)


def _run_main(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(bf, "load_dotenv", lambda: None)
    monkeypatch.setattr(sys, "argv", ["backfill", *args])
    bf.main()


def _done(caplog: pytest.LogCaptureFixture) -> str:
    (line,) = [r.getMessage() for r in caplog.records if "Backfill done" in r.message]
    return line


def test_main_offline_run_then_rerun_then_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The CLI summary: written count = enriched + offline, or total - skipped."""
    caplog.set_level(logging.INFO, logger="torchcell.literature.backfill")
    _make_exact_key(tmp_path)
    _make_data_key(tmp_path)
    _run_main(monkeypatch, "--root", str(tmp_path), "--no-zotero")
    assert _done(caplog) == (
        "Backfill done: 2 dirs (enriched=0 offline=2 skipped=0); "
        "2 manifests written this run"
    )
    caplog.clear()
    _run_main(monkeypatch, "--root", str(tmp_path), "--no-zotero")
    assert _done(caplog) == (
        "Backfill done: 2 dirs (enriched=0 offline=0 skipped=2); "
        "0 manifests written this run"
    )
    caplog.clear()
    _run_main(monkeypatch, "--root", str(tmp_path), "--no-zotero", "--force")
    assert _done(caplog) == (
        "Backfill done: 2 dirs (enriched=0 offline=2 skipped=0); "
        "2 manifests written this run"
    )


def test_main_defaults_to_the_data_root_mirror_and_dry_run_reports_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No ``--root``: ``$DATA_ROOT/torchcell-library``; ``--dry-run`` writes nothing."""
    caplog.set_level(logging.INFO, logger="torchcell.literature.backfill")
    mirror = tmp_path / "torchcell-library"
    paper = _make_exact_key(mirror)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    _run_main(monkeypatch, "--no-zotero", "--dry-run")
    assert _done(caplog) == (
        "Backfill done: 1 dirs (enriched=0 offline=1 skipped=0); "
        "0 manifests written this run"
    )
    assert not (paper / MANIFEST_FILENAME).exists()
