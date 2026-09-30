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
listed images/fig1.jpg, paper.md, paper.pdf, si/si1.pdf. Offline, only the OCR roles
get a default source (``"mineru-ocr"``). Enriched, the Zotero children are listed SI
first; ``pdf_attachments`` sorts the main article first, so ``paper.pdf`` gets ATT1 (md5
``"md5main"``) and ``si/si1.pdf`` gets ATT2 (no md5 in Zotero, so ``zotero_md5`` None);
the item's collection keys ``["C1", "C9"]`` resolve to ``["yeast", "C9"]`` (C9 is not in
the library, so its key is kept).

Findings pinned here (source lines in ``torchcell/literature/backfill.py`` unless
named): an enriched manifest is ``provenance_complete=True`` even when Zotero has no DOI
or title (line 114), although line 44 says their absence marks incomplete provenance; a
top-level ``thesis.txt`` (a born-digital extraction, per ``manifest._role_for``) is
tagged ``source="mineru-ocr"`` (``manifest.py`` line 261); every subdirectory of the
mirror, including a non-key one such as ``_bib``, is treated as a citation key (line
200); an existing ``manifest.json`` is skipped without being read, so a corrupt one
survives and the skipped result reports ``provenance_complete=True`` (lines 139-140);
two Zotero items resolving to one citation key keep the later one silently (line 95).
"""

import hashlib
import json
import logging
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
    _role_for,
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
    key: str, data: dict[str, Any]
) -> dict[str, Any]:  # a Zotero item as pyzotero returns it
    return {"key": key, "data": data}


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


def test_born_digital_thesis_text_is_tagged_as_mineru_ocr(
    tmp_path: Path, frozen: None
) -> None:
    """Finding: a top-level ``thesis.txt`` is a born-digital extraction by
    ``_role_for``'s own comment, yet ``build_manifest`` (manifest.py line 261) gives
    every ``paper_ocr`` file ``source="mineru-ocr"``, so the backfill records an OCR
    step that never ran. Pinned until the default source distinguishes the two.
    """
    key = tmp_path / "lopezThesis2024"
    key.mkdir()
    (key / "thesis.pdf").write_bytes(b"%PDF-1.5 thesis")
    (key / "thesis.txt").write_bytes(b"thesis text\n")
    backfill_key(key)
    assert _read(key / MANIFEST_FILENAME)["files"] == [
        _record("thesis.pdf", "paper_pdf", 15, _SHA_THESIS_PDF),
        _record("thesis.txt", "paper_ocr", 12, _SHA_THESIS_TXT, "mineru-ocr"),
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
    # the paginated index scan, then the attachments (line 103), then the
    # collection names (line 111): three reads, no write method exists to be called
    assert zot.calls == [
        ("items", None),
        ("everything",),
        ("children", "ITEM1"),
        ("everything",),
        ("collections",),
    ]


def test_enriched_manifest_without_doi_or_title_still_claims_complete(
    tmp_path: Path, frozen: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: backfill.py line 114 passes ``provenance_complete=True`` for every
    Zotero match, while line 44 says a missing doi/title/zotero_item_key marks
    incomplete provenance. An item with an empty DOI and no title is reported in
    ``null_metadata`` yet written as complete. Pinned until the two agree.
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
        "provenance_complete": True,
        "null_metadata": ["doi", "title"],
        "doi": None,
    }
    written = _read(key / MANIFEST_FILENAME)
    assert (written["doi"], written["title"], written["collections"]) == (
        None,
        None,
        [],
    )
    assert written["provenance_complete"] is True


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


def test_citation_index_keys_by_native_then_extra_and_last_duplicate_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding: two items resolving to one key collapse to the LATER item with no
    warning (dict comprehension, line 95). Pinned until duplicates are refused.
    """
    first = _item("A", {"citationKey": "dupKey2020", "title": "first"})
    second = _item("B", {"citationKey": "dupKey2020", "title": "second"})
    extra = _item("C", {"extra": "tex.x: 1\nCitation Key: extraKey2021"})
    library = make_library(monkeypatch, FakeZot(items=[first, extra, second]))
    index = build_citation_index(library)
    assert {key: item["key"] for key, item in index.items()} == {
        "dupKey2020": "B",
        "extraKey2021": "C",
    }


def test_existing_manifest_is_skipped_unread_even_when_corrupt(tmp_path: Path) -> None:
    """Finding: the skip test is ``exists()`` only (lines 139-140). A corrupt
    manifest is left in place, and the skipped result reports ``n_files=0`` and
    ``provenance_complete=True`` whatever the file says. Pinned until skip validates.
    """
    paper = _make_exact_key(tmp_path)
    (paper / MANIFEST_FILENAME).write_text("not json")
    result = backfill_key(paper)
    assert result.model_dump() == {
        "citation_key": "fakePaperKey2020",
        "mode": "skipped",
        "n_files": 0,
        "provenance_complete": True,
        "null_metadata": [],
        "doi": None,
    }
    assert (paper / MANIFEST_FILENAME).read_text() == "not json"


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


def test_mirror_scan_takes_every_subdirectory_as_a_key_and_ignores_files(
    tmp_path: Path, frozen: None
) -> None:
    """Finding: ``backfill_mirror`` treats EVERY subdirectory as a citation key (line
    200), so the mirror's non-key ``_bib`` store gets a manifest of its own; an empty
    directory gets a manifest listing no files. Loose files at the root are ignored.
    Pinned until the scan filters non-key directories.
    """
    _make_exact_key(tmp_path)
    (tmp_path / "_bib").mkdir()
    (tmp_path / "_bib" / "paper.bib").write_bytes(b"@article{a,}\n")
    (tmp_path / "emptyKey2000").mkdir()
    (tmp_path / "README.txt").write_text("not a key")

    report = backfill_mirror(tmp_path, use_zotero=False)

    assert [(r.citation_key, r.mode, r.n_files) for r in report.results] == [
        ("_bib", "offline", 1),
        ("emptyKey2000", "offline", 0),
        ("fakePaperKey2020", "offline", 4),
    ]
    assert _read(tmp_path / "_bib" / MANIFEST_FILENAME)["files"] == [
        _record(
            "paper.bib",
            "other",
            13,
            "e108d2a13ec55c7385439422efbadaf13de68bbe875ab7060fdb697ac1dbe823",
        )
    ]
    assert _read(tmp_path / "emptyKey2000" / MANIFEST_FILENAME) == _manifest(
        "emptyKey2000", []
    )
    assert not (tmp_path / "README.txt.json").exists()


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
