# tests/torchcell/literature/test_capture.py
# [[tests.torchcell.literature.test_capture]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_capture.py
"""``capture_by_doi`` end to end with Zotero, SI fetching and OCR all faked.

Fixture: a real ``ZoteroLibrary`` over the read-only ``FakeZot`` (no write methods, so
no Zotero write can even be attempted), holding one item (DOI ``10.1073/pnas.1``,
native citation key ``ohyaHighdimensional2005``, collections ``C1`` and an unknown
``CX``) whose children are, in listed order: the SI PDF, a non-PDF snapshot, and the
main PDF titled "Full Text PDF" (md5 ``m5main``). ``fetch_si_data`` and ``ocr_artifact``
are replaced at their import site in ``capture``: the SI fake writes
``si/si_data/T1.csv`` and returns its source URL; the OCR fake writes ``paper.md``.
``data_root`` is ``tmp_path``. sha256 values are recomputed with ``hashlib``.

Contract pinned: the main PDF is ``paper.pdf`` even when listed after the SI; manifest
sources are ``zotero:attachment:<key>`` for the PDFs, the URL for SI data and
``mineru-ocr`` for the OCR markdown; an unknown collection key is kept as the key; the
manifest is written last and lists every file with its role and hash.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

import torchcell.literature.capture as capture
from tests.torchcell.literature._fake_zotero import FakeZot, collection, make_library

ITEM = {
    "key": "ITEM1",
    "data": {
        "DOI": "10.1073/pnas.1",
        "title": "High-dimensional phenotyping",
        "citationKey": "ohyaHighdimensional2005",
        "collections": ["C1", "CX"],
    },
}
CHILDREN = [
    {
        "key": "ATT2",
        "data": {
            "contentType": "application/pdf",
            "title": "Supplementary Information",
            "parentItem": "ITEM1",
        },
    },
    {
        "key": "SNAP",
        "data": {
            "contentType": "text/html",
            "title": "Snapshot",
            "parentItem": "ITEM1",
        },
    },
    {
        "key": "ATT1",
        "data": {
            "contentType": "application/pdf",
            "title": "Full Text PDF",
            "md5": "m5main",
            "parentItem": "ITEM1",
        },
    },
]
FILES = {"ATT1": b"%PDF main", "ATT2": b"%PDF si"}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def lib(monkeypatch: pytest.MonkeyPatch) -> Any:
    zot = FakeZot(
        items=[ITEM],
        collections=[collection("C1", "database")],
        children={"ITEM1": CHILDREN},
        files=FILES,
    )
    return make_library(monkeypatch, zot), zot


def test_capture_by_doi_writes_pdfs_si_ocr_and_manifest(
    lib: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    library, zot = lib
    si_calls: list[tuple[Any, ...]] = []
    ocr_calls: list[Path] = []

    def fake_fetch(
        artifact_dir: Path, *, dryad_doi: str | None, extra_urls: list[str] | None
    ) -> list[tuple[Path, str]]:
        si_calls.append((artifact_dir, dryad_doi, extra_urls))
        dest = artifact_dir / "si" / "si_data" / "T1.csv"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"a,b\n")
        return [(dest, "https://datadryad.org/api/v2/files/1/download")]

    def fake_ocr(artifact_dir: Path) -> None:
        ocr_calls.append(artifact_dir)
        (artifact_dir / "paper.md").write_text("# paper\n")

    monkeypatch.setattr(capture, "fetch_si_data", fake_fetch)
    monkeypatch.setattr(capture, "ocr_artifact", fake_ocr)

    out = capture.capture_by_doi(
        library,
        "10.1073/PNAS.1",
        dryad_doi="10.5061/dryad.x",
        si_expected=["Table S1"],
        data_root=tmp_path,
    )
    art = tmp_path / "torchcell-library" / "ohyaHighdimensional2005"
    assert out == art
    assert si_calls == [(art, "10.5061/dryad.x", None)]
    assert ocr_calls == [art]
    assert (art / "paper.pdf").read_bytes() == b"%PDF main"
    assert (art / "si" / "si1.pdf").read_bytes() == b"%PDF si"
    manifest = json.loads((art / "manifest.json").read_text())
    manifest.pop("created_at")
    assert manifest == {
        "version": 1,
        "citation_key": "ohyaHighdimensional2005",
        "doi": "10.1073/pnas.1",
        "title": "High-dimensional phenotyping",
        "library_id": "1",
        "zotero_item_key": "ITEM1",
        "collections": ["database", "CX"],
        "files": [
            {
                "path": "paper.md",
                "role": "paper_ocr",
                "bytes": 8,
                "sha256": _sha(b"# paper\n"),
                "source": "mineru-ocr",
                "zotero_md5": None,
                "retrieval": None,
                "processing": None,
            },
            {
                "path": "paper.pdf",
                "role": "paper_pdf",
                "bytes": 9,
                "sha256": _sha(b"%PDF main"),
                "source": "zotero:attachment:ATT1",
                "zotero_md5": "m5main",
                "retrieval": None,
                "processing": None,
            },
            {
                "path": "si/si1.pdf",
                "role": "si_pdf",
                "bytes": 7,
                "sha256": _sha(b"%PDF si"),
                "source": "zotero:attachment:ATT2",
                "zotero_md5": None,
                "retrieval": None,
                "processing": None,
            },
            {
                "path": "si/si_data/T1.csv",
                "role": "si_data",
                "bytes": 4,
                "sha256": _sha(b"a,b\n"),
                "source": "https://datadryad.org/api/v2/files/1/download",
                "zotero_md5": None,
                "retrieval": None,
                "processing": None,
            },
        ],
        "si_data_sources": ["https://datadryad.org/api/v2/files/1/download"],
        "si_expected": ["Table S1"],
        "provenance_complete": True,
    }
    # only reads, and both attachment downloads by key
    assert ("file", "ATT1") in zot.calls and ("file", "ATT2") in zot.calls
    assert ("file", "SNAP") not in zot.calls


def test_capture_without_si_or_ocr_calls_neither(
    lib: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    library, _ = lib
    called: list[str] = []
    monkeypatch.setattr(capture, "fetch_si_data", lambda *a, **k: called.append("si"))
    monkeypatch.setattr(capture, "ocr_artifact", lambda *a, **k: called.append("ocr"))
    art = capture.capture_by_doi(
        library, "10.1073/pnas.1", do_ocr=False, data_root=tmp_path
    )
    assert called == []
    manifest = json.loads((art / "manifest.json").read_text())
    assert [f["path"] for f in manifest["files"]] == ["paper.pdf", "si/si1.pdf"]
    assert manifest["si_data_sources"] == []
    assert manifest["si_expected"] == []


def test_capture_extra_urls_alone_trigger_si_fetching(
    lib: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    library, _ = lib
    calls: list[tuple[Any, ...]] = []

    def fake_fetch(
        artifact_dir: Path, *, dryad_doi: str | None, extra_urls: list[str] | None
    ) -> list[tuple[Path, str]]:
        calls.append((dryad_doi, extra_urls))
        return []

    monkeypatch.setattr(capture, "fetch_si_data", fake_fetch)
    art = capture.capture_by_doi(
        library,
        "10.1073/pnas.1",
        extra_si_urls=["https://h/x.csv"],
        do_ocr=False,
        data_root=tmp_path,
    )
    assert calls == [(None, ["https://h/x.csv"])]
    manifest = json.loads((art / "manifest.json").read_text())
    assert manifest["si_data_sources"] == []
    assert [f["path"] for f in manifest["files"]] == ["paper.pdf", "si/si1.pdf"]


def test_capture_refuses_a_doi_not_in_the_library(lib: Any, tmp_path: Path) -> None:
    library, zot = lib
    with pytest.raises(
        ValueError, match=re.escape("No Zotero item with DOI 10.9/none in library 1")
    ):
        capture.capture_by_doi(library, "10.9/none", data_root=tmp_path)
    assert not (tmp_path / "torchcell-library").exists()


def test_collection_names_empty_keys_skip_the_listing(lib: Any) -> None:
    library, zot = lib
    assert capture._collection_names(library, []) == []
    assert zot.calls == []
