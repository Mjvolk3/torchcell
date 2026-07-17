"""Tests for torchcell.literature.sync (Zotero collection diff)."""

from pathlib import Path
from typing import Any

import pytest

from torchcell.literature.backfill import library_root
from torchcell.literature.sync import (
    SyncMode,
    plan_collection_sync,
    plan_database_sync,
    sync_collection,
    sync_collections,
    sync_database,
)


def _item(key: str, doi: str | None, citation_key: str) -> dict[str, Any]:
    """A minimal Zotero item dict as the sync path reads it."""
    data: dict[str, Any] = {"itemType": "journalArticle", "citationKey": citation_key}
    if doi is not None:
        data["DOI"] = doi
    return {"key": key, "data": data}


class _FakeZot:
    """Stand-in for the pyzotero client the sync path touches."""

    def __init__(self, items: list[dict[str, Any]], has_pdf: set[str]) -> None:
        self._items = items
        self._has_pdf = has_pdf

    def everything(self, x: Any) -> Any:
        return x

    def collection_items(self, _coll_key: str) -> list[dict[str, Any]]:
        return self._items

    def children(self, item_key: str) -> list[dict[str, Any]]:
        if item_key in self._has_pdf:
            return [{"data": {"contentType": "application/pdf", "filename": "p.pdf"}}]
        return []


class _FakeLib:
    """Stand-in for ZoteroLibrary exposing only what sync uses."""

    def __init__(self, items: list[dict[str, Any]], has_pdf: set[str]) -> None:
        self.zot = _FakeZot(items, has_pdf)

    def collection_key(self, _name: str) -> str:
        return "COLLKEY"

    def pdf_attachments(self, item_key: str) -> list[dict[str, Any]]:
        return self.zot.children(item_key)


def _mirror_key(root: Path, name: str) -> None:
    """Create a captured-looking mirror dir (dir + manifest)."""
    d = root / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text("{}")


def test_plan_classifies_present_capture_and_unsupported(tmp_path: Path) -> None:
    lib_root = tmp_path / "torchcell-library"
    lib_root.mkdir()
    _mirror_key(lib_root, "alreadyMirrored2020")

    items = [
        _item("K1", "10.1/present", "alreadyMirrored2020"),  # present
        _item("K2", "10.2/new", "newPaper2021"),  # would_capture (doi + pdf)
        _item("K3", None, "noDoi2022"),  # unsupported (no doi)
        _item("K4", "10.4/nopdf", "noPdf2023"),  # unsupported (no pdf)
    ]
    lib = _FakeLib(items, has_pdf={"K2"})

    report = plan_database_sync(lib, data_root=tmp_path)  # type: ignore[arg-type]

    modes = {r.citation_key: r.mode for r in report.results}
    assert modes["alreadyMirrored2020"] == SyncMode.PRESENT
    assert modes["newPaper2021"] == SyncMode.WOULD_CAPTURE
    assert modes["noDoi2022"] == SyncMode.UNSUPPORTED
    assert modes["noPdf2023"] == SyncMode.UNSUPPORTED
    assert report.n_collection_items == 4


def test_dry_run_captures_nothing(tmp_path: Path) -> None:
    lib_root = tmp_path / "torchcell-library"
    lib_root.mkdir()
    items = [_item("K2", "10.2/new", "newPaper2021")]
    lib = _FakeLib(items, has_pdf={"K2"})

    report = sync_database(lib, data_root=tmp_path, dry_run=True)  # type: ignore[arg-type]

    assert report.by_mode(SyncMode.WOULD_CAPTURE)[0].citation_key == "newPaper2021"
    # Nothing was written to the mirror.
    assert not (lib_root / "newPaper2021").exists()


def test_summary_tallies_modes(tmp_path: Path) -> None:
    lib_root = tmp_path / "torchcell-library"
    lib_root.mkdir()
    _mirror_key(lib_root, "alreadyMirrored2020")
    items = [
        _item("K1", "10.1/present", "alreadyMirrored2020"),
        _item("K3", None, "noDoi2022"),
    ]
    lib = _FakeLib(items, has_pdf=set())

    report = plan_database_sync(lib, data_root=tmp_path)  # type: ignore[arg-type]
    summary = report.summary()
    assert "present=1" in summary
    assert "unsupported=1" in summary


def test_plan_collection_sync_labels_the_named_collection(tmp_path: Path) -> None:
    lib_root = tmp_path / "torchcell-library"
    lib_root.mkdir()
    items = [_item("K2", "10.2/new", "newPaper2021")]
    lib = _FakeLib(items, has_pdf={"K2"})

    report = plan_collection_sync(lib, "paper", data_root=tmp_path)  # type: ignore[arg-type]

    # The report carries the collection it was asked for -- not a hardcoded name.
    assert report.collection == "paper"
    assert report.by_mode(SyncMode.WOULD_CAPTURE)[0].citation_key == "newPaper2021"


def test_sync_collections_returns_one_report_per_collection(tmp_path: Path) -> None:
    lib_root = tmp_path / "torchcell-library"
    lib_root.mkdir()
    items = [_item("K2", "10.2/new", "newPaper2021")]
    lib = _FakeLib(items, has_pdf={"K2"})

    reports = sync_collections(
        lib,  # type: ignore[arg-type]
        ["database", "paper"],
        data_root=tmp_path,
        dry_run=True,
    )

    assert [r.collection for r in reports] == ["database", "paper"]


def test_sync_collections_captures_once_then_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key in both collections is captured on the first pass, present on the second."""
    lib_root = tmp_path / "torchcell-library"
    lib_root.mkdir()
    items = [_item("K2", "10.2/new", "newPaper2021")]
    lib = _FakeLib(items, has_pdf={"K2"})
    doi_to_key = {"10.2/new": "newPaper2021"}

    def fake_capture(_lib: Any, doi: str, *, do_ocr: bool, data_root: Any) -> Path:
        key = doi_to_key[doi]
        _mirror_key(library_root(data_root), key)  # dir + manifest, as capture does
        return library_root(data_root) / key

    monkeypatch.setattr("torchcell.literature.sync.capture_by_doi", fake_capture)

    reports = sync_collections(
        lib,  # type: ignore[arg-type]
        ["database", "paper"],
        data_root=tmp_path,
    )

    assert reports[0].by_mode(SyncMode.CAPTURED)[0].citation_key == "newPaper2021"
    assert reports[1].by_mode(SyncMode.PRESENT)[0].citation_key == "newPaper2021"


def test_sync_collection_limit_bounds_captures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``limit`` caps captures per pass; the remainder is reported would_capture."""
    lib_root = tmp_path / "torchcell-library"
    lib_root.mkdir()
    items = [
        _item("K1", "10.1/a", "paperA2021"),
        _item("K2", "10.2/b", "paperB2021"),
        _item("K3", "10.3/c", "paperC2021"),
    ]
    lib = _FakeLib(items, has_pdf={"K1", "K2", "K3"})
    doi_to_key = {
        "10.1/a": "paperA2021",
        "10.2/b": "paperB2021",
        "10.3/c": "paperC2021",
    }

    def fake_capture(_lib: Any, doi: str, *, do_ocr: bool, data_root: Any) -> Path:
        key = doi_to_key[doi]
        _mirror_key(library_root(data_root), key)
        return library_root(data_root) / key

    monkeypatch.setattr("torchcell.literature.sync.capture_by_doi", fake_capture)

    report = sync_collection(lib, "paper", data_root=tmp_path, limit=2)  # type: ignore[arg-type]

    assert len(report.by_mode(SyncMode.CAPTURED)) == 2
    assert len(report.by_mode(SyncMode.WOULD_CAPTURE)) == 1
