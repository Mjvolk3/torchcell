"""Tests for torchcell.literature.backfill (offline manifest regularization)."""

from pathlib import Path

from torchcell.literature.backfill import backfill_key, backfill_mirror
from torchcell.literature.manifest import (
    MANIFEST_FILENAME,
    Manifest,
    _role_for,
    sha256_file,
)


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
