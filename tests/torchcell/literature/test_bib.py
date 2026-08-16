# tests/torchcell/literature/test_bib.py
"""Tests for the Zotero -> bib.bib generator.

The invariants that matter are the destructive ones: a sync must never drop an
entry the bibliography already had, and must never write an attachment stub.
"""

import pytest

from torchcell.literature.bib import (
    BibFileSyncReport,
    merge_bib_entries,
    plan_bib_sync,
    read_bib_entries,
    sync_bib_file,
    write_bib_entries,
)

LEGACY = """
@article{legacyOnlyEntry2019,
  author = {Legacy, Ada},
  title = {A Pre-Zotero Reference},
  journal = {Journal of Old Notes},
  year = {2019}
}

@article{sharedEntry2020,
  author = {Shared, Bob},
  title = {Stale Local Title},
  journal = {Journal of Drift},
  year = {2020}
}
"""


def _entry(key: str, title: str, **extra: str) -> dict[str, str]:
    """A minimal bibtexparser entry dict."""
    return {
        "ID": key,
        "ENTRYTYPE": "article",
        "author": "Zotero, Zed",
        "title": title,
        "year": "2020",
        **extra,
    }


@pytest.fixture
def legacy_bib(tmp_path):
    """A bib file holding one Zotero-unknown entry and one Zotero-known entry."""
    path = tmp_path / "bib.bib"
    path.write_text(LEGACY)
    return path


def test_read_write_round_trip_preserves_entries(legacy_bib, tmp_path):
    """Parsing then writing must not lose or alter entries."""
    before = read_bib_entries(legacy_bib)
    out = tmp_path / "out.bib"
    write_bib_entries(out, before)
    after = read_bib_entries(out)
    assert {e["ID"] for e in before} == {e["ID"] for e in after}
    assert len(after) == 2


def test_merge_preserves_entries_zotero_does_not_have():
    """A legacy-only key survives a merge -- this is the non-truncation invariant."""
    existing = [_entry("legacyOnly2019", "Old"), _entry("shared2020", "Stale")]
    incoming = [_entry("shared2020", "Fresh"), _entry("brandNew2024", "New")]

    merged, changes, preserved = merge_bib_entries(existing, incoming)

    keys = {e["ID"] for e in merged}
    assert keys == {"legacyOnly2019", "shared2020", "brandNew2024"}
    assert preserved == ["legacyOnly2019"]
    assert {c.citation_key: c.mode for c in changes} == {
        "shared2020": "unchanged",
        "brandNew2024": "added",
    }


def test_default_merge_does_not_touch_existing_entries():
    """Add-only is the default: a shared key keeps its local value verbatim."""
    existing = [_entry("shared2020", "Local Title", shortjournal="Nat Methods")]
    incoming = [_entry("shared2020", "Zotero Title")]

    merged, _, _ = merge_bib_entries(existing, incoming)

    assert merged[0]["title"] == "Local Title"
    assert merged[0]["shortjournal"] == "Nat Methods"


def test_update_existing_takes_zotero_fields_but_keeps_local_only_fields():
    """Field-wise update: Zotero wins per field; local-only fields survive.

    ``shortjournal`` is the field that matters -- the Nature CSL uses it for
    abbreviated journal names, and a wholesale overwrite would drop it.
    """
    existing = [_entry("shared2020", "Local Title", shortjournal="Nat Methods")]
    incoming = [_entry("shared2020", "Zotero Title")]

    merged, changes, _ = merge_bib_entries(existing, incoming, update_existing=True)

    assert merged[0]["title"] == "Zotero Title"
    assert merged[0]["shortjournal"] == "Nat Methods"
    assert [c.mode for c in changes] == ["updated"]


def test_merge_reports_unchanged_when_identical():
    """An identical entry is reported unchanged, not updated."""
    entry = _entry("same2021", "Same")
    _, changes, _ = merge_bib_entries([entry], [dict(entry)], update_existing=True)
    assert [c.mode for c in changes] == ["unchanged"]


def test_sanitizes_invalid_citation_key(tmp_path):
    """A key with a `$` makes the file unparseable to pandoc, so it is stripped."""
    path = tmp_path / "bad.bib"
    path.write_text(
        "@online{yun$On$Connections2020, title = {Sparse}, year = {2020}}\n"
    )
    assert {e["ID"] for e in read_bib_entries(path)} == {"yunOnConnections2020"}


def test_plan_does_not_write(legacy_bib):
    """The dry-run path classifies without touching the file."""
    before = legacy_bib.read_text()
    report = plan_bib_sync(legacy_bib, [_entry("brandNew2024", "New")])
    assert isinstance(report, BibFileSyncReport)
    assert report.written is False
    assert report.n_before == 2
    assert report.n_after == 3
    assert legacy_bib.read_text() == before


def test_sync_writes_merged_file(legacy_bib):
    """A real sync adds the new entry and keeps the legacy one."""
    report = sync_bib_file(legacy_bib, [_entry("brandNew2024", "New")])
    assert report.written is True
    keys = {e["ID"] for e in read_bib_entries(legacy_bib)}
    assert keys == {"legacyOnlyEntry2019", "sharedEntry2020", "brandNew2024"}


def test_sync_of_empty_pull_keeps_every_entry(legacy_bib):
    """An empty Zotero pull must leave the bibliography intact, not blank it."""
    report = sync_bib_file(legacy_bib, [])
    assert report.n_after == 2
    assert len(read_bib_entries(legacy_bib)) == 2


def test_sync_refuses_to_shrink(legacy_bib, monkeypatch):
    """A merge that would drop entries raises instead of writing a truncated file."""
    import torchcell.literature.bib as bib_mod

    monkeypatch.setattr(bib_mod, "merge_bib_entries", lambda e, i, **kw: ([], [], []))
    with pytest.raises(RuntimeError, match="refusing to write"):
        sync_bib_file(legacy_bib, [_entry("brandNew2024", "New")])


def test_duplicate_keys_collapse(tmp_path):
    """Duplicate citation keys collapse to one entry rather than emitting both."""
    path = tmp_path / "dup.bib"
    path.write_text(
        "@article{dup2020, author = {A, B}, title = {First}, year = {2020}}\n"
        "@article{dup2020, author = {A, B}, title = {Second}, year = {2020}}\n"
    )
    report = plan_bib_sync(path, [])
    assert report.n_before == 1
    assert report.n_after == 1
