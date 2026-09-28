# tests/torchcell/literature/test_bib_more.py
# [[tests.torchcell.literature.test_bib_more]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_bib_more.py
"""Zotero -> BibTeX pulls over a read-only fake Zotero library (``_fake_zotero.FakeZot``).

Each fake library serves a ``format="bibtex"`` response per scope (``None`` = whole
library, else a collection key) and the matching JSON item list, which is what
``citable_citation_keys`` reads to detect key drift. Entries are bibtexparser dicts; an
attachment stub has neither ``title`` nor ``author`` and must be dropped. A key holding
``$`` is sanitized (``yun$On$Connections2020`` -> ``yunOnConnections2020``).
"""

import logging
import re
from pathlib import Path
from typing import Any

import pytest

from tests.torchcell.literature._fake_zotero import (
    FakeZot,
    bib_db,
    collection,
    make_library,
)
from torchcell.literature.bib import (
    BibEntryChange,
    BibFileSyncReport,
    _collection_selector,
    _collection_tree,
    _is_citable_entry,
    citable_citation_keys,
    fetch_bibtex_entries,
    fetch_paired_collection_entries,
    fetch_union_bibtex_entries,
    read_bib_entries,
    sync_bib_file,
)


def _entry(key: str, title: str = "T", author: str = "A, B") -> dict[str, str]:
    return {"ID": key, "ENTRYTYPE": "article", "title": title, "author": author}


def _item(key: str, citation_key: str, item_type: str = "journalArticle") -> Any:
    return {"key": key, "data": {"itemType": item_type, "citationKey": citation_key}}


STUB = {"ID": "noauthor_notitle_nodate", "ENTRYTYPE": "misc"}


def test_report_by_mode_and_summary() -> None:
    """Counts per mode, alphabetical, plus preserved and the Zotero pull size."""
    report = BibFileSyncReport(
        path="bib.bib",
        n_before=2,
        n_after=3,
        n_zotero=2,
        changes=[
            BibEntryChange(citation_key="a", mode="added"),
            BibEntryChange(citation_key="b", mode="unchanged"),
        ],
        preserved=["c"],
    )
    assert report.by_mode("added") == [BibEntryChange(citation_key="a", mode="added")]
    assert report.summary() == (
        "bib.bib: 2 -> 3 entries | added=1 unchanged=1 updated=0 preserved=1 (zotero=2)"
    )


@pytest.mark.parametrize(
    ("entry", "citable"),
    [
        (STUB, False),
        ({"ID": "x", "title": "  "}, False),
        ({"ID": "x", "title": "Real"}, True),
        ({"ID": "x", "author": "Doe, J"}, True),
    ],
)
def test_is_citable_entry(entry: dict[str, str], citable: bool) -> None:
    """A title or an author (non-blank) marks a real work."""
    assert _is_citable_entry(entry) is citable


def test_collection_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eight uppercase alphanumerics is a key; anything else is a name (no call made)."""
    zot = FakeZot()
    lib = make_library(monkeypatch, zot)
    assert _collection_selector(lib, "W46ATS7B") == {"collection_key": "W46ATS7B"}
    assert _collection_selector(lib, "microbe-perturb-seq") == {
        "collection": "microbe-perturb-seq"
    }
    assert _collection_selector(lib, "w46ats7b") == {"collection": "w46ats7b"}
    assert zot.calls == []


def test_citable_citation_keys_three_scopes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whole library, by name (resolved to a key), and by key; attachments excluded."""
    zot = FakeZot(
        collections=[collection("COLL0001", "papers")],
        items=[_item("I1", "whole2020"), _item("I2", "stub", "attachment")],
        collection_members={"COLL0001": [_item("I3", "inColl2021")]},
    )
    lib = make_library(monkeypatch, zot)
    assert citable_citation_keys(lib) == {"whole2020"}
    assert citable_citation_keys(lib, "papers") == {"inColl2021"}
    assert citable_citation_keys(lib, collection_key="COLL0001") == {"inColl2021"}


def test_fetch_bibtex_entries_filters_sanitizes_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The stub is dropped, ``$`` stripped, and a stored-but-unexported key warned.

    Drift is checked on the RAW exported keys, before sanitizing, so only
    ``drifted2019`` (stored, never exported) is reported.
    """
    zot = FakeZot(
        items=[_item("I1", "yun$On$Connections2020"), _item("I2", "drifted2019")],
        bibtex={None: bib_db([_entry("yun$On$Connections2020"), dict(STUB)])},
    )
    lib = make_library(monkeypatch, zot)
    with caplog.at_level(logging.WARNING, logger="torchcell.literature.bib"):
        entries = fetch_bibtex_entries(lib)
    assert entries == [_entry("yunOnConnections2020")]
    assert [r.getMessage() for r in caplog.records] == [
        "bib: 1 stored citationKey(s) differ from the BibTeX export and will not "
        "resolve if cited by the stored key: drifted2019",
        "bib: Zotero citation key 'yun$On$Connections2020' is not a valid BibTeX "
        "key; writing 'yunOnConnections2020' (fix it in Zotero so the two agree)",
    ]


def test_fetch_bibtex_entries_by_name_and_empty_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A named collection resolves to its key; an empty one returns a plain list."""
    zot = FakeZot(
        collections=[collection("COLL0001", "papers"), collection("EMPTY001", "empty")],
        collection_members={"COLL0001": [_item("I1", "a2020")]},
        bibtex={"COLL0001": bib_db([_entry("a2020")]), "EMPTY001": []},
    )
    lib = make_library(monkeypatch, zot)
    assert fetch_bibtex_entries(lib, "papers") == [_entry("a2020")]
    assert fetch_bibtex_entries(lib, collection_key="EMPTY001") == []


def test_fetch_paired_collection_entries_personal_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Group by key, personal by name; the shared key takes the personal entry."""
    group = make_library(
        monkeypatch,
        FakeZot(
            collection_members={"GROUP001": [_item("G", "shared2020")]},
            bibtex={
                "GROUP001": bib_db(
                    [_entry("shared2020", title="Group"), _entry("groupOnly2019")]
                )
            },
        ),
    )
    user = make_library(
        monkeypatch,
        FakeZot(
            collections=[collection("USER0001", "microbe-perturb-seq")],
            collection_members={"USER0001": [_item("U", "shared2020")]},
            bibtex={"USER0001": bib_db([_entry("shared2020", title="Personal")])},
        ),
    )
    entries = fetch_paired_collection_entries(
        group, user, group_collection="GROUP001", user_collection="microbe-perturb-seq"
    )
    assert entries == [_entry("shared2020", title="Personal"), _entry("groupOnly2019")]


def _user_tree_library(monkeypatch: pytest.MonkeyPatch) -> Any:
    return make_library(
        monkeypatch,
        FakeZot(
            collections=[
                collection("ROOT0001", "torchcell"),
                collection("KID00001", "topics", parent="ROOT0001"),
            ],
            bibtex={
                "ROOT0001": [],
                "KID00001": bib_db(
                    [_entry("shared2020", title="Personal"), _entry("mine2024")]
                ),
            },
        ),
    )


def test_collection_tree_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Root key first, then its child."""
    assert _collection_tree(_user_tree_library(monkeypatch), "torchcell") == [
        "ROOT0001",
        "KID00001",
    ]


def test_fetch_union_bibtex_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Group only without a user; with one, the personal tree overrides and adds."""
    group = make_library(
        monkeypatch,
        FakeZot(
            bibtex={
                None: bib_db([_entry("shared2020", title="Group"), _entry("g2019")])
            }
        ),
    )
    assert fetch_union_bibtex_entries(group) == [
        _entry("shared2020", title="Group"),
        _entry("g2019"),
    ]
    user = _user_tree_library(monkeypatch)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "user_root_collection is required when a user library is given"
        ),
    ):
        fetch_union_bibtex_entries(group, user)
    assert fetch_union_bibtex_entries(
        group, user, user_root_collection="torchcell"
    ) == [_entry("shared2020", title="Personal"), _entry("g2019"), _entry("mine2024")]


def test_read_missing_file_is_empty(tmp_path: Path) -> None:
    """An absent .bib reads as no entries."""
    assert read_bib_entries(tmp_path / "absent.bib") == []


def test_sync_dry_run_writes_nothing(tmp_path: Path) -> None:
    """``dry_run`` classifies the add but leaves the file absent and unwritten."""
    path = tmp_path / "bib.bib"
    report = sync_bib_file(path, [_entry("new2024")], dry_run=True)
    assert report.model_dump() == {
        "path": str(path),
        "n_before": 0,
        "n_after": 1,
        "n_zotero": 1,
        "changes": [{"citation_key": "new2024", "mode": "added"}],
        "preserved": [],
        "written": False,
    }
    assert not path.exists()
