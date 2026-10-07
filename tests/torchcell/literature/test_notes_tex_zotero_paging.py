# tests/torchcell/literature/test_notes_tex_zotero_paging.py
# [[tests.torchcell.literature.test_notes_tex_zotero_paging]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_notes_tex_zotero_paging.py
"""The notes-tex Zotero scripts read every page of an item's children (issue #563).

``notes-tex/common/zotero_publish.py`` and ``zotero_comments.py`` are scripts, not a
package, so they are imported from their directory (``zotero_comments`` imports
``zotero_publish`` by bare name). The client is the read-only ``FakeZot``, which answers
at most 100 rows per request and pages through ``everything``/``follow`` like pyzotero
1.13. It has no write methods, so nothing here can create, modify or upload a Zotero
item; only the two READ helpers are exercised, never the publish path.

Fixture: one published document with 150 versions, attachment ``A000`` .. ``A149`` with
filenames ``doc_2026-01-01-00-00-00_<sha8>.pdf`` where ``<sha8>`` is ``f"{i:08x}"``, so
the newest by filename is ``A149`` (sha8 ``00000095``), which sits on page two.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.torchcell.literature._fake_zotero import FakeZot, collection

COMMON = Path(__file__).resolve().parents[3] / "notes-tex" / "common"
N_VERSIONS = 150


def _filename(i: int) -> str:
    return f"doc_2026-01-01-00-00-00_{i:08x}.pdf"


def _attachments() -> list[dict[str, Any]]:
    return [
        {
            "key": f"A{i:03d}",
            "data": {"itemType": "attachment", "filename": _filename(i)},
        }
        for i in range(N_VERSIONS)
    ]


@pytest.fixture
def scripts(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[ModuleType, ModuleType]]:
    """``(zotero_publish, zotero_comments)`` imported from notes-tex/common."""
    monkeypatch.syspath_prepend(str(COMMON))
    publish = importlib.import_module("zotero_publish")
    comments = importlib.import_module("zotero_comments")
    yield publish, comments
    for name in ("zotero_publish", "zotero_comments"):
        sys.modules.pop(name, None)


def test_existing_hashes_reads_children_past_first_page(
    scripts: tuple[ModuleType, ModuleType],
) -> None:
    """All 150 versions are known, so a 150th build is not re-uploaded as new."""
    publish, _ = scripts
    zot = FakeZot(children={"PARENT": _attachments()})
    hashes = publish.existing_hashes(zot, "PARENT")
    assert hashes == {f"{i:08x}": _filename(i) for i in range(N_VERSIONS)}
    assert hashes["00000095"] == "doc_2026-01-01-00-00-00_00000095.pdf"
    assert zot.calls == [("children", "PARENT"), ("everything",), ("follow",)]


def test_fetch_latest_version_is_the_newest_past_first_page(
    scripts: tuple[ModuleType, ModuleType],
) -> None:
    """With 150 versions the default (latest) is ``A149``, and ``--version 101`` exists."""
    _, comments = scripts
    parent = {"key": "PARENT", "data": {"extra": "Doc Key: notes-tex/grp/doc"}}
    annotation = {
        "key": "ANN1",
        "data": {
            "itemType": "annotation",
            "annotationPageLabel": "3",
            "annotationType": "highlight",
            "annotationColor": "#ffd400",
            "annotationText": " quoted ",
            "annotationComment": " fix this ",
            "annotationSortIndex": "00002|000010|00100",
        },
    }
    zot = FakeZot(
        collections=[
            collection("ROOT", "torchcell"),
            collection("NOTES", "notes-tex", parent="ROOT"),
            collection("GRP", "grp", parent="NOTES"),
            collection("DOC", "doc", parent="GRP"),
        ],
        collection_members={"DOC": [parent]},
        children={"PARENT": _attachments(), "A149": [annotation], "A100": []},
    )
    filename, out = comments.fetch(zot, "notes-tex/grp/doc", None)
    assert filename == "doc_2026-01-01-00-00-00_00000095.pdf"
    assert [c.model_dump() for c in out] == [
        {
            "key": "ANN1",
            "index": 1,
            "page": "3",
            "kind": "highlight",
            "color": "#ffd400",
            "quote": "quoted",
            "comment": "fix this",
        }
    ]
    assert comments.fetch(zot, "notes-tex/grp/doc", 101) == (
        "doc_2026-01-01-00-00-00_00000064.pdf",
        [],
    )
