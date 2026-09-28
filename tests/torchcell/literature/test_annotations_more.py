# tests/torchcell/literature/test_annotations_more.py
# [[tests.torchcell.literature.test_annotations_more]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_annotations_more.py
r"""Annotation capture over a read-only fake Zotero library (``_fake_zotero.FakeZot``).

The personal library fixture holds two papers, ``P1`` (``citationKey`` ``paperOne2020``)
and ``P2`` (``paperTwo2021``), each with a PDF attachment (``A1``, ``A2``), plus:

* ``N1``: an annotation on ``A1`` with text, comment, page 3 and yellow ``#FFD400``;
* ``N2``: an annotation on ``A1`` with text only, page 1, an unknown color;
* ``N3``: an empty annotation on ``A1`` (dropped: no text and no comment);
* ``N4``: an annotation on an attachment ``AX`` that is not in the library (dropped);
* ``T1``: a note on ``P2`` in HTML (``<p>Key point</p><ul><li>a &amp; b</li></ul>``),
  flattened to ``Key point\n- a & b``;
* ``T2``: a note whose parent is not a top-level item (dropped);
* ``N5``: an annotation on ``A2`` (kept only when ``P2`` is in ``parent_keys``).

The personal collection tree is ``torchcell`` (``R``) -> ``topics`` (``C``), with P1
filed directly in ``R`` and P2 in ``C``; an unrelated collection ``Z`` holds P3.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from tests.torchcell.literature._fake_zotero import FakeZot, collection, make_library
from torchcell.literature.annotations import (
    Annotation,
    PaperAnnotations,
    capture_annotations,
    collect_all,
    collect_library_annotations,
    merge_annotations,
    personal_tree_item_keys,
    render_markdown,
    unmirrored_with_annotations,
)

EM = "\u2014"  # the em-dash character, escaped so the source holds none


def _item(key: str, **data: Any) -> dict[str, Any]:
    return {"key": key, "data": data}


ITEMS = [
    _item("P1", itemType="journalArticle", citationKey="paperOne2020"),
    _item("P2", itemType="journalArticle", citationKey="paperTwo2021"),
    _item("A1", itemType="attachment", parentItem="P1"),
    _item("A2", itemType="attachment", parentItem="P2"),
    _item(
        "N1",
        itemType="annotation",
        parentItem="A1",
        annotationText=" the words ",
        annotationComment=" our words ",
        annotationPageLabel="3",
        annotationColor="#FFD400",
    ),
    _item(
        "N2",
        itemType="annotation",
        parentItem="A1",
        annotationText="just a highlight",
        annotationPageLabel="1",
        annotationColor="#123456",
    ),
    _item("N3", itemType="annotation", parentItem="A1", annotationText="  "),
    _item("N4", itemType="annotation", parentItem="AX", annotationText="orphan"),
    _item(
        "T1",
        itemType="note",
        parentItem="P2",
        note="<p>Key point</p><ul><li>a &amp; b</li></ul>",
    ),
    _item("T2", itemType="note", parentItem="A1", note="<p>on an attachment</p>"),
    _item("N5", itemType="annotation", parentItem="A2", annotationText="second paper"),
]

N1 = Annotation(
    kind="highlight",
    text="the words",
    comment="our words",
    page="3",
    color="#FFD400",
    color_name="yellow",
    sources=["personal"],
    item_keys={"personal": "N1"},
)
N2 = Annotation(
    kind="highlight",
    text="just a highlight",
    page="1",
    color="#123456",
    sources=["personal"],
    item_keys={"personal": "N2"},
)
T1 = Annotation(
    kind="note",
    comment="Key point\n- a & b",
    sources=["personal"],
    item_keys={"personal": "T1"},
)
N5 = Annotation(
    kind="highlight",
    text="second paper",
    sources=["personal"],
    item_keys={"personal": "N5"},
)


def _dump(out: dict[str, list[Annotation]]) -> dict[str, list[dict[str, Any]]]:
    return {ck: [a.model_dump() for a in anns] for ck, anns in out.items()}


def test_collect_library_annotations_whole_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N3 (empty), N4 (unknown attachment) and T2 (non-top parent) are dropped."""
    lib = make_library(monkeypatch, FakeZot(items=ITEMS))
    out = collect_library_annotations(lib, "personal")
    assert _dump(out) == {
        "paperOne2020": [N1.model_dump(), N2.model_dump()],
        "paperTwo2021": [T1.model_dump(), N5.model_dump()],
    }


def test_collect_library_annotations_respects_parent_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only P1's records survive ``parent_keys={"P1"}``, notes and highlights alike."""
    lib = make_library(monkeypatch, FakeZot(items=ITEMS))
    out = collect_library_annotations(lib, "personal", parent_keys={"P1"})
    assert _dump(out) == {"paperOne2020": [N1.model_dump(), N2.model_dump()]}


def test_merge_unions_sources_and_item_keys_once() -> None:
    """Identical content from both libraries merges; a repeated source is not doubled."""
    group_copy = N1.model_copy(
        update={"sources": ["group"], "item_keys": {"group": "G1"}}
    )
    personal_again = N1.model_copy(update={"item_keys": {"personal": "N1b"}})
    merged = merge_annotations(
        {"paperOne2020": [N1]},
        {"paperOne2020": [group_copy]},
        {"paperOne2020": [personal_again]},
    )
    [only] = merged["paperOne2020"].annotations
    assert only.sources == ["personal", "group"]
    assert only.item_keys == {"personal": "N1b", "group": "G1"}
    assert N1.sources == ["personal"]


def test_render_markdown_all_sections() -> None:
    """Finding: annotations.py:97 counts a NOTE as a comment, so it renders twice.

    ``comments`` keeps every record with a non-blank ``comment``, and a note stores its
    text in ``comment``; the note T1 therefore appears under "## Comments" (no page, so
    a dash, and no quoted highlight) and again under "## Notes", and the tally reads 2
    comments. A comment-free highlight with neither page nor color reads "dash dash".
    """
    pa = PaperAnnotations(citation_key="paperOne2020", annotations=[N1, T1, N2, N5])
    assert pa.comments == [N1, T1]
    assert render_markdown(pa) == "\n".join(
        [
            f"# Annotations {EM} paperOne2020",
            "",
            "2 comments · 2 highlights · 1 notes · sources: personal",
            "",
            "## Comments",
            "",
            "**[personal]** p3 · yellow",
            "> the words",
            "",
            "our words",
            "",
            f"**[personal]** {EM}",
            "",
            "Key point\n- a & b",
            "",
            "## Notes",
            "",
            "**[personal]**",
            "",
            "Key point\n- a & b",
            "",
            "## Highlights (no comment)",
            "",
            f"- **[personal]** p1 {EM} just a highlight",
            f"- **[personal]** {EM} {EM} second paper",
            "",
        ]
    )


def test_render_markdown_empty_paper() -> None:
    """No records: zero tallies and ``sources: none``, no section headings."""
    pa = PaperAnnotations(citation_key="empty2020")
    assert render_markdown(pa) == "\n".join(
        [
            f"# Annotations {EM} empty2020",
            "",
            "0 comments · 0 highlights · 0 notes · sources: none",
            "",
        ]
    )


def _tree_zot() -> FakeZot:
    return FakeZot(
        items=ITEMS,
        collections=[
            collection("R", "torchcell"),
            collection("C", "topics", parent="R"),
            collection("Z", "elsewhere"),
        ],
        collection_members={
            "R": [ITEMS[0], _item("A9", itemType="attachment")],
            "C": [ITEMS[1], _item("T9", itemType="note")],
            "Z": [_item("P3", itemType="book")],
        },
    )


def test_personal_tree_item_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """R and its child C are walked; attachments/notes and collection Z are excluded."""
    zot = _tree_zot()
    lib = make_library(monkeypatch, zot)
    assert personal_tree_item_keys(lib, "torchcell") == {"P1", "P2"}
    assert [c for c in zot.calls if c[0] == "collection_items_top"] == [
        ("collection_items_top", "R"),
        ("collection_items_top", "C"),
    ]


def test_collect_all_merges_personal_tree_and_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same library as both sources: every record lists personal + group."""
    lib = make_library(monkeypatch, _tree_zot())
    merged = collect_all(lib, lib, root_collection="torchcell")
    assert sorted(merged) == ["paperOne2020", "paperTwo2021"]
    assert [
        (a.kind, a.text, a.sources) for a in merged["paperOne2020"].annotations
    ] == [
        ("highlight", "the words", ["personal", "group"]),
        ("highlight", "just a highlight", ["personal", "group"]),
    ]
    assert merged["paperTwo2021"].annotations[0].item_keys == {
        "personal": "N5",
        "group": "N5",
    }


def test_capture_only_mirrored_and_unmirrored_listing(tmp_path: Path) -> None:
    """Only a key with ``manifest.json`` is written by default; the other is listed."""
    (tmp_path / "paperOne2020").mkdir()
    (tmp_path / "paperOne2020" / "manifest.json").write_text("{}")
    merged = {
        "paperOne2020": PaperAnnotations(citation_key="paperOne2020", annotations=[N1]),
        "paperTwo2021": PaperAnnotations(citation_key="paperTwo2021", annotations=[N5]),
    }
    assert capture_annotations(tmp_path, merged) == ["paperOne2020"]
    assert sorted(p.name for p in (tmp_path / "paperOne2020").iterdir()) == [
        "annotations.json",
        "annotations.md",
        "manifest.json",
    ]
    assert (
        json.loads((tmp_path / "paperOne2020" / "annotations.json").read_text())
        == merged["paperOne2020"].model_dump()
    )
    assert not (tmp_path / "paperTwo2021").exists()
    assert unmirrored_with_annotations(tmp_path, merged) == ["paperTwo2021"]
    assert capture_annotations(tmp_path, merged, only_mirrored=False) == [
        "paperOne2020",
        "paperTwo2021",
    ]
    assert (
        tmp_path / "paperTwo2021" / "annotations.md"
    ).read_text() == render_markdown(merged["paperTwo2021"])
