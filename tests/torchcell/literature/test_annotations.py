# tests/torchcell/literature/test_annotations.py
"""Tests for annotation capture.

The load-bearing behaviour is provenance: a paper annotated in both libraries must
not double its notes, and must never lose track of which library each came from.
"""

import json

from torchcell.literature.annotations import (
    Annotation,
    PaperAnnotations,
    _strip_html,
    merge_annotations,
    render_markdown,
    write_annotations,
)


def _ann(
    source, text="the paper said", comment="my thought", page="3", kind="highlight"
):
    """An annotation from one library."""
    return Annotation(
        kind=kind,
        text=text,
        comment=comment,
        page=page,
        color="#ffd400",
        color_name="yellow",
        sources=[source],
        item_keys={source: f"{source.upper()}KEY"},
    )


def test_identical_content_in_both_libraries_collapses_but_keeps_both_sources():
    """The same note on a personal and a group copy is ONE record citing both."""
    merged = merge_annotations(
        {"paper2024": [_ann("personal")]}, {"paper2024": [_ann("group")]}
    )
    anns = merged["paper2024"].annotations
    assert len(anns) == 1
    assert sorted(anns[0].sources) == ["group", "personal"]
    assert set(anns[0].item_keys) == {"personal", "group"}


def test_differing_content_is_kept_separately():
    """Different comments on the same paper stay distinct records."""
    merged = merge_annotations(
        {"p2024": [_ann("personal", comment="first")]},
        {"p2024": [_ann("group", comment="second")]},
    )
    anns = merged["p2024"].annotations
    assert len(anns) == 2
    assert {a.sources[0] for a in anns} == {"personal", "group"}


def test_comment_and_highlight_are_distinguished():
    """A bare highlight is not counted as a comment."""
    pa = merge_annotations(
        {
            "p2024": [
                _ann("personal", comment=""),
                _ann("personal", comment="mine", page="9"),
            ]
        }
    )["p2024"]
    assert len(pa.comments) == 1
    assert len(pa.highlights) == 1
    assert pa.comments[0].comment == "mine"


def test_notes_are_separated_from_highlights():
    """Free-standing note items land in `notes`, not `highlights`."""
    pa = merge_annotations(
        {"p2024": [_ann("personal", kind="note", text="", comment="a standalone note")]}
    )["p2024"]
    assert len(pa.notes) == 1
    assert pa.highlights == []


def test_comments_sort_before_highlights():
    """Comments come first -- that is the order they are useful to read in."""
    pa = merge_annotations(
        {"p": [_ann("personal", comment=""), _ann("personal", comment="c", page="1")]}
    )["p"]
    assert pa.annotations[0].has_comment


def test_markdown_labels_the_source_library():
    """Rendered markdown must say where each record came from."""
    merged = merge_annotations(
        {"p2024": [_ann("personal", comment="only mine")]},
        {"p2024": [_ann("group", comment="only theirs", page="8")]},
    )
    md = render_markdown(merged["p2024"])
    assert "**[personal]**" in md
    assert "**[group]**" in md
    assert "only mine" in md and "only theirs" in md


def test_markdown_labels_shared_source_as_both():
    """Content in both libraries renders one entry naming both."""
    merged = merge_annotations({"p": [_ann("personal")]}, {"p": [_ann("group")]})
    md = render_markdown(merged["p"])
    assert "[group+personal]" in md or "[personal+group]" in md


def test_write_annotations_emits_json_and_markdown(tmp_path):
    """Both artifacts land in the paper's directory and the JSON round-trips."""
    pa = PaperAnnotations(citation_key="p2024", annotations=[_ann("personal")])
    paths = write_annotations(tmp_path, pa)
    assert {p.name for p in paths} == {"annotations.json", "annotations.md"}
    data = json.loads((tmp_path / "annotations.json").read_text())
    assert data["citation_key"] == "p2024"
    assert data["annotations"][0]["sources"] == ["personal"]


def test_strip_html_flattens_a_zotero_note():
    """Zotero stores notes as HTML; the mirror keeps readable text."""
    out = _strip_html("<p>First line</p><p>Second &amp; third</p>")
    assert "First line" in out
    assert "Second & third" in out
    assert "<p>" not in out


def test_summary_lists_sources():
    """The log line names the libraries involved."""
    pa = merge_annotations({"p": [_ann("personal")]}, {"p": [_ann("group", page="7")]})[
        "p"
    ]
    s = pa.summary()
    assert "group" in s and "personal" in s
