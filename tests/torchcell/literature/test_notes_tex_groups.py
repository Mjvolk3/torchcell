# tests/torchcell/literature/test_notes_tex_groups.py
# [[tests.torchcell.literature.test_notes_tex_groups]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_notes_tex_groups.py
"""The notes-tex group layer: ``notes-tex/<group>/<slug>/`` and the Zotero path it derives.

``zotero_publish.py`` resolves a bare slug by looking under ``notes-tex/*/`` in the
tree, refuses the pre-group flat form, and refuses to publish while the document's
collection still sits flat under ``torchcell/notes-tex``. ``zotero_regroup.py`` plans
the one-shot move of those flat collections. Both are scripts imported from their
directory; the client is the read-only ``FakeZot``, so only the planning and the
refusals are exercised, never a write.
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


@pytest.fixture
def scripts(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[ModuleType, ModuleType]]:
    """``(zotero_publish, zotero_regroup)`` imported from notes-tex/common."""
    monkeypatch.syspath_prepend(str(COMMON))
    publish = importlib.import_module("zotero_publish")
    regroup = importlib.import_module("zotero_regroup")
    yield publish, regroup
    for name in ("zotero_publish", "zotero_regroup", "zotero_comments"):
        sys.modules.pop(name, None)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A repo skeleton: two groups, three documents, and the ``common`` machinery."""
    for rel in (
        "notes-tex/common",
        "notes-tex/trigenic/010-index-defect",
        "notes-tex/trigenic/010-positive-panel",
        "notes-tex/wet-lab/024-perturb-seq-costing",
        "paper/nature-biotech",
    ):
        (tmp_path / rel).mkdir(parents=True)
    (tmp_path / "notes-tex" / "README.md").write_text("not a group\n")
    return tmp_path


def test_groups_are_read_from_the_tree(
    scripts: tuple[ModuleType, ModuleType], tree: Path
) -> None:
    publish, _ = scripts
    assert publish.notes_tex_groups(str(tree)) == {
        "010-index-defect": "trigenic",
        "010-positive-panel": "trigenic",
        "024-perturb-seq-costing": "wet-lab",
    }


def test_a_slug_under_two_groups_is_refused(
    scripts: tuple[ModuleType, ModuleType], tree: Path
) -> None:
    publish, _ = scripts
    (tree / "notes-tex" / "wet-lab" / "010-index-defect").mkdir()
    with pytest.raises(SystemExit, match="under both trigenic/ and wet-lab/"):
        publish.notes_tex_groups(str(tree))


def test_resolve_bare_slug_grouped_path_and_other_trees(
    scripts: tuple[ModuleType, ModuleType], tree: Path
) -> None:
    publish, _ = scripts
    repo = str(tree)
    assert (
        publish.resolve_doc_dir(repo, "010-index-defect")
        == "notes-tex/trigenic/010-index-defect"
    )
    assert (
        publish.resolve_doc_dir(repo, "notes-tex/wet-lab/024-perturb-seq-costing/")
        == "notes-tex/wet-lab/024-perturb-seq-costing"
    )
    assert (
        publish.resolve_doc_dir(repo, "paper/nature-biotech") == "paper/nature-biotech"
    )


@pytest.mark.parametrize(
    ("doc", "message"),
    [
        ("no-such-doc", "no notes-tex/<group>/no-such-doc/ directory"),
        ("notes-tex/010-index-defect", "not a notes-tex/<group>/<slug> directory"),
        (
            "notes-tex/common/zotero_publish.py",
            "not a notes-tex/<group>/<slug> directory",
        ),
        (
            "notes-tex/trigenic/010-index-defect/sections",
            "not a notes-tex/<group>/<slug> directory",
        ),
    ],
)
def test_resolve_refuses_unknown_and_flat_forms(
    scripts: tuple[ModuleType, ModuleType], tree: Path, doc: str, message: str
) -> None:
    publish, _ = scripts
    with pytest.raises(SystemExit, match=message):
        publish.resolve_doc_dir(str(tree), doc)


def _built(publish: ModuleType, doc_dir: str) -> Any:
    return publish.BuiltDoc(
        doc_dir=doc_dir,
        pdf_stem="x",
        pdf_path="/x.pdf",
        title="T",
        subtitle=None,
        authors=[],
        sha256="0" * 64,
        n_bytes=1,
        git_commit="abc",
        git_branch="main",
        built_at="2026-01-01-00-00-00",
    )


def test_publish_refuses_while_the_flat_collection_exists(
    scripts: tuple[ModuleType, ModuleType],
) -> None:
    """The pre-group collection is still under notes-tex, so publishing is refused with
    the exact regroup command rather than creating a second history under the group.
    """
    publish, _ = scripts
    zot = FakeZot(
        collections=[
            collection("ROOT", "torchcell"),
            collection("NOTES", "notes-tex", parent="ROOT"),
            collection("FLAT", "010-index-defect", parent="NOTES"),
        ]
    )
    with pytest.raises(SystemExit) as exc:
        publish.refuse_flat_legacy(
            zot, "ROOT", _built(publish, "notes-tex/trigenic/010-index-defect")
        )
    assert "torchcell/notes-tex/010-index-defect (FLAT) still sits flat" in str(
        exc.value
    )
    assert "--assign 010-index-defect=trigenic --dry-run" in str(exc.value)


def test_publish_proceeds_when_grouped_or_not_notes_tex(
    scripts: tuple[ModuleType, ModuleType],
) -> None:
    """A grouped document is looked up (two collection reads: the index, then the flat
    name under it) and passes; a non-notes-tex path is not looked up at all.
    """
    publish, _ = scripts
    zot = FakeZot(
        collections=[
            collection("ROOT", "torchcell"),
            collection("NOTES", "notes-tex", parent="ROOT"),
            collection("GRP", "trigenic", parent="NOTES"),
            collection("DOC", "010-index-defect", parent="GRP"),
            collection("PAPER", "paper", parent="ROOT"),
        ]
    )
    publish.refuse_flat_legacy(
        zot, "ROOT", _built(publish, "notes-tex/trigenic/010-index-defect")
    )
    assert zot.calls == [
        ("collections",),
        ("everything",),
        ("collections",),
        ("everything",),
    ]
    zot.calls.clear()
    publish.refuse_flat_legacy(zot, "ROOT", _built(publish, "paper/nature-biotech"))
    assert zot.calls == []


def _item(key: str, title: str, doc_key: str | None) -> dict[str, Any]:
    extra = f"Doc Key: {doc_key}\nGit Commit: abc\nGit Branch: main" if doc_key else ""
    return {"key": key, "data": {"title": title, "extra": extra, "collections": []}}


def test_regroup_plan_moves_rewrites_and_reports(
    scripts: tuple[ModuleType, ModuleType],
) -> None:
    """Three flat collections: one mapped to an existing group, one to a group that
    does not exist yet (its item has a non-flat Doc Key, which is kept), one unmapped.
    A collection already under a group is reported, not moved.
    """
    _, regroup = scripts
    zot = FakeZot(
        collections=[
            collection("ROOT", "torchcell"),
            collection("NOTES", "notes-tex", parent="ROOT"),
            collection("TRI", "trigenic", parent="NOTES"),
            collection("DONE", "010-positive-panel", parent="TRI"),
            collection("FLAT1", "010-index-defect", parent="NOTES"),
            collection("FLAT2", "experiments.026-flux", parent="NOTES"),
            collection("FLAT3", "orphan-doc", parent="NOTES"),
            {
                "key": "GONE",
                "data": {
                    "name": "deleted-doc",
                    "parentCollection": "NOTES",
                    "deleted": True,
                },
            },
        ],
        collection_members={
            "FLAT1": [_item("I1", "Index defect", "notes-tex/010-index-defect")],
            "FLAT2": [_item("I2", "Flux note", "notes/experiments.026-flux.md")],
        },
    )
    p = regroup.plan(
        zot,
        {
            "010-index-defect": "trigenic",
            "experiments.026-flux": "metabolism",
            "010-positive-panel": "trigenic",
        },
    )
    assert p.model_dump() == {
        "moves": [
            {
                "slug": "010-index-defect",
                "group": "trigenic",
                "collection_key": "FLAT1",
                "group_key": "TRI",
                "items": [
                    {
                        "key": "I1",
                        "title": "Index defect",
                        "old_doc_key": "notes-tex/010-index-defect",
                        "new_doc_key": "notes-tex/trigenic/010-index-defect",
                    }
                ],
            },
            {
                "slug": "experiments.026-flux",
                "group": "metabolism",
                "collection_key": "FLAT2",
                "group_key": None,
                "items": [
                    {
                        "key": "I2",
                        "title": "Flux note",
                        "old_doc_key": "notes/experiments.026-flux.md",
                        "new_doc_key": None,
                    }
                ],
            },
        ],
        "unassigned": ["orphan-doc"],
        "already_grouped": ["trigenic/010-positive-panel"],
    }


def test_regroup_dry_run_writes_nothing(
    scripts: tuple[ModuleType, ModuleType], capsys: pytest.CaptureFixture[str]
) -> None:
    """FakeZot has no write methods, so a dry run that tried to write would raise."""
    _, regroup = scripts
    zot = FakeZot(
        collections=[
            collection("ROOT", "torchcell"),
            collection("NOTES", "notes-tex", parent="ROOT"),
            collection("FLAT1", "010-index-defect", parent="NOTES"),
        ],
        collection_members={
            "FLAT1": [
                _item("I1", "Index defect", "notes-tex/010-index-defect"),
                _item("I9", "Someone else's paper", None),
            ]
        },
    )
    p = regroup.plan(zot, {"010-index-defect": "trigenic"})
    regroup.apply(zot, p, "NOTES", dry=True)
    out = capsys.readouterr().out
    assert "[dry-run] would create group collection 'trigenic'" in out
    assert "010-index-defect  ->  notes-tex/trigenic/" in out
    assert (
        "Doc Key notes-tex/010-index-defect -> notes-tex/trigenic/010-index-defect"
        in out
    )
    assert 'item I9  "Someone else\'s paper"  no Doc Key; not ours, left alone' in out


def test_doc_key_rewrite_touches_only_the_marker_line(
    scripts: tuple[ModuleType, ModuleType],
) -> None:
    _, regroup = scripts
    extra = "Doc Key: notes-tex/a\nGit Commit: abc\nGit Branch: main"
    assert regroup._rewrite_doc_key(extra, "notes-tex/g/a") == (
        "Doc Key: notes-tex/g/a\nGit Commit: abc\nGit Branch: main"
    )


@pytest.mark.parametrize(
    ("doc_dir", "pdf_stem", "expected"),
    [
        # the default build of a notes-tex document: no stem in the name
        ("notes-tex/trigenic/010-index-defect", "010-index-defect", "010-index-defect"),
        ("notes-tex/trigenic/010-index-defect", "main", "010-index-defect"),
        # the share view: the suffix survives with its hyphen
        (
            "notes-tex/trigenic/010-index-defect",
            "010-index-defect-clean",
            "010-index-defect-clean",
        ),
        # the manuscript's views: a stem that is neither the document nor prefixed by it
        ("paper/nature-biotech", "submission", "nature-biotech-submission"),
        ("paper/nature-biotech", "editing", "nature-biotech-editing"),
        # the renamed editing view (2026-10-08): the `-nature-biotech` suffix is dropped so
        # the attachment name is byte-identical to the ones published before the rename
        ("paper/nature-biotech", "editing-nature-biotech", "nature-biotech-editing"),
    ],
)
def test_attachment_filename_drops_the_document_name_wherever_it_sits(
    scripts: tuple[ModuleType, ModuleType], doc_dir: str, pdf_stem: str, expected: str
) -> None:
    publish, _ = scripts
    built = _built(publish, doc_dir).model_copy(update={"pdf_stem": pdf_stem})
    assert built.filename == f"{expected}_2026-01-01-00-00-00_00000000.pdf"
