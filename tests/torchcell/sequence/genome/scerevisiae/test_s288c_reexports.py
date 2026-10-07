# tests/torchcell/sequence/genome/scerevisiae/test_s288c_reexports.py
# [[tests.torchcell.sequence.genome.scerevisiae.test_s288c_reexports]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/genome/scerevisiae/test_s288c_reexports.py
"""The s288c module after the 2026.10.07 base extraction: one namespace with ``base``.

Everything organism-agnostic moved to ``torchcell.sequence.genome.base``. These tests
pin the compatibility contract the move relies on: every name ``base`` defines is
reachable from ``s288c`` as the same object; rebinding a shared name on ``s288c`` (as
``monkeypatch.setattr(s288c, ...)`` does throughout the synthetic suite) rebinds it in
``base``, where the code that looks it up now lives, and undoing restores both; a name
``s288c`` owns is not mirrored; and the pre-extraction unpickling target still reopens
a genome.
"""

import ast
from pathlib import Path
from typing import Any

import pytest

import torchcell.sequence.genome.base as base
import torchcell.sequence.genome.scerevisiae.s288c as s288c
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


def _base_definitions() -> list[str]:
    """Every top-level name ``base`` defines (functions, classes, assignments)."""
    names: list[str] = []
    for node in ast.parse(Path(base.__file__).read_text()).body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names.extend(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return sorted(names)


def test_every_base_definition_is_reachable_from_s288c() -> None:
    """All of them but ``base``'s own logger, as the identical object."""
    names = [n for n in _base_definitions() if n != "log"]
    assert len(names) == 66
    assert [n for n in names if getattr(s288c, n, None) is not getattr(base, n)] == []


def test_rebinding_a_shared_name_on_s288c_rebinds_it_in_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``base.untrusted_reason`` looks up ``_change_counter`` in base's globals, so a
    patch made on ``s288c`` must land there; ``undo`` restores both modules.
    """
    original = base._change_counter

    def fake(path: str) -> int:
        return -1

    monkeypatch.setattr(s288c, "_change_counter", fake)
    assert (s288c._change_counter, base._change_counter) == (fake, fake)
    monkeypatch.undo()
    assert (s288c._change_counter, base._change_counter) == (original, original)


def test_deleting_a_shared_name_on_s288c_deletes_it_in_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``delattr`` mirrors too, and ``undo`` puts the name back in both."""
    original = base.nucleotides
    monkeypatch.delattr(s288c, "nucleotides")
    assert (hasattr(s288c, "nucleotides"), hasattr(base, "nucleotides")) == (
        False,
        False,
    )
    monkeypatch.undo()
    assert (s288c.nucleotides, base.nucleotides) == (original, original)
    assert original == ["A", "T", "G", "C"]


def test_a_name_s288c_owns_is_not_mirrored(monkeypatch: pytest.MonkeyPatch) -> None:
    """``download_url`` (the SGD GO download) and ``SCerevisiaeGenome`` stay local."""
    monkeypatch.setattr(s288c, "download_url", lambda url, folder: None)
    monkeypatch.setattr(s288c, "SCerevisiaeGenome", object)
    assert (hasattr(base, "download_url"), hasattr(base, "SCerevisiaeGenome")) == (
        False,
        False,
    )


def test_the_pre_extraction_unpickling_target_reopens_through_the_base_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pickle written before the extraction names
    ``s288c._restore_genome(cls, genome_root, go_root, private_db_path)``; it hands
    the base restore the init fields with ``overwrite=False`` and returns its genome.
    """
    calls: list[tuple[Any, ...]] = []
    reopened = object()

    def restore(cls: Any, init_kwargs: dict[str, Any], private: str | None) -> Any:
        calls.append((cls, init_kwargs, private))
        return reopened

    monkeypatch.setattr(s288c, "_restore_annotated_genome", restore)
    result = s288c._restore_genome(SCerevisiaeGenome, "/g", "/go", "/tmp/copy.db")
    assert result is reopened
    assert calls == [
        (
            SCerevisiaeGenome,
            {"genome_root": "/g", "go_root": "/go", "overwrite": False},
            "/tmp/copy.db",
        )
    ]
