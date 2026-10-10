# tests/torchcell/candidates/test_store.py
# [[tests.torchcell.candidates.test_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/candidates/test_store.py
"""The verdict store: round trip, key checks, row lookup, GRANDFATHERED and enforcement."""

import sys
import types
from pathlib import Path

import pytest

from tests.torchcell.candidates._builders import admissible_verdict, verdict
from torchcell.candidates import store


def test_round_trip_is_byte_stable(tmp_path: Path) -> None:
    """Write -> read returns an equal verdict; the file is the indented JSON."""
    v = verdict()
    path = store.write_verdict(v, tmp_path / "candidates")
    assert path == tmp_path / "candidates" / "keyPaper2020.json"
    assert path.read_text() == v.model_dump_json(indent=2) + "\n"
    assert store.read_verdict("keyPaper2020", tmp_path / "candidates") == v


def test_read_refuses_a_file_under_the_wrong_key(tmp_path: Path) -> None:
    """A verdict saved under another key's file name is refused."""
    path = store.write_verdict(verdict(), tmp_path)
    path.rename(tmp_path / "otherKey.json")
    with pytest.raises(ValueError, match="holds the verdict of 'keyPaper2020'"):
        store.read_verdict("otherKey", tmp_path)


@pytest.mark.parametrize("key", ["", "a/b", ".hidden"])
def test_verdict_path_rejects_non_keys(key: str, tmp_path: Path) -> None:
    """An empty key, a path, or a dotfile is not a citation key."""
    with pytest.raises(ValueError, match="not a citation key"):
        store.verdict_path(key, tmp_path)


def test_load_store_and_rows(tmp_path: Path) -> None:
    """load_store keys by citation key; verdicts_by_row splits tables; a twin row raises."""
    assert store.load_store(tmp_path / "absent") == {}
    store.write_verdict(verdict(), tmp_path)
    store.write_verdict(
        verdict(citation_key="yeastKey", table="yeast", row_name="Y"), tmp_path
    )
    assert sorted(store.load_store(tmp_path)) == ["keyPaper2020", "yeastKey"]
    assert list(store.verdicts_by_row("yeast", tmp_path)) == ["Y"]
    store.write_verdict(verdict(citation_key="twinKey"), tmp_path)
    with pytest.raises(ValueError, match="row 'Key 2020' has two verdicts"):
        store.verdicts_by_row("bacteria", tmp_path)


def test_grandfathered_is_sorted_and_counted() -> None:
    """125 classes, sorted, unique, as derived from the registry on 2026-10-10."""
    assert len(store.GRANDFATHERED) == 125
    assert list(store.GRANDFATHERED) == sorted(set(store.GRANDFATHERED))
    assert store.GRANDFATHERED[0] == "AminoAcidCooper2010Dataset"
    assert "SynthLethalityYeastSynthLethDbDataset" in store.GRANDFATHERED


@pytest.fixture
def fake_registry(monkeypatch: pytest.MonkeyPatch) -> dict[str, type]:
    """Three classes in two throwaway modules, one module keyed and one not."""
    keyed = types.ModuleType("_fake_keyed")
    keyed.CITATION_KEY = "keyPaper2020"  # type: ignore[attr-defined]
    bare = types.ModuleType("_fake_bare")
    monkeypatch.setitem(sys.modules, "_fake_keyed", keyed)
    monkeypatch.setitem(sys.modules, "_fake_bare", bare)
    classes = {
        name: type(name, (), {"__module__": module})
        for name, module in (
            ("OldDataset", "_fake_bare"),
            ("KeyedDataset", "_fake_keyed"),
            ("BareDataset", "_fake_bare"),
        )
    }
    return classes


def test_citation_key_of(fake_registry: dict[str, type]) -> None:
    """The module-level constant, or None."""
    assert store.citation_key_of(fake_registry["KeyedDataset"]) == "keyPaper2020"
    assert store.citation_key_of(fake_registry["BareDataset"]) is None


def test_enforcement_violations(fake_registry: dict[str, type]) -> None:
    """Grandfathered classes pass; others need a key and an admissible verdict."""
    assert store.enforcement_violations(fake_registry, ("OldDataset",), {}) == [
        "BareDataset: its module declares no CITATION_KEY",
        "KeyedDataset: no verdict at database/candidates/keyPaper2020.json",
    ]
    refused = {"keyPaper2020": verdict()}
    assert store.enforcement_violations(
        fake_registry, ("BareDataset", "OldDataset"), refused
    ) == ["KeyedDataset: verdict keyPaper2020 is refused"]
    for passing in (admissible_verdict(), admissible_verdict(gap_issue=854)):
        assert (
            store.enforcement_violations(
                fake_registry, ("BareDataset", "OldDataset"), {"keyPaper2020": passing}
            )
            == []
        )
