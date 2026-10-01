# tests/scripts/test_lit_reocr_si.py
# [[tests.scripts.test_lit_reocr_si]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/scripts/test_lit_reocr_si.py
"""``scripts/lit_reocr_si.py`` on a synthetic mirror: phase order, refusals, retirement.

The script is loaded from its file. ``ocr_pdf`` is replaced by a fake that does what
the runner does for an SI PDF (writes ``<stem>.md`` referencing
``images/<stem>/<stem>.jpg`` and that figure). The Zotero side
(``ZoteroLibrary.from_env``, ``build_citation_index``, ``backfill_key``) is replaced by
recorders, so no MinerU, network or Zotero is touched. ``unresolved_si_figure_refs``
and ``retire_flat_figures`` are wrapped to record each call and then run for real,
and ``retire_flat_figures`` runs the real ``scripts/deprecate.sh`` into a graveyard
under ``tmp_path`` with ``DATA_ROOT`` set to ``tmp_path/data``.

Contract (PR #585 review): every key is resolved in Zotero before any OCR; then each
phase runs over every key before the next starts, in the exact order Zotero, OCR,
check, retire, check, backfill; a failure in any phase stops before the next, so an
OCR failure retires nothing and writes no manifest; a dangling reference after the
retirement refuses before any manifest; a graveyard inside ``DATA_ROOT`` is refused
by the real ``deprecate.sh`` (exit 2) and the figures are moved back; a staging
directory left by a killed retirement is retired on the next run; a backfill that is
not ``enriched`` refuses.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from torchcell.literature.backfill import KeyBackfillResult

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "lit_reocr_si.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lit_reocr_si", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reocr = _load()


def _key(root: Path, name: str, si_names: list[str], flat: list[str]) -> Path:
    key = root / name
    (key / "si" / "images").mkdir(parents=True)
    (key / "paper.pdf").write_bytes(b"%PDF paper")
    for si in si_names:
        (key / "si" / si).write_bytes(b"%PDF si")
    for fig in flat:
        (key / "si" / "images" / fig).write_bytes(b"old")
    return key


class _Recorder:
    """Records every phase call in ``events`` and fakes OCR and Zotero."""

    def __init__(
        self,
        fail_on: str | None = None,
        write_figure: bool = True,
        indexed: tuple[str, ...] = ("lee", "ohya"),
        mode: str = "enriched",
    ) -> None:
        self.events: list[str] = []
        self.fail_on = fail_on
        self.write_figure = write_figure
        self.indexed = indexed
        self.mode = mode

    def ocr_pdf(self, pdf: Path, *, device_mode: str) -> Path:
        self.events.append(f"ocr {pdf.parent.parent.name}/{pdf.name} {device_mode}")
        if f"{pdf.parent.parent.name}/{pdf.name}" == self.fail_on:
            raise RuntimeError(f"MinerU failed on {pdf.name}")
        if self.write_figure:
            figure = pdf.parent / "images" / pdf.stem / f"{pdf.stem}.jpg"
            figure.parent.mkdir(parents=True, exist_ok=True)
            figure.write_bytes(b"new")
        md = pdf.with_suffix(".md")
        md.write_text(f"![](images/{pdf.stem}/{pdf.stem}.jpg)\n")
        return md

    def build_citation_index(self, lib: str) -> dict[str, dict[str, Any]]:
        self.events.append(f"zotero {lib}")
        return {key: {"key": f"ITEM-{key}"} for key in self.indexed}

    def backfill_key(self, key_dir: Path, **kwargs: Any) -> KeyBackfillResult:
        self.events.append(
            f"backfill {key_dir.name} force={kwargs['force']} lib={kwargs['lib']} "
            f"item={kwargs['citation_index'][key_dir.name]['key']}"
        )
        return KeyBackfillResult(citation_key=key_dir.name, mode=self.mode, n_files=1)


def _install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rec: _Recorder) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(reocr, "ocr_pdf", rec.ocr_pdf)
    monkeypatch.setattr(reocr, "backfill_key", rec.backfill_key)
    monkeypatch.setattr(reocr, "build_citation_index", rec.build_citation_index)
    monkeypatch.setattr(reocr.ZoteroLibrary, "from_env", classmethod(lambda cls: "LIB"))
    check: Callable[[Path], list[str]] = reocr.unresolved_si_figure_refs
    retire: Callable[[Path, str], Path | None] = reocr.retire_flat_figures

    def recorded_check(key_dir: Path) -> list[str]:
        rec.events.append(f"check {key_dir.name}")
        return check(key_dir)

    def recorded_retire(key_dir: Path, graveyard: str) -> Path | None:
        rec.events.append(f"retire {key_dir.name}")
        return retire(key_dir, graveyard)

    monkeypatch.setattr(reocr, "unresolved_si_figure_refs", recorded_check)
    monkeypatch.setattr(reocr, "retire_flat_figures", recorded_retire)


def _two_keys(root: Path) -> tuple[Path, Path]:
    lee = _key(root, "lee", ["si1.pdf", "si2.pdf"], ["f1.jpg", "f2.jpg"])
    ohya = _key(root, "ohya", ["si10.pdf", "si2.pdf"], ["g1.jpg"])
    return lee, ohya


def _graveyard_entry(graveyard: Path, key: str) -> Path:
    (entry,) = [
        p
        for p in graveyard.iterdir()
        if p.name.endswith(f"__{key}__si-images-flat-pre579")
    ]
    return entry / f"{key}__si-images-flat-pre579"


# ------------------------------------------------------------------ pure parts


def test_si_pdfs_are_natural_and_a_key_without_any_refuses(tmp_path: Path) -> None:
    key = _key(tmp_path, "k", ["si10.pdf", "si2.pdf", "si1.pdf", "table.pdf"], [])
    assert [p.name for p in reocr.si_pdfs(key)] == ["si1.pdf", "si2.pdf", "si10.pdf"]
    empty = _key(tmp_path, "empty", [], [])
    with pytest.raises(reocr.NoSiPdfError) as refused:
        reocr.si_pdfs(empty)
    assert str(refused.value) == f"no si/si*.pdf under {empty}"


def test_flat_figures_are_only_files_directly_in_si_images(tmp_path: Path) -> None:
    key = _key(tmp_path, "k", ["si1.pdf"], ["b.jpg", "a.jpg"])
    (key / "si" / "images" / "si1").mkdir()
    (key / "si" / "images" / "si1" / "c.jpg").write_bytes(b"x")
    assert [p.name for p in reocr.flat_si_figures(key)] == ["a.jpg", "b.jpg"]
    assert reocr.flat_si_figures(_key(tmp_path, "none", [], [])) == []


def test_unresolved_refs_name_the_markdown_and_the_reference(tmp_path: Path) -> None:
    """Markdown and HTML forms are read from ``si/*.md`` only; a prose URL is not a
    reference.
    """
    key = _key(tmp_path, "k", [], [])
    (key / "si" / "images" / "si1").mkdir()
    (key / "si" / "images" / "si1" / "ok.jpg").write_bytes(b"x")
    (key / "si" / "si1.md").write_text(
        "![](images/si1/ok.jpg)\n![](images/si1/gone.jpg)\n"
        "see https://example.org/images/logo.png\n"
    )
    (key / "si" / "si10.md").write_text('<img src="images/si10/x.png">\n')
    (key / "si" / "si2.md").write_text("no figures\n")
    assert reocr.unresolved_si_figure_refs(key) == [
        "si1.md: images/si1/gone.jpg",
        "si10.md: images/si10/x.png",
    ]


# ------------------------------------------------------------------ phase order


def test_every_phase_runs_over_every_key_in_the_exact_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two keys, both with flat figures: Zotero, then all OCR (natural order, device
    passed), all checks, all retirements, all checks again, all backfills. Each
    key's flat figures land in its own graveyard entry.
    """
    root = tmp_path / "torchcell-library"
    lee, ohya = _two_keys(root)
    graveyard = tmp_path / "graveyard"
    rec = _Recorder()
    _install(monkeypatch, tmp_path, rec)

    reocr.reocr_keys(root, ["lee", "ohya"], graveyard=str(graveyard), device_mode="cpu")

    assert rec.events == [
        "zotero LIB",
        "ocr lee/si1.pdf cpu",
        "ocr lee/si2.pdf cpu",
        "ocr ohya/si2.pdf cpu",
        "ocr ohya/si10.pdf cpu",
        "check lee",
        "check ohya",
        "retire lee",
        "retire ohya",
        "check lee",
        "check ohya",
        "backfill lee force=True lib=LIB item=ITEM-lee",
        "backfill ohya force=True lib=LIB item=ITEM-ohya",
    ]
    assert sorted(p.name for p in _graveyard_entry(graveyard, "lee").iterdir()) == [
        "f1.jpg",
        "f2.jpg",
    ]
    assert sorted(p.name for p in _graveyard_entry(graveyard, "ohya").iterdir()) == [
        "g1.jpg"
    ]
    for key in (lee, ohya):
        assert reocr.flat_si_figures(key) == []
        assert not reocr.staging_dir(key).exists()


def test_an_ocr_failure_on_the_second_key_retires_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "torchcell-library"
    lee, _ = _two_keys(root)
    rec = _Recorder(fail_on="ohya/si10.pdf")
    _install(monkeypatch, tmp_path, rec)
    with pytest.raises(RuntimeError, match="MinerU failed on si10.pdf"):
        reocr.reocr_keys(
            root, ["lee", "ohya"], graveyard=str(tmp_path / "g"), device_mode="cuda"
        )
    assert rec.events == [
        "zotero LIB",
        "ocr lee/si1.pdf cuda",
        "ocr lee/si2.pdf cuda",
        "ocr ohya/si2.pdf cuda",
        "ocr ohya/si10.pdf cuda",
    ]
    assert [p.name for p in reocr.flat_si_figures(lee)] == ["f1.jpg", "f2.jpg"]
    assert not (tmp_path / "g").exists()


def test_a_key_missing_from_zotero_refuses_before_any_ocr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "torchcell-library"
    _two_keys(root)
    rec = _Recorder(indexed=("lee",))
    _install(monkeypatch, tmp_path, rec)
    with pytest.raises(reocr.KeyNotInZoteroError) as refused:
        reocr.reocr_keys(
            root, ["lee", "ohya"], graveyard=str(tmp_path / "g"), device_mode="cuda"
        )
    assert str(refused.value) == "not in the Zotero citation index, nothing OCR'd: ohya"
    assert rec.events == ["zotero LIB"]


def test_a_backfill_that_is_not_enriched_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "torchcell-library"
    _two_keys(root)
    rec = _Recorder(mode="offline")
    _install(monkeypatch, tmp_path, rec)
    with pytest.raises(reocr.NotEnrichedError) as refused:
        reocr.reocr_keys(
            root, ["lee", "ohya"], graveyard=str(tmp_path / "g"), device_mode="cuda"
        )
    assert (
        str(refused.value) == "backfill of lee wrote a offline manifest, not enriched"
    )
    assert rec.events[-1] == "backfill lee force=True lib=LIB item=ITEM-lee"


def test_an_unresolved_reference_after_the_ocr_refuses_before_retiring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "torchcell-library"
    lee = _key(root, "lee", ["si1.pdf"], ["f1.jpg"])
    rec = _Recorder(write_figure=False, indexed=("lee",))
    _install(monkeypatch, tmp_path, rec)
    with pytest.raises(reocr.UnresolvedFigureError) as refused:
        reocr.reocr_keys(
            root, ["lee"], graveyard=str(tmp_path / "g"), device_mode="cuda"
        )
    assert str(refused.value) == (
        "1 SI figure reference(s) unresolved after re-OCR, nothing retired; no "
        "manifest written: lee/si/si1.md: images/si1/si1.jpg"
    )
    assert rec.events == ["zotero LIB", "ocr lee/si1.pdf cuda", "check lee"]
    assert (lee / "si" / "images" / "f1.jpg").read_bytes() == b"old"


def test_a_reference_to_a_flat_figure_refuses_after_retiring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``si1.md`` (not re-OCR'd) still points at the flat ``images/f1.jpg``: it
    resolves before the retirement and dangles after it, so the second check refuses
    by name before any manifest is written; ``f1.jpg`` is in the graveyard.
    """
    root = tmp_path / "torchcell-library"
    lee = _key(root, "lee", ["si2.pdf"], ["f1.jpg"])
    (lee / "si" / "si1.md").write_text("![](images/f1.jpg)\n")
    graveyard = tmp_path / "g"
    rec = _Recorder(indexed=("lee",))
    _install(monkeypatch, tmp_path, rec)
    with pytest.raises(reocr.UnresolvedFigureError) as refused:
        reocr.reocr_keys(root, ["lee"], graveyard=str(graveyard), device_mode="cuda")
    assert str(refused.value) == (
        f"1 SI figure reference(s) unresolved after retiring flat figures to "
        f"{graveyard}; no manifest written: lee/si/si1.md: images/f1.jpg"
    )
    assert rec.events == [
        "zotero LIB",
        "ocr lee/si2.pdf cuda",
        "check lee",
        "retire lee",
        "check lee",
    ]
    assert [p.name for p in _graveyard_entry(graveyard, "lee").iterdir()] == ["f1.jpg"]


# ------------------------------------------------------------------ retirement


def test_a_graveyard_inside_data_root_is_refused_and_nothing_is_moved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``DATA_ROOT`` reaches ``deprecate.sh``, whose guard exits 2: ``RetireError``
    carries the exit code, the flat figures are back in ``si/images/``, the staging
    directory is gone and nothing reached the graveyard.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    lee = _key(tmp_path / "data" / "torchcell-library", "lee", ["si1.pdf"], ["f1.jpg"])
    graveyard = tmp_path / "data" / "torchcell-deprecated"
    with pytest.raises(reocr.RetireError) as refused:
        reocr.retire_flat_figures(lee, str(graveyard))
    staging = reocr.staging_dir(lee)
    assert str(refused.value).startswith(
        f"deprecate.sh exited 2 retiring {staging}; flat figures moved back: "
        "deprecate: refusing graveyard inside DATA_ROOT"
    )
    assert [p.name for p in reocr.flat_si_figures(lee)] == ["f1.jpg"]
    assert not staging.exists()
    assert not graveyard.exists()


def test_a_staging_directory_left_by_a_killed_retirement_is_retired(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A killed earlier run moved ``f1.jpg`` into the staging directory and left
    ``f2.jpg`` flat: the next retirement takes both to the graveyard. With neither
    flat figures nor a staging directory there is nothing to retire.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    lee = _key(tmp_path, "lee", ["si1.pdf"], ["f1.jpg", "f2.jpg"])
    staging = reocr.staging_dir(lee)
    staging.mkdir()
    (lee / "si" / "images" / "f1.jpg").rename(staging / "f1.jpg")
    graveyard = tmp_path / "g"

    assert reocr.retire_flat_figures(lee, str(graveyard)) == staging

    assert sorted(p.name for p in _graveyard_entry(graveyard, "lee").iterdir()) == [
        "f1.jpg",
        "f2.jpg",
    ]
    assert not staging.exists()
    assert reocr.retire_flat_figures(lee, str(graveyard)) is None


# ------------------------------------------------------------------ main


def test_main_passes_keys_root_device_and_graveyard_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The real ``reocr_keys`` behind ``main``: ``--graveyard`` decides where the
    flat figures land, ``--device-mode`` reaches the OCR, ``--root`` the mirror.
    """
    root = tmp_path / "mirror"
    _two_keys(root)
    rec = _Recorder()
    _install(monkeypatch, tmp_path, rec)
    monkeypatch.setattr(reocr, "load_dotenv", lambda path: None)
    graveyard = tmp_path / "custom-graveyard"

    assert (
        reocr.main(
            [
                "--key",
                "lee",
                "--key",
                "ohya",
                "--root",
                str(root),
                "--graveyard",
                str(graveyard),
                "--device-mode",
                "cpu",
            ]
        )
        == 0
    )

    assert rec.events[1] == "ocr lee/si1.pdf cpu"
    assert sorted(p.name for p in _graveyard_entry(graveyard, "lee").iterdir()) == [
        "f1.jpg",
        "f2.jpg",
    ]


def test_main_defaults_root_to_the_data_root_mirror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, list[str], str, str]] = []
    monkeypatch.setattr(reocr, "load_dotenv", lambda path: None)
    monkeypatch.setattr(
        reocr,
        "reocr_keys",
        lambda root, keys, *, graveyard, device_mode: calls.append(
            (root, keys, graveyard, device_mode)
        ),
    )
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert reocr.main(["--key", "a", "--key", "b"]) == 0
    assert calls == [
        (tmp_path / "torchcell-library", ["a", "b"], reocr.DEFAULT_GRAVEYARD, "cuda")
    ]
