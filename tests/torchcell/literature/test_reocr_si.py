# tests/torchcell/literature/test_reocr_si.py
# [[tests.torchcell.literature.test_reocr_si]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_reocr_si.py
"""``torchcell.literature.reocr_si`` on a synthetic mirror: phase order, refusals,
retirement.

``ocr_pdf`` is replaced by a fake that does what the runner does for an SI PDF (writes
``<stem>.md`` referencing ``images/<stem>/<stem>.jpg`` and that figure). The Zotero
side (``ZoteroLibrary.from_env``, ``build_citation_index``, ``backfill_key``) is
replaced by recorders; the fake ``backfill_key`` WRITES ``manifest.json`` unless
``dry_run``, as the real one does, so a refusal after it is tested against the bytes on
disk. ``unresolved_si_figure_refs`` and ``retire_flat_figures`` are wrapped to record
each call and then run for real; ``retire_flat_figures`` runs the real
``scripts/deprecate.sh``. An autouse fixture points ``DEFAULT_GRAVEYARD`` and
``DATA_ROOT`` into ``tmp_path``, so no test, and no mutant of the module, can reach the
real graveyard. No MinerU, network or Zotero is touched.

Contract (PR #585 reviews): every key is resolved in Zotero, dry-run backfilled as
``enriched`` and checked for markdown that the run would leave pointing at a retired
flat figure, all before any OCR; then each phase runs over every key before the next
starts, in the exact order Zotero, dry-run, OCR, check, retire, check, backfill; a
failure in any phase stops before the next; a final backfill that is not ``enriched``
restores the previous manifest bytes; ``deprecate.sh``'s refusal of a graveyard inside
``DATA_ROOT`` applies; a staging directory left by a killed retirement is retired.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from torchcell.literature import reocr_si as reocr
from torchcell.literature.backfill import KeyBackfillResult
from torchcell.literature.zotero import ZoteroLibrary

_PREVIOUS = b'{"previous": "manifest"}'


@pytest.fixture(autouse=True)
def _tmp_graveyard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(reocr, "DEFAULT_GRAVEYARD", str(tmp_path / "default-graveyard"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))


def _key(root: Path, name: str, si_names: list[str], flat: list[str]) -> Path:
    key = root / name
    (key / "si" / "images").mkdir(parents=True)
    (key / "paper.pdf").write_bytes(b"%PDF paper")
    (key / "manifest.json").write_bytes(_PREVIOUS)
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
        dry_mode: str = "enriched",
        mode: str = "enriched",
        flat_ref: bool = False,
    ) -> None:
        self.events: list[str] = []
        self.fail_on = fail_on
        self.write_figure = write_figure
        self.indexed = indexed
        self.dry_mode = dry_mode
        self.mode = mode
        self.flat_ref = flat_ref

    def ocr_pdf(self, pdf: Path, *, device_mode: str) -> Path:
        self.events.append(f"ocr {pdf.parent.parent.name}/{pdf.name} {device_mode}")
        if f"{pdf.parent.parent.name}/{pdf.name}" == self.fail_on:
            raise RuntimeError(f"MinerU failed on {pdf.name}")
        if self.write_figure:
            figure = pdf.parent / "images" / pdf.stem / f"{pdf.stem}.jpg"
            figure.parent.mkdir(parents=True, exist_ok=True)
            figure.write_bytes(b"new")
        md = pdf.with_suffix(".md")
        ref = "images/f1.jpg" if self.flat_ref else f"images/{pdf.stem}/{pdf.stem}.jpg"
        md.write_text(f"![]({ref})\n")
        return md

    def build_citation_index(self, lib: str) -> dict[str, dict[str, Any]]:
        self.events.append(f"zotero {lib}")
        return {key: {"key": f"ITEM-{key}"} for key in self.indexed}

    def backfill_key(self, key_dir: Path, **kwargs: Any) -> KeyBackfillResult:
        dry = kwargs.get("dry_run", False)
        item = kwargs["citation_index"][key_dir.name]["key"]
        mode = self.dry_mode if dry else self.mode
        self.events.append(
            f"{'dry' if dry else 'backfill'} {key_dir.name} force={kwargs['force']} "
            f"lib={kwargs['lib']} item={item}"
        )
        if not dry:
            (key_dir / "manifest.json").write_text(f'{{"mode": "{mode}"}}')
        return KeyBackfillResult(citation_key=key_dir.name, mode=mode, n_files=1)


def _install(monkeypatch: pytest.MonkeyPatch, rec: _Recorder) -> None:
    monkeypatch.setattr(reocr, "ocr_pdf", rec.ocr_pdf)
    monkeypatch.setattr(reocr, "backfill_key", rec.backfill_key)
    monkeypatch.setattr(reocr, "build_citation_index", rec.build_citation_index)
    monkeypatch.setattr(ZoteroLibrary, "from_env", classmethod(lambda cls: "LIB"))
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


def _names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


_DRY = [
    "zotero LIB",
    "dry lee force=True lib=LIB item=ITEM-lee",
    "dry ohya force=True lib=LIB item=ITEM-ohya",
]
_OCR = [
    "ocr lee/si1.pdf cuda",
    "ocr lee/si2.pdf cuda",
    "ocr ohya/si2.pdf cuda",
    "ocr ohya/si10.pdf cuda",
]


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


def test_unrewritten_flat_refs_are_flat_references_of_markdown_with_no_pdf(
    tmp_path: Path,
) -> None:
    """``si1.md`` has ``si1.pdf`` (the re-OCR rewrites it); ``notes.md`` and
    ``si3.md`` have none, so only their flat references count, not a per-PDF one.
    """
    key = _key(tmp_path, "k", ["si1.pdf"], ["f1.jpg"])
    (key / "si" / "si1.md").write_text("![](images/f1.jpg)\n")
    (key / "si" / "notes.md").write_text("![](images/f1.jpg)\n![](images/si1/x.jpg)\n")
    (key / "si" / "si3.md").write_text('<img src="images/f9.png">\n')
    assert reocr.unrewritten_flat_refs(key) == [
        "notes.md: images/f1.jpg",
        "si3.md: images/f9.png",
    ]


# ------------------------------------------------------------------ phase order


def test_every_phase_runs_over_every_key_in_the_exact_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two keys, both with flat figures: Zotero, dry-run backfills, all OCR
    (natural order), all checks, all retirements, all checks again, all backfills.
    Each key's flat figures land in its own graveyard entry and each manifest is the
    enriched one the backfill wrote.
    """
    root = tmp_path / "torchcell-library"
    lee, ohya = _two_keys(root)
    graveyard = tmp_path / "graveyard"
    rec = _Recorder()
    _install(monkeypatch, rec)

    reocr.reocr_keys(
        root, ["lee", "ohya"], graveyard=str(graveyard), device_mode="cuda"
    )

    assert rec.events == [
        *_DRY,
        *_OCR,
        "check lee",
        "check ohya",
        "retire lee",
        "retire ohya",
        "check lee",
        "check ohya",
        "backfill lee force=True lib=LIB item=ITEM-lee",
        "backfill ohya force=True lib=LIB item=ITEM-ohya",
    ]
    assert _names(_graveyard_entry(graveyard, "lee")) == ["f1.jpg", "f2.jpg"]
    assert _names(_graveyard_entry(graveyard, "ohya")) == ["g1.jpg"]
    for key in (lee, ohya):
        assert reocr.flat_si_figures(key) == []
        assert not reocr.staging_dir(key).exists()
        assert (key / "manifest.json").read_text() == '{"mode": "enriched"}'


def test_an_ocr_failure_on_the_second_key_retires_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "torchcell-library"
    lee, ohya = _two_keys(root)
    rec = _Recorder(fail_on="ohya/si10.pdf")
    _install(monkeypatch, rec)
    with pytest.raises(RuntimeError, match="MinerU failed on si10.pdf"):
        reocr.reocr_keys(
            root, ["lee", "ohya"], graveyard=str(tmp_path / "g"), device_mode="cuda"
        )
    assert rec.events == [*_DRY, *_OCR]
    assert [p.name for p in reocr.flat_si_figures(lee)] == ["f1.jpg", "f2.jpg"]
    assert not (tmp_path / "g").exists()
    for key in (lee, ohya):
        assert (key / "manifest.json").read_bytes() == _PREVIOUS


def test_a_key_missing_from_zotero_refuses_before_any_ocr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "torchcell-library"
    _two_keys(root)
    rec = _Recorder(indexed=("lee",))
    _install(monkeypatch, rec)
    with pytest.raises(reocr.KeyNotInZoteroError) as refused:
        reocr.reocr_keys(
            root, ["lee", "ohya"], graveyard=str(tmp_path / "g"), device_mode="cuda"
        )
    assert str(refused.value) == "not in the Zotero citation index, nothing OCR'd: ohya"
    assert rec.events == ["zotero LIB"]


def test_a_dry_run_backfill_that_is_not_enriched_refuses_before_any_ocr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "torchcell-library"
    lee, _ = _two_keys(root)
    rec = _Recorder(dry_mode="offline")
    _install(monkeypatch, rec)
    with pytest.raises(reocr.NotEnrichedError) as refused:
        reocr.reocr_keys(
            root, ["lee", "ohya"], graveyard=str(tmp_path / "g"), device_mode="cuda"
        )
    assert str(refused.value) == (
        "dry-run backfill of lee returned mode 'offline', not 'enriched'; nothing OCR'd"
    )
    assert rec.events == _DRY[:2]
    assert (lee / "manifest.json").read_bytes() == _PREVIOUS


def test_a_final_backfill_that_is_not_enriched_restores_the_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The dry run said enriched but the real backfill wrote an offline manifest:
    the previous bytes are back on disk and the refusal names the key.
    """
    root = tmp_path / "torchcell-library"
    lee, _ = _two_keys(root)
    rec = _Recorder(mode="offline")
    _install(monkeypatch, rec)
    with pytest.raises(reocr.NotEnrichedError) as refused:
        reocr.reocr_keys(
            root, ["lee", "ohya"], graveyard=str(tmp_path / "g"), device_mode="cuda"
        )
    assert str(refused.value) == (
        "backfill of lee returned mode 'offline', not 'enriched'; the previous "
        "manifest was restored"
    )
    assert rec.events[-1] == "backfill lee force=True lib=LIB item=ITEM-lee"
    assert (lee / "manifest.json").read_bytes() == _PREVIOUS


def test_a_final_backfill_with_no_previous_manifest_leaves_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "torchcell-library"
    lee = _key(root, "lee", ["si1.pdf"], [])
    (lee / "manifest.json").rename(tmp_path / "moved.json")
    _install(monkeypatch, _Recorder(mode="offline", indexed=("lee",)))
    with pytest.raises(reocr.NotEnrichedError):
        reocr.reocr_keys(
            root, ["lee"], graveyard=str(tmp_path / "g"), device_mode="cuda"
        )
    assert not (lee / "manifest.json").exists()


def test_markdown_left_pointing_at_a_flat_figure_refuses_before_any_ocr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``si3.md`` has no ``si3.pdf`` and references the flat ``images/f1.jpg``:
    retiring would break it and no re-OCR rewrites it, so the run refuses up front.
    """
    root = tmp_path / "torchcell-library"
    lee = _key(root, "lee", ["si1.pdf"], ["f1.jpg"])
    (lee / "si" / "si3.md").write_text("![](images/f1.jpg)\n")
    rec = _Recorder(indexed=("lee",))
    _install(monkeypatch, rec)
    with pytest.raises(reocr.UnrewrittenFlatReferenceError) as refused:
        reocr.reocr_keys(
            root, ["lee"], graveyard=str(tmp_path / "g"), device_mode="cuda"
        )
    assert str(refused.value) == (
        "markdown with no SI PDF to re-OCR references flat si/images/ figures, which "
        "retirement would break; nothing OCR'd: lee/si/si3.md: images/f1.jpg"
    )
    assert rec.events == _DRY[:2]
    assert [p.name for p in reocr.flat_si_figures(lee)] == ["f1.jpg"]


def test_an_unresolved_reference_after_the_ocr_refuses_before_retiring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "torchcell-library"
    lee = _key(root, "lee", ["si1.pdf"], ["f1.jpg"])
    rec = _Recorder(write_figure=False, indexed=("lee",))
    _install(monkeypatch, rec)
    with pytest.raises(reocr.UnresolvedFigureError) as refused:
        reocr.reocr_keys(
            root, ["lee"], graveyard=str(tmp_path / "g"), device_mode="cuda"
        )
    assert str(refused.value) == (
        "1 SI figure reference(s) unresolved after re-OCR, nothing retired; no "
        "manifest written: lee/si/si1.md: images/si1/si1.jpg"
    )
    assert rec.events == [*_DRY[:2], "ocr lee/si1.pdf cuda", "check lee"]
    assert (lee / "si" / "images" / "f1.jpg").read_bytes() == b"old"


def test_the_check_after_retiring_is_a_backstop_before_any_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An OCR that wrote a flat reference (``images/f1.jpg``) passes the first check
    and dangles once ``f1.jpg`` is retired: the second check refuses by name and the
    manifest keeps its previous bytes; ``f1.jpg`` is in the graveyard.
    """
    root = tmp_path / "torchcell-library"
    lee = _key(root, "lee", ["si1.pdf"], ["f1.jpg"])
    graveyard = tmp_path / "g"
    rec = _Recorder(indexed=("lee",), flat_ref=True)
    _install(monkeypatch, rec)
    with pytest.raises(reocr.UnresolvedFigureError) as refused:
        reocr.reocr_keys(root, ["lee"], graveyard=str(graveyard), device_mode="cuda")
    assert str(refused.value) == (
        f"1 SI figure reference(s) unresolved after retiring flat figures to "
        f"{graveyard}; no manifest written: lee/si/si1.md: images/f1.jpg"
    )
    assert rec.events == [
        *_DRY[:2],
        "ocr lee/si1.pdf cuda",
        "check lee",
        "retire lee",
        "check lee",
    ]
    assert _names(_graveyard_entry(graveyard, "lee")) == ["f1.jpg"]
    assert (lee / "manifest.json").read_bytes() == _PREVIOUS


# ------------------------------------------------------------------ retirement


def test_a_graveyard_inside_data_root_is_refused_and_nothing_is_moved(
    tmp_path: Path,
) -> None:
    """``DATA_ROOT`` reaches ``deprecate.sh``, whose guard exits 2: ``RetireError``
    carries the exit code, the flat figures are back in ``si/images/``, the staging
    directory is gone and nothing reached the graveyard.
    """
    lee = _key(tmp_path / "data" / "torchcell-library", "lee", ["si1.pdf"], ["f1.jpg"])
    graveyard = tmp_path / "data" / "torchcell-deprecated"
    with pytest.raises(reocr.RetireError) as refused:
        reocr.retire_flat_figures(lee, str(graveyard))
    staging = reocr.staging_dir(lee)
    assert str(refused.value).startswith(
        f"deprecate.sh exited 2 retiring {staging}; flat figures moved back into "
        "si/images/: deprecate: refusing graveyard inside DATA_ROOT"
    )
    assert [p.name for p in reocr.flat_si_figures(lee)] == ["f1.jpg"]
    assert not staging.exists()
    assert not graveyard.exists()


def test_a_deprecate_failure_after_its_move_reports_the_exit_code_and_where(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ``deprecate.sh`` that moves the staging directory and then exits 3: the
    error keeps exit code 3 and says to look in the graveyard, where the file is.
    """
    lee = _key(tmp_path, "lee", ["si1.pdf"], ["f1.jpg"])
    script = tmp_path / "deprecate.sh"
    script.write_text(
        'mkdir -p "$DEPRECATED_DIR" && mv "$1" "$DEPRECATED_DIR/" && '
        "echo boom >&2 && exit 3\n"
    )
    monkeypatch.setattr(reocr, "DEPRECATE_SH", script)
    graveyard = tmp_path / "g"
    with pytest.raises(reocr.RetireError) as refused:
        reocr.retire_flat_figures(lee, str(graveyard))
    staging = reocr.staging_dir(lee)
    assert str(refused.value) == (
        f"deprecate.sh exited 3 retiring {staging}; {staging.name} is no longer in "
        f"si/; look for it in {graveyard}: boom"
    )
    assert _names(graveyard / staging.name) == ["f1.jpg"]


@pytest.mark.parametrize(
    ("staged", "flat"), [(["f1.jpg"], ["f2.jpg"]), (["f1.jpg", "f2.jpg"], [])]
)
def test_a_staging_directory_left_by_a_killed_retirement_is_retired(
    tmp_path: Path, staged: list[str], flat: list[str]
) -> None:
    """A killed earlier run moved ``staged`` into the staging directory and left
    ``flat`` in ``si/images/`` (with nothing flat left: a kill inside
    ``deprecate.sh``). The next retirement takes all of them to the graveyard; then
    there is nothing to retire.
    """
    lee = _key(tmp_path, "lee", ["si1.pdf"], flat)
    staging = reocr.staging_dir(lee)
    staging.mkdir()
    for name in staged:
        (staging / name).write_bytes(b"old")
    graveyard = tmp_path / "g"

    assert reocr.retire_flat_figures(lee, str(graveyard)) == staging

    assert _names(_graveyard_entry(graveyard, "lee")) == ["f1.jpg", "f2.jpg"]
    assert not staging.exists()
    assert reocr.retire_flat_figures(lee, str(graveyard)) is None


# ------------------------------------------------------------------ main


def test_main_passes_keys_root_device_and_graveyard_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The real ``reocr_keys`` behind ``main``: ``--graveyard`` decides where the
    flat figures land (not the default, also pointed into ``tmp_path``),
    ``--device-mode`` reaches the OCR, ``--root`` the mirror.
    """
    root = tmp_path / "mirror"
    _two_keys(root)
    rec = _Recorder()
    _install(monkeypatch, rec)
    default = tmp_path / "default"
    monkeypatch.setattr(reocr, "DEFAULT_GRAVEYARD", str(default))
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

    assert rec.events[3] == "ocr lee/si1.pdf cpu"
    assert _names(_graveyard_entry(graveyard, "lee")) == ["f1.jpg", "f2.jpg"]
    assert not default.exists()


def test_main_defaults_root_and_graveyard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, list[str], str, str]] = []
    monkeypatch.setattr(
        reocr,
        "reocr_keys",
        lambda root, keys, *, graveyard, device_mode: calls.append(
            (root, keys, graveyard, device_mode)
        ),
    )
    assert reocr.main(["--key", "a", "--key", "b"]) == 0
    assert calls == [
        (
            tmp_path / "data" / "torchcell-library",
            ["a", "b"],
            str(tmp_path / "default-graveyard"),
            "cuda",
        )
    ]
